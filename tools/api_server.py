# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import asyncio
import base64
import uuid
from io import BytesIO

import msgspec
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.disaggregated.proxy import Proxy
from vllm.multimodal.image import convert_image_mode
from vllm.sampling_params import SamplingParams

app = FastAPI()

IMAGE_PATH = "tools/512p.jpg"
image = convert_image_mode(Image.open(IMAGE_PATH), "RGB")
IMAGE_ARRAY = np.array(image)


# 注册路由
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        request_data = await request.json()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        is_streaming = request_data.get("stream", False)

        # Extract parameters from request
        prompt = request_data.get("messages", [])[-1].get("content", [])
        # For simplicity, we'll use the last message content as the prompt
        if prompt and isinstance(prompt, list):
            prompt_text = prompt[-1].get("text", "")
        else:
            prompt_text = ""

        binary_list = []
        image_num = len(prompt) - 1
        # load image and base64 decode
        if app.state.is_load_image:
            for i in range(image_num):
                image_base64 = prompt[i].get("image_url",
                                             "").get("url", "").split(",")[-1]
                image_data = base64.b64decode(image_base64.encode("utf-8"))
                image_buffer = BytesIO(image_data)
                image = convert_image_mode(Image.open(image_buffer), "RGB")
                binary_data = np.array(image)
                binary_list.append(binary_data)

        ## the np of image
        else:
            for _ in range(image_num):
                binary_list.append(IMAGE_ARRAY)

        if "qwen" in app.state.proxy.model_config.model.lower():
            image_pad = "<|image_pad|>"
        else:
            image_pad = "<image>"

        image_str = ""
        for i in range(image_num):
            image_str += image_pad

        prompt_text = ("<|im_start|>system\n"
                       "You are a helpful assistant.<|im_end|>\n"
                       "<|im_start|>user\n"
                       f"<|vision_start|>{image_str}<|vision_end|>"
                       f"{prompt_text}<|im_end|>\n"
                       "<|im_start|>assistant\n")

        prompt = {
            "prompt": prompt_text,
            "multi_modal_data": {
                "image": binary_list
            },
        }

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 1.0),
            top_k=request_data.get("top_k", 10),
            max_tokens=request_data.get("max_tokens", 100),
            stop=request_data.get("stop", None),
            seed=request_data.get("seed", 77),
            repetition_penalty=request_data.get("repetition_penalty", 1.0),
            stop_token_ids=request_data.get("stop_token_ids", None),
        )

        if is_streaming:

            async def stream_generator():
                async for output in app.state.proxy.generate(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                ):
                    prompt_tokens = len(output.prompt_token_ids)
                    completion_tokens = len(output.outputs[0].token_ids)
                    total_tokens = prompt_tokens + completion_tokens
                    # Format according to OpenAI's streaming format
                    chunk = {
                        "id":
                        request_id,
                        "object":
                        "chat.completion.chunk",
                        "created":
                        int(asyncio.get_event_loop().time()),
                        "model":
                        app.state.proxy.model_config.model,
                        "choices": [{
                            "index":
                            0,
                            "delta": {
                                "content": output.outputs[0].text
                            },
                            "finish_reason":
                            output.outputs[0].finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                    }
                    yield f"data: {msgspec.json.encode(chunk).decode()}\n\n"
                # End of stream
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(),
                                     media_type="text/event-stream")
        else:
            # For non-streaming, collect all outputs
            final_output = None
            async for output in app.state.proxy.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
            ):
                final_output = output

            if final_output:
                prompt_tokens = len(final_output.prompt_token_ids)
                completion_tokens = len(final_output.outputs[0].token_ids)
                total_tokens = prompt_tokens + completion_tokens
                response = {
                    "id":
                    request_id,
                    "object":
                    "chat.completion",
                    "created":
                    int(asyncio.get_event_loop().time()),
                    "model":
                    app.state.proxy.model_config.model,
                    "choices": [{
                        "index":
                        0,
                        "message": {
                            "role": "assistant",
                            "content": final_output.outputs[0].text
                        },
                        "finish_reason":
                        final_output.outputs[0].finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
                return JSONResponse(content=response)
            else:
                raise HTTPException(status_code=500,
                                    detail="No response from proxy")
    except Exception as e:
        print("Error processing chat completion request: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/completions")
async def completions(request: Request):
    try:
        request_data = await request.json()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        is_streaming = request_data.get("stream", False)

        # Extract parameters from request
        prompt = request_data.get("prompt", "")

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request_data.get("temperature", 0.7),
            top_p=request_data.get("top_p", 1.0),
            max_tokens=request_data.get("max_tokens", 100),
            stop=request_data.get("stop", None),
            seed=request_data.get("seed", 77),
            repetition_penalty=request_data.get("repetition_penalty", 1.0),
            stop_token_ids=request_data.get("stop_token_ids", None),
        )

        if is_streaming:

            async def stream_generator():
                async for output in app.state.proxy.generate(
                        prompt=prompt,
                        sampling_params=sampling_params,
                        request_id=request_id,
                ):
                    prompt_tokens = len(output.prompt_token_ids)
                    completion_tokens = len(output.outputs[0].token_ids)
                    total_tokens = prompt_tokens + completion_tokens
                    # Format according to OpenAI's streaming format
                    chunk = {
                        "id":
                        request_id,
                        "object":
                        "text_completion.chunk",
                        "created":
                        int(asyncio.get_event_loop().time()),
                        "model":
                        app.state.proxy.model_config.model,
                        "choices": [{
                            "index":
                            0,
                            "text":
                            output.outputs[0].text,
                            "finish_reason":
                            output.outputs[0].finish_reason
                        }],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        }
                    }
                    yield f"data: {msgspec.json.encode(chunk).decode()}\n\n"
                # End of stream
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(),
                                     media_type="text/event-stream")
        else:
            # For non-streaming, collect all outputs
            final_output = None
            async for output in app.state.proxy.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
            ):
                final_output = output

            if final_output:
                prompt_tokens = len(final_output.prompt_token_ids)
                completion_tokens = len(final_output.outputs[0].token_ids)
                total_tokens = prompt_tokens + completion_tokens
                response = {
                    "id":
                    request_id,
                    "object":
                    "text_completion",
                    "created":
                    int(asyncio.get_event_loop().time()),
                    "model":
                    app.state.proxy.model_config.model,
                    "choices": [{
                        "index":
                        0,
                        "text":
                        final_output.outputs[0].text,
                        "finish_reason":
                        final_output.outputs[0].finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
                return JSONResponse(content=response)
            else:
                raise HTTPException(status_code=500,
                                    detail="No response from proxy")
    except Exception as e:
        print("Error processing completion request: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM Disaggregated Proxy")
    parser.add_argument("--host",
                        type=str,
                        default="127.0.0.1",
                        help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")
    parser.add_argument("--proxy-addr",
                        type=str,
                        required=True,
                        help="Proxy address")
    parser.add_argument("--e-addr-list",
                        type=str,
                        required=True,
                        help="encode worker ipc address")
    parser.add_argument("--pd-addr-list",
                        type=str,
                        required=True,
                        help="pd worker ipc address")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--is-load-image",
                        action='store_true',
                        help="load image from path")

    args = parser.parse_args()
    app.state.proxy = Proxy(proxy_addr=args.proxy_addr,
                            encode_addr_list=args.e_addr_list.split(","),
                            pd_addr_list=args.pd_addr_list.split(","),
                            enable_health_monitor=True,
                            model_name=args.model)
    app.state.is_load_image = args.is_load_image
    print(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app=app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=True,
                loop="asyncio")
