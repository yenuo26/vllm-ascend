#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/blob/main/tests/conftest.py
#

import asyncio
import contextlib
import gc
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, List, Optional, Tuple, TypeVar, Union

import httpx
import numpy as np
import openai
import psutil
import pytest
import requests
import torch
import yaml
from PIL import Image
from modelscope import snapshot_download  # type: ignore[import-untyped]
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, BatchFeature)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from vllm import LLM, SamplingParams
from vllm.config.model import TaskOption, _get_and_verify_dtype
from vllm.disaggregated.protocol import ServerType
from vllm.disaggregated.proxy import Proxy
from vllm.inputs import TextPrompt
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils import get_open_port

from tests.e2e.model_utils import (TokensTextLogprobs,
                                   TokensTextLogprobsPromptLogprobs)
from vllm_ascend.ascend_config import clear_ascend_config
# TODO: remove this part after the patch merged into vllm, if
# we not explicitly patch here, some of them might be effectiveless
# in pytest scenario
from vllm_ascend.utils import adapt_patch  # noqa E402

adapt_patch(True)
adapt_patch(False)

from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment, destroy_model_parallel)

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature, dict)
_M = TypeVar("_M")

_PromptMultiModalInput = Union[List[_M], List[List[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[Tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]

_TEST_DIR = os.path.dirname(__file__)


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray
        ray.shutdown()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


class RemoteOpenAIServer:
    DUMMY_API_KEY = "token-abc123"  # vLLM's OpenAI server does not need API key

    def _start_server(self, model: str, server_cmd: list[str],
                      env_dict: Optional[dict[str, str]]) -> None:
        """Subclasses override this method to customize server process launch
        """
        env = os.environ.copy()
        # the current process might initialize npu,
        # to be safe, we should use spawn method
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        if env_dict is not None:
            env.update(env_dict)
        self.proc: subprocess.Popen = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def __init__(self,
                 model: str,
                 vllm_serve_args: Union[list[str], str],
                 *,
                 server_host: str = "0.0.0.0",
                 server_port: int = 8080,
                 env_dict: Optional[dict[str, str]] = None,
                 seed: Optional[int] = None,
                 auto_port: bool = True,
                 max_wait_seconds: Optional[float] = None,
                 override_hf_configs: Optional[dict[str, Any]] = None) -> None:
        if isinstance(vllm_serve_args, str):
            vllm_serve_args = shlex.split(vllm_serve_args)
        else:
            vllm_serve_args = ["vllm", "serve", model, *vllm_serve_args]
        if auto_port:
            if "-p" in vllm_serve_args or "--port" in vllm_serve_args:
                raise ValueError("You have manually specified the port "
                                 "when `auto_port=True`.")

            # No need for a port if using unix sockets
            if "--uds" not in vllm_serve_args:
                # Don't mutate the input args
                vllm_serve_args = vllm_serve_args + [
                    "--port", str(get_open_port())
                ]
        if seed is not None:
            if "--seed" in vllm_serve_args:
                raise ValueError("You have manually specified the seed "
                                 f"when `seed={seed}`.")

            vllm_serve_args = vllm_serve_args + ["--seed", str(seed)]

        if override_hf_configs is not None:
            vllm_serve_args = vllm_serve_args + [
                "--hf-overrides",
                json.dumps(override_hf_configs)
            ]
        self.host = str(server_host)
        self.port = int(server_port)

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 7200
        self._wait_for_server(url=self.url_for("health"),
                              timeout=max_wait_seconds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

    def _poll(self) -> Optional[int]:
        """Subclasses override this method to customize process polling"""
        return self.proc.poll()

    def hang_until_terminated(self) -> None:
        """
        Wait until the server process terminates.
        This is for headless mode, where the api server
        process only exists in the leader node.
        """
        client = requests
        try:
            while True:
                try:
                    resp = client.get(self.url_for("health"), timeout=5)
                    if resp.status_code != 200:
                        break
                    time.sleep(5)
                except Exception:
                    break
        finally:
            if isinstance(client, httpx.Client):
                client.close()

    def _wait_for_server(self, *, url: str, timeout: float):
        # run health check
        start = time.time()
        client = requests
        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                # this exception can only be raised by requests.get,
                # which means the server is not ready yet.
                # the stack trace is not useful, so we suppress it
                # by using `raise from None`.
                result = self._poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None

                time.sleep(5)
                if time.time() - start > timeout:
                    raise RuntimeError(
                        "Server failed to start in time.") from None

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def get_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.OpenAI(
            base_url=self.url_for("v1"),
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = 600
        return openai.AsyncOpenAI(base_url=self.url_for("v1"),
                                  api_key=self.DUMMY_API_KEY,
                                  max_retries=0,
                                  **kwargs)


class VllmRunner:

    def __init__(
        self,
        model_name: str,
        task: TaskOption = "auto",
        tokenizer_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        # Use smaller max model length, otherwise bigger model cannot run due
        # to kv cache size limit.
        max_model_len: int = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: bool = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            task=task,
            tokenizer=tokenizer_name,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=True,
            dtype=dtype,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            disable_log_stats=disable_log_stats,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
            quantization=quantization,
            **kwargs,
        )

    def get_inputs(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[TextPrompt]:
        if images is not None:
            assert len(prompts) == len(images)

        if videos is not None:
            assert len(prompts) == len(videos)

        if audios is not None:
            assert len(prompts) == len(audios)

        inputs = [TextPrompt(prompt=prompt) for prompt in prompts]
        if images is not None:
            for i, image in enumerate(images):
                if image is not None:
                    inputs[i]["multi_modal_data"] = {"image": image}

        if videos is not None:
            for i, video in enumerate(videos):
                if video is not None:
                    inputs[i]["multi_modal_data"] = {"video": video}

        if audios is not None:
            for i, audio in enumerate(audios):
                if audio is not None:
                    inputs[i]["multi_modal_data"] = {"audio": audio}

        return inputs

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params)

        outputs: List[Tuple[List[List[int]], List[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: List[List[int]] = []
            req_sample_output_strs: List[str] = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    @staticmethod
    def _final_steps_generate_w_logprobs(
        req_outputs: List[RequestOutput],
    ) -> List[TokensTextLogprobsPromptLogprobs]:
        outputs: List[TokensTextLogprobsPromptLogprobs] = []
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs,
                            req_output.prompt_logprobs))
        return outputs

    def generate_w_logprobs(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params)

        toks_str_logsprobs_prompt_logprobs = (
            self._final_steps_generate_w_logprobs(req_outputs))
        # Omit prompt logprobs if not required by sampling params
        return ([x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
                if sampling_params.prompt_logprobs is None else
                toks_str_logsprobs_prompt_logprobs)

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[Tuple[List[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts,
                                greedy_params,
                                images=images,
                                videos=videos,
                                audios=audios)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop: Optional[List[str]] = None,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids,
            stop=stop)

        return self.generate_w_logprobs(prompts,
                                        greedy_logprobs_params,
                                        images=images,
                                        audios=audios,
                                        videos=videos)

    def encode(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[List[float]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.embed(inputs)
        return [req_output.outputs.embedding for req_output in req_outputs]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        clear_ascend_config()
        cleanup_dist_env_and_memory()


class HfRunner:

    def get_default_device(self):

        return ("cpu"
                if current_platform.is_cpu() else current_platform.device_type)

    def wrap_device(self, x: _T, device: Optional[str] = None) -> _T:
        if x is None or isinstance(x, (bool, )):
            return x

        if device is None:
            device = self.device

        if isinstance(x, dict):
            return {k: self.wrap_device(v, device) for k, v in x.items()}

        if hasattr(x, "device") and x.device.type == device:
            return x

        return x.to(device)

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        *,
        model_kwargs: Optional[dict[str, Any]] = None,
        trust_remote_code: bool = True,
        is_sentence_transformer: bool = False,
        is_cross_encoder: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: type[_BaseAutoModelClass] = AutoModelForCausalLM,
    ) -> None:
        model_name = maybe_model_redirect(model_name)
        self.model_name = model_name

        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.device = self.get_default_device()
        self.dtype = torch_dtype = _get_and_verify_dtype(
            self.model_name,
            self.config,
            dtype=dtype,
            is_pooling_model=is_sentence_transformer or is_cross_encoder,
        )

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        model_kwargs.setdefault("torch_dtype", torch_dtype)

        if is_sentence_transformer:
            # Lazy init required for AMD CI
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        elif is_cross_encoder:
            # Lazy init required for AMD CI
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                model_name,
                device=self.device,
                automodel_args=model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = auto_cls.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )

            # in case some unquantized custom models are not in same dtype
            if (getattr(model, "quantization_method", None) is None
                    and any(p.dtype != self.dtype
                            for p in model.parameters())):
                model = model.to(dtype=self.dtype)

            if (getattr(model, "quantization_method", None) != "bitsandbytes"
                    and len({p.device
                             for p in model.parameters()}) < 2):
                model = model.to(device=self.device)

            self.model = model

        if not skip_tokenizer_init:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )

        # don't put this import at the top level
        # it will call torch.cuda.device_count()
        from transformers import AutoProcessor  # noqa: F401
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if skip_tokenizer_init:
            self.tokenizer = self.processor.tokenizer

    def encode(self, prompts: list[str], *args,
               **kwargs) -> list[list[torch.Tensor]]:
        return self.model.encode(prompts, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def ilama_lora_files():
    return snapshot_download(repo_id="vllm-ascend/ilama-text2sql-spider")


def qwen_prompt(questions: List[str]) -> List[str]:
    placeholder = "<|image_pad|>"
    return [("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
             f"{q}<|im_end|>\n<|im_start|>assistant\n") for q in questions]


PROMPT_TEMPLATES = {
    "qwen2.5vl": qwen_prompt,
}


@pytest.fixture(params=list(PROMPT_TEMPLATES.keys()))
def prompt_template(request):
    return PROMPT_TEMPLATES[request.param]


@pytest.fixture
def load_config():
    """读取配置文件"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'configs', 'test_config.yaml')
    print(config_path)
    config_path = os.getenv('TEST_CONFIG', config_path)
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'配置文件不存在：{config_path}')
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("no config find")


class RemoteEPDServer:

    def get_proxy(self) -> Proxy:
        return self.p

    def _run_server(self, server_cmd: list[str],
                    env_dict: Optional[dict[str, str]]) -> None:
        """Subclasses override this method to customize server process launch
        """
        env = os.environ.copy()
        # the current process might initialize npu,
        # to be safe, we should use spawn method
        if env_dict is not None:
            env.update(env_dict)
        proc = subprocess.Popen(
            server_cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._proc_list.append(proc)

    def _read_output(self, pipe, prefix):
        """在单独线程中读取输出"""
        try:
            with pipe:
                for line in iter(pipe.readline, ''):
                    if line:  # 避免空行
                        print(f"{prefix}: {line}", end='')
        except Exception as e:
            print(f"error: {e}")

    def _run_server_new_session(self, server_cmd: list[str],
                                env_dict: Optional[dict[str, str]]) -> None:
        """Subclasses override this method to customize server process launch
        """
        env = os.environ.copy()
        # the current process might initialize npu,
        # to be safe, we should use spawn method
        if env_dict is not None:
            env.update(env_dict)
        proc = subprocess.Popen(
            server_cmd,
            cwd=None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            text=True,
            bufsize=1,  # 行缓冲
            universal_newlines=True)

        # 创建线程读取输出
        stdout_thread = threading.Thread(target=self._read_output,
                                         args=(proc.stdout, "UVICORN-STDOUT"),
                                         daemon=True)
        stderr_thread = threading.Thread(target=self._read_output,
                                         args=(proc.stderr, "UVICORN-STDERR"),
                                         daemon=True)

        stdout_thread.start()
        stderr_thread.start()
        self._proc_list.append(proc)

    def _start_api_server(self) -> None:
        api_server_args = [
            "--host", "127.0.0.1", "--port",
            str(self.api_server_port), "--model", self.model, "--proxy-addr",
            self.proxy_addr, "--e-addr-list", ",".join(self.e_addr_list),
            "--pd-addr-list", ",".join(self.pd_addr_list)
        ]
        if self.is_image_load:
            api_server_args.append("--is-load-image")
        api_server_args = ["python", "tools/api_server.py", *api_server_args]
        self._run_server_new_session(api_server_args, None)

    def _start_vllm(self):
        if self.env_dict is None:
            self.env_dict = dict()
        self.env_dict['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"
        self.env_dict['VLLM_USE_V1'] = "1"
        if isinstance(self.e_serve_args, str):
            self.e_serve_args = shlex.split(self.e_serve_args)
        if isinstance(self.pd_serve_args, str):
            self.pd_serve_args = shlex.split(self.pd_serve_args)
        else:
            self.e_serve_args = [
                "python", "-m", "vllm.entrypoints.disaggregated.worker",
                *self.e_serve_args
            ]
            self.pd_serve_args = [
                "python", "-m", "vllm.entrypoints.disaggregated.worker",
                *self.pd_serve_args
            ]

        if "--proxy-addr" not in self.e_serve_args and "--proxy-addr" not in self.pd_serve_args:
            # defaut proxy-addr is /tmp/proxy
            self.e_serve_args = self.e_serve_args + [
                "--proxy-addr", self._default_addr_prefix + "proxy"
            ]
            self.pd_serve_args = self.pd_serve_args + [
                "--proxy-addr", self._default_addr_prefix + "proxy"
            ]
        else:
            try:
                index_e = self.e_serve_args.index("--proxy-addr")
                index_pd = self.pd_serve_args.index("--proxy-addr")
            except ValueError:
                print("e instance proxy addr must be same as pd instance")
                return
            self.proxy_addr = self.e_serve_args[index_e + 1]

        if "--model" not in self.e_serve_args or "--model" not in self.pd_serve_args:
            raise ValueError("must carry --model")

        else:
            index_e = self.e_serve_args.index("--model")
            self.model = self.e_serve_args[index_e + 1]

        for i in range(self.e_num):
            self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            if "--worker-addr" not in self.e_serve_args:
                # defaut encode-addr is /tmp/encode_{i}
                self.e_serve_args = self.e_serve_args + [
                    "--worker-addr",
                    self._default_addr_prefix + "encode_" + str(i)
                ]
                self.e_addr_list.append(self._default_addr_prefix + "encode_" +
                                        str(i))
            else:
                index_e = self.e_serve_args.index("--worker-addr")
                self.e_addr_list.append(self.e_serve_args[index_e + 1])
            self._run_server(self.e_serve_args, self.env_dict)
        for i in range(self.pd_num):
            self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i + self.e_num)
            if "--worker-addr" not in self.pd_serve_args:
                # defaut worker-addr is /tmp/pd_{i}
                self.pd_serve_args = self.pd_serve_args + [
                    "--worker-addr", self._default_addr_prefix + "pd_" + str(i)
                ]
                self.pd_addr_list.append(self._default_addr_prefix + "pd_" +
                                         str(i))
            else:
                index_pd = self.pd_serve_args.index("--worker-addr")
                self.pd_addr_list.append(self.pd_serve_args[index_pd + 1])
            self._run_server(self.pd_serve_args, self.env_dict)

    async def _wait_for_vllm_server(self, max_wait_seconds) -> None:
        sleep_times = 10
        timeout_times = 3
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            tasks_0 = [
                asyncio.create_task(
                    asyncio.wait_for(self.p.check_health(
                        ServerType.E_INSTANCE, iid),
                                     timeout=timeout_times))
                for iid in range(self.e_num)
            ]
            tasks_1 = [
                asyncio.create_task(
                    asyncio.wait_for(self.p.check_health(
                        ServerType.PD_INSTANCE, iid),
                                     timeout=timeout_times))
                for iid in range(self.pd_num)
            ]
            tasks = tasks_0 + tasks_1

            results = await asyncio.gather(*tasks, return_exceptions=True)
            if all([isinstance(result, bool) and result
                    for result in results]):
                print("All instances are ready")
                return
            else:
                print(f"current results: {results}")
                await asyncio.sleep(sleep_times)

        raise RuntimeError("epd instance start failed!")

    def _kill_process_tree(self, pid):
        """kill process and its children"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            gone, still_alive = psutil.wait_procs(children, timeout=10)

            for child in still_alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            try:
                parent.terminate()
                parent.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            pass

    async def _wait_for_api_server(self,
                                   timeout: int = 300,
                                   check_interval: float = 0.5) -> bool:

        base_url = f"http://127.0.0.1:{self.api_server_port}"
        health_url = f"{base_url}/health"

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    print(
                        f"✅ api server is ready: {data.get('status', 'unknown')}"
                    )
                    return True
                else:
                    print(
                        f"❌ api server start error, http error code: {response.status_code}"
                    )
            except requests.exceptions.ConnectionError:
                print("⏳ waiting for ready ...")
            except requests.exceptions.RequestException as e:
                print(f"api server start error: {e}")

            await asyncio.sleep(check_interval)
        print("api server start timeout")
        return False

    def __init__(self,
                 start_mode: Optional[str],
                 e_num: Optional[int],
                 pd_num: Optional[int],
                 e_serve_args: Union[list[str], str],
                 pd_serve_args: Union[list[str], str],
                 api_server_port: Optional[int] = 10001,
                 is_image_load: Optional[bool] = False,
                 env_dict: Optional[dict[str, str]] = None) -> None:
        self._proc_list = list()
        self.e_num = e_num
        self.pd_num = pd_num
        self.start_mode = start_mode
        self.is_image_load = is_image_load
        self.api_server_port = api_server_port
        self.e_addr_list = list()
        self.pd_addr_list = list()

        self.model = str()
        self.e_serve_args = e_serve_args
        self.pd_serve_args = pd_serve_args
        self.env_dict = env_dict
        self._default_addr_prefix = "/tmp/"
        self.proxy_addr = self._default_addr_prefix + "proxy"

    async def __aenter__(self):
        # start with
        max_wait_seconds = 1800
        self._start_vllm()
        self.p = Proxy(proxy_addr=self.proxy_addr,
                       encode_addr_list=self.e_addr_list,
                       pd_addr_list=self.pd_addr_list,
                       enable_health_monitor=True,
                       model_name=self.model)
        await self._wait_for_vllm_server(max_wait_seconds=max_wait_seconds)
        if self.start_mode == "http":
            self.p.shutdown()
            self._start_api_server()
            await self._wait_for_api_server()
        elif self.start_mode == "api":
            self.p.shutdown()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # exit with
        for proc in self._proc_list:
            self._kill_process_tree(proc.pid)
        print("vllm instance and api server is stoping")
        self.p.shutdown()
        print("proxy is stoping")

