import uuid
import numpy as np
from PIL import Image

import pytest
import os
from vllm import SamplingParams
from vllm.multimodal.image import convert_image_mode
from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases

model_path = load_config().get("model_path")
MODELS = [
    os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")
]
DATASET_PATH = load_config().get("dataset_path")

TENSOR_PARALLELS = [1]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_base_001(model: str, tp_size: int):
    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "30000", "--max-num-batched-tokens",
        "40000", "--max-num-seqs", "1", "--enforce-eager",
        "--gpu-memory-utilization", "0.0", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--no-enable-prefix-caching", "--model", model, "--max-model-len",
        "30000", "--tensor-parallel-size",
        str(tp_size), "--max-num-batched-tokens", "40000", "--max-num-seqs",
        "400", "--gpu-memory-utilization", "0.9", "--enforce-eager",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "textvqa"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 512,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0,
        "top_k": -1,
        "top_p": 1,
        "repetition_penalty": 1,
        "request_rate": 0,
        "baseline": 1,
        "threshold": 0.97
    },{
    "case_type": "accuracy",
    "dataset_path": os.path.join(DATASET_PATH, "textvqa-subset"),
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "textvqa/textvqa_gen_base64",
    "max_out_len": 2048,
    "batch_size": 128,
    "baseline": 81,
    "threshold": 5
    }]
    api_port = 10001
    async with RemoteEPDServer(start_mode="http",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)
