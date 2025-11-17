import os

import pytest
import pytest_asyncio
import copy

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from tools.aisbench import create_result_plot

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")

TENSOR_PARALLELS = [1]
#DATASET_NAME = ["simulate_truth"]
DATASET_NAME = ["image_2", "image_3", "image_4"]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"


@pytest_asyncio.fixture(scope="session")
async def teardown():
    yield
    for dataset in DATASET_NAME:
        create_result_plot(result_file_names=[
            f"qwen2_5_vl_7b_{dataset}_PD_mix",
            f"qwen2_5_vl_7b_{dataset}_1E1PD_sc",
            f"qwen2_5_vl_7b_{dataset}_1E2PD", f"qwen2_5_vl_7b_{dataset}_1E1PD"
        ],result_figure_prefix=dataset)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_pd_mix_001(model: str, tp_size: int, dataset_name: str, teardown):
    api_port = 10001
    vllm_server_args = [
        "--port",
        str(api_port), "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "100", "--enforce-eager",
        "--gpu-memory-utilization", "0.95"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, dataset_name),
        "request_conf":
        "vllm_api_stream_chat",
        "dataset_conf":
        "textvqa/textvqa_gen",
        "num_prompts":
        50,
        "max_out_len":
        256,
        "batch_size":
        16,
        "temperature":
        0.5,
        "top_k":
        10,
        "top_p":
        0.7,
        "repetition_penalty":
        1.2,
        "request_rate":
        0,
        "seed":
        77,
    }]

    request_rate = [0.28, 0.56, 0.84, 1.12, 1.4, 1.68]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"qwen2_5_vl_7b_{dataset_name}_PD_mix",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    with RemoteOpenAIServer(model,
                            vllm_server_args,
                            server_host="127.0.0.1",
                            server_port=api_port,
                            auto_port=False) as server:

        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1pd_sharecard_001(model: str, tp_size: int, dataset_name: str, teardown):
    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1", "--enforce-eager",
        "--gpu-memory-utilization", "0.0", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "100", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, dataset_name),
        "request_conf":
        "vllm_api_stream_chat",
        "dataset_conf":
        "textvqa/textvqa_gen",
        "num_prompts":
        50,
        "max_out_len":
        256,
        "batch_size":
        16,
        "temperature":
        0.5,
        "top_k":
        10,
        "top_p":
        0.7,
        "repetition_penalty":
        1.2,
        "request_rate":
        0,
        "seed":
        77,
    }]

    request_rate = [0.28, 0.56, 0.84, 1.12, 1.4, 1.68]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"qwen2_5_vl_7b_{dataset_name}_1E1PD_sc",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(start_mode="http",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               is_epd_same_card=True,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:

        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1pd_001(model: str, tp_size: int, dataset_name: str, teardown):
    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1", "--enforce-eager",
        "--gpu-memory-utilization", "0.0", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "100", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, dataset_name),
        "request_conf":
        "vllm_api_stream_chat",
        "dataset_conf":
        "textvqa/textvqa_gen",
        "num_prompts":
        50,
        "max_out_len":
        256,
        "batch_size":
        16,
        "temperature":
        0.5,
        "top_k":
        10,
        "top_p":
        0.7,
        "repetition_penalty":
        1.2,
        "request_rate":
        0,
        "seed":
        77,
    }]

    request_rate = [0.56, 1.12, 1.68, 2.24, 2.8, 3.36]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"qwen2_5_vl_7b_{dataset_name}_1E1PD",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(start_mode="http",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=2,
                           aisbench_cases=aisbench_cases)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_001(model: str, tp_size: int,dataset_name: str, teardown):
    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1", "--enforce-eager",
        "--gpu-memory-utilization", "0.0", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "100", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, dataset_name),
        "request_conf":
        "vllm_api_stream_chat",
        "dataset_conf":
        "textvqa/textvqa_gen",
        "num_prompts":
        50,
        "max_out_len":
        256,
        "batch_size":
        16,
        "temperature":
        0.5,
        "top_k":
        10,
        "top_p":
        0.7,
        "repetition_penalty":
        1.2,
        "request_rate":
        0,
        "seed":
        77,
    }]
    request_rate = [0.84, 1.68, 2.52, 3.36, 4.2, 5.04]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 256,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"qwen2_5_vl_7b_{dataset_name}_1E2PD",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(start_mode="http",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=3,
                           aisbench_cases=aisbench_cases)
