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
SHARED_STORAGE_PATH = "/dev/shm/epd/storage"

TENSOR_PARALLELS = [1]
DATASET_NAME = ["simulate_truth"]

MOONCAKE_PRODUCER_CONFIG_PATH = load_config().get("mooncake_config_path") + "producer.json"
MOONCAKE_CONSUMER_CONFIG_PATH = load_config().get("mooncake_config_path") + "consumer.json"

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1pd_mooncake_ipc_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_mooncake_IPC",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e1pd_merge_mooncake_ipc_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_merge_mooncake_IPC",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               is_epd_same_card=True,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_ipc_001(model: str, tp_size: int, dataset_name: str):
    """同图片同请求开启前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_samereq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_IPC_samereq",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_ipc_002(model: str, tp_size: int, dataset_name: str):
    """同图片同请求关闭前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_samereq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_IPC_samereq_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_ipc_003(model: str, tp_size: int, dataset_name: str):
    """同图片跨请求开启前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_IPC_diffreq",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_ipc_004(model: str, tp_size: int, dataset_name: str):
    """同图片跨请求关闭前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_IPC_diffreq_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_ipc_005(model: str, tp_size: int, dataset_name: str):
    """不同图片开启前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_IPC",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_ipc_006(model: str, tp_size: int, dataset_name: str):
    """不同图片关闭前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_IPC_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_3e5pd_mooncake_ipc_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_3E5PD_mooncake_IPC",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=5,
                               e_num=3,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_ipc_acc_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth"),
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
    acc_cases = [{
        "case_type":
            "accuracy",
        "dataset_path":
            os.path.join(DATASET_PATH, "textvqa_subset"),
        "request_conf":
            "vllm_api_general_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "max_out_len":
            2048,
        "batch_size":
            32,
        "temperature":
            0,
        "top_k":
            -1,
        "top_p":
            1,
        "repetition_penalty":
            1,
        "request_rate":
            0,
        "seed":
            77,
        "baseline":81,
        "threshold":1
    }]


    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=acc_cases)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_mooncake_ipc_acc_002(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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
    acc_cases = [{
        "case_type":
            "accuracy",
        "dataset_path":
            os.path.join(DATASET_PATH, "textvqa_subset"),
        "request_conf":
            "vllm_api_general_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "max_out_len":
            2048,
        "batch_size":
            32,
        "temperature":
            0,
        "top_k":
            -1,
        "top_p":
            1,
        "repetition_penalty":
            1,
        "request_rate":
            0,
        "seed":
            77,
        "baseline": 81,
        "threshold": 1
    }]

    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=acc_cases)

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_mooncake_tcp_transfer_protocol_001(model: str, tp_size: int, dataset_name: str):
    """transfer_protocol为空"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    proxy_args = [
        "--transfer-protocol", ""
    ]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9"
    ]
    api_port = 10001
    try:
        RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args,
                               proxy_args=proxy_args)
    except Exception as message:
        print(f"error message is: {str(message)}")
        assert "Invalid value" in str(message), "init success"

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_mooncake_tcp_transfer_protocol_002(model: str, tp_size: int, dataset_name: str):
    """transfer_protocol为异常值"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    proxy_args = [
        "--transfer-protocol", "http"
    ]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "http",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "http",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9"
    ]
    api_port = 10001
    try:
        async with RemoteEPDServer(run_mode="worker",
                                   store_type="mooncake",
                                   proxy_type="api_server",
                                   api_server_port=api_port,
                                   pd_num=2,
                                   e_num=1,
                                   env_dict=env_dict,
                                   e_serve_args=e_server_args,
                                   pd_serve_args=pd_server_args,
                                   mooncake_args=mooncake_args,
                                   proxy_args=proxy_args) as server:

            # warm up
            run_aisbench_cases(model=model,
                               port=api_port,
                               aisbench_cases=warmup_cases,
                               verify=False,
                               save=False)
    except Exception as message:
        print(f"error message is: {str(message)}")
        assert "Invaalid value" in str(message), "init success"

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_mooncake_tcp_transfer_protocol_003(model: str, tp_size: int, dataset_name: str):
    """transfer_protocol为不携带"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9"
    ]
    api_port = 10001
    try:
        RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args)
    except Exception as message:
        print(f"error message is: {str(message)}")
        assert "Invalid value" in str(message), "init success"

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_mooncake_tcp_env_transfer_protocol_001(model: str, tp_size: int, dataset_name: str):
    """TRANSFER_PROTOCOL为空"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = ""
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_tcp_export_transfer_protocol_002(model: str, tp_size: int, dataset_name: str):
    """TRANSFER_PROTOCOL为异常值"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "HTTP"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_tcp_export_transfer_protocol_003(model: str, tp_size: int, dataset_name: str):
    """TRANSFER_PROTOCOL不携带"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e1pd_mooncake_tcp_001(model: str, tp_size: int, dataset_name: str):
    '''
    使用环境变量的方式
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_mooncake_TCP",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e1pd_merge_mooncake_tcp_001(model: str, tp_size: int, dataset_name: str):
    '''
    使用环境变量的方式
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_merge_mooncake_TCP",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               env_dict=env_dict,
                               is_epd_same_card=True,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_tcp_001(model: str, tp_size: int, dataset_name: str):
    '''
    同图片同请求开启前缀缓存
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_samereq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_TCP_samereq",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_tcp_002(model: str, tp_size: int, dataset_name: str):
    '''
    同图片同请求关闭前缀缓存
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_samereq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_TCP_samereq_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

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
async def test_1e2pd_mooncake_tcp_003(model: str, tp_size: int, dataset_name: str):
    '''
    同图片跨请求开启前缀缓存
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_TCP_diffreq",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:
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
async def test_1e2pd_mooncake_tcp_004(model: str, tp_size: int, dataset_name: str):
    '''
    同图片跨请求关闭前缀缓存
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    proxy_args = [
        "--transfer-protocol", "tcp"
    ]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9"
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_TCP_diffreq_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args,
                               proxy_args=proxy_args) as server:
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
async def test_1e2pd_mooncake_tcp_005(model: str, tp_size: int, dataset_name: str):
    '''
    不同图片开启前缀缓存
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    proxy_args = [
        "--transfer-protocol", "tcp"
    ]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9"
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_TCP",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args,
                               proxy_args=proxy_args) as server:
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
async def test_1e2pd_mooncake_tcp_006(model: str, tp_size: int, dataset_name: str):
    '''
    不同图片关闭前缀缓存
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    proxy_args = [
        "--transfer-protocol", "tcp"
    ]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9"
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_mooncake_TCP_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args,
                               proxy_args=proxy_args) as server:
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
async def test_3e5pd_mooncake_tcp_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_3E5PD_mooncake_TCP",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=5,
                               e_num=3,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:
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
async def test_1e1pd_shared_tcp_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_shared_TCP",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               env_dict=env_dict,
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
async def test_1e1pd_merge_shared_tcp_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_merge_shared_TCP",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               env_dict=env_dict,
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
async def test_1e2pd_shared_tcp_001(model: str, tp_size: int, dataset_name: str):
    """相同图片同请求开启前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth_samereq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shared_TCP_samereq",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
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
async def test_1e2pd_shared_tcp_002(model: str, tp_size: int, dataset_name: str):
    """相同图片同请求关闭前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth_samereq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_samereq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shared_TCP_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
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
async def test_1e2pd_shared_tcp_003(model: str, tp_size: int, dataset_name: str):
    """相同图片跨请求开启前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shared_TCP_diffreq",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
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
async def test_1e2pd_shared_tcp_004(model: str, tp_size: int, dataset_name: str):
    """相同图片跨请求关闭前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    proxy_args = [
        "--transfer-protocol", "tcp"
    ]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth_diffreq"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shared_TCP_diffreq_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args) as server:
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
async def test_1e2pd_shared_tcp_005(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    proxy_args = [
        "--transfer-protocol", "tcp"
    ]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shared_TCP",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args) as server:
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
async def test_1e2pd_shared_tcp_006(model: str, tp_size: int, dataset_name: str):
    """相同图片跨请求关闭前缀缓存"""
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    proxy_args = [
        "--transfer-protocol", "tcp"
    ]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "tcp",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shared_TCP_noprefix_caching",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args) as server:
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
async def test_3e5pd_shared_tcp_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
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

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "max_out_len": 150,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_3E5PD_shared_TCP",
        "threshold": 0.97
    }
    aisbench_cases = []
    for i in range(len(request_rate)):
        case_dict["request_rate"] = request_rate[i]
        new_case_dict = copy.deepcopy(case_dict)
        aisbench_cases.append(new_case_dict)
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=5,
                               e_num=3,
                               env_dict=env_dict,
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
async def test_1e2pd_mooncake_tcp_acc_001(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    acc_cases = [{
        "case_type":
            "accuracy",
        "dataset_path":
            os.path.join(DATASET_PATH, "textvqa_subset"),
        "request_conf":
            "vllm_api_general_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "max_out_len":
            2048,
        "batch_size":
            32,
        "temperature":
            0,
        "top_k":
            -1,
        "top_p":
            1,
        "repetition_penalty":
            1,
        "request_rate":
            0,
        "seed":
            77,
        "baseline": 81,
        "threshold": 1
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:

        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=acc_cases)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_mooncake_tcp_acc_002(model: str, tp_size: int, dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "1",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size", str(tp_size), "--enforce-eager",
        "--max-model-len", "10000", "--max-num-batched-tokens",
        "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_CONSUMER_CONFIG_PATH +
        '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]
    mooncake_args = [
        "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
        "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
    ]
    acc_cases = [{
        "case_type":
            "accuracy",
        "dataset_path":
            os.path.join(DATASET_PATH, "textvqa_subset"),
        "request_conf":
            "vllm_api_general_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "max_out_len":
            2048,
        "batch_size":
            32,
        "temperature":
            0,
        "top_k":
            -1,
        "top_p":
            1,
        "repetition_penalty":
            1,
        "request_rate":
            0,
        "seed":
            77,
        "baseline": 81,
        "threshold": 1
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               mooncake_args=mooncake_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=acc_cases)
# @pytest.mark.asyncio
# @pytest.mark.parametrize("model", MODELS)
# @pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
# @pytest.mark.parametrize("dataset_name", DATASET_NAME)
# async def test_1e2pd_mooncake_tcp_001(model: str, tp_size: int, dataset_name: str):
#     '''
#     使用环境变量的方式
#     '''
#     env_dict = {}
#     env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
#     env_dict["TRANSFER_PROTOCOL"] = "tcp"
#     e_server_args = [
#         "--model", model, "--gpu-memory-utilization", "0.0",
#         "--tensor-parallel-size",str(tp_size), "--enforce-eager",
#         "--no-enable-prefix-caching",
#         "--max-model-len", "10000", "--max-num-batched-tokens",
#         "10000", "--max-num-seqs", "1",
#         "--ec-transfer-config",
#         '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
#         MOONCAKE_PRODUCER_CONFIG_PATH +
#         '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
#     ]
#     pd_server_args = [
#         "--model", model, "--gpu-memory-utilization", "0.95",
#         "--tensor-parallel-size", str(tp_size), "--enforce-eager",
#         "--max-model-len", "10000", "--max-num-batched-tokens",
#         "10000", "--max-num-seqs", "128",
#         "--ec-transfer-config",
#         '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
#         MOONCAKE_CONSUMER_CONFIG_PATH +
#         '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
#     ]
#     mooncake_args = [
#         "--rpc_port", "50052", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
#         "--http_metadata_server_port=8082", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
#         "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
#     ]
#     warmup_cases = [{
#         "case_type":
#         "performance",
#         "dataset_path":
#         os.path.join(DATASET_PATH, dataset_name),
#         "request_conf":
#         "vllm_api_stream_chat",
#         "dataset_conf":
#         "textvqa/textvqa_gen",
#         "num_prompts":
#         50,
#         "max_out_len":
#         256,
#         "batch_size":
#         16,
#         "temperature":
#         0.5,
#         "top_k":
#         10,
#         "top_p":
#         0.7,
#         "repetition_penalty":
#         1.2,
#         "request_rate":
#         0,
#         "seed":
#         77,
#     }]
#
#     request_rate = [0.28, 0.78, 1.28, 1.78]
#     case_dict = {
#         "case_type": "performance",
#         "dataset_path": os.path.join(DATASET_PATH, dataset_name),
#         "request_conf": "vllm_api_stream_chat",
#         "dataset_conf": "textvqa/textvqa_gen",
#         "num_prompts": 200,
#         "max_out_len": 150,
#         "batch_size": 128,
#         "temperature": 0.5,
#         "top_k": 10,
#         "top_p": 0.7,
#         "repetition_penalty": 1.2,
#         "request_rate": 0.28,
#         "baseline": 1,
#         "seed": 77,
#         "result_file_name": f"{dataset_name}_1E2PD_mooncake",
#         "threshold": 0.97
#     }
#     aisbench_cases = []
#     for i in range(len(request_rate)):
#         case_dict["request_rate"] = request_rate[i]
#         new_case_dict = copy.deepcopy(case_dict)
#         aisbench_cases.append(new_case_dict)
#     api_port = 10001
#     async with RemoteEPDServer(run_mode="worker",
#                                store_type="mooncake",
#                                proxy_type="api_server",
#                                api_server_port=api_port,
#                                pd_num=2,
#                                e_num=1,
#                                env_dict=env_dict,
#                                e_serve_args=e_server_args,
#                                pd_serve_args=pd_server_args,
#                                mooncake_args=mooncake_args) as server:
#
#         # warm up
#         run_aisbench_cases(model=model,
#                            port=api_port,
#                            aisbench_cases=warmup_cases,
#                            verify=False,
#                            save=False)
#         # aisbench test
#         run_aisbench_cases(model=model,
#                            port=api_port,
#                            aisbench_cases=aisbench_cases)
#
#
#
# @pytest.mark.asyncio
# @pytest.mark.parametrize("model", MODELS)
# @pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
# @pytest.mark.parametrize("dataset_name", DATASET_NAME)
# async def test_1e2pd_mooncake_tcp_002(model: str, tp_size: int, dataset_name: str):
#     '''
#     使用命令行的方式
#     '''
#     env_dict = {}
#     env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
#     e_server_args = [
#         "--model", model, "--gpu-memory-utilization", "0.0","--transfer-protocol", "tcp",
#         "--tensor-parallel-size",str(tp_size), "--enforce-eager",
#         "--no-enable-prefix-caching",
#         "--max-model-len", "10000", "--max-num-batched-tokens",
#         "10000", "--max-num-seqs", "1",
#         "--ec-transfer-config",
#         '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
#         MOONCAKE_PRODUCER_CONFIG_PATH +
#         '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
#     ]
#     pd_server_args = [
#         "--model", model, "--gpu-memory-utilization", "0.95","--transfer-protocol", "tcp",
#         "--tensor-parallel-size", str(tp_size), "--enforce-eager",
#         "--max-model-len", "10000", "--max-num-batched-tokens",
#         "10000", "--max-num-seqs", "128",
#         "--ec-transfer-config",
#         '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
#         MOONCAKE_CONSUMER_CONFIG_PATH +
#         '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
#     ]
#     mooncake_args = [
#         "--rpc_port", "50051", "--enable_http_metadata_server=true", "--http_metadata_server_host=0.0.0.0",
#         "--http_metadata_server_port=8081", "--rpc_thread_num", "8", "--default_kv_lease_ttl", "10000",
#         "eviction_ratio", "0.05", "--eviction_high_watermark_ratio", "0.9"
#     ]
#     proxy_args = [
#         "--transfer-protocol", "tcp"
#     ]
#     warmup_cases = [{
#         "case_type":
#         "performance",
#         "dataset_path":
#         os.path.join(DATASET_PATH, dataset_name),
#         "request_conf":
#         "vllm_api_stream_chat",
#         "dataset_conf":
#         "textvqa/textvqa_gen",
#         "num_prompts":
#         50,
#         "max_out_len":
#         256,
#         "batch_size":
#         16,
#         "temperature":
#         0.5,
#         "top_k":
#         10,
#         "top_p":
#         0.7,
#         "repetition_penalty":
#         1.2,
#         "request_rate":
#         0,
#         "seed":
#         77,
#     }]
#
#     request_rate = [0.28, 0.56, 0.84, 1.12, 1.4, 1.68]
#     case_dict = {
#         "case_type": "performance",
#         "dataset_path": os.path.join(DATASET_PATH, dataset_name),
#         "request_conf": "vllm_api_stream_chat",
#         "dataset_conf": "textvqa/textvqa_gen",
#         "num_prompts": 200,
#         "max_out_len": 150,
#         "batch_size": 128,
#         "temperature": 0.5,
#         "top_k": 10,
#         "top_p": 0.7,
#         "repetition_penalty": 1.2,
#         "request_rate": 0.28,
#         "baseline": 1,
#         "seed": 77,
#         "result_file_name": f"{dataset_name}_1E2PD_mooncake",
#         "threshold": 0.97
#     }
#     aisbench_cases = []
#     for i in range(len(request_rate)):
#         case_dict["request_rate"] = request_rate[i]
#         new_case_dict = copy.deepcopy(case_dict)
#         aisbench_cases.append(new_case_dict)
#     api_port = 10001
#     async with RemoteEPDServer(run_mode="worker",
#                                store_type="mooncake",
#                                proxy_type="api_server",
#                                api_server_port=api_port,
#                                pd_num=2,
#                                e_num=1,
#                                env_dict=env_dict,
#                                e_serve_args=e_server_args,
#                                pd_serve_args=pd_server_args,
#                                mooncake_args=mooncake_args,
#                                proxy_args=proxy_args) as server:
#
#         # warm up
#         run_aisbench_cases(model=model,
#                            port=api_port,
#                            aisbench_cases=warmup_cases,
#                            verify=False,
#                            save=False)
#         # aisbench test
#         run_aisbench_cases(model=model,
#                            port=api_port,
#                            aisbench_cases=aisbench_cases)


