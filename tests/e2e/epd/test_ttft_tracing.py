import os
import copy
import pytest
import datetime
import time

from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
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
        "--model", model, "--max-model-len", "30000",
        "--max-num-batched-tokens", "40000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    env_dict = {"TIMECOUNT_ENABLED": "1","VLLM_LOG_STATS_INTERVAL": "10"}

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
            100,
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

    request_rate = [0.28, 0.78, 1.28, 1.78, 2.28, 2.78, 3.28]
    case_dict = {
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "num_prompts":
            200,
        "max_out_len":
            256,
        "batch_size":
            128,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0.28,
        "baseline":
            1,
        "seed":
            77,
        "result_file_name":
            "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold":
            0.97
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
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases, verify=False, save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_timeout_enabled_001(model: str, tp_size: int):
    """timeout_enabled为1，实例拉起成功，打印对应统计时间"""
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
        "--model", model, "--max-model-len", "30000",
        "--max-num-batched-tokens", "40000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    env_dict = {"TIMECOUNT_ENABLED": "1","VLLM_LOG_STATS_INTERVAL": "10"}

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
            100,
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
            0.0,
        "seed":
            77,
    }]

    request_rate = [0.28]
    case_dict = {
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "num_prompts":
            200,
        "max_out_len":
            256,
        "batch_size":
            128,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0.28,
        "baseline":
            1,
        "seed":
            77,
        "result_file_name":
            "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold":
            0.97
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
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases, verify=False, save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_timeout_enabled_002(model: str, tp_size: int):
    """timeout_enabled为非0/1值，拉起实例失败，打印对应报错信息"""
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
        "--model", model, "--max-model-len", "30000",
        "--max-num-batched-tokens", "40000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    env_dict = {"TIMECOUNT_ENABLED": "0.5","VLLM_LOG_STATS_INTERVAL": "10"}

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
            100,
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
            0.0,
        "seed":
            77,
    }]

    request_rate = [0.28]
    case_dict = {
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "num_prompts":
            200,
        "max_out_len":
            256,
        "batch_size":
            128,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0.28,
        "baseline":
            1,
        "seed":
            77,
        "result_file_name":
            "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold":
            0.97
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
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases, verify=False, save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_vllm_log_stats_interval_001(model: str, tp_size: int):
    """vllm_log_stats_interval为20，实例拉起成功，打印对应统计时间"""
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
        "--model", model, "--max-model-len", "30000",
        "--max-num-batched-tokens", "40000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    env_dict = {"TIMECOUNT_ENABLED": "1","VLLM_LOG_STATS_INTERVAL": "20"}

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
            100,
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
            0.0,
        "seed":
            77,
    }]

    request_rate = [0.28]
    case_dict = {
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "num_prompts":
            200,
        "max_out_len":
            256,
        "batch_size":
            128,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0.28,
        "baseline":
            1,
        "seed":
            77,
        "result_file_name":
            "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold":
            0.97
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
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases, verify=False, save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_performance_001(model: str, tp_size: int):
    """开启计时功能，QPS从0.28-1.7，实例拉起成功，打印对应统计时间，与关闭计时功能相比，看TTFT，TPOT是否劣化"""
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
        "--model", model, "--max-model-len", "30000",
        "--max-num-batched-tokens", "40000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    env_dict = {"TIMECOUNT_ENABLED": "1","VLLM_LOG_STATS_INTERVAL": "10"}

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
            0.0,
        "seed":
            77,
    }]

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "num_prompts":
            100,
        "max_out_len":
            256,
        "batch_size":
            128,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0.28,
        "baseline":
            1,
        "seed":
            77,
        "result_file_name":
            "94_performance_001_open",
        "threshold":
            0.97
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
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases, verify=False, save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_performance_002(model: str, tp_size: int):
    """关闭计时功能，QPS从0.28-1.7，实例拉起成功，打印对应统计时间"""
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
        "--model", model, "--max-model-len", "30000",
        "--max-num-batched-tokens", "40000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    env_dict = {"TIMECOUNT_ENABLED": "0","VLLM_LOG_STATS_INTERVAL": "10"}

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
            0.0,
        "seed":
            77,
    }]

    request_rate = [0.28, 0.78, 1.28, 1.78]
    case_dict = {
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "num_prompts":
            100,
        "max_out_len":
            256,
        "batch_size":
            128,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0.28,
        "baseline":
            1,
        "seed":
            77,
        "result_file_name":
            "94_performance_001_close",
        "threshold":
            0.97
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
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases, verify=False, save=False)
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_longterm_001(model: str, tp_size: int):
    """开启计时功能，QPS从0.28，实例拉起成功，持续请求2h，打印对应统计时间，"""
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
        "--model", model, "--max-model-len", "30000",
        "--max-num-batched-tokens", "40000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    env_dict = {"TIMECOUNT_ENABLED": "1","VLLM_LOG_STATS_INTERVAL": "10"}

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
            100,
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
            0.0,
        "seed":
            77,
    }]

    request_rate = [0.28]
    case_dict = {
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen",
        "num_prompts":
            200,
        "max_out_len":
            256,
        "batch_size":
            128,
        "temperature":
            0.5,
        "top_k":
            10,
        "top_p":
            0.7,
        "repetition_penalty":
            1.2,
        "request_rate":
            0.28,
        "baseline":
            1,
        "seed":
            77,
        "result_file_name":
            "94_longterm_001",
        "threshold":
            0.97
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
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases, verify=False, save=False)
        print("sleep 5min")
        time.sleep(5*60)
        # aisbench test for 2h
        # end_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        #
        # while datetime.datetime.now() < end_time:
        #     run_aisbench_cases(model=model,
        #                        port=api_port,
        #                        aisbench_cases=aisbench_cases)

        print("2 hour test completed")