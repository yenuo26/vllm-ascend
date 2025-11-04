import os

import pytest

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
async def test_1e1pd_merge_001(model: str, tp_size: int):
    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "30000", "--max-num-seqs", "1",
        "--enforce-eager", "--gpu-memory-utilization", "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--max-model-len", "30000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    aisbench_cases = [{
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
    }, {
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
        0.78,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold":
        0.97
    }, {
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
        1.28,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold":
        0.97
    }, {
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
        1.78,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E1PD_merge",
        "threshold":
        0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(start_mode="http",
                               api_server_port=api_port,
                               pd_num=1,
                               e_num=1,
                               is_epd_same_card=True,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_1e1pd_001(model: str, tp_size: int):
    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "30000", "--max-num-seqs", "1",
        "--enforce-eager", "--gpu-memory-utilization", "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--max-model-len", "30000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    aisbench_cases = [{
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
        0.56,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E1PD",
        "threshold":
        0.97
    }, {
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
        1.56,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E1PD",
        "threshold":
        0.97
    }, {
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
        2.56,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E1PD",
        "threshold":
        0.97
    }, {
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
        3.56,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E1PD",
        "threshold":
        0.97
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


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_1e2pd_001(model: str, tp_size: int):
    e_server_args = [
        "--no-enable-prefix-caching", "--model", model,
        "--tensor-parallel-size",
        str(tp_size), "--max-model-len", "30000", "--max-num-seqs", "1",
        "--enforce-eager", "--gpu-memory-utilization", "0.0",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        "--model", model, "--max-model-len", "30000", "--tensor-parallel-size",
        str(tp_size), "--max-num-seqs", "128", "--gpu-memory-utilization",
        "0.95", "--enforce-eager", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]

    aisbench_cases = [{
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
        0.84,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E2PD",
        "threshold":
        0.97
    }, {
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
        2.34,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E2PD",
        "threshold":
        0.97
    }, {
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
        3.84,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E2PD",
        "threshold":
        0.97
    }, {
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
        5.34,
        "baseline":
        1,
        "seed":
        77,
        "result_file_name":
        "qwen2_5_vl_7b_perf_custom_1E2PD",
        "threshold":
        0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(start_mode="http",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)
