import copy
import os

import pytest

from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from tests.e2e.nightly.multi_node.config.utils import get_cluster_ips

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")

TENSOR_PARALLELS = [1]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"
ENABLE_PREFIX = [True, False]




DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1p1d_ipc_storage_mooncake_001(model: str, tp_size: int,
                                               dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"shared_storage_path":"' +
            SHARED_STORAGE_PATH +
            '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_producer","mooncake_rpc_port": "50051",'
            '"kv_connector_extra_config": {"local_hostname": "localhost", "metadata_server": "http://localhost:8081/metadata",'
            '"protocol": "tcp", "device_name": "", "master_server_address": "localhost:50051", "global_segment_size": 30000000000}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_consumer","mooncake_rpc_port": "50051",'
            '"kv_connector_extra_config": {"local_hostname": "localhost", "metadata_server": "http://localhost:8081/metadata",'
            '"protocol": "tcp", "device_name": "", "master_server_address": "localhost:50051", "global_segment_size": 30000000000}}'
        ]
    ]

    mooncake_args = [
        "--rpc_port", "50051", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8081", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9"
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    request_rate = [5.34]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_storage_mooncake",
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
                               kv_store_type="mooncake",
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
                           card_num=3,
                           aisbench_cases=aisbench_cases)


REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["image_4", "simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("enable_prefix", ENABLE_PREFIX)
async def test_1e1p1d_ipc_mooncake_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate:float, enable_prefix:bool):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"local_hostname":"localhost","metadata_server": "http://localhost:8085/metadata",'
        '"global_segment_size": 32212254720, "local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        '"master_server_address": "localhost:50055","replica_num": 1, "fast_transfer":true, "fast_transfer_buffer_size": 1'
        '"ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"local_hostname":"localhost","metadata_server": "http://localhost:8085/metadata",'
            '"global_segment_size": 32212254720, "local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            '"master_server_address": "localhost:50055","replica_num": 1, "fast_transfer":true, "fast_transfer_buffer_size": 1'
            '"ec_max_num_scheduled_tokens": "1000000000000000000"},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"},'
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_producer","mooncake_rpc_port": "50051"}'
            '"kv_connector_extra_config": {"local_hostname": "localhost", "metadata_server": "http://localhost:8081/metadata",'
            '"protocol": "tcp", "device_name": "", "master_server_address": "localhost:50051", "global_segment_size": 30000000000}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_consumer","mooncake_rpc_port": "50051"}'
            '"kv_connector_extra_config": {"local_hostname": "localhost", "metadata_server": "http://localhost:8081/metadata",'
            '"protocol": "tcp", "device_name": "", "master_server_address": "localhost:50051", "global_segment_size": 30000000000}}'
        ]
    ]

    mooncake_args = [[
        "--rpc_port", "50051", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8081", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9", "--metrics_port 9005"
    ],[
        "--rpc_port", "50055", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8085", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9", "--metrics_port 9004"
    ]]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    aisbench_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate*3,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_mooncake",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
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
                           card_num=3,
                           aisbench_cases=aisbench_cases)



@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", ["image_4"])
async def test_1e1p1d_ipc_mooncake_002(model: str, tp_size: int,
                                       dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
            MOONCAKE_CONSUMER_CONFIG_PATH +
            '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_producer","mooncake_rpc_port": "50051"}'

        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_consumer","mooncake_rpc_port": "50051"}'
        ]
    ]

    mooncake_args = [
        "--rpc_port", "50051", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8081", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9"
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    request_rate = [0.42, 1.17, 1.92, 2.67]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "batch_size": 128,
        "max_out_len": 150,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_mooncake",
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
                               kv_store_type="mooncake",
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
                           card_num=3,
                           aisbench_cases=aisbench_cases)



@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", ["image_4"])
async def test_1e1p1d_ipc_mooncake_003(model: str, tp_size: int,
                                       dataset_name: str):
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--no-enable-prefix-caching",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
            MOONCAKE_CONSUMER_CONFIG_PATH +
            '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_producer","mooncake_rpc_port": "50051"}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--no-enable-prefix-caching",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_consumer","mooncake_rpc_port": "50051"}'
        ]
    ]

    mooncake_args = [
        "--rpc_port", "50051", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8081", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9"
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    request_rate = [0.42, 1.17, 1.92, 2.67]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "batch_size": 128,
        "max_out_len": 150,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_mooncake",
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
                               kv_store_type="mooncake",
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
                           card_num=3,
                           aisbench_cases=aisbench_cases)



@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1p1d_ipc_mooncake_004(model: str, tp_size: int,
                                       dataset_name: str):
    '''
    roundrobin策略+mooncake
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--no-enable-prefix-caching",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
            MOONCAKE_CONSUMER_CONFIG_PATH +
            '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_producer","mooncake_rpc_port": "50051"}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--no-enable-prefix-caching",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_consumer","mooncake_rpc_port": "50051"}'
        ]
    ]
    proxy_args = ["--router", "RoundRobinRouter"]

    mooncake_args = [
        "--rpc_port", "50051", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8081", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9"
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    request_rate = [0.42, 1.17, 1.92, 2.67]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "batch_size": 128,
        "max_out_len": 150,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_roundroubin_mooncake",
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
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args,
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
                           card_num=3,
                           aisbench_cases=aisbench_cases)



@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1p1d_ipc_mooncake_005(model: str, tp_size: int,
                                       dataset_name: str):
    '''
    LeastInFlightRouter策略+mooncake
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
        MOONCAKE_PRODUCER_CONFIG_PATH +
        '", "ec_max_num_scheduled_tokens": "1000000000000000000"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--no-enable-prefix-caching",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"ec_mooncake_config_file_path":"' +
            MOONCAKE_CONSUMER_CONFIG_PATH +
            '"},"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_producer","mooncake_rpc_port": "50051"}'

        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--no-enable-prefix-caching",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector": "MooncakeConnectorStoreV1","kv_role": "kv_consumer","mooncake_rpc_port": "50051"}'
        ]
    ]
    proxy_args = ["--router", "LeastInFlightRouter"]

    mooncake_args = [
        "--rpc_port", "50051", "--enable_http_metadata_server=true",
        "--http_metadata_server_host=0.0.0.0",
        "--http_metadata_server_port=8081", "--rpc_thread_num", "8",
        "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
        "--eviction_high_watermark_ratio", "0.9"
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 50,
        "max_out_len": 256,
        "batch_size": 16,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0,
        "seed": 77,
    }]

    request_rate = [0.42, 1.17, 1.92, 2.67]
    case_dict = {
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen",
        "num_prompts": 200,
        "batch_size": 128,
        "max_out_len": 150,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_leastinfilight_mooncake",
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
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=2,
                               e_num=1,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args,
                               proxy_args=proxy_args,
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
                           card_num=3,
                           aisbench_cases=aisbench_cases)