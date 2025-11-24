import os

import pytest
import pytest_asyncio
import copy

from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from tests.e2e.nightly.multi_node.config.multi_node_epd_config import ClusterManager

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")
SHARED_STORAGE_PATH = "/dev/shm/epd/storage"

TENSOR_PARALLELS = [1]
DATASET_NAME = ["simulate_truth"]

MOONCAKE_PRODUCER_CONFIG_PATH = load_config().get("mooncake_config_path") + "producer.json"
MOONCAKE_CONSUMER_CONFIG_PATH = load_config().get("mooncake_config_path") + "consumer.json"
PREFIX_CACHE = [True, False]

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("prefix_cache", PREFIX_CACHE)
async def test_1e2pd_cross_p_e_pd_mooncake_tcp_001(model: str, tp_size: int, prefix_cache: bool):
    '''
    proxy-e-pd 跨机部署, 1E2PD
    前缀缓存： 开启/关闭
    数据集：使用同请求相同图片，跨请求相同图片，不同图片数据集, textvqa-subset
    测试类型：perf, acc
    ec transfer: mooncake
    '''

    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_num = 1
    pd_num = 2
    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, "epd_vllm_ascend_mooncake")
    for i in range(pd_num):
        cluster.add_node_info("pd", 2, "epd_vllm_ascend_mooncake")
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
    if not prefix_cache:
        pd_server_args.append("--no-enable-prefix-caching")

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
            "textvqa/textvqa_gen_base64",
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
    dataset_names = ["simulate_truth_samereq", "simulate_truth_diffreq", "simulate_truth"]
    case_dict = {
        "case_type": "performance",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "threshold": 0.97
    }
    aisbench_cases = []
    for dataset_name in dataset_names:
        for i in range(len(request_rate)):
            case_dict["request_rate"] = request_rate[i]
            case_dict["dataset_path"] = os.path.join(DATASET_PATH, dataset_name)
            case_dict["result_file_name"] = f"{dataset_name}_1E2PD_cross_p_e_pd_mooncake_TCP"
            new_case_dict = copy.deepcopy(case_dict)
            aisbench_cases.append(new_case_dict)

    acc_cases = [{
        "case_type": "accuracy",
        "dataset_path": os.path.join(DATASET_PATH, "textvqa_subset"),
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "max_out_len": 2048,
        "batch_size": 128,
        "temperature": 0,
        "top_k": -1,
        "top_p": 1,
        "repetition_penalty": 1,
        "request_rate": 0,
        "baseline": 81,
        "seed": 77,
        "threshold": 1
    }]

    api_port = 10002
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
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
        # test perf
        run_aisbench_cases(model=model,
                               port=api_port,
                               aisbench_cases=aisbench_cases)

        # test acc
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=acc_cases,save=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("prefix_cache", PREFIX_CACHE)
async def test_1e2pd_cross_p_epd_mooncake_tcp_001(model: str, tp_size: int, prefix_cache: bool):
    '''
    proxy-epd 跨机部署, 1E2PD
    前缀缓存： 开启/关闭
    数据集：使用同请求相同图片，跨请求相同图片，不同图片数据集
    测试类型：perf
    ec transfer: mooncake
    '''

    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_num = 1
    pd_num = 2
    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, "epd_vllm_ascend_mooncake")
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, "epd_vllm_ascend_mooncake")
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
    if not prefix_cache:
        pd_server_args.append("--no-enable-prefix-caching")

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
            "textvqa/textvqa_gen_base64",
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
    dataset_names = ["simulate_truth_samereq", "simulate_truth_diffreq", "simulate_truth"]
    case_dict = {
        "case_type": "performance",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "threshold": 0.97
    }
    aisbench_cases = []
    for dataset_name in dataset_names:
        for i in range(len(request_rate)):
            case_dict["request_rate"] = request_rate[i]
            case_dict["dataset_path"] = os.path.join(DATASET_PATH, dataset_name)
            case_dict["result_file_name"] = f"{dataset_name}_1E2PD_cross_p_epd_mooncake_TCP"
            new_case_dict = copy.deepcopy(case_dict)
            aisbench_cases.append(new_case_dict)

    api_port = 10002
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
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
        # test perf
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)



@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_1e1pd_sc_cross_p_epd_mooncake_tcp_001(model: str, tp_size: int):
    '''
    proxy-epd 跨机部署, 1e1pd共卡
    前缀缓存： 开启
    数据集：不同图片数据集
    测试类型：perf
    ec transfer: mooncake
    '''

    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_num = 1
    pd_num = 1
    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, "epd_vllm_ascend_mooncake")
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, "epd_vllm_ascend_mooncake")
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
            "textvqa/textvqa_gen_base64",
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
    dataset_names = ["simulate_truth"]
    case_dict = {
        "case_type": "performance",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "threshold": 0.97
    }
    aisbench_cases = []
    for dataset_name in dataset_names:
        for i in range(len(request_rate)):
            case_dict["request_rate"] = request_rate[i]
            case_dict["dataset_path"] = os.path.join(DATASET_PATH, dataset_name)
            case_dict["result_file_name"] = f"{dataset_name}_1E1PD_sc_cross_p_epd_mooncake_TCP"
            new_case_dict = copy.deepcopy(case_dict)
            aisbench_cases.append(new_case_dict)

    api_port = 10002
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
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
        # test perf
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)



@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_1e1pd_sc_cross_p_epd_storage_tcp_001(model: str, tp_size: int):
    '''
    proxy-epd 跨机部署, 1e1pd共卡
    前缀缓存： 开启
    数据集：不同图片数据集
    测试类型：perf
    ec transfer: storage
    '''

    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_num = 1
    pd_num = 1
    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, "epd_vllm_ascend_mooncake")
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, "epd_vllm_ascend_mooncake")
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
            "textvqa/textvqa_gen_base64",
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
    dataset_names = ["simulate_truth"]
    case_dict = {
        "case_type": "performance",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "threshold": 0.97
    }
    aisbench_cases = []
    for dataset_name in dataset_names:
        for i in range(len(request_rate)):
            case_dict["request_rate"] = request_rate[i]
            case_dict["dataset_path"] = os.path.join(DATASET_PATH, dataset_name)
            case_dict["result_file_name"] = f"{dataset_name}_1E1PD_sc_cross_p_epd_storage_TCP"
            new_case_dict = copy.deepcopy(case_dict)
            aisbench_cases.append(new_case_dict)

    api_port = 10002
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
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
        # test perf
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)



@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("prefix_cache", PREFIX_CACHE)
async def test_1e2pd_cross_p_epd_storage_tcp_001(model: str, tp_size: int, prefix_cache: bool):
    '''
    proxy-epd 跨机部署, 1E2PD
    前缀缓存： 开启/关闭
    数据集：使用同请求相同图片，跨请求相同图片，不同图片数据集
    测试类型：perf
    ec transfer: storage
    '''

    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_num = 1
    pd_num = 2
    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, "epd_vllm_ascend_mooncake")
    for i in range(pd_num):
        cluster.add_node_info("pd", 1, "epd_vllm_ascend_mooncake")
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
    if not prefix_cache:
        pd_server_args.append("--no-enable-prefix-caching")

    warmup_cases = [{
        "case_type":
            "performance",
        "dataset_path":
            os.path.join(DATASET_PATH, "simulate_truth"),
        "request_conf":
            "vllm_api_stream_chat",
        "dataset_conf":
            "textvqa/textvqa_gen_base64",
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
    dataset_names = ["simulate_truth_samereq", "simulate_truth_diffreq", "simulate_truth"]
    case_dict = {
        "case_type": "performance",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "threshold": 0.97
    }
    aisbench_cases = []
    for dataset_name in dataset_names:
        for i in range(len(request_rate)):
            case_dict["request_rate"] = request_rate[i]
            case_dict["dataset_path"] = os.path.join(DATASET_PATH, dataset_name)
            case_dict["result_file_name"] = f"{dataset_name}_1E2PD_cross_p_epd_storage_TCP"
            new_case_dict = copy.deepcopy(case_dict)
            aisbench_cases.append(new_case_dict)

    api_port = 10002
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               is_epd_same_card=True,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:

        # warm up
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=warmup_cases,
                           verify=False,
                           save=False)
        # test perf
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_1e2pd_cross_p_e_pd_mooncake_tcp_002(model: str, tp_size: int):
    '''
    proxy-epd 跨机部署, 1E2PD
    前缀缓存：开启
    数据集：模拟现网数据集
    测试类型：stability
    ec transfer: mooncake
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_num = 1
    pd_num = 2
    cluster = ClusterManager()
    for i in range(e_num):
        cluster.add_node_info("e", 1, "epd_vllm_ascend_mooncake")
    for i in range(pd_num):
        cluster.add_node_info("pd", 2, "epd_vllm_ascend_mooncake")
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
            "textvqa/textvqa_gen_base64",
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
    dataset_name = "simulate_truth"
    aisbench_cases = [{
        "case_type": "pressure",
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "batch_size": 128,
        "temperature": 0.5,
        "pressure_time": 86400,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": 0.28,
        "baseline": 1,
        "seed": 77,
        "threshold": 0.97,
        "result_file_name": f"{dataset_name}_1E2PD_cross_p_e_pd_mooncake_TCP"
    }]

    api_port = 10002
    async with RemoteEPDServer(run_mode="worker",
                               store_type="storage",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               node_info=cluster,
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
        # test stability
        run_aisbench_cases(model=model,
                           port=api_port,
                           aisbench_cases=aisbench_cases)

