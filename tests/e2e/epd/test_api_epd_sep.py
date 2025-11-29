import copy
import os

import pytest

from tests.e2e.conftest import RemoteEPDServer
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from tests.e2e.nightly.multi_node.config.utils import get_cluster_ips
from tests.e2e.nightly.multi_node.config.multi_node_epd_config import ClusterManager
from vllm.utils import get_open_port

model_path = load_config().get("model_path")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")
CONTAINER_NAME = load_config().get("container_name")

TENSOR_PARALLELS = [1]

SHARED_STORAGE_PATH = "/dev/shm/epd/storage"
ENABLE_PREFIX = [True, False]

DATASET_NAME = ["simulate_truth"]
ROUTER = ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1p1d_ipc_storage_mooncake_001(model: str, tp_size: int,
                                               dataset_name: str):
    '''
    数据集： simulate_truth
    前缀缓存：开启
    部署形态： 1E1P1D、单机
    存储类型：EC storage , KV mooncake
    '''
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
            '{"kv_connector_extra_config": {"local_hostname": "localhost", '
            '"metadata_server": "http://localhost:8081/metadata","protocol": "tcp", '
            '"device_name": "", "master_server_address": "localhost:50051", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            '"kv_role": "kv_producer", "mooncake_rpc_port": "50051"}'],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector_extra_config": {"local_hostname": "localhost", '
            '"metadata_server": "http://localhost:8081/metadata","protocol": "tcp", '
            '"device_name": "", "master_server_address": "localhost:50051", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            '"kv_role": "kv_consumer", "mooncake_rpc_port": "50051"}']
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("router", ROUTER)
async def test_1e1p1d_ipc_mooncake_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float, router: str):
    '''
    数据集： simulate_truth、image_4
    部署形态： 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    调度策略：RandomRouter， RoundRobinRouter，LeastInFlightRouter
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"local_hostname":"localhost",'
        '"metadata_server": "http://localhost:8085/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        '"master_server_address": "localhost:50055","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"local_hostname":"localhost",'
            '"metadata_server": "http://localhost:8085/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            '"master_server_address": "localhost:50055","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            '{"kv_connector_extra_config": {"local_hostname": "localhost", '
            '"metadata_server": "http://localhost:8081/metadata","protocol": "tcp", '
            '"device_name": "", "master_server_address": "localhost:50051", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            '"kv_role": "kv_producer", "mooncake_rpc_port": "50051"}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector_extra_config": {"local_hostname": "localhost", '
            '"metadata_server": "http://localhost:8081/metadata","protocol": "tcp", '
            '"device_name": "", "master_server_address": "localhost:50051", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            '"kv_role": "kv_consumer", "mooncake_rpc_port": "50051"}'
        ]
    ]

    mooncake_args = [
        [
            "--rpc_port", "50051", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            "--http_metadata_server_port=8081", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9005"
        ],
        [
            "--rpc_port", "50055", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            "--http_metadata_server_port=8085", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
        ]
    ]

    proxy_args = ["--router", router]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
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
                               proxy_args=proxy_args,
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
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("router", ROUTER)
async def test_1e1p1d_cross_tcp_mooncake_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float, router: str):
    '''
    数据集： simulate_truth
    部署形态： 1E1P1D、proxyE-P-D跨机
    存储类型：EC mooncake , KV mooncake
    调度策略：RandomRouter， RoundRobinRouter，LeastInFlightRouter
    通信方式： TCP
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    e_num = 1
    p_num = 1
    d_num = 1
    cluster = ClusterManager()
    for i in range(p_num):
        cluster.add_node_info("p", 1, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 2, CONTAINER_NAME)

    node_ips = get_cluster_ips()
    rpc_port = 50053
    http_metadata_server_port = 8083
    metrics_port = 9003

    mooncake_ip = node_ips[0]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", f"{mooncake_ip}", "--enable_http_metadata_server=true",
            f"--http_metadata_server_host={mooncake_ip}",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]


    proxy_args = ["--router", router]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num+p_num+d_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_mooncake_ipv4",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=p_num+d_num,
                               e_num=e_num,
                               node_info=cluster,
                               proxy_args=proxy_args,
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
                           card_num=e_num+p_num+d_num,
                           aisbench_cases=aisbench_cases)

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_2e3p3d_tcp_mooncake_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth、image_4
    部署形态： 2E3P3D、单机
    存储类型：EC mooncake , KV mooncake
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    e_num = 2
    p_num = 3
    d_num = 3
    e_server_args = list()
    pd_server_args = list()

    e_arg = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"local_hostname":"localhost",'
        '"metadata_server": "http://localhost:8085/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        '"master_server_address": "localhost:50055","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    p_arg = [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"local_hostname":"localhost",'
            '"metadata_server": "http://localhost:8085/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            '"master_server_address": "localhost:50055","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            '{"kv_connector_extra_config": {"local_hostname": "localhost", '
            '"metadata_server": "http://localhost:8081/metadata","protocol": "tcp", '
            '"device_name": "", "master_server_address": "localhost:50051", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            '"kv_role": "kv_producer", "mooncake_rpc_port": "50051"}'
        ]
    d_arg = [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector_extra_config": {"local_hostname": "localhost", '
            '"metadata_server": "http://localhost:8081/metadata","protocol": "tcp", '
            '"device_name": "", "master_server_address": "localhost:50051", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            '"kv_role": "kv_consumer", "mooncake_rpc_port": "50051"}'
        ]
    for _ in range(e_num):
        e_server_args.append(e_arg)
    for _ in range(p_num):
        pd_server_args.append(p_arg)
    for _ in range(d_num):
        pd_server_args.append(d_arg)


    mooncake_args = [
        [
            "--rpc_port", "50051", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            "--http_metadata_server_port=8081", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9005"
        ],
        [
            "--rpc_port", "50055", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            "--http_metadata_server_port=8085", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", "9004"
        ]
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num+p_num+d_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_2E3P3D_mooncake",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=d_num+p_num,
                               e_num=e_num,
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
                           card_num=e_num+p_num+d_num,
                           aisbench_cases=aisbench_cases)





REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["image_4", "simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e1p1d_ipc_mooncake_ipv6_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth、image_4
    部署形态： 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    通信方式：ipv6
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "1"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"

    rpc_port = 50053
    http_metadata_server_port = 8083
    metrics_port = 9003

    mooncake_ip = "::1"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "::", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=::",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_mooncake_ipv6",
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

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["image_4", "simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e1p1d_tcp_mooncake_ipv4_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth、image_4
    部署形态： 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    ipv4
    通信方式： TCP
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "0"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = "0.0.0.0"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "0.0.0.0", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_mooncake_tcp_ipv4",
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

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["simulate_truth_samereq", "simulate_truth_diffreq"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("enable_prefix", ENABLE_PREFIX)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1p1d_ipc_mooncake_ipv4_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float, enable_prefix: bool):
    '''
    数据集： simulate_truth_samereq、simulate_truth_diffreq
    部署形态： 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    ipv4
    前缀缓存： 开启、关闭
    通信方式： IPC
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "0"
    env_dict["TRANSFER_PROTOCOL"] = "ipc"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = "0.0.0.0"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
        ]
    ]
    if not enable_prefix:
        for args in pd_server_args:
            args.append("--no-enable-prefix-caching")
    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "0.0.0.0", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_mooncake_ipc_ipv4",
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

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["simulate_truth_samereq", "simulate_truth_diffreq"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("enable_prefix", ENABLE_PREFIX)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e1p1d_ipc_mooncake_ipv6_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float, enable_prefix: bool):
    '''
    数据集： simulate_truth_samereq、simulate_truth_diffreq
    部署形态： 1E1P1D、单机
    存储类型：EC mooncake , KV mooncake
    ipv6
    前缀缓存： 开启、关闭
    通信方式： IPC
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "1"
    env_dict["TRANSFER_PROTOCOL"] = "ipc"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = "::1"
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
        ]
    ]
    if not enable_prefix:
        for args in pd_server_args:
            args.append("--no-enable-prefix-caching")
    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "::", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=::",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 3,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_mooncake_ipc_ipv6",
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

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("router", ROUTER)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_2e3p3d_tcp_mooncake_ipv4_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float, router: str):
    '''
    数据集： simulate_truth
    部署形态： 2E3P3D、单机
    存储类型：EC mooncake , KV mooncake
    ipv4
    开启前缀缓存
    调度策略：RandomRouter， RoundRobinRouter，LeastInFlightRouter
    通信方式： TCP

    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "0"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = "0.0.0.0"
    e_num = 2
    p_num = 3
    d_num = 3
    e_server_args = list()
    pd_server_args = list()

    e_arg = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    p_arg = [
        "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
    ]
    d_arg = [
        "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
    ]
    for _ in range(e_num):
        e_server_args.append(e_arg)
    for _ in range(p_num):
        pd_server_args.append(p_arg)
    for _ in range(d_num):
        pd_server_args.append(d_arg)


    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "0.0.0.0", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=0.0.0.0",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]
    proxy_args = ["--router", router]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 8,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_2E3P3D_mooncake_tcp_ipv4",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=6,
                               e_num=2,
                               env_dict=env_dict,
                               proxy_args=proxy_args,
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
                           card_num=8,
                           aisbench_cases=aisbench_cases)

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_2e3p3d_tcp_mooncake_ipv6_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： 2E3P3D、单机
    存储类型：EC mooncake , KV mooncake
    ipv6
    开启前缀缓存
    通信方式： TCP
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "1"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    e_num = 2
    p_num = 3
    d_num = 3
    e_server_args = list()
    pd_server_args = list()
    mooncake_ip = "::1"

    e_arg = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    p_arg = [
        "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{mooncake_ip}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
    ]
    d_arg = [
        "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{mooncake_ip}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
    ]
    for _ in range(e_num):
        e_server_args.append(e_arg)
    for _ in range(p_num):
        pd_server_args.append(p_arg)
    for _ in range(d_num):
        pd_server_args.append(d_arg)


    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "::", "--enable_http_metadata_server=true",
            "--http_metadata_server_host=::",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]

    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * 8,
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_2E3P3D_mooncake_tcp_ipv6",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=6,
                               e_num=2,
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
                           card_num=8,
                           aisbench_cases=aisbench_cases)

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["image_4", "simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_proxy_1e1p1d_cross_tcp_mooncake_ipv4_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth、image_4
    部署形态： 1E1P1D、proxy-EPD跨机
    存储类型：EC mooncake , KV mooncake
    通信方式： TCP
    开启前缀缓存
    ipv4
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    e_num = 1
    p_num = 1
    d_num = 1
    cluster = ClusterManager()
    for i in range(p_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(p_num):
        cluster.add_node_info("p", 1, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 1, CONTAINER_NAME)

    node_ips = get_cluster_ips()
    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = node_ips[0]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "{mooncake_ip}:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", f"{mooncake_ip}", "--enable_http_metadata_server=true",
            f"--http_metadata_server_host={mooncake_ip}",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]



    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num+p_num+d_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_proxy_1E1P1D_cross_tcp_mooncake_ipv4",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=p_num+d_num,
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
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=e_num+p_num+d_num,
                           aisbench_cases=aisbench_cases)

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["image_4", "simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_proxy_1e1p1d_cross_tcp_mooncake_ipv6_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth、image_4
    部署形态： 1E1P1D、proxy-EPD跨机
    存储类型：EC mooncake , KV mooncake
    通信方式： TCP
    开启前缀缓存
    ipv6
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "1"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    e_num = 1
    p_num = 1
    d_num = 1
    cluster = ClusterManager()
    for i in range(p_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(p_num):
        cluster.add_node_info("p", 1, CONTAINER_NAME)
    for i in range(d_num):
        cluster.add_node_info("d", 1, CONTAINER_NAME)

    node_ips = get_cluster_ips(family=socket.AF_INET6)
    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = node_ips[0]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]

    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "[{mooncake_ip}]:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://[{mooncake_ip}]:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", "::", "--enable_http_metadata_server=true",
            f"--http_metadata_server_host=::",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]



    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num+p_num+d_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_proxy_1E1P1D_cross_tcp_mooncake_ipv6",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=p_num+d_num,
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
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=e_num+p_num+d_num,
                           aisbench_cases=aisbench_cases)

REQUEST_RATE = [0.28, 0.78, 1.28, 1.78]
DATASET_NAME = ["image_4", "simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_proxy_1e_2pd_cross_tcp_mooncake_ipv4_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： 1E2PD、proxy-E-PD跨机
    存储类型：EC mooncake , KV mooncake
    通信方式： TCP
    开启前缀缓存
    ipv4
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "0"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    e_num = 1
    pd_num = 2
    cluster = ClusterManager()
    for i in range(p_num):
        cluster.add_node_info("e", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("pd", 2, CONTAINER_NAME)


    node_ips = get_cluster_ips()
    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = node_ips[0]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[2]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}'
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", f"{mooncake_ip}", "--enable_http_metadata_server=true",
            f"--http_metadata_server_host={mooncake_ip}",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]



    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num+p_num+d_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_proxy_1E1P1D_cross_tcp_mooncake_ipv4",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
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
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=e_num+pd_num,
                           aisbench_cases=aisbench_cases)

REQUEST_RATE = [1.78]
DATASET_NAME = ["simulate_truth"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_proxy1e_1p_1d_cross_tcp_mooncake_ipv4_001(model: str, tp_size: int,
                                       dataset_name: str, request_rate: float):
    '''
    数据集： simulate_truth
    部署形态： 1E1P1D、proxyE-P_D跨机
    存储类型：EC mooncake , KV mooncake
    通信方式： TCP
    开启前缀缓存
    ipv4
    '''
    env_dict = {}
    env_dict["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "6000"
    env_dict["LM_SERVICE_REQUEST_TIMEOUT_SECONDS"] = "300"
    env_dict["MC_MS_AUTO_DISC"] = "0"
    env_dict["MC_USE_IPV6"] = "0"
    env_dict["TRANSFER_PROTOCOL"] = "tcp"
    env_dict["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    e_num = 1
    p_num = 1
    d_num = 1
    cluster = ClusterManager()
    for i in range(p_num):
        cluster.add_node_info("e", 0, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("p", 1, CONTAINER_NAME)
    for i in range(pd_num):
        cluster.add_node_info("d", 2, CONTAINER_NAME)


    node_ips = get_cluster_ips()
    rpc_port = get_open_port()
    http_metadata_server_port = get_open_port()
    metrics_port = get_open_port()

    mooncake_ip = node_ips[0]
    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.0",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[0]}",'
        f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 32212254720, '
        '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
        f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
        '"fast_transfer_buffer_size": 1, "ec_max_num_scheduled_tokens": "1000000000000000000"},'
        '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            f'{{"ec_connector_extra_config":{{"local_hostname":"{node_ips[1]}",'
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","global_segment_size": 0, '
            '"local_buffer_size": 1073741824, "protocol": "tcp", "device_name": "",'
            f'"master_server_address": "{mooncake_ip}:{rpc_port}","replica_num": 1, "fast_transfer":true, '
            '"fast_transfer_buffer_size": 1},'
            '"ec_connector":"ECMooncakeStorageConnector","ec_role": "ec_consumer"}',
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[1]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_producer", "mooncake_rpc_port": "{rpc_port}"}}'
        ],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            f'{{"kv_connector_extra_config": {{"local_hostname": "{node_ips[2]}", '
            f'"metadata_server": "http://{mooncake_ip}:{http_metadata_server_port}/metadata","protocol": "tcp", '
            f'"device_name": "", "master_server_address": "[{mooncake_ip}]:{rpc_port}", '
            '"global_segment_size": 30000000000},"kv_connector": "MooncakeConnectorStoreV1", '
            f'"kv_role": "kv_consumer", "mooncake_rpc_port": "{rpc_port}"}}'
        ]
    ]

    mooncake_args = [
            "--rpc_port", str(rpc_port), "--rpc_address", f"{mooncake_ip}", "--enable_http_metadata_server=true",
            f"--http_metadata_server_host={mooncake_ip}",
            f"--http_metadata_server_port={http_metadata_server_port}", "--rpc_thread_num", "8",
            "--default_kv_lease_ttl", "10000", "eviction_ratio", "0.05",
            "--eviction_high_watermark_ratio", "0.9", "--metrics_port", str(metrics_port)
    ]



    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
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
        "case_type": "pressure",
        "pressure_time":86400,
        "dataset_path": os.path.join(DATASET_PATH, dataset_name),
        "request_conf": "vllm_api_stream_chat",
        "dataset_conf": "textvqa/textvqa_gen_base64",
        "num_prompts": 200,
        "batch_size": 128,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.7,
        "repetition_penalty": 1.2,
        "request_rate": request_rate * (e_num+p_num+d_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_proxy1E_1P_1D_cross_tcp_mooncake_ipv4",
        "threshold": 0.97
    }]
    api_port = 10001
    async with RemoteEPDServer(run_mode="worker",
                               store_type="mooncake",
                               kv_store_type="mooncake",
                               proxy_type="api_server",
                               api_server_port=api_port,
                               pd_num=p_num+d_num,
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
        # aisbench test
        run_aisbench_cases(model=model,
                           port=api_port,
                           card_num=e_num+p_num+d_num,
                           aisbench_cases=aisbench_cases)