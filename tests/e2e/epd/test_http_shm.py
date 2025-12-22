import os

import pytest


from tests.e2e.conftest import RemoteEPDServer,RemoteOpenAIServer,DisaggEpdProxy
from tests.e2e.epd.conftest import load_config
from tools.aisbench import run_aisbench_cases
from tests.e2e.nightly.multi_node.config.multi_node_epd_config import ClusterManager, EnvManager
from vllm.utils import get_open_port

model_path = load_config().get("model_path")
CONTAINER_NAME = load_config().get("container_name")
MODELS = [os.path.join(model_path, "Qwen2.5-VL-7B-Instruct")]
DATASET_PATH = load_config().get("dataset_path")
SHARED_STORAGE_PATH = "/dev/shm/epd/storage"


TENSOR_PARALLELS = [1]
PREFIX_CACHE = [True, False]
ROUTER = ["RandomRouter", "RoundRobinRouter", "LeastInFlightRouter"]

REQUEST_RATE = [0.28, 0.56, 0.84]
DATASET_NAME = ["simulate_truth"]


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e1pd_shm_http_001(model: str, tp_size: int, dataset_name: str,
                                 request_rate: float):
    '''
    1E1PD 单机部署
    前缀缓存： 开启
    数据集：模拟ZJ
    ec transfer: shm
    通信方式：http
    '''

    env_dict = EnvManager()
    e_num = 1
    pd_num = 1
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.01",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    warmup_cases = [{
        "case_type": "performance",
        "dataset_path": os.path.join(DATASET_PATH, "image_4"),
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
        "request_rate": request_rate * (e_num + pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1PD_shm_http",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="serve",
                               store_type="storage",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        async with DisaggEpdProxy(port=api_port, server=server) as proxy:
            # warm up
            run_aisbench_cases(model=model,
                               port=api_port,
                               aisbench_cases=warmup_cases,
                               verify=False,
                               save=False)
            # aisbench test
            run_aisbench_cases(model=model,
                               port=api_port,
                               card_num=pd_num+e_num,
                               aisbench_cases=aisbench_cases)


DATASET_NAME = ["simulate_truth", "image_4"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e2pd_shm_http_001(model: str, tp_size: int,
                                    dataset_name: str, request_rate: float):
    '''
    1E2PD 单机部署
    前缀缓存： 开启
    数据集：模拟zj，image_4
    ec transfer: shm
    通信方式：http
    '''

    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.01",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
       "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
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
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E2PD_shm_http_001",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="serve",
                               store_type="storage",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        async with DisaggEpdProxy(port=api_port, server=server) as proxy:
            # warm up
            run_aisbench_cases(model=model,
                               port=api_port,
                               aisbench_cases=warmup_cases,
                               verify=False,
                               save=False)
            # aisbench test
            run_aisbench_cases(model=model,
                               port=api_port,
                               card_num=pd_num + e_num,
                               aisbench_cases=aisbench_cases)


@pytest.mark.asyncio
@pytest.mark.acc
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
async def test_1e2pd_shm_http_003(model: str, tp_size: int,
                                 dataset_name: str):
    '''
    1E2PD 单机部署
    前缀缓存： 开启
    数据集：textvqa-subset
    ec transfer: shm
    通信方式：http
    '''

    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.01",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "20000", "--max-num-batched-tokens", "20000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "20000",
        "--max-num-batched-tokens", "20000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
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

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="serve",
                               store_type="storage",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        async with DisaggEpdProxy(port=api_port, server=server) as proxy:
            # aisbench test
            run_aisbench_cases(model=model,
                               port=api_port,
                               card_num=pd_num + e_num,
                               aisbench_cases=acc_cases, save=False)



DATASET_NAME = ["simulate_truth", "image_4"]
@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dataset_name", DATASET_NAME)
@pytest.mark.parametrize("request_rate", REQUEST_RATE)
async def test_1e1p1d_shm_http_001(model: str, tp_size: int,
                                    dataset_name: str, request_rate: float):
    '''
    1E1P1D 单机部署
    前缀缓存： 开启
    数据集：模拟zj，image_4
    ec transfer: shm
    通信方式：http
    '''

    env_dict = EnvManager()
    e_num = 1
    pd_num = 2
    for i in range(e_num):
        env_dict.add_env("e", "ASCEND_RT_VISIBLE_DEVICES", str(i))
    for i in range(pd_num):
        env_dict.add_env("pd",
                         "ASCEND_RT_VISIBLE_DEVICES",
                         str(i + e_num),
                         index=i)

    e_server_args = [
        "--model", model, "--gpu-memory-utilization", "0.01",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [[
        "--model", model, "--gpu-memory-utilization", "0.95",
       "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}',
        "--kv-transfer-config",
        '{"kv_connector_extra_config": {"use_ascend_direct": true, '
        '"prefill": {"dp_size": 1, "tp_size": 1}, "decode": {"dp_size": 1, "tp_size": 1}},"kv_connector": "MooncakeConnectorV1", '
        '"kv_role": "kv_producer", "kv_buffer_device": "npu", "kv_parallel_size": 1, "kv_port": "20001",'
        '"engine_id":"0", "kv_rank": 0, "kv_connector_module_paht": "vllm_ascend.distriuted.mooncake_connector"}'],
        [
            "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--kv-transfer-config",
            '{"kv_connector_extra_config": {"use_ascend_direct": true, '
            '"prefill": {"dp_size": 1, "tp_size": 1}, "decode": {"dp_size": 1, "tp_size": 1}},"kv_connector": "MooncakeConnectorV1", '
            '"kv_role": "kv_consumer", "kv_buffer_device": "npu", "kv_parallel_size": 1, "kv_port": "20001",'
            '"engine_id":"0", "kv_rank": 0, "kv_connector_module_paht": "vllm_ascend.distriuted.mooncake_connector"}']

    ]
    warmup_cases = [{
        "case_type":
        "performance",
        "dataset_path":
        os.path.join(DATASET_PATH, "simulate_truth_samereq"),
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
        "request_rate": request_rate*(e_num+pd_num),
        "baseline": 1,
        "seed": 77,
        "result_file_name": f"{dataset_name}_1E1P1D_shm_http_001",
        "threshold": 0.97
    }]

    api_port = get_open_port()
    async with RemoteEPDServer(run_mode="serve",
                               store_type="storage",
                               api_server_port=api_port,
                               pd_num=pd_num,
                               e_num=e_num,
                               env_dict=env_dict,
                               e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        async with DisaggEpdProxy(port=api_port, server=server) as proxy:
            # warm up
            run_aisbench_cases(model=model,
                               port=api_port,
                               aisbench_cases=warmup_cases,
                               verify=False,
                               save=False)
            # aisbench test
            run_aisbench_cases(model=model,
                               port=api_port,
                               card_num=pd_num + e_num,
                               aisbench_cases=aisbench_cases)