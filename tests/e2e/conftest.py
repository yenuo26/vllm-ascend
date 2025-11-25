import asyncio
import contextlib
import copy
import gc
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypeVar, Union, Literal

import httpx
import numpy as np
import openai
import pandas as pd
import psutil
import requests
import torch
from PIL import Image
from torch import nn
from transformers import (BatchEncoding, BatchFeature)
from vllm.disaggregated.proxy import Proxy

from tests.e2e.nightly.multi_node.config.multi_node_config import NodeInfo
# TODO: remove this part after the patch merged into vllm, if
# we not explicitly patch here, some of them might be effectiveless
# in pytest scenario
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.9.1"):
    from vllm.utils import get_open_port
else:
    from vllm.utils.network_utils import get_open_port


from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment, destroy_model_parallel)

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature, dict)
_M = TypeVar("_M")

_PromptMultiModalInput = Union[List[_M], List[List[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[Tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]

_TEST_DIR = os.path.dirname(__file__)


def get_package_location(package_name):
    try:
        distribution = importlib.metadata.distribution(package_name)
        return str(distribution.locate_file(''))
    except importlib.metadata.PackageNotFoundError:
        return None


VLLM_PATH = get_package_location("vllm")


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



def write_to_execl(data, path):
    if path is not None:
        if not os.path.exists(path):
            df = pd.DataFrame(data, index=[0])
            df.to_csv(path, index=False)
        else:
            existing_df = pd.read_csv(path)
            new_df = pd.DataFrame(data, index=[0])
            combined_df = pd.concat([existing_df, new_df],
                                    ignore_index=True)
            combined_df.to_csv(path, index=False)


class RemoteEPDServer:

    def get_proxy(self) -> Proxy:
        return self.p

    def _read_output(self, pipe, prefix):
        """在单独线程中读取输出"""
        try:
            with pipe:
                for line in iter(pipe.readline, ''):
                    if line:
                        print(f"{prefix}: {line}", end='')
                        if self.env_dict.get("TIMECOUNT_ENABLED", 0)=="1":
                            self._extract_ttft_data(line, prefix)

        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc()

    def _extract_ttft_data(self, text, prefix):
        if "PROXY" in prefix.upper():
            patterns = {
                'transfer_to_encode': r'Avg proxy to encoder requests: ([\d.]+) ms',
                'transfer_to_pd': r'Avg proxy to pd requests: ([\d.]+) ms',
            }
            for i, flag in enumerate(self.e_addr_list):
                patterns[f'E{i}_queue'] = fr'{flag}.*Avg queue time requests: ([\d.]+) ms'
                patterns[f'E{i}_prefill'] = fr'{flag}.*Avg prefill time requests: ([\d.]+) ms'
            for i, flag in enumerate(self.pd_addr_list):
                patterns[f'PD{i}_ttft'] = fr'{flag}.*Avg proxy ttft: ([\d.]+) ms'
                patterns[f'PD{i}_queue'] = fr'{flag}.*Avg queue time requests: ([\d.]+) ms'
                patterns[f'PD{i}_prefill'] = fr'{flag}.*Avg prefill time requests: ([\d.]+) ms'
            for key, pattern in patterns.items():
                match = re.search(pattern, text)
                if match:
                    self.metrics[key] = float(match.group(1))

    def save_ttft_data(self, file_name, index):
        data = {
            "index": index
        }
        data.update(self.metrics)
        write_to_execl(data, f"./{file_name}.csv")
        print(f"TTFT Analysis csv file is locate in ./{file_name}.csv")


    def _run_server(self, server_cmd: list[str], env_dict: Optional[dict[str,
                                                                         str]],
                    log_prefix: str) -> None:
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # 文本模式
            bufsize=1)
        # 创建线程读取输出
        stdout_thread = threading.Thread(target=self._read_output,
                                         args=(proc.stdout, log_prefix),
                                         daemon=True)
        stderr_thread = threading.Thread(target=self._read_output,
                                         args=(proc.stderr, log_prefix),
                                         daemon=True)

        stdout_thread.start()
        stderr_thread.start()
        self._proc_list.append(proc)

    def _run_server_new_session(self, server_cmd: list[str],
                                env_dict: Optional[dict[str, str]],
                                log_prefix: str) -> None:
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
                                         args=(proc.stdout, log_prefix),
                                         daemon=True)
        stderr_thread = threading.Thread(target=self._read_output,
                                         args=(proc.stderr, log_prefix),
                                         daemon=True)

        stdout_thread.start()
        stderr_thread.start()
        self._proc_list.append(proc)

    def _start_api_server(self) -> None:
        api_server_args = [
            "--host", "127.0.0.1", "--port",
            str(self.api_server_port), "--proxy-config", json.dumps(self.proxy_config)
        ]
        if self.is_image_load:
            api_server_args.append("--is-load-image")

        print(f"proxy params is: {api_server_args}")
        api_server_path = Path(
            __file__).parent.parent.parent / "tools" / "api_server.py"
        api_server_args = ["python", api_server_path, *api_server_args]
        self._run_server_new_session(api_server_args, self.env_dict,
                                     "[PROXY] ")

    def _start_mooncake(self) -> None:
        self._init_mooncake_config()
        self.mooncake_args = ["mooncake_master", *self.mooncake_args]
        self._run_server_new_session(self.mooncake_args, None, "[MOONCAKE] ")

    def _init_mooncake_config(self) -> None:
        mooncake_json = {
            "local_hostname": "0.0.0.0",
            "global_segment_size": 32212254720,
            "local_buffer_size": 1073741824,
            "protocol": "tcp",
            "device_name": "",
            "replica_num": 1,
            "fast_transfer": True,
            "fast_transfer_buffer_size": 1
        }

        for i, arg in enumerate(self.mooncake_args):
            if "--http_metadata_server_port" in arg:
                metadata_server_port = self.mooncake_args[i].split("=")[-1]
                mooncake_json[
                    "metadata_server"] = f"http://0.0.0.0:{metadata_server_port}/metadata"
            if "--rpc_port" in arg:
                rpc_port = self.mooncake_args[i + 1]
                mooncake_json[
                    "master_server_address"] = f"0.0.0.0:{rpc_port}"

        config_path = ""
        if self.store_type == "mooncake":
            for i, arg in enumerate(self.e_serve_args_list + self.pd_serve_args_list):
                index = arg.index("--ec-transfer-config")
                config_path = json.loads(arg[index + 1]).get(
                    "ec_connector_extra_config").get("ec_mooncake_config_file_path")

        if self.kv_store_type == "mooncake":
            config_path = self.env_dict["MOONCAKE_CONFIG_PATH"]

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(mooncake_json, f, ensure_ascii=False, indent=4)
        print(f"The mooncake producer config is\n {mooncake_json}")

    def _get_addr_config(self, args, i, role):
        if self.env_dict.get("TRANSFER_PROTOCOL") is not None:
            self.protocol = self.env_dict["TRANSFER_PROTOCOL"].lower()
        elif "--transfer-protocol" in args:
            protocol_index = args.index("--transfer-protocol") + 1
            if protocol_index < len(args):
                self.protocol = args[protocol_index].lower()
        else:
            self.protocol = "ipc"

        if self.protocol == "tcp":
            if role == "E":
                return {
                    "proxy_addr": "127.0.0.1:37000",
                    "worker_addr": f"127.0.0.1:3800{i}"
                }
            else:
                return {
                    "proxy_addr": "127.0.0.1:37000",
                    "worker_addr": f"127.0.0.1:3900{i}"
                }
        else:
            if role == "E":
                return {
                    "proxy_addr": f"{self._default_addr_prefix}proxy",
                    "worker_addr": f"{self._default_addr_prefix}encoder_{i}"
                }
            else:
                return {
                    "proxy_addr": f"{self._default_addr_prefix}proxy",
                    "worker_addr": f"{self._default_addr_prefix}pd_{i}"
                }

    def _start_vllm_worker(self):
        if self.env_dict is None:
            self.env_dict = dict()
        self.env_dict['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"
        self.env_dict['VLLM_USE_V1'] = "1"

        serve_arg_cmd = [
                "taskset", "-c", "0-96", "python", "-m",
                "vllm.entrypoints.disaggregated.worker"
            ]

        for i, e_serve_arg in enumerate(self.e_serve_args_list):
            self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            e_serve_arg = [*serve_arg_cmd, *e_serve_arg]

            config = self._get_addr_config(e_serve_arg, i, "E")
            if "--proxy-addr" not in e_serve_arg:
                e_serve_arg.extend(["--proxy-addr", config["proxy_addr"]])
            if "--worker-addr" not in e_serve_arg:
                e_serve_arg.extend(["--worker-addr", config["worker_addr"]])

            index_e = e_serve_arg.index("--proxy-addr")
            if self.proxy_addr is not None and e_serve_arg[
                    index_e + 1] != self.proxy_addr:
                raise ValueError("proxy addr must be same between workers")
            self.proxy_addr = e_serve_arg[index_e + 1]

            index_e = e_serve_arg.index("--worker-addr")
            self.e_addr_list.append(e_serve_arg[index_e + 1])

            if "--model" not in e_serve_arg:
                raise ValueError("must carry --model")
            else:
                index_e = e_serve_arg.index("--model")
                if self.model is not None and e_serve_arg[index_e +
                                                          1] != self.model:
                    raise ValueError("model must be same between workers")
                self.model = e_serve_arg[index_e + 1]

            self._run_server(e_serve_arg, self.env_dict, f"[ENCODE_{i}] ")

        for i, pd_serve_arg in enumerate(self.pd_serve_args_list):
            if self.is_epd_same_card:
                self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            else:
                self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(
                    i + self.e_num)
            pd_serve_arg = [*serve_arg_cmd, *pd_serve_arg]
            if "--model" not in pd_serve_arg:
                raise ValueError("must carry --model")
            else:
                index_pd = pd_serve_arg.index("--model")
                if self.model is not None and pd_serve_arg[
                    index_pd + 1] != self.model:
                    raise ValueError("model must be same between workers")

            config = self._get_addr_config(pd_serve_arg, i, "PD")
            if "--proxy-addr" not in pd_serve_arg:
                pd_serve_arg.extend(["--proxy-addr", config["proxy_addr"]])
            if "--worker-addr" not in pd_serve_arg:
                pd_serve_arg.extend(["--worker-addr", config["worker_addr"]])

            index_pd = pd_serve_arg.index("--proxy-addr")
            if self.proxy_addr is not None and pd_serve_arg[
                    index_pd + 1] != self.proxy_addr:
                raise ValueError("proxy addr must be same between workers")

            worker_index = pd_serve_arg.index("--worker-addr")
            log_prefix = ""
            if "--kv-transfer-config" in pd_serve_arg:
                kv_index = pd_serve_arg.index("--kv-transfer-config")
                if "kv_consumer" in pd_serve_arg[kv_index + 1]:
                    self.d_addr_list.append(pd_serve_arg[worker_index + 1])
                    log_prefix = f"[D_{i}] "
                elif "kv_producer" in pd_serve_arg[kv_index + 1]:
                    self.p_addr_list.append(pd_serve_arg[worker_index + 1])
                    log_prefix = f"[P_{i}] "
            else:
                self.pd_addr_list.append(pd_serve_arg[worker_index + 1])
                log_prefix = f"[PD_{i}] "

            self._run_server(pd_serve_arg, self.env_dict, log_prefix)

    def _start_zmq_proxy(self):
        for key, value in self.env_dict.items():
            os.environ[key] = value
        self.proxy_config = {
            'proxy_addr': self.proxy_addr,
            'encode_addr_list': self.e_addr_list,
            'model_name': self.model
        }
        if self.pd_addr_list:
            self.proxy_config['pd_addr_list'] = self.pd_addr_list
        else:
            self.proxy_config.update({
                'p_addr_list': self.p_addr_list,
                'd_addr_list': self.d_addr_list
            })
        if self.proxy_args is not None and "--transfer_protocol" in self.proxy_args:
            self.proxy_config['transfer_protocol'] = self.proxy_args[self.proxy_args.index("--transfer_protocol")+1]
        if self.proxy_args is not None and "--enable-health-monitor" in self.proxy_args:
            self.proxy_config['enable_health_monitor'] = self.proxy_args[self.proxy_args.index("--enable-health-monitor")+1]

        p = Proxy(**self.proxy_config)
        return p

    def _start_disagg_proxy(self):
        proxy_args = [
            "--host", "127.0.0.1", "--port",
            str(self.api_server_port), "--encode-servers-urls",
            ",".join(self.e_addr_list), "--decode-servers-urls",
            ",".join(self.pd_addr_list), "--prefill-servers-urls", "disable"
        ]
        proxy_path = os.path.join(
            VLLM_PATH,
            "examples/online_serving/disaggregated_encoder/mooncake_connector/disagg_epd_proxy.py"
        )
        proxy_args = ["python", proxy_path, *proxy_args]
        self._run_server_new_session(proxy_args, self.env_dict, "[PRXOY] ")

    def _start_vllm_serve(self):
        if self.env_dict is None:
            self.env_dict = dict()
        self.env_dict['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"
        self.env_dict['VLLM_USE_V1'] = "1"

        serve_arg_cmd = ["taskset", "-c", "0-96", "vllm", "serve"]

        for i, e_serve_arg in enumerate(self.e_serve_args_list):
            self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            e_serve_arg = [*serve_arg_cmd, *e_serve_arg]
            index_e = e_serve_arg.index("--port")
            self.e_addr_list.append(
                f"http://localhost:{e_serve_arg[index_e + 1]}")
            self._run_server(e_serve_arg, self.env_dict,
                             f"[ENCODE_{i}] ")

        for i, pd_serve_arg in enumerate(self.pd_serve_args_list):
            self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            pd_serve_arg = [*serve_arg_cmd, *pd_serve_arg]
            index_pd = pd_serve_arg.index("--port")
            self.pd_addr_list.append(
                f"http://localhost:{pd_serve_arg[index_pd + 1]}")
            self._run_server(pd_serve_arg, self.env_dict, f"[PD_{i}] ")


    async def _wait_for_vllm_worker(self, max_wait_seconds) -> None:
        sleep_times = 10
        timeout_times = 3
        start_time = time.time()
        await asyncio.sleep(90)



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

    async def _wait_for_server(self,
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
                 run_mode: Literal["serve", "worker"],
                 store_type: Literal["mooncake", "storage"],
                 e_num: Optional[int],
                 pd_num: Optional[int],
                 e_serve_args: Union[list[str], str],
                 pd_serve_args: Union[list[str], str],
                 proxy_type: Literal["disagg_proxy", "proxy",
                                     "api_server"] = None,
                 kv_store_type: Literal["mooncake"] = "",
                 mooncake_args: Union[list[str], str] = None,
                 proxy_args: Union[list[str], str] = None,
                 api_server_port: Optional[int] = 10001,
                 is_image_load: Optional[bool] = True,
                 is_epd_same_card: Optional[bool] = False,
                 env_dict: Optional[dict[str, str]] = None) -> None:
        self._proc_list = list()
        self.e_num = e_num
        self.pd_num = pd_num
        self.p = None
        if run_mode not in ["serve", "worker"]:
            raise ValueError(f"run mode must be serve or worker")
        if store_type not in ["mooncake", "storage"]:
            raise ValueError(f"store type must be mooncake or storage")
        if kv_store_type not in ["mooncake", ""]:
            raise ValueError(f"kv store type must be mooncake")
        if proxy_type is not None and proxy_type not in [
                "disagg_proxy", "proxy", "api_server"
        ]:
            raise ValueError(
                f"proxy type must be disagg_proxy, proxy or api_server")
        self.run_mode = run_mode
        self.store_type = store_type
        self.protocol = ""
        self.kv_store_type = kv_store_type
        self.proxy_type = proxy_type
        self.is_image_load = is_image_load
        self.is_epd_same_card = is_epd_same_card
        self.api_server_port = api_server_port
        self.e_addr_list = list()
        self.pd_addr_list = list()
        self.p_addr_list = list()
        self.d_addr_list = list()
        self.e_serve_args_list = list()
        self.pd_serve_args_list = list()

        self.model = None
        if isinstance(e_serve_args, list):
            if not all(isinstance(item, list) for item in e_serve_args):
                for i in range(self.e_num):
                    self.e_serve_args_list.append(copy.deepcopy(e_serve_args))
            else:
                self.e_serve_args_list = e_serve_args
        else:
            raise RuntimeError("e_serve_args must be a list")

        if isinstance(pd_serve_args, list):
            if not all(isinstance(item, list) for item in pd_serve_args):
                for i in range(self.pd_num):
                    self.pd_serve_args_list.append(copy.deepcopy(pd_serve_args))
            else:
                self.pd_serve_args_list = pd_serve_args
        else:
            raise RuntimeError("pd_serve_args must be a list")

        self.mooncake_args = mooncake_args
        self.proxy_args = proxy_args

        self.env_dict = env_dict
        self._default_addr_prefix = "/tmp/"
        self.proxy_addr = None
        self.metrics = {}

    async def __aenter__(self):
        # start with
        max_wait_seconds = 1800
        if self.store_type == "mooncake" or self.kv_store_type == "mooncake":
            self._start_mooncake()
        if self.run_mode == "worker":
            self._start_vllm_worker()
            self.p = self._start_zmq_proxy()
            await self._wait_for_vllm_worker(max_wait_seconds=max_wait_seconds)
        elif self.run_mode == "serve":
            self._start_vllm_serve()
        if self.proxy_type is None:
            self.p.shutdown()
        elif self.proxy_type == "disagg_proxy":
            self._start_disagg_proxy()
            await self._wait_for_server()
        elif self.proxy_type == "api_server":
            self.p.shutdown()
            self._start_api_server()
            await self._wait_for_server()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # exit with
        for proc in self._proc_list:
            self._kill_process_tree(proc.pid)
        print("vllm instance and api server is stoping")
        if self.p is not None:
            self.p.shutdown()
        print("proxy is stoping")


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
        env['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"
        env['VLLM_USE_V1'] = "1"
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
                 server_host: str = '0.0.0.0',
                 server_port: int = 8080,
                 env_dict: Optional[dict[str, str]] = None,
                 seed: Optional[int] = None,
                 auto_port: bool = True,
                 nodes_info: Optional[list[NodeInfo]] = None,
                 disaggregated_prefill: Optional[dict] = None,
                 proxy_port: Optional[int] = None,
                 max_wait_seconds: Optional[float] = None,
                 override_hf_configs: Optional[dict[str, Any]] = None) -> None:
        if isinstance(vllm_serve_args, str):
            vllm_serve_args = shlex.split(vllm_serve_args)
        else:
            vllm_serve_args = [
                "taskset", "-c", "0-96", "vllm", "serve", model,
                *vllm_serve_args
            ]
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
        # for multi-nodes test
        self.nodes_info = nodes_info
        self.disaggregated_prefill = disaggregated_prefill
        self.cur_index = os.getenv("LWS_WORKER_INDEX", 0)
        self.proxy_port = proxy_port

        self._start_server(model, vllm_serve_args, env_dict)
        max_wait_seconds = max_wait_seconds or 1800
        if self.disaggregated_prefill:
            assert proxy_port is not None, "for disaggregated_prefill, proxy port must be provided"
            self._wait_for_server_pd(proxy_port=proxy_port,
                                     timeout=max_wait_seconds)
        else:
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

    def hang_until_terminated(self, url) -> None:
        """
        Wait until the server process terminates.
        This is for headless mode, where the api server
        process only exists in the leader node.
        """
        client = requests
        try:
            while True:
                try:
                    resp = client.get(url, timeout=5)
                    if resp.status_code != 200:
                        break
                    time.sleep(5)
                except Exception:
                    break
        finally:
            if isinstance(client, httpx.Client):
                client.close()

    def _wait_for_server_pd(self, proxy_port: int, timeout: float):
        # Wait for all api_server nodes ready
        assert self.nodes_info is not None, "cluster info must be provided"
        for node_info in self.nodes_info:
            if node_info.headless:
                continue

            url_health = f"http://{node_info.ip}:{node_info.server_port}/health"
            self._wait_for_server(url=url_health, timeout=timeout)

        # Wait for proxy ready
        master_node = self.nodes_info[0]
        url_proxy = f"http://{master_node.ip}:{proxy_port}/healthcheck"
        self._wait_for_server(url=url_proxy, timeout=timeout)

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



