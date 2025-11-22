import asyncio
import contextlib
import gc
import json
import os
import re
import shlex
import copy
import subprocess
import tempfile
import sys
import threading
import traceback
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypeVar, Union, Literal

import httpx
import numpy as np
import pandas as pd
import openai
import psutil
import pytest
import requests
import torch
import importlib
from PIL import Image
from datetime import datetime
from lm_service.apis.vllm.proxy import Proxy
from lm_service.protocol.protocol import ServerType
from lm_service.routing_logic import RandomRouter, RoundRobinRouter, LeastInFlightRouter
from modelscope import snapshot_download  # type: ignore[import-untyped]
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, BatchFeature)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from vllm import LLM, SamplingParams
from vllm.config.model import TaskOption, _get_and_verify_dtype
from vllm.inputs import TextPrompt
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.transformers_utils.utils import maybe_model_redirect

from tests.e2e.model_utils import (TokensTextLogprobs,
                                   TokensTextLogprobsPromptLogprobs)
from tests.e2e.nightly.multi_node.config.multi_node_config import NodeInfo
from tests.e2e.nightly.multi_node.config.multi_node_epd_config import ClusterManager
from tests.e2e.nightly.multi_node.config.utils import get_cluster_ips

from vllm_ascend.ascend_config import clear_ascend_config
# TODO: remove this part after the patch merged into vllm, if
# we not explicitly patch here, some of them might be effectiveless
# in pytest scenario
from vllm_ascend.utils import adapt_patch  # noqa E402
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.11.0"):
    from vllm.utils import get_open_port
else:
    from vllm.utils.network_utils import get_open_port

adapt_patch(True)
adapt_patch(False)

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
                        if self.env_dict.get("TIMECOUNT_ENABLED", 0) == "1":
                            self._extract_ttft_data(line, prefix)

        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc()

    def _extract_ttft_data(self, text, prefix):
        if "PROXY" in prefix.upper():
            patterns = {
                'transfer_to_encode':
                r'Avg proxy to encoder requests: ([\d.]+) ms',
                'transfer_to_pd': r'Avg proxy to pd requests: ([\d.]+) ms'}
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
            bufsize=1,
            universal_newlines=True)

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
            str(self.api_server_port), "--proxy-config",
            json.dumps(self.proxy_config)
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
            "fast_transfer_buffer_size": 1,
            "metadata_server": "http://0.0.0.0:8081/metadata",
            "master_server_address": "0.0.0.0:50051"
        }
        for i, arg in enumerate(self.mooncake_args):
            if "--http_metadata_server_port" in arg:
                metadata_server_port = self.mooncake_args[i].split("=")[-1]
                if self.node_info is not None:
                    mooncake_json[
                        "metadata_server"] = f"http://{self.cluster_ips[0]}:{metadata_server_port}/metadata"

            if "--rpc_port" in arg:
                rpc_port = self.mooncake_args[i + 1]
                if self.node_info is not None:
                    mooncake_json["master_server_address"] = f"{self.cluster_ips[0]}:{rpc_port}"

        config_paths = set()
        if self.store_type == "mooncake":
            for i, arg in enumerate(self.e_serve_args_list +
                                    self.pd_serve_args_list):
                index = arg.index("--ec-transfer-config")
                config_paths.add(json.loads(
                    arg[index + 1]).get("ec_connector_extra_config").get(
                        "ec_mooncake_config_file_path"))

        if self.kv_store_type == "mooncake":
            config_paths.add(self.env_dict["MOONCAKE_CONFIG_PATH"])
        if self.node_info is not None:
            for host in self.cluster_ips[1:]:
                mooncake_json["local_hostname"] = host
                for config_path in config_paths:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(mooncake_json, f, ensure_ascii=False, indent=4)
                        local_config_path = f.name
                    try:
                        scp_cmd = ["scp", local_config_path, f"root@{host}:{config_path}"]
                        subprocess.run(scp_cmd, check=True)
                    finally:
                        os.unlink(local_config_path)
            mooncake_json["local_hostname"] = self.cluster_ips[0]
        for config_path in config_paths:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(mooncake_json, f, ensure_ascii=False, indent=4)


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
            if self.node_info is not None:
                if self.node_info.get_node_info(role.lower()) is not None:
                    node_id = self.node_info.get_node_info(role.lower(), i).node_id
                    host = self.cluster_ips[node_id]
                else:
                    host = self.cluster_ips[0]
                proxy_host = self.cluster_ips[0]
            else:
                host = "127.0.0.1"
                proxy_host = "127.0.0.1"
            if role.lower() == "e":
                return {
                    "proxy_addr": f"{proxy_host}:37000",
                    "worker_addr": f"{host}:3800{i}"
                }
            else:
                return {
                    "proxy_addr": f"{proxy_host}:37000",
                    "worker_addr": f"{host}:3900{i}"
                }
        else:
            if role.lower() == "e":
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
                "lm_service.entrypoints.worker"
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

            if self.node_info is not None and self.node_info.get_node_info(
                    "e") is not None:
                node_id = self.node_info.get_node_info("e", i).node_id
                self._run_in_remote_container(
                    host=self.cluster_ips[node_id],
                    container_name=self.node_info.get_node_info(
                        "e", i).container_name,
                    server_cmd=e_serve_arg,
                    env_dict=self.env_dict,
                    log_prefix=f"[ENCODE_{i}] ",
                )
            else:
                self._run_server(e_serve_arg, self.env_dict, f"[ENCODE_{i}] ")

        for i, pd_serve_arg in enumerate(self.pd_serve_args_list):
            if self.is_epd_same_card:
                self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            else:
                self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i +
                                                                 self.e_num)
            pd_serve_arg = [*serve_arg_cmd, *pd_serve_arg]
            if "--model" not in pd_serve_arg:
                raise ValueError("must carry --model")
            else:
                index_pd = pd_serve_arg.index("--model")
                if self.model is not None and pd_serve_arg[index_pd +
                                                           1] != self.model:
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
            role = ""
            if "--kv-transfer-config" in pd_serve_arg:
                kv_index = pd_serve_arg.index("--kv-transfer-config")
                if "kv_consumer" in pd_serve_arg[kv_index + 1]:
                    self.d_addr_list.append(pd_serve_arg[worker_index + 1])
                    log_prefix = f"[D_{i}] "
                    role = "d"
                elif "kv_producer" in pd_serve_arg[kv_index + 1]:
                    self.p_addr_list.append(pd_serve_arg[worker_index + 1])
                    log_prefix = f"[P_{i}] "
                    role = "p"
            else:
                self.pd_addr_list.append(pd_serve_arg[worker_index + 1])
                log_prefix = f"[PD_{i}] "
                role = "pd"

            if self.node_info is not None and self.node_info.get_node_info(
                    role) is not None:
                node_id = self.node_info.get_node_info(role, i).node_id
                self._run_in_remote_container(
                    host=self.cluster_ips[node_id],
                    container_name=self.node_info.get_node_info(
                        role, i).container_name,
                    server_cmd=pd_serve_arg,
                    env_dict=self.env_dict,
                    log_prefix=log_prefix,
                )
            else:
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
            self.proxy_config['transfer_protocol'] = self.proxy_args[
                self.proxy_args.index("--transfer_protocol") + 1]
        if self.proxy_args is not None and "--enable-health-monitor" in self.proxy_args:
            self.proxy_config['enable_health_monitor'] = self.proxy_args[
                self.proxy_args.index("--enable-health-monitor") + 1]
        if self.proxy_args is not None and "--router" in self.proxy_args:
            self.proxy_config['router'] = self.proxy_args[
                self.proxy_args.index("--router") + 1]
            if self.proxy_args[self.proxy_args.index("router") +
                               1] == "RandomRouter":
                self.proxy_config['router'] = RandomRouter
            elif self.proxy_args[self.proxy_args.index("router") +
                                 1] == "RoundRobinRouter":
                self.proxy_config['router'] = RoundRobinRouter
            else:
                self.proxy_config['router'] = LeastInFlightRouter
        p = Proxy(**self.proxy_config)
        if self.proxy_args is not None and "--router" in self.proxy_args:
            self.proxy_config['router'] = self.proxy_args[
                self.proxy_args.index("--router") + 1]
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
            self._run_server(e_serve_arg, self.env_dict, f"[ENCODE_{i}] ")

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

        while time.time() - start_time < max_wait_seconds:
            tasks_0 = [
                asyncio.create_task(
                    asyncio.wait_for(self.p.check_health(
                        ServerType.E_INSTANCE, f"{self.protocol}://{addr}"),
                                     timeout=timeout_times))
                for addr in self.e_addr_list
            ]
            if self.pd_addr_list:
                tasks_1 = [
                    asyncio.create_task(
                        asyncio.wait_for(self.p.check_health(
                            ServerType.PD_INSTANCE, f"{self.protocol}://{addr}"),
                                         timeout=timeout_times))
                    for addr in self.pd_addr_list
                ]
                tasks = tasks_0 + tasks_1
            else:
                tasks_1 = [
                    asyncio.create_task(
                        asyncio.wait_for(self.p.check_health(
                            ServerType.P_INSTANCE, f"{self.protocol}://{addr}"),
                            timeout=timeout_times))
                    for addr in self.p_addr_list
                ]
                tasks_2 = [
                    asyncio.create_task(
                        asyncio.wait_for(self.p.check_health(
                            ServerType.D_INSTANCE, f"{self.protocol}://{addr}"),
                            timeout=timeout_times))
                    for addr in self.d_addr_list
                ]
                tasks = tasks_0 + tasks_1 + tasks_2

            results = await asyncio.gather(*tasks, return_exceptions=True)
            if all([isinstance(result, bool) and result
                    for result in results]):
                print("All instances are ready")
                return
            else:
                print(f"current results: {results}")
                await asyncio.sleep(sleep_times)

        raise RuntimeError("epd instance start failed!")

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
                 node_info: ClusterManager = None,
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
        self.node_info = node_info

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
                    self.pd_serve_args_list.append(
                        copy.deepcopy(pd_serve_args))
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
        if node_info is not None:
            self.cluster_ips = get_cluster_ips()

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


class VllmRunner:

    def __init__(
        self,
        model_name: str,
        task: TaskOption = "auto",
        tokenizer_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        # Use smaller max model length, otherwise bigger model cannot run due
        # to kv cache size limit.
        max_model_len: int = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: bool = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            task=task,
            tokenizer=tokenizer_name,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=True,
            dtype=dtype,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            disable_log_stats=disable_log_stats,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
            quantization=quantization,
            **kwargs,
        )

    def get_inputs(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[TextPrompt]:
        if images is not None:
            assert len(prompts) == len(images)

        if videos is not None:
            assert len(prompts) == len(videos)

        if audios is not None:
            assert len(prompts) == len(audios)

        inputs = [TextPrompt(prompt=prompt) for prompt in prompts]
        if images is not None:
            for i, image in enumerate(images):
                if image is not None:
                    inputs[i]["multi_modal_data"] = {"image": image}

        if videos is not None:
            for i, video in enumerate(videos):
                if video is not None:
                    inputs[i]["multi_modal_data"] = {"video": video}

        if audios is not None:
            for i, audio in enumerate(audios):
                if audio is not None:
                    inputs[i]["multi_modal_data"] = {"audio": audio}

        return inputs

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params)

        outputs: List[Tuple[List[List[int]], List[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: List[List[int]] = []
            req_sample_output_strs: List[str] = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    @staticmethod
    def _final_steps_generate_w_logprobs(
        req_outputs: List[RequestOutput],
    ) -> List[TokensTextLogprobsPromptLogprobs]:
        outputs: List[TokensTextLogprobsPromptLogprobs] = []
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs,
                            req_output.prompt_logprobs))
        return outputs

    def generate_w_logprobs(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params)

        toks_str_logsprobs_prompt_logprobs = (
            self._final_steps_generate_w_logprobs(req_outputs))
        # Omit prompt logprobs if not required by sampling params
        return ([x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
                if sampling_params.prompt_logprobs is None else
                toks_str_logsprobs_prompt_logprobs)

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[Tuple[List[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts,
                                greedy_params,
                                images=images,
                                videos=videos,
                                audios=audios)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop: Optional[List[str]] = None,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids,
            stop=stop)

        return self.generate_w_logprobs(prompts,
                                        greedy_logprobs_params,
                                        images=images,
                                        audios=audios,
                                        videos=videos)

    def encode(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[List[float]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.embed(inputs)
        return [req_output.outputs.embedding for req_output in req_outputs]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        clear_ascend_config()
        cleanup_dist_env_and_memory()


class HfRunner:

    def get_default_device(self):

        return ("cpu"
                if current_platform.is_cpu() else current_platform.device_type)

    def wrap_device(self, x: _T, device: Optional[str] = None) -> _T:
        if x is None or isinstance(x, (bool, )):
            return x

        if device is None:
            device = self.device

        if isinstance(x, dict):
            return {k: self.wrap_device(v, device) for k, v in x.items()}

        if hasattr(x, "device") and x.device.type == device:
            return x

        return x.to(device)

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        *,
        model_kwargs: Optional[dict[str, Any]] = None,
        trust_remote_code: bool = True,
        is_sentence_transformer: bool = False,
        is_cross_encoder: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: type[_BaseAutoModelClass] = AutoModelForCausalLM,
    ) -> None:
        model_name = maybe_model_redirect(model_name)
        self.model_name = model_name

        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.device = self.get_default_device()
        self.dtype = torch_dtype = _get_and_verify_dtype(
            self.model_name,
            self.config,
            dtype=dtype,
            is_pooling_model=is_sentence_transformer or is_cross_encoder,
        )

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        model_kwargs.setdefault("torch_dtype", torch_dtype)

        if is_sentence_transformer:
            # Lazy init required for AMD CI
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        elif is_cross_encoder:
            # Lazy init required for AMD CI
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                model_name,
                device=self.device,
                automodel_args=model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = auto_cls.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )

            # in case some unquantized custom models are not in same dtype
            if (getattr(model, "quantization_method", None) is None
                    and any(p.dtype != self.dtype
                            for p in model.parameters())):
                model = model.to(dtype=self.dtype)

            if (getattr(model, "quantization_method", None) != "bitsandbytes"
                    and len({p.device
                             for p in model.parameters()}) < 2):
                model = model.to(device=self.device)

            self.model = model

        if not skip_tokenizer_init:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )

        # don't put this import at the top level
        # it will call torch.cuda.device_count()
        from transformers import AutoProcessor  # noqa: F401
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if skip_tokenizer_init:
            self.tokenizer = self.processor.tokenizer

    def encode(self, prompts: list[str], *args,
               **kwargs) -> list[list[torch.Tensor]]:
        return self.model.encode(prompts, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def ilama_lora_files():
    return snapshot_download(repo_id="vllm-ascend/ilama-text2sql-spider")


def qwen_prompt(questions: List[str]) -> List[str]:
    placeholder = "<|image_pad|>"
    return [("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
             f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
             f"{q}<|im_end|>\n<|im_start|>assistant\n") for q in questions]


PROMPT_TEMPLATES = {
    "qwen2.5vl": qwen_prompt,
}


@pytest.fixture(params=list(PROMPT_TEMPLATES.keys()))
def prompt_template(request):
    return PROMPT_TEMPLATES[request.param]
