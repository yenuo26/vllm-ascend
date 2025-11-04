import asyncio
import os
import shlex
import subprocess
import sys
import time

import psutil
import threading
import requests

from pathlib import Path
from typing import Optional, Union
from llm_service.protocol.protocol import ServerType
from llm_service.apis.vllm.proxy import Proxy


class RemoteEPDServer:

    def get_proxy(self) -> Proxy:
        return self.p

    def _run_server(self, server_cmd: list[str],
                    env_dict: Optional[dict[str, str]]) -> None:
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
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._proc_list.append(proc)

    def _read_output(self, pipe, prefix):
        """在单独线程中读取输出"""
        try:
            with pipe:
                for line in iter(pipe.readline, ''):
                    if line:  # 避免空行
                        print(f"{prefix}: {line}", end='')
        except Exception as e:
            print(f"error: {e}")

    def _run_server_new_session(self, server_cmd: list[str],
                                env_dict: Optional[dict[str, str]]) -> None:
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
                                         args=(proc.stdout, "UVICORN-STDOUT"),
                                         daemon=True)
        stderr_thread = threading.Thread(target=self._read_output,
                                         args=(proc.stderr, "UVICORN-STDERR"),
                                         daemon=True)

        stdout_thread.start()
        stderr_thread.start()
        self._proc_list.append(proc)

    def _start_api_server(self) -> None:
        api_server_args = [
            "--host", "127.0.0.1", "--port",
            str(self.api_server_port), "--model", self.model, "--proxy-addr",
            self.proxy_addr, "--e-addr-list", ",".join(self.e_addr_list),
            "--pd-addr-list", ",".join(self.pd_addr_list)
        ]
        if self.is_image_load:
            api_server_args.append("--is-load-image")
        api_server_path = Path(
            __file__).parent.parent.parent / "tools" / "api_server.py"
        api_server_args = ["python", api_server_path, *api_server_args]
        self._run_server_new_session(api_server_args, None)

    def _start_vllm(self):
        if self.env_dict is None:
            self.env_dict = dict()
        self.env_dict['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = "1"
        self.env_dict['VLLM_USE_V1'] = "1"
        if isinstance(self.e_serve_args, str):
            self.e_serve_args = shlex.split(self.e_serve_args)
        if isinstance(self.pd_serve_args, str):
            self.pd_serve_args = shlex.split(self.pd_serve_args)
        else:
            self.e_serve_args = [
                "python", "-m", "llm_service.entrypoints.worker",
                *self.e_serve_args
            ]
            self.pd_serve_args = [
                "python", "-m", "llm_service.entrypoints.worker",
                *self.pd_serve_args
            ]

        if "--proxy-addr" not in self.e_serve_args and "--proxy-addr" not in self.pd_serve_args:
            # defaut proxy-addr is /tmp/proxy
            self.e_serve_args = self.e_serve_args + [
                "--proxy-addr", self._default_addr_prefix + "proxy"
            ]
            self.pd_serve_args = self.pd_serve_args + [
                "--proxy-addr", self._default_addr_prefix + "proxy"
            ]
        else:
            try:
                index_e = self.e_serve_args.index("--proxy-addr")
                index_pd = self.pd_serve_args.index("--proxy-addr")
            except ValueError:
                print("e instance proxy addr must be same as pd instance")
                return
            self.proxy_addr = self.e_serve_args[index_e + 1]

        if "--model" not in self.e_serve_args or "--model" not in self.pd_serve_args:
            raise ValueError("must carry --model")

        else:
            index_e = self.e_serve_args.index("--model")
            self.model = self.e_serve_args[index_e + 1]

        for i in range(self.e_num):
            if self.is_e_same_card:
                self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = "0"
            else:
                self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            if "--worker-addr" not in self.e_serve_args:
                # defaut encode-addr is /tmp/encode_{i}
                self.e_serve_args = self.e_serve_args + [
                    "--worker-addr",
                    self._default_addr_prefix + "encoder_" + str(i)
                ]
                self.e_addr_list.append(self._default_addr_prefix +
                                        "encoder_" + str(i))
            else:
                index_e = self.e_serve_args.index("--worker-addr")
                self.e_addr_list.append(self.e_serve_args[index_e + 1])
            self._run_server(self.e_serve_args, self.env_dict)
        for i in range(self.pd_num):
            if self.is_epd_same_card:
                self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i)
            else:
                self.env_dict["ASCEND_RT_VISIBLE_DEVICES"] = str(i +
                                                                 self.e_num)
            if "--worker-addr" not in self.pd_serve_args:
                # defaut worker-addr is /tmp/pd_{i}
                self.pd_serve_args = self.pd_serve_args + [
                    "--worker-addr", self._default_addr_prefix + "pd_" + str(i)
                ]
                self.pd_addr_list.append(self._default_addr_prefix + "pd_" +
                                         str(i))
            else:
                index_pd = self.pd_serve_args.index("--worker-addr")
                self.pd_addr_list.append(self.pd_serve_args[index_pd + 1])
            self._run_server(self.pd_serve_args, self.env_dict)

    async def _wait_for_vllm_server(self, max_wait_seconds) -> None:
        sleep_times = 10
        timeout_times = 3
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            tasks_0 = [
                asyncio.create_task(
                    asyncio.wait_for(self.p.check_health(
                        ServerType.E_INSTANCE, iid),
                                     timeout=timeout_times))
                for iid in range(self.e_num)
            ]
            tasks_1 = [
                asyncio.create_task(
                    asyncio.wait_for(self.p.check_health(
                        ServerType.PD_INSTANCE, iid),
                                     timeout=timeout_times))
                for iid in range(self.pd_num)
            ]
            tasks = tasks_0 + tasks_1

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

    async def _wait_for_api_server(self,
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
                 start_mode: Optional[str],
                 e_num: Optional[int],
                 pd_num: Optional[int],
                 e_serve_args: Union[list[str], str],
                 pd_serve_args: Union[list[str], str],
                 api_server_port: Optional[int] = 10001,
                 is_image_load: Optional[bool] = True,
                 is_epd_same_card: Optional[bool] = False,
                 is_e_same_card: Optional[bool] = False,
                 env_dict: Optional[dict[str, str]] = None) -> None:
        self._proc_list = list()
        self.e_num = e_num
        self.pd_num = pd_num
        self.start_mode = start_mode
        self.is_image_load = is_image_load
        self.is_epd_same_card = is_epd_same_card
        self.is_e_same_card = is_e_same_card
        self.api_server_port = api_server_port
        self.e_addr_list = list()
        self.pd_addr_list = list()

        self.model = str()
        self.e_serve_args = e_serve_args
        self.pd_serve_args = pd_serve_args
        self.env_dict = env_dict
        self._default_addr_prefix = "/tmp/"
        self.proxy_addr = self._default_addr_prefix + "proxy"

    async def __aenter__(self):
        # start with
        max_wait_seconds = 1800
        self._start_vllm()
        self.p = Proxy(proxy_addr=self.proxy_addr,
                       encode_addr_list=self.e_addr_list,
                       pd_addr_list=self.pd_addr_list,
                       enable_health_monitor=True,
                       model_name=self.model)
        await self._wait_for_vllm_server(max_wait_seconds=max_wait_seconds)
        if self.start_mode == "http":
            self.p.shutdown()
            self._start_api_server()
            await self._wait_for_api_server()
        elif self.start_mode == "api":
            self.p.shutdown()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # exit with
        for proc in self._proc_list:
            self._kill_process_tree(proc.pid)
        print("vllm instance and api server is stoping")
        self.p.shutdown()
        print("proxy is stoping")
