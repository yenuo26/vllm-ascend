# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import importlib
import json
import os
import re
import subprocess
import traceback
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd


def get_package_location(package_name):
    try:
        distribution = importlib.metadata.distribution(package_name)
        return str(distribution.locate_file(''))
    except importlib.metadata.PackageNotFoundError:
        return None


benchmark_path = get_package_location("ais_bench_benchmark")
DATASET_CONF_DIR = os.path.join(benchmark_path,
                                "ais_bench/benchmark/configs/datasets")
REQUEST_CONF_DIR = os.path.join(benchmark_path,
                                "ais_bench/benchmark/configs/models/vllm_api")
DATASET_DIR = os.path.join(benchmark_path, "ais_bench/datasets")


class AisbenchRunner:
    RESULT_MSG = {
        "performance": "Performance Result files locate in ",
        "accuracy": "write csv to "
    }
    DATASET_RENAME = {
        "aime2024": "aime",
        "gsm8k-lite": "gsm8k",
        "textvqa-lite": "textvqa"
    }

    def _run_aisbench_task(self):
        dataset_conf = self.dataset_conf.split('/')[-1]
        if self.task_type == "accuracy":
            aisbench_cmd = [
                "taskset", "-c", "97-192", 'ais_bench', '--models',
                self.request_conf, '--datasets', f'{dataset_conf}'
            ]
        if self.task_type == "performance":
            aisbench_cmd = [
                "taskset", "-c", "97-192", 'ais_bench', '--models',
                self.request_conf, '--datasets', f'{dataset_conf}', '--mode',
                'perf'
            ]
            if self.num_prompts:
                aisbench_cmd.extend(['--num-prompts', str(self.num_prompts)])
        print(f"running aisbench cmd: {' '.join(aisbench_cmd)}")
        self.proc: subprocess.Popen = subprocess.Popen(aisbench_cmd,
                                                       stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE,
                                                       text=True)

    def __init__(self,
                 model: str,
                 port: int,
                 aisbench_config: dict,
                 verify=True,
                 save=True):
        self.model = model
        self.port = port
        self.task_type = aisbench_config["case_type"]
        self.request_conf = aisbench_config["request_conf"]
        self.dataset_conf = aisbench_config.get("dataset_conf")
        self.dataset_path = aisbench_config.get("dataset_path")
        self.num_prompts = aisbench_config.get("num_prompts")
        self.max_out_len = aisbench_config["max_out_len"]
        self.batch_size = aisbench_config["batch_size"]
        self.request_rate = aisbench_config.get("request_rate", 0)
        self.temperature = aisbench_config.get("temperature")
        self.top_k = aisbench_config.get("top_k")
        self.result_file_name = aisbench_config.get("result_file_name", "test")
        self.top_p = aisbench_config.get("top_p")
        self.seed = aisbench_config.get("seed")
        self.repetition_penalty = aisbench_config.get("repetition_penalty")
        self.exp_folder = None
        self.result_line = None
        self._init_dataset_conf()
        self._init_request_conf()
        self._run_aisbench_task()
        self._wait_for_task()
        if verify:
            self.baseline = aisbench_config.get("baseline", 1)
            if self.task_type == "accuracy":
                self.threshold = aisbench_config.get("threshold", 1)
                self._accuracy_verify()
            if self.task_type == "performance":
                self.threshold = aisbench_config.get("threshold", 0.97)
                self._performance_verify()
        if save:
            self._performance_result_save()
            self.create_result_plot([self.result_file_name])

    def create_result_plot(self, result_file_names):
        import matplotlib.ticker as ticker
        plt.rcParams['axes.unicode_minus'] = False  #display a minus sign

        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0, 0].set_title('TTFT')
            axes[0, 0].set_ylabel('TTFT(ms)')

            axes[0, 1].set_title('TPOT')
            axes[0, 1].set_ylabel('TPOT(ms)')

            axes[0, 2].set_ylabel('E2E(ms)')
            axes[0, 2].set_title('E2E')

            axes[1, 0].set_title('Request Throughput')
            axes[1, 0].set_ylabel('Request Throughput(req/s)')

            axes[1, 1].set_title('Total Token Throughput')
            axes[1, 1].set_ylabel('Total Token Throughput(token/s)')

            axes_indexs = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
            for axes in axes_indexs:
                axes.set_xlabel('Request Rate(req/s)')
                axes.grid(True, alpha=0.3)
                axes.xaxis.set_major_locator(ticker.AutoLocator())
                axes.xaxis.set_major_formatter(ticker.ScalarFormatter())

            for name in result_file_names:
                df = pd.read_csv(f"./{name}.csv")
                x = df['Request rate']
                #remove data unit
                df['TTFT_Average'] = df['TTFT_Average'].str.extract(
                    '(\d+\.?\d*)').astype(float)
                df['TPOT_Average'] = df['TPOT_Average'].str.extract(
                    '(\d+\.?\d*)').astype(float)
                df['E2EL_Average'] = df['E2EL_Average'].str.extract(
                    '(\d+\.?\d*)').astype(float)
                df['Request Throughput_total'] = df[
                    'Request Throughput_total'].str.extract('(\d+\.?\d*)').astype(
                        float)
                df['Total Token Throughput_total'] = df[
                    'Total Token Throughput_total'].str.extract(
                        '(\d+\.?\d*)').astype(float)

                # TTFT
                axes[0, 0].plot(x, df['TTFT_Average'], 'b-', linewidth=2, labels=name)
                axes[0, 0].plot(x, df['TTFT_Average'], 'bo', markersize=4)
                # display num for data point
                for i, (xi, yi) in enumerate(zip(x, df['TTFT_Average'])):
                    axes[0, 0].annotate(f'{yi:.4f}',
                                        (xi, yi),
                                        textcoords="offset points",
                                        xytext=(0, 10),  # 在点上方10像素显示
                                        ha='center',  # 水平居中
                                        va='bottom',  # 垂直底部对齐
                                        fontsize=8,
                                        color='black')
                # TPOT
                axes[0, 1].plot(x, df['TPOT_Average'], 'r-', linewidth=2, labels=name)
                axes[0, 1].plot(x, df['TPOT_Average'], 'ro', markersize=4)

                for i, (xi, yi) in enumerate(zip(x, df['TPOT_Average'])):
                    axes[0, 1].annotate(f'{yi:.4f}',
                                        (xi, yi),
                                        textcoords="offset points",
                                        xytext=(0, 10),  # 在点上方10像素显示
                                        ha='center',  # 水平居中
                                        va='bottom',  # 垂直底部对齐
                                        fontsize=8,
                                        color='black')

                # E2E
                axes[0, 2].plot(x, df['E2EL_Average'], 'g-', linewidth=2, labels=name)
                axes[0, 2].plot(x, df['E2EL_Average'], 'go', markersize=4)

                for i, (xi, yi) in enumerate(zip(x, df['E2EL_Average'])):
                    axes[0, 2].annotate(f'{yi:.4f}',
                                        (xi, yi),
                                        textcoords="offset points",
                                        xytext=(0, 10),  # 在点上方10像素显示
                                        ha='center',  # 水平居中
                                        va='bottom',  # 垂直底部对齐
                                        fontsize=8,
                                        color='black')

                # Request Throughput
                axes[1, 0].plot(x,
                                df['Request Throughput_total'],
                                'm-',
                                linewidth=2, labels=name)
                axes[1, 0].plot(x,
                                df['Request Throughput_total'],
                                'mo',
                                markersize=4)

                for i, (xi, yi) in enumerate(zip(x, df['Request Throughput_total'])):
                    axes[1, 0].annotate(f'{yi:.4f}',
                                        (xi, yi),
                                        textcoords="offset points",
                                        xytext=(0, 10),  # 在点上方10像素显示
                                        ha='center',  # 水平居中
                                        va='bottom',  # 垂直底部对齐
                                        fontsize=8,
                                        color='black')


                # Total Token Throughput
                axes[1, 1].plot(x,
                                df['Total Token Throughput_total'],
                                'c-',
                                linewidth=2, labels=name)
                axes[1, 1].plot(x,
                                df['Total Token Throughput_total'],
                                'co',
                                markersize=4)

                for i, (xi, yi) in enumerate(zip(x, df['Total Token Throughput_total'])):
                    axes[1, 1].annotate(f'{yi:.4f}',
                                        (xi, yi),
                                        textcoords="offset points",
                                        xytext=(0, 10),  # 在点上方10像素显示
                                        ha='center',  # 水平居中
                                        va='bottom',  # 垂直底部对齐
                                        fontsize=8,
                                        color='black')


            axes[1, 2].set_visible(False)
            plt.tight_layout()

            fig.suptitle('', fontsize=16, y=0.98)


            if len(result_file_names) == 1:
                plt.savefig(f'./{result_file_names[0]}.png',
                            dpi=300,
                            bbox_inches='tight')
                print(f"Result figure is locate in {result_file_names[0]}.png")
            else:
                today = date.today()
                plt.savefig(f'./test_perf_result_{today}.png',
                            dpi=300,
                            bbox_inches='tight')
                print(f"Result figure is locate in test_perf_result_{today}.png")



        except Exception as e:
            print(f"ERROR: {str(e)}")

    def _performance_result_save(self):
        try:
            csv_result = defaultdict(dict)
            for index, row in self.result_csv.iterrows():
                performance_param = row['Performance Parameters']
                data = {
                    'Average':
                    str(row['Average']) if pd.notna(row['Average']) else None,
                    'Min':
                    str(row['Min']) if pd.notna(row['Min']) else None,
                    'Max':
                    str(row['Max']) if pd.notna(row['Max']) else None,
                    'Median':
                    str(row['Median']) if pd.notna(row['Median']) else None,
                    'P75':
                    str(row['P75']) if pd.notna(row['P75']) else None,
                    'P90':
                    str(row['P90']) if pd.notna(row['P90']) else None,
                    'P99':
                    str(row['P99']) if pd.notna(row['P99']) else None
                }

                if performance_param not in csv_result:
                    csv_result[performance_param] = {}

                csv_result[performance_param] = data
                csv_result = dict(csv_result)
            merged_json = {"Request rate": self.request_rate}
            merged_json.update(self.result_json)
            merged_json.update(csv_result)
            self._write_to_execl(merged_json, f"./{self.result_file_name}.csv")
            print(f"Result csv file is locate in {self.result_file_name}.csv")
        except Exception as e:
            print(
                f"save result failed, reason is: {str(e)}, traceback is: {traceback.print_exc()}"
            )

    def _flatten_dict(self, data, parent_key='', sep="_"):
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(
                    self._flatten_dict(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    items.append((f"{new_key}{sep}{i}", item))
            else:
                items.append((new_key, value))
        return dict(items)

    def _write_to_execl(self, data, path):
        data = self._flatten_dict(data)
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

    def _init_dataset_conf(self):
        if self.task_type == "accuracy":
            dataset_name = os.path.basename(self.dataset_path)
            dataset_rename = self.DATASET_RENAME.get(dataset_name, "")
            dst_dir = os.path.join(DATASET_DIR, dataset_rename)
            command = ["cp", "-r", self.dataset_path, dst_dir]
            subprocess.call(command)
        if self.task_type == "performance":
            conf_path = os.path.join(DATASET_CONF_DIR,
                                     f'{self.dataset_conf}.py')
            if self.dataset_conf.startswith("textvqa"):
                self.dataset_path = os.path.join(self.dataset_path,
                                                 "textvqa_val.jsonl")
            with open(conf_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = re.sub(r'path=.*', f'path="{self.dataset_path}",',
                             content)
            conf_path_new = os.path.join(DATASET_CONF_DIR,
                                         f'{self.dataset_conf}.py')
            with open(conf_path_new, 'w', encoding='utf-8') as f:
                f.write(content)

    def _init_request_conf(self):
        conf_path = os.path.join(REQUEST_CONF_DIR, f'{self.request_conf}.py')
        with open(conf_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'model=.*', f'model="{self.model}",', content)
        content = re.sub(r'host_port.*', f'host_port = {self.port},', content)
        content = re.sub(r'max_out_len.*',
                         f'max_out_len = {self.max_out_len},', content)
        content = re.sub(r'batch_size.*', f'batch_size = {self.batch_size},',
                         content)
        if self.task_type == "performance":
            content = re.sub(r'path=.*', f'path="{self.model}",', content)
            content = re.sub(r'request_rate.*',
                             f'request_rate = {self.request_rate},', content)
            if "ignore_eos" not in content:
                content = re.sub(
                    r"temperature.*",
                    "temperature = 0,\n            ignore_eos = True,",
                    content)
        if self.task_type == "accuracy":
            if "ignore_eos" not in content:
                content = re.sub(
                    r"temperature.*",
                    "temperature = 0.6,\n            ignore_eos = False,",
                    content)
        if self.temperature:
            content = re.sub(r"temperature.*",
                             f"temperature = {self.temperature},", content)
        if self.top_p:
            content = re.sub(r"top_p.*", f"top_p = {self.top_p},", content)
        if self.top_k:
            content = re.sub(r"top_k.*", f"top_k = {self.top_k},", content)
        if self.seed:
            content = re.sub(r"seed.*", f"seed = {self.seed},", content)
        if self.repetition_penalty:
            content = re.sub(
                r"repetition_penalty.*",
                f"repetition_penalty = {self.repetition_penalty},", content)
        conf_path_new = os.path.join(REQUEST_CONF_DIR,
                                     f'{self.request_conf}.py')
        with open(conf_path_new, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"The request config is\n {content}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.terminate()
        try:
            self.proc.wait(8)
        except subprocess.TimeoutExpired:
            # force kill if needed
            self.proc.kill()

    def _wait_for_exp_folder(self):
        while True:
            line = self.proc.stdout.readline().strip()
            print(line)
            if "Current exp folder: " in line:
                self.exp_folder = re.search(r'Current exp folder: (.*)',
                                            line).group(1)
                return
            if "ERROR" in line:
                raise RuntimeError(
                    "Some errors happen to Aisbench task.") from None

    def _wait_for_task(self):
        self._wait_for_exp_folder()
        result_msg = self.RESULT_MSG[self.task_type]
        while True:
            line = self.proc.stdout.readline().strip()
            print(line)
            if result_msg in line:
                self.result_line = line
                return
            if "ERROR" in line:
                raise RuntimeError(
                    "Some errors happen to Aisbench task.") from None

    def _get_result_performance(self):
        result_dir = re.search(r'Performance Result files locate in (.*)',
                               self.result_line).group(1)[:-1]
        dataset_type = self.dataset_conf.split('/')[0]
        result_csv_file = os.path.join(result_dir,
                                       f"{dataset_type}dataset.csv")
        result_json_file = os.path.join(result_dir,
                                        f"{dataset_type}dataset.json")
        self.result_csv = pd.read_csv(result_csv_file)
        print("Getting performance results from file: ", result_json_file)
        with open(result_json_file, 'r', encoding='utf-8') as f:
            self.result_json = json.load(f)

    def _get_result_accuracy(self):
        acc_file = re.search(r'write csv to (.*)', self.result_line).group(1)
        df = pd.read_csv(acc_file)
        return float(df.loc[0][-1])

    def _performance_verify(self):
        self._get_result_performance()
        output_throughput = self.result_json["Output Token Throughput"][
            "total"].replace("token/s", "")
        assert float(
            output_throughput
        ) >= self.threshold * self.baseline, f"Performance verification failed. The current Output Token Throughput is {output_throughput} token/s, which is not greater than or equal to {self.threshold} * baseline {self.baseline}."

    def _accuracy_verify(self):
        acc_value = self._get_result_accuracy()
        assert self.baseline - self.threshold <= acc_value <= self.baseline + self.threshold, f"Accuracy verification failed. The accuracy of {self.dataset_path} is {acc_value}, which is not within {self.threshold} relative to baseline {self.baseline}."


def run_aisbench_cases(model, port, aisbench_cases, verify=True, save=True):
    aisbench_errors = []
    for aisbench_case in aisbench_cases:
        try:
            with AisbenchRunner(model, port, aisbench_case, verify, save):
                pass
        except Exception as e:
            aisbench_errors.append([aisbench_case, e, traceback.print_exc()])
            print(e)
    for failed_case, error_info, error_traceback in aisbench_errors:
        print(
            f"The following aisbench case failed: {failed_case}, reason is {error_info}, traceback is: {error_traceback}."
        )
    assert not aisbench_errors, "some aisbench cases failed, info were shown above."
