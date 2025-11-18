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
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np


def get_package_location(package_name):
    try:
        distribution = importlib.metadata.distribution(package_name)
        return str(distribution.locate_file(''))
    except importlib.metadata.PackageNotFoundError:
        return None


def create_result_plot(result_file_names, result_figure_prefix="test_perf_result"):
    plt.rcParams['axes.unicode_minus'] = False  #display a minus sign
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(result_file_names)}

    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 18))
        axes[0, 0].set_title('TTFT')
        axes[0, 0].set_ylabel('TTFT(ms)')

        axes[0, 1].set_title('TPOT')
        axes[0, 1].set_ylabel('TPOT(ms)')

        axes[0, 2].set_ylabel('E2E(ms)')
        axes[0, 2].set_title('E2E')

        axes[1, 0].set_title('Request Throughput/Card')
        axes[1, 0].set_ylabel('Request Throughput/Card(req/s)')

        axes[1, 1].set_title('Total Token Throughput/Card')
        axes[1, 1].set_ylabel('Total Token Throughput/Card(token/s)')


        for i, name in enumerate(result_file_names):
            df = pd.read_csv(f"./{name}.csv")
            x = df['Request rate/Card']
            #remove data unit
            df['TTFT_Average'] = df['TTFT_Average'].str.extract(
                r'(\d+\.?\d*)').astype(float)
            df['TPOT_Average'] = df['TPOT_Average'].str.extract(
                r'(\d+\.?\d*)').astype(float)
            df['E2EL_Average'] = df['E2EL_Average'].str.extract(
                r'(\d+\.?\d*)').astype(float)

            color = color_map[name]
            # TTFT
            axes[0, 0].plot(x,
                            df['TTFT_Average'],
                            linewidth=2,
                            color=color,
                            label=name)
            axes[0, 0].plot(x, df['TTFT_Average'], color=color, markersize=4)
            # display num for data point
            for i, (xi, yi) in enumerate(zip(x, df['TTFT_Average'])):
                axes[0, 0].annotate(
                    f'{yi:.2f}',
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),  # 在点上方10像素显示
                    ha='center',  # 水平居中
                    va='bottom',  # 垂直底部对齐
                    fontsize=8,
                    color='black')
            # TPOT
            axes[0, 1].plot(x,
                            df['TPOT_Average'],
                            linewidth=2,
                            color=color,
                            label=name)
            axes[0, 1].plot(x, df['TPOT_Average'], color=color, markersize=4)

            for i, (xi, yi) in enumerate(zip(x, df['TPOT_Average'])):
                axes[0, 1].annotate(
                    f'{yi:.2f}',
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),  # 在点上方10像素显示
                    ha='center',  # 水平居中
                    va='bottom',  # 垂直底部对齐
                    fontsize=8,
                    color='black')

            # E2E
            axes[0, 2].plot(x,
                            df['E2EL_Average'],
                            linewidth=2,
                            color=color,
                            label=name)
            axes[0, 2].plot(x, df['E2EL_Average'], color=color, markersize=4)

            for i, (xi, yi) in enumerate(zip(x, df['E2EL_Average'])):
                axes[0, 2].annotate(
                    f'{yi:.2f}',
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),  # 在点上方10像素显示
                    ha='center',  # 水平居中
                    va='bottom',  # 垂直底部对齐
                    fontsize=8,
                    color='black')

            # Request Throughput
            axes[1, 0].plot(x,
                            df['Request Throughput/Card'],
                            linewidth=2,
                            color=color,
                            label=name)
            axes[1, 0].plot(x,
                            df['Request Throughput/Card'],
                            color=color,
                            markersize=4)

            for i, (xi, yi) in enumerate(zip(x,
                                             df['Request Throughput/Card'])):
                axes[1, 0].annotate(
                    f'{yi:.3f}',
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),  # 在点上方10像素显示
                    ha='center',  # 水平居中
                    va='bottom',  # 垂直底部对齐
                    fontsize=8,
                    color='black')

            # Total Token Throughput
            axes[1, 1].plot(x,
                            df['Total Token Throughput/Card'],
                            linewidth=2,
                            color=color,
                            label=name)
            axes[1, 1].plot(x,
                            df['Total Token Throughput/Card'],
                            color=color,
                            markersize=4)

            for i, (xi,
                    yi) in enumerate(zip(x,
                                         df['Total Token Throughput/Card'])):
                axes[1, 1].annotate(
                    f'{yi:.2f}',
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),  # 在点上方10像素显示
                    ha='center',  # 水平居中
                    va='bottom',  # 垂直底部对齐
                    fontsize=8,
                    color='black')

        axes_indexs = [
            axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]
        ]
        for axes_obj in axes_indexs:
            axes_obj.set_xlabel('Request Rate/Card(req/s)')
            axes_obj.grid(True, alpha=0.3)
            axes_obj.xaxis.set_major_locator(ticker.AutoLocator())
            axes_obj.xaxis.set_major_formatter(ticker.ScalarFormatter())
            axes_obj.legend()

        axes[1, 2].set_visible(False)
        plt.tight_layout()

        fig.suptitle('', fontsize=16, y=0.98)

        if len(result_file_names) == 1:
            plt.savefig(f'./{result_file_names[0]}.png',
                        dpi=200,
                        bbox_inches='tight')
            print(f"Result figure is locate in {result_file_names[0]}.png")
        else:
            today = date.today()
            plt.savefig(f'./{result_figure_prefix}_{today}.png',
                        dpi=200,
                        bbox_inches='tight')
            print(f"Result figure is locate in {result_figure_prefix}_{today}.png")

    except Exception as e:
        print(f"ERROR: {str(e)}")

def create_ttft_plot(result_file_names, result_figure_prefix="test_perf_result"):
    plt.rcParams['axes.unicode_minus'] = False  #display a minus sign
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(result_file_names)}

    try:
        all_data = []
        file_names = []

        for file in result_file_names:
            try:
                df = pd.read_csv(f"./{file}.csv")
                file_name = os.path.basename(file).replace('.csv', '')
                df['file_source'] = file_name
                all_data.append(df)
                file_names.append(file_name)
                print(f"\n{file_name} 数据:")
                print(df.head())
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

        if not all_data:
            return

        # 合并数据
        combined_data = pd.concat(all_data, ignore_index=True)

        # 设置图形
        plt.figure(figsize=(14, 8))

        # 准备分组柱状图数据
        bar_width = 0.8 / len(result_file_names)  # 动态调整柱宽
        indices = combined_data['index'].unique() if 'index' in combined_data.columns else range(1, len(
            combined_data) // len(result_file_names) + 1)

        # 绘制 e2e 数据的柱状图
        for i, file_name in enumerate(result_file_names):
            file_data = combined_data[combined_data['file_source'] == file_name]
            x_pos = np.arange(len(indices)) + i * bar_width

            plt.bar(x_pos, file_data['e2e'], width=bar_width,
                    color=colors[i], alpha=0.7, label=f'{file_name}-e2e')

        plt.title('多个CSV文件 e2e 数据对比', fontsize=16, fontweight='bold')
        plt.xlabel('索引', fontsize=12)
        plt.ylabel('e2e 值', fontsize=12)
        plt.xticks(np.arange(len(indices)) + bar_width * (len(result_file_names) - 1) / 2, indices)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if len(result_file_names) == 1:
            plt.savefig(f'./{result_file_names[0]}.png',
                        dpi=200,
                        bbox_inches='tight')
            print(f"Result figure is locate in {result_file_names[0]}.png")
        else:
            today = date.today()
            plt.savefig(f'./{result_figure_prefix}_{today}.png',
                        dpi=200,
                        bbox_inches='tight')
            print(f"Result figure is locate in {result_figure_prefix}_{today}.png")

    except Exception as e:
        print(f"ERROR: {str(e)}")


benchmark_path = get_package_location("ais_bench_benchmark")
DATASET_CONF_DIR = os.path.join(benchmark_path,
                                "ais_bench/benchmark/configs/datasets")
REQUEST_CONF_DIR = os.path.join(benchmark_path,
                                "ais_bench/benchmark/configs/models/vllm_api")
DATASET_DIR = os.path.join(benchmark_path, "ais_bench/datasets")
CONSTS_DIR = os.path.join(benchmark_path, "ais_bench/benchmark/global_consts.py")



class AisbenchRunner:
    RESULT_MSG = {
        "performance": "Performance Result files locate in ",
        "accuracy": "write csv to ",
        "pressure": "Performance Result files locate in "
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
                f"{self.request_conf}_custom", '--datasets', f'{dataset_conf}_custom'
            ]
        if self.task_type == "performance":
            aisbench_cmd = [
                "taskset", "-c", "97-192", 'ais_bench', '--models',
                f"{self.request_conf}_custom", '--datasets', f'{dataset_conf}_custom', '--mode',
                'perf'
            ]
            if self.num_prompts:
                aisbench_cmd.extend(['--num-prompts', str(self.num_prompts)])
        if self.task_type == "pressure":
            aisbench_cmd = [
                "taskset", "-c", "97-192", 'ais_bench', '--models',
                f"{self.request_conf}_custom", '--datasets', f'{dataset_conf}_custom', '--mode',
                'perf', '--pressure'
            ]
        print(f"running aisbench cmd: {' '.join(aisbench_cmd)}")
        self.proc: subprocess.Popen = subprocess.Popen(aisbench_cmd,
                                                       stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE,
                                                       text=True)

    def __init__(self,
                 model: str,
                 port: int,
                 aisbench_config: dict,
                 card_num: int,
                 verify=True,
                 save=True):
        self.model = model
        self.port = port
        self.task_type = aisbench_config["case_type"]
        self.request_conf = aisbench_config["request_conf"]
        self.dataset_conf = aisbench_config.get("dataset_conf")
        self.dataset_path = aisbench_config.get("dataset_path")
        self.num_prompts = aisbench_config.get("num_prompts")
        self.max_out_len = aisbench_config.get("max_out_len", None)
        self.batch_size = aisbench_config["batch_size"]
        self.request_rate = aisbench_config.get("request_rate", 0)
        self.temperature = aisbench_config.get("temperature")
        self.top_k = aisbench_config.get("top_k")
        self.result_file_name = aisbench_config.get("result_file_name", "test")
        self.top_p = aisbench_config.get("top_p")
        self.seed = aisbench_config.get("seed")
        self.repetition_penalty = aisbench_config.get("repetition_penalty")
        self.pressure_time = aisbench_config.get("pressure_time", 0)
        self.exp_folder = None
        self.card_num = card_num
        self.result_line = None
        self._init_dataset_conf()
        self._init_request_conf()
        if self.task_type == "pressure":
            self._init_consts_conf()
        self._run_aisbench_task()
        self._wait_for_task()
        if verify:
            self.baseline = aisbench_config.get("baseline", 1)
            if self.task_type == "accuracy":
                self.threshold = aisbench_config.get("threshold", 1)
                self._accuracy_verify()
            if self.task_type == "performance" or self.task_type == "pressure":
                self.threshold = aisbench_config.get("threshold", 0.97)
                self._performance_verify()
        if save:
            self._performance_result_save()
            create_result_plot([self.result_file_name])

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
            merged_json["Request rate/Card"] = round(self.request_rate / self.card_num, 2)
            merged_json.update(self.result_json)
            merged_json.update(csv_result)
            merged_json["Total Token Throughput/Card"] = round(float(merged_json.get("Total Token Throughput").get("total").split(" ")[0]) / self.card_num, 4)
            merged_json["Request Throughput/Card"] = round(float(merged_json.get("Request Throughput").get("total").split(" ")[0]) / self.card_num, 4)
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
        conf_path = os.path.join(DATASET_CONF_DIR,
                                 f'{self.dataset_conf}.py')
        if self.dataset_conf.startswith("textvqa"):
            self.dataset_path = os.path.join(self.dataset_path,
                                             "textvqa_val.jsonl")
        with open(conf_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'path=.*', f'path="{self.dataset_path}",',
                         content)
        if self.max_out_len is None:
            if "max_tokens" not in content:
                content = re.sub(
                    r"output_column.*",
                    "output_column='answer',\n            max_tokens_column = 'max_tokens'",
                    content)

        conf_path_new = os.path.join(DATASET_CONF_DIR,
                                     f'{self.dataset_conf}_custom.py')
        with open(conf_path_new, 'w', encoding='utf-8') as f:
            f.write(content)




    def _init_consts_conf(self):
        with open(CONSTS_DIR, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'PRESSURE_TIME.*', f'PRESSURE_TIME = {self.pressure_time}',
                         content)
        with open(CONSTS_DIR, 'w', encoding='utf-8') as f:
            f.write(content)

    def _init_request_conf(self):
        conf_path = os.path.join(REQUEST_CONF_DIR, f'{self.request_conf}.py')
        with open(conf_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'model=.*', f'model="{self.model}",', content)
        content = re.sub(r'host_port.*', f'host_port = {self.port},', content)

        content = re.sub(r'batch_size.*', f'batch_size = {self.batch_size},',
                         content)
        content = re.sub(r'path=.*', f'path="{self.model}",', content)
        content = re.sub(r'request_rate.*',
                         f'request_rate = {self.request_rate},', content)

        if self.task_type == "performance" or self.task_type == "pressure":
            if "ignore_eos" not in content:
                content = re.sub(
                    r"temperature.*",
                    "temperature = 0,\n            ignore_eos = True,",
                    content)
        if self.task_type == "accuracy":
            if "ignore_eos" not in content:
                content = re.sub(
                    r"temperature.*",
                    "temperature = 0,\n            ignore_eos = False,",
                    content)

        if self.max_out_len is not None:
            content = re.sub(r'max_out_len.*',
                             f'max_out_len = {self.max_out_len},', content)
        if self.temperature is not None:
            content = re.sub(r"temperature.*",
                             f"temperature = {self.temperature},", content)
        if self.top_p is not None:
            content = re.sub(r"top_p.*", f"top_p = {self.top_p},", content)
        if self.top_k is not None:
            content = re.sub(r"top_k.*", f"top_k = {self.top_k},", content)
        if self.seed is not None:
            content = re.sub(r"seed.*", f"seed = {self.seed},", content)
        if self.repetition_penalty is not None:
            content = re.sub(
                r"repetition_penalty.*",
                f"repetition_penalty = {self.repetition_penalty},", content)
        conf_path_new = os.path.join(REQUEST_CONF_DIR,
                                     f'{self.request_conf}_custom.py')
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


def run_aisbench_cases(model, port, aisbench_cases, card_num=1, verify=True, save=True):
    aisbench_errors = []
    for aisbench_case in aisbench_cases:
        try:
            with AisbenchRunner(model, port, aisbench_case, verify=verify, save=save, card_num=card_num):
                pass
        except Exception as e:
            aisbench_errors.append([aisbench_case, e, traceback.print_exc()])
            print(e)
    for failed_case, error_info, error_traceback in aisbench_errors:
        print(
            f"The following aisbench case failed: {failed_case}, reason is {error_info}, traceback is: {error_traceback}."
        )
    assert not aisbench_errors, "some aisbench cases failed, info were shown above."
