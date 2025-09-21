# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
将 GSM8k 数据集（小学数学推理问答数据集）进行预处理，并保存为 Parquet 格式，同时可选择上传到 HDFS
"""

import argparse
import os
import re

import datasets

from myverl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    """
    - GSM8k 的答案字符串格式中，最终答案以 "#### <数字>" 的形式出现。
    - 这个函数用正则匹配到 #### 后面的数字，并去掉千位分隔符 ,。
    - 例如：
        "The result is 25 #### 25" → 提取出 "25"
    """
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # 加载数据集
    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # 预处理逻辑
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'


    def make_map_fn(split):
        def process_fn(example, idx):
            """
            把原始数据转换为统一格式：
                • question：在原始问题后追加提示 "Let's think step by step..."
                • answer：提取 #### 后的最终答案
                • 最终数据格式：
                    {
                      "data_source": "openai/gsm8k",
                      "prompt": [{"role": "user", "content": "原始问题 + Let's think step by step..."}],
                      "ability": "math",
                      "reward_model": {"style": "rule", "ground_truth": "正确答案"},
                      "extra_info": {
                        "split": "train/test",
                        "index": 样本索引,
                        "answer": 原始答案文本,
                        "question": 原始问题
                      }
                    }
            """
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn


    # 使用 Hugging Face 的 .map() 方法，对每个样本调用 process_fn 进行格式转换
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    # 保存为 Parquet 格式
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    # 可选上传到 HDFS：如果提供了 --hdfs_dir，会在 HDFS 中创建目录并复制保存的数据
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
