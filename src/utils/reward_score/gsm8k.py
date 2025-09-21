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

import re
# 为了优化正则匹配的速度，只在答案字符串的最后 300 个字符内搜索
# （因为数学题的最终答案几乎总在最后部分出现）
_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str: str, method="strict") -> str | None:
    """
    从模型生成的答案字符串中提取最终数值答案

    :param solution_str: 模型生成的答案字符串
    :param method: 提取答案的方法， choices are 'strict' and 'flexible'

    :return: 提取到的答案（字符串），提取不到则为 None
    """
    assert method in ["strict", "flexible"]

    # 如果字符串太长，只截取最后 300 个字符来匹配
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # 严格模式：要求答案前必须有 "#### " 格式
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # 取最后一个匹配结果，并去掉千位逗号和美元符号
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        # 宽松模式：只要是数字（带小数点、逗号、负号）都提取
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # 如果没有找到数字，返回 None（无 reward）
            pass
        else:
            invalid_str = ["", "."]
            # 从后往前找，取最后一个不是 "." 的数字
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(solution_str: str, ground_truth: str, method="strict", format_score=0.0, score=1.0) -> float:
    """GSM8k 数据集的评分函数，用于判断模型答案是否正确

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    :param solution_str: 型生成的答案字符串
    :param ground_truth: 标准答案
    :param method: 提取答案的方法， choices are 'strict' and 'flexible'
    :param format_score: 格式化分数（如果提取到了答案但数值不对，给此分数）
    :param score: 正确答案的满分（默认 1.0）

    :return: 评分, 0 表示答案不正确, score 表示答案正确, format_score 表示答案格式正确但数值不正确
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0    # 没有答案，直接 0 分
    else:
        if answer == ground_truth:
            return score    # 答案完全正确，满分
        else:
            return format_score     # 提取到了答案，但和标准答案不同，给格式分
