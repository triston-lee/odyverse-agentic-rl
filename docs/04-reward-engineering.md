# 04 · 奖励工程（函数式奖励）

目标：按 **VERL 官方接口**实现与接入奖励函数，既能直接复用**内置数据集的规则奖励**，也能通过**`custom_reward_function`**挂载你自己的函数。

## 0）你需要知道的官方事实

\- VERL 的 **RewardManager** 在 PPO 训练入口（`main_ppo.py`）里调用，用来组织“按数据源选择奖励函数 → 计算每条响应的分数”。输入是一个 `DataProto`，其中包含 **chat_template 之后的 `input_ids/attention_mask`**、**生成的 `responses` token**，以及从 parquet 预处理进来的 **`ground_truth`** 与 **`data_source`** 等字段（位于 `non_tensor_batch`） [oai_citation:0‡Verl]( https://verl.readthedocs.io/en/latest/preparation/reward_function.html )。 

\- 官方已经**预实现**了 GSM8K、MATH 等数据集的奖励：例如 **GSM8K** 强制输出 “四个 `####` 后给最终答案”，**完全匹配给 1 分；只要格式正确给 0.1 分**（否则 0 分） [oai_citation:1‡Verl]( https://verl.readthedocs.io/en/latest/preparation/reward_function.html )。 

\- 如果你要自定义奖励函数，**无需改 VERL 源码**，直接在配置里指定： 

-  `custom_reward_function.path`（文件路径）
-  `custom_reward_function.name`（函数名，默认 `compute_score`）
- 函数签名推荐为 

 `fn(data_source, solution_str, ground_truth, extra_info=None) -> float`（单样本打分） [oai_citation:2‡Verl]( https://verl.readthedocs.io/en/latest/preparation/reward_function.html )。 

\- `data.prompt_key` 默认是 `prompt`，训练集/验证集需要是 **parquet**；其余常见数据字段与开关见「Config Explanation」页面（例如 `max_prompt_length`/`max_response_length` 等） [oai_citation:3‡Verl]( https://verl.readthedocs.io/en/latest/examples/config.html )。



## 1）两种接法一览

**A. 直接用内置奖励**

适用于：GSM8K/MATH 等官方已支持的数据源。 

要求：在你的 parquet 中预处理好 **`ground_truth`** 与 **`data_source`**（例如 `gsm8k`），RewardManager 会自动选到对应的内置规则函数来评分 [oai_citation:4‡Verl]( https://verl.readthedocs.io/en/latest/preparation/reward_function.html )。

**B. 自定义奖励函数**

适用于：你自己的任务。 

做法：写一个 **单样本打分函数**，在配置里通过 `custom_reward_function.*` 指向它即可（不需要修改 VERL 注册表） [oai_citation:5‡Verl]( https://verl.readthedocs.io/en/latest/preparation/reward_function.html )。

## 2）数据准备（与官方字段对齐）



最小字段建议（parquet）：

\- `prompt`（或用 `data.prompt_key` 指向其他列名） [oai_citation:6‡Verl]( https://verl.readthedocs.io/en/latest/examples/config.html ) 

\- `ground_truth`（该样本的标准答案/参考） [oai_citation:7‡Verl]( https://verl.readthedocs.io/en/latest/preparation/reward_function.html ) 

\- `data_source`（数据集名称；用内置奖励时用于路由） [oai_citation:8‡Verl]( https://verl.readthedocs.io/en/latest/preparation/reward_function.html )

\> 备注：如果你只做自定义奖励，不强制需要 `data_source`，但保留该列有助于在一个文件里混合多数据源与多种奖励逻辑。

## 3）自定义奖励：最小实现（JSON 结构化 + 正确性）

在你的项目里建 [my_reward.py](..%2Fverl%2Fexamples%2Freward_score%2Fmy_reward.py):

```python
# examples/reward_score/my_reward.py
import json, re

def _safe_last_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: 
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    官方建议的签名：返回单样本分数 in [0,1]
    规则（示例）：
      +0.8 正确性：若最终 JSON 中 `city` 等于 ground_truth
      +0.2 结构化：只包含一个字段 {"city": <非空字符串>}
    """
    js = _safe_last_json(solution_str or "")
    sc = 0.0
    city = (js.get("city") or "").strip() if isinstance(js, dict) else ""
    gt = (ground_truth or "").strip()
    if city and gt and city.lower() == gt.lower():
        sc += 0.8
    if isinstance(js, dict) and set(js.keys()) == {"city"} and isinstance(city, str) and city:
        sc += 0.2
    return max(0.0, min(1.0, sc))
```

- 单样本函数签名、通过配置注入，是官方推荐做法；如果你只测这一个函数，函数名直接叫 compute_score，可省略 custom_reward_function.name。

- 生产建议：保持确定性、边界安全（JSON 解析异常要兜底）、分值尽量落在 [0,1]，便于与 KL 控制配合（是否把 KL 加在 reward 里取决于 algorithm.use_kl_in_reward，避免与 actor 端的 use_kl_loss 双重惩罚）。

## 4）配置与启动命令

只需在命令里追加两行 custom_reward_function.* 即可（其他参数沿用你的 PPO 命令）：
```bash
python -m src.trainer.main_ppo \
  data.train_files=$HOME/data/your/train.parquet \
  data.val_files=$HOME/data/your/val.parquet \
  data.prompt_key=prompt \
  # —— 指向你刚写的奖励函数 —— #
  custom_reward_function.path=examples/reward_score/my_reward.py \
  custom_reward_function.name=compute_score \
  # —— 常见 PPO / rollout 参数（示例） —— #
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=False \
  trainer.total_epochs=2
```

- custom_reward_function.path 指向文件路径，name 可省略（函数名默认为 compute_score） 。

- 想跑内置 GSM8K：把 parquet 的 data_source 设成 gsm8k，并按官方约定生成“####”格式的最终答案，RewardManager 会用内置规则打分（正确=1，格式=0.1） 。

## 5）奖励模型 / 其它任务

- 对于 RLHF 数据集（如 full_hh_rlhf）和代码生成任务（如 APPS），官方建议分别用 Reward Model 与 SandBox 评测路线（后者即将开源），不走简单的规则分法 。

- 如果你要用基于模型的 RM，配置里有 reward_model.* 与 reward_model.reward_manager（naive 为默认，若你的校验函数支持多进程，可切到 prime 以并行评分）。