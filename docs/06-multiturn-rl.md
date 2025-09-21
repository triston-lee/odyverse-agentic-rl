# 06 · 多轮对话（AIOps 事故分诊 · 无工具）

- **目标**：启用 VERL 的 **SGLang 多轮**，完成一个“两轮事故分诊”最小 PPO 实验：  

  \- 第 1 轮：只说明排查思路（不下结论）  

  \- 第 2 轮：只输出 JSON：`{"service": string, "incident_type": string}`

\> 多轮由 SGLang 引擎驱动，只需把 rollout 切到 `sglang` 并开启 `multi_turn` 即可。 [oai_citation:0‡verl.readthedocs.io](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)  

\> 切换后端的官方说法：启动脚本加 `actor_rollout_ref.rollout.name=sglang` 即可无缝从 vLLM 切换到 SGLang。 [oai_citation:1‡verl.readthedocs.io](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html?utm_source=chatgpt.com)

## 1）开启多轮（最小官方配置）

官方文档给出的最小配置是把 rollout 切到 **SGLang**，并启用 `multi_turn`： 

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn: true
```

> 上述配置会启用 **SGLang** 并以**多轮交互**执行 rollout。 

作为后端切换的通用说明：安装好 SGLang 后，只需在启动脚本加 actor_rollout_ref.rollout.name=sglang 即可从 vLLM 无缝切换到 SGLang（功能覆盖一致）。 

## 2）数据（Parquet · 与官方字段对齐）

我们只用两列：prompt 与 ground_truth。如果你的列名不同，可用 data.prompt_key 指定（见 Config Explanation）。 

- 生成数据：

```bash
python scripts/aiops_triage_build_parquet.py \
  --out-train $HOME/data/aiops_mturn/train.parquet \
  --out-val   $HOME/data/aiops_mturn/val.parquet
```

- 数据拼接脚本 scripts/aiops_triage_build_parquet.py

```python
import random, argparse, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

SERVICES = ["api", "db", "cache"]
INCIDENTS = ["latency_spike", "error_rate_spike", "oom"]

TPL = """你将用【两轮】完成事故分诊：
- 第1轮：只说明你的排查思路（不要直接给结论）。
- 第2轮：只输出 JSON：{"service": string, "incident_type": string}。
现象：{symptom}
日志摘录：{logline}
链路提示：{topo}
"""

def synth(n=600, seed=0):
    rng = random.Random(seed)
    cues = [
      ("api","latency_spike", "p99延迟升高", "timeout 错误增多", "上游网关无异常"),
      ("db","error_rate_spike", "5xx暴涨", "deadlock / lock wait", "慢查询占比上升"),
      ("cache","oom", "QPS平稳但命中率下降", "OOM killer 日志", "上游读写变多"),
      ("api","error_rate_spike", "4xx/5xx混合增长", "连接池耗尽", "下游无明显异常"),
    ]
    rows=[]
    for _ in range(n):
        svc, inc, s, l, t = rng.choice(cues)
        rows.append({"prompt": TPL.format(symptom=s, logline=l, topo=t),
                     "ground_truth": f"{svc}:{inc}"})
    return pd.DataFrame(rows)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    a=ap.parse_args()
    df = synth()
    pa.parquet.write_table(pa.Table.from_pandas(df), a.out_train)
    pa.parquet.write_table(pa.Table.from_pandas(df), a.out_val)
```

## 3）奖励函数（官方签名 · 自定义路径注入）

在 VERL 中，**最推荐**的自定义奖励做法是：把单样本评分函数放到独立文件，然后用

custom_reward_function.path/name 指过去，无需改源码。函数签名（官方）建议为：

fn(data_source, solution_str, ground_truth, extra_info=None) -> float。 

examples/reward_score/aiops_mturn.py

```python
import json, re

def _last_json(s: str):
    m=re.search(r"\{.*\}", s, re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except: return {}

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    js = _last_json(solution_str or "")
    sc = 0.0
    if isinstance(js, dict):
        svc = (js.get("service") or "").strip().lower()
        inc = (js.get("incident_type") or "").strip().lower()
        try:
            gt_svc, gt_inc = (ground_truth or "").split(":")
            if svc == gt_svc and inc == gt_inc:
                sc += 0.9  # 正确性主奖
        except: pass
        if set(js.keys()) == {"service","incident_type"} and svc and inc:
            sc += 0.1  # 结构化加分
    return max(0.0, min(1.0, sc))
```

## 4）训练命令（单卡最小可跑）

```bash
python -m src.trainer.main_ppo \
  data.train_files=$HOME/data/aiops_mturn/train.parquet \
  data.val_files=$HOME/data/aiops_mturn/val.parquet \
  data.prompt_key=prompt \
  custom_reward_function.path=examples/reward_score/aiops_mturn.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.multi_turn=true \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.total_epochs=2
```

## 5）进阶（了解即可）

- **交互插入/提醒**：如果你希望在两轮之间自动插入一条系统提示（例如“第二轮必须只输出 JSON”），可在 rollout 增加 interaction_config_file 指向一个交互 YAML；最小 Demo 不需要。 

- **安装提示**：SGLang 的安装与后端切换说明见官方 Backend 页面与 Install 页。 