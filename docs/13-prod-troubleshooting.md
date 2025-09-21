好，继续更两篇：**13「数据工程与合成」** + **14「复现实验报告模板」**。两篇都尽量“贴源码/可执行”，关键点我查过 VERL/SGLang/vLLM 官方资料并标注引用。



------





## **docs/13-data-engineering-and-synthesis.md**



```
# 13 · 数据工程与合成（偏好对 / 负样本 / 规则校验 · 贴源码）

> 主题：把我们在 04/06/07/11 里用到的数据“做厚做稳”。目标是：
> 1) 合成**可控**的 AIOps 训练/验证集（两轮分诊、可选工具）；
> 2) 产出 **偏好对**（DPO/对比学习用）与 **负样本**（稳健性）；
> 3) 提供**质量闸门**（schema/泄漏/冲突检查）；
> 4) 让数据与 VERL 的 **reward_fn**、**multi_turn** 配置完全对齐（`data.prompt_key`、`custom_reward_function.*`）。 [oai_citation:0‡Verl](https://verl.readthedocs.io/en/latest/examples/config.html?utm_source=chatgpt.com)

---

## 0）与 VERL 的接口对齐

- 训练/验证数据建议用 **parquet**；最少列：`prompt`、`ground_truth`。若列名不同，用 `data.prompt_key` 覆盖。 [oai_citation:1‡Verl](https://verl.readthedocs.io/en/latest/examples/config.html?utm_source=chatgpt.com)  
- 自定义奖励函数通过 `custom_reward_function.path/name` 注入，不改框架源码。函数签名：  
  `fn(data_source, solution_str, ground_truth, extra_info=None) -> float`。 [oai_citation:2‡Verl](https://verl.readthedocs.io/en/latest/preparation/reward_function.html?utm_source=chatgpt.com)  
- 多轮 rollout（SGLang）只需 `actor_rollout_ref.rollout.name=sglang`、`multi_turn=true`；第二轮上下文由后端自动拼接。 [oai_citation:3‡Verl](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html?utm_source=chatgpt.com)

---

## 1）合成主任务数据（两轮分诊）

**脚本：`scripts/synth_aiops_pairs.py`**（训练/验证两用，支持随机种子）
```python
import random, argparse, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

CUES = [
  ("api","latency_spike","p99延迟升高","timeout 错误增多","上游网关无异常"),
  ("db","error_rate_spike","5xx暴涨","deadlock / lock wait","慢查询占比上升"),
  ("cache","oom","命中率下降","OOM killer 日志","上游读写变多"),
]

PROMPT_TPL = """你将用【两轮】完成事故分诊：
- 第1轮：只说排查思路（不下结论）。
- 第2轮：只输出 JSON：{"service": string, "incident_type": string}.
现象：{s}
日志摘录：{l}
链路：{t}
"""

def build(n=1200, seed=0):
    rng=random.Random(seed); rows=[]
    for _ in range(n):
        svc,inc,s,l,t = rng.choice(CUES)
        rows.append({"prompt": PROMPT_TPL.format(s=s,l=l,t=t),
                     "ground_truth": f"{svc}:{inc}"})
    return pd.DataFrame(rows)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a=ap.parse_args()
    pq.write_table(pa.Table.from_pandas(build(a.n_train,a.seed)), a.out_train)
    pq.write_table(pa.Table.from_pandas(build(a.n_val,a.seed+1)), a.out_val)
```



------





## **2）合成**

## **偏好对**

## **（DPO/对比学习）**





**脚本：scripts/synth_aiops_prefs.py**（一条 prompt → chosen/rejected）

```
import json, random, argparse, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from synth_aiops_pairs import CUES, PROMPT_TPL

NEG_POOL=[("api","error_rate_spike"),("db","latency_spike"),("cache","latency_spike")]

def build(n=1000, seed=0):
    rng=random.Random(seed); rows=[]
    for _ in range(n):
        svc,inc,s,l,t = rng.choice(CUES)
        prompt = PROMPT_TPL.format(s=s,l=l,t=t)
        chosen = json.dumps({"service":svc,"incident_type":inc}, ensure_ascii=False)
        wsvc,winc = rng.choice([p for p in NEG_POOL if p!=(svc,inc)])
        rejected = json.dumps({"service":wsvc,"incident_type":winc}, ensure_ascii=False)
        rows.append({"prompt":prompt,"chosen":chosen,"rejected":rejected})
    return pd.DataFrame(rows)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=1000)
    a=ap.parse_args(); pq.write_table(pa.Table.from_pandas(build(a.n)), a.out)
```

> 这些偏好对可直接喂给 **DPO** 或作为 **GRPO** 的离线 warmup；DPO 在 VERL 里可按“扩展到 DPO”的教程接入，或先用 TRL 的 DPOTrainer 跑，再回到 VERL 做 PPO/GRPO。 



------





## **3）**

## **负样本**

## **与抗干扰（鲁棒性）**





- **字段缺失**：随机把 日志摘录/链路 置空；
- **冲突提示**：让“现象”和“日志”暗示不同组件（模型要学会权衡）；
- **诱导噪声**：加入“过时经验/谣言段落”，奖励端只看最终 JSON 正确性。
- **标注错配**：少量故意错配标签，用评测脚本观测模型对噪声的敏感度（期望：分数略降，但整体稳）。





**脚本：scripts/perturb_aiops.py**（把原 parquet 加扰写出）

```
import argparse, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, random, re
NOISE = "\n【过时经验】缓存命中率低通常说明数据库死锁，请直接重启。"
def perturb_row(r, rng):
    p=r["prompt"]; gt=r["ground_truth"]
    # 10% 缺字段
    if rng.random()<0.1: p=re.sub(r"日志摘录：.*\n","日志摘录：\n",p)
    # 10% 冲突线索
    if rng.random()<0.1: p=p+"\n额外日志：连接池耗尽"
    # 10% 诱导噪声
    if rng.random()<0.1: p=p+NOISE
    return {"prompt":p,"ground_truth":gt}
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True); ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a=ap.parse_args(); rng=random.Random(a.seed)
    df=pa.read_table(a.in_parquet).to_pandas()
    out=[perturb_row(r, rng) for _,r in df.iterrows()]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(out)), a.out_parquet)
```



------





## **4）质量闸门（Schema / 泄漏 / 冲突）**





**脚本：scripts/data_quality_gate.py**

```
import argparse, json, re, pandas as pd, pyarrow.parquet as pq
from jsonschema import validate

JSON_PATTERN=re.compile(r"\{.*\}", re.S)
SCHEMA={"type":"object",
  "properties":{"service":{"type":"string"}, "incident_type":{"type":"string"}},
  "required":["service","incident_type"]}

def leak_check(prompt, gt):
    # “标签泄漏”粗查：ground_truth 的词直接出现在 prompt（真实生产应更精细）
    svc, inc = gt.split(":")
    return (svc in prompt) or (inc in prompt)

def main(parquet):
    df=pq.read_table(parquet).to_pandas()
    errs=[]
    for i,row in df.iterrows():
        p,gt=row["prompt"], row["ground_truth"]
        # 1) 基础校验
        if not isinstance(p,str) or ":" not in gt:
            errs.append((i,"schema","bad types"))
        # 2) 泄漏
        if leak_check(p, gt):
            errs.append((i,"leak","gt tokens in prompt"))
        # 3) 冲突（示例规则：现象/日志互相指向不同组件）
        if "p99延迟升高" in p and "deadlock" in p:
            errs.append((i,"conflict","latency + deadlock combo"))
    print("total:", len(df), "errors:", len(errs))
    for e in errs[:50]: print(e)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--parquet", required=True)
    a=ap.parse_args(); main(a.parquet)
```

> 质量闸门让训练/评测前就能**阻断脏数据**，避免 reward 函数“背锅”。（自定义奖励与配置的接口见官方文档。） 



------





## **5）与 08 评测打通（eg. “证据率/合规率”）**





第 08 篇给了 scripts/eval_offline.py，它按**与训练同源的 reward_fn**打分；你可在评测表里追加“扰动类型 → 成绩”的切片统计，看模型对噪声的敏感度。



------





## **6）小结**





- 数据是 RL 成败的第一因子：**分布稳定、标签可信、评测同源**。
- 在 VERL：只要对齐 data.prompt_key + custom_reward_function.* +（多轮）rollout.name=sglang,multi_turn=true，其余就能按教程推进。 

