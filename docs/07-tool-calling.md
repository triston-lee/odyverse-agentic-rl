# 07 · Tool 调用（AIOps 事故分诊 · 离线 KB）

目标：延续 06 的任务，但第 1 轮引导模型调用**本地知识库检索工具** `kb_search(query)` 获取“证据”，第 2 轮输出 JSON 结论。完全离线可复现。

## 1）知识库（本地 JSON）

```bash
mkdir -p data/kb && cat > data/kb/aiops_kb.json <<'JSON'
[
  {"pattern": "p99延迟|timeout", "service": "api", "incident_type": "latency_spike",
   "snippet": "API 延迟升高/超时，先看网关与下游，常见为 API 线程池耗尽或限流未配。"},
  {"pattern": "deadlock|lock wait", "service": "db", "incident_type": "error_rate_spike",
   "snippet": "数据库死锁/锁等待导致 5xx 暴涨，关注慢查询与事务粒度。"},
  {"pattern": "命中率下降|OOM", "service": "cache", "incident_type": "oom",
   "snippet": "缓存命中率下降伴随 OOM，检查驱逐策略与大 key。"}
]
JSON
```



## 2）数据（难度稍大，鼓励用工具）



```bash
python scripts/aiops_tool_build_parquet.py \
  --out-train $HOME/data/aiops_tool/train.parquet \
  --out-val   $HOME/data/aiops_tool/val.parquet
```

scripts/aiops_tool_build_parquet.py

```python
import random, argparse, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

TPL = """你将用【两轮】完成事故分诊：
- 第1轮：先说你的排查思路；如果需要，你可以调用 kb_search(query) 获取一段知识库证据。
- 第2轮：只输出 JSON：{"service": string, "incident_type": string, "evidence": string?}。
现象：{symptom}
日志摘录：{logline}
链路提示：{topo}
"""

def synth(n=800, seed=1):
    rng = random.Random(seed)
    cues = [
      ("api","latency_spike", "p99延迟升高", "timeout 错误增多", "上游网关无异常"),
      ("db","error_rate_spike", "5xx暴涨", "deadlock / lock wait", "慢查询占比上升"),
      ("cache","oom", "QPS平稳但命中率下降", "OOM killer 日志", "上游读写变多"),
    ]
    rows=[]
    for _ in range(n):
        svc, inc, s, l, t = rng.choice(cues)
        prompt = TPL.format(symptom=s, logline=l, topo=t)
        rows.append({"prompt": prompt, "ground_truth": f"{svc}:{inc}"})
    return pd.DataFrame(rows)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    a=ap.parse_args()
    df = synth(1200)
    pa.parquet.write_table(pa.Table.from_pandas(df), a.out_train)
    pa.parquet.write_table(pa.Table.from_pandas(df), a.out_val)
```



## 3）KB 工具（类 + YAML）

**工具类**（my_tools/kb_tool.py）

```python
# my_tools/kb_tool.py
import json, re, pathlib, hashlib, time
from myverl.tools.base_tool import BaseTool, ToolResponse

KB_PATH = pathlib.Path("data/kb/aiops_kb.json")
KB = json.loads(KB_PATH.read_text(encoding="utf-8")) if KB_PATH.exists() else []


def _pick_snippet(query: str):
    for item in KB:
        if re.search(item["pattern"], query, re.I):
            return item
    return KB[0] if KB else None


def _kb_ticket(snippet: str) -> str:
    # 生成一次性的证据标签，奖励只认带标签的证据
    h = hashlib.sha1(f"{snippet}|{int(time.time()) // 3600}".encode()).hexdigest()[:8]
    return f"KBID:{h}"


class KBTool(BaseTool):
    name = "kb_search"
    description = "Return a relevant AIOps KB snippet for given query text."
    tool_schema = {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": "Retrieve a short snippet from local KB",
            "parameters": {
                "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]
            }
        }
    }

    async def create(self, *, instance_id: str, **kwargs):
        q = (kwargs.get("query") or "").strip()
        ans = _pick_snippet(q)
        if not ans:
            text = "[KB] (empty)"
        else:
            tag = _kb_ticket(ans["snippet"])
            text = f"[KB] {tag} service={ans['service']} incident={ans['incident_type']} :: {ans['snippet']}"
        return instance_id, ToolResponse(text=text)
```

> 解释：返回里包含 KBID:xxxx，奖励函数只认这个**动态标签**，模型自己编证据不会加分。

**工具 YAML**（tools/kb.yaml）

```yaml
tools:
  - class_name: my_tools.kb_tool.KBTool
    config:
      type: native
```

> 你当前的 VERL 版本如果要求在 rollout 配置里用 actor_rollout_ref.rollout.tool_kwargs.tools_config_file=tools/kb.yaml 加载，即可生效。

## 4）奖励（以正确性为主，证据加分，不使用工具也不扣分）

examples/reward_score/aiops_tool_reward.py

```python
import json, re

KB_TAG = re.compile(r"KBID:[0-9a-f]{8}")

def _last_json(s: str):
    m=re.search(r"\{.*\}", s, re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except: return {}

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    js = _last_json(solution_str or ""); sc = 0.0
    svc = (js.get("service") or "").strip().lower() if isinstance(js, dict) else ""
    inc = (js.get("incident_type") or "").strip().lower() if isinstance(js, dict) else ""
    ev  = (js.get("evidence") or "").strip() if isinstance(js, dict) else ""

    # 1) 正确性（主奖）
    try:
        gt_svc, gt_inc = (ground_truth or "").split(":")
        if svc == gt_svc and inc == gt_inc:
            sc += 0.85
    except: pass

    # 2) 结构化（键齐全即可；不强求 evidence 必填）
    if isinstance(js, dict) and "service" in js and "incident_type" in js:
        sc += 0.10

    # 3) 证据加分（可选）。只认带 KBID 标签的证据，防“假装用工具”
    if ev and KB_TAG.search(ev):
        sc += 0.05

    # （可选）工具成本：如果你能拿到调用次数（extra_info），这里可做轻微扣分
    # if extra_info and extra_info.get("tool_calls", 0) > 1: sc -= 0.02 * (extra_info["tool_calls"] - 1)

    return max(0.0, min(1.0, sc))
```

> 设计哲学：
>
> - 不使用工具**：只要答案对 + 格式对，最高拿 **0.95**；**
>
> - **使用工具且带真凭证**：再加 **0.05**（总分不超过 1.0）；
>
> - （可选）你可以引入**成本项**，如多次工具调用轻微减分，防过度依赖。

## 5）训练命令（多轮 + 工具）

```bash
python -m src.trainer.main_ppo \
  data.train_files=$HOME/data/aiops_tool/train.parquet \
  data.val_files=$HOME/data/aiops_tool/val.parquet \
  data.prompt_key=prompt \
  custom_reward_function.path=examples/reward_score/aiops_tool_reward.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.multi_turn=true \
  actor_rollout_ref.rollout.tool_kwargs.tools_config_file=tools/kb.yaml \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  trainer.total_epochs=2
```
