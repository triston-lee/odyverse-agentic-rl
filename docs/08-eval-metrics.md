> 

```
---

## docs/08-eval-metrics.md

```markdown
# 08 · 评测与对齐（离线指标 · 源码化实现）

> 目标：用**与训练同源的奖励函数**对验证集做**离线评测**，并输出 A/B 报告（无工具 vs 可选工具、多轮 vs 单轮、不同 KL/超参配置）。本页给出完整脚本与指标设计。

---

## 0）为什么一定要“离线评测”？

- 训练中的 reward 是“**on-policy 估计**”，受当前采样/温度等影响；  
- 离线评测用**固定模型（或不同 checkpoint）** + **固定验证集**，可复现、可对比；  
- 与训练同源的 `compute_score(...)` 能确保**目标一致**，避免“训练目标/评测目标不一致”。

---

## 1）输出什么？（指标清单）

**主指标**
- `score_mean` / `score_std`（奖励平均/方差）
- `acc_main`：主任务正确率（例如 AIOps：`service:incident` 全命中率）
- `json_ok`：JSON 合规率（能被解析 + 字段齐全）

**多轮/工具特有**
- `avg_turns`：平均轮数（如果你记录了）
- `evidence_rate`：含证据比例（07 可选工具）
- `kbid_rate`：含 `KBID:xxxx` 真凭证比例（防“装用工具”）
- `tool_calls_mean`：平均工具调用次数（若能获取）

**效率与稳定**
- `len_response_mean`：平均输出长度（token）
- `kl_mean`（若可得）、`sps`（训练阶段）

---

## 2）准备评测数据与模型输出

两种来源：
1) **推理脚本**对 `val.parquet` 逐条生成 `response`；或  
2) 训练期间**周期验证**的导出文件（推荐你在 Trainer 的 `test_freq` 钩子里把 `(prompt, response)` 写到 JSONL）。

我们使用统一格式 `pred.jsonl`：
```json
{"prompt": "...", "response": "...", "ground_truth": "...?", "meta": {"tool_calls": 1}}
```



------





## **3）评测脚本（完整可用）**





scripts/eval_offline.py

```
import argparse, json, pandas as pd, numpy as np
import pyarrow.parquet as pq
from importlib import import_module
import re, json as pyjson
from collections import Counter
from pathlib import Path

KB_TAG = re.compile(r"KBID:[0-9a-f]{8}")

def load_reward(fn_spec: str):
    # e.g. "examples.reward_score.aiops_mturn:compute_score"
    mod, fn = fn_spec.split(":")
    return getattr(import_module(mod), fn)

def try_json(text: str):
    if not isinstance(text, str): return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try: return pyjson.loads(m.group(0))
    except: return None

def main(val_parquet, pred_jsonl, reward_fn, out_csv):
    # 1) 读取验证集与预测
    val = pq.read_table(val_parquet).to_pandas()
    preds = [json.loads(l) for l in open(pred_jsonl, "r", encoding="utf-8")]
    dfp  = pd.DataFrame(preds)
    # 2) 合并（用 prompt 对齐；生产建议用 id）
    df = val.merge(dfp, on="prompt", how="inner", suffixes=("",""))
    # 3) 评分（与训练同源）
    score_fn = load_reward(reward_fn)
    samples = df[["prompt","response","ground_truth"]].to_dict("records")
    scores  = [score_fn(None, s["response"], s.get("ground_truth"), None) for s in samples]
    df["score"] = scores
    # 4) 任务级指标（AIOps 示例）
    svc_ok, inc_ok, both_ok, json_ok, kbid = [], [], [], [], []
    svc_pred, inc_pred, svc_gt, inc_gt = [], [], [], []
    for _, r in df.iterrows():
        js = try_json(r.get("response","")) or {}
        svc = (js.get("service") or "").strip().lower() if isinstance(js, dict) else ""
        inc = (js.get("incident_type") or "").strip().lower() if isinstance(js, dict) else ""
        gt  = (r.get("ground_truth") or "")
        ok_json = isinstance(js, dict) and (("service" in js) and ("incident_type" in js))
        json_ok.append(1 if ok_json and svc and inc else 0)
        try:
            gsvc, ginc = gt.split(":")
            ok_both = (svc == gsvc and inc == ginc)
            both_ok.append(1 if ok_both else 0)
            svc_ok.append(1 if svc == gsvc else 0)
            inc_ok.append(1 if inc == ginc else 0)
            svc_pred.append(svc); inc_pred.append(inc); svc_gt.append(gsvc); inc_gt.append(ginc)
        except:
            both_ok.append(0); svc_ok.append(0); inc_ok.append(0)
        kbid.append(1 if KB_TAG.search((js.get("evidence") or "")) else 0)
    # 5) 汇总
    def mean(x): return float(np.mean(x)) if len(x) else 0.0
    report = {
        "n": len(df),
        "score_mean": mean(df["score"]),
        "score_std": float(np.std(df["score"])),
        "acc_main": mean(both_ok),
        "acc_service": mean(svc_ok),
        "acc_incident": mean(inc_ok),
        "json_ok": mean(json_ok),
        "kbid_rate": mean(kbid),
        "len_response_mean": float(np.mean([len((r or "")) for r in df["response"]])),
    }
    # 6) 混淆（可选）：服务/事故类型
    c_service = pd.crosstab(pd.Series(svc_gt, name="gt_service"),
                            pd.Series(svc_pred, name="pred_service"))
    c_inc     = pd.crosstab(pd.Series(inc_gt, name="gt_incident"),
                            pd.Series(inc_pred, name="pred_incident"))
    # 7) 输出
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("==== Report ====")
    for k,v in report.items(): print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("Confusion(Service):\n", c_service)
    print("Confusion(Incident):\n", c_inc)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--reward-fn", required=True, help="module:path.to.fn e.g. examples.reward_score.aiops_mturn:compute_score")
    ap.add_argument("--out", default="reports/eval.csv")
    a=ap.parse_args()
    main(a.val, a.pred, a.reward_fn, a.out)
```

**使用示例**

```
# 无工具（06）的评测
python scripts/eval_offline.py \
  --val  $HOME/data/aiops_mturn/val.parquet \
  --pred outputs/06_multiturn_pred.jsonl \
  --reward-fn examples.reward_score.aiops_mturn:compute_score \
  --out  reports/06_eval.csv

# 可选工具（07）的评测（你也可以沿用同一奖励）
python scripts/eval_offline.py \
  --val  $HOME/data/aiops_tool/val.parquet \
  --pred outputs/07_tool_pred.jsonl \
  --reward-fn examples.reward_score.aiops_tool_reward:compute_score \
  --out  reports/07_eval.csv
```

> 小技巧：把不同 checkpoint 的 pred.jsonl 放进同一目录，文件名带上 global_step，评测脚本跑完即可画“训练进程 → 指标曲线”。



------





## **4）怎么得到** 

## **pred.jsonl**

## **？**







### **方案 A：训练过程的** 

### **test_freq**

###  **钩子里导出**





- 最稳：与你的 rollout/模板完全一致。
- 做法：在验证阶段把 (prompt, response, ground_truth, tool_calls) 写入 outputs/${exp}/step_${k}.jsonl。







### **方案 B：独立推理脚本（快速但要注意模板一致）**





最小基线（transformers 直出，**仅用于离线评测**）：

```
# scripts/predict_hf.py （简化示例）
import argparse, json, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re

def main(model, val_parquet, out_jsonl, max_new_tokens=128, temperature=0.2):
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).eval().cuda()
    val = pq.read_table(val_parquet).to_pandas()
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in val.iterrows():
            prompt = row["prompt"]
            x = tok.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True, return_tensors="pt").cuda()
            y = m.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)
            text = tok.decode(y[0][x.shape[-1]:], skip_special_tokens=True)
            f.write(json.dumps({"prompt": prompt, "response": text, "ground_truth": row.get("ground_truth","")}, ensure_ascii=False)+"\n")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", required=True)  # 训练后合并到 HF 格式的目录
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args(); main(a.model, a.val, a.out)
```

> 注意：要用**与训练一致**的 chat template（apply_chat_template）；否则评测有偏差。



------





## **5）A/B 对比与显著性（Bootstrap）**





scripts/ab_compare.py

```
import argparse, pandas as pd, numpy as np
def load(path): return pd.read_csv(path)["score"].to_numpy()
def bootstrap_diff(a, b, n=10000, seed=0):
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(n):
        sa=rng.choice(a, size=a.size, replace=True)
        sb=rng.choice(b, size=b.size, replace=True)
        diffs.append(sa.mean()-sb.mean())
    diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5])
    return diffs.mean(), (lo,hi)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    a=ap.parse_args(); A=load(a.a); B=load(a.b)
    dm,(lo,hi)=bootstrap_diff(A,B)
    print(f"Δscore_mean={dm:.4f}, 95% CI=({lo:.4f},{hi:.4f})")
```



------





## **6）图表建议（你可以用 Notebook/Matplotlib 补上）**





- score 直方图（06 vs 07）
- json_ok/kbid_rate 的条形图
- KL vs score 的散点（如从训练日志导出）
- checkpoint 维度的 score_mean 折线图（观察收敛）





------





## **7）把评测接入 CI（可选）**





- 训练结束触发 predict_hf.py → 生成 pred.jsonl；
- 跑 eval_offline.py → 得 eval.csv；
- 设置阈值：acc_main ≥ X 且 json_ok ≥ Y，不达标则失败；
- 归档：把 eval.csv、图表、若干失败样例打包到 artifacts。





------





## **8）错误分析（脚本化）**





- 抽样 top-N 低分样本，打印 prompt/response/ground_truth；
- 统计**常见混淆**（比如 api latency_spike vs db error_rate_spike 的混淆）；
- 如是工具场景，统计“未带 KBID 的 evidence 占比”。





------





## **9）小结：让训练与评测真正“对齐”**





- **同一套 compute_score**：训练与评测公用，避免“目标错位”。
- **固定验证集** + **固定模板**：确保可比性。
- **A/B 明确**：无工具 ↔ 可选工具；不同 KL；不同 batch/长度。
- **把统计和图表脚本存进仓库**：一键复跑，写报告就拷贝图表。



```
---

要继续的话，我下一轮可以补 **09「多机扩展（Ray / SkyPilot）」** 和 **10「吞吐与显存优化」**。这两篇会结合“进程/Actor 架构、参数如何影响 SPS/显存峰值、典型 OOM 处方”，依然给出可复制的 YAML/脚本与源码定位点。
```

