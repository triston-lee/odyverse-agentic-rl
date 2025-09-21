

```
---

## docs/14-repro-report-template.md

```markdown
# 14 · 复现实验报告模板（SPS/显存/指标图表 · 一键脚本）

> 目标：把“从训练到评测”的关键事实**结构化记录**下来，便于复现、汇报与 A/B 对比。我们提供：
> 1) 报告模板（可直接拷仓库）；
> 2) 指标聚合脚本（含：score/json_ok/kbid_rate/混淆矩阵）；
> 3) 吞吐/显存的抓取与图表；
> 4) 常见坑（VERL 的生成上限 / 保存 / 合并 等问题）与出处。

---

## 0）报告模板（Markdown）

`reports/template.md`
```markdown
# 实验报告：AIOps 两轮分诊（VERL）

## 1. 基本信息
- 任务：两轮分诊（第二轮 JSON）
- 框架：VERL {commit/hash}；后端：{sglang|vllm}
- 模型：{Qwen2.5-0.5B-Instruct 等}
- 数据：train/val 路径、行数、质量闸门通过率
- 奖励：{路径与函数名}（与训练/评测同源） [oai_citation:7‡Verl](https://verl.readthedocs.io/en/latest/preparation/reward_function.html?utm_source=chatgpt.com)
- 配置片段：data/model/rollout/actor/critic/algorithm/trainer 关键键（详见 02 篇）

## 2. 训练过程
- 日志截图：reward/KL/loss/SPS（TB 或 CSV）
- 超参：lr/kl_coef/micro_batch/mini_batch/长度限制等
- 事件：断点、异常、重启说明（如保存/生成上限……）

## 3. 评测（离线）
- 指标表：score_mean/std、acc_main、json_ok、kbid_rate、len_resp
- 混淆矩阵：service / incident
- 错误样例（top-k 低分）

## 4. A/B
- 无工具 vs 可选工具、KL 强弱、GRPO vs PPO、DPO 前后
- 显著性：bootstrap Δscore_mean + 95% CI

## 5. 部署与一致性
- 合并到 HF 目录（如走 FSDP 合并脚本）并在 vLLM/SGLang 起 OpenAI 兼容服务（命令附录） [oai_citation:8‡vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?utm_source=chatgpt.com)
- 线上模板/温度与训练一致性说明
```



------





## **1）指标聚合脚本（复用 08 + 汇总）**





**脚本：scripts/make_report.py**（把多次评测 CSV 叠在一个 Markdown 表）

```
import argparse, pandas as pd, json, os
def main(csvs, out_md):
    rows=[]
    for path in csvs:
        df=pd.read_csv(path)
        n=len(df); sc=df["score"].mean(); js=(df["response"].apply(lambda x: "{" in str(x))).mean()
        acc=(df.apply(lambda r: int(str(r["ground_truth"]).split(":")== (lambda js: [(js.get("service") or ""), (js.get("incident_type") or "")])(json.loads(r["response"][r["response"].find('{'):])) ) if "{" in str(r["response"]) else 0, axis=1)).mean()
        rows.append({"file":os.path.basename(path),"n":n,"score_mean":sc,"json_ratio":js,"acc_est":acc})
    with open(out_md,"w",encoding="utf-8") as f:
        f.write("| file | n | score_mean | json_ratio | acc_est |\n|---|---:|---:|---:|---:|\n")
        for r in rows: f.write(f"| {r['file']} | {r['n']} | {r['score_mean']:.4f} | {r['json_ratio']:.2f} | {r['acc_est']:.2f} |\n")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True); ap.add_argument("--out", required=True)
    a=ap.parse_args(); main(a.csv, a.out)
```

> 你也可以直接用 08 篇的 eval_offline.py 输出多列指标，再由本脚本汇总。“训练-评测对齐”是**同一 reward_fn** 的复用，这点与官方推荐一致。 



------





## **2）吞吐（SPS）与显存：日志抓取与画图**





**训练端抓取**：建议在循环里把 SPS、time_rollout、time_update、gpu_mem_peak 输出到一行（CSV/TB）。

**解析脚本**（示例，配合 09 篇）：

```
# scripts/plot_sps_mem.py
import re, sys, numpy as np, matplotlib.pyplot as plt
sps,mem=[],[]
for ln in open(sys.argv[1],encoding="utf-8"):
    m=re.search(r"SPS=(\d+(\.\d+)?)",ln); n=re.search(r"mem_peak=(\d+)",ln)
    if m: sps.append(float(m.group(1)))
    if n: mem.append(int(n.group(1)))
plt.figure(); plt.plot(sps); plt.title("SPS over steps"); plt.savefig("reports/sps.png")
plt.figure(); plt.plot(mem); plt.title("GPU mem_peak (MB)"); plt.savefig("reports/mem.png")
```



------





## **3）“保存/恢复/生成上限”等常见坑（附出处）**





- **周期评测/生成上限**：trainer.test_freq 太频繁、数据太难 → “Generated too many… max_num_gen_batches” 报错，设置 max_num_gen_batches=0 可放开上限（issue 讨论给出原因与解法）。 

- **多机脚本**：官方提供了多机训练的入门文档（含 SLURM/Ray/Docker 步骤），可对照本项目的 09 篇调整。 

- **权重合并到 HF**：VERL 提供了合并工具把 Actor/FSDP 分片合成 HF 目录，便于后续用 vLLM/SGLang 起 **OpenAI 兼容**服务。

- **OpenAI 兼容服务**：

  

  - vLLM：vllm serve --api-key ...，支持官方 OpenAI 客户端直连。 
  - SGLang：支持原生与 OpenAI 兼容端点，示例调用在多处文档/教程中给出。 

  





------





## **4）一键生成复现实验包（Markdown + 图 + CSV）**





**脚本：scripts/gen_artifacts.sh**

```
#!/usr/bin/env bash
set -e
EXP=${1:-exp001}
mkdir -p reports/$EXP
# 1) 评测（离线）
python scripts/eval_offline.py --val $HOME/data/aiops_mturn/val.parquet \
  --pred outputs/$EXP/pred.jsonl \
  --reward-fn examples.reward_score.aiops_mturn:compute_score \
  --out reports/$EXP/eval.csv
# 2) 汇总表
python scripts/make_report.py --csv reports/$EXP/eval.csv --out reports/$EXP/summary.md
# 3) 吞吐/显存图（如有日志）
python scripts/plot_sps_mem.py outputs/$EXP/train.log || true
echo "Artifacts in reports/$EXP/"
```



------





## **5）报告应包含的“可复现元数据”**





- **git commit / pip freeze**；
- **模型/权重来源**（HF URL 或本地路径）；
- **chat template** 与 **温度/长度/stop**；
- **数据版本**（parquet 的 MD5/行数/通过质量闸门的比率）；
- **reward_fn 路径与版本**（与训练/评测一致）；
- **后端参数**：SGLang/vLLM 的关键启动项（TP/并发/dtype）。 





------





## **6）结语**





- 报告不是“漂亮图表”，而是**让别人/未来的你能复跑出来**。
- 用同一 reward_fn 做训练与离线评测，可保证“目标对齐”并且可查来源（这是 VERL 文档的推荐路径之一）。 



```
---

两篇先到这儿。下一轮我可以继续 15/16（例如**“错误分析脚本集 / 典型 Badcase 分类”** + **“FAQ & 附录（配置速查、字段表、术语）”**），或者你指定想优先补哪块。
```

