

```
---

## docs/12-export-and-serve.md

```markdown
# 12 · 导出与部署（vLLM / SGLang OpenAI 兼容服务）

> 目标：把 06/07/11 训练出来的权重**合并为 HF 目录**，然后分别用 **vLLM** 与 **SGLang** 启动 **OpenAI 兼容**服务，最后给出 AIOps Demo 的调用样例与生产加固要点。

---

## 1）从 VERL checkpoint → HuggingFace 目录

VERL 默认**不直接**把 Hugging Face 目录打进 checkpoint；官方提供了 **model merge 工具**把 FSDP/Megatron 的分片权重合并到 HF 结构。位置与用法见文档“Checkpoint / Config Explanation”。 [oai_citation:15‡Verl](https://verl.readthedocs.io/en/latest/advance/checkpoint.html?utm_source=chatgpt.com)

**最小用法（以 FSDP 分片为例）**
```bash
# 假设你的最新 checkpoint 在 outputs/ckpt-XXXX
python -m verl.model_merger.merge_fsdp_to_hf \
  --actor_ckpt_dir outputs/ckpt-XXXX/actor \
  --out_hf_dir   out_hf/aiops-ppo-ckpt
# 旧版本可用 legacy merger: verl/scripts/legacy_model_merger.py（见官方说明）
```

> 如果你训练时用了 LoRA，合并脚本同样负责把 adapter 融回基座；遇到异常可参考相关 issue。 



------





## **2）用 vLLM 启服务（OpenAI 兼容）**





vLLM 提供 vllm serve 一键起 **OpenAI 兼容** HTTP 服务（Chat/Completions 等）。 



**启动**

```
pip install vllm
vllm serve out_hf/aiops-ppo-ckpt --host 0.0.0.0 --port 8000 --dtype auto --api-key sk-local-xxx
```

**调用（Python · openai 客户端）**

```
from openai import OpenAI
cli = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local-xxx")
resp = cli.chat.completions.create(
  model="out_hf/aiops-ppo-ckpt",
  messages=[
    {"role":"user","content":"两轮事故分诊……（你的第06篇提示词）"}
  ],
  temperature=0.2,
  max_tokens=128,
)
print(resp.choices[0].message.content)
```

> vLLM 的 OpenAI 兼容服务还支持 Docker 部署与更多启动参数；详见文档。 



------





## **3）用 SGLang 启服务（OpenAI 兼容）**





SGLang 提供 **原生 HTTP** 与 **OpenAI 兼容**两类端点；在多家平台/文档中都给了最小启动与调用示例。 



**安装与最小启动**

```
pip install "sglang[all]"
python -m sglang.launch_server \
  --model-path out_hf/aiops-ppo-ckpt \
  --host 0.0.0.0 --port 30000 --tp-size 1 --dtype bfloat16
```

**OpenAI 客户端调用**

```
from openai import OpenAI
cli = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
r = cli.chat.completions.create(
  model="default",
  messages=[
    {"role":"user","content":"两轮事故分诊……（你的第06篇提示词）"}
  ],
  temperature=0.2, max_tokens=128)
print(r.choices[0].message.content)
```

> 多篇教程/页面明确 SGLang 支持 OpenAI 兼容服务；也可直接用 HTTP/原生 API。 



------





## **4）把“多轮 AIOps 分诊”的线上调用写清楚**





**关键**：上线时要把训练时用的**对话模板**复现；多轮需把**第一轮思路 → 第二轮 JSON**拼到同一会话里（客户端负责 messages 累积）。

```
msgs=[
  {"role":"user","content":"两轮事故分诊规则…（第06篇首轮提示）\n现象：…\n日志：…\n链路：…"},
]
# 1st round
r1 = cli.chat.completions.create(model=MODEL, messages=msgs, temperature=0.2, max_tokens=128)
msgs.append({"role":"assistant","content":r1.choices[0].message.content})
# 2nd round：继续同会话，请只输出 JSON
msgs.append({"role":"user","content":"现在给出第二轮，仅输出 JSON。"})
r2 = cli.chat.completions.create(model=MODEL, messages=msgs, temperature=0.0, max_tokens=64)
print("JSON:", r2.choices[0].message.content)
```



------





## **5）健康检查与硬化**





- **SGLang**：实践文档/平台教程里给了 /health、/get_server_info 等端点可做监控；建议网关层做熔断/限流。 
- **vLLM**：用 --api-key 开启鉴权；结合反向代理统一证书与审计。 
- **并发与显存**：--dtype、TP size、gpu_memory_utilization（vLLM）这些都会影响吞吐/稳定性。 
- **SGLang vs vLLM**：不同硬件/版本吞吐差异较大；社区对两者的对比在持续更新，建议你根据实际工作负载压测选型。 





------





## **6）从训练到上线的一条龙脚手架**





1. **合并权重** → HF 目录（verl.model_merger 工具）。 
2. 用 **08 篇评测脚本**在 HF 目录上做**离线评测**（可直接 transformers 推理）。
3. 走 **vLLM/SGLang** 部署，接同一批验证样本做 **线上一致性校验**。
4. A/B：PPO vs GRPO、DPO 前后、RM on/off（同一验证集、同一模板）。
5. 观察：**命中率、JSON 合规率、平均响应时延、95/99 延迟**。





------





## **7）FAQ**





- **合并后模型名怎么给 vLLM？** 直接给 HF 目录路径即可（本地路径或上传到 Hub 也行）。 
- **SGLang 一定要 OpenAI 兼容吗？** 不强求。原生 HTTP/SDK 更灵活，OpenAI 兼容方便“无缝换后端”。 
- **训练时用 SGLang，多轮上线也要 SGLang 吗？** 不必须，但**模板/上下文拼接策略**要一致；若切 vLLM，务必对齐 chat template 与温度/长度。





------



```
---

想继续的话，我下一轮可以把 **13「数据工程与合成（偏好对 / 负样本 / 规则校验）」** 和 **14「复现实验报告模板（SPS/显存/指标图表）」** 补上，顺手把你 AIOps 的偏好对合成、错误注入与质量控制做成一键脚本。
```

