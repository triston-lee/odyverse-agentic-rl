下面是 A3-multiturn-tool-mini.md 成稿（直接放进 docs/）。我把“多轮 + 工具”讲成一条从配置→数据→奖励→日志的短旅程：先点亮最短路径，再把“为什么这样设计”与“常见坑位”讲明白。关键结论我都查过 VERL / SGLang 官方文档与页面，并在文内标注出处。

⸻


# A3 · 多轮 / 工具迷你版：把“环境”接成会说话的搭档

> 单轮 RL 像是扔球：给一个 prompt，接一个回复，打个分结束。  
> 多轮 RL 像是打网球：来回好几拍，中途还能“喊个队友”（工具）帮忙。  
> 这一章我们在 **VERL + SGLang** 里把这场球打起来：**两段配置 + 一段奖励 + 一眼日志**。

---

## 0）先把“球场”搭起来：两行配置点亮多轮

VERL 已经把 **SGLang** 封成 rollout 引擎，多轮只要把型号拨到 `sglang`，打开 `multi_turn`：

```yaml
# A3/conf/multiturn.yaml
actor_rollout_ref:
  rollout:
    name: sglang           # 或 sglang_async（见 §3 提升并发）
    multi_turn: true       # 打开多轮模式

这就是官方“Multi-turn Rollout Support”的基础配置，意思很直白：让 rollout 阶段用 SGLang 与模型进行多轮交互。 ￼

小抖机灵：多轮的难点不是“会话怎么拼”，而是训练时只给“本轮助手 token”记损失。VERL 在多轮里采用delta-based tokenization（增量分词）解决了这个对齐地狱——只会把“新生成”的助手 token 打进 loss mask（response_mask）。 ￼

⸻

1）让搭档“会干活”：给 rollout 注入工具

多轮只是会说话；工具让它会“做事”。在 VERL 里自定义工具走一个套路：
	1.	写一个工具类（继承 BaseTool），实现入参与执行；
	2.	写一份 YAML 告诉 rollout 用哪些工具；
	3.	在 rollout 配置里挂上这份 YAML。

1.1 最小工具类（示例骨架）

# A3/tools/kb_lookup.py
from typing import Any, Dict, Tuple, Union
from verl.tools.base_tool import BaseTool, ToolResponse
from pydantic import BaseModel, Field

class KBArgs(BaseModel):
    query: str = Field(..., description="用户的查询关键词")

class KBLookup(BaseTool[KBArgs]):
    name = "kb_lookup"
    description = "查内部知识库，返回简要结论与证据片段"
    args_schema = KBArgs

    async def create(self, *args, **kwargs) -> tuple[str, ToolResponse]:
        # 如果工具会产出多模态输入，需返回 ToolResponse(image=..., video=..., text=...)
        return "inst-0", ToolResponse(text="ready")

    async def execute(self, args: KBArgs, **kwargs) -> Tuple[Union[str, Dict[str, Any]], float, dict]:
        # 这里调用你的检索/HTTP 服务
        doc = {"title":"cache OOM", "snippet":"命中率下降导致..."}
        # 返回 (文本或结构化结果, 工具耗时/或奖励片段, 附加日志)
        return {"summary":"怀疑缓存击穿", "evidence":[doc]}, 0.0, {}

VERL 的多轮文档明确：自定义环境交互工具基于 BaseTool；若工具返回多模态输入，需要在 ToolResponse 里提前处理图片/视频（如 Qwen2.5-VL），并在 rollout 里启用对应字段。 ￼

1.2 工具 YAML（告诉 rollout 怎么用它）

# A3/conf/tool.kb.yaml
tools:
  - class_name: A3.tools.kb_lookup.KBLookup
    config:
      type: native          # 自定义 Python 工具
    tool_schema:            # （可选）OpenAI function/schema 风格
      name: kb_lookup
      description: "查询内部知识库"
      parameters:
        type: object
        properties:
          query: { type: string }
        required: ["query"]

挂到 rollout：

# 继续 A3/conf/multiturn.yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn: true
    tool_kwargs:
      tools_config_file: A3/conf/tool.kb.yaml

官方文档给出了“Custom Tool Configuration + tools_config_file”的接入点，并附了 GSM8K 的工具示例可对照。 ￼

另外一条官方路线是 MCP 工具：如果你的工具是经 MCP（Model Context Protocol）提供的，配置里把 type: mcp 与 mcp_servers_config_path 指向相应 JSON 即可。 ￼

⸻

2）把“多轮 + 工具”落到任务：两轮分诊的对话脚手架

在 SGLang rollout 里，会话消息由后端自动维护。你在数据里只准备第一轮任务描述，例如：

第1轮：只说排查思路（不下结论）
第2轮：只输出 JSON：{"service": string, "incident_type": string}
现象：……
日志摘录：……
链路：……

	•	第 1 轮：模型输出“思路”；如需工具，模型在回复中按工具 schema 发起 tool_calls，SGLang 执行工具并把结果接回同一回合。
	•	第 2 轮：我们再追加一条 user：“现在仅输出 JSON”，奖励只看这一轮。
	•	增量分词保证只有“第 2 轮助手的新 token”入损失，从而避免把历史也算进来。文档直接给了 delta 分词的伪代码和“模板差异的健壮性选项”。 ￼

有些模型在渲染模板时会“去掉思考内容/思维标记”，可能造成“训练时与推理时模板不一致”。VERL 给了一个开关：
actor_rollout_ref.rollout.multi_turn.use_inference_chat_template = True，以及tokenization_sanity_check 的模式（strict / ignore_strippable / disable）来降低误报。 ￼

⸻

3）要更快？把引擎切到 sglang_async（异步多轮）

在“Search Tool Integration”示例里，VERL演示了 rollout.name: sglang_async 的配置：一边多轮生成、一边异步并发工具/检索，配合外部“本地 dense 检索服务”一同压榨吞吐。 ￼

actor_rollout_ref:
  rollout:
    name: sglang_async
    multi_turn:
      enable: true

官方还把“启动检索服务、并发与限速、输入输出格式”写成了教程（包括 Docker / uv / conda 两套环境与 60~70GB 的索引下载说明）。如果你只想接入自己已有的检索/HTTP 服务，按它的 I/O 协议改即可。 ￼

相关进展在 issue 里也有追踪：SGLang rollout 的 异步/多轮/工具正在往服务端形态演进（AsyncServerBase）。 ￼

⸻

4）奖励：还是老朋友“函数式奖励”，但只看最后一轮

你不需要为多轮改 RewardManager 的形态。还是那句熟悉的签名：

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    ...

	•	挂法：custom_reward_function.path/name 两个键，指到你写的 py 文件与函数名；
	•	数据：ground_truth / data_source 都在 non_tensor_batch 里；
	•	流程：先 detokenize 得到 solution_str，再把分数铺到 token（乘 response_mask）。这些都在官方的 RewardManager/Config 页写得很清楚。 ￼

训练循环也不用你操心：RayPPOTrainer 里已经把“生成 → 参考对数似然 → 价值 →（可选）RM → 函数式奖励 → KL 惩罚（可选）→ 优势 → 更新”串好了，并把 token 级分数写回 batch['token_level_scores']。 ￼

⸻

5）一眼看懂日志：多轮/工具要盯哪几根线
	•	任务指标：reward_mean / reward_std（是否在涨）、你自定义的 acc/json_ok（离线评测算）。
	•	对齐指标：kl（别双重惩罚）、response_mask 的有效长度（多轮每回合是否合理）。
	•	系统吞吐：timing/gen、timing/update_actor/critic 与 SPS；若使用检索或外部工具，采样阶段耗时会显著上浮，考虑开 sglang_async 与提升工具并发。 ￼

⸻

6）“多轮 + 工具”的三类常见坑
	1.	Token 对齐告警不断
	•	现象：训练 log 频繁提示“delta 与 full tokenization 不一致”。
	•	处理：把 tokenization_sanity_check_mode 暂调到 ignore_strippable 排除空白字符差异；若你的生产模板确实不同，可开 use_inference_chat_template=true 跟生产对齐。 ￼
	2.	工具接回多模态内容
	•	现象：工具返回了图片/视频，但 rollout 不认。
	•	处理：在工具的 create/execute 里用文档提供的 process_image/process_video 先做预处理；同时把数据集的 return_multi_modal_inputs 设为 False，避免冲突。 ￼
	3.	评测太勤，训练节奏被打断
	•	现象：每次到验证都要等很久（尤其接了检索，验证集成千上万）。
	•	处理：扩大 test_freq，或只在关键步保存/评测一次（官方检索示例里也备注了大验证集的耗时问题）。 ￼

⸻

7）对照：为什么 VERL 的多轮首推 SGLang？
	•	SGLang：原生支持多轮/工具、函数调用 schema、以及 OpenAI 兼容接口，配合 VERL 的 delta 分词 和 sglang_async，是“多轮 RL + 工具”的一等公民。 ￼
	•	vLLM：也有工具调用（Chat API 下的 function-calling / guided decoding），但 VERL 的多轮/工具训练教程与代码样例是围绕 SGLang 写的；如果你只做单轮，vLLM 是高吞吐首选。 ￼

⸻

8）把一切串起来：最小跑通清单
	1.	多轮开关：rollout.name=sglang + multi_turn=true；必要时加
multi_turn.tokenization_sanity_check_mode=ignore_strippable。 ￼
	2.	工具 YAML：写好 tools_config_file 并在 rollout 里挂上（自定义 Python 或 MCP）。 ￼
	3.	奖励函数：沿用单轮签名，只解析最后一轮输出；配置里 custom_reward_function.path/name 指过去。 ￼
	4.	（可选）异步引擎：把 name 切成 sglang_async，外接你的检索/工具服务，按官方 I/O 协议适配。 ￼
	5.	日志体感：看 reward_mean / kl / timing/* / SPS，确认采样与更新占比、KL 是否健康。 ￼

⸻

参考与延伸
	•	Multi-turn Rollout Support（基础配置、工具接入、delta 分词、多模态注意事项）。 ￼
	•	Search Tool Integration（sglang_async、检索服务 I/O、并发/限速建议、全流程脚本）。 ￼
	•	Reward Function / Config Explanation（RewardManager、函数签名、custom_reward_function.*）。 ￼
	•	PPO Ray Trainer（训练循环里各阶段的输入/输出与计时指标）。 ￼

---

要不要我顺手把这一篇里用到的 **`A3/tools/kb_lookup.py`** 和 **两份 YAML** 真文件加到你的仓库结构（再配一个极简 `run.sh`）？或者你想先看 **A4：Ray 与分布式 Buffer 的“小黑盒观察法”**，我也可以直接开写。