下面是 A2-dataflow.md 成稿（直接放进 docs/ 就能用）。我把一条样本从 parquet 读进来、经 rollout、奖励、优势、更新、到日志与保存的整条流水线讲清楚；每个“会踩坑”的点都配了“怎么在日志里看出来”。

⸻


# A2 · VERL 的数据流与日志：一条样本的奇幻漂流

> 先给你一句话的全景图：  
> **控制流像单机**（一条清爽的 PPO 训练循环），**计算流像工厂**（vLLM/SGLang 负责生成、FSDP/Megatron 负责并行、Ray 把人马都调起来）——这就是 VERL 的 HybridFlow。 [oai_citation:0‡Verl Documentation](https://verl.readthedocs.io/en/latest/hybrid_flow.html?utm_source=chatgpt.com)

---

## 0）从 1 条样本开始：一眼望穿主链路

**数据流（Driver 视角）**

1. **读数据**：从 `train.parquet` 取出一批 `prompt`（还有你自己存的 `ground_truth` 等非张量字段）。这些列名用 `data.prompt_key` 等配置对上即可。 [oai_citation:1‡Verl Documentation](https://verl.readthedocs.io/en/latest/examples/config.html?utm_source=chatgpt.com)  
2. **rollout 采样**：把 batch 丢给 **Rollout 引擎**（单轮首选 vLLM；多轮/工具首选 **SGLang**），取回 `responses / logp_actor / logp_ref / values / response_mask …`。 [oai_citation:2‡GitHub](https://github.com/sgl-project/sglang?utm_source=chatgpt.com)  
3. **奖励**：调用你**热插拔**的函数式奖励（或 RM），输出每条样本的分数（建议 `[0,1]`），再按 `response_mask` 铺到 token 级。 [oai_citation:3‡Verl Documentation](https://verl.readthedocs.io/en/latest/preparation/reward_function.html?utm_source=chatgpt.com)  
4. **优势 / 回报**：按 GAE(λ) 估计 `advantage` 和 `returns`，形状与遮罩全都对齐到“**本轮助手的 token**”。  
5. **PPO 更新**：按 clip 公式更新 actor/critic；**KL** 既可以走 **loss 分支**，也可以并入 **reward**（二选一！）。 [oai_citation:4‡Verl Documentation](https://verl.readthedocs.io/en/latest/examples/config.html?utm_source=chatgpt.com)  
6. **日志与保存**：把 `reward/kl/loss/SPS` 等指标写到后端（console/TensorBoard/W&B/MLflow…），到 `save_freq/test_freq` 再做保存/评测。 [oai_citation:5‡Verl Documentation](https://verl.readthedocs.io/en/latest/_modules/verl/utils/tracking.html?utm_source=chatgpt.com)

---

## 1）入口与数据：最小“门面三件套”

- **入口脚本**：`main_ppo`（或你自己的 `run.sh`）把配置与命令行覆盖项合并，然后构造 RayPPOTrainer 并 `fit()`。Trainer 在 Driver 进程上运行，负责**数据准备 → WorkerGroup 初始化 → PPO 训练循环**三件大事。 [oai_citation:6‡Verl Documentation](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html?utm_source=chatgpt.com)  
- **数据格式**：官方 **Quickstart (GSM8K)** 就是两列起步（`prompt` + `ground_truth`），你也可以在同一行塞更多自定义信息。列名不一致，用 `data.prompt_key=...` 对上即可。 [oai_citation:7‡Verl Documentation](https://verl.readthedocs.io/en/latest/start/quickstart.html?utm_source=chatgpt.com)  
- **一个良好的习惯**：把**同源的奖励函数**（训练与评测）放进单独 `.py`，通过  
  `custom_reward_function.path/name` 指过去，不改框架源码。 [oai_citation:8‡Verl Documentation](https://verl.readthedocs.io/en/latest/preparation/reward_function.html?utm_source=chatgpt.com)

---

## 2）“环境”= 推理后端：vLLM vs SGLang

- **vLLM**：单轮高吞吐的“直线加速器”，适合函数式奖励的 baseline 或离线评测。  
- **SGLang**：多轮/工具一等公民；VERL 的多轮是**增量（delta）分词**——**只为“本轮助手新生成”的 token 计损失/奖励**，这解决了多轮模板对齐地狱。开启方法只需：  
  ```yaml
  actor_rollout_ref:
    rollout:
      name: sglang
      multi_turn: true

之后历史与（可选）工具的返回由后端自动拼接，你在奖励里只管看最后一轮。 ￼

记好一个词：response_mask。它告诉你哪些 token 属于“当前轮助手”，后面的奖励/优势/损失都要乘这个 mask。

⸻

3）奖励：把业务规则装进一个小函数
	•	签名与注入：

def compute_score(data_source, solution_str, ground_truth, extra_info=None)->float:
    ...

在配置里用
custom_reward_function.path=... 与 custom_reward_function.name=compute_score 指过去，VERL 就会在训练循环里自动调用它。 ￼

	•	常见做法：
	•	解析最后一轮文本（例如抽出 {"service","incident_type"}），命中就给高分；
	•	结构化合规加点分；
	•	（可选）证据/工具标签加点分。
	•	铺到 token：把样本级分数乘 response_mask 铺到 token 维度；有的任务也会只把末 token收满奖。

⸻

4）优势、KL 与 PPO：容易搞混的三件事
	•	GAE(λ)：对齐到 token/样本的优势、回报，必须配合 mask 去掉 padding 与历史。
	•	KL 的两条路（择一）：
	1.	use_kl_loss=true：在 actor 损失里加 KL 项；
	2.	把 KL 合到奖励：等价于在 reward_token 上做 r' = r - β·KL；
这两条别叠加，否则会“双算惩罚”，训练走不动。 ￼
	•	Ref 模型：用来算 KL 的“参考分布”（常指向同一路的 SFT/初始策略）。

⸻

5）日志怎么读：把“乱麻”拽成四根主线

不同后端/版本的键名略有差异，但意义大体相通；下列条目你基本都会在 console/TensorBoard/W&B 看到。VERL 的 tracking 模块里能看到多个后端适配（console、file、TensorBoard、ClearML…）。 ￼

一、学习是否在“涨”
	•	reward_mean / reward_std：主目标在涨没？方差是否变小？（方差剧烈波动 = 奖励不稳）
	•	acc / json_ok / pass@1：如果你在验证阶段导出了预测文件，离线再算这些“业务指标”。

二、分布漂没漂
	•	kl：策略偏离参考的幅度。过大=“放飞”，过小=“被 KL 压住学不动”。（也可在 critic/kl 或自定义键下看到，取决于你走哪条 KL 接法。） ￼

三、更新是否健康
	•	loss_actor / loss_critic：是否在合理区间收敛。critic 爆炸常见于值函数学习率偏大或回报尺度不稳。
	•	clip_ratio（若实现暴露）：有多少样本被 PPO clip 到了下/上界，长期接近 1 说明学习率/优势过大。（有些 RLHF 工具链会常规记录该指标。） ￼

四、系统层面
	•	SPS（samples/sec）与采样/更新用时：采样比更新慢很多 → 往往是 SGLang/工具 I/O 成瓶颈；多开并发或用 Ray 并行采样。
	•	验证/保存频率：trainer.test_freq / save_freq / total_epochs 会直接影响“训练节奏”（测得太勤吞吐会掉）。社区问题里也常讨论这几个参数的作用。 ￼

一段典型 console 片段（示意）

step=120  reward_mean=0.63  kl=0.004  loss_actor=0.71  loss_critic=0.38  SPS=92
step=130  reward_mean=0.68  kl=0.007  loss_actor=0.65  loss_critic=0.35  SPS=88

	•	读法：reward 在涨、KL 在可控区间（1e-3~1e-2 量级典型），SPS 稳；如果下一次 KL 飙到 0.05，你就要考虑调小温度/调大 KL 系数了。

⸻

6）验证与保存：别把训练节奏“打断”
	•	保存/评测：trainer.save_freq 控制 checkpoint 周期；trainer.test_freq 控制验证节奏；有时“评测样本很长 + 评测太勤”会显著拖慢训练，先把频率放大。（社区问答里就围绕这几个键讨论过影响。） ￼
	•	日志后端：默认有 console；你也可以启用 TensorBoard/W&B/MLflow/文件等（跟踪模块里能看到适配器的注入）。 ￼

⸻

7）两张“工程化心智卡”

7.1 数据面（批内你拥有什么？）

名称	说明
prompt	任务输入，可能是多轮“第 1 轮：思路 / 第 2 轮：只输出 JSON”
ground_truth	你自带的标签（例如 api:latency_spike）用于奖励
responses	rollout 生成的文本列表
response_mask	本轮助手 token 的 0/1 掩码（多轮对齐的关键）
logp_actor / logp_ref	策略与参考的 token 对数似然，用于 PPO/KL
values	critic 估计的 V 值（token 或聚合后样本级）
extra_info	可自定义记录：工具调用次数、证据标签等

记住：mask 一定要乘上（loss / advantage / KL 都要遮掉 padding 与历史）。多轮的 token 对齐靠的就是它。 ￼

7.2 日志面（你该盯什么？）
	•	任务指标：reward_mean、acc/pass@1、json_ok
	•	对齐指标：kl（以及你是否“只看最后一轮”的打分）
	•	优化稳定性：loss_actor/critic、（可选）clip_ratio
	•	系统健康：SPS、显存峰值、采样/更新占比
	•	训练节奏：test/save 的频率是否合适

⸻

8）三种经典“异常形态”，以及怎么排查
	1.	reward 不涨、kl 很小
	•	现象：loss 在抖，KL 常年 < 1e-3
	•	排查：KL 系数是不是太大了？奖励是不是过于苛刻（大多数样本被打 0）？
	•	手段：先只留主奖（命中就满分），把格式/证据的 shaping 暂关一轮；再把 KL 从 1e-3 降到 5e-4 试。 ￼
	2.	kl 暴走
	•	现象：reward 短时涨、随后质量崩，KL 飙升
	•	排查：是不是双重 KL（reward 里减了、loss 里也加了）？或者温度太高/生成长度过长？
	•	手段：确保 use_kl_loss 与“KL 入奖励”二选一；降低温度 / 缩 max_response_length。 ￼
	3.	多轮训练“对不上号”
	•	现象：第二轮该是 JSON，却老是念叨解释；或者奖励在第一轮文本里触发
	•	排查：SGLang 是否开启 multi_turn:true？奖励是否只解析最后一轮？
	•	手段：在奖励里只抓最后一个 {...}；必要时在两轮间插一条系统提醒。 ￼

⸻

9）把“数据流 → 日志 → 调参”闭环起来
	•	同源奖励：训练与离线评测用同一个 compute_score，否则目标会“错位”。 ￼
	•	多轮必看 mask：所有张量（adv/kl/loss）都乘 response_mask；这是你判断“对齐是否正确”的第一证据。 ￼
	•	频率与节奏：合理设定 test_freq/save_freq/total_epochs，让训练别被频繁评测掣肘。 ￼
	•	日志后端：本地先 console，稳了再开 TensorBoard/W&B；tracking 模块里能看到支持的后端适配器。 ￼

⸻

延伸阅读
	•	HybridFlow 编程指南（为什么要把控制流和计算流分离；RayPPOTrainer 的职责）。 ￼
	•	Quickstart：GSM8K（数据与最小 PPO 路线）。 ￼
	•	Multi-turn Rollout Support（SGLang、多轮增量分词/工具）。 ￼
	•	Config Explanation（custom_reward_function.*、use_kl_loss 等）。 ￼
	•	RayPPOTrainer（数据准备 / WorkerGroup / 训练循环三大块）。 ￼

---

如果你想，我可以再加一版 **“日志对照表（字段名别名）+ 真实训练截图位”**，把你仓库里的实际键名映射成这篇的通用名词，这样读日志就更顺手了。