源码导览地图（从入口到训练循环的最短路径）
 
0 如何开始读：四个“搜什么”的关键词

1 入口层：命令行 → 配置 → 构造 Trainer

2 数据层：Parquet → 批次 → 多轮增量分词

3 Rollout 层：vLLM / SGLang

4 奖励层：RewardManager & 自定义函数

5 优势估计：GAE(lambda)、value_target

6 日志与保存



```
---

## docs/03-arch-tour.md

```markdown
# 03 · 源码导览地图（从入口到训练循环的最短路径）

> 这篇不是“完整 API 文档”，而是**帮你 1–2 小时内把主链路读明白**：入口 → 配置 → 数据 → rollout → 奖励 → 优势/更新 → 日志/保存。不同版本文件位置会有差异，建议用 `ripgrep/rg` 或 IDE 全局搜索定位函数名。

---

## 0）如何开始读：四个“搜什么”的关键词

1. **训练入口**：搜索 `main_ppo`  
2. **Trainer 主循环**：搜索 `train_one_epoch` / `train_loop` / `ppo_trainer`  
3. **多轮/后端**：搜索 `rollout`、`sglang`、`vllm`  
4. **奖励接入**：搜索 `reward`、`compute_score`、`custom_reward_function`

---

## 1）入口层：命令行 → 配置 → 构造 Trainer

**你会看到的流程（伪代码）**

```python
def main_ppo():
    cfg = parse_config_and_cli_overrides()
    # 1) 构造 tokenizer / 模型句柄（actor/ref/critic）
    actor, ref, critic = build_models(cfg.actor_rollout_ref.model, cfg.critic)
    # 2) 构造数据流水线（parquet → dataloader）
    train_loader, val_loader = build_dataloaders(cfg.data)
    # 3) 构造 rollout 引擎（vLLM / SGLang）
    rollout = build_rollout_engine(cfg.actor_rollout_ref.rollout, actor, ref)
    # 4) 奖励函数（内置/自定义）与 RewardManager
    reward_fn = load_custom_reward(cfg.custom_reward_function) or builtin_reward(cfg.data)
    # 5) 组装 PPO Trainer
    trainer = PPOTrainer(
        actor=actor, critic=critic, ref=ref,
        rollout=rollout, reward_fn=reward_fn, cfg=cfg
    )
    trainer.fit()
```

**看点**



- CLI 覆盖：大多使用 OmegaConf/DictConfig 体系；命令行 a.b.c=... 会覆盖配置树
- 入口会在“训练前评估”与“保存频率”上读 trainer.* 键





------





## **2）数据层：Parquet → 批次 → 多轮增量分词**





**典型职责**



- 读取 parquet → 校验存在 prompt/ground_truth（或你在 04/06/07 用到的其它列）
- 训练时按 data.train_batch_size 构造批次；多轮场景下对**用户/助手**分别应用模板
- **增量/差分 tokenization**：只把“当前轮 assistant 的 tokens”计入损失/优势计算（上轮历史拼接进上下文但不重复计）





**代码阅读建议**



- 找“dataloader”或“dataset”的构造函数，看它如何从 non_tensor_batch 中带过 ground_truth 等非张量字段
- 确认多轮时的“截断策略”：上下文是否可能超过 max_model_len，是否有安全阈值





------





## **3）Rollout 层：vLLM / SGLang**





**单轮（vLLM）**



- 关注：generate 的批内并发、KV Cache 使用、logprob 的计算路径
- 常见优化点：gpu_memory_utilization、max_model_len、log_prob_micro_batch_size_per_gpu





**多轮/工具（SGLang）**



- 关注：multi_turn=true 时，历史对话如何拼接；工具调用的事件流（函数 schema → 触发 → 执行器 → 结果写回）
- 工具 YAML 加载点：一般在 rollout 构造时读取 tool_kwargs.tools_config_file，把工具类注册到可用列表
- 若提供 interaction_config_file：会在轮与轮之间插入“系统提示/提醒/证据落盘”等中间消息





**代码阅读建议**



- 找到“后端抽象接口”（如 RolloutEngine/Sampler）与 vLLM/SGLang 的实现类
- Grep multi_turn 字眼看看它在何处控制对话组装与分词窗口





------





## **4）奖励层：RewardManager & 自定义函数**





**关键路径**



- Trainer 在每个 batch 采样后，拿到 responses（文本/ids）与 non_tensor_batch（包含 ground_truth / data_source / 你自定义的字段）
- 调用 RewardManager → 若配置了 custom_reward_function 则走你提供的 compute_score；否则按 data_source 路由到内置数据集奖励
- 返回 rewards ∈ [0,1]（建议），进入优势估计





**代码阅读建议**



- 搜 custom_reward_function 的解析与导入（importlib/SourceFileLoader）
- 搜 ground_truth 在 reward 函数中的使用（你会看到它来自 parquet 的 non_tensor_batch）





------





## **5）优势与更新：PPO 训练循环**





**主干（伪代码）**

```
for epoch in range(E):
    for batch in train_loader:
        # 1) 采样：rollout.generate(prompts) → responses, logprob(actor/ref)
        outs = rollout.generate(...)
        # 2) 奖励：reward = reward_fn(data_source, solution_str, ground_truth, extra)
        R = compute_reward(outs.responses, batch.non_tensor)
        # 3) 估计优势：GAE(lambda)、value_target
        adv, vt = estimate_advantage(R, critic_values, gamma, lam)
        # 4) PPO 更新（分 mini-batch / micro-batch）
        for mb in minibatches(outs, adv, vt, size=cfg.actor.ppo_mini_batch_size):
            loss_actor, kl = ppo_actor_step(...)
            loss_critic   = critic_step(...)
        # 5) 记录日志/保存/评测
        log_metrics(...)
    maybe_eval_and_save(...)
```

**看点**



- KL 的两个分支：**并入 reward** vs **加入 loss**（二选一）
- 维度对齐：advantage/value_target 的 shape 是否按“样本级”或“token 级”（不同实现会先聚合到样本级）
- 微批/梯度累积：PPO 更新里如何在显存受限时分摊





------





## **6）日志与保存**





**常见指标**



- 训练：reward_mean / reward_std / kl / loss_actor / loss_critic / sps
- 系统：GPU 利用率/显存、数据加载耗时、采样时间/更新时间占比
- 多轮：平均轮数、工具调用率、JSON 合规率（若你在 reward 或 hook 里统计）





**阅读建议**



- 找到 logger 抽象（add_scalar / log_dict），确认它把哪些键写到 TB/CSV
- 保存策略：save_freq + global_step 下的 actor/critic 权重目录；若有“合并权重”脚本，看看它如何把 FSDP shard 合并成 HF 目录





------





## **7）把源码与配置对上号（速查表）**



| **你在 CLI 改的键**         | **影响的源码区域（搜索提示）** | **作用**               |
| --------------------------- | ------------------------------ | ---------------------- |
| data.*                      | build_dataloader / dataset     | 加载/截断/批次         |
| actor_rollout_ref.model.*   | build_models                   | 初始化 tokenizer/模型  |
| actor_rollout_ref.rollout.* | build_rollout_engine / `sglang | vllm`                  |
| actor_rollout_ref.ref.*     | compute_logprob 路径           | KL/对比对数似然        |
| actor.* / critic.*          | ppo_actor_step / critic_step   | 学习率/批次/优化器     |
| algorithm.*                 | kl_coeff_schedule / ppo_loss   | 稳定性/探索-利用权衡   |
| custom_reward_function.*    | RewardManager                  | 正确性/结构化/证据加分 |
| trainer.*                   | fit()/eval()/save()            | 训练节奏/日志/保存     |



------





## **8）建议的读代码顺序（两小时拿下主链）**





1. main_ppo（10 分钟）：入口与配置注入
2. build_rollout_engine（20 分钟）：vLLM/SGLang 差异与多轮开关
3. RewardManager（20 分钟）：自定义函数如何被调用
4. ppo_trainer（40 分钟）：优势/更新/微批/日志
5. （可选）工具与交互 YAML 的加载流程（30 分钟）：对照 07





------





## **9）常见“读不下去”的卡点与解法**





- **找不到文件名/路径不一致**：版本差异；请全局搜函数名/类名（main_ppo, RewardManager, PPOTrainer）。
- **维度错乱**：在优势/损失里打印 shape（或看单元测试）；多轮时注意是否已经“样本级聚合”。
- **多轮上下文爆长**：确认有最大轮数/最大长度的截断；必要时在 prompt 明确“两轮”，奖励对超轮惩罚。





------





## **10）阅读清单（带问题去代码里找答案）**





- KL 系数是**静态**还是**动态调度**？
- 优势估计用的 **GAE(λ)** 具体实现在哪？
- Actor/Ref 的 logprob 是在后端算还是在本地算？
- 多轮时，**上一轮**的 assistant tokens 是否从损失中排除了？
- 断点续训如何恢复 optimizer/step 状态？
- 合并权重到 HF 目录的脚本在什么地方、如何处理 FSDP shard？





> 把这些问题逐一定位，你就已经“走通一遍 VERL 的主脉络”了。

```
---

要不要我下一轮继续补**05「训练循环拆解（PPO）」**和**08「评测与对齐（离线指标）」**？我会把 05 写成“张量形状 + 数学公式 + 实操调参”，08 写成“评测脚本 + 图表模板 + 无工具/可选工具对比”的权威版。
```

