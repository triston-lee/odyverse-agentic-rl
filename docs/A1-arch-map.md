下面是 **A1-arch-map.md** 的成稿（可直接放进 docs/ 目录）。我把它写成“从 CartPole 的小宇宙走进 VERL 的大世界”的导览，既讲故事也给出源码/文档支撑与对照表，方便你把课本 RL 一步步映射到 VERL 的工程化 RL。



------



```
# A1 · 从 CartPole 的 RL 看 VERL 的架构地图

> 一根棍子、一辆小车、左右两个动作——**CartPole** 是很多人 RL 的启蒙。而到了 LLM 的后训练世界，环境不再是物理引擎，**“环境”变成了推理引擎**（vLLM / SGLang），奖励来自**函数或奖励模型**，训练循环由 **Ray** 调度的大型流水线驱动。  
> 这篇文章带你把“CartPole 世界”的概念，一一映射到 **VERL** 的工程化架构。

---

## 0. 先回家看看：CartPole 的最小 PPO 是什么样？

在 **Gym/Gymnasium** 里，CartPole 的目标是让杆子不倒，动作为 `left/right`，在每个时间步得到 +1 奖励；`v1` 版本最长 500 步，阈值 475 分视作“解题”。 [oai_citation:0‡健身房文档](https://gymnasium.farama.org/environments/classic_control/cart_pole/?utm_source=chatgpt.com)

**极简训练循环的心智图**（伪代码）：

```python
obs = env.reset()
for step in range(K):
    action = policy(obs)                      # πθ(a|s)
    obs_next, reward, done, info = env.step(action)
    buffer.add(obs, action, reward)          # 轨迹
    if done or len(buffer) >= B:
        adv, ret = estimate_gae(buffer)      # GAE(λ)
        update_actor_critic(adv, ret)        # PPO clip
        buffer.clear()
    obs = obs_next if not done else env.reset()
```



- **环境**：env.step(a) 真正产生下一状态与奖励；
- **奖励**：来自物理模拟，简单明了；
- **更新**：PPO“**在旧策略附近更新**”，用 clip 约束改动幅度。 





------





## **1. VERL 的世界观：把 RL 变成“数据流”**





来到 LLM 的后训练（RLHF/RLAIF/GRPO/PPO）里，**VERL** 主张用“**HybridFlow**”把 RL 拆成**控制流**（算法逻辑）与**计算流**（大规模并行）。控制流保持“像单机代码一样清晰”，计算流交给后端（FSDP/Megatron 训练侧；vLLM/SGLang 采样侧）与 Ray 调度去扩展。 



> **翻译成人话**：你写的还是“采样→算奖励→估优势→PPO 更新”，但“谁来并行生成/并行算 logprob/多机聚合”这些繁琐事，交给 VERL 的 **RayPPOTrainer + WorkerGroup** 去做。 



------





## **2. 一张“概念对照表”：CartPole ↔ VERL（LLM）**



| **CartPole 里的概念** | **在 VERL（LLM 后训练）里的等价物**   | **说明**                                                     |
| --------------------- | ------------------------------------- | ------------------------------------------------------------ |
| 环境 env.step(a)      | **Rollout 引擎**（vLLM / **SGLang**） | 给定“提示”（状态）和“生成配置”（动作分布），产出**模型回复**（下一状态）与对数似然等。SGLang 支持**多轮**和**工具调用**。 |
| 奖励 reward_t         | **自定义奖励函数**或**奖励模型 RM**   | VERL推荐将奖励函数写成**独立文件**，通过 custom_reward_function.path/name 注入，不改源码。 |
| 轨迹/回合             | **一组 prompt→response**（可多轮）    | LLM 里“一个回合”常指一次或多轮对话；多轮用**增量分词**只训练“本轮助手输出”。 |
| 策略 πθ               | **Actor 模型**（被训练的 LLM）        | 例如 Qwen2.5-Instruct；Ref 模型用于 KL 参考。                |
| 值函数 Vϕ             | **Critic 模型**                       | 单独的 value 头或共享底座。                                  |
| PPO 更新              | **RayPPOTrainer** 的训练循环          | 驱动多进程采样/回传/聚合/反传。                              |
| 渲染/日志             | **Logger + 验证导出**                 | 记录 reward/KL/loss/SPS，并可导出 pred.jsonl 离线评测。      |



------





## **3. “环境”是怎么变成 vLLM / SGLang 的？**





在 CartPole 里，环境是物理引擎；在 LLM RL 里，**环境是“把提示变成回复”的推理系统**。VERL 将其抽象为 **Rollout**：



- **vLLM**：单轮高吞吐生成（适合单轮/函数式奖励 baseline）。
- **SGLang**：多轮/工具优选，VERL 原生支持“**多轮 token 对齐**”（下节详述）。 





换后端，本质上只需**改配置**（例如 actor_rollout_ref.rollout.name=sglang），算法控制流保持不变，这正是 HybridFlow 的收益。 



------





## **4. 多轮的“对齐地狱”与 VERL 的解法**





Chat 模板把多轮消息扁平化成**一串 token**，很难知道哪些 token 属于“当前轮的 assistant”。VERL 在多轮里采用了**增量（delta）分词**：每生成一轮，就把“到此为止的模板文本”重新分词，**仅对“新增的助手片段”计损失与奖励**。因此我们可以放心地写“第 1 轮思路 + 第 2 轮只输出 JSON”的任务，奖励只看**最后一轮**。 



------





## **5. RayPPOTrainer：把“单机式写法”搬到多机**





VERL 的 **RayPPOTrainer** 在**Driver 进程**上跑控制流，包含三件大事：



1. **准备数据**（加载 parquet、组 batch）；
2. **初始化 WorkerGroup**（Actor/Ref/Critic 的远程工作者）；
3. **训练循环**（采样→评分→优势→PPO 更新），并行细节由 Worker 承担。 





这就像你在单机里写的那段伪代码，但在 VERL 里，“谁去生成/谁去算 logprob/谁去更新参数”是一个分布式团队协作。



------





## **6. 把 CartPole 伪代码，翻译成 VERL 的“算法骨架”**





**CartPole 伪代码**（再看一眼）：

```
# sample -> reward -> advantage -> PPO update
outs = env.step(policy(obs))
R = reward(outs)             # 环境给的
A, Vt = estimate_adv(R)      # GAE
loss = ppo_loss(outs.logp, A, Vt, KL)
update(loss)
```

**在 VERL 里的同构骨架**（伪代码化，不同版本函数名略异）：

```
# Driver (RayPPOTrainer)
for batch in train_loader:
    # 1) 采样：Rollout（vLLM/SGLang）分布式生成
    outs = workers.actor_rollout_ref.generate(batch.prompts)
    #    -> responses, logp_actor, logp_ref, values, masks, extra

    # 2) 奖励：调用你注入的 compute_score（函数式/RM）
    R = [compute_score(ds, resp, gt) for resp, gt in zip(outs.responses, batch.gt)]
    R_token = expand_to_tokens(R, outs.response_mask)   # 铺到有效 token

    # 3) 优势/回报
    A, Rt = compute_gae(R_token, outs.values, outs.response_mask)

    # 4) PPO 更新（DDP + 梯度累积）
    loss_actor, loss_critic, kl = ppo_update(outs, A, Rt, cfg.kl_ctrl)
    logger.log(...)
```



- **同构点**：采样→奖励→优势→更新，一样的；
- **不同点**：采样发生在 **vLLM/SGLang**，奖励可以完全自定义（比如“第 2 轮 JSON 命中才给分”），多轮靠**delta 分词**对齐。 





------





## **7. 一张“路线图”：从入口到各部件（Mermaid）**



```
flowchart LR
    A[main_ppo / RayPPOTrainer\n(控制流/Driver)] --> B[DataLoader\nparquet->batch]
    A --> C[WorkerGroup\n(Actor/Ref/Critic)]
    C --> C1[Rollout Engine\nvLLM / SGLang]
    C --> C2[LogProb & Values\n(Ref/Critic)]
    A --> D[RewardManager\ncustom_reward_function]
    A --> E[PPO Update\nclip+KL]
    C1 -->|responses, masks| A
    D -->|scores [0,1] (tokenized)| A
    C2 -->|logp_actor/ref, V| A
    A --> F[Logger/Checkpoint]
```



- **Rollout Engine** 就是“环境”；
- **RewardManager** 是“奖励来源”（函数/模型）；
- **RayPPOTrainer** 把一切串起来。 





------





## **8. 具体到“写代码”：奖励函数与后端切换**





**（1）奖励函数：完全“热插拔”**



VERL 官方建议把奖励写在独立文件，函数签名为

fn(data_source, solution_str, ground_truth, extra_info=None) -> float，

并通过配置注入：

```
custom_reward_function.path=examples/reward_score/aiops_mturn.py \
custom_reward_function.name=compute_score
```

这让你无需改框架源码，就能把“只看第二轮 JSON 是否命中”的逻辑接进去。 



**（2）后端切换：把“环境”换成 SGLang（多轮/工具）**

```
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.multi_turn=true
```

切到 SGLang 后，多轮历史与工具结果由后端自动拼接，**delta 分词**确保只对“当前轮助手输出”计损失。 



------





## **9. 为什么这套设计对“工程化 RL”很重要？**





- **换后端不用改算法**：今天单轮（vLLM），明天多轮+工具（SGLang），你只是改配置。 
- **奖励的“开放性”**：从 GSM8K 的“答案是否正确”，到我们 AIOps 的“第二轮 JSON 是否命中”，都能靠**函数式奖励**迅速落地。 
- **多轮的“对齐难题”被框架兜住**：delta 分词把“谁的 token 该算损失”这件棘手的工程问题抽掉了。 
- **分布式的复杂度放在 Ray 层**：你专注“训练学不学得会”，而不是“多机能不能跑起来”。 





------





## **10. 小练习（把 CartPole 思维迁移过来）**





1. 把你在 06 篇用的多轮 JSON 任务，当成“**离散动作空间**”：输出 service/incident 就像 left/right，试着写一个**“只命中就 +1，否则 0”** 的极简奖励，看看收敛速度。
2. 把 **KL 系数** 当作“**步长护栏**”：太小会放飞，太大会不学；用 3–4 个取值跑 A/B。
3. 用 **vLLM ↔ SGLang** 做一次“环境对换”实验：同数据、同奖励，观察**多轮对齐**带来的训练稳定性差异。
4. 读一眼 **RayPPOTrainer** 文档，找到“数据准备—WorkerGroup—PPO 循环”的三段代码位置，写下你自己的“源码导航笔记”。 





------





## **参考**





- CartPole 与 Gym/Gymnasium 文档。 
- PPO 原理与动机（SpinningUp + 论文）。 
- VERL：HybridFlow/多轮 tokenization/自定义奖励/Trainer。 
- VERL 官方仓库与更新要点。 



```
---

要不要我接着把 **A2（数据流与日志）** 也写出来？那一篇我会顺着 `RayPPOTrainer` 的三段主流程，把“batch 里到底有什么、rollout 回复里都带回了哪些张量、日志如何看出训练是否发散/卡 KL”讲得很“落地看板化”。
```