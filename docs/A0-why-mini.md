VERL 是什么 & 生态对比



# VERL 是什么？——从单机 DQN 到分布式 RL 的第一步

强化学习在实验室里常常是单机小实验：CartPole、MountainCar、Atari。
但一旦放到工业环境，就会遇到 **采样效率低、训练不稳定、分布式资源利用差** 等痛点。

字节跳动开源的 **VERL (Versatile RL Framework)**，目标就是解决这些问题：
- **模块化**：环境、算法、策略、训练器解耦，像搭积木一样拼装。
- **分布式**：采样和学习解耦，支持多机多卡，吞吐大幅提升。
- **工业落地**：不仅能跑经典 RL，还能和大模型（LLM、Diffusion）结合，用于推荐、运维、搜索。

这一篇，我们先从一个最小的例子出发：
- **单机 DQN** 在 CartPole 上跑起来。
- 再展示一个小小的「多进程采样」雏形，感受 VERL 为什么要强调分布式。

---

## 单机 vs 分布式 RL

| 特性 | 单机 RL             | VERL 风格的 RL                       |
| ---- | ------------------- | ------------------------------------ |
| 环境 | 一个进程内运行      | 多个 worker 并行收集                 |
| 算法 | 环境交互 + 学习耦合 | 解耦：Actor 只采样，Learner 专心训练 |
| 扩展 | 只能 CPU/GPU 单卡   | 支持集群、跨机 GPU                   |
| 应用 | 学术 Demo           | 工业落地（推荐、运维、搜索、广告）   |

---

## 本文的 Demo

我们实现了一个简化版 DQN：
- **单机模式**：一个进程玩 CartPole。
- **并行模式**：4 个进程同时玩，数据汇总给 Learner。

运行方式：

```bash
# 单机 DQN
python -m src.cli.train --config configs/cartpole/dqn.yaml

# 并行 DQN（4 workers）
python -m src.cli.train --config configs/cartpole/dqn.yaml runtime.num_workers=4
```





# A0 · verl 的前世今生：为什么又是一个 RL 框架？

> 如果你只在 CartPole 里写过几行 PPO，见到 LLM 的 RLHF 训练现场，多半会怀疑人生：  
> rollout 跑在独立推理引擎里，奖励函数半数是规则一半是模型，日志像瀑布一样刷，单机显存一眨眼就满了，集群还要和 Ray/K8s 打交道——这是“工程化 RL”，和教科书上的 RL 不是一个物种。

**verl**（Volcano Engine Reinforcement Learning for LLMs）就是在这样的背景里长出来的：  
它是一个**面向大模型后训练（post-training）**的强化学习框架，号称“**灵活、效率高、生产可落地**”，而且是 **HybridFlow 论文**的开源实现——也就是把 RL 这件事抽象为“**数据流**”，把**算法的控制流**和**底层算力/并行的计算流**拆开管理。 [oai_citation:0‡GitHub](https://github.com/volcengine/verl?utm_source=chatgpt.com)

---

## 1) 诞生动机：LLM 时代的 RL，不再是单进程小玩具

传统 DRL 的“控制流 + 计算流”常常塞在同一个进程里，模型也不大；但到了 LLM，**计算流天然是多进程/多机**（FSDP、Megatron 等），这逼着我们做一个选择：  
- 要么**把控制流也变成多进程**，和计算流绑死；  
- 要么**把两者解耦**：控制流单进程、计算流多进程，通过消息把数据来回传。  

**verl 选择了第二条**：控制流保持简单、可重用，计算流可随硬件/后端替换（FSDP、Megatron、vLLM、SGLang 等），代价是控制进程与工作进程之间需要高频数据传递。这个取舍在官方“HybridFlow Programming Guide”里写得很直白。 [oai_citation:1‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

> 一句话版：**算法循环**写得像单机，**并行优化**交给后端；换后端≠推倒重来。 [oai_citation:2‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

---

## 2) 现在进行时：verl 长成了什么样？

- **后训练一条龙**：从 SFT（可选）到 PPO/GRPO 等 RL 算法，再到多机训练、权重合并、在线服务，官方给了完整路径与样例（GSM8K 快速上手就是一个典型）。 [oai_citation:3‡ Verl](https://verl.readthedocs.io/en/latest/start/quickstart.html?utm_source=chatgpt.com)  
- **算法家族**：PPO、GRPO 以及一系列“Recipe/自博弈/偏好优化”的扩展，框架层面都给了入口与配置位。 [oai_citation:4‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)  
- **奖励体系**：推荐用**函数式奖励**或**RM（奖励模型）**，最关键的是——**自定义奖励**不用改框架源码，直接 `custom_reward_function.path/name` 指过去即可（我们在本仓第 04/06/07 篇一直这么用）。 [oai_citation:5‡ Verl](https://verl.readthedocs.io/en/latest/preparation/reward_function.html?utm_source=chatgpt.com)  
- **多轮/工具能力**：官方专门写了“**Multi-turn Rollout Support**”，提出**增量（delta）分词**来只训练“本轮助手输出”，解决了多轮下 token 对齐难的问题；SGLang 是一等公民，支持多轮 agent 式 RL、工具/函数调用等。 [oai_citation:6‡ Verl](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html?utm_source=chatgpt.com)  
- **后端与并行**：FSDP / Megatron 负责训练侧并行；vLLM / SGLang 负责 rollout 侧高吞吐；Ray 作为调度/远程执行基座，把 Controller 与各 Worker 组织起来跑。官方文档给到了入口函数、Ray Trainer、WorkerGroup 等源码导览。 [oai_citation:7‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)  
- **生态与落地**：  
  - **KubeRay 指南**：演示如何在 Kubernetes 上用 verl 做 GSM8K 的 PPO 训练； [oai_citation:8‡Ray](https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html?utm_source=chatgpt.com)  
  - **SkyPilot**：官方示例直接把 verl 标为“主流开源 RL 框架”，给了云上便捷起训练的模版； [oai_citation:9‡SkyPilot](https://docs.skypilot.co/en/latest/examples/training/verl.html?utm_source=chatgpt.com)  
  - **AMD ROCm**：AMD 官方写了延伸文与兼容页，说明 verl 在 MI300X 等硬件上的吞吐/收敛表现与起步方法。 [oai_citation:10‡ROCm Documentation](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/verl-compatibility.html?utm_source=chatgpt.com)

---

## 3) 一张“心智图”：从入口到训练循环（你会在源码里看到的）

- **入口**：`verl/trainer/main_ppo.py` 定义了一个单进程 **controller**（Ray 远程函数），在这里装配 RewardManager、构造 RayPPOTrainer，再调用 `fit()`； [oai_citation:11‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)  
- **RayPPOTrainer**：负责主循环、拉起/管理 **WorkerGroup**；  
- **WorkerGroup/Workers**：常见三类：`ActorRolloutRef`、`Critic`、`RewardModel`。它们暴露出 `generate_sequences / compute_log_prob / update_actor ...` 等接口，由装饰器指定**怎么切分/并行/聚合**； [oai_citation:12‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)  
- **Rollout 后端**：vLLM/SGLang/HF TGI；训练侧是 FSDP/Megatron；  
- **多轮与工具**：在 SGLang 里把前轮历史与工具结果自动拼接，**delta tokenization** 确保只有“新生成”的助手 token 计损失。 [oai_citation:13‡ Verl](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html?utm_source=chatgpt.com)

> 这套分层，让“写算法循环”和“调硬件/后端”解耦——你能把心思放在**奖励函数**与**稳定性/收敛**上。 [oai_citation:14‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

---

## 4) 为什么是“现在用 verl”，而不是“再等等”？

**功能成熟 + 快速跟进**：  
- 官方更新节奏很密，最近的版本/讨论里能看到**多轮高效分词与遮罩**、**Nsight/verl profiler**、**多模态/多节点增强**等改动；这意味着工程细节在持续被打磨，能跟得上生产侧的需求。 [oai_citation:15‡GitHub](https://github.com/volcengine/verl/discussions/2225?utm_source=chatgpt.com)  
- GitHub README 明确强调了和 **SGLang** 的深度协作，涉及“多轮 agent RL、VLM RLHF、Server-based RL、局部 rollout”等方向。对于“要上工具/要多轮/要服务端一体化”的团队，这些都是**一线能力**。 [oai_citation:16‡GitHub](https://github.com/volcengine/verl?utm_source=chatgpt.com)

**生态打通**：  
- 你可以用 Ray/KubeRay 管集群、用 SkyPilot 跑云上实验、用 ROCm 在 AMD 卡上训、最后把权重合并到 HF 目录，再用 vLLM/SGLang 起 **OpenAI 兼容**服务（我们在第 12 篇会手把手示范）。这些组件都有官方教程或示例，**不是 PPT。** [oai_citation:17‡Ray](https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html?utm_source=chatgpt.com)

**学习成本可控**：  
- 快速上手一条命令跑 GSM8K（PPO），奖励就是规则抽取“#### 后的答案”是否正确；这对入门者是友好的“第一口糖”。 [oai_citation:18‡ Verl](https://verl.readthedocs.io/en/latest/start/quickstart.html?utm_source=chatgpt.com)  
- 写自定义奖励函数走“文件注入”而非改源码；多轮/工具也都是配开关+YAML；把复杂度收敛在**数据与奖励**这两个你能掌控的点上。 [oai_citation:19‡ Verl](https://verl.readthedocs.io/en/latest/preparation/reward_function.html?utm_source=chatgpt.com)

---

## 5) 误区与正解

- **“多轮就是把两轮 prompt 拼一起？”**  
  不。多轮最大坑在**token 对齐**：历史消息经过模板展平后，assistant 的 token 和 role 不一一对应。verl 用**增量分词**只截取“新生成”的助手片段计损失；你只要开 `multi_turn`，训练端就会做对齐处理。 [oai_citation:20‡ Verl](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html?utm_source=chatgpt.com)

- **“奖励只能靠 RM 吗？”**  
  不。**函数式奖励**是第一公民：GSM8K/MATH 都有现成示范，你也能把结构化/格式/证据等逻辑写成轻量函数，或把 RM 分数与函数奖组合。 [oai_citation:21‡ Verl](https://verl.readthedocs.io/en/latest/preparation/reward_function.html?utm_source=chatgpt.com)

- **“我就单机玩玩，还用得着 Ray 吗？”**  
  需要。Controller/Worker 的编排与远程调用就是用 Ray；即便是单机多卡，RayPPOTrainer 也在跑，只是你感知不到而已。要上多机/K8s，只是把 Ray 从本地扩到集群。 [oai_citation:22‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

---

## 6) 一条“最小正路”：如果你今天就想开始

1. 跑官方 **GSM8K Quickstart**，熟悉数据与奖励函数的接口； [oai_citation:23‡ Verl](https://verl.readthedocs.io/en/latest/start/quickstart.html?utm_source=chatgpt.com)  
2. 把后端切到 **SGLang** 并开 **multi_turn**，看一眼多轮 JSON 任务的收敛曲线； [oai_citation:24‡ Verl](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html?utm_source=chatgpt.com)  
3. 写你自己的 **函数式奖励**（比如我们在 AIOps 里只看第二轮 JSON 是否命中 `service:incident`）； [oai_citation:25‡ Verl](https://verl.readthedocs.io/en/latest/preparation/reward_function.html?utm_source=chatgpt.com)  
4. 用 **KubeRay/SkyPilot** 在云上放大 batch 做一次 A/B（无工具 vs 可选工具、KL 大小对收敛的影响）； [oai_citation:26‡Ray](https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html?utm_source=chatgpt.com)  
5. 合并 checkpoint，走 **vLLM/SGLang** 起 **OpenAI 兼容**服务，做线上一致性校验。*(这一步我们在 A12 会给脚本。)*  [oai_citation:27‡GitHub](https://github.com/volcengine/verl?utm_source=chatgpt.com)

---

## 7) 预告：A1–A4 会讲什么？

- **A1**：把你熟悉的 CartPole PPO 与 verl 的训练循环并排，画一张“**概念等价表**”，把“课本 RL → 工程化 RL”一次性打通。  
- **A2**：沿着官方“入口→Ray Trainer→Workers→rollout/reward→日志”的路径，把**数据流与日志**梳成“火焰图式”的笔记。 [oai_citation:28‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)  
- **A3**：多轮/工具如何在 SGLang 里落地、奖励如何只看“最后一轮”、为什么 delta 分词能解决对齐。 [oai_citation:29‡ Verl](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html?utm_source=chatgpt.com)  
- **A4**：Ray 是什么？Buffer/Placements/WorkerGroup 怎么映射到显卡，为什么“分布式 Buffer”是训练稳定性的保险丝。*(我们会边讲概念，边贴最小源码片段。)*

---

### 参考与延伸阅读
- 官方文档首页与 Quickstart（GSM8K、PPO）。 [oai_citation:30‡ Verl](https://verl.readthedocs.io/?utm_source=chatgpt.com)  
- HybridFlow 编程指南（控制流/计算流解耦、RayPPOTrainer 与 Workers 源码导览、多轮支持）。 [oai_citation:31‡ Verl](https://verl.readthedocs.io/en/latest/hybrid_flow.html)  
- Multi-turn Rollout Support（delta tokenization）。 [oai_citation:32‡ Verl](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html?utm_source=chatgpt.com)  
- GitHub README（SGLang 深度支持与路线）。 [oai_citation:33‡GitHub](https://github.com/volcengine/verl?utm_source=chatgpt.com)  
- KubeRay 与 SkyPilot 集成示例；AMD ROCm 支持与性能报告。 [oai_citation:34‡Ray](https://docs.ray.io/en/latest/cluster/kubernetes/examples/verl-post-training.html?utm_source=chatgpt.com)