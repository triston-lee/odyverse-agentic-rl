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