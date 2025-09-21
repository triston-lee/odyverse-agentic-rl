没问题——这轮我把 **05「训练循环拆解（PPO）」** 和 **08「评测与对齐（离线指标）」** 两篇一次性写全，尽量结合“源码层面的实现思路 + 可运行最小片段”。直接把下面两段保存到 docs/05-training-loop.md 和 docs/08-eval-metrics.md 即可。



------





## **docs/05-training-loop.md**



```
# 05 · 训练循环拆解（PPO · 结合源码与张量形状）

> 目标：把 VERL 的 PPO 主链路“从数据到更新”吃透。我们用**贴近源码的伪代码** + **张量形状**讲清楚：
> 数据 → Rollout 采样 → 奖励 → 优势/回报 → PPO 更新（actor/ref/critic）→ 日志与保存。
> 文中函数名均为“搜索建议”，不同版本位置略异，请用 IDE 全局搜索定位。

---

## 0）全景：一图看懂数据流

**一次完整迭代：**
1. 从 `train.parquet` 取一个 batch（`prompt` + 可能的 `ground_truth` 等）。
2. **Rollout 引擎**（vLLM / SGLang）基于 `prompt` 生成 `response`，同时拿到 **actor/ref 的对数似然**。
3. **RewardManager** 调用你在 04/06/07 注册的 `compute_score(...) -> [0,1]`，得到**样本级奖励**。
4. 把样本级奖励铺到**token 级**（按 mask），用 **GAE(λ)** 得到**优势**与**回报**。
5. 做 **PPO 更新**：clip 策略损失 + 值函数损失（+ 可选熵/KL 正则）。
6. 记录指标（reward/kl/loss/SPS）——若到 `save_freq`/`test_freq` 则保存/评测。

---

## 1）Rollout 采样（单轮 vs 多轮）与关键张量

**形状约定（常见实现）：**
- `B`：batch size（语义批大小）
- `T`：生成序列上限（`data.max_response_length`）
- `V`：词表大小

**典型输出：**
- `responses: List[str]`（长度 B，文本）
- `resp_ids: LongTensor[B, T]`（pad 到 T）
- `attn_mask: BoolTensor[B, T]`（响应有效 token 的 mask）
- `logp_actor: FloatTensor[B, T]`（策略在响应 token 上的对数似然）
- `logp_ref:   FloatTensor[B, T]`（参考模型的对数似然；用于 KL）
- `values:     FloatTensor[B, T]`（critic 估计的 V 值；多实现也会先聚合成样本级）

**单轮（vLLM）**：一次 `prompt -> response`。  
**多轮（SGLang）**：SGLang 会把**上一轮对话历史**（含工具返回）自动拼进当前轮的输入；**只对“本轮新生成的助手 tokens”计损失**（关键！）。

> 搜索建议：`build_rollout_engine`, `generate`, `compute_logprob`, `multi_turn`.

---

## 2）奖励：样本级 → token 级（mask 铺展）

你的 `compute_score(data_source, solution_str, ground_truth, extra)->float` 返回 **[0,1]** 的样本分。  
PPO 需要**每个 token**的 advantage，因此常见做法是把样本分 `r_i` 铺到该样本有效 token 上：

```python
# 伪代码（张量化）
# 输入：sample_reward: FloatTensor[B] in [0,1]; attn_mask: [B,T] (仅针对“响应部分”)
token_reward = sample_reward[:, None].expand(-1, T) * attn_mask  # [B,T]
```

> 也有实现会做“**末 token 收全奖、其余 0**”或“**线性衰减**”。核心是**与 attn_mask 对齐**，不要给 pad token 奖励。



------





## **3）优势与回报：GAE(λ)（样本级/序列级两种粒度）**





**定义（按 token 粒度）**



- 即时回报 r_t：上一步铺展后的 token_reward[b, t]
- 价值 V(s_t)：来自 critic 的 values[b, t]（同维度）
- TD 误差：δ_t = r_t + γ V_{t+1} - V_t
- GAE：A_t = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...





常见实现会把**同一响应的所有 token 的优势**再**聚合/平均到样本级**（adv_sample = (A_t ⊙ mask).sum / mask.sum），以减小噪声；也有保持 token 级的实现。

```
# 伪代码：沿时间维反向递推
def compute_gae(r, v, mask, gamma=0.99, lam=0.95):
    # r, v, mask: [B,T]
    T = r.size(1)
    A = torch.zeros_like(r)
    gae = torch.zeros_like(r[:, 0])
    for t in reversed(range(T)):
        not_last = (t < T-1).float()
        v_next = v[:, t+1] * not_last
        delta = r[:, t] + gamma * v_next - v[:, t]
        gae = delta + gamma * lam * gae * not_last
        A[:, t] = gae
    A = A * mask  # 掩掉无效位
    # 样本级聚合（可选）
    A_sample = (A.sum(dim=1) / (mask.sum(dim=1) + 1e-8))  # [B]
    return A, A_sample
```

**回报（target for critic）**：R_t = A_t + V_t（或样本级对应）。



> 搜索建议：gae, estimate_advantage, returns_targets, masking.



------





## **4）KL 两种接法（别“双算惩罚”）**





- **KL in loss（常见）**：

  L_actor = - E[ min(r_t A_t, clip(r_t,1±ε) A_t) ] + β * KL(actor||ref)

- **KL in reward**：

  在奖励端做 r'_t = r_t - β * KL_t，然后像普通奖励一样走 GAE。





**注意**：二者**二选一**，否则会把 KL 惩罚叠加两次，训练会“走不动”。

```
if use_kl_in_reward:
    reward_token = reward_token - kl_coef * (logp_actor - logp_ref).exp().log()  # ≈ KL 局部项
else:
    kl_term = kl_coef * (logp_actor - logp_ref)  # 用于损失
```

> 搜索建议：kl_ctrl, use_kl_in_reward, kl_coef_schedule.



------





## **5）PPO 更新（actor / critic）**





**比值**：r_t = exp(logp_actor_new - logp_actor_old)

**剪切**：clip(r_t, 1-ε, 1+ε)

**策略损失**：L_π = - E[ min(r_t * A_t, clip(r_t)*A_t) ]

**值函数损失**：L_V = E[ (V_θ - R)^2 ]

**总损失**：L = L_π + c1 * L_V (+ c2 * entropy)



张量实现（示意，样本级优势）：

```
def ppo_step(batch):
    # 重新前向 → 新 logp_actor_new / values_new
    logp_new, values_new = model.forward_logp_value(batch.input_ids, batch.attn_mask, batch.resp_ids)
    ratio = torch.exp(logp_new_sum_per_sample - batch.logp_old_sum_per_sample)  # 样本级
    surr1 = ratio * batch.adv_sample
    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * batch.adv_sample
    loss_actor = -torch.mean(torch.min(surr1, surr2))
    loss_v = F.mse_loss(values_new_sample, batch.returns_sample)
    loss = loss_actor + c1*loss_v
    if not use_kl_in_reward:
        loss = loss + kl_coef * torch.mean(batch.kl_sample)  # 来自 old actor/ref
    loss.backward()
```

**mini/micro-batch**：



- **micro-batch**：**显存**维度（一次能塞多少样本/序列）
- **mini-batch**：**优化**维度（一次 optimizer.step() 之前用多少样本做反传）
- VERL 常见做法：将 train_batch_size 切成若干 **mini-batch**，每个 mini 再切成多个 **micro-batch** 做梯度累积。





> 搜索建议：ppo_mini_batch_size, ppo_micro_batch_size_per_gpu, grad_accum.



------





## **6）多轮细节：只训练“本轮助手输出”**





- 构造 **response mask**：仅覆盖“本轮 assistant 的输出 tokens”。
- 旧轮历史拼入上下文，但 mask=0，**不参与损失**。
- 工具返回文本（如 TOOL[...]）也作为历史的一部分，由 SGLang 拼接。





> 搜索建议：multi_turn, delta tokenization, response_mask.



------





## **7）数值与稳定性：9 条硬规则**





1. **奖励范围**尽量归一到 [0,1]。
2. **mask everywhere**：loss/adv/kl 计算都要乘有效位。
3. epsilon、1e-8 该加就加，避免 log(0) 与除零。
4. 初期 reward 噪声大 → **加大 KL**、**减小 LR**、**增大 mini-batch**。
5. **只留主奖**先跑稳，再加格式/证据等 shaping。
6. micro-batch 先设 1，能跑再升；语料长就缩 max_prompt/response_length。
7. 日志里同时看 **reward/kl/loss/SPS** 的变化趋势，不要只盯一个指标。
8. Clip ratio（ε）默认 0.2，**不收敛可以降到 0.1** 看看。
9. 值函数爆炸（loss_v 巨大）→ 降 critic.lr，或给 value clip。





------





## **8）你在源码里该看什么（Checklist）**





- **Rollout**：generate() 返回了哪些张量？logprob 是如何对齐到响应 token 的？
- **Reward**：non_tensor_batch 中你的 ground_truth 等字段如何传入？
- **Advantage**：按 token 还是按样本？在哪里做了聚合？
- **KL**：走 reward 端还是 loss 端？有没有系数调度？
- **Mask**：哪些地方用了 response mask？（策略/值/kl 都要遮掉 pad 与历史）
- **Batching**：mini / micro 的两层循环在哪里？如何做梯度累积？
- **Logging**：哪些指标被写到 TB/CSV？





------





## **9）最小单元测试（建议你加到仓库）**



```
def test_mask_shapes():
    B,T = 3,7
    mask = torch.tensor([[1,1,1,0,0,0,0],
                         [1,1,0,0,0,0,0],
                         [1,1,1,1,1,0,0]], dtype=torch.float32)
    r = torch.rand(B,T); v = torch.zeros(B,T)
    A, As = compute_gae(r, v, mask)
    assert A.shape==(B,T) and As.shape==(B,)
    assert (A[mask==0]==0).all()
```

> 把这类 shape/遮罩单测写起来，你会省掉大量“莫名其妙 reward 不升”的时间。