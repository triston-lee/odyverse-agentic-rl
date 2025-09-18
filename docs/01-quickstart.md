# 01 · 快速上手（单卡 PPO）

> 目标：**5 分钟**把 VERL 跑起来（最小 PPO），看到日志与 checkpoint。
>

## 0）准备（二选一）

**A. 官方 Docker（推荐）**

```bash
docker create --runtime=nvidia --gpus all --net=host --shm-size=10g \
  -v $PWD:/workspace/verl --name verl \
  verlai/verl:app-latest sleep infinity
docker start verl && docker exec -it verl bash

# 容器内
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
```

**B. 本地环境（你熟悉包管理再用）**

- Python ≥ 3.10，CUDA ≥ 12.x，PyTorch & vLLM/SGLang 需和 CUDA 匹配
- 安装：pip install -e .（在 VERL 源码目录）

> 显存紧张时：把下面所有 *_micro_batch_size_per_gpu 先设为 1。

## 1）准备数据（GSM8K 或最小玩具集）

**GSM8K（官方示例）**

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k
```

**或：极小玩具集（10 条，快速验证）**

```bash
python3 - <<'PY'
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, os
os.makedirs(os.path.expanduser("~/data/toy"), exist_ok=True)
rows=[{"prompt":f"Say hello #{i} in JSON: {{\"msg\": string}}", "label":"hello"} for i in range(10)]
tbl=pa.Table.from_pandas(pd.DataFrame(rows))
pq.write_table(tbl, os.path.expanduser("~/data/toy/train.parquet"))
pq.write_table(tbl, os.path.expanduser("~/data/toy/val.parquet"))
print("toy data -> ~/data/toy/{train,val}.parquet")
PY
```

## 2）一键开训（PPO · 单卡）

> 默认用极小模型 Qwen/Qwen2.5-0.5B-Instruct；你也可以换成任意可加载的指令模型。
>
> 奖励函数位置位于[gsm8k.py](..%2Fverl%2Futils%2Freward_score%2Fgsm8k.py)

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \	# 
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \ # Actor 初始等于 Qwen2.5-0.5B-Instruct
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_quickstart.log
```

**显存不够？**把两处 *_micro_batch_size_per_gpu 调成 1；或把 gpu_memory_utilization 降到 0.3。

1. 读数据（Parquet）
   - **data.train_files**, **data.val_files** 指向你前面预处理好的 GSM8k parquet。每条样本里有：
     - prompt: 用户消息（问题 + “Let’s think step by step … ####” 提示）
     - reward_model: {"style": "rule", "ground_truth": <提取出的正确数值>}
   - **data.train_batch_size=256**：每次 PPO 更新要采样的样本数（一次 rollouts 的规模）。
   - **data.max_prompt_length=512** / **data.max_response_length=256**：分别裁剪输入与生成长度；直接影响显存与吞吐。

2. 模型三件套

   - **actor_rollout_ref.model.path = Qwen2.5-0.5B-Instruct**
     - Actor：要被优化的策略；
     - Rollout：实际用来生成数据的推理副本（由 vLLM 驱动）；
     - Ref：冻结的初始策略（只前向不训练）用于 KL。

3. 优化与批次粒度（容易搞混的三层批次）

   - **data.train_batch_size=256**（rollout 批次）：一次 PPO 更新收集 256 个样本的生成与奖励。

   - **actor_rollout_ref.actor.ppo_mini_batch_size=64**（PPO 小批次）：把 256 切成 4 个小批次做梯度更新（每小批次一次优化步）。

   - **actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4**（显存微批）：再把每个 小批次=64 拆成显存可承受的微批，在 GPU 上 梯度累积：

   > - 现在是 trainer.n_gpus_per_node=1；
   >
   > - 每小批需要的累积步数 = 64 / (1 GPU × 4/微批) = 16 次 累积 → 再做一次 optimizer step。

   - 其他的微批设置（vLLM的前向传播也用微批，Critic 与 Ref 的前向/反向同理做微批）：
     -  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
     - critic.ppo_micro_batch_size_per_gpu=4
     - actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2

   - 学习率：Actor 1e-6（更谨慎，防“破坏”语言能力），Critic 1e-5（值函数一般学得更快）。
     - *Actor 的学习率1e-6 是保守安全的起点，若 reward 长期 0，可以小幅上调（例如 2e-6）。*

4. KL 系数

   - **algorithm.kl_ctrl.kl_coef=0.001**：越大越“保守”（贴近初始策略），越小越“敢改”（更可能探索但不稳定）。
     - *如果发现语言风格“飘”了、答非所问，适当加大 kl_coef；若一直学不动，适当减小。*

   - 在 稀疏奖励（对错 0/1）场景下，KL 是非常重要的稳定器。

5. 推理引擎（vLLM）占用

   - **gpu_memory_utilization=0.4**：vLLM 占显存的配额，剩余显存留给训练中的 Actor/Critic。
     - 单卡 (**trainer.n_gpus_per_node=1**) 下，这个值过大或过小都会影响吞吐/稳定性；0.3~0.6 是常见起点。

6. 其余参数

   - trainer.val_before_train=False：跳过训练前的初始验证，直接开跑。

   - trainer.nnodes=1, trainer.n_gpus_per_node=1：单机单卡。

   - trainer.logger=console：控制台日志；

   - Shell：

     - PYTHONUNBUFFERED=1：Python 标准输出不缓存，日志实时刷。

     - 2>&1 | tee verl_quickstart.log：stdout/stderr 合流并写入文件。

## 3）结果在哪

- 日志：verl_quickstart.log（控制台也会持续打印）
- checkpoint：checkpoints/<project>/<experiment>/.../actor
- （可选）合并到 Hugging Face 目录（便于推理服务）：

```bash
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/<project>/<experiment>/global_step_1/actor \
  --target_dir checkpoints/<project>/<experiment>/global_step_1/actor/hf
```

## 附录

1. GSM8k相关资料

   - Paper: https://arxiv.org/pdf/2110.14168

   - Dataset: https://huggingface.co/datasets/gsm8k