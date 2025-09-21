太好了！这轮给你连上 **09「多机扩展（Ray / SkyPilot）」** 和 **10「吞吐与显存优化」** 两篇，都是**可直接放进 docs/** 的成稿，尽量结合实现与落地操作。依然沿用你前面已经写好的 AIOps 任务与 SGLang 多轮场景。



------





## **docs/09-scaling-distributed.md**



```
# 09 · 多机扩展（Ray / SkyPilot / 本地多卡）

> 目标：把第 06/07 篇的多轮 PPO 从**单卡**扩到**单机多卡**甚至**多机**；并给出“怎么量化吞吐（SPS）”与“常见网络/NCCL 坑”的成体系做法。

---

## 0）扩展思路（先打通再提速）

- **优先级**：先单卡能稳训 → 单机多卡（DDP）→ 多机（DDP + TP/服务并行）。  
- **并发来源**：  
  1) **采样并发**（rollout 后端：vLLM/SGLang 的内部并行/TP）  
  2) **更新并行**（PPO 的 DDP：`trainer.n_gpus_per_node × trainer.nnodes`）  
- **吞吐指标**：Samples Per Second（SPS），用**训练循环**里采样/更新的计时统一统计，避免只看后端 tokens/s。

---

## 1）单机多卡（最小命令）

> 适用：一台 4–8 卡机器。核心是把 `trainer.n_gpus_per_node` 调高，保持其余逻辑不变。

```bash
python -m verl.trainer.main_ppo \
  data.train_files=$HOME/data/aiops_mturn/train.parquet \
  data.val_files=$HOME/data/aiops_mturn/val.parquet \
  data.prompt_key=prompt \
  custom_reward_function.path=examples/reward_score/aiops_mturn.py \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.multi_turn=true \
  # --- 单机多卡关键 --- #
  trainer.n_gpus_per_node=4 trainer.nnodes=1 \
  # --- 显存安全做法：先小 micro，再放大 --- #
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  data.train_batch_size=512 \
  trainer.total_epochs=3
```

**解释**



- DDP 由框架内部用 torchrun 或等价器起进程；你只需改 trainer.n_gpus_per_node。
- **吞吐先不追求极限**：微批设 1，跑通后再拉到 2/4，或加大 ppo_mini_batch_size。





------





## **2）本地 + vLLM 的 TP（单机张量并行）**





> 如果你临时想把 rollout 换回 vLLM 跑单轮/对比吞吐，可用 TP 撑大模型或提高并行：

```
# 仅供参考：单轮/函数式奖励的 vLLM 跑法
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
```

**要点**



- TP>1 会占用多卡；SGLang 的多轮一般不需要你手动开 TP（保持 1 即可），重点是 DDP 的扩展。





------





## **3）多机（DDP，最小两节点）**







### **3.1 端口与主机地址**





- 选一个**主节点**（rank 0）：MASTER_ADDR=ip.of.head，MASTER_PORT=29500
- 所有节点同步 trainer.nnodes=2、各自 node_rank（多数框架会从环境里自动获取）
- 打开防火墙端口或内网安全组（至少 29500）







### **3.2 环境变量（NCCL 常见设置）**



```
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1        # 没有 IB/RDMA 的机器上建议关掉
export NCCL_SOCKET_IFNAME=eth0  # 或者你的内网网卡名
export CUDA_LAUNCH_BLOCKING=0
```



### **3.3 训练命令（各节点一致）**



```
MASTER_ADDR=10.0.0.10 MASTER_PORT=29500 \
python -m verl.trainer.main_ppo \
  ... \
  trainer.n_gpus_per_node=4 trainer.nnodes=2 \
  # （如需）固定随机端口避免冲突
  trainer.torchrun_master_addr=$MASTER_ADDR \
  trainer.torchrun_master_port=$MASTER_PORT
```

> 你的集群管理方式不同，可能由调度器自动注入 MASTER_ADDR/MASTER_PORT，这里给手动模式以方便本地/云机直连测试。



------





## **4）用 Ray 起一个“并行采样器”（本地/多机通吃）**





> 场景：多轮+工具时，**rollout 采样**成为瓶颈。可以用 Ray 起若干**采样 Actor**，每个 Actor 内部走 SGLang，主进程收集完成 PPO 更新。





### **4.1 Ray 本地启动（单机）**



```
ray start --head --port=6379
# 校验
python - <<'PY'
import ray; ray.init(address="auto"); print(ray.cluster_resources())
PY
```



### **4.2 最小采样 Actor（教学版骨架）**





distributed/ray_sampler.py

```
import ray, time
@ray.remote(num_gpus=1)
class Sampler:
    def __init__(self, cfg):
        self.cfg = cfg
        # 在这里初始化 SGLang/vLLM 客户端、tokenizer 等
    def sample(self, batch):
        # 这里调用 rollout 后端生成，并返回 {responses, logp_actor, logp_ref, values}
        return {"responses":[...], "logp_actor":..., "logp_ref":..., "values":...}

def map_sample(cfg, batches, num_actors=2):
    actors=[Sampler.options(name=f"sampler_{i}").remote(cfg) for i in range(num_actors)]
    futs=[actors[i%num_actors].sample.remote(b) for i,b in enumerate(batches)]
    return ray.get(futs)
```

**接入 Trainer**：在你的训练循环中，把原先“单进程 rollout”的那一段替换成 map_sample。

**收益**：采样与更新**并行**，特别是工具 I/O 很慢时，吞吐可明显提升。



> Ray 多机：在 head 节点 ray start --head，worker 节点用 ray start --address='head:6379' 加入；确保节点 GPU 可见且驱动一致。



------





## **5）SkyPilot：一键把任务提交到云（最小模板）**





> SkyPilot 适合“我只要一条命令把训练扔到云上”的场景。下面模板以 1×A100 为例；如需要多机，把 num_nodes 拉高，或在多任务并行上做阵列。



sky-verl.yaml

```
name: verl-ppo-aiops

resources:
  accelerators: A100:1
  disk_size: 200
  ports: [29500, 6379]   # torchrun / ray

envs:
  EXP_NAME: aiops-mturn

setup: |
  conda create -y -n verl python=3.10
  source activate verl
  git clone https://github.com/volcengine/verl
  cd verl && pip install --no-deps -e .
  # 你的项目代码/数据同步（可用 sky storage）

run: |
  source activate verl
  python -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/aiops_mturn/train.parquet \
    data.val_files=$HOME/data/aiops_mturn/val.parquet \
    data.prompt_key=prompt \
    custom_reward_function.path=examples/reward_score/aiops_mturn.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn=true \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.total_epochs=3
```

提交：

```
sky launch -c verl-a100 sky-verl.yaml
```



------





## **6）如何量化“吞吐”（SPS）**







### **6.1 训练端：记录采样/更新用时**





在训练循环中加入（或确保已有）：



- time_rollout：采样 + 计算 logprob
- time_update：PPO 更新
- samples_processed：本 step 的样本数（B）





SPS 计算：

```
SPS = samples_processed / (time_rollout + time_update)
```



### **6.2 小脚本从日志抓取（示例）**





scripts/parse_sps.py

```
import re, sys
ts=0;s=0
for line in open(sys.argv[1],"r",encoding="utf-8"):
    m=re.search(r"SPS=(\d+(\.\d+)?)", line)
    if m: ts+=float(m.group(1)); s+=1
print("avg_SPS", ts/max(s,1))
```



------





## **7）常见坑（Checklist）**





- **端口**：29500（DDP）、6379（Ray）；被占用 or 防火墙未放行 → 训练挂起。
- **NCCL**：没有 IB 的机型记得 NCCL_IB_DISABLE=1；否则卡在 init。
- **驱动/容器**：CUDA/驱动/容器 CUDA Runtime 不一致 → vLLM/SGLang 崩。
- **时间不同步**：多机日志时间差过大，建议开启 NTP。
- **数据路径**：多机各节点读同一个绝对路径（或用对象存储/网络盘）。
- **显存碎片**：频繁 OOM → 降 gpu_memory_utilization、缩 max_*_length、重启进程清碎片。





------





## **8）建议的扩展顺序**





1. 单机 2–4 卡 DDP（微批=1 跑通）
2. 把微批提到 2/4、提高 train_batch_size
3. 引入 Ray 并行采样（多轮+工具用处最大）
4. 多机 DDP（1–2 台先打通，再扩）
5. 云上 SkyPilot（把数据/脚本打包、参数固化）

