## 配置与超参速查

> 这一页是你“改参数就翻”的备忘单。所有键名都来自前面我们跑通 01/04/06/07 的命令行；不同版本可能有细微差别，但思路一致：能被 CLI 覆盖的层级键，都可以在命令里 `a.b.c=value` 覆盖默认配置。

### 1 最小需要学会的配置

#### 1.1 基础模块配置

1. **数据与采样（Data & Prompt）**

| 模块 | 键 | 作用 | 典型值 / 建议 |
|---|---|---|---|
| `data` | `data.train_files` / `data.val_files` | 指定训练 / 验证用的 parquet 文件路径；可多个文件 | 使用绝对路径，多文件用逗号分隔 |
| `data` | `data.prompt_key` | 数据中负责 “prompt” 的列名 | 默认为 `prompt`，如列名不同要改 |
| `data` | `data.train_batch_size` | 全局训练 batch 大小（样本数） | 通常设为 **128–2048**，平衡吞吐量、显存与训练稳定性 |
| `data` | `data.max_prompt_length` | Prompt 输入长度限制（截断） | 建议 **256–1024**，越长显存越吃紧 |
| `data` | `data.max_response_length` | 模型生成输出最大长度 | 建议 **64–512**，过大可能降低奖励密度 |

2. **策略 Actor / 价值 Critic 与 PPO**

| 模块 | 键                                                      | 作用 | 典型值 / 建议 |
|---|--------------------------------------------------------|---|---|
| `model` | `actor_rollout_ref.model.path`                         | 指令模型路径或 HF 模型名称 | 初期可选 **0.5B-3B** 模型验通路后，再上大模型 |
| `actor` | `actor_rollout_ref.actor.optim.lr`                     | 策略网络学习率 | 建议 **1e-6 ~ 5e-6**（小模型可稍大） |
| `actor` | **`actor_rollout_ref.actor.ppo_mini_batch_size`**       | PPO 的小批量训练大小 | **32–256**，设大能更稳定但训练慢 |
| `actor` | **`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`** | actor 的单卡微批大小 | 若出现 OOM，首先调这个小 |
| `critic` | `critic.model.path`                                    | 价值网络模型路径 | 通常与 actor 使用相同或匹配的基础模型 |
| `critic` | `critic.optim.lr`                                      | Critic 的学习率 | 常设约 **1e-5** |
| `critic` | **`critic.ppo_micro_batch_size_per_gpu`**                  | Critic 的单卡微批大小 | 与 actor 微批大小保持协调 |

3. **Rollout 与并行（Rollout Backend / Parallelism）**

| 模块 | 键 | 作用                                    | 典型值 / 建议                                  |
|---|---|---------------------------------------|-------------------------------------------|
| `rollout` | `actor_rollout_ref.rollout.name` | 指定后端：`vllm` 或 `sglang`                | **单轮任务** 用 `vllm`，**多轮交互**或工具链推荐 `sglang` |
| `rollout` | `actor_rollout_ref.rollout.multi_turn` | 是否开启多轮交互                              | 多轮任务设 `true`                              |
| `rollout` | `actor_rollout_ref.rollout.gpu_memory_utilization` | 控制生成期间显存占用比率                          | 建议设 **0.3–0.8**，显存不够时调小                  |
| `rollout` | `actor_rollout_ref.rollout.tensor_model_parallel_size` | 张量并行（TP）度，即单层参数切分到多少卡                 | 小模型设 `1`，大模型 / 多卡设 `2, 4, 8…`           |
| `rollout` | **`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`** | 计算 log_prob 的单卡微批大小 |  与 actor 微批大小保持协调                         |

4. **参考 / LogProb 模型（Ref）**

| 模块 | 键 | 作用 | 典型值 / 建议 |
|---|---|---|---|
| `ref` | **`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`** | log-prob 计算时，单卡微批大小 | 与 actor/critic 微批协调配合，显存紧张就调小 |

5. **KL 控制与奖励结合（KL & Reward）**

| 模块 | 键 | 作用 | 典型值 / 建议 |
|---|---|---|---|
| `algo` | `algorithm.kl_ctrl.kl_coef` | KL 惩罚系数，用于限制策略与参考模型偏差 | 常见范围 **1e-4 ~ 1e-2**，若训练不稳定或策略漂移大则调高 |
| `algo` | `algorithm.use_kl_in_reward` | 是否把 KL 整合到 reward 中 | 与 `use_kl_loss` 二选一，避免重复惩罚 |

6. **训练控制 (Trainer) 与工具集 (Tools / Reward)**

| 模块 | 键 | 作用 | 典型值 / 建议 |
|---|---|---|---|
| `trainer` | `trainer.total_epochs` | 总训练轮数（遍历数据集次数） | 初期用 **2-5** 轮，稳定后可适度增加 |
| `reward` | `custom_reward_function.path` / `name` | 自定义奖励函数注入路径或标识 | 按需注入，重要任务可用复杂奖励函数 |
| `tools` | `actor_rollout_ref.rollout.tool_kwargs.tools_config_file` | 工具链配置文件（YAML）路径 | 多轮交互 / 工具调用场景常用 |

#### 1.2  理解批大小（各种 Batch Size）

可以看到，整个配置参数中，有非常多BatchSize参数，包括 batch_size / mini_batch_size / micro_batch_size，其次各模块（Actor / Critic / Rollout / Ref）中这些批次参数的用途也不一样

1. **batch_size / mini_batch_size / micro_batch_size_per_gpu**

| 名称 | 含义 / 用途 | 视角 | 官方说明 / 注意事项 |
|---|------------------------------|----------------------------------------|---------------------------|
| `train_batch_size` | 全局训练样本数：一次训练更新 /采样阶段处理的总 prompt 数量 | Global | 决定总体样本效率，太大容易显存或通信瓶颈。  [ref](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html?utm_source=chatgpt.com) |
| `ppo_mini_batch_size` | PPO 更新阶段，把 `train_batch_size` 或采样到的数据拆成几组小 batch 来做多轮更新 | Global | mini_batch 的拆分用于梯度稳定性与训练效率，多个 worker /GPU 共享该全局拆分。  [ref](https://verl.readthedocs.io/en/latest/examples/config.html?utm_source=chatgpt.com) |
| `*_micro_batch_size_per_gpu` | 每张 GPU 在一次前向 /反向 /log_prob /rollout 等操作中实际处理的子批量 | Local<br/>Per GPU | 用来控制显存占用与吞吐效率，是PERFORMANCE-related parameter，官方建议用这个而非旧的 `micro_batch_size`。  [ref](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html?utm_source=chatgpt.com) |

2. 各模块（Actor / Critic / Rollout / Ref）中这些批次参数的用途区分

| 模块 | 主要任务 | 用到的批次参数类型                                                                                                  | 建议用途 /典型设定对比 |
|---|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| **Actor** | PPO 策略训练：包含前向 + 反向 + KL 损失 +策略梯度等 | `train_batch_size`（全局）<br/>`ppo_mini_batch_size`（全局拆分更新用）<br>`ppo_micro_batch_size_per_gpu`（控制每 GPU 的前向/反向子批） | 这个模块主要是反向，通常算力 & 显存负担最大。micro_batch_per_gpu 要设得不会 OOM，一般都先小后大，mini_batch_size 拆分要平衡稳定与速度。  [ref](https://verl.readthedocs.io/en/latest/examples/config.html?utm_source=chatgpt.com) |
| **Critic** | Value 网络训练估值 / 输出标量 /通常反向传播也有但开销较小 | `critic.ppo_micro_batch_size_per_gpu` 或类似前向 +反向微批 /全局 mini batch 拆分                                        | 因为输出维度小、softmax等计算少，可以设比 actor 微批稍大，以提高吞吐率。  [ref](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html?utm_source=chatgpt.com) |
| **Rollout** | 采样 /生成 /推理 /计算 log_prob（前向为主） | `rollout.log_prob_micro_batch_size_per_gpu` 等前向微批 + 借助 `train_batch_size` 与 rollout 中的采样 N（response 数）     | Rollout 阶段通常是纯前向，开销比训练更新小，可以设更大 micro_batch；但要考虑 prompt/response 长度 + GPU 并行度 + 内存消耗。  [ref](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html?utm_source=chatgpt.com) |
| **Ref** | 参考模型 /baseline /计算 ref_log_prob 或 KL 对比 /通常不训练 | `ref.log_prob_micro_batch_size_per_gpu`（前向）                                                                | Ref 模型同样也是前向，可设较大 batch 来减少推理时间；但如果模型 very large 或 TP/PP 并行度高，也要注意显存限制。  [ref](https://verl.readthedocs.io/en/latest/examples/config.html?utm_source=chatgpt.com) |

### 2 具体配置解析

#### 2.1 数据（data.*）

```yaml
data:
  # 路径 & 基本字段
  tokenizer: null                    # 可不填；与 RM/Actor 的模板不一致时再指定
  train_files: /abs/path/train.parquet
  val_files:   /abs/path/val.parquet
  prompt_key: prompt

  # 序列长度（吞吐&显存第一控制器）
  max_prompt_length: 512
  max_response_length: 256

  # 批大小（全局语义批）
  train_batch_size: 512

  # 训练/加载行为
  shuffle: true
  filter_overlong_prompts: false     # 超长报错 or 过滤，可配 truncation
  filter_overlong_prompts_workers: 1
  truncation: error                  # 可选：left/right/middle/error
  return_raw_input_ids: false        # 当 Policy 与 RM 的 chat template 不一致时置 true
  return_raw_chat: false
  return_full_prompt: false
  trust_remote_code: true
```

- max_prompt_length/max_response_length 决定**序列长度**，直接影响显存与吞吐；RL rollout 会按该上限生成。 
- train_batch_size 是**一次采样/训练迭代的全局样本数**（“语义批”），并不等同于每卡实际装载量；后者由 *_micro_batch_size_per_gpu 决定。 
- truncation 支持 left/right/middle/error；middle 会保留首尾、丢中间。 

#### 2.2 三元组与后端（actor_rollout_ref.\*）

##### 2.2.1 模型

```yaml
actor_rollout_ref:
  hybrid_engine: true                # 混合引擎（Actor/Rollout/Ref 协同），默认开启
  model:
    path: Qwen/Qwen2.5-0.5B-Instruct # 仅需改 path 即可切模
    external_libs: null
    override_config:
      model_config: {}
    enable_gradient_checkpointing: false
    enable_activation_offload: false
    trust_remote_code: false
    use_remove_padding: false        # 去 padding 能显著提效（HF/FSDP 路径）
```

- path 支持本地或远端（HDFS/Hub）；首次建议 0.5B–3B 把流程跑稳。 
- use_remove_padding 能明显提升算子利用率（HF/FSDP）。 

##### 2.2.2 Rollout（推理/采样引擎）

```yaml
actor_rollout_ref:
  rollout:
    name: vllm                     # vllm / sglang / hf
    multi_turn: false              # 多轮任务改 true（且配合 name: sglang）
    # 采样参数（对齐 vLLM SamplingParams；sglang/hf 也可按需传）
    temperature: 1.0
    top_k: -1                      # vLLM: -1 表示禁用；HF: 0
    top_p: 1.0
    n: 1                           # 每个 prompt 生成条数；GRPO/RLOO 可设 >1
    do_sample: true
    calculate_log_probs: false     # 需要重算 logprob 时置 true

    # 长度控制（与 data.* 对齐）
    prompt_length:   ${data.max_prompt_length}   # opensource 路径下通常忽略
    response_length: ${data.max_response_length}

    # vLLM/SGLang 专属参数
    dtype: bfloat16                 # 建议与 FSDP 模型 dtype 对齐
    gpu_memory_utilization: 0.5     # vLLM: 占总显存比例；SGLang: mem_fraction_static
    tensor_model_parallel_size: 1    # 仅 vLLM 生效；>1 时做 TP 切分
    max_num_batched_tokens: 8192
    max_num_seqs: 1024
    free_cache_engine: true          # 推理后释放 KVCache
    enforce_eager: true              # 关闭 CUDA Graph（vLLM 某版本必要）
    load_format: dummy_dtensor       # FSDP: dtensor/hf/dummy_* 三选

    # 引擎透传（按官方文档传递）
    engine_kwargs:
      vllm: {}
      sglang: {}

    # 验证集的采样参数（尽量确定性）
    val_kwargs:
      top_k: -1
      top_p: 1.0
      temperature: 0
      n: 1
      do_sample: false

    # logprob 重算的微批（每卡）
    log_prob_micro_batch_size_per_gpu: 16
    log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
    log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
```

- **多轮**：multi_turn: true 且 name: sglang；SGLang 是多轮/工具首选。 
- gpu_memory_utilization：vLLM≥0.7 表示“使用总显存的比例”；SGLang 对应静态内存占比 mem_fraction_static。 
- tensor_model_parallel_size（TP）**仅对 vLLM 生效**；TP>1 用于大模型/多卡。 
- free_cache_engine/enforce_eager/load_format：与权重同步与 vLLM 行为强相关，按官方建议搭配。

##### 2.2.3 Ref（参考模型，用于 KL / logprob）

```yaml
actor_rollout_ref:
  ref:
    # FSDP 与 Actor 对齐的配置（可选）
    fsdp_config:
      param_offload: false
      wrap_policy:
        min_num_params: 0
    # logprob 仅前向，可设大些
    log_prob_micro_batch_size_per_gpu: 16
    log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
    log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
    ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size}
```

- _per_gpu 是**本地每卡**的微批；旧字段（非 per_gpu）已标注弃用。 
- Ref 只做**前向**，可比训练阶段更大胆地放大微批。 

#### 2.3 优化器与批大小

##### 2.3.1 Actor（策略）参数详解（actor.\*）

```yaml
actor_rollout_ref:
  actor:
    # PPO 批次层级
    ppo_mini_batch_size: 64          # 全局 mini-batch（用于PPO更新时的拆分）
    ppo_micro_batch_size_per_gpu: 2  # 每卡微批（OOM 首先调小它）
    use_dynamic_bsz: false           # 按需动态调小以避 OOM
    ppo_max_token_len_per_gpu: 16384 # 每卡最大 token（PPO 批），用于硬限制，估算: n * max_prompt + max_response
 
    # PPO/损失
    ppo_epochs: 1                    # 每批样本重复多少轮更新
    clip_ratio: 0.2                  # PPO clip 范围
    entropy_coeff: 0.0               # 自 v0.3.x 默认 0.0
    grad_clip: 1.0                   # 全局梯度裁剪
 
    # KL 选项（二选一：要么走loss，要么走reward）
    use_kl_loss: false               # 默认走Reward端，若走loss：在Actor端加KL
    kl_loss_coef: 0.001
    kl_loss_type: kl                 # kl/abs/mse/low_var_kl/full（支持带“+”直通梯度）
    
    # 训练后端
    strategy: fsdp2                  # 推荐 FSDP2（主干维护）
    use_torch_compile: true

    # 优化器与调度
    optim:
      lr: 1e-6
      total_training_steps: -1       # 由程序注入
      warmup_style: constant         # or linear/cosine 等（Megatron命名略有差异）
      lr_warmup_steps: -1            # <0 走 ratio
      lr_warmup_steps_ratio: 0.0
      
    # 并行/内存路径
    strategy: fsdp2                  # 官方推荐 FSDP2（相对FSDP1显存↓、吞吐↑）
    fsdp_config:
      param_offload: false
      optimizer_offload: false
      wrap_policy:
        min_num_params: 0

    # Checkpoint
    checkpoint:
      save_contents: ['model', 'optimizer', 'extra']
      load_contents: ${actor_rollout_ref.actor.checkpoint.save_contents}
```

- **批层级**：mini-batch（全局）用于把一次采样得到的轨迹批再拆小做多次更新；**micro-batch（每卡）** 控制单次前/反向的装载量（直接对应显存）。

- **PPO 核心超参**：ppo_epochs、clip_ratio、entropy_coeff、grad_clip 均在官方“[PPO 配置](https://verl.readthedocs.io/en/latest/examples/config.html)”处有定义（默认/建议）。 

- **KL 双通道，二选一**：

  - 走 **loss**：actor.use_kl_loss=true（再配 kl_loss_coef/kl_loss_type）。

  - 走 **reward**：algorithm.use_kl_in_reward=true。

    **不要双开**，[官方文档](https://verl.readthedocs.io/en/latest/algo/ppo.html)明确指出两者是互斥设计。 

- **FSDP2**：在 VERL 中直接 strategy: fsdp2 启用，[官方性能指南](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html?utm_source=chatgpt.com)推荐迁移到 FSDP2（相较 FSDP1 平均显存更低、吞吐略升、与 DTensor 更好组合）。 

##### 2.3.2 Critic（价值）参数详解（critic.\*）

```yaml
critic:
  model:
    path: Qwen/Qwen2.5-0.5B-Instruct  # 常与 actor 同路或同族
  strategy: fsdp2
  optim:
    lr: 1e-5
  ppo_epochs: ${actor_rollout_ref.actor.ppo_epochs}    # 不设则常与 actor 对齐
  ppo_micro_batch_size_per_gpu: 2                      # 每卡微批；可略大于 actor
  # （可选）也支持 ppo_mini_batch_size（global），若未设与 actor 相同
```

- Critic 只预测 value（标量或低维），**计算量/显存压力通常小于 actor**，因此 **微批可略放大** 以提吞吐（仍需以 OOM 为上限）。官方对 actor/critic 配置“多数相同”的说明与示例命令里，都把[两者的 micro-batch](https://verl.readthedocs.io/en/latest/algo/ppo.html) 并列给出。 
- **epochs / mini-batch**：若未单独设，通常与 actor 保持一致；VERL 文档在“[PPO 配置](https://verl.readthedocs.io/en/latest/algo/ppo.html)”中给出 critic.ppo_epochs 与 actor.ppo_epochs 的对齐关系。 

#### 2.4 算法（algorithm.*)

```yaml
algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: gae         # 可选: gae / grpo / reinforce_plus_plus / reinforce_plus_plus_baseline / rloo / gpg / opo
  use_kl_in_reward: false    # 在“奖励端”加入 KL 惩罚，与 actor.use_kl_loss 互斥
  kl_penalty: kl             # KL 估计方式: kl / abs / mse / low_var_kl / full
  kl_ctrl:
    type: fixed              # fixed / adaptive
    kl_coef: 0.001           # 初始 KL 系数，不稳 → 加大；稳后可降以增强探索
    horizon: 10000           # 仅 adaptive 使用
    target_kl: 0.1           # 仅 adaptive 使用
```

- adv_estimator 支持 gae、grpo、reinforce_plus_plus、reinforce_plus_plus_baseline、rloo 等；[文档](https://verl.readthedocs.io/en/latest/examples/config.html)亦给出 GPG、OPO 等配方的开启方法（如 adv_estimator: gpg 或 opo）。 
- KL 相关有两条路径：
  1. 在 **奖励端**：algorithm.use_kl_in_reward=true，通过 kl_penalty 与 kl_ctrl 控制；
  2. 在 **损失端**：启用 actor_rollout_ref.actor.use_kl_loss（见上文 Actor 部分）。两者不要同时开，以免“双重惩罚”。

**实务建议：**

- 先用 fixed + 小的 kl_coef（如 1e-3）稳住分布，若生成发散再切 adaptive 并设置 target_kl。这是通用调参手法；target_kl/horizon 的精确定义以源码与文档为准。 

#### 2.5 训练器（traner.*）

```yaml
trainer:
  total_epochs: 30
  project_name: verl_examples
  experiment_name: gsm8k
  logger: ["console", "wandb"]   # 支持: console / wandb / swanlab / mlflow / tensorboard / trackio
  log_val_generations: 0         # 验证时最多记录多少条生成
  nnodes: 1
  n_gpus_per_node: 8
  save_freq: -1                  # 每多少个 iteration 保存；-1 表示按内部策略/关闭
  val_before_train: true
  test_freq: 2                   # 每多少个 iteration 验证
  critic_warmup: 0               # 仅先训 critic 的迭代数
  default_hdfs_dir: null
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
  resume_mode: auto              # disable / auto / resume_path
  resume_from_path: null
  remove_previous_ckpt_in_save: false
  del_local_ckpt_after_load: false
  ray_wait_register_center_timeout: 300
```

- logger 多后端可并列；Ray/KubeRay 场景按 nnodes/n_gpus_per_node 设置多机多卡。 
- 频繁 test_freq 会影响吞吐，初期建议 10–50 步一次；真正的推荐频率请结合你数据量/吞吐做权衡。此条为经验性建议；字段本身释义以官方为准。 

#### 2.6 工具调用 / 多轮交互（rollout.tool_kwargs 与 interaction_config_file）

开启多轮 + 工具，一般切换到 **SGLang** 引擎，并提供工具与交互的 YAML：

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn: true                  # 开多轮
    tool_kwargs:
      tools_config_file: tools/kb.yaml   # 工具清单
    interaction_config_file: tools/interaction.yaml  # 可选：模拟交互脚本（评价/反馈/引导）
```

**如何写工具 YAML（简化示例）：**

```yaml
tools:
  - class_name: my_tools.search.SearchTool  # 继承 verl.tools.base_tool.BaseTool
    config:
      type: native
    tool_schema:
      name: "web_search"
      description: "search the web"
      parameters:
        - name: query
          type: string
```

- 多轮最小配置：rollout.name: "sglang" + multi_turn: true。 
- 自定义工具：实现 BaseTool，在 YAML 的 class_name 中 里登记，然后把 tools_config_file 填到 rollout 配置的 rollout.tool_kwargs.tools_config_file 即可；另外可通过 interaction_config_file 注入「模拟交互/判错/提示」。[官方文档](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html?utm_source=chatgpt.com)包含样例与多模态工具返回的处理方式（如 process_image/process_video）。 

#### 2.7 自定义奖励（custom_reward_function.\*）与奖励模型（reward_model.\*）

##### 2.7.1 规则/程序化奖励

```yaml
custom_reward_function:
  path: examples/reward_score/math.py   # 你的奖励函数文件
  name: compute_score                   # 函数名，默认 compute_score 可不写
```

**实现规范（函数签名）：**

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # 返回一个标量分数，建议归一在 [0, 1] 便于稳定
    ...
```

- [官方文档](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)明确指出可在单文件中实现多个奖励函数，切换时仅改 name；若只测一个函数，命名成 compute_score 可不填 name。 

- [官方示例](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)（如 GSM8K/MATH）会强制输出格式并用字符串匹配给分；当然可以在此基础上叠加「格式分」「正确性分」等 shaping。 

##### 2.7.2 奖励模型（RM）

```yaml
reward_model:
  enable: false                   # 开启后将用 RM 打分
  model:
    input_tokenizer: ${actor_rollout_ref.model.path} # 若 policy 与 RM 模板不同，这里放 RM 的 tokenizer
    path: /abs/path/RM            # AutoModelForSequenceClassification
    external_lib: ${actor_rollout_ref.model.external_lib}
    trust_remote_code: false
    fsdp_config:
      min_num_params: 0
      param_offload: false
  micro_batch_size_per_gpu: 16
  max_length: null
  reward_manager: naive           # 或 prime（若验证函数均可多进程安全）
```

- 一旦启用 reward_model.enable 并且批次里已经算出了 rm_scores，RewardManager 会**直接返回 RM 分数**，**不再调用**你的规则打分函数（custom_reward_function）。
- 输入 tokenizer 与 chat 模板不一致时，需要先解码再套 RM 的模板。 
  - **策略模型（Actor）**用的是 **A** 牌 tokenizer + chat 模板（比如 Qwen 的模板）；
  - **奖励模型（RM）**用的是 **B** 牌 tokenizer + chat 模板（比如 Llama3 或自定义的 RM 模板）。
  - 这两套**词表与对话包装规则不同**。因此**不能**把 Actor 产出的 **token IDs** 直接丢给 RM。
    - **tokenizer 不同**：不同模型的词表/特殊符号不同（如 <s>, <|im_start|> 等），直接复用 token 会变成无效或错位的符号。
    - **chat 模板不同**：RM 训练时学习的是“按它的模板包好的对话格式”，比如角色分隔、系统提示、结尾标记等。如果不按它的模板拼，RM 的分数就不可信。 
  - 正确做法是：**先用 Actor 的 tokenizer 把 token 解码成纯文本**，再用 **RM 的 tokenizer 的 chat 模板 apply_chat_template** 重新包装后**再**喂给 RM 打分。这样才能和 RM 的训练格式对齐。 

- 例子：Actor 生成了回答（是 **Qwen** 模板/词表，Qwen2.5-Instruct），我们要用 **Llama3-RM** 去打分。

  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
  
  # 1) 加载 Actor 与 RM
  actor_name = "Qwen/Qwen2.5-0.5B-Instruct"
  rm_name    = "your-org/llama3-reward-model"
  
  actor_tok = AutoTokenizer.from_pretrained(actor_name)
  rm_tok    = AutoTokenizer.from_pretrained(rm_name, use_fast=True)
  rm_model  = AutoModelForSequenceClassification.from_pretrained(rm_name)
  
  # 2) Actor 侧：得到生成（注意：这里省略了真正的生成代码，假设拿到了 Actor 的 token 序列）
  actor_output_ids = actor_generate_ids  # e.g. [batch, seq_len]
  
  # === 关键步骤 A：先用 Actor 的 tokenizer 解码成纯文本 ===
  resp_text = actor_tok.decode(actor_output_ids[0], skip_special_tokens=True)
  
  # 3) 构造 RM 需要的消息列表（把 user 和 actor 的回答分别放到各自角色）
  user_prompt = original_user_prompt_text  # 你送给 Actor 的原始用户问题（纯文本）
  messages = [
      {"role": "system", "content": "You are a reward model that rates helpfulness & harmlessness."},
      {"role": "user", "content":   user_prompt},
      {"role": "assistant", "content": resp_text},
  ]
  
  # === 关键步骤 B：用 RM 的 chat 模板重新“套壳” ===
  #   - tokenize=False: 先拿到一整串符合 RM 训练格式的字符串
  #   - add_generation_prompt=False: 评估已有回答，不再追加“请继续生成”的提示
  prompt_for_rm = rm_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)  
  # 参考 HF 文档：apply_chat_template 的用法
  
  # 4) 用 RM 自己的 tokenizer 编码，再送进 RM 打分
  rm_inputs = rm_tok(prompt_for_rm, return_tensors="pt", truncation=True)
  with torch.no_grad():
      out = rm_model(**rm_inputs).logits.squeeze()
  
  score = float(out.item())  # 你的 RM 分数（按具体 RM 解释：越大越好或反之）
  ```

  - **A→文本→B**：先把 **Actor token → 文本**，再**按 RM 模板**拼装 → 交给 **RM 的 tokenizer** 编码 → **RM** 打分。
  - apply_chat_template 就是让你“按 RM 的规范”把 *system/user/assistant* 序列化成 RM 训练时见过的样子。

### 3 Cookbook：五种常见场景怎么改

#### 3.1 显存不够（OOM）

**思路优先级**：先降每卡负载 → 再降序列长 → 再调引擎显存占比 → 最后再上并行或换更小模型。

- *_micro_batch_size_per_gpu 降到 1 起步；这类 **per-GPU 微批**是官方建议优先调的性能开关。 
- 收紧 max_prompt_length / max_response_length，序列长是显存一号放大器。 
- vLLM 提高或降低 gpu_memory_utilization（占总显存比例，0~1）；过高可能在某些版本上仍逼近满显存。 
- SGLang 调低 --mem-fraction-static（KV 池占比），通常 0.7~0.9；越低越不易 OOM、但并发/吞吐会降。 

**关键修改部位**

```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 1
  rollout:
    # vLLM
    name: vllm
    gpu_memory_utilization: 0.5   # 先保守
    # SGLang（如用）
    engine_kwargs:
      sglang:
        mem_fraction_static: 0.8  # 0.7~0.9 之间试

data:
  max_prompt_length: 384
  max_response_length: 128
```

#### 3.2 收敛不稳（奖励上不去 / 来回跳）

**思路**：先稳住策略偏移（KL）→ 再降学习率 → 再加大 mini-batch。

- 增大 KL 系数（reward 端或 loss 端**二选一**），如从 0.001 增加到 0.004，更强地限制策略偏离，防止策略飘远。 
- 降 actor.optim.lr 一档，如从 1e-6 降低到 5e-7，降低步长，减小震荡。
- 加大 ppo_mini_batch_size，一般从 64 增加到 128，更大的小批能平滑梯度、提升稳定性（代价是吞吐略降）。

**关键修改部位**，具体配置参考[官方文档](https://verl.readthedocs.io/en/latest/algo/ppo.html?utm_source=chatgpt.com)

```yaml
algorithm:
  use_kl_in_reward: true
  kl_penalty: low_var_kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.004     # 在 1e-3~1e-2 内加到更保守

actor_rollout_ref:
  actor:
    optim:
      lr: 5e-7         # 例如从 1e-6 降一档
    ppo_mini_batch_size: 128
    use_kl_loss: false # 避免与 reward 端双罚
```

#### 3.3 吞吐低（GPU 吃不满 / TPS 偏低）

**思路**：以 rollout 引擎为中心做“批合并”与“显存占比”调优。

- vLLM：提高 gpu_memory_utilization、调 max_num_seqs / max_num_batched_tokens。 
- SGLang：适度提高 mem_fraction_static 提升 KV 池容量，但注意 OOM 风险。 
- 适当放大 *_micro_batch_size_per_gpu（前向-only 的 rollout/ref 通常能设更大）。

**关键修改部位**

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    gpu_memory_utilization: 0.7
    max_num_batched_tokens: 16384
    max_num_seqs: 2048

  ref:
    log_prob_micro_batch_size_per_gpu: 16  # ref 前向-only，可更大

actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 2        # 逐步抬，观察显存
```

#### 3.4 多轮生成跑歪（第 2 轮还在解释 / 历史拼接错）

**思路**：切换到 SGLang 多轮、强制模板约束、验证时用确定性采样。

- 开启多轮的**最小配置**：name: "sglang" + multi_turn: true。 
- 验证/评测用 temperature=0, do_sample=false，避免不确定性。 

**关键修改部位**

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn: true
    # 验证期采样确定性
    val_kwargs:
      temperature: 0
      do_sample: false
      top_p: 1.0
```

#### 3.5 工具依赖过强 / 过弱（想多用 or 少用工具）

**思路**：在奖励里调“证据/工具调用”的权重；在多轮交互用 SGLang + 工具 YAML。

- 多轮 + 工具最小线：rollout.name: sglang、multi_turn: true、tools_config_file 指向你的工具清单。 
- 奖励中对“命中工具/证据”加分或对“多次调用”扣小分（规则奖励端实现）。

**关键修改部位**

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn: true
    tool_kwargs:
      tools_config_file: tools/kb.yaml  # 你的工具清单

custom_reward_function:
  path: examples/reward_score/my_task.py
  name: compute_score
  reward_kwargs:
    use_tool_bonus: true
    tool_bonus: 0.05
    multi_tool_penalty: 0.02
```

### 4 可复制配置片段（模板合集）

#### 4.1. 单轮高吞吐（vLLM）

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    multi_turn: false
    gpu_memory_utilization: 0.6   # 0.5~0.8 试，vLLM 的 gpu_memory_utilization = 占总显存比例
    max_num_batched_tokens: 16384
    max_num_seqs: 1024
```

#### 4.2. 多轮 + 工具（SGLang）

```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn: true
    engine_kwargs:
      sglang:
        mem_fraction_static: 0.85  # 视显存/并发调，SGLang 的静态内存占比用来控制 KV 池容量
    tool_kwargs:
      tools_config_file: tools/kb.yaml
```

#### 4.3. 不同位置的 KL（避免 reward 与 loss 双罚）

##### 4.3.1 KL在 reward 端

```yaml
algorithm:
  use_kl_in_reward: true
  kl_penalty: kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.001

actor_rollout_ref:
  actor:
    use_kl_loss: false	# PPO/KL 语义与二选一原则
```

##### 4.3.2. KL 在 loss 端

```yaml
algorithm:
  use_kl_in_reward: false	 # PPO/KL 语义与二选一原则

actor_rollout_ref:
  actor:
    use_kl_loss: true
    kl_loss_type: kl          # 或 abs/mse/low_var_kl/full
    kl_loss_coef: 0.001
```

#### 4.4. per-GPU 微批与 token 上限（避 OOM）

```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 1
    ppo_max_token_len_per_gpu: 12288  # 按 n*(prompt)+response 估
    use_dynamic_bsz: true             # 超过就自动缩小
  ref:
    log_prob_micro_batch_size_per_gpu: 16
```

- per-GPU 微批 & token 上限字段、以及“前向-only 可更大”的建议均见[官方调优页](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html?utm_source=chatgpt.com)

### 5 排错速查（配置相关）

| **症状** | **可能原因** | **快速核查** | **配置/修复建议（可直接改）** | **参考文档/issue** |
|---|---|---|---|---|
| 推理/采样 OOM（vLLM） | KV Cache 预留过大；批合并参数过激 | 查看日志里的 GPU/Cache 利用率；是否刚开始第二步就 OOM | `rollout.gpu_memory_utilization: 0.5~0.8`；必要时下调 `max_num_batched_tokens / max_num_seqs`；升级到对应版本并观察释放缓存是否正常 | [vLLM 参数说明](https://docs.vllm.ai/en/latest/serving/configuration.html) |
| 推理/采样 OOM（SGLang） | KV Pool 过大 | 查看 `--mem-fraction-static` 当前值 | 下调 `engine_kwargs.sglang.mem_fraction_static: 0.7~0.9` 区间内试 | [SGLang 参数](https://github.com/sgl-project/sglang) |
| 吞吐很低，GPU 吃不满（vLLM） | 批合并过小；Cache 预留保守 | 观察 “GPU cache utilization / running seqs” | 上调 `gpu_memory_utilization`、`max_num_batched_tokens`、`max_num_seqs` | [vLLM 性能调优](https://docs.vllm.ai/en/latest/serving/performance.html) |
| 吞吐很低（Ref/rollout 仅前向） | 前向微批过小 | 观察 Ref/logprob 的 batch | 放大 `ref.log_prob_micro_batch_size_per_gpu`、`rollout.log_prob_micro_batch_size_per_gpu`（前向-only 可更大） | [VERL 配置指南](https://verl.readthedocs.io/) |
| 训练不稳：奖励上不去/来回跳 | 策略偏移过大；步长过大；mini-batch 太小 | 看 KL 值是否飙升；loss 抖动 | **二选一**加 KL：`algorithm.use_kl_in_reward: true` **或** `actor.use_kl_loss: true`；增大 `kl_coef`；减小 `actor.optim.lr`；增大 `actor.ppo_mini_batch_size` | [PPO & KL 控制](https://huggingface.co/blog/trl-ppo) |
| KL 惩罚过强、生成塌缩 | 同时在 reward 与 loss 两端加了 KL（双罚） | grep `use_kl_in_reward` 与 `use_kl_loss` | 仅保留一端；按目标 KL 回调 `kl_coef` | [VERL algorithm 文档](https://verl.readthedocs.io/en/latest/config/algorithm.html) |
| 多轮对话没有生效 / 历史没拼上 | 仍在单轮引擎；未开 multi_turn | 检查 rollout 引擎与开关 | `rollout.name: sglang` + `rollout.multi_turn: true`；验证时用确定性采样 | [VERL rollout 文档](https://verl.readthedocs.io/en/latest/config/rollout.html) |
| 用 vLLM async / DAPO 报错 | 版本/组合尚不稳定 | 查看对应 issue | 切回 `mode: sync` 或升级到建议版本；按 issue workaround 调整 | [vLLM GitHub issues](https://github.com/vllm-project/vllm/issues) |
| RM 启用后规则打分不生效 | RM 分数**默认短路替换**规则分 | 查看 batch 是否已有 `rm_scores` | 要“混合”就自写合成逻辑（自定义 reward_fn）；或禁用 `reward_model.enable` | [VERL reward 文档](https://verl.readthedocs.io/en/latest/config/reward.html) |
| 不同 tokenizer / 模板导致 RM 打分异常 | Actor 与 RM 模板/词表不一致 | 看输入是否按 RM 的 chat 模板 | **先用 Actor 解码为文本**→**用 RM 的 `apply_chat_template` 重新拼装**→再喂 RM | [HF Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating) |
| 验证/展示结果不稳定 | 评测仍在随机采样 | 检查 val/test 采样参数 | `rollout.val_kwargs: {temperature: 0, do_sample: false, top_p: 1.0}` | [VERL rollout 文档](https://verl.readthedocs.io/en/latest/config/rollout.html) |
