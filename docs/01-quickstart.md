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

## **1）准备数据（GSM8K 或最小玩具集）**

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

