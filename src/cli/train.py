# src/cli/train.py
import argparse
from pathlib import Path
import time
import torch
import numpy as np
import gymnasium as gym

from verl_playground.utils.config import load_config
from verl_playground.envs.cartpole import make_env
from verl_playground.algos.dqn import DQN
from verl_playground.utils.loggers import Logger

# 多进程 runner（需要下方的“兼容小补丁”）
from verl_playground.runners.multiprocess import MultiProcessRunner


def _set_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动降级到 CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def _single_process_train(cfg) -> None:
    """单机版本：环境交互 + 学习在同一进程内"""
    device = _set_device(cfg["runtime"]["device"])
    env = make_env(cfg["env"]["name"], seed=cfg["seed"])
    algo = DQN(env.observation_space, env.action_space, cfg)
    # 将 algo 的设备与配置对齐
    for p in algo.q.parameters():
        p.device = device

    logger = Logger(run_dir="experiments")
    obs, _ = env.reset(seed=cfg["seed"])

    total_steps = int(cfg["train"]["total_steps"])
    log_itv = int(cfg["train"]["log_interval"])
    eval_itv = int(cfg["train"]["eval_interval"])
    save_itv = int(cfg["train"]["save_interval"])

    st = time.time()
    for step in range(1, total_steps + 1):
        action = algo.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        algo.observe(obs, action, reward, next_obs, done)
        algo.learn(step)
        obs = next_obs if not done else env.reset()[0]

        if step % log_itv == 0:
            logger.log({"step": step, "eps": algo.eps, "elapsed_s": time.time() - st})

        if step % eval_itv == 0:
            eval_ret = algo.evaluate(make_env(cfg["env"]["name"], seed=cfg["seed"] + 1))
            logger.log({"step": step, "eval_return": float(eval_ret)})

        if step % save_itv == 0:
            ckpt = Path("experiments/checkpoints/latest.pt")
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(algo.state_dict(), ckpt)

    ckpt = Path("experiments/checkpoints/final.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(algo.state_dict(), ckpt)
    print(f"[done] saved -> {ckpt}")


def _multiprocess_train(cfg) -> None:
    """
    多进程版本：多个 Actor 进程并行采样，主进程充当 Learner。
    需要 MultiProcessRunner 实现 reset() 返回初始观测（见文末“兼容小补丁”）。
    """
    device = _set_device(cfg["runtime"]["device"])
    env_name = cfg["env"]["name"]
    num_workers = int(cfg["runtime"]["num_workers"])
    seed = int(cfg["seed"])

    # 先用一个临时 env 拿到空间定义（obs/act）以构建 Algo
    probe_env = gym.make(env_name)
    algo = DQN(probe_env.observation_space, probe_env.action_space, cfg)
    probe_env.close()

    logger = Logger(run_dir="experiments")

    runner = MultiProcessRunner(env_name, num_workers=num_workers, seed=seed)
    # 重要：获取每个 worker 的初始 obs
    obs_batch = runner.reset()  # List[np.ndarray]，长度 = num_workers

    total_steps = int(cfg["train"]["total_steps"])
    log_itv = int(cfg["train"]["log_interval"])
    eval_itv = int(cfg["train"]["eval_interval"])
    save_itv = int(cfg["train"]["save_interval"])

    st = time.time()
    # 我们这里把一次“并行 step”计作 num_workers 条 transition
    # 为了对齐 single_process 的“step”语义，这里用累加计数。
    transition_count = 0

    while transition_count < total_steps:
        # 基于每个 worker 当前 obs 选择动作
        actions = []
        for obs in obs_batch:
            actions.append(algo.act(obs))

        # 并行执行一步，返回每个 worker 的 transition
        results = runner.step(actions)
        # results: List[(obs, action, reward, next_obs, done)]
        next_obs_batch = []

        for (obs, action, reward, next_obs, done) in results:
            algo.observe(obs, action, reward, next_obs, done)
            algo.learn(transition_count)  # 用全局 transition 计数驱动 target 更新等
            next_obs_batch.append(next_obs)
            transition_count += 1
            if transition_count >= total_steps:
                break

        obs_batch = next_obs_batch

        # 日志与评估
        if transition_count % log_itv == 0:
            logger.log(
                {
                    "step": transition_count,
                    "eps": algo.eps,
                    "num_workers": num_workers,
                    "elapsed_s": time.time() - st,
                }
            )

        if transition_count % eval_itv == 0:
            eval_ret = algo.evaluate(make_env(env_name, seed=seed + 1))
            logger.log({"step": transition_count, "eval_return": float(eval_ret)})

        if transition_count % save_itv == 0:
            ckpt = Path("experiments/checkpoints/latest.pt")
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(algo.state_dict(), ckpt)

    runner.close()
    ckpt = Path("experiments/checkpoints/final.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(algo.state_dict(), ckpt)
    print(f"[done] saved -> {ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # 允许命令行临时覆盖 YAML：a.b=1 a.c=foo
    parser.add_argument(
        "overrides", nargs="*", help="临时覆盖配置，例如: train.total_steps=20000 runtime.num_workers=4"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)

    num_workers = int(cfg["runtime"].get("num_workers", 1))
    if num_workers <= 1:
        print("[info] 进入单机训练模式")
        _single_process_train(cfg)
    else:
        print(f"[info] 进入多进程训练模式：num_workers={num_workers}")
        _multiprocess_train(cfg)


if __name__ == "__main__":
    main()