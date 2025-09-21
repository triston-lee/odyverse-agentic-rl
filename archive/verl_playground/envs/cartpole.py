import gymnasium as gym


def make_env(env_name: str, seed: int = 42):
    env = gym.make(env_name)
    try:
        env.reset(seed=seed)
    except TypeError:
        # 兼容旧版 gym
        env.seed(seed)
    return env
