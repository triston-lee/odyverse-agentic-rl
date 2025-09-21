class SingleProcessRunner:
    """
    单机 runner：对齐多进程接口，统一 step 返回字典。
    """
    def __init__(self, env):
        self.env = env
        self._obs, _ = env.reset()

    def reset(self):
        self._obs, _ = self.env.reset()
        return self._obs

    def step(self, actions):
        # 单机时 actions 是 len=1 的列表
        action = actions[0] if isinstance(actions, (list, tuple)) else actions
        next_obs, reward, term, trunc, _ = self.env.step(action)
        done = bool(term or trunc)

        tr = {
            "obs": self._obs,
            "action": action,
            "reward": float(reward),
            "next_obs": next_obs,
            "done": done,
        }
        if done:
            next_obs, _ = self.env.reset()
        self._obs = next_obs
        return [tr]