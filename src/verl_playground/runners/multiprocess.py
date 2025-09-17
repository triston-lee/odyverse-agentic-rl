# src/verl_playground/runners/multiprocess.py
import gymnasium as gym, multiprocessing as mp

def worker_task(env_name, conn, seed):
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    # 先把初始观测发给主进程
    conn.send(obs)
    while True:
        action = conn.recv()
        if action is None:
            break
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        if done:
            next_obs, _ = env.reset()
        # 返回一步 transition（含“上一步的 obs”）
        conn.send((obs, action, reward, next_obs, done))
        obs = next_obs

class MultiProcessRunner:
    def __init__(self, env_name, num_workers=4, seed=42):
        self.num_workers = num_workers
        self.parent_conns, self.procs = [], []
        for i in range(num_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=worker_task, args=(env_name, child_conn, seed + i))
            p.daemon = True
            p.start()
            self.parent_conns.append(parent_conn)
            self.procs.append(p)

    def reset(self):
        """收集每个 worker 的初始观测"""
        return [conn.recv() for conn in self.parent_conns]

    def step(self, actions):
        for conn, a in zip(self.parent_conns, actions):
            conn.send(a)
        results = [conn.recv() for conn in self.parent_conns]
        return results

    def close(self):
        for conn in self.parent_conns:
            conn.send(None)
        for p in self.procs:
            p.join()