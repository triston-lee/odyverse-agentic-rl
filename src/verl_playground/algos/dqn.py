import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(self, obs_space, act_space, cfg):
        self.act_dim = act_space.n
        self.q = MLP(obs_space.shape[0], self.act_dim)
        self.target_q = MLP(obs_space.shape[0], self.act_dim)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=cfg["algo"]["lr"])
        self.gamma = cfg["algo"]["gamma"]
        self.batch_size = cfg["algo"]["batch_size"]
        self.buffer = deque(maxlen=cfg["algo"]["buffer_size"])
        self.eps, self.eps_end = 1.0, cfg["algo"]["exploration"]["eps_end"]
        self.eps_decay = (1.0 - self.eps_end) / cfg["algo"]["exploration"]["eps_decay_steps"]
        self.device = torch.device(cfg["runtime"]["device"])
        self.q.to(self.device)
        self.target_q.to(self.device)
        self.ep_return = 0

    def act(self, obs):
        if random.random() < self.eps:
            return random.randrange(self.act_dim)
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.q(obs_t).argmax(dim=-1).item())

    def observe(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))
        self.ep_return += reward
        if done:
            self.ep_return = 0

    def learn(self, step):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        obs, act, rew, nxt, done = zip(*batch)
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        act = torch.tensor(act).to(self.device)
        rew = torch.tensor(rew, dtype=torch.float32).to(self.device)
        nxt = torch.tensor(nxt, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_val = self.q(obs).gather(1, act.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next = self.target_q(nxt).max(1)[0]
            target = rew + self.gamma * (1 - done) * max_next
        loss = ((q_val - target) ** 2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if step % 1000 == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.eps = max(self.eps_end, self.eps - self.eps_decay)

    def state_dict(self):
        return self.q.state_dict()

    def load_state_dict(self, s):
        self.q.load_state_dict(s)

    def evaluate(self, env, episodes=5):
        returns = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done, ret = False, 0
            while not done:
                act = self.act(obs)
                obs, r, term, trunc, _ = env.step(act)
                ret += r
                done = term or trunc
            returns.append(ret)
        return np.mean(returns)
