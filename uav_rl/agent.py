from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import TrainingConfig
from .replay_buffer import ReplayBuffer


class QNet(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(256, 1)
        self.adv_head = nn.Linear(256, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        value = self.value_head(z)
        adv = self.adv_head(z)
        return value + adv - adv.mean(dim=1, keepdim=True)


@dataclass
class TrainStats:
    loss: float = 0.0
    q_value: float = 0.0


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, cfg: TrainingConfig | None = None):
        self.cfg = cfg or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.steps = 0

        self.online = QNet(state_size, action_size).to(self.device)
        self.target = QNet(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.replay_capacity)
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self) -> float:
        ratio = min(1.0, self.steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * math.exp(-4.0 * ratio)

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        eps = 0.0 if deterministic else self.epsilon()
        self.steps += 1
        if random.random() < eps:
            return random.randrange(self.action_size)
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(st)
        return int(torch.argmax(q, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push((state, action, reward, next_state, done))

    def learn(self, batch_size: int | None = None) -> TrainStats:
        bs = batch_size or self.cfg.batch_size
        if len(self.buffer) < max(bs, self.cfg.warmup_steps):
            return TrainStats()

        batch = self.buffer.sample(bs)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(-1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        s2 = torch.tensor(np.asarray(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q = self.online(s).gather(1, a)
        with torch.no_grad():
            next_actions = self.online(s2).argmax(dim=1, keepdim=True)
            next_q = self.target(s2).gather(1, next_actions)
            target = r + (1.0 - d) * self.cfg.gamma * next_q

        loss = self.loss_fn(q, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        self.optim.step()

        if self.steps % self.cfg.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return TrainStats(loss=float(loss.item()), q_value=float(q.mean().item()))

    def save(self, path: str) -> None:
        torch.save({"online": self.online.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(payload["online"])
        self.target.load_state_dict(payload["online"])
