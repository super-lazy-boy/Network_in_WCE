from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .agent import DQNAgent
from .config import EnvConfig, TrainingConfig
from .env import UAVNetworkEnv


class Trainer:
    def __init__(self, env_cfg: EnvConfig | None = None, train_cfg: TrainingConfig | None = None):
        self.env = UAVNetworkEnv(env_cfg, seed=(train_cfg.seed if train_cfg else 42))
        self.agent = DQNAgent(self.env.state_size, self.env.action_size, train_cfg)
        self.train_cfg = train_cfg or TrainingConfig()

    def train(self, checkpoint_dir: str = "checkpoints") -> Dict[str, List[float]]:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logs: Dict[str, List[float]] = {
            "episode_reward": [],
            "throughput": [],
            "connectivity": [],
            "weak_link_ratio": [],
            "energy_penalty": [],
            "collision_penalty": [],
            "reliability_proxy": [],
            "loss": [],
        }
        for ep in range(1, self.train_cfg.episodes + 1):
            state = self.env.reset()
            done = False
            reward_sum = 0.0
            last_info = {}
            losses = []

            while not done:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                stats = self.agent.learn()
                if stats.loss > 0:
                    losses.append(stats.loss)
                state = next_state
                reward_sum += reward
                last_info = info

            logs["episode_reward"].append(reward_sum)
            logs["throughput"].append(last_info.get("throughput", 0.0))
            logs["connectivity"].append(last_info.get("connectivity", 0.0))
            logs["loss"].append(float(np.mean(losses)) if losses else 0.0)
            logs["weak_link_ratio"].append(last_info.get("weak_link_ratio", 0.0))
            logs["energy_penalty"].append(last_info.get("energy_penalty", 0.0))
            logs["collision_penalty"].append(last_info.get("collision_penalty", 0.0))
            logs["reliability_proxy"].append(last_info.get("reliability_proxy", 0.0))

            if ep % 100 == 0:
                ckpt = Path(checkpoint_dir) / f"dqn_ep{ep}.pt"
                self.agent.save(str(ckpt))
                print(
                    f"[ep={ep}] reward={reward_sum:.3f} thr={logs['throughput'][-1]:.3f} "
                    f"conn={logs['connectivity'][-1]:.3f} eps={self.agent.epsilon():.3f}"
                    f"loss={logs['loss'][-1]:.3f} wlr={logs['weak_link_ratio'][-1]:.3f} "
                    f"ep={logs['episode_reward'][-1]:.3f} ep_penalty={logs['energy_penalty'][-1]:.3f} "
                    f"col_penalty={logs['collision_penalty'][-1]:.3f} rel_proxy={logs['reliability_proxy'][-1]:.3f}"
                )
        self.agent.save(str(Path(checkpoint_dir) / "final.pt"))
        return logs
