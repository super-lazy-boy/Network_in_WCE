from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .agent import DQNAgent
from .config import EnvConfig, TrainingConfig
from .env import UAVNetworkEnv


class Trainer:
    def __init__(self, env_cfg: EnvConfig | None = None, train_cfg: TrainingConfig | None = None):
        self.env_cfg = env_cfg or EnvConfig()
        self.train_cfg = train_cfg or TrainingConfig()

        self.env = UAVNetworkEnv(self.env_cfg, seed=self.train_cfg.seed)
        self.agent = DQNAgent(self.env.state_size, self.env.action_size, self.train_cfg)

    def evaluate_policy(
        self,
        eval_episodes: int = 5,
        eval_seed_start: int = 10000,
    ) -> Dict[str, float]:
        """
        固定种子 + deterministic policy 的评估函数。
        不进行探索，不写入 replay buffer，不更新网络参数。

        作用：
        1. 让 best.pt 的选择不受训练时随机探索影响
        2. 减小单个 episode 偶然性
        3. 更接近真正部署时的策略表现
        """
        metrics = {
            "episode_reward": [],
            "throughput": [],
            "connectivity": [],
            "weak_link_ratio": [],
            "energy_penalty": [],
            "collision_penalty": [],
            "reliability_proxy": [],
        }

        for k in range(eval_episodes):
            eval_env = UAVNetworkEnv(self.env_cfg, seed=eval_seed_start + k)
            state = eval_env.reset()
            done = False
            reward_sum = 0.0
            last_info = {}

            while not done:
                action = self.agent.act(state, deterministic=True)
                next_state, reward, done, info = eval_env.step(action)
                state = next_state
                reward_sum += reward
                last_info = info

            metrics["episode_reward"].append(reward_sum)
            metrics["throughput"].append(last_info.get("throughput", 0.0))
            metrics["connectivity"].append(last_info.get("connectivity", 0.0))
            metrics["weak_link_ratio"].append(last_info.get("weak_link_ratio", 0.0))
            metrics["energy_penalty"].append(last_info.get("energy_penalty", 0.0))
            metrics["collision_penalty"].append(last_info.get("collision_penalty", 0.0))
            metrics["reliability_proxy"].append(last_info.get("reliability_proxy", 0.0))

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    @staticmethod
    def _make_eval_score(eval_stats: Dict[str, float]) -> float:
        """
        用评估集平均表现构造综合评分。
        这里优先考虑：
        1. 连通性
        2. 可靠性
        3. 吞吐量
        然后惩罚：
        4. 弱链路
        5. 能耗
        6. 碰撞

        注意：不要直接用单纯 reward 做 best 选择，
        因为 reward 在 RL 中可能和最终 NS-3 指标不完全一致。
        """
        score = (
            3.0 * eval_stats["connectivity"]
            + 2.5 * eval_stats["reliability_proxy"]
            + 2.0 * eval_stats["throughput"]
            + 0.2 * eval_stats["episode_reward"] / 100.0
            - 1.5 * eval_stats["weak_link_ratio"]
            - 0.8 * eval_stats["energy_penalty"]
            - 2.0 * eval_stats["collision_penalty"]
        )
        return float(score)

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
            "eval_score": [],
            "eval_reward": [],
            "eval_throughput": [],
            "eval_connectivity": [],
            "eval_reliability_proxy": [],
        }

        best_score = -1e18
        best_ckpt = Path(checkpoint_dir) / "best.pt"

        # 建议每隔固定轮数做 deterministic evaluation
        eval_interval = 50
        eval_episodes = 5

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

            # ---- 训练日志 ----
            logs["episode_reward"].append(reward_sum)
            logs["throughput"].append(last_info.get("throughput", 0.0))
            logs["connectivity"].append(last_info.get("connectivity", 0.0))
            logs["loss"].append(float(np.mean(losses)) if losses else 0.0)
            logs["weak_link_ratio"].append(last_info.get("weak_link_ratio", 0.0))
            logs["energy_penalty"].append(last_info.get("energy_penalty", 0.0))
            logs["collision_penalty"].append(last_info.get("collision_penalty", 0.0))
            logs["reliability_proxy"].append(last_info.get("reliability_proxy", 0.0))

            # ---- 周期性保存普通 checkpoint ----
            if ep % 100 == 0:
                ckpt = Path(checkpoint_dir) / f"dqn_ep{ep}.pt"
                self.agent.save(str(ckpt))

            # ---- 周期性评估：用评估均值而不是训练单回合结果选 best.pt ----
            if ep % eval_interval == 0:
                eval_stats = self.evaluate_policy(
                    eval_episodes=eval_episodes,
                    eval_seed_start=10000,
                )
                eval_score = self._make_eval_score(eval_stats)

                logs["eval_score"].append(eval_score)
                logs["eval_reward"].append(eval_stats["episode_reward"])
                logs["eval_throughput"].append(eval_stats["throughput"])
                logs["eval_connectivity"].append(eval_stats["connectivity"])
                logs["eval_reliability_proxy"].append(eval_stats["reliability_proxy"])

                if eval_score > best_score:
                    best_score = eval_score
                    self.agent.save(str(best_ckpt))

            if ep % 100 == 0:
                last_eval_score = logs["eval_score"][-1] if logs["eval_score"] else float("nan")
                last_eval_thr = logs["eval_throughput"][-1] if logs["eval_throughput"] else float("nan")
                last_eval_conn = logs["eval_connectivity"][-1] if logs["eval_connectivity"] else float("nan")
                last_eval_rel = logs["eval_reliability_proxy"][-1] if logs["eval_reliability_proxy"] else float("nan")

                print(
                    f"[ep={ep}] "
                    f"train_reward={reward_sum:.3f} "
                    f"train_thr={logs['throughput'][-1]:.3f} "
                    f"train_conn={logs['connectivity'][-1]:.3f} "
                    f"eps={self.agent.epsilon():.3f} "
                    f"loss={logs['loss'][-1]:.3f} "
                    f"wlr={logs['weak_link_ratio'][-1]:.3f} "
                    f"energy={logs['energy_penalty'][-1]:.3f} "
                    f"col={logs['collision_penalty'][-1]:.3f} "
                    f"rel={logs['reliability_proxy'][-1]:.3f} "
                    f"| eval_score={last_eval_score:.3f} "
                    f"eval_thr={last_eval_thr:.3f} "
                    f"eval_conn={last_eval_conn:.3f} "
                    f"eval_rel={last_eval_rel:.3f}"
                )

        self.agent.save(str(Path(checkpoint_dir) / "final.pt"))
        print(f"best_score={best_score:.4f}, best_ckpt={best_ckpt}")
        return logs