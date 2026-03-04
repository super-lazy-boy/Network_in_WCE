"""Backward-compatible entrypoint.

Use `python scripts/train.py` for full training options.
"""

from uav_rl.config import EnvConfig, TrainingConfig
from uav_rl.trainer import Trainer


if __name__ == "__main__":
    trainer = Trainer(EnvConfig(), TrainingConfig(episodes=50))
    trainer.train(checkpoint_dir="artifacts/checkpoints")
