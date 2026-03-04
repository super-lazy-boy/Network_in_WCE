import argparse
import json
from pathlib import Path

from uav_rl.config import EnvConfig, TrainingConfig
from uav_rl.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", type=str, default="artifacts")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainingConfig(episodes=args.episodes)
    trainer = Trainer(EnvConfig(), train_cfg)
    logs = trainer.train(checkpoint_dir=str(out / "checkpoints"))

    with (out / "train_logs.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


if __name__ == "__main__":
    main()
