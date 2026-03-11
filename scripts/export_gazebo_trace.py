#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# 让脚本能够从项目根目录导入 uav_rl 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_rl.agent import DQNAgent
from uav_rl.config import EnvConfig, TrainingConfig
from uav_rl.env import UAVNetworkEnv


def compute_yaw_from_velocity(vx: float, vy: float, fallback_yaw: float = 0.0) -> float:
    """
    根据水平速度计算航向角 yaw。
    如果水平速度过小，则使用 fallback_yaw，避免 atan2(0,0) 导致方向抖动。
    """
    if abs(vx) + abs(vy) < 1e-8:
        return fallback_yaw
    return float(math.atan2(vy, vx))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained best.pt policy rollout to gazebo_trace.json"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained checkpoint, e.g. artifacts/best.pt",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for gazebo trace json, e.g. artifacts/gazebo_trace.json",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Maximum rollout steps to export",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Environment seed for deterministic rollout",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to export continuously; usually 1 is enough",
    )
    return parser


def make_frame(env: UAVNetworkEnv, t: float, last_yaws: list[float]) -> dict:
    """
    将当前环境状态导出为 Gazebo 一帧：
    {
      "t": 0.0,
      "uavs": [
        {"id":0,"x":...,"y":...,"z":...,"yaw":...},
        ...
      ]
    }
    """
    uavs = []
    for i in range(env.cfg.num_uavs):
        x = float(env.positions[i, 0])
        y = float(env.positions[i, 1])
        z = float(env.positions[i, 2])

        # 优先使用 velocities 推 yaw；如果当前 env 没有 velocities，就退化为 0
        if hasattr(env, "velocities"):
            vx = float(env.velocities[i, 0])
            vy = float(env.velocities[i, 1])
            yaw = compute_yaw_from_velocity(vx, vy, fallback_yaw=last_yaws[i])
        else:
            yaw = last_yaws[i]

        last_yaws[i] = yaw

        uavs.append(
            {
                "id": i,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
            }
        )

    return {"t": float(t), "uavs": uavs}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path = Path(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1) 构造环境和 agent ----
    env_cfg = EnvConfig()
    train_cfg = TrainingConfig()

    env = UAVNetworkEnv(env_cfg, seed=args.seed)
    agent = DQNAgent(env.state_size, env.action_size, train_cfg)

    # ---- 2) 加载 best.pt ----
    # 当前 agent.load() 会把 online 权重加载到 online 和 target 网络。:contentReference[oaicite:2]{index=2}
    agent.load(str(model_path))

    # ---- 3) rollout 并导出轨迹 ----
    frames: list[dict] = []
    step_dt = float(env.cfg.dt)
    global_step = 0

    # 为了让速度接近 0 时 yaw 不乱跳，维护一个上一次的 yaw
    last_yaws = [0.0 for _ in range(env.cfg.num_uavs)]

    for ep in range(args.episodes):
        state = env.reset()
        done = False

        # 把 reset 后的初始状态也保存一帧
        frames.append(make_frame(env, t=global_step * step_dt, last_yaws=last_yaws))

        while not done and global_step < args.steps:
            # deterministic=True：导出最优策略轨迹，而不是带 epsilon 的探索轨迹
            action = agent.act(state, deterministic=True)
            next_state, reward, done, info = env.step(action)

            global_step += 1
            frames.append(make_frame(env, t=global_step * step_dt, last_yaws=last_yaws))

            state = next_state

            if global_step >= args.steps:
                break

        if global_step >= args.steps:
            break

    trace = {
        "step_dt": step_dt,
        "frames": frames,
        "meta": {
            "model": str(model_path),
            "seed": args.seed,
            "episodes": args.episodes,
            "max_steps": args.steps,
            "num_uavs": env.cfg.num_uavs,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)

    print(f"[OK] Exported Gazebo trace to: {out_path}")
    print(f"[INFO] frames={len(frames)}, step_dt={step_dt}, num_uavs={env.cfg.num_uavs}")


if __name__ == "__main__":
    main()
    