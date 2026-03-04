"""Gazebo adapter skeleton.

This bridge keeps the RL interface stable while allowing deployment to Gazebo/ROS2.
Use `build_waypoint_message` output in your ROS2 publisher node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class WaypointCommand:
    uav_id: int
    x: float
    y: float
    z: float
    velocity: float


class GazeboBridge:
    def __init__(self, max_velocity: float = 20.0):
        self.max_velocity = max_velocity

    def build_waypoint_message(self, positions: np.ndarray, targets: np.ndarray) -> List[WaypointCommand]:
        commands: List[WaypointCommand] = []
        for i, (p, t) in enumerate(zip(positions, targets)):
            direction = t - p
            dist = float(np.linalg.norm(direction) + 1e-6)
            speed = min(self.max_velocity, 0.5 * dist)
            commands.append(WaypointCommand(i, float(t[0]), float(t[1]), float(t[2]), speed))
        return commands

    def parse_feedback(self, pose_dict: Dict[int, List[float]]) -> np.ndarray:
        ids = sorted(pose_dict.keys())
        return np.array([pose_dict[i] for i in ids], dtype=np.float32)
