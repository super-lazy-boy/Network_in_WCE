from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from .config import EnvConfig


class UAVNetworkEnv:
    """Centralized UAV swarm env with weak-connectivity constraints.

    Action space is a single discrete integer encoding:
    - global movement strategy (7 choices)
    - power profile (3 choices)
    - bandwidth scheduling policy (3 choices)
    - topology policy (3 choices)
    """

    def __init__(self, config: EnvConfig | None = None, seed: int = 0):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)

        self.movement_vectors = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float32,
        )
        self.power_levels = np.array([0.35, 0.65, 1.0], dtype=np.float32)
        self.bandwidth_splits = np.array([0.6, 0.8, 1.0], dtype=np.float32)
        self.action_size = (
            len(self.movement_vectors)
            * len(self.power_levels)
            * len(self.bandwidth_splits)
            * 3
        )

        self.positions = np.zeros((self.cfg.num_uavs, 3), dtype=np.float32)
        self.targets = np.zeros((self.cfg.num_uavs, 3), dtype=np.float32)
        self.tx_powers = np.full(self.cfg.num_uavs, self.cfg.min_tx_power_w, dtype=np.float32)
        self.bandwidth_allocation = np.full(
            self.cfg.num_uavs, self.cfg.total_bandwidth_hz / self.cfg.num_uavs, dtype=np.float32
        )
        self.link_matrix = np.zeros((self.cfg.num_uavs, self.cfg.num_uavs), dtype=np.float32)
        self.step_count = 0

    @property
    def state_size(self) -> int:
        return self.cfg.num_uavs * 8 + self.cfg.num_uavs * self.cfg.num_uavs

    def reset(self) -> np.ndarray:
        self.positions = self.rng.uniform(
            low=[0, 0, self.cfg.min_height],
            high=[self.cfg.grid_size, self.cfg.grid_size, self.cfg.max_height],
            size=(self.cfg.num_uavs, 3),
        ).astype(np.float32)
        self.targets = self.rng.uniform(
            low=[0, 0, self.cfg.min_height],
            high=[self.cfg.grid_size, self.cfg.grid_size, self.cfg.max_height],
            size=(self.cfg.num_uavs, 3),
        ).astype(np.float32)
        self.tx_powers.fill(self.cfg.min_tx_power_w)
        self.bandwidth_allocation.fill(self.cfg.total_bandwidth_hz / self.cfg.num_uavs)
        self.link_matrix.fill(0.0)
        self.step_count = 0
        self._update_links(topology_policy=0)
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        move_id, power_id, bw_id, topo_id = self._decode_action(action)
        self.step_count += 1

        self._apply_motion(move_id)
        self._apply_radio_controls(power_id, bw_id)
        self._update_links(topology_policy=topo_id)

        throughput, weak_ratio = self._estimate_network_metrics()
        collision_penalty = self._collision_penalty()
        progress_reward = self._path_progress_reward()
        energy_penalty = float(np.mean(self.tx_powers / self.cfg.max_tx_power_w))
        connectivity_bonus = self._largest_component_ratio()

        reward = (
            2.5 * throughput
            + 2.0 * connectivity_bonus
            + 0.5 * progress_reward
            - 1.5 * weak_ratio
            - 1.2 * energy_penalty
            - 2.0 * collision_penalty
        )

        done = self.step_count >= self.cfg.max_episode_steps
        info = {
            "throughput": throughput,
            "weak_link_ratio": weak_ratio,
            "connectivity": connectivity_bonus,
            "energy_penalty": energy_penalty,
            "collision_penalty": collision_penalty,
        }
        return self._build_state(), float(reward), done, info

    def _decode_action(self, action: int) -> Tuple[int, int, int, int]:
        a = int(action) % self.action_size
        topo_id = a % 3
        a //= 3
        bw_id = a % len(self.bandwidth_splits)
        a //= len(self.bandwidth_splits)
        power_id = a % len(self.power_levels)
        a //= len(self.power_levels)
        move_id = a % len(self.movement_vectors)
        return move_id, power_id, bw_id, topo_id

    def _apply_motion(self, move_id: int) -> None:
        direction = self.movement_vectors[move_id]
        random_jitter = self.rng.normal(0, 0.15, size=self.positions.shape).astype(np.float32)
        heading = self.targets - self.positions
        norms = np.linalg.norm(heading, axis=1, keepdims=True) + 1e-6
        heading = heading / norms
        velocity = 0.65 * heading + 0.35 * direction + random_jitter
        self.positions += velocity * self.cfg.speed * self.cfg.dt

        self.positions[:, 0:2] = np.clip(self.positions[:, 0:2], 0, self.cfg.grid_size)
        self.positions[:, 2] = np.clip(self.positions[:, 2], self.cfg.min_height, self.cfg.max_height)

        reached = np.linalg.norm(self.targets - self.positions, axis=1) < 30.0
        if np.any(reached):
            self.targets[reached] = self.rng.uniform(
                low=[0, 0, self.cfg.min_height],
                high=[self.cfg.grid_size, self.cfg.grid_size, self.cfg.max_height],
                size=(int(np.sum(reached)), 3),
            )

    def _apply_radio_controls(self, power_id: int, bw_id: int) -> None:
        power_scale = self.power_levels[power_id]
        self.tx_powers[:] = np.clip(
            power_scale * self.cfg.max_tx_power_w,
            self.cfg.min_tx_power_w,
            self.cfg.max_tx_power_w,
        )

        distances_to_center = np.linalg.norm(
            self.positions - np.array([self.cfg.grid_size / 2, self.cfg.grid_size / 2, 150.0], dtype=np.float32),
            axis=1,
        )
        urgency = 1.0 / (distances_to_center + 1.0)
        urgency /= urgency.sum()
        split = self.bandwidth_splits[bw_id]
        uniform = np.full(self.cfg.num_uavs, 1.0 / self.cfg.num_uavs)
        weights = split * urgency + (1.0 - split) * uniform
        self.bandwidth_allocation = weights * self.cfg.total_bandwidth_hz

    def _update_links(self, topology_policy: int) -> None:
        n = self.cfg.num_uavs
        self.link_matrix.fill(0.0)
        dist = self._pairwise_distance()
        rx_power_dbm = self._rx_power_dbm(dist)

        for i in range(n):
            candidates = np.argsort(dist[i])
            selected = 0
            for j in candidates:
                if i == j:
                    continue
                if rx_power_dbm[i, j] < self.cfg.weak_link_threshold_dbm:
                    continue
                if topology_policy == 0 and dist[i, j] > 350:
                    continue
                if topology_policy == 1 and abs(self.positions[i, 2] - self.positions[j, 2]) > 80:
                    continue
                self.link_matrix[i, j] = 1.0
                selected += 1
                if selected >= self.cfg.max_neighbors:
                    break

            if topology_policy == 2 and selected == 0:
                # resiliency policy: ensure fallback weak edge
                j = int(np.argsort(dist[i])[1])
                self.link_matrix[i, j] = 1.0

    def _pairwise_distance(self) -> np.ndarray:
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
        return dist

    def _rx_power_dbm(self, distance: np.ndarray) -> np.ndarray:
        c = 3e8
        lam = c / self.cfg.carrier_freq_hz
        d = np.maximum(distance, 1.0)
        path_loss_db = 20.0 * np.log10(4 * np.pi * d / lam)
        tx_dbm = 30.0 + 10.0 * np.log10(np.maximum(self.tx_powers, 1e-6))
        return tx_dbm[:, None] - path_loss_db

    def _estimate_network_metrics(self) -> Tuple[float, float]:
        dist = self._pairwise_distance()
        rx_dbm = self._rx_power_dbm(dist)
        noise_dbm = self.cfg.noise_density_dbm_hz + 10 * np.log10(
            self.bandwidth_allocation.mean() + 1e-9
        )
        snr_linear = 10 ** ((rx_dbm - noise_dbm) / 10)

        rates = []
        weak_links = 0
        total_links = int(np.sum(self.link_matrix)) + 1
        for i in range(self.cfg.num_uavs):
            neighbors = np.where(self.link_matrix[i] > 0)[0]
            if len(neighbors) == 0:
                continue
            bw_share = self.bandwidth_allocation[i] / len(neighbors)
            for j in neighbors:
                if rx_dbm[i, j] < self.cfg.weak_link_threshold_dbm:
                    weak_links += 1
                rate = bw_share * np.log2(1 + max(snr_linear[i, j], 1e-8))
                rates.append(rate)

        net_rate = float(np.mean(rates) / 1e6) if rates else 0.0
        weak_ratio = weak_links / total_links
        return net_rate, weak_ratio

    def _largest_component_ratio(self) -> float:
        n = self.cfg.num_uavs
        visited = np.zeros(n, dtype=bool)
        best = 0

        undirected = (self.link_matrix + self.link_matrix.T) > 0
        for s in range(n):
            if visited[s]:
                continue
            stack = [s]
            visited[s] = True
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                for nxt in np.where(undirected[node])[0]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(int(nxt))
            best = max(best, size)
        return best / n

    def _path_progress_reward(self) -> float:
        d = np.linalg.norm(self.targets - self.positions, axis=1)
        return float(np.exp(-np.mean(d) / self.cfg.grid_size))

    def _collision_penalty(self) -> float:
        d = self._pairwise_distance()
        np.fill_diagonal(d, 1e6)
        collisions = np.sum(d < self.cfg.collision_distance)
        return float(collisions / (self.cfg.num_uavs * (self.cfg.num_uavs - 1)))

    def _build_state(self) -> np.ndarray:
        pos_norm = self.positions / np.array([self.cfg.grid_size, self.cfg.grid_size, self.cfg.max_height])
        tgt_norm = self.targets / np.array([self.cfg.grid_size, self.cfg.grid_size, self.cfg.max_height])
        p_norm = (self.tx_powers / self.cfg.max_tx_power_w)[:, None]
        bw_norm = (self.bandwidth_allocation / self.cfg.total_bandwidth_hz)[:, None]
        per_uav = np.concatenate([pos_norm, tgt_norm, p_norm, bw_norm], axis=1).flatten()
        return np.concatenate([per_uav, self.link_matrix.flatten()]).astype(np.float32)

    def export_config(self) -> Dict:
        return asdict(self.cfg)
