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
        self.shadowing_db = np.zeros((self.cfg.num_uavs, self.cfg.num_uavs), dtype=np.float32)
        self.fast_fading_db = np.zeros((self.cfg.num_uavs, self.cfg.num_uavs), dtype=np.float32)
        self.channel_bad_state = np.zeros((self.cfg.num_uavs, self.cfg.num_uavs), dtype=np.int32)
        self.velocities = np.zeros((self.cfg.num_uavs, 3), dtype=np.float32)

    @property
    def state_size(self) -> int:
        # 每个 UAV:
        # pos(3) + target(3) + power(1) + bw(1) + degree(1) = 9
        # 全局:
        # link_matrix(n*n) + connectivity(1) + weak_ratio(1)
        return self.cfg.num_uavs * 9 + self.cfg.num_uavs * self.cfg.num_uavs + 2

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
        self.shadowing_db = self.rng.normal(
            0.0, self.cfg.shadowing_std_db, size=(self.cfg.num_uavs, self.cfg.num_uavs)
        ).astype(np.float32)
        self.fast_fading_db.fill(0.0)
        self.channel_bad_state.fill(0)
        self.shadowing_db = 0.5 * (self.shadowing_db + self.shadowing_db.T)
        np.fill_diagonal(self.shadowing_db, 0.0)
        self.velocities = self.rng.normal(0.0, 1.0, size=(self.cfg.num_uavs, 3)).astype(np.float32)
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        move_id, power_id, bw_id, topo_id = self._decode_action(action)
        self.step_count += 1

        # ---------- 1) 记录动作前状态，用于构造“增量奖励” ----------
        prev_connectivity = self._largest_component_ratio()
        prev_avg_target_dist = self._mean_target_distance()
        prev_weak_ratio = self._estimate_network_metrics()[1]

        # ---------- 2) 执行动作 ----------
        self._apply_motion(move_id)
        self._apply_radio_controls(power_id, bw_id)
        self._update_links(topology_policy=topo_id)

        # ---------- 3) 计算动作后网络指标 ----------
        throughput_mbps, weak_ratio = self._estimate_network_metrics()
        connectivity = self._largest_component_ratio()
        collision_penalty = self._collision_penalty()
        energy_penalty = float(np.mean(self.tx_powers / self.cfg.max_tx_power_w))
        avg_target_dist = self._mean_target_distance()

        # ---------- 4) 构造归一化指标 ----------
        # 吞吐量压缩到 [0,1) 区间，避免 reward 尺度过大
        thr_norm = throughput_mbps / (throughput_mbps + self.cfg.reward_thr_scale_mbps)

        # 连通性本身就在 [0,1]
        conn_norm = connectivity

        # 弱链路、能耗、碰撞，本身越大越差
        weak_pen = np.clip(weak_ratio, 0.0, 1.0)
        energy_pen = np.clip(energy_penalty, 0.0, 1.0)
        collision_pen = np.clip(collision_penalty, 0.0, 1.0)

        # 路径推进改成“距离改善量”，比 exp(-distance) 更稳定
        progress_gain = (prev_avg_target_dist - avg_target_dist) / (self.cfg.grid_size + 1e-6)

        # 连通性改善量：如果动作让网络更连通，应明确奖励
        conn_gain = connectivity - prev_connectivity

        # 弱链路改善量：如果弱链路减少，也应奖励
        weak_gain = prev_weak_ratio - weak_ratio

        # ---------- 5) 分层 reward 设计 ----------
        # 第一层：安全与连通性优先
        # 如果连通率低于阈值，主要任务不是冲吞吐，而是先修复拓扑
        if connectivity < self.cfg.min_connectivity_ratio:
            reward = (
                - self.cfg.reward_disconnect_base
                + self.cfg.reward_conn_repair * conn_gain
                + self.cfg.reward_progress_under_disconnect * progress_gain
                - self.cfg.reward_collision_weight * collision_pen
                - self.cfg.reward_energy_weight_under_disconnect * energy_pen
                - self.cfg.reward_weak_weight_under_disconnect * weak_pen
            )
        else:
            # 第二层：在连通性达标后，再鼓励高吞吐、低弱链路、低能耗
            reward = (
                self.cfg.reward_throughput_weight * thr_norm
                + self.cfg.reward_connectivity_weight * conn_norm
                + self.cfg.reward_progress_weight * progress_gain
                + self.cfg.reward_conn_gain_weight * conn_gain
                + self.cfg.reward_weak_improve_weight * weak_gain
                - self.cfg.reward_weak_weight * weak_pen
                - self.cfg.reward_energy_weight * energy_pen
                - self.cfg.reward_collision_weight * collision_pen
            )

        # ---------- 6) 终局奖励/惩罚 ----------
        done = self.step_count >= self.cfg.max_episode_steps
        if done:
            if connectivity >= self.cfg.final_success_connectivity:
                reward += self.cfg.reward_terminal_success * thr_norm
            else:
                reward -= self.cfg.reward_terminal_failure

        info = {
            "throughput": float(throughput_mbps),
            "throughput_norm": float(thr_norm),
            "weak_link_ratio": float(weak_ratio),
            "connectivity": float(connectivity),
            "energy_penalty": float(energy_penalty),
            "collision_penalty": float(collision_penalty),
            "progress_gain": float(progress_gain),
            "conn_gain": float(conn_gain),
            "avg_target_dist": float(avg_target_dist),
            "reward_raw": float(reward),
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
        """
        基于群集运动 + RL方向残差控制 的移动模型。
        这样弱通联环境中的拓扑变化会更自然，也更接近文献中的群集移动思想。
        """
        rl_dir = self.movement_vectors[move_id].astype(np.float32)
        n = self.cfg.num_uavs
        dist = self._pairwise_distance()
        new_vel = np.zeros_like(self.velocities)

        for i in range(n):
            nbrs = np.where((dist[i] < self.cfg.neighbor_radius_m) & (dist[i] > 1e-6))[0]

            alignment = np.zeros(3, dtype=np.float32)
            cohesion = np.zeros(3, dtype=np.float32)
            separation = np.zeros(3, dtype=np.float32)

            if len(nbrs) > 0:
                # 对齐：趋向邻居平均速度
                alignment = np.mean(self.velocities[nbrs], axis=0)

                # 凝聚：趋向邻居质心
                center = np.mean(self.positions[nbrs], axis=0)
                cohesion = center - self.positions[i]

                # 分离：避免过近碰撞
                close_nbrs = np.where((dist[i] < self.cfg.separation_radius_m) & (dist[i] > 1e-6))[0]
                if len(close_nbrs) > 0:
                    sep_vecs = self.positions[i] - self.positions[close_nbrs]
                    sep_norm = np.linalg.norm(sep_vecs, axis=1, keepdims=True) + 1e-6
                    separation = np.sum(sep_vecs / sep_norm, axis=0)

            target_vec = self.targets[i] - self.positions[i]
            target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-6)

            alignment = alignment / (np.linalg.norm(alignment) + 1e-6)
            cohesion = cohesion / (np.linalg.norm(cohesion) + 1e-6)
            separation = separation / (np.linalg.norm(separation) + 1e-6)

            jitter = self.rng.normal(0, self.cfg.jitter_std, size=3).astype(np.float32)

            v = (
                self.cfg.alignment_weight * alignment
                + self.cfg.cohesion_weight * cohesion
                + self.cfg.separation_weight * separation
                + self.cfg.target_weight * target_vec
                + self.cfg.rl_direction_weight * rl_dir
                + jitter
            )

            v = v / (np.linalg.norm(v) + 1e-6)
            new_vel[i] = v

        self.velocities = new_vel
        self.positions += self.velocities * self.cfg.speed * self.cfg.dt

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
        dist = self._pairwise_distance()
        rx_dbm = self._rx_power_dbm(dist)
        avg_link_margin = np.mean(rx_dbm - self.cfg.weak_link_threshold_dbm, axis=1)
        urgency = 1.0 / (np.maximum(avg_link_margin, -20.0) + 25.0)
        urgency = np.clip(urgency, 1e-3, None)
        urgency /= urgency.sum()
        split = self.bandwidth_splits[bw_id]
        uniform = np.full(self.cfg.num_uavs, 1.0 / self.cfg.num_uavs)
        weights = split * urgency + (1.0 - split) * uniform
        self.bandwidth_allocation = weights * self.cfg.total_bandwidth_hz

    def _update_links(self, topology_policy: int) -> None:
        """
        综合考虑：
        - 接收功率 / 链路质量
        - 相对速度（越小越稳定）
        - 邻居重叠度（越高越利于稳定拓扑）
        - 能量负担（避免总压榨少数节点）
        """
        n = self.cfg.num_uavs
        self.link_matrix.fill(0.0)
        dist = self._pairwise_distance()
        rx_power_dbm = self._rx_power_dbm(dist)

        # 先计算邻居集合，用于邻居重叠度
        candidate_sets = []
        for i in range(n):
            cand = set(np.where((dist[i] < self.cfg.neighbor_radius_m) & (np.arange(n) != i))[0].tolist())
            candidate_sets.append(cand)

        for i in range(n):
            scores = []
            for j in range(n):
                if i == j:
                    continue
                if rx_power_dbm[i, j] < self.cfg.weak_link_threshold_dbm - 8.0:
                    continue

                # 链路质量分
                link_quality = rx_power_dbm[i, j]

                # 相对速度分：越小越稳定
                rel_speed = np.linalg.norm(self.velocities[i] - self.velocities[j])

                # 邻居相关度：公共邻居越多，越有利于形成稳定局部结构
                inter = len(candidate_sets[i].intersection(candidate_sets[j]))
                union = max(1, len(candidate_sets[i].union(candidate_sets[j])))
                overlap = inter / union

                # 能量分：更偏向剩余“低消耗”的节点
                energy_score = 1.0 - float(self.tx_powers[j] / self.cfg.max_tx_power_w)

                # 不同 topology policy 影响打分侧重点
                if topology_policy == 0:
                    score = 1.0 * link_quality - 2.0 * rel_speed + 8.0 * overlap + 2.0 * energy_score
                elif topology_policy == 1:
                    alt_pen = abs(self.positions[i, 2] - self.positions[j, 2])
                    score = 1.0 * link_quality - 2.0 * rel_speed + 8.0 * overlap + 2.0 * energy_score - 0.05 * alt_pen
                else:
                    # 韧性优先：更看重覆盖和可恢复性
                    score = 0.8 * link_quality - 1.5 * rel_speed + 10.0 * overlap + 3.0 * energy_score

                scores.append((score, j))

            scores.sort(key=lambda x: x[0], reverse=True)
            for _, j in scores[: self.cfg.max_neighbors]:
                self.link_matrix[i, j] = 1.0

    def _pairwise_distance(self) -> np.ndarray:
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
        return dist
    
    def _update_channel_states(self) -> None:
        """
        使用简化的马尔可夫链更新链路好/坏状态，
        用于模拟弱通联环境中的深衰落持续性，而不是每一步完全独立随机。
        """
        n = self.cfg.num_uavs
        for i in range(n):
            for j in range(i + 1, n):
                bad = self.channel_bad_state[i, j]
                if bad == 0:
                    if self.rng.random() < self.cfg.channel_good_to_bad:
                        self.channel_bad_state[i, j] = 1
                        self.channel_bad_state[j, i] = 1
                else:
                    if self.rng.random() < self.cfg.channel_bad_to_good:
                        self.channel_bad_state[i, j] = 0
                        self.channel_bad_state[j, i] = 0

    def _rx_power_dbm(self, distance: np.ndarray) -> np.ndarray:
        """
        更真实的接收功率模型：
        接收功率 = 发射功率 - 大尺度路径损耗 - 阴影衰落 + 小尺度衰落 - 深衰落附加损耗

        支持：
        - free_space
        - log_distance
        - rician
        - rayleigh
        """
        d = np.maximum(distance, self.cfg.reference_distance_m)

        # ---------- 1) 大尺度路径损耗 ----------
        if self.cfg.path_loss_model == "free_space":
            c = 3e8
            lam = c / self.cfg.carrier_freq_hz
            path_loss_db = 20.0 * np.log10(4 * np.pi * d / lam)

        elif self.cfg.path_loss_model in ["log_distance", "rician", "rayleigh"]:
            path_loss_db = (
                self.cfg.reference_loss_db
                + 10.0 * self.cfg.path_loss_exponent * np.log10(d / self.cfg.reference_distance_m)
            )
        else:
            raise ValueError(f"Unknown path loss model: {self.cfg.path_loss_model}")

        # ---------- 2) 小尺度衰落 ----------
        # 采用时间平滑，避免每一步完全独立随机，增强时间相关性
        n = self.cfg.num_uavs
        fading_new = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                if self.cfg.path_loss_model == "rician":
                    K = self.cfg.rician_k_linear
                    # 复高斯+LOS分量近似
                    los_amp = np.sqrt(K / (K + 1.0))
                    nlos_std = np.sqrt(1.0 / (2.0 * (K + 1.0)))
                    h_real = los_amp + self.rng.normal(0.0, nlos_std)
                    h_imag = self.rng.normal(0.0, nlos_std)
                    amp = np.sqrt(h_real * h_real + h_imag * h_imag)
                elif self.cfg.path_loss_model == "rayleigh":
                    h_real = self.rng.normal(0.0, 1.0 / np.sqrt(2.0))
                    h_imag = self.rng.normal(0.0, 1.0 / np.sqrt(2.0))
                    amp = np.sqrt(h_real * h_real + h_imag * h_imag)
                else:
                    amp = 1.0

                fading_db = 20.0 * np.log10(max(amp, 1e-6))
                fading_new[i, j] = fading_db
                fading_new[j, i] = fading_db

        self.fast_fading_db = (
            self.cfg.fading_memory * self.fast_fading_db
            + (1.0 - self.cfg.fading_memory) * fading_new
        )

        # ---------- 3) 深衰落坏状态 ----------
        self._update_channel_states()
        deep_fade_penalty = self.channel_bad_state.astype(np.float32) * self.cfg.deep_fade_extra_loss_db

        # ---------- 4) 接收功率 ----------
        tx_dbm = 30.0 + 10.0 * np.log10(np.maximum(self.tx_powers, 1e-6))
        rx_dbm = (
            tx_dbm[:, None]
            - path_loss_db
            - self.shadowing_db
            + self.fast_fading_db
            - deep_fade_penalty
        )

        np.fill_diagonal(rx_dbm, -1e9)
        return rx_dbm

    def _estimate_network_metrics(self) -> Tuple[float, float]:
        """
        使用简化 SINR 估计链路速率，而不是只用 SNR。
        干扰项来自其他同时发射节点的泄漏功率。
        """
        dist = self._pairwise_distance()
        rx_dbm = self._rx_power_dbm(dist)

        rates = []
        weak_links = 0
        total_links = int(np.sum(self.link_matrix)) + 1

        # 噪声功率（按每条链路实际带宽计算）
        noise_density_linear_w_hz = 10 ** ((self.cfg.noise_density_dbm_hz - 30) / 10)

        for i in range(self.cfg.num_uavs):
            neighbors = np.where(self.link_matrix[i] > 0)[0]
            if len(neighbors) == 0:
                continue

            bw_share = self.bandwidth_allocation[i] / len(neighbors)

            for j in neighbors:
                signal_w = 10 ** ((rx_dbm[i, j] - 30) / 10)
                noise_w = noise_density_linear_w_hz * bw_share

                # 干扰项：所有其他发射机 k->j 的接收功率累计
                interference_w = 0.0
                for k in range(self.cfg.num_uavs):
                    if k == i:
                        continue
                    # 只要 k 当前也有出边，就视作可能发射干扰
                    if np.sum(self.link_matrix[k]) > 0:
                        interference_w += 10 ** ((rx_dbm[k, j] - 30) / 10)

                sinr = signal_w / max(noise_w + interference_w, 1e-12)

                if rx_dbm[i, j] < self.cfg.weak_link_threshold_dbm:
                    weak_links += 1

                rate = bw_share * np.log2(1.0 + max(sinr, 1e-8))
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

    def _mean_target_distance(self) -> float:
        """
        计算所有无人机到各自目标点的平均距离。
        用于构造逐步推进的增量奖励，比直接用 exp(-distance) 更适合强化学习。
        """
        d = np.linalg.norm(self.targets - self.positions, axis=1)
        return float(np.mean(d))

    # def _path_progress_reward(self) -> float:
    #     d = np.linalg.norm(self.targets - self.positions, axis=1)
    #     return float(np.exp(-np.mean(d) / self.cfg.grid_size))

    def _collision_penalty(self) -> float:
        d = self._pairwise_distance()
        np.fill_diagonal(d, 1e6)
        collisions = np.sum(d < self.cfg.collision_distance)
        return float(collisions / (self.cfg.num_uavs * (self.cfg.num_uavs - 1)))

    def _build_state(self) -> np.ndarray:
        """
        状态向量包含：
        1. 每个 UAV 的归一化位置 / 目标 / 发射功率 / 带宽 / 节点度
        2. 整体链路矩阵
        3. 全局连通率、弱链路比例

        这样智能体不仅看到几何位置，也能显式感知当前拓扑质量。
        """
        pos_norm = self.positions / np.array(
            [self.cfg.grid_size, self.cfg.grid_size, self.cfg.max_height],
            dtype=np.float32,
        )
        tgt_norm = self.targets / np.array(
            [self.cfg.grid_size, self.cfg.grid_size, self.cfg.max_height],
            dtype=np.float32,
        )
        p_norm = (self.tx_powers / self.cfg.max_tx_power_w)[:, None]
        bw_norm = (self.bandwidth_allocation / self.cfg.total_bandwidth_hz)[:, None]

        # 节点度：每个节点当前有多少邻居
        degree = np.sum(self.link_matrix > 0, axis=1, keepdims=True).astype(np.float32)
        degree_norm = degree / max(1.0, self.cfg.num_uavs - 1)

        per_uav = np.concatenate(
            [pos_norm, tgt_norm, p_norm, bw_norm, degree_norm],
            axis=1,
        ).flatten()

        connectivity = np.array([self._largest_component_ratio()], dtype=np.float32)
        weak_ratio = np.array([self._estimate_network_metrics()[1]], dtype=np.float32)

        return np.concatenate(
            [per_uav, self.link_matrix.flatten().astype(np.float32), connectivity, weak_ratio]
        ).astype(np.float32)

    def export_config(self) -> Dict:
        return asdict(self.cfg)
