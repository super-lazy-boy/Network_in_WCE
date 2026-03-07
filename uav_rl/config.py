from dataclasses import dataclass


@dataclass
class EnvConfig:
    # =========================
    # 基础环境参数
    # =========================
    num_uavs: int = 8
    grid_size: float = 1200.0
    max_height: float = 300.0
    min_height: float = 50.0
    max_episode_steps: int = 1000

    # 仿真步长与运动参数
    dt: float = 1.0
    speed: float = 18.0

    # =========================
    # 无线通信参数
    # =========================
    weak_link_threshold_dbm: float = -86.0   # 弱连接判定阈值
    max_neighbors: int = 3
    carrier_freq_hz: float = 2.4e9
    noise_density_dbm_hz: float = -174.0
    total_bandwidth_hz: float = 20e6

    # 发射功率范围
    max_tx_power_w: float = 1.2
    min_tx_power_w: float = 0.01

    # 碰撞判定距离
    collision_distance: float = 8.0

    # =========================
    # 信道 / 衰落模型参数
    # =========================
    # 可选: 'free_space', 'rician', 'rayleigh'
    path_loss_model: str = "rician"
    rician_factor: float = 10.0
    # 阴影衰落标准差（dB）
    shadowing_std_db: float = 4.0

    # 参考距离与参考路径损耗
    reference_distance_m: float = 1.0
    reference_loss_db: float = 40.0

    # 对数距离路径损耗指数
    path_loss_exponent: float = 2.4

    # Rician K 因子（线性域）
    rician_k_linear: float = 6.0

    # 链路状态马尔可夫过程：好/坏信道持续性
    channel_good_to_bad: float = 0.03
    channel_bad_to_good: float = 0.10
    deep_fade_extra_loss_db: float = 12.0

    # 时间相关衰落平滑系数
    fading_memory: float = 0.85

    # =========================
    # Reward 设计参数（论文级分层奖励）
    # =========================

    # 吞吐量归一化压缩因子：
    # thr_norm = thr / (thr + reward_thr_scale_mbps)
    # 作用：防止吞吐量数值过大导致 reward 被单一目标主导
    reward_thr_scale_mbps: float = 20.0

    # 连通性门限：
    # 当 connectivity < min_connectivity_ratio 时，奖励函数进入“修复拓扑模式”
    min_connectivity_ratio: float = 0.75

    # 终局判定成功时的连通率阈值
    final_success_connectivity: float = 0.875

    # -------------------------
    # 断连/弱连通模式下的奖励参数
    # -------------------------
    # 基础断连惩罚
    reward_disconnect_base: float = 2.0

    # 连通性修复奖励（对 conn_gain 的奖励）
    reward_conn_repair: float = 5.0

    # 断连状态下，允许保留少量路径推进激励
    reward_progress_under_disconnect: float = 0.3

    # 断连状态下的能耗惩罚权重
    reward_energy_weight_under_disconnect: float = 0.2

    # 断连状态下的弱链路惩罚权重
    reward_weak_weight_under_disconnect: float = 1.0

    # -------------------------
    # 连通达标模式下的主奖励参数
    # -------------------------
    # 吞吐量奖励权重
    reward_throughput_weight: float = 2.0

    # 连通性奖励权重
    reward_connectivity_weight: float = 2.5

    # 路径推进奖励权重
    reward_progress_weight: float = 0.5

    # 连通性改善增量奖励权重
    reward_conn_gain_weight: float = 1.2

    # 弱链路改善增量奖励权重
    reward_weak_improve_weight: float = 0.8

    # -------------------------
    # 通用惩罚项权重
    # -------------------------
    # 弱链路比例惩罚
    reward_weak_weight: float = 1.0

    # 能耗惩罚
    reward_energy_weight: float = 0.3

    # 碰撞惩罚
    reward_collision_weight: float = 4.0

    # -------------------------
    # 终局奖励项
    # -------------------------
    # 终局成功奖励（按 thr_norm 放大）
    reward_terminal_success: float = 2.5

    # 终局失败惩罚
    reward_terminal_failure: float = 2.5

    # 群集运动参数
    neighbor_radius_m: float = 220.0
    separation_radius_m: float = 60.0
    alignment_weight: float = 0.30
    cohesion_weight: float = 0.25
    separation_weight: float = 0.35
    target_weight: float = 0.35
    rl_direction_weight: float = 0.15
    jitter_std: float = 0.08


@dataclass
class TrainingConfig:
    # =========================
    # 训练轮数
    # =========================
    episodes: int = 2000

    # 经验池 warmup：让 buffer 先积累足够多样本再开始学习
    warmup_steps: int = 5000

    # 批大小
    batch_size: int = 128

    # 折扣因子
    gamma: float = 0.99

    # 学习率
    lr: float = 3e-4

    # 旧版硬更新参数，保留兼容；如果你已改为 soft update，则此参数基本不用
    target_update_interval: int = 350

    # =========================
    # 探索策略参数
    # =========================
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05

    # 放慢 epsilon 衰减，避免在复杂环境中过早贪心
    epsilon_decay_steps: int = 150_000

    # =========================
    # Replay Buffer
    # =========================
    replay_capacity: int = 150_000

    # =========================
    # Soft target update
    # =========================
    # target = (1 - tau) * target + tau * online
    tau: float = 0.005

    # =========================
    # Reward scaling
    # =========================
    # 若环境 reward 已基本控制在 [-5, 5]，可设为 1.0
    # 若实际训练中 Q 值仍波动很大，可改成 0.5 或 0.2
    reward_scale: float = 1.0

    # 随机种子
    seed: int = 42