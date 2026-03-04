from dataclasses import dataclass


@dataclass
class EnvConfig:
    num_uavs: int = 8
    grid_size: float = 1200.0
    max_height: float = 300.0
    min_height: float = 50.0
    max_episode_steps: int = 250
    dt: float = 1.0
    speed: float = 18.0
    weak_link_threshold_dbm: float = -86.0
    max_neighbors: int = 3
    carrier_freq_hz: float = 2.4e9
    noise_density_dbm_hz: float = -174.0
    total_bandwidth_hz: float = 20e6
    max_tx_power_w: float = 1.2
    min_tx_power_w: float = 0.1
    collision_distance: float = 8.0


@dataclass
class TrainingConfig:
    episodes: int = 400
    warmup_steps: int = 2500
    batch_size: int = 128
    gamma: float = 0.99
    lr: float = 3e-4
    target_update_interval: int = 350
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000
    replay_capacity: int = 150_000
    seed: int = 42
