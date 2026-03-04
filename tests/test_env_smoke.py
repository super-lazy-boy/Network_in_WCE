from uav_rl.config import EnvConfig
from uav_rl.env import UAVNetworkEnv


def test_env_reset_and_step():
    env = UAVNetworkEnv(EnvConfig(num_uavs=5, max_episode_steps=5), seed=1)
    state = env.reset()
    assert state.shape[0] == env.state_size

    for _ in range(5):
        next_state, reward, done, info = env.step(0)
        assert next_state.shape == state.shape
        assert isinstance(reward, float)
        assert "throughput" in info
        state = next_state
    assert done is True
