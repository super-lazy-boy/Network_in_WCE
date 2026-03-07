import argparse
import sys
import numpy as np
sys.path.append("E:/code/Network_in_WCE")
from uav_rl.agent import DQNAgent
from uav_rl.config import EnvConfig
from uav_rl.env import UAVNetworkEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = UAVNetworkEnv(EnvConfig())
    agent = DQNAgent(env.state_size, env.action_size)
    agent.load(args.model)

    rewards = []
    for _ in range(args.episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.act(s, deterministic=True)
            s, r, done, _ = env.step(a)
            total += r
        rewards.append(total)

    print(f"mean_reward={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")


if __name__ == "__main__":
    main()
