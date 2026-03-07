# scripts/export_ns3_trace.py
import argparse
import json
from pathlib import Path
import sys
sys.path.append("/home/zaq/code/Network_in_WCE")
from uav_rl.agent import DQNAgent
from uav_rl.config import EnvConfig
from uav_rl.env import UAVNetworkEnv
from uav_rl.ns3_bridge import Ns3Bridge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", type=str, default="artifacts/ns3_trace.json")
    args = parser.parse_args()

    env = UAVNetworkEnv(EnvConfig())
    agent = DQNAgent(env.state_size, env.action_size)
    agent.load(args.model)
    bridge = Ns3Bridge()

    s = env.reset()
    timeline = []
    for t in range(args.steps):
        a = agent.act(s, deterministic=True)
        s, r, done, info = env.step(a)

        topo = bridge.export_topology(env.positions, env.link_matrix)
        power = bridge.export_power_schedule(env.tx_powers)

        timeline.append({
            "t": t,
            "reward": r,
            "info": info,
            "topology": topo,
            "power": power,
        })
        if done:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frames": timeline}, indent=2), encoding="utf-8")
    print(f"saved: {out}")

if __name__ == "__main__":
    main()