# UAV Swarm Weak-Connectivity RL (DQN)

本项目实现了一个面向**弱通联无人集群**的 3D 网络环境 `UAVNetworkEnv`，并提供了可训练的 DQN（Dueling + Double DQN）智能体，用于联合优化：

- 动态链路调度（邻居选择 / 拓扑策略）
- 带宽与发射功率动态分配
- 无人机路径动态调整（目标跟踪 + 全局动作策略）

同时给出与 **Gazebo** 和 **NS-3** 的桥接模块（工程化接口），便于后续接入高保真物理仿真与网络协议仿真。

## 1. 项目结构

```text
uav_rl/
  config.py            # 环境/训练超参数
  env.py               # UAVNetworkEnv核心仿真
  agent.py             # Dueling Double DQN
  replay_buffer.py
  trainer.py           # 训练循环
  gazebo_bridge.py     # Gazebo/ROS2 接口适配
  ns3_bridge.py        # NS-3 拓扑与功率导出
scripts/
  train.py             # 训练入口
  evaluate.py          # 推理评估入口
tests/
  test_env_smoke.py 
```

## 2. 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy torch pytest
python scripts/train.py --episodes 100 --out artifacts
python scripts/evaluate.py --model artifacts/checkpoints/final.pt --episodes 10
pytest -q
# 用训练好的模型滚动环境，导出 NS-3 输入文件
python scripts/export_ns3_trace.py --model artifacts/checkpoints/best.pt --steps 250 --out artifacts/ns3_trace.json
# 用训练好的模型滚动环境，导出 Gazebo 输入文件
python scripts/export_gazebo_trace.py \
  --model artifacts/checkpoints/best.pt \
  --out artifacts/gazebo_trace.json \
  --steps 250 \
  --seed 42
```

## 3. 环境设计要点

- **状态**：所有无人机位置、目标点、功率、带宽分配 + 当前链路邻接矩阵。
- **动作（离散）**：
  - 全局运动策略（7）
  - 功率等级（3）
  - 带宽调度模式（3）
  - 拓扑策略（3）
- **奖励**：吞吐量、连通性、路径进度、弱链路率、功耗、碰撞风险 的加权组合。

## 4. Gazebo 与 NS-3 对接说明

- `uav_rl/gazebo_bridge.py` 提供 waypoint 指令构建与反馈解析结构，可直接封装进 ROS2 节点（订阅位姿，发布航点）。
- `uav_rl/ns3_bridge.py` 导出可用于 NS-3 场景构建的节点位置、链路拓扑、PHY 参数与功率调度表。
- 使用NS仿真（$NS3_ROOT为NS-3根目录）
- 参考教程链接 https://blog.csdn.net/weixin_33759613/article/details/149821361
```bash
cp ns3/uav_rl_scenario.cc $NS3_ROOT/scratch/
cp artifacts/ns3_trace.json $NS3_ROOT/
sudo apt-get update
sudo apt-get install -y nlohmann-json3-dev #如果没安装
cd $NS3_ROOT
./ns3 configure
./ns3 build
./ns3 run "scratch/uav_rl_scenario \
  --trace=ns3_trace.json \
  --step=1.0 \
  --applyTxPower=true \
  --routing=OLSR \
  --lossModel=LogDistance \
  --trafficMode=random-pairs \
  --topologyRange=350 \
  --seed=1 \
  --run=1"
```
- 使用 Gazebo 仿真环境（$GAZEBO_MODEL_PATH 包含 `uav_rl/models/`）（未完成）
```bash
cp artifacts/gazebo_trace.json ./ros2_ws/src/uav_simulation/traces
cd ./ros2_ws
./start.sh
```
## 5. 研究扩展建议

- 引入 Prioritized Replay、NoisyNet、Distributional DQN。
- 将单智能体全局调度扩展到 CTDE 多智能体框架（QMIX/MAPPO）。
- 在 NS-3 中接入移动信道模型（Nakagami/Rician）与队列时延奖励。
- 在 Gazebo 中结合真实动力学与风场扰动，做 sim2real domain randomization。
