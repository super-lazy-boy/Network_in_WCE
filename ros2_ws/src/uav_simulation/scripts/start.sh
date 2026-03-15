#!/usr/bin/env bash
set -e

echo "===== 退出 Conda 环境 ====="
if [ -n "${CONDA_PREFIX:-}" ]; then
  # 尽量彻底退出 conda，避免 Python / libstdc++ 被污染
  while [ -n "${CONDA_PREFIX:-}" ]; do
    conda deactivate 2>/dev/null || break
  done
fi

echo "===== 清理 Conda / Python 环境变量 ====="
unset PYTHONPATH
unset PYTHONHOME
unset LD_LIBRARY_PATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PROMPT_MODIFIER
unset CONDA_SHLVL

# 使用系统基础 PATH，避免优先命中 anaconda
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo "===== 设置 ROS / DDS 环境 ====="
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# 禁用 FastDDS SHM，减少 open_and_lock_file 相关报错
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

echo "===== 关闭残留 Gazebo / ROS / Python 进程 ====="
pkill -9 -f gzserver 2>/dev/null || true
pkill -9 -f gzclient 2>/dev/null || true
pkill -9 -f "gazebo --verbose" 2>/dev/null || true
pkill -9 -f "/opt/ros/.*/spawn_entity.py" 2>/dev/null || true
pkill -9 -f "ros2 launch uav_simulation" 2>/dev/null || true
pkill -9 -f trace_player 2>/dev/null || true

echo "===== 等待端口释放 ====="
sleep 2

echo "===== 检查 Gazebo 默认端口 11345 ====="
if ss -ltnp 2>/dev/null | grep -q ":11345"; then
  echo "错误：Gazebo 默认端口 11345 仍被占用。"
  echo "当前占用情况："
  ss -ltnp 2>/dev/null | grep ":11345" || true
  echo
  echo "请先关闭残留 Gazebo 进程，或手动执行："
  echo "  pkill -9 gzserver"
  echo "  pkill -9 gzclient"
  echo "  pkill -9 gazebo"
  echo "  lsof -i :11345"
  exit 1
fi

echo "===== 清理 FastDDS 共享内存残留 ====="
rm -rf /dev/shm/fastrtps_* 2>/dev/null || true
rm -rf /tmp/fastdds_shm_* 2>/dev/null || true
rm -rf /tmp/fastrtps* 2>/dev/null || true

echo "===== 进入工作空间 ====="
# 假设当前脚本位于 ros2_ws/ 或 ros2_ws/src/uav_simulation/scripts/ 之类位置
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 兼容几种常见放置位置：
# 1) ros2_ws/start.sh
# 2) ros2_ws/src/uav_simulation/start.sh
# 3) ros2_ws/src/uav_simulation/scripts/start.sh
if [ -f "$SCRIPT_DIR/src/uav_simulation/package.xml" ]; then
  WS_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/package.xml" ] && [ -d "$SCRIPT_DIR/../../.." ]; then
  WS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
elif [ -f "$SCRIPT_DIR/../package.xml" ] && [ -d "$SCRIPT_DIR/../../../" ]; then
  WS_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
else
  echo "错误：无法自动识别 ROS2 工作空间根目录。"
  echo "当前脚本目录：$SCRIPT_DIR"
  exit 1
fi

cd "$WS_DIR"
echo "工作空间目录：$WS_DIR"

echo "===== 加载 ROS2 ====="
source /opt/ros/humble/setup.bash

echo "===== 清理旧构建 ====="
rm -rf build install log

echo "===== 编译 ====="
colcon build --packages-select uav_simulation --symlink-install

echo "===== 加载工作空间 ====="
source "$WS_DIR/install/setup.bash"

echo "===== 配置 Gazebo 插件路径 ====="
# 让 Gazebo 能找到你自己编译的 world plugin
if [ -n "${GAZEBO_PLUGIN_PATH:-}" ]; then
  export GAZEBO_PLUGIN_PATH="$WS_DIR/install/uav_simulation/lib:/opt/ros/humble/lib:$GAZEBO_PLUGIN_PATH"
else
  export GAZEBO_PLUGIN_PATH="$WS_DIR/install/uav_simulation/lib:/opt/ros/humble/lib"
fi

# 这里重新补回系统 ROS 库路径，不要指向 conda
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="/opt/ros/humble/lib:$WS_DIR/install/uav_simulation/lib:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="/opt/ros/humble/lib:$WS_DIR/install/uav_simulation/lib"
fi

echo "===== 当前关键环境 ====="
echo "PATH=$PATH"
echo "GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "which python3 -> $(which python3)"
echo "python3 version -> $(python3 --version)"

echo "===== 再次确认 Gazebo 端口 11345 未被占用 ====="
if ss -ltnp 2>/dev/null | grep -q ":11345"; then
  echo "错误：启动前端口 11345 又被占用了。"
  ss -ltnp 2>/dev/null | grep ":11345" || true
  exit 1
fi

echo "===== 启动 ====="
ros2 launch uav_simulation play_trace.launch.py