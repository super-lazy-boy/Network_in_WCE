#!/bin/bash
set -e

echo "===== 清理 Conda / Python 环境 ====="
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset PYTHONHOME
unset LD_LIBRARY_PATH
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo "===== 关闭残留 Gazebo 进程 ====="
pkill -f gzserver 2>/dev/null || true
pkill -f gzclient 2>/dev/null || true
pkill -f gazebo 2>/dev/null || true

cd "$(dirname "$0")"

echo "===== 清理旧构建 ====="
rm -rf build install log

echo "===== 加载 ROS2 ====="
source /opt/ros/humble/setup.bash

echo "===== 编译 ====="
colcon build --packages-select uav_simulation --symlink-install

echo "===== 加载工作空间 ====="
source install/setup.bash

export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:$GAZEBO_PLUGIN_PATH
export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH

echo "===== 启动 ====="
ros2 launch uav_simulation play_trace.launch.py