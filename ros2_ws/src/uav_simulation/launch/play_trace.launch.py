#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('uav_simulation').find('uav_simulation')

    urdf_path = os.path.join(pkg_share, 'urdf', 'simple_uav.urdf')
    world_path = os.path.join(pkg_share, 'worlds', 'uav.world')
    trace_path = os.path.join(pkg_share, 'traces', 'gazebo_trace.json')

    with open(urdf_path, 'r', encoding='utf-8') as f:
        robot_desc = f.read()

    with open(trace_path, 'r', encoding='utf-8') as f:
        trace_data = json.load(f)

    frames = trace_data.get('frames', [])
    if not frames:
        raise RuntimeError('gazebo_trace.json 中没有 frames')

    first_frame = frames[0]
    uavs = first_frame.get('uavs', [])

    # 统一缩放，避免初始位置太远看不见
    position_scale = 0.05
    z_offset = 2.0

    # 更稳的 Gazebo 启动方式：同时加载 init + factory
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo',
            '--verbose',
            world_path,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'
        ],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_desc
        }],
        output='screen'
    )

    # 串行生成 UAV，避免同一时刻并发 spawn
    spawn_actions = []
    for i, uav in enumerate(uavs):
        entity_id = int(uav['id'])
        name = f'uav_{entity_id}'

        x = float(uav['x']) * position_scale
        y = float(uav['y']) * position_scale
        z = float(uav['z']) * position_scale + z_offset
        yaw = float(uav['yaw'])

        spawn_node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', name,
                '-file', urdf_path,
                '-x', str(x),
                '-y', str(y),
                '-z', str(z),
                '-Y', str(yaw)
            ],
            output='screen'
        )

        # 第1架 8 秒开始，每架间隔 2 秒
        spawn_actions.append(
            TimerAction(
                period=8.0 + i * 2.0,
                actions=[spawn_node]
            )
        )

    # 更晚启动 trace_player，等所有模型基本生成完再播放
    trace_player = TimerAction(
        period=26.0,
        actions=[
            Node(
                package='uav_simulation',
                executable='trace_player',
                name='trace_player',
                output='screen',
                parameters=[{
                    'trace_file': trace_path,
                    'loop': True,
                    'playback_rate': 1.0,
                    'position_scale': position_scale,
                    'z_offset': z_offset
                }]
            )
        ]
    )

    actions = [
        gazebo,
        robot_state_publisher,
    ]
    actions.extend(spawn_actions)
    actions.append(trace_player)

    return LaunchDescription(actions)