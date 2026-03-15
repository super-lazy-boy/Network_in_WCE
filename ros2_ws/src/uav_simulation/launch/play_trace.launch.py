#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('uav_simulation').find('uav_simulation')

    # 安装后的 package 前缀目录：.../install/uav_simulation
    pkg_prefix = os.path.dirname(os.path.dirname(pkg_share))
    plugin_lib_path = os.path.join(pkg_prefix, 'lib')

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
    if not uavs:
        raise RuntimeError('gazebo_trace.json 第一帧没有 uavs')

    position_scale = 0.05
    z_offset = 2.0
    playback_rate = 1.0
    loop = 'true'
    start_delay = 0.5

    old_gazebo_plugin_path = os.environ.get('GAZEBO_PLUGIN_PATH', '')
    if old_gazebo_plugin_path:
        new_gazebo_plugin_path = f'{plugin_lib_path}:{old_gazebo_plugin_path}'
    else:
        new_gazebo_plugin_path = plugin_lib_path

    # 让 Gazebo 能找到我们自己编译的插件
    set_gazebo_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value=new_gazebo_plugin_path
    )

    # 给 world plugin 注入参数
    set_trace_file = SetEnvironmentVariable(
        name='UAV_TRACE_FILE',
        value=trace_path
    )
    set_position_scale = SetEnvironmentVariable(
        name='UAV_POSITION_SCALE',
        value=str(position_scale)
    )
    set_z_offset = SetEnvironmentVariable(
        name='UAV_Z_OFFSET',
        value=str(z_offset)
    )
    set_playback_rate = SetEnvironmentVariable(
        name='UAV_PLAYBACK_RATE',
        value=str(playback_rate)
    )
    set_loop = SetEnvironmentVariable(
        name='UAV_LOOP',
        value=loop
    )
    set_start_delay = SetEnvironmentVariable(
        name='UAV_START_DELAY',
        value=str(start_delay)
    )

    # 可选：减少 FastDDS SHM 报错
    set_fastdds_transport = SetEnvironmentVariable(
        name='FASTDDS_BUILTIN_TRANSPORTS',
        value='UDPv4'
    )

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

    # 仍然按第一帧把 UAV spawn 进来
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

        spawn_actions.append(
            TimerAction(
                period=8.0 + i * 2.0,
                actions=[spawn_node]
            )
        )

    actions = [
        set_gazebo_plugin_path,
        set_trace_file,
        set_position_scale,
        set_z_offset,
        set_playback_rate,
        set_loop,
        set_start_delay,
        set_fastdds_transport,
        gazebo,
        robot_state_publisher,
    ]

    actions.extend(spawn_actions)

    return LaunchDescription(actions)
