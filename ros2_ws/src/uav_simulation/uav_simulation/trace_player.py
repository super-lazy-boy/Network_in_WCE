#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState


def yaw_to_quaternion(yaw: float):
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


class TracePlayer(Node):
    def __init__(self):
        super().__init__('trace_player')

        self.declare_parameter('trace_file', '')
        self.declare_parameter('loop', True)
        self.declare_parameter('playback_rate', 1.0)
        self.declare_parameter('position_scale', 1.0)
        self.declare_parameter('z_offset', 0.0)

        self.trace_file = self.get_parameter('trace_file').value
        self.loop = self.get_parameter('loop').value
        self.playback_rate = self.get_parameter('playback_rate').value
        self.position_scale = self.get_parameter('position_scale').value
        self.z_offset = self.get_parameter('z_offset').value

        if not self.trace_file:
            raise RuntimeError('trace_file 参数不能为空')

        with open(self.trace_file, 'r', encoding='utf-8') as f:
            self.trace_data = json.load(f)

        self.frames = self.trace_data.get('frames', [])
        self.step_dt = float(self.trace_data.get('step_dt', 0.1))

        if not self.frames:
            raise RuntimeError('轨迹文件 frames 为空')

        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.get_logger().info('等待 /gazebo/set_entity_state 服务...')
        self.client.wait_for_service()

        self.frame_index = 0
        timer_period = self.step_dt / max(self.playback_rate, 1e-6)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(f'轨迹加载成功，共 {len(self.frames)} 帧')

    def timer_callback(self):
        if self.frame_index >= len(self.frames):
            if self.loop:
                self.frame_index = 0
            else:
                self.get_logger().info('轨迹播放完成')
                self.timer.cancel()
                return

        frame = self.frames[self.frame_index]
        uavs = frame.get('uavs', [])

        for uav in uavs:
            entity_id = int(uav['id'])
            name = f'uav_{entity_id}'

            x = float(uav['x']) * self.position_scale
            y = float(uav['y']) * self.position_scale
            z = float(uav['z']) * self.position_scale + self.z_offset
            yaw = float(uav['yaw'])

            qx, qy, qz, qw = yaw_to_quaternion(yaw)

            state = EntityState()
            state.name = name
            state.reference_frame = 'world'

            state.pose = Pose()
            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.position.z = z
            state.pose.orientation.x = qx
            state.pose.orientation.y = qy
            state.pose.orientation.z = qz
            state.pose.orientation.w = qw

            state.twist = Twist()

            req = SetEntityState.Request()
            req.state = state

            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        self.frame_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = TracePlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()