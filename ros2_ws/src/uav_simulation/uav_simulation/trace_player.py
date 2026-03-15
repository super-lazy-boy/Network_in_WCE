#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from typing import Dict, List

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Quaternion, Twist
from tf_transformations import quaternion_from_euler


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def shortest_angular_distance(from_angle: float, to_angle: float) -> float:
    return normalize_angle(to_angle - from_angle)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = quaternion_from_euler(0.0, 0.0, yaw)
    msg = Quaternion()
    msg.x = q[0]
    msg.y = q[1]
    msg.z = q[2]
    msg.w = q[3]
    return msg


class TracePlayer(Node):
    def __init__(self):
        super().__init__('trace_player')

        self.declare_parameter('trace_file', '')
        self.declare_parameter('loop', True)
        self.declare_parameter('playback_rate', 1.0)
        self.declare_parameter('position_scale', 1.0)
        self.declare_parameter('z_offset', 0.0)
        self.declare_parameter('update_rate', 30.0)
        self.declare_parameter(
            'service_candidates',
            ['/gazebo/set_entity_state', '/set_entity_state']
        )

        self.trace_file = self.get_parameter('trace_file').value
        self.loop = bool(self.get_parameter('loop').value)
        self.playback_rate = float(self.get_parameter('playback_rate').value)
        self.position_scale = float(self.get_parameter('position_scale').value)
        self.z_offset = float(self.get_parameter('z_offset').value)
        self.update_rate = float(self.get_parameter('update_rate').value)
        self.service_candidates = list(
            self.get_parameter('service_candidates').value
        )

        if not self.trace_file:
            raise RuntimeError('trace_file 参数为空')

        with open(self.trace_file, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)

        self.step_dt = float(trace_data.get('step_dt', 1.0))
        self.frames: List[Dict] = trace_data.get('frames', [])
        if len(self.frames) < 2:
            raise RuntimeError('轨迹帧数不足，至少需要 2 帧')

        self.uav_ids = [int(u['id']) for u in self.frames[0]['uavs']]
        self.total_duration = self.step_dt * (len(self.frames) - 1)
        self.play_time = 0.0

        self.get_logger().info(
            f'加载轨迹成功: frames={len(self.frames)}, '
            f'step_dt={self.step_dt}, update_rate={self.update_rate}, '
            f'playback_rate={self.playback_rate}'
        )

        self.client = None
        self.service_name = None

        # 尝试连接多个候选服务名
        for srv_name in self.service_candidates:
            client = self.create_client(SetEntityState, srv_name)
            self.get_logger().info(f'尝试连接服务: {srv_name}')
            if client.wait_for_service(timeout_sec=3.0):
                self.client = client
                self.service_name = srv_name
                break

        if self.client is None:
            raise RuntimeError(
                '未找到 SetEntityState 服务。请检查 Gazebo 是否加载了状态插件，'
                '并用 ros2 service list | grep entity_state 确认服务名。'
            )

        self.get_logger().info(f'已连接服务: {self.service_name}')
        self.get_logger().info('开始播放轨迹')

        timer_period = 1.0 / self.update_rate
        self.timer = self.create_timer(timer_period, self.on_timer)

    def on_timer(self):
        dt = 1.0 / self.update_rate
        self.play_time += dt * self.playback_rate

        if self.play_time > self.total_duration:
            if self.loop:
                self.play_time = self.play_time % self.total_duration
            else:
                self.play_time = self.total_duration

        frame_index = int(self.play_time / self.step_dt)
        if frame_index >= len(self.frames) - 1:
            frame_index = len(self.frames) - 2

        next_index = frame_index + 1

        t0 = self.frames[frame_index]['t']
        t1 = self.frames[next_index]['t']

        alpha = 0.0 if abs(t1 - t0) < 1e-9 else (self.play_time - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, alpha))

        frame0_map = {int(u['id']): u for u in self.frames[frame_index]['uavs']}
        frame1_map = {int(u['id']): u for u in self.frames[next_index]['uavs']}

        for uav_id in self.uav_ids:
            if uav_id not in frame0_map or uav_id not in frame1_map:
                continue

            u0 = frame0_map[uav_id]
            u1 = frame1_map[uav_id]

            x = (1.0 - alpha) * float(u0['x']) + alpha * float(u1['x'])
            y = (1.0 - alpha) * float(u0['y']) + alpha * float(u1['y'])
            z = (1.0 - alpha) * float(u0['z']) + alpha * float(u1['z'])

            yaw0 = float(u0['yaw'])
            yaw1 = float(u1['yaw'])
            dyaw = shortest_angular_distance(yaw0, yaw1)
            yaw = normalize_angle(yaw0 + alpha * dyaw)

            x = x * self.position_scale
            y = y * self.position_scale
            z = z * self.position_scale + self.z_offset

            self.set_entity_pose(f'uav_{uav_id}', x, y, z, yaw)

    def set_entity_pose(self, entity_name: str, x: float, y: float, z: float, yaw: float):
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = entity_name
        req.state.reference_frame = 'world'

        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z
        req.state.pose.orientation = yaw_to_quaternion(yaw)

        req.state.twist = Twist()
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0

        future = self.client.call_async(req)
        future.add_done_callback(
            lambda f, name=entity_name: self._handle_response(f, name)
        )

    def _handle_response(self, future, entity_name: str):
        try:
            result = future.result()
            if result is not None and not result.success:
                self.get_logger().warn(
                    f'设置 {entity_name} 状态失败: {result.status_message}'
                )
        except Exception as e:
            self.get_logger().warn(f'调用服务异常 [{entity_name}]: {e}')


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