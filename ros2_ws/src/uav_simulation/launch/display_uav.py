from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    pkg_share = FindPackageShare('uav_simulation').find('uav_simulation')

    urdf_path = os.path.join(pkg_share, 'urdf', 'simple_uav.urdf')
    world_path = os.path.join(pkg_share, 'worlds', 'uav.world')

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    gazebo = ExecuteProcess(
        cmd=[
            'gazebo',
            '--verbose',
            world_path,
            '-s',
            'libgazebo_ros_factory.so'
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

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'uav1',
            '-file', urdf_path,
            '-x', '0',
            '-y', '0',
            '-z', '1'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])