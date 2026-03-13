from setuptools import setup
import os
from glob import glob

package_name = 'uav_simulation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'traces'), glob('traces/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zaq',
    maintainer_email='zaq@example.com',
    description='UAV Gazebo trace playback package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trace_player = uav_simulation.trace_player:main',
        ],
    },
)