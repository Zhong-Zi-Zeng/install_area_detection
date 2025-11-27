import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'install_area_detection'
    
    config_file = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='install_area_detection',
            executable='install_area_detection',
            name='install_area_detection_node',
            output='screen',
            parameters=[config_file]
        ),
    ])
