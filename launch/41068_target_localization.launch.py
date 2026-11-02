"""Start target geolocation after the Parrot simulator is running."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('fixed_frame', default_value='parrot1_odom',
                              description='Use parrot1_map only when SLAM is running.'),
        DeclareLaunchArgument('ground_z_m', default_value='0.0',
                              description='Height of the ground plane in fixed_frame.'),
        DeclareLaunchArgument('require_human_shape', default_value='false',
                              description='Require upright human-like thermal silhouette for rescue alerts.'),
        Node(
            package='41068_ignition_bringup',
            executable='target_localizer',
            name='target_localizer',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'fixed_frame': LaunchConfiguration('fixed_frame'),
                'ground_z_m': LaunchConfiguration('ground_z_m'),
                'require_human_shape': LaunchConfiguration('require_human_shape'),
            }],
            remappings=[
                ('/tf', '/parrot1/tf'),
                ('/tf_static', '/parrot1/tf_static'),
            ],
        ),
    ])
