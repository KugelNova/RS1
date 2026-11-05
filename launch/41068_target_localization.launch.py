"""Run confirmed-human detection when Gazebo is launched separately."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('fixed_frame', default_value='parrot1_odom',
                              description='Use parrot1_map only when SLAM is running.'),
        DeclareLaunchArgument('ground_mesh', default_value='',
                              description='Path to this run’s generated terrain_new collision mesh.'),
        DeclareLaunchArgument('ground_z_m', default_value='0.0',
                              description='Compatibility argument. RGB-D now supplies target depth.'),
        DeclareLaunchArgument('require_human_shape', default_value='false',
                              description='Also require an upright thermal silhouette.'),
        Node(
            package='41068_ignition_bringup',
            executable='target_localizer',
            name='target_localizer',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'fixed_frame': LaunchConfiguration('fixed_frame'),
                'ground_mesh': ParameterValue(LaunchConfiguration('ground_mesh'), value_type=str),
                'require_human_shape': LaunchConfiguration('require_human_shape'),
            }],
            remappings=[
                ('/tf', '/parrot1/tf'),
                ('/tf_static', '/parrot1/tf_static'),
            ],
        ),
    ])
