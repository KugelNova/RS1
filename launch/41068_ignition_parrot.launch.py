from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ld = LaunchDescription()

    use_sim_time = LaunchConfiguration('use_sim_time')
    rviz = LaunchConfiguration('rviz')
    slam = LaunchConfiguration('slam')
    nav2 = LaunchConfiguration('nav2')
    world = LaunchConfiguration('world')

    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time',
    ))
    ld.add_action(DeclareLaunchArgument(
        'rviz',
        default_value='False',
        description='Flag to launch RViz',
    ))
    ld.add_action(DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Flag to launch SLAM Toolbox',
    ))
    ld.add_action(DeclareLaunchArgument(
        'nav2',
        default_value='False',
        description='Flag to launch Nav2. Nav2 also starts SLAM.',
    ))
    ld.add_action(DeclareLaunchArgument(
        'world',
        default_value='simple_trees',
        description='Which world to load',
        choices=['simple_trees', 'large_demo', 'terrain_new'],
    ))

    for name, count in (('humans', '3'), ('animals', '5'), ('trees', '80'), ('rocks', '25')):
        ld.add_action(DeclareLaunchArgument(
            name,
            default_value=PythonExpression([
                "'", count, "' if '", world, "' == 'terrain_new' else '0'"]),
            description='Startup population count for terrain_new; 0 disables this kind',
        ))
    for name, value, description in (
            ('map_size', '150', 'Generated map width and length in metres'),
            ('terrain_height', '8', 'Ground relief in metres; 0 makes flat ground'),
            ('terrain_roughness', '0.5', 'Ground detail from 0 (smooth) to 1 (rough)')):
        ld.add_action(DeclareLaunchArgument(
            name, default_value=value, description=description))
    ld.add_action(DeclareLaunchArgument(
        'spawn_seed', default_value='42', description='Repeatable population placement seed',
    ))
    ld.add_action(DeclareLaunchArgument(
        'spawn_edge_margin', default_value='10',
        description='Minimum metres between generated rescue targets and map edge',
    ))
    ld.add_action(DeclareLaunchArgument(
        'animal_species', default_value='bear,deer,bird,mouse,kangaroo,horse',
        description='Comma-separated animal types for startup spawning',
    ))

    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('41068_ignition_bringup'),
            'launch',
            '41068_ignition.launch.py',
        ])),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rviz': rviz,
            'slam': slam,
            'nav2': nav2,
            'world': world,
            'humans': LaunchConfiguration('humans'),
            'animals': LaunchConfiguration('animals'),
            'trees': LaunchConfiguration('trees'),
            'rocks': LaunchConfiguration('rocks'),
            'map_size': LaunchConfiguration('map_size'),
            'terrain_height': LaunchConfiguration('terrain_height'),
            'terrain_roughness': LaunchConfiguration('terrain_roughness'),
            'spawn_seed': LaunchConfiguration('spawn_seed'),
            'spawn_edge_margin': LaunchConfiguration('spawn_edge_margin'),
            'animal_species': LaunchConfiguration('animal_species'),
            'husky': 'False',
            'parrot': 'True',
        }.items(),
    ))

    return ld
