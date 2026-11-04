from pathlib import Path
import shlex
import tempfile

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


_TRUE_STRINGS = "['true', '1', 'yes', 'on']"


def _is_true_expression(name):
    return ["'", LaunchConfiguration(name), "'.lower() in ", _TRUE_STRINGS]


def _if_true(name):
    return IfCondition(PythonExpression(_is_true_expression(name)))


def _if_all_true(*names):
    pieces = []
    for i, name in enumerate(names):
        if i > 0:
            pieces.append(' and ')
        pieces.extend(_is_true_expression(name))
    return IfCondition(PythonExpression(pieces))


def add_robot(
    ld,
    *,
    pkg_path,
    config_path,
    use_sim_time,
    enabled_arg,
    namespace,
    frame_prefix,
    gz_model_name,
    xacro_parts,
    bridge_config,
    localization_config,
    x,
    y,
    z,
    yaw='0.0',
    spawn_delay=0.0,
):
    """Add one robot instance to the launch description.

    Convention used here:
      - ROS nodes/topics are under /<namespace>/...
      - Gazebo entity/model is named <gz_model_name>
      - Gazebo plugin topics are under /model/<gz_model_name>/...
      - TF topics are under /<namespace>/tf and /<namespace>/tf_static
      - TF frame ids use <frame_prefix>, e.g. husky1_base_link
    """

    robot_description_content = ParameterValue(
        Command([
            'xacro ',
            PathJoinSubstitution([pkg_path] + xacro_parts),
            ' ',
            'prefix:=', frame_prefix,
            ' ',
            'gz_model_name:=', gz_model_name,
        ]),
        value_type=str,
    )

    robot_enabled = _if_true(enabled_arg)

    ld.add_action(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=namespace,
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
        condition=robot_enabled,
    ))

    ld.add_action(Node(
        package='robot_localization',
        executable='ekf_node',
        namespace=namespace,
        name='robot_localization',
        output='screen',
        parameters=[
            PathJoinSubstitution([config_path, localization_config]),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            ('odometry/filtered', 'odom'),
        ],
        condition=robot_enabled,
    ))

    robot_spawner = Node(
        package='ros_ign_gazebo',
        executable='create',
        name='spawn_' + namespace,
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-topic', '/' + namespace + '/robot_description',
            '-name', gz_model_name,
            '-x', x,
            '-y', y,
            '-z', z,
            '-Y', yaw,
        ],
    )

    # Start Gazebo first, then insert robot entities in a predictable order.
    ld.add_action(TimerAction(
        period=spawn_delay,
        actions=[robot_spawner],
        condition=robot_enabled,
    ))

    ld.add_action(Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        namespace=namespace,
        name='gazebo_bridge',
        output='screen',
        parameters=[{
            'config_file': PathJoinSubstitution([config_path, bridge_config]),
            'use_sim_time': use_sim_time,
        }],
        condition=robot_enabled,
    ))


def add_navigation_instance(
    ld,
    *,
    pkg_path,
    use_sim_time,
    robot_arg,
    robot_namespace,
    start_delay,
):
    nav_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                pkg_path,
                'launch',
                '41068_navigation.launch.py',
            ])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'namespace': robot_namespace,
            'config_filename_suffix': '_' + robot_namespace,
            'slam': LaunchConfiguration('slam'),
            'nav2': LaunchConfiguration('nav2'),
        }.items(),
    )

    # Scope is important: without it, repeated IncludeLaunchDescription calls can
    # leak/overwrite launch configurations such as namespace and config suffix.
    ld.add_action(TimerAction(
        period=start_delay,
        actions=[GroupAction(
            scoped=True,
            actions=[nav_include],
        )],
        condition=_if_true(robot_arg),
    ))


def add_rviz_instance(
    ld,
    *,
    config_path,
    use_sim_time,
    robot_arg,
    robot_namespace,
    rviz_config,
    start_delay,
):
    # Each RViz process is itself placed in the robot namespace so the Nav2 RViz
    # panel resolves relative action/service names such as navigate_to_pose and
    # lifecycle_manager_navigation to the correct robot.
    ld.add_action(TimerAction(
        period=start_delay,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            namespace=robot_namespace,
            name='rviz2',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=['-d', PathJoinSubstitution([config_path, rviz_config])],
            remappings=[
                ('/tf', '/' + robot_namespace + '/tf'),
                ('/tf_static', '/' + robot_namespace + '/tf_static'),
            ],
        )],
        condition=_if_all_true('rviz', robot_arg),
    ))


def _launch_gazebo(context):
    # Generate the population before the server starts, so counts are exact from
    # the first simulation step and no asynchronous spawn processes are needed.
    from rescue_population.world import create_population_world, nonnegative_integer

    package_dir = Path(FindPackageShare('41068_ignition_bringup').perform(context))
    world = LaunchConfiguration('world').perform(context)
    humans = nonnegative_integer(LaunchConfiguration('humans').perform(context), 'humans')
    animals = nonnegative_integer(LaunchConfiguration('animals').perform(context), 'animals')
    seed = nonnegative_integer(LaunchConfiguration('spawn_seed').perform(context), 'spawn_seed')
    world_path = package_dir / 'worlds' / (world + '.sdf')
    actions = []
    if humans or animals:
        temporary = tempfile.TemporaryDirectory(prefix='rescue-population-')
        try:
            result = create_population_world(
                package_dir, world, Path(temporary.name) / (world + '.sdf'),
                humans=humans, animals=animals, spawn_seed=seed,
                species=LaunchConfiguration('animal_species').perform(context))
        except Exception:
            temporary.cleanup()
            raise
        world_path = result.path

        def cleanup(_event, _context):
            temporary.cleanup()

        actions.extend([
            RegisterEventHandler(OnShutdown(on_shutdown=cleanup)),
            LogInfo(msg='Placed %d humans and %d animals on the terrain (seed %d).'
                    % (humans, animals, seed)),
        ])
    gui = LaunchConfiguration('gui').perform(context).lower() in ('true', '1', 'yes', 'on')
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('ros_ign_gazebo'), 'launch', 'ign_gazebo.launch.py'])),
        launch_arguments={
            'ign_args': shlex.quote(str(world_path)) + ' -r' + ('' if gui else ' -s'),
        }.items(),
    ))
    return actions


def generate_launch_description():

    ld = LaunchDescription()

    pkg_path = FindPackageShare('41068_ignition_bringup')
    config_path = PathJoinSubstitution([pkg_path, 'config'])

    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time',
    )
    use_sim_time = LaunchConfiguration('use_sim_time')
    ld.add_action(use_sim_time_launch_arg)

    # 1. Declare the gui argument
    gui_launch_arg = DeclareLaunchArgument(
        'gui',
        default_value='True',  # Default True to launch Gazebo 
        description='Flag to launch Gazebo GUI client',
    )
    ld.add_action(gui_launch_arg)

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz',
        default_value='False',
        description='Flag to launch RViz',
    )
    ld.add_action(rviz_launch_arg)

    slam_launch_arg = DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Flag to launch SLAM Toolbox for each enabled robot',
    )
    ld.add_action(slam_launch_arg)

    nav2_launch_arg = DeclareLaunchArgument(
        'nav2',
        default_value='False',
        description='Flag to launch Nav2 for each enabled robot. Nav2 also starts SLAM.',
    )
    ld.add_action(nav2_launch_arg)

    husky_launch_arg = DeclareLaunchArgument(
        'husky',
        default_value='True',
        description='Launch the Husky UGV instance in namespace /husky1',
    )
    ld.add_action(husky_launch_arg)

    parrot_launch_arg = DeclareLaunchArgument(
        'parrot',
        default_value='True',
        description='Launch the Parrot drone instance in namespace /parrot1',
    )
    ld.add_action(parrot_launch_arg)

    world_launch_arg = DeclareLaunchArgument(
        'world',
        #default_value='rescue_forest',
        default_value='simple_trees',
        description='Which world to load',
        choices=['simple_trees', 'large_demo', 'rescue_forest', 'terrain_new'],
    )
    ld.add_action(world_launch_arg)

    for name, count, description in (
            ('humans', '3', 'Number of stationary humans at startup; 0 disables them'),
            ('animals', '5', 'Number of stationary animals at startup; 0 disables them')):
        ld.add_action(DeclareLaunchArgument(
            name,
            default_value=PythonExpression([
                "'", count, "' if '", LaunchConfiguration('world'),
                "' == 'terrain_new' else '0'"]),
            description=description,
        ))
    ld.add_action(DeclareLaunchArgument(
        'spawn_seed', default_value='42',
        description='Nonnegative random seed for repeatable population positions',
    ))
    ld.add_action(DeclareLaunchArgument(
        'animal_species', default_value='bear,deer,bird,mouse,kangaroo,horse',
        description='Comma-separated animal types to use for the startup population',
    ))

    # Resolve bundled models for all worlds, including native OBJ terrain.
    ld.add_action(SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            PathJoinSubstitution([pkg_path, 'models']),
            ':',
            EnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', default_value=''),
        ],
    ))

    # Load common Gazebo server systems from a shared config file. This keeps
    # required systems such as Sensors out of individual robot and world files,
    # so custom student worlds do not need to copy plugin blocks.
    server_config_file = PathJoinSubstitution([
        config_path,
        'ignition_server.config',
    ])
    ld.add_action(SetEnvironmentVariable(
        name='IGN_GAZEBO_SERVER_CONFIG_PATH',
        value=server_config_file,
    ))
    ld.add_action(SetEnvironmentVariable(
        name='GZ_SIM_SERVER_CONFIG_PATH',
        value=server_config_file,
    ))

    ld.add_action(OpaqueFunction(function=_launch_gazebo))

    # Bridge the global simulation clock once.
    ld.add_action(Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge_clock',
        output='screen',
        parameters=[{
            'config_file': PathJoinSubstitution([config_path, 'gazebo_bridge_clock.yaml']),
            'use_sim_time': use_sim_time,
        }],
    ))

    add_robot(
        ld,
        pkg_path=pkg_path,
        config_path=config_path,
        use_sim_time=use_sim_time,
        enabled_arg='husky',
        namespace='husky1',
        frame_prefix='husky1_',
        gz_model_name='husky1',
        xacro_parts=['urdf_husky', 'husky.urdf.xacro'],
        bridge_config='gazebo_bridge_husky1.yaml',
        localization_config='robot_localization_husky1.yaml',
        x='0.0',
        y='0.0',
        z='0.4',
        spawn_delay=3.0,
    )

    add_robot(
        ld,
        pkg_path=pkg_path,
        config_path=config_path,
        use_sim_time=use_sim_time,
        enabled_arg='parrot',
        namespace='parrot1',
        frame_prefix='parrot1_',
        gz_model_name='parrot1',
        xacro_parts=['urdf_parrot', 'parrot.urdf.xacro'],
        bridge_config='gazebo_bridge_parrot1.yaml',
        localization_config='robot_localization_parrot1.yaml',
        x='-10.0',
        y='7.0',
        z='5.0',
        yaw='1.5708',
        spawn_delay=6.0,
    )

    add_navigation_instance(
        ld,
        pkg_path=pkg_path,
        use_sim_time=use_sim_time,
        robot_arg='husky',
        robot_namespace='husky1',
        start_delay=7.0,
    )

    add_navigation_instance(
        ld,
        pkg_path=pkg_path,
        use_sim_time=use_sim_time,
        robot_arg='parrot',
        robot_namespace='parrot1',
        start_delay=10.0,
    )

    add_rviz_instance(
        ld,
        config_path=config_path,
        use_sim_time=use_sim_time,
        robot_arg='husky',
        robot_namespace='husky1',
        rviz_config='41068_husky1.rviz',
        start_delay=11.0,
    )

    add_rviz_instance(
        ld,
        config_path=config_path,
        use_sim_time=use_sim_time,
        robot_arg='parrot',
        robot_namespace='parrot1',
        rviz_config='41068_parrot1.rviz',
        start_delay=13.0,
    )

    return ld
