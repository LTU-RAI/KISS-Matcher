from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('kiss_matcher_ros')
    namespace = LaunchConfiguration('namespace')

    # NOTE: Python launch file (not YAML) because Humble's launch frontend cannot parse
    # sigterm_timeout/sigkill_timeout/on_exit on a node; the destructor writes result.pcd
    # and needs more than the default 5s before SIGTERM/SIGKILL escalation.
    start_rviz_arg = DeclareLaunchArgument(
        'start_rviz',
        default_value='false'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='husky',
        description='robot namespace; frames/topics derive from it'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Set true when replaying a bag so TF stamps share the sim clock with Fast-LIO'
    )

    rviz_path_arg = DeclareLaunchArgument(
        'rviz_path',
        default_value=PathJoinSubstitution([pkg_share, 'rviz', 'kiss_matcher_sam.rviz'])
    )

    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'husky_slam.yaml'])
    )

    # Namespaced to match spark-fast-lio: <ns>/map -> <ns>/odom -> <ns>/base_link.
    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value=[namespace, '/map'],
        description='Global (loop-closure corrected) frame'
    )

    odom_frame_arg = DeclareLaunchArgument(
        'odom_frame',
        default_value=[namespace, '/odom'],
        description='Fast-LIO odometry frame (child of map)'
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value=[namespace, '/base_link'],
        description="Robot's base frame (kept for compatibility)"
    )

    # Absolute topics: namespace does NOT auto-prefix these, so they name the ns explicitly.
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value=['/', namespace, '/odometry']
    )

    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value=['/', namespace, '/fast_lio/cloud_registered']
    )

    # The frontend infers the parameter type from the substitution; Python does not, so
    # use_sim_time has to be declared as a bool explicitly.
    use_sim_time_param = ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)

    km_sam_node = Node(
        namespace=namespace,
        package='kiss_matcher_ros',
        executable='kiss_matcher_sam',
        name='kiss_matcher_sam',
        sigterm_timeout='60.0',
        sigkill_timeout='60.0',
        output='screen',
        on_exit=Shutdown(),
        remappings=[
            # Keep the map->odom correction on the per-robot tree (/<ns>/tf), like Fast-LIO.
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
            # Subscriber topics
            ('/cloud', LaunchConfiguration('scan_topic')),
            ('/odom', LaunchConfiguration('odom_topic')),
        ],
        parameters=[
            {
                'use_sim_time': use_sim_time_param,
                'map_frame': LaunchConfiguration('map_frame'),
                'odom_frame': LaunchConfiguration('odom_frame'),
                'base_frame': LaunchConfiguration('base_frame'),
            },
            LaunchConfiguration('config_path'),
        ]
    )

    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('start_rviz')),
        namespace=namespace,
        package='rviz2',
        executable='rviz2',
        name='km_sam_rviz',
        prefix='nice',
        output='screen',
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static'),
        ],
        parameters=[{'use_sim_time': use_sim_time_param}],
        arguments=['-d', LaunchConfiguration('rviz_path')],
    )

    return LaunchDescription([
        start_rviz_arg,
        namespace_arg,
        use_sim_time_arg,
        rviz_path_arg,
        config_path_arg,
        map_frame_arg,
        odom_frame_arg,
        base_frame_arg,
        odom_topic_arg,
        scan_topic_arg,
        km_sam_node,
        rviz_node,
    ])
