from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('kiss_matcher_ros')
    
    # Declare launch arguments
    start_rviz_arg = DeclareLaunchArgument(
        'start_rviz',
        default_value='false',
        description='Whether to start RViz'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='km_sam',
        description='Namespace for the KISS-Matcher-SAM node'
    )
    
    rviz_path_arg = DeclareLaunchArgument(
        'rviz_path',
        default_value=PathJoinSubstitution([pkg_share, 'rviz', 'kiss_matcher_sam.rviz']),
        description='Path to RViz config file'
    )
    
    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'slam_config_walp2.yaml']),
        description='Path to SLAM config file'
    )
    
    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='odom',
        description='Global map frame ID'
    )
    
    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base',
        description='Robot base frame ID'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/walp/fast_lio/odometry',
        description='Odometry topic'
    )
    
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/walp/fast_lio/cloud_registered',
        description='Scan topic'
    )
    
    # KISS-Matcher-SAM node
    km_sam_node = Node(
        package='kiss_matcher_ros',
        executable='kiss_matcher_sam',
        namespace=LaunchConfiguration('namespace'),
        name='kiss_matcher_sam',
        output='screen',
        on_exit=Shutdown(),
        remappings=[
            ('/cloud', LaunchConfiguration('scan_topic')),
            ('/odom', LaunchConfiguration('odom_topic')),
        ],
        parameters=[
            {
                'map_frame': 'odom',
                'base_frame': 'walp/vectornav',
            },
            LaunchConfiguration('config_path'),
        ]
    )
    
    # RViz node (optional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='km_sam_rviz',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_path')],
        condition=IfCondition(LaunchConfiguration('start_rviz')),
    )
    
    return LaunchDescription([
        start_rviz_arg,
        namespace_arg,
        rviz_path_arg,
        config_path_arg,
        map_frame_arg,
        base_frame_arg,
        odom_topic_arg,
        scan_topic_arg,
        km_sam_node,
        rviz_node,
    ])
