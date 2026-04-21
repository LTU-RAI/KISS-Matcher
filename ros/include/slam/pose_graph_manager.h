#pragma once

#ifndef KISS_MATCHER_POSE_GRAPH_MANAGER_H
#define KISS_MATCHER_POSE_GRAPH_MANAGER_H

#include <chrono>
#include <cmath>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>

// message_filters in ROS2
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

// TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
// #include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// #include <tf2_eigen/tf2_eigen.hpp>

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/inference/Symbol.h>

#include "../tictoc.hpp"
#include "slam/loop_closure.h"
#include "slam/loop_detector.h"
#include "slam/pose_graph_node.hpp"
#include "slam/utils.hpp"

// #include <pose_graph_tools_msgs/msg/pose_graph.hpp>

namespace fs = std::filesystem;
using namespace std::chrono;
typedef message_filters::sync_policies::ApproximateTime<nav_msgs::msg::Odometry,
                                                        sensor_msgs::msg::PointCloud2>
    NodeSyncPolicy;

class PoseGraphManager : public rclcpp::Node {
 public:
  PoseGraphManager() = delete;
  explicit PoseGraphManager(const rclcpp::NodeOptions &options);
  ~PoseGraphManager();

 private:
  void appendKeyframePose(const kiss_matcher::PoseGraphNode &node);

  void callbackNode(const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg,
                    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &scan_msg);
  /**** Timer functions ****/
  // void loopPubTimerFunc();
  void buildMap();
  void detectLoopClosureByLoopDetector();
  void detectLoopClosureByNNSearch();
  void detectInterSessionLoopClosure();
  void performInterSessionRegistration();

  void visualizeCurrentData(const Eigen::Matrix4d &current_odom,
                            const rclcpp::Time &timestamp,
                            const std::string &frame_id);
  void visualizePoseGraph();
  // Fixed-rate broadcaster for `map -> odom`. Runs on its own callback group so
  // it keeps publishing even when the sync callback is stalled inside iSAM2.
  void broadcastMapOdomTf();

  void performRegistration();

  void visualizeLoopClosureClouds();

  visualization_msgs::msg::Marker visualizeLoopMarkers(const gtsam::Values &corrected_poses) const;
  visualization_msgs::msg::Marker visualizeLoopDetectionRadius(
      const geometry_msgs::msg::Point &latest_position) const;

  bool checkIfKeyframe(const kiss_matcher::PoseGraphNode &query_node,
                       const kiss_matcher::PoseGraphNode &latest_node);

  void saveFlagCallback(const std_msgs::msg::String::ConstSharedPtr &msg);

  // Returns true if relocalization has just succeeded on this tick (caller can
  // proceed with normal pose-graph initialization). Returns false if more
  // scans are still being accumulated or if the current attempt failed — in
  // either case the caller should skip the rest of the tick.
  bool tryRelocalize();

  // Load a previously-saved session (scans/ + poses_tum.txt + graph.g2o) from
  // `prior_session_dir_`, populate `prior_keyframes_`, re-key the loaded graph
  // with `prior_session_prefix_`, and seed ISAM2 with the prior session so
  // inter-session BetweenFactors can attach to real nodes. Must be called once
  // in the constructor after `isam_handler_` is constructed.
  bool loadPriorSession();

  std::string map_frame_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string package_path_;
  std::string seq_name_;

  std::mutex realtime_pose_mutex_;
  std::mutex keyframes_mutex_;
  std::mutex graph_mutex_;
  std::mutex lc_mutex_;
  std::mutex vis_mutex_;

  Eigen::Matrix4d last_corrected_pose_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d odom_delta_          = Eigen::Matrix4d::Identity();

  // Cache consumed by the fixed-rate TF broadcaster. The sync callback writes
  // here; the timer reads and re-emits at a steady stamp so RViz never sees a
  // gap in `map -> odom` during long iSAM2 updates after loop closures.
  std::mutex tf_cache_mutex_;
  Eigen::Matrix4d cached_T_map_odom_ = Eigen::Matrix4d::Identity();
  bool tf_cache_ready_               = false;
  kiss_matcher::PoseGraphNode current_frame_;
  std::vector<kiss_matcher::PoseGraphNode> keyframes_;

  bool is_initialized_           = false;
  bool loop_closure_added_       = false;
  bool need_map_update_          = false;
  bool need_graph_vis_update_    = false;
  bool need_lc_cloud_vis_update_ = false;

  std::shared_ptr<gtsam::ISAM2> isam_handler_ = nullptr;
  gtsam::NonlinearFactorGraph gtsam_graph_;
  // Mirror of every factor ever added to `gtsam_graph_`. `gtsam_graph_` is
  // cleared after each ISAM2 update, so it cannot be used to serialize the
  // complete pose graph. `persistent_graph_` is appended to at every add site
  // and never cleared, so `writeG2o` can dump the full session on save.
  gtsam::NonlinearFactorGraph persistent_graph_;
  gtsam::Values init_esti_;
  gtsam::Values corrected_esti_;

  double keyframe_thr_;
  double scan_voxel_res_;
  double map_voxel_res_;
  double save_voxel_res_;
  double loop_pub_delayed_time_;
  double loop_detection_radius_;  // Only for visualization
  int sub_key_num_;

  size_t succeeded_query_idx_;
  std::vector<std::pair<size_t, size_t>> vis_loop_edges_;
  // Inter-session LC edges: first = new-session query idx, second = prior idx
  std::vector<std::pair<size_t, size_t>> vis_inter_loop_edges_;
  // pose_graph_tools_msgs::msg::PoseGraph loop_msgs_;
  std::queue<LoopIdxPair> loop_idx_pair_queue_;
  // Inter-session queue: (new-session query idx, prior-session match idx)
  std::queue<std::pair<size_t, size_t>> inter_loop_idx_pair_queue_;

  kiss_matcher::TicToc timer_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  pcl::PointCloud<pcl::PointXYZ> odoms_, corrected_odoms_;
  nav_msgs::msg::Path odom_path_, corrected_path_;

  bool store_voxelized_scan_ = false;
  bool scan_in_sensor_frame_ = false;

  bool save_map_bag_         = false;
  bool save_map_pcd_         = false;
  bool save_in_kitti_format_ = false;
  bool save_pose_graph_      = false;
  double last_lc_time_       = 0.0;

  // Multi-session GTSAM symbol prefixes. When no prior session is loaded,
  // only `new_session_prefix_` is used and behavior matches a plain-integer
  // key space (Symbol just wraps the same index with a prefix tag).
  char prior_session_prefix_ = 'a';
  char new_session_prefix_   = 'b';

  // Prior session state (loaded once at startup when prior_session_dir is set).
  std::string prior_session_dir_;
  std::vector<kiss_matcher::PoseGraphNode> prior_keyframes_;

  // Relocalization state
  bool reloc_enabled_                = false;
  bool reloc_succeeded_              = false;
  std::string prior_map_pcd_path_;
  // Radius-based inter-session bootstrap parameters.
  double bootstrap_radius_           = 15.0;
  double bootstrap_scan_distance_    = 0.5;
  size_t bootstrap_max_attempts_per_tick_ = 5;
  Eigen::Matrix4d reloc_last_accum_pose_ = Eigen::Matrix4d::Identity();
  bool reloc_has_last_accum_pose_        = false;
  // Loaded from `relocalization.prior_map_pcd` solely for publishing on
  // /prior_map. Actual bootstrap alignment now matches against prior_keyframes_.
  pcl::PointCloud<PointType>::Ptr prior_map_cloud_;
  Eigen::Matrix4d T_priormap_from_newodom_ = Eigen::Matrix4d::Identity();

  std::shared_ptr<kiss_matcher::LoopClosure> loop_closure_;

  // NOTE(hlim): We do not provide a loop detector implementation directly,
  // but you can plug in your own detector via this interface.
  std::shared_ptr<kiss_matcher::LoopDetector> loop_detector_;

  pcl::PointCloud<PointType>::Ptr map_cloud_;

  // ROS2 interface
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr corrected_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr prior_path_pub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr loop_detection_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr loop_detection_radius_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr realtime_odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_src_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_tgt_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_coarse_aligned_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_fine_aligned_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr prior_map_pub_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_save_flag_;

  // rclcpp::Publisher<pose_graph_tools_msgs::msg::PoseGraph>::SharedPtr loop_closures_pub_;

  // message_filters
  std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> sub_odom_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> sub_scan_;
  std::shared_ptr<message_filters::Synchronizer<NodeSyncPolicy>> sub_node_;

  // Timers
  rclcpp::TimerBase::SharedPtr hydra_loop_timer_;
  rclcpp::TimerBase::SharedPtr map_timer_;
  rclcpp::TimerBase::SharedPtr loop_detector_timer_;
  rclcpp::TimerBase::SharedPtr loop_nnsearch_timer_;
  rclcpp::TimerBase::SharedPtr graph_vis_timer_;
  rclcpp::TimerBase::SharedPtr lc_reg_timer_;
  rclcpp::TimerBase::SharedPtr inter_lc_detect_timer_;
  rclcpp::TimerBase::SharedPtr inter_lc_reg_timer_;
  rclcpp::TimerBase::SharedPtr lc_vis_timer_;
  rclcpp::TimerBase::SharedPtr tf_broadcast_timer_;
  rclcpp::CallbackGroup::SharedPtr tf_broadcast_cb_group_;
};

#endif  // KISS_MATCHER_POSE_GRAPH_MANAGER_H
