#include "slam/pose_graph_manager.h"

using namespace kiss_matcher;

PoseGraphManager::PoseGraphManager(const rclcpp::NodeOptions &options)
    : rclcpp::Node("km_sam", options) {
  double loop_pub_hz;
  double loop_detector_hz;
  double loop_nnsearch_hz;
  double map_update_hz;
  double vis_hz;
  double tf_broadcast_hz;

  LoopClosureConfig lc_config;
  LoopDetectorConfig ld_config;
  auto &gc = lc_config.gicp_config_;
  auto &mc = lc_config.matcher_config_;

  map_frame_             = declare_parameter<std::string>("map_frame", "map");
  odom_frame_            = declare_parameter<std::string>("odom_frame", "odom");
  base_frame_            = declare_parameter<std::string>("base_frame", "base");
  loop_pub_hz            = declare_parameter<double>("loop_pub_hz", 0.1);
  loop_detector_hz       = declare_parameter<double>("loop_detector_hz", 1.0);
  loop_nnsearch_hz       = declare_parameter<double>("loop_nnsearch_hz", 1.0);
  loop_pub_delayed_time_ = declare_parameter<double>("loop_pub_delayed_time", 60.0);
  map_update_hz          = declare_parameter<double>("map_update_hz", 0.2);
  vis_hz                 = declare_parameter<double>("vis_hz", 0.5);
  tf_broadcast_hz        = declare_parameter<double>("tf_broadcast_hz", 50.0);

  store_voxelized_scan_            = declare_parameter<bool>("store_voxelized_scan", false);
  scan_in_sensor_frame_            = declare_parameter<bool>("scan_in_sensor_frame", false);
  lc_config.voxel_res_             = declare_parameter<double>("voxel_resolution", 0.3);
  scan_voxel_res_                  = lc_config.voxel_res_;
  map_voxel_res_                   = declare_parameter<double>("map_voxel_resolution", 1.0);
  save_voxel_res_                  = declare_parameter<double>("save_voxel_resolution", 0.3);
  keyframe_thr_                    = declare_parameter<double>("keyframe.keyframe_threshold", 1.0);
  lc_config.num_submap_keyframes_  = declare_parameter<int>("keyframe.num_submap_keyframes", 5);
  lc_config.verbose_               = declare_parameter<bool>("loop.verbose", false);
  lc_config.is_multilayer_env_     = declare_parameter<bool>("loop.is_multilayer_env", false);
  lc_config.loop_detection_radius_ = declare_parameter<double>("loop.loop_detection_radius", 15.0);
  lc_config.loop_detection_timediff_threshold_ =
      declare_parameter<double>("loop.loop_detection_timediff_threshold", 10.0);

  gc.num_threads_               = declare_parameter<int>("local_reg.num_threads", 8);
  gc.correspondence_randomness_ = declare_parameter<int>("local_reg.correspondences_number", 20);
  gc.max_num_iter_              = declare_parameter<int>("local_reg.max_num_iter", 32);
  gc.scale_factor_for_corr_dist_ =
      declare_parameter<double>("local_reg.scale_factor_for_corr_dist", 5.0);
  gc.overlap_threshold_ = declare_parameter<double>("local_reg.overlap_threshold", 90.0);

  lc_config.enable_global_registration_ = declare_parameter<bool>("global_reg.enable", false);
  lc_config.num_inliers_threshold_ =
      declare_parameter<int>("global_reg.num_inliers_threshold", 100);

  reloc_enabled_ = declare_parameter<bool>("relocalization.enabled", false);
  prior_map_pcd_path_ = declare_parameter<std::string>("relocalization.prior_map_pcd", "");
  bootstrap_radius_ =
      declare_parameter<double>("relocalization.bootstrap_radius", 15.0);
  bootstrap_scan_distance_ =
      declare_parameter<double>("relocalization.bootstrap_scan_distance", 0.5);
  bootstrap_submap_scans_ = static_cast<size_t>(
      declare_parameter<int>("relocalization.bootstrap_submap_scans", 5));
  bootstrap_max_attempts_per_tick_ = static_cast<size_t>(
      declare_parameter<int>("relocalization.bootstrap_max_attempts_per_tick", 5));
  bootstrap_voxel_resolution_ = declare_parameter<double>(
      "relocalization.bootstrap_voxel_resolution", -1.0);
  {
    const std::vector<double> identity16 = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0};
    const auto t_init_vec = declare_parameter<std::vector<double>>(
        "relocalization.bootstrap_T_init", identity16);
    if (t_init_vec.size() != 16) {
      RCLCPP_ERROR(this->get_logger(),
                   "relocalization.bootstrap_T_init must be 16 doubles "
                   "(row-major 4x4), got %lu. Using identity.",
                   t_init_vec.size());
    } else {
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          bootstrap_T_init_(r, c) = t_init_vec[r * 4 + c];
      if (!bootstrap_T_init_.isApprox(Eigen::Matrix4d::Identity())) {
        const Eigen::Vector3d t = bootstrap_T_init_.block<3, 1>(0, 3);
        RCLCPP_INFO(this->get_logger(),
                    "Bootstrap T_init: translation = (%.3f, %.3f, %.3f) "
                    "(applied to query pose before reloc).",
                    t.x(), t.y(), t.z());
      }
    }
  }
  prior_session_dir_ =
      declare_parameter<std::string>("relocalization.prior_session_dir", "");
  {
    const std::string prior_prefix_str =
        declare_parameter<std::string>("relocalization.prior_session_prefix", "a");
    const std::string new_prefix_str =
        declare_parameter<std::string>("relocalization.new_session_prefix", "b");
    if (!prior_prefix_str.empty()) prior_session_prefix_ = prior_prefix_str.front();
    if (!new_prefix_str.empty())   new_session_prefix_   = new_prefix_str.front();
    if (prior_session_prefix_ == new_session_prefix_) {
      RCLCPP_ERROR(this->get_logger(),
                   "prior_session_prefix and new_session_prefix must differ "
                   "(got '%c' for both). Forcing new_session_prefix = 'b'.",
                   prior_session_prefix_);
      new_session_prefix_ = 'b';
    }
  }

  if (reloc_enabled_) {
    // prior_map_pcd is optional — used only for publishing a reference cloud
    // on `/prior_map`. The actual bootstrap relocalization now matches
    // against `prior_keyframes_` (loaded from prior_session_dir).
    if (!prior_map_pcd_path_.empty() && fs::exists(prior_map_pcd_path_)) {
      prior_map_cloud_.reset(new pcl::PointCloud<PointType>());
      if (pcl::io::loadPCDFile<PointType>(prior_map_pcd_path_, *prior_map_cloud_) != 0) {
        RCLCPP_WARN(this->get_logger(),
                    "Failed to load prior map PCD: %s. Skipping /prior_map publishing.",
                    prior_map_pcd_path_.c_str());
        prior_map_cloud_ = nullptr;
      } else {
        const auto &voxelized = voxelize(prior_map_cloud_, map_voxel_res_);
        *prior_map_cloud_     = *voxelized;
        RCLCPP_INFO(this->get_logger(),
                    "Loaded prior map '%s' with %lu points (visualization only).",
                    prior_map_pcd_path_.c_str(),
                    prior_map_cloud_->size());
      }
    }
  }

  save_map_bag_         = declare_parameter<bool>("result.save_map_bag", false);
  save_map_pcd_         = declare_parameter<bool>("result.save_map_pcd", false);
  save_in_kitti_format_ = declare_parameter<bool>("result.save_in_kitti_format", false);
  save_pose_graph_      = declare_parameter<bool>("result.save_pose_graph", false);
  seq_name_             = declare_parameter<std::string>("result.seq_name", "");
  package_path_         = declare_parameter<std::string>("result.save_dir", "");
  if (package_path_.empty()) {
    package_path_ = fs::current_path().string();
  }
  if (!fs::exists(package_path_)) {
    fs::create_directories(package_path_);
  }
  RCLCPP_INFO(this->get_logger(), "Save directory: %s", package_path_.c_str());

  rclcpp::QoS qos(1);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // With relocalization enabled the sync callback returns early until the
  // prior-map alignment succeeds, so the cache would never be written and
  // `map -> odom` would be absent during accumulation. Seed it with identity
  // so the broadcaster timer emits a valid TF from startup; the first
  // successful reloc tick overwrites it with the real transform.
  if (reloc_enabled_) {
    std::lock_guard<std::mutex> lock(tf_cache_mutex_);
    cached_T_map_odom_ = Eigen::Matrix4d::Identity();
    tf_cache_ready_    = true;
  }

  loop_closure_          = std::make_shared<LoopClosure>(lc_config, this->get_logger());
  loop_detection_radius_ = lc_config.loop_detection_radius_;

  loop_detector_ = std::make_shared<LoopDetector>(ld_config, this->get_logger());

  gtsam::ISAM2Params isam_params_;
  isam_params_.relinearizeThreshold = 0.01;
  isam_params_.relinearizeSkip      = 1;
  isam_handler_                     = std::make_shared<gtsam::ISAM2>(isam_params_);

  if (reloc_enabled_ && prior_session_dir_.empty()) {
    RCLCPP_ERROR(this->get_logger(),
                 "relocalization.enabled is true but prior_session_dir is empty. "
                 "Bootstrap reloc now matches against prior_keyframes_ loaded from disk, "
                 "so prior_session_dir is required. Disabling relocalization.");
    reloc_enabled_ = false;
  }
  if (!prior_session_dir_.empty()) {
    if (!reloc_enabled_) {
      RCLCPP_ERROR(this->get_logger(),
                   "prior_session_dir is set but relocalization.enabled is false. "
                   "Prior keyframes cannot be placed in the new-session frame without "
                   "a bootstrap reloc. Ignoring prior_session_dir.");
      prior_session_dir_.clear();
    } else if (!loadPriorSession()) {
      RCLCPP_ERROR(this->get_logger(),
                   "loadPriorSession() failed. Disabling relocalization.");
      prior_session_dir_.clear();
      prior_keyframes_.clear();
      reloc_enabled_ = false;
    }
  }

  odom_path_.header.frame_id      = map_frame_;
  corrected_path_.header.frame_id = map_frame_;

  // NOTE(hlim): To make this node compatible with being launched under different namespaces,
  // I deliberately avoided adding a '/' in front of the topic names.
  path_pub_           = this->create_publisher<nav_msgs::msg::Path>("path/original", qos);
  corrected_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("path/corrected", qos);
  prior_path_pub_     = this->create_publisher<nav_msgs::msg::Path>("path/prior", qos);
  map_pub_            = this->create_publisher<sensor_msgs::msg::PointCloud2>("global_map", qos);
  scan_pub_           = this->create_publisher<sensor_msgs::msg::PointCloud2>("curr_scan", qos);
  loop_detection_pub_ =
      this->create_publisher<visualization_msgs::msg::Marker>("loop_detection", qos);
  loop_detection_radius_pub_ =
      this->create_publisher<visualization_msgs::msg::Marker>("loop_detection_radius", qos);

  // loop_closures_pub_ =
  // this->create_publisher<pose_graph_tools_msgs::msg::PoseGraph>("/hydra_ros_node/external_loop_closures",
  // 10);
  realtime_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom_corrected", qos);
  debug_src_pub_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("lc/src", qos);
  debug_tgt_pub_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("lc/tgt", qos);
  debug_coarse_aligned_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("lc/coarse_alignment", qos);
  debug_fine_aligned_pub_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("lc/fine_alignment", qos);
  debug_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("lc/debug_cloud", qos);
  prior_map_pub_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("prior_map", qos);

  if (reloc_enabled_ && prior_map_cloud_ && !prior_map_cloud_->empty()) {
    // TRANSIENT_LOCAL QoS latches this for late subscribers (e.g. RViz).
    prior_map_pub_->publish(toROSMsg(*prior_map_cloud_, map_frame_, this->now()));
  }

  if (!prior_keyframes_.empty()) {
    nav_msgs::msg::Path prior_path;
    prior_path.header.frame_id = map_frame_;
    prior_path.header.stamp    = this->now();
    prior_path.poses.reserve(prior_keyframes_.size());
    for (const auto &kf : prior_keyframes_) {
      prior_path.poses.push_back(eigenToPoseStamped(kf.pose_corrected_, map_frame_));
    }
    prior_path_pub_->publish(prior_path);
  }

  sub_odom_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(this, "/odom");
  sub_scan_ =
      std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, "/cloud");

  sub_node_ = std::make_shared<message_filters::Synchronizer<NodeSyncPolicy>>(
      NodeSyncPolicy(100), *sub_odom_, *sub_scan_);
  sub_node_->registerCallback(std::bind(
      &PoseGraphManager::callbackNode, this, std::placeholders::_1, std::placeholders::_2));

  sub_save_flag_ = this->create_subscription<std_msgs::msg::String>(
      "save_dir", 1, std::bind(&PoseGraphManager::saveFlagCallback, this, std::placeholders::_1));

  // hydra_loop_timer_ = this->create_wall_timer(
  //   std::chrono::duration<double>(1.0 / loop_pub_hz),
  //   std::bind(&PoseGraphManager::loopPubTimerFunc, this));

  map_cloud_.reset(new pcl::PointCloud<PointType>());
  map_timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / map_update_hz),
                                       std::bind(&PoseGraphManager::buildMap, this));

  loop_detector_timer_ =
      this->create_wall_timer(std::chrono::duration<double>(1.0 / loop_detector_hz),
                              std::bind(&PoseGraphManager::detectLoopClosureByLoopDetector, this));

  loop_nnsearch_timer_ =
      this->create_wall_timer(std::chrono::duration<double>(1.0 / loop_nnsearch_hz),
                              std::bind(&PoseGraphManager::detectLoopClosureByNNSearch, this));

  graph_vis_timer_ =
      this->create_wall_timer(std::chrono::duration<double>(1.0 / vis_hz),
                              std::bind(&PoseGraphManager::visualizePoseGraph, this));

  lc_reg_timer_ = this->create_wall_timer(std::chrono::duration<double>(1.0 / 100.0),
                                          std::bind(&PoseGraphManager::performRegistration, this));

  if (!prior_keyframes_.empty()) {
    // Detect cross-session candidates at the same cadence as intra-session NN
    // search; run registrations on a faster timer so we don't bottleneck.
    inter_lc_detect_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / loop_nnsearch_hz),
        std::bind(&PoseGraphManager::detectInterSessionLoopClosure, this));
    inter_lc_reg_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / 100.0),
        std::bind(&PoseGraphManager::performInterSessionRegistration, this));
  }

  // 20 Hz is enough as long as it's faster than the full registration process.
  lc_vis_timer_ =
      this->create_wall_timer(std::chrono::duration<double>(1.0 / 20.0),
                              std::bind(&PoseGraphManager::visualizeLoopClosureClouds, this));

  // Fixed-rate `map -> odom` broadcaster on a dedicated reentrant group so it
  // keeps emitting TFs with fresh stamps even while the sync callback is
  // blocked inside a long iSAM2 update after a loop closure.
  tf_broadcast_cb_group_ =
      this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  tf_broadcast_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / tf_broadcast_hz),
      std::bind(&PoseGraphManager::broadcastMapOdomTf, this),
      tf_broadcast_cb_group_);

  if (!lc_config.is_multilayer_env_) {
    RCLCPP_WARN(
        get_logger(),
        "'loop.is_multilayer_env' is set to `false`. "
        "This setting is recommended for outdoor environments to ignore the effect of Z-drift. "
        "However, if you're running SLAM in an indoor multi-layer environment, "
        "consider setting it to true to enable full 3D NN search for loop candidates.");
  }
  RCLCPP_INFO(this->get_logger(), "Main class, starting node...");
}

PoseGraphManager::~PoseGraphManager() {
  if (save_map_bag_) {
    RCLCPP_INFO(this->get_logger(), "NOTE(hlim): skipping final bag save in ROS2 example code.");
  }
  if (save_map_pcd_) {
    pcl::PointCloud<PointType>::Ptr corrected_map(new pcl::PointCloud<PointType>());
    corrected_map->reserve(keyframes_[0].scan_.size() * keyframes_.size());

    {
      std::lock_guard<std::mutex> lock(keyframes_mutex_);
      for (size_t i = 0; i < keyframes_.size(); ++i) {
        *corrected_map += transformPcd(keyframes_[i].scan_, keyframes_[i].pose_corrected_);
      }
    }
    const auto &voxelized_map = voxelize(corrected_map, save_voxel_res_);
    pcl::io::savePCDFileASCII<PointType>(package_path_ + "/result.pcd", *voxelized_map);
    RCLCPP_INFO(this->get_logger(), "Result saved in .pcd format (Destructor).");
  }
}

void PoseGraphManager::appendKeyframePose(const PoseGraphNode &node) {
  odoms_.points.emplace_back(node.pose_(0, 3), node.pose_(1, 3), node.pose_(2, 3));

  corrected_odoms_.points.emplace_back(
      node.pose_corrected_(0, 3), node.pose_corrected_(1, 3), node.pose_corrected_(2, 3));

  odom_path_.poses.emplace_back(eigenToPoseStamped(node.pose_, map_frame_));
  corrected_path_.poses.emplace_back(eigenToPoseStamped(node.pose_corrected_, map_frame_));
  return;
}

void PoseGraphManager::callbackNode(const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg,
                                    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &scan_msg) {
  static size_t latest_keyframe_idx = 0;

  // NOTE(hlim): For clarification, 'current' refers to the real-time incoming messages,
  // while 'latest' indicates the last keyframe information already appended to keyframes_.
  Eigen::Matrix4d current_odom = current_frame_.pose_;
  current_frame_               = PoseGraphNode(*odom_msg,
                                  *scan_msg,
                                  latest_keyframe_idx,
                                  scan_voxel_res_,
                                  store_voxelized_scan_,
                                  scan_in_sensor_frame_);

  kiss_matcher::TicToc total_timer;
  kiss_matcher::TicToc local_timer;

  // Relocalization: accumulate scans from the front-end until enough are
  // available to attempt a global alignment to the prior map. Once successful,
  // rewrite every incoming pose into the prior-map frame so the rest of the
  // pipeline (pose graph, TF, saving) operates there transparently.
  if (reloc_enabled_ && !reloc_succeeded_) {
    if (!tryRelocalize()) {
      return;
    }
    current_frame_.pose_           = T_priormap_from_newodom_ * current_frame_.pose_;
    current_frame_.pose_corrected_ = current_frame_.pose_;
    current_odom                   = current_frame_.pose_;
    last_corrected_pose_           = current_frame_.pose_;
    odom_delta_                    = Eigen::Matrix4d::Identity();
  } else if (reloc_enabled_ && reloc_succeeded_) {
    current_frame_.pose_           = T_priormap_from_newodom_ * current_frame_.pose_;
    current_frame_.pose_corrected_ = current_frame_.pose_;
  }

  visualizeCurrentData(current_odom, odom_msg->header.stamp, scan_msg->header.frame_id);

  if (!is_initialized_) {
    keyframes_.push_back(current_frame_);
    appendKeyframePose(current_frame_);

    auto variance_vector = (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished();
    gtsam::noiseModel::Diagonal::shared_ptr prior_noise =
        gtsam::noiseModel::Diagonal::Variances(variance_vector);

    const gtsam::Symbol sym_first(new_session_prefix_, 0);
    gtsam::PriorFactor<gtsam::Pose3> prior_factor(
        sym_first, eigenToGtsam(current_frame_.pose_), prior_noise);
    gtsam_graph_.add(prior_factor);
    persistent_graph_.add(prior_factor);

    init_esti_.insert(gtsam::Symbol(new_session_prefix_, latest_keyframe_idx),
                      eigenToGtsam(current_frame_.pose_));
    ++latest_keyframe_idx;
    is_initialized_ = true;

    RCLCPP_INFO(this->get_logger(), "The first node comes. Initialization complete.");

  } else {
    const auto t_keyframe_processing = local_timer.toc();
    if (checkIfKeyframe(current_frame_, keyframes_.back())) {
      {
        std::lock_guard<std::mutex> lock(keyframes_mutex_);
        keyframes_.push_back(current_frame_);
      }

      auto variance_vector = (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished();
      gtsam::noiseModel::Diagonal::shared_ptr odom_noise =
          gtsam::noiseModel::Diagonal::Variances(variance_vector);

      gtsam::Pose3 pose_from = eigenToGtsam(keyframes_[latest_keyframe_idx - 1].pose_corrected_);
      gtsam::Pose3 pose_to   = eigenToGtsam(current_frame_.pose_corrected_);

      {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        const gtsam::Symbol sym_prev(new_session_prefix_, latest_keyframe_idx - 1);
        const gtsam::Symbol sym_curr(new_session_prefix_, latest_keyframe_idx);
        gtsam::BetweenFactor<gtsam::Pose3> odom_factor(
            sym_prev, sym_curr, pose_from.between(pose_to), odom_noise);
        gtsam_graph_.add(odom_factor);
        persistent_graph_.add(odom_factor);
        init_esti_.insert(sym_curr, pose_to);
      }

      ++latest_keyframe_idx;
      {
        std::lock_guard<std::mutex> lock(vis_mutex_);
        appendKeyframePose(current_frame_);
      }

      local_timer.tic();
      {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        isam_handler_->update(gtsam_graph_, init_esti_);
        isam_handler_->update();
        if (loop_closure_added_) {
          isam_handler_->update();
          isam_handler_->update();
          isam_handler_->update();
        }
        gtsam_graph_.resize(0);
        init_esti_.clear();
      }
      const auto t_optim = local_timer.toc();

      {
        std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
        corrected_esti_ = isam_handler_->calculateEstimate();
        const gtsam::Symbol sym_latest(new_session_prefix_, latest_keyframe_idx - 1);
        last_corrected_pose_ =
            gtsamToEigen(corrected_esti_.at<gtsam::Pose3>(sym_latest));
        odom_delta_ = Eigen::Matrix4d::Identity();
      }
      if (loop_closure_added_) {
        std::lock_guard<std::mutex> lock(keyframes_mutex_);
        for (size_t i = 0; i < keyframes_.size(); ++i) {
          const gtsam::Symbol sym(new_session_prefix_, i);
          if (corrected_esti_.exists(sym)) {
            keyframes_[i].pose_corrected_ = gtsamToEigen(corrected_esti_.at<gtsam::Pose3>(sym));
          }
        }
        for (size_t i = 0; i < prior_keyframes_.size(); ++i) {
          const gtsam::Symbol sym(prior_session_prefix_, i);
          if (corrected_esti_.exists(sym)) {
            prior_keyframes_[i].pose_corrected_ =
                gtsamToEigen(corrected_esti_.at<gtsam::Pose3>(sym));
          }
        }
        loop_closure_added_ = false;
      }

      const auto t_total = total_timer.toc();

      RCLCPP_INFO(
          this->get_logger(),
          "# of Keyframes: %zu. Timing (msec) → Total: %.1f | Keyframe: %.1f | Optim.: %.1f",
          keyframes_.size(),
          t_total,
          t_keyframe_processing,
          t_optim);
    }
  }
}

// void PoseGraphManager::loopPubTimerFunc()
// {
//   if (loop_msgs_.edges.empty()) {
//     RCLCPP_WARN(this->get_logger(),
//       "`loop_msgs_.edges` is empty. Skipping loop closure publishing.");
//     return;
//   }
//   if (last_lc_time_ + loop_pub_delayed_time_ < this->now().seconds()) {
//     loop_closures_pub_->publish(loop_msgs_);
//     loop_msgs_.nodes.clear();
//     loop_msgs_.edges.clear();
//     RCLCPP_INFO(this->get_logger(), "`loop_msgs_` is successfully published!");
//     return;
//   }
// }

void PoseGraphManager::buildMap() {
  static size_t start_idx = 0;

  if (map_pub_->get_subscription_count() > 0) {
    {
      std::lock_guard<std::mutex> lock(keyframes_mutex_);
      if (need_map_update_) {
        map_cloud_->clear();
        start_idx = 0;
      }

      if (keyframes_.empty()) return;

      // NOTE(hlim): Building the full map causes RViz delay when keyframes > 500.
      // Since the map is for visualization only, we apply a heuristic to reduce cost.
      for (size_t i = start_idx; i < keyframes_.size(); ++i) {
        const auto &i_th_scan = [&]() {
          // It's already voxelized
          if (store_voxelized_scan_) {
            return keyframes_[i].scan_;
          }

          if (keyframes_[i].voxelized_scan_.empty()) {
            keyframes_[i].voxelized_scan_ = *voxelize(keyframes_[i].scan_, scan_voxel_res_);
          }
          return keyframes_[i].voxelized_scan_;
        }();

        *map_cloud_ += transformPcd(i_th_scan, keyframes_[i].pose_corrected_);
      }

      start_idx = keyframes_.size();
    }

    const auto &voxelized_map = voxelize(map_cloud_, map_voxel_res_);
    map_pub_->publish(
        toROSMsg(*voxelized_map, map_frame_, toRclcppTime(keyframes_.back().timestamp_)));
  }

  if (need_map_update_) {
    need_map_update_ = false;
  }
}

void PoseGraphManager::detectLoopClosureByLoopDetector() {
  auto &query = keyframes_.back();
  if (!is_initialized_ || keyframes_.empty() || query.loop_detector_processed_) {
    return;
  }
  query.loop_detector_processed_ = true;

  kiss_matcher::TicToc ld_timer;
  const auto &loop_idx_pairs = loop_detector_->fetchLoopCandidates(query, keyframes_);

  for (const auto &loop_candidate : loop_idx_pairs) {
    loop_idx_pair_queue_.push(loop_candidate);
  }

  const auto t_ld = ld_timer.toc();
}

void PoseGraphManager::detectLoopClosureByNNSearch() {
  auto &query = keyframes_.back();
  if (!is_initialized_ || keyframes_.empty() || query.nnsearch_processed_) {
    return;
  }
  query.nnsearch_processed_ = true;

  const auto &loop_idx_pairs = loop_closure_->fetchLoopCandidates(query, keyframes_);

  for (const auto &loop_candidate : loop_idx_pairs) {
    loop_idx_pair_queue_.push(loop_candidate);
  }
}

void PoseGraphManager::performRegistration() {
  kiss_matcher::TicToc reg_timer;
  if (loop_idx_pair_queue_.empty()) {
    return;
  }
  const auto [query_idx, match_idx] = loop_idx_pair_queue_.front();
  loop_idx_pair_queue_.pop();

  const RegOutput &reg_output = loop_closure_->performLoopClosure(keyframes_, query_idx, match_idx);
  need_lc_cloud_vis_update_   = true;

  if (reg_output.is_valid_) {
    RCLCPP_INFO(this->get_logger(), "LC accepted. Overlapness: %.3f", reg_output.overlapness_);
    gtsam::Pose3 pose_from = eigenToGtsam(reg_output.pose_ * keyframes_[query_idx].pose_corrected_);
    gtsam::Pose3 pose_to   = eigenToGtsam(keyframes_[match_idx].pose_corrected_);

    // TODO(hlim): Parameterize
    auto variance_vector = (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished();
    gtsam::noiseModel::Diagonal::shared_ptr loop_noise =
        gtsam::noiseModel::Diagonal::Variances(variance_vector);

    {
      std::lock_guard<std::mutex> lock(graph_mutex_);
      const gtsam::Symbol sym_q(new_session_prefix_, query_idx);
      const gtsam::Symbol sym_m(new_session_prefix_, match_idx);
      gtsam::BetweenFactor<gtsam::Pose3> loop_factor(
          sym_q, sym_m, pose_from.between(pose_to), loop_noise);
      gtsam_graph_.add(loop_factor);
      persistent_graph_.add(loop_factor);
    }

    vis_loop_edges_.emplace_back(query_idx, match_idx);
    succeeded_query_idx_   = query_idx;
    loop_closure_added_    = true;
    need_map_update_       = true;
    need_graph_vis_update_ = true;

    // --------------------------------------------------
    // TODO(hlim): resurrect pose_graph_tools_msgs
    // pose_graph_tools_msgs::msg::PoseGraphEdge edge;
    // double lidar_end_time_compensation = 0.1;
    // edge.header.stamp = this->now();
    // edge.robot_from = 0;
    // edge.robot_to = 0;
    // edge.type = 1;

    // edge.key_to = static_cast<uint64_t>(
    //   (keyframes_.back().timestamp_ - lidar_end_time_compensation) * 1e9);
    // edge.key_from = static_cast<uint64_t>(
    //   (keyframes_[closest_keyframe_idx].timestamp_ - lidar_end_time_compensation) * 1e9);

    // Eigen::Matrix4d pose_inv = pose_to.matrix().inverse() * pose_from.matrix();
    // edge.pose = poseEigToPoseGeo(pose_inv);
    // loop_msgs_.edges.emplace_back(edge);
    // last_lc_time_ = this->now().seconds();
    // --------------------------------------------------
  } else {
    if (reg_output.overlapness_ == 0.0) {
      RCLCPP_WARN(this->get_logger(), "LC rejected. KISS-Matcher failed");
    } else {
      RCLCPP_WARN(this->get_logger(), "LC rejected. Overlapness: %.3f", reg_output.overlapness_);
    }
  }
  RCLCPP_INFO(this->get_logger(), "Reg: %.1f msec", reg_timer.toc());
}

void PoseGraphManager::detectInterSessionLoopClosure() {
  // Only meaningful once the bootstrap reloc has placed new-session poses in
  // the prior-map frame; before that, radius queries against prior_keyframes_
  // would be nonsense.
  if (!reloc_succeeded_ || prior_keyframes_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(keyframes_mutex_);
  if (!is_initialized_ || keyframes_.empty()) {
    return;
  }
  const PoseGraphNode &query = keyframes_.back();
  const auto idx_pairs =
      loop_closure_->fetchInterSessionLoopCandidates(query, prior_keyframes_);
  for (const auto &pair : idx_pairs) {
    inter_loop_idx_pair_queue_.push(pair);
  }
}

void PoseGraphManager::performInterSessionRegistration() {
  if (inter_loop_idx_pair_queue_.empty()) {
    return;
  }
  const auto [query_idx, match_idx] = inter_loop_idx_pair_queue_.front();
  inter_loop_idx_pair_queue_.pop();

  // Snapshot both vectors under their respective locks so registration
  // doesn't race with sync-callback appends / ISAM2 pose rewrites.
  std::vector<PoseGraphNode> new_snapshot;
  std::vector<PoseGraphNode> prior_snapshot;
  {
    std::lock_guard<std::mutex> lock(keyframes_mutex_);
    if (query_idx >= keyframes_.size() || match_idx >= prior_keyframes_.size()) {
      return;
    }
    new_snapshot   = keyframes_;
    prior_snapshot = prior_keyframes_;
  }

  const RegOutput reg_output = loop_closure_->performInterSessionLoopClosure(
      new_snapshot, prior_snapshot, query_idx, match_idx);

  if (!reg_output.is_valid_) {
    if (reg_output.overlapness_ == 0.0) {
      RCLCPP_WARN(this->get_logger(), "Inter-session LC rejected. KISS-Matcher failed");
    } else {
      RCLCPP_WARN(this->get_logger(),
                  "Inter-session LC rejected. Overlapness: %.3f",
                  reg_output.overlapness_);
    }
    return;
  }

  RCLCPP_INFO(this->get_logger(),
              "Inter-session LC accepted (q=%lu ↔ prior=%lu). Overlapness: %.3f",
              query_idx, match_idx, reg_output.overlapness_);

  gtsam::Pose3 pose_from =
      eigenToGtsam(reg_output.pose_ * new_snapshot[query_idx].pose_corrected_);
  gtsam::Pose3 pose_to = eigenToGtsam(prior_snapshot[match_idx].pose_corrected_);

  auto variance_vector =
      (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished();
  gtsam::noiseModel::Diagonal::shared_ptr loop_noise =
      gtsam::noiseModel::Diagonal::Variances(variance_vector);

  {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    const gtsam::Symbol k_from(new_session_prefix_, query_idx);
    const gtsam::Symbol k_to(prior_session_prefix_, match_idx);
    gtsam::BetweenFactor<gtsam::Pose3> inter_factor(
        k_from, k_to, pose_from.between(pose_to), loop_noise);
    gtsam_graph_.add(inter_factor);
    persistent_graph_.add(inter_factor);
  }

  vis_inter_loop_edges_.emplace_back(query_idx, match_idx);
  loop_closure_added_    = true;
  need_map_update_       = true;
  need_graph_vis_update_ = true;
  need_lc_cloud_vis_update_ = true;
  succeeded_query_idx_   = query_idx;
}

void PoseGraphManager::visualizeCurrentData(const Eigen::Matrix4d &current_odom,
                                            const rclcpp::Time &timestamp,
                                            const std::string &frame_id) {
  // NOTE(hlim): Instead of visualizing only when adding keyframes (node-wise), which can feel
  // choppy, we visualize the current frame every cycle to ensure smoother, real-time visualization.
  {
    std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
    odom_delta_                    = odom_delta_ * current_odom.inverse() * current_frame_.pose_;
    current_frame_.pose_corrected_ = last_corrected_pose_ * odom_delta_;

    const geometry_msgs::msg::PoseStamped ps =
        eigenToPoseStamped(current_frame_.pose_corrected_, map_frame_);
    nav_msgs::msg::Odometry odom_out;
    odom_out.header.stamp    = timestamp;
    odom_out.header.frame_id = map_frame_;
    odom_out.child_frame_id  = base_frame_;
    odom_out.pose.pose       = ps.pose;
    realtime_odom_pub_->publish(odom_out);

    // Follow REP-105: cache `map -> odom` correction for the broadcaster timer
    // to emit at a steady rate. T_map_odom = T_map_base_corrected *
    // (T_odom_base)^(-1). Fast-LIO keeps publishing `odom -> base` so the two
    // broadcasters no longer collide.
    const Eigen::Matrix4d T_map_odom =
        current_frame_.pose_corrected_ * current_frame_.pose_.inverse();
    {
      std::lock_guard<std::mutex> tf_lock(tf_cache_mutex_);
      cached_T_map_odom_ = T_map_odom;
      tf_cache_ready_    = true;
    }
  }

  scan_pub_->publish(toROSMsg(
      transformPcd(current_frame_.scan_, current_frame_.pose_corrected_), map_frame_, timestamp));
  if (!corrected_path_.poses.empty()) {
    loop_detection_radius_pub_->publish(
        visualizeLoopDetectionRadius(corrected_path_.poses.back().pose.position));
  }
}

void PoseGraphManager::broadcastMapOdomTf() {
  Eigen::Matrix4d T;
  {
    std::lock_guard<std::mutex> lock(tf_cache_mutex_);
    if (!tf_cache_ready_) return;
    T = cached_T_map_odom_;
  }

  geometry_msgs::msg::TransformStamped transform_stamped;
  transform_stamped.header.stamp    = this->now();
  transform_stamped.header.frame_id = map_frame_;
  transform_stamped.child_frame_id  = odom_frame_;
  const Eigen::Quaterniond q(T.block<3, 3>(0, 0));
  transform_stamped.transform.translation.x = T(0, 3);
  transform_stamped.transform.translation.y = T(1, 3);
  transform_stamped.transform.translation.z = T(2, 3);
  transform_stamped.transform.rotation.x    = q.x();
  transform_stamped.transform.rotation.y    = q.y();
  transform_stamped.transform.rotation.z    = q.z();
  transform_stamped.transform.rotation.w    = q.w();
  tf_broadcaster_->sendTransform(transform_stamped);
}

void PoseGraphManager::visualizePoseGraph() {
  if (!is_initialized_) {
    return;
  }

  if (need_graph_vis_update_) {
    gtsam::Values corrected_esti_copied;
    pcl::PointCloud<pcl::PointXYZ> corrected_odoms;
    nav_msgs::msg::Path corrected_path;

    {
      std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
      corrected_esti_copied = corrected_esti_;
    }
    // Only emit new-session poses on the corrected_path. Prior-session poses
    // are rendered separately on path/prior so the two trajectories can be
    // styled/color-coded independently.
    for (size_t i = 0; i < keyframes_.size(); ++i) {
      const gtsam::Symbol sym(new_session_prefix_, i);
      if (!corrected_esti_copied.exists(sym)) continue;
      gtsam::Pose3 pose_ = corrected_esti_copied.at<gtsam::Pose3>(sym);
      corrected_odoms.points.emplace_back(
          pose_.translation().x(), pose_.translation().y(), pose_.translation().z());

      corrected_path.poses.push_back(gtsamToPoseStamped(pose_, map_frame_));
    }
    if (!vis_loop_edges_.empty() || !vis_inter_loop_edges_.empty()) {
      loop_detection_pub_->publish(visualizeLoopMarkers(corrected_esti_copied));
    }
    {
      std::lock_guard<std::mutex> lock(vis_mutex_);
      corrected_odoms_      = corrected_odoms;
      corrected_path_.poses = corrected_path.poses;
    }
    need_graph_vis_update_ = false;
  }

  {
    std::lock_guard<std::mutex> lock(vis_mutex_);
    path_pub_->publish(odom_path_);
    corrected_path_pub_->publish(corrected_path_);
  }
}

void PoseGraphManager::visualizeLoopClosureClouds() {
  if (!need_lc_cloud_vis_update_) {
    return;
  }

  rclcpp::Time query_timestamp;
  {
    std::lock_guard<std::mutex> lock(keyframes_mutex_);
    if (succeeded_query_idx_ >= keyframes_.size()) {
      return;
    }
    query_timestamp = toRclcppTime(keyframes_[succeeded_query_idx_].timestamp_);
  }

  debug_src_pub_->publish(toROSMsg(loop_closure_->getSourceCloud(), map_frame_, query_timestamp));
  debug_tgt_pub_->publish(toROSMsg(loop_closure_->getTargetCloud(), map_frame_, query_timestamp));
  debug_fine_aligned_pub_->publish(
      toROSMsg(loop_closure_->getFinalAlignedCloud(), map_frame_, query_timestamp));
  debug_coarse_aligned_pub_->publish(
      toROSMsg(loop_closure_->getCoarseAlignedCloud(), map_frame_, query_timestamp));
  debug_cloud_pub_->publish(toROSMsg(loop_closure_->getDebugCloud(), map_frame_, query_timestamp));
  need_lc_cloud_vis_update_ = false;
}

visualization_msgs::msg::Marker PoseGraphManager::visualizeLoopMarkers(
    const gtsam::Values &corrected_poses) const {
  visualization_msgs::msg::Marker edges;
  edges.type               = visualization_msgs::msg::Marker::LINE_LIST;
  edges.scale.x            = 0.1f;
  edges.header.frame_id    = map_frame_;
  edges.pose.orientation.w = 1.0f;
  edges.color.r            = 1.0f;
  edges.color.g            = 1.0f;
  edges.color.b            = 1.0f;
  edges.color.a            = 1.0f;

  // Per-point colors override the global color, so intra-session edges can be
  // drawn white and inter-session edges in a different color on the same
  // marker.
  std_msgs::msg::ColorRGBA intra_color;
  intra_color.r = 1.0f; intra_color.g = 1.0f; intra_color.b = 1.0f; intra_color.a = 1.0f;
  std_msgs::msg::ColorRGBA inter_color;
  inter_color.r = 0.0f; inter_color.g = 1.0f; inter_color.b = 0.3f; inter_color.a = 1.0f;

  auto push_segment = [&](const gtsam::Pose3 &p_a,
                          const gtsam::Pose3 &p_b,
                          const std_msgs::msg::ColorRGBA &color) {
    geometry_msgs::msg::Point p, p2;
    p.x  = p_a.translation().x();
    p.y  = p_a.translation().y();
    p.z  = p_a.translation().z();
    p2.x = p_b.translation().x();
    p2.y = p_b.translation().y();
    p2.z = p_b.translation().z();
    edges.points.push_back(p);
    edges.points.push_back(p2);
    edges.colors.push_back(color);
    edges.colors.push_back(color);
  };

  for (size_t i = 0; i < vis_loop_edges_.size(); ++i) {
    const gtsam::Symbol sym_a(new_session_prefix_, vis_loop_edges_[i].first);
    const gtsam::Symbol sym_b(new_session_prefix_, vis_loop_edges_[i].second);
    if (!corrected_poses.exists(sym_a) || !corrected_poses.exists(sym_b)) {
      continue;
    }
    push_segment(corrected_poses.at<gtsam::Pose3>(sym_a),
                 corrected_poses.at<gtsam::Pose3>(sym_b),
                 intra_color);
  }

  for (size_t i = 0; i < vis_inter_loop_edges_.size(); ++i) {
    const gtsam::Symbol sym_new(new_session_prefix_, vis_inter_loop_edges_[i].first);
    const gtsam::Symbol sym_prior(prior_session_prefix_, vis_inter_loop_edges_[i].second);
    if (!corrected_poses.exists(sym_new) || !corrected_poses.exists(sym_prior)) {
      continue;
    }
    push_segment(corrected_poses.at<gtsam::Pose3>(sym_new),
                 corrected_poses.at<gtsam::Pose3>(sym_prior),
                 inter_color);
  }
  return edges;
}

visualization_msgs::msg::Marker PoseGraphManager::visualizeLoopDetectionRadius(
    const geometry_msgs::msg::Point &latest_position) const {
  visualization_msgs::msg::Marker sphere;
  sphere.header.frame_id = map_frame_;
  sphere.id              = 100000;  // arbitrary number
  sphere.type            = visualization_msgs::msg::Marker::SPHERE;
  sphere.pose.position.x = latest_position.x;
  sphere.pose.position.y = latest_position.y;
  sphere.pose.position.z = latest_position.z;
  sphere.scale.x         = 2 * loop_detection_radius_;
  sphere.scale.y         = 2 * loop_detection_radius_;
  sphere.scale.z         = 2 * loop_detection_radius_;
  // Use transparanet cyan color
  sphere.color.r = 0.0;
  sphere.color.g = 0.824;
  sphere.color.b = 1.0;
  sphere.color.a = 0.5;

  return sphere;
}

bool PoseGraphManager::checkIfKeyframe(const PoseGraphNode &query_node,
                                       const PoseGraphNode &latest_node) {
  return keyframe_thr_ < (latest_node.pose_corrected_.block<3, 1>(0, 3) -
                          query_node.pose_corrected_.block<3, 1>(0, 3))
                             .norm();
}

void PoseGraphManager::saveFlagCallback(const std_msgs::msg::String::ConstSharedPtr &msg) {
  std::string save_dir        = !msg->data.empty() ? msg->data : package_path_;
  std::string seq_directory   = save_dir + "/" + seq_name_;
  std::string scans_directory = seq_directory + "/scans";

  if (save_in_kitti_format_) {
    RCLCPP_INFO(this->get_logger(),
                "Scans are saved in %s, following the KITTI and TUM format",
                scans_directory.c_str());

    if (fs::exists(seq_directory)) {
      fs::remove_all(seq_directory);
    }
    fs::create_directories(scans_directory);

    std::ofstream kitti_pose_file(seq_directory + "/poses_kitti.txt");
    std::ofstream tum_pose_file(seq_directory + "/poses_tum.txt");
    tum_pose_file << "#timestamp x y z qx qy qz qw\n";

    {
      std::lock_guard<std::mutex> lock(keyframes_mutex_);
      for (size_t i = 0; i < keyframes_.size(); ++i) {
        std::stringstream ss_;
        ss_ << scans_directory << "/" << std::setw(6) << std::setfill('0') << i << ".pcd";
        RCLCPP_INFO(this->get_logger(), "Saving %s...", ss_.str().c_str());
        pcl::io::savePCDFileASCII<PointType>(ss_.str(), keyframes_[i].scan_);

        const auto &pose_ = keyframes_[i].pose_corrected_;
        kitti_pose_file << pose_(0, 0) << " " << pose_(0, 1) << " " << pose_(0, 2) << " "
                        << pose_(0, 3) << " " << pose_(1, 0) << " " << pose_(1, 1) << " "
                        << pose_(1, 2) << " " << pose_(1, 3) << " " << pose_(2, 0) << " "
                        << pose_(2, 1) << " " << pose_(2, 2) << " " << pose_(2, 3) << "\n";

        const auto &lidar_optim_pose_ =
            eigenToPoseStamped(keyframes_[i].pose_corrected_, map_frame_);
        tum_pose_file << std::fixed << std::setprecision(8) << keyframes_[i].timestamp_ << " "
                      << lidar_optim_pose_.pose.position.x << " "
                      << lidar_optim_pose_.pose.position.y << " "
                      << lidar_optim_pose_.pose.position.z << " "
                      << lidar_optim_pose_.pose.orientation.x << " "
                      << lidar_optim_pose_.pose.orientation.y << " "
                      << lidar_optim_pose_.pose.orientation.z << " "
                      << lidar_optim_pose_.pose.orientation.w << "\n";
      }
    }
    kitti_pose_file.close();
    tum_pose_file.close();
    RCLCPP_INFO(this->get_logger(), "Scans and poses saved in .pcd and KITTI format");
  }
  if (save_map_bag_) {
    RCLCPP_INFO(this->get_logger(),
                "NOTE(hlim): rosbag2 saving not directly implemented; skipping.");
  }
  if (save_map_pcd_) {
    pcl::PointCloud<PointType>::Ptr corrected_map(new pcl::PointCloud<PointType>());
    corrected_map->reserve(keyframes_[0].scan_.size() * keyframes_.size());

    {
      std::lock_guard<std::mutex> lock(keyframes_mutex_);
      for (size_t i = 0; i < keyframes_.size(); ++i) {
        *corrected_map += transformPcd(keyframes_[i].scan_, keyframes_[i].pose_corrected_);
      }
    }
    const auto &voxelized_map = voxelize(corrected_map, save_voxel_res_);
    pcl::io::savePCDFileASCII<PointType>(seq_directory + "/" + seq_name_ + "_map.pcd",
                                         *voxelized_map);
    RCLCPP_INFO(this->get_logger(), "Accumulated map cloud saved in .pcd format");
  }
  if (save_pose_graph_) {
    if (!fs::exists(seq_directory)) {
      fs::create_directories(seq_directory);
    }
    const std::string g2o_path = seq_directory + "/graph.g2o";
    {
      std::lock_guard<std::mutex> lock(graph_mutex_);
      std::lock_guard<std::mutex> lock_rt(realtime_pose_mutex_);
      gtsam::writeG2o(persistent_graph_, corrected_esti_, g2o_path);
    }
    RCLCPP_INFO(this->get_logger(), "Pose graph saved to %s", g2o_path.c_str());
  }
}

bool PoseGraphManager::tryRelocalize() {
  if (prior_keyframes_.empty()) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(),
                          *this->get_clock(),
                          2000,
                          "Bootstrap reloc requires prior_session_dir "
                          "(prior_keyframes_ is empty).");
    return false;
  }

  // Throttle by motion: only push a new scan into the query-side buffer once
  // per `bootstrap_scan_distance_` of movement so buffered scans are spatially
  // distributed rather than piled up while the robot is stationary.
  const Eigen::Vector3d current_pos = current_frame_.pose_.block<3, 1>(0, 3);
  const bool first_try = !reloc_has_last_accum_pose_;
  const bool moved_enough =
      first_try ||
      (current_pos - reloc_last_accum_pose_.block<3, 1>(0, 3)).norm() >= bootstrap_scan_distance_;
  if (!moved_enough) return false;
  reloc_last_accum_pose_     = current_frame_.pose_;
  reloc_has_last_accum_pose_ = true;

  // Push current scan into the query-side ring buffer. pose_corrected_ is
  // the new-odom pose pre-multiplied by `bootstrap_T_init_` so the query
  // lives in (an approximation of) the prior-map frame. Radius filtering +
  // KISS-Matcher absorb the remaining residual misalignment.
  PoseGraphNode buffered   = current_frame_;
  buffered.pose_corrected_ = bootstrap_T_init_ * current_frame_.pose_;
  reloc_scan_buffer_.push_back(std::move(buffered));
  while (reloc_scan_buffer_.size() > bootstrap_submap_scans_) {
    reloc_scan_buffer_.pop_front();
  }

  // Copy the deque into a contiguous vector for performInterSessionLoopClosure.
  // The last element is the "center" of the query submap (all preceding
  // scans are folded in by accumulateSubmap's start = center - submap_range).
  const std::vector<PoseGraphNode> query_vec(reloc_scan_buffer_.begin(),
                                             reloc_scan_buffer_.end());
  const size_t query_center_idx = query_vec.size() - 1;
  const PoseGraphNode &query    = query_vec[query_center_idx];

  const auto candidates = loop_closure_->fetchInterSessionLoopCandidates(
      query, prior_keyframes_, bootstrap_max_attempts_per_tick_, bootstrap_radius_);

  if (candidates.empty()) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Bootstrap reloc: no prior keyframes within %.1fm of current pose. "
        "Waiting for motion.",
        bootstrap_radius_);
    return false;
  }

  const Eigen::Vector3d qpos = query.pose_corrected_.block<3, 1>(0, 3);
  RCLCPP_INFO(this->get_logger(),
              "Bootstrap reloc: query @ (%.2f, %.2f, %.2f), buffer=%lu, "
              "trying %lu candidates within %.1fm:",
              qpos.x(), qpos.y(), qpos.z(),
              query_vec.size(),
              candidates.size(),
              bootstrap_radius_);
  for (const auto &pair : candidates) {
    const size_t m             = pair.second;
    const Eigen::Vector3d ppos = prior_keyframes_[m].pose_corrected_.block<3, 1>(0, 3);
    const double dxy           = (ppos.head<2>() - qpos.head<2>()).norm();
    RCLCPP_INFO(this->get_logger(),
                "  prior[%lu] @ (%.2f, %.2f, %.2f)  dXY=%.2fm",
                m, ppos.x(), ppos.y(), ppos.z(), dxy);
  }

  for (const auto &pair : candidates) {
    const size_t match_idx = pair.second;
    const RegOutput reg    = loop_closure_->performInterSessionLoopClosure(
        query_vec, prior_keyframes_, query_center_idx, match_idx,
        bootstrap_voxel_resolution_);

    // Publish the submaps KISS-Matcher was actually fed, so we can see them in
    // RViz even when bootstrap is failing (the normal /lc/* pubs only fire on
    // successful LCs). Frame is the prior-map frame since each scan was already
    // transformed by its pose_corrected_ inside performInterSessionLoopClosure.
    const auto stamp = this->now();
    debug_src_pub_->publish(toROSMsg(loop_closure_->getSourceCloud(), map_frame_, stamp));
    debug_tgt_pub_->publish(toROSMsg(loop_closure_->getTargetCloud(), map_frame_, stamp));
    debug_coarse_aligned_pub_->publish(
        toROSMsg(loop_closure_->getCoarseAlignedCloud(), map_frame_, stamp));

    if (!reg.is_valid_) {
      RCLCPP_WARN(this->get_logger(),
                  "  candidate prior=%lu rejected (overlap=%.1f%%).",
                  match_idx,
                  reg.overlapness_);
      continue;
    }
    // reg.pose_ aligns the pre-transformed query (bootstrap_T_init_ * new-odom)
    // into the prior frame, so compose to recover the raw new-odom -> prior
    // transform.
    T_priormap_from_newodom_ = reg.pose_ * bootstrap_T_init_;
    reloc_succeeded_         = true;
    reloc_scan_buffer_.clear();
    RCLCPP_INFO(this->get_logger(),
                "\033[1;32mBootstrap reloc succeeded via prior keyframe %lu "
                "(inliers=%lu, overlap=%.1f%%). Pose graph will run in the "
                "prior-map frame.\033[0m",
                match_idx,
                reg.num_final_inliers_,
                reg.overlapness_);
    return true;
  }
  return false;
}

bool PoseGraphManager::loadPriorSession() {
  const fs::path dir(prior_session_dir_);
  const fs::path poses_path = dir / "poses_tum.txt";
  const fs::path scans_dir  = dir / "scans";
  const fs::path g2o_path   = dir / "graph.g2o";

  RCLCPP_INFO(this->get_logger(),
              "[prior] Loading prior session from %s", dir.c_str());

  if (!fs::exists(poses_path)) {
    RCLCPP_ERROR(this->get_logger(),
                 "prior_session_dir has no poses_tum.txt: %s",
                 poses_path.c_str());
    return false;
  }
  if (!fs::exists(scans_dir)) {
    RCLCPP_ERROR(this->get_logger(),
                 "prior_session_dir has no scans/ subdir: %s",
                 scans_dir.c_str());
    return false;
  }

  RCLCPP_INFO(this->get_logger(), "[prior] Parsing poses_tum.txt ...");
  std::ifstream pf(poses_path);
  std::string line;
  std::vector<std::pair<double, Eigen::Matrix4d>> tum_poses;
  while (std::getline(pf, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    double t, x, y, z, qx, qy, qz, qw;
    if (!(ss >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
    tf2::Quaternion q(qx, qy, qz, qw);
    tf2::Matrix3x3 rot_tf(q);
    Eigen::Matrix3d rot;
    matrixTF2ToEigen(rot_tf, rot);
    Eigen::Matrix4d pose       = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0)     = rot;
    pose.block<3, 1>(0, 3)     = Eigen::Vector3d(x, y, z);
    tum_poses.emplace_back(t, pose);
  }
  if (tum_poses.empty()) {
    RCLCPP_ERROR(this->get_logger(), "poses_tum.txt is empty: %s", poses_path.c_str());
    return false;
  }
  RCLCPP_INFO(this->get_logger(),
              "[prior] Parsed %lu poses. Loading scans ...", tum_poses.size());

  prior_keyframes_.clear();
  prior_keyframes_.reserve(tum_poses.size());
  const size_t log_every = std::max<size_t>(1, tum_poses.size() / 10);
  for (size_t i = 0; i < tum_poses.size(); ++i) {
    std::stringstream scan_ss;
    scan_ss << scans_dir.string() << "/" << std::setw(6) << std::setfill('0') << i << ".pcd";
    const std::string scan_path = scan_ss.str();

    kiss_matcher::PoseGraphNode node;
    if (pcl::io::loadPCDFile<PointType>(scan_path, node.scan_) != 0) {
      RCLCPP_WARN(this->get_logger(),
                  "Failed to load prior scan %s. Skipping keyframe %lu.",
                  scan_path.c_str(), i);
      continue;
    }
    node.pose_                   = tum_poses[i].second;
    node.pose_corrected_         = tum_poses[i].second;
    node.timestamp_              = tum_poses[i].first;
    node.idx_                    = i;
    // Prior keyframes never originate outgoing candidate queries.
    node.nnsearch_processed_      = true;
    node.loop_detector_processed_ = true;
    prior_keyframes_.push_back(std::move(node));

    if ((i + 1) % log_every == 0 || i + 1 == tum_poses.size()) {
      RCLCPP_INFO(this->get_logger(),
                  "[prior]   scans loaded: %lu / %lu", i + 1, tum_poses.size());
    }
  }
  if (prior_keyframes_.empty()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to load any prior keyframes from %s",
                 scans_dir.c_str());
    return false;
  }
  RCLCPP_INFO(this->get_logger(),
              "Loaded %lu prior keyframes from %s",
              prior_keyframes_.size(), prior_session_dir_.c_str());

  gtsam::NonlinearFactorGraph prior_graph;
  gtsam::Values prior_values;
  bool loaded_g2o = false;
  if (fs::exists(g2o_path)) {
    RCLCPP_INFO(this->get_logger(),
                "[prior] Loading g2o graph from %s ...", g2o_path.c_str());
    try {
      auto parsed = gtsam::readG2o(g2o_path.string(), true /* is3D */);
      RCLCPP_INFO(this->get_logger(),
                  "[prior]   readG2o returned %lu factors, %lu values. Re-keying ...",
                  parsed.first->size(), parsed.second->size());
      // Re-key into prior_session_prefix_ namespace using the saved keys' indices.
      // readG2o returns keys that may themselves be Symbols (when the writer
      // used Symbol keys) — gtsam::Symbol handles a plain integer key as
      // prefix='\0', so Symbol::index() extracts the original integer regardless.
      for (const auto &key_pose : *parsed.second) {
        const gtsam::Symbol old_sym(key_pose.key);
        const gtsam::Symbol new_sym(prior_session_prefix_, old_sym.index());
        prior_values.insert(new_sym,
                            key_pose.value.cast<gtsam::Pose3>());
      }
      for (const auto &factor : *parsed.first) {
        // Clone factor into a re-keyed version. Only BetweenFactor and
        // PriorFactor appear in a kiss_matcher session.
        if (auto bf = boost::dynamic_pointer_cast<
                gtsam::BetweenFactor<gtsam::Pose3>>(factor)) {
          const gtsam::Symbol old_k1(bf->key1());
          const gtsam::Symbol old_k2(bf->key2());
          prior_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
              gtsam::Symbol(prior_session_prefix_, old_k1.index()),
              gtsam::Symbol(prior_session_prefix_, old_k2.index()),
              bf->measured(),
              bf->noiseModel()));
        } else if (auto pf2 = boost::dynamic_pointer_cast<
                       gtsam::PriorFactor<gtsam::Pose3>>(factor)) {
          const gtsam::Symbol old_k(pf2->key());
          prior_graph.add(gtsam::PriorFactor<gtsam::Pose3>(
              gtsam::Symbol(prior_session_prefix_, old_k.index()),
              pf2->prior(),
              pf2->noiseModel()));
        }
      }
      // writeG2o does not serialize PriorFactor, so the deserialized graph
      // has no anchor and is gauge-free (6-DoF). Re-add a tight prior on the
      // first prior-session node so ISAM2 can linearize without hitting an
      // indeterminant linear system.
      const gtsam::Symbol sym0(prior_session_prefix_, 0);
      if (prior_values.exists(sym0)) {
        auto anchor_variance =
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
        auto anchor_noise = gtsam::noiseModel::Diagonal::Variances(anchor_variance);
        prior_graph.add(gtsam::PriorFactor<gtsam::Pose3>(
            sym0, prior_values.at<gtsam::Pose3>(sym0), anchor_noise));
        RCLCPP_INFO(this->get_logger(),
                    "[prior]   Added anchor PriorFactor on %c0",
                    prior_session_prefix_);
      }

      loaded_g2o = true;
      RCLCPP_INFO(this->get_logger(),
                  "[prior] Loaded prior graph from %s (%lu factors, %lu values)",
                  g2o_path.c_str(), prior_graph.size(), prior_values.size());
    } catch (const std::exception &e) {
      RCLCPP_WARN(this->get_logger(),
                  "Failed to parse %s: %s. Falling back to synthesized chain.",
                  g2o_path.c_str(), e.what());
    }
  }

  if (!loaded_g2o) {
    // Fallback: synthesize a rigid chain of BetweenFactors from consecutive
    // pose deltas with a default noise model. No intra-prior LC edges are
    // recovered, so the prior trajectory is effectively frozen shape-wise but
    // can still rigidly translate/rotate as a block.
    RCLCPP_WARN(this->get_logger(),
                "No graph.g2o in %s; synthesizing prior chain from TUM poses.",
                prior_session_dir_.c_str());
    auto variance_vector =
        (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished();
    auto default_noise = gtsam::noiseModel::Diagonal::Variances(variance_vector);

    const gtsam::Symbol sym0(prior_session_prefix_, 0);
    prior_graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        sym0, eigenToGtsam(prior_keyframes_[0].pose_corrected_), default_noise));
    prior_values.insert(sym0, eigenToGtsam(prior_keyframes_[0].pose_corrected_));

    for (size_t i = 1; i < prior_keyframes_.size(); ++i) {
      const gtsam::Symbol sym_prev(prior_session_prefix_, i - 1);
      const gtsam::Symbol sym_curr(prior_session_prefix_, i);
      gtsam::Pose3 p_prev = eigenToGtsam(prior_keyframes_[i - 1].pose_corrected_);
      gtsam::Pose3 p_curr = eigenToGtsam(prior_keyframes_[i].pose_corrected_);
      prior_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
          sym_prev, sym_curr, p_prev.between(p_curr), default_noise));
      prior_values.insert(sym_curr, p_curr);
    }
  }

  // Merge into persistent graph so a subsequent save round-trips prior + new
  // as a single combined session, and seed ISAM2 with the prior values so
  // inter-session BetweenFactors can attach.
  RCLCPP_INFO(this->get_logger(),
              "[prior] Seeding ISAM2 with %lu factors and %lu values ...",
              prior_graph.size(), prior_values.size());
  persistent_graph_ += prior_graph;
  {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    isam_handler_->update(prior_graph, prior_values);
    isam_handler_->update();
  }
  RCLCPP_INFO(this->get_logger(), "[prior] ISAM2 initial update OK. Calculating estimate ...");
  {
    std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
    corrected_esti_ = isam_handler_->calculateEstimate();
  }

  RCLCPP_INFO(this->get_logger(),
              "[prior] Prior session incorporated into ISAM2 (prefix '%c').",
              prior_session_prefix_);
  return true;
}

// ----------------------------------------------------------------------

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;

  auto node = std::make_shared<PoseGraphManager>(options);

  // To allow timer callbacks to run concurrently using multiple threads
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
