#pragma once

#ifndef KISS_MATCHER_LOOP_CLOSURE_H
#define KISS_MATCHER_LOOP_CLOSURE_H

///// C++ common headers
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Eigen>
#include <kiss_matcher/KISSMatcher.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>

#include "rclcpp/rclcpp.hpp"
#include "slam/loop_types.hpp"
#include "slam/pose_graph_node.hpp"
#include "slam/utils.hpp"

using NodePair = std::tuple<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>;

namespace kiss_matcher {
struct GICPConfig {
  int num_threads_               = 4;
  int correspondence_randomness_ = 20;
  int max_num_iter_              = 20;

  double max_corr_dist_              = 1.0;
  double scale_factor_for_corr_dist_ = 5.0;
  double overlap_threshold_          = 90.0;
};

struct LoopClosureConfig {
  bool verbose_                    = false;
  bool enable_global_registration_ = true;
  bool is_multilayer_env_          = false;
  size_t num_submap_keyframes_     = 11;
  size_t num_inliers_threshold_    = 100;
  double voxel_res_                = 0.1;
  double loop_detection_radius_;
  double loop_detection_timediff_threshold_;
  GICPConfig gicp_config_;
  KISSMatcherConfig matcher_config_;
};

// Registration Output
struct RegOutput {
  bool is_valid_            = false;
  bool is_converged_        = false;
  size_t num_final_inliers_ = 0;
  double overlapness_       = 0.0;
  Eigen::Matrix4d pose_     = Eigen::Matrix4d::Identity();
};

class LoopClosure {
 private:
  // For coarse-to-fine alignment
  std::shared_ptr<kiss_matcher::KISSMatcher> global_reg_handler_                        = nullptr;
  std::shared_ptr<small_gicp::RegistrationPCL<PointType, PointType>> local_reg_handler_ = nullptr;

  pcl::PointCloud<PointType>::Ptr src_cloud_;
  pcl::PointCloud<PointType>::Ptr tgt_cloud_;
  pcl::PointCloud<PointType>::Ptr coarse_aligned_;
  pcl::PointCloud<PointType>::Ptr aligned_;
  pcl::PointCloud<PointType>::Ptr debug_cloud_;
  LoopClosureConfig config_;

  rclcpp::Logger logger_;

  std::chrono::steady_clock::time_point last_success_icp_time_;
  bool has_success_icp_time_ = false;

 public:
  explicit LoopClosure(const LoopClosureConfig &config, const rclcpp::Logger &logger);
  ~LoopClosure();
  double calculateDistance(const Eigen::Matrix4d &pose1, const Eigen::Matrix4d &pose2);

  LoopCandidates getLoopCandidatesFromQuery(const PoseGraphNode &query_frame,
                                            const std::vector<PoseGraphNode> &keyframes);

  LoopCandidate getClosestCandidate(const LoopCandidates &candidates);

  LoopIdxPairs fetchClosestLoopCandidate(const PoseGraphNode &query_frame,
                                         const std::vector<PoseGraphNode> &keyframes);

  LoopIdxPairs fetchLoopCandidates(const PoseGraphNode &query_frame,
                                   const std::vector<PoseGraphNode> &keyframes,
                                   const size_t num_max_candidates  = 3,
                                   const double reliable_window_sec = 30);

  NodePair setSrcAndTgtCloud(const std::vector<PoseGraphNode> &keyframes,
                             const size_t src_idx,
                             const size_t tgt_idx,
                             const size_t num_submap_keyframes,
                             const double voxel_res,
                             const bool enable_global_registration);

  void setSrcAndTgtCloud(const pcl::PointCloud<PointType> &src_cloud,
                         const pcl::PointCloud<PointType> &tgt_cloud);

  RegOutput icpAlignment(const pcl::PointCloud<PointType> &src,
                         const pcl::PointCloud<PointType> &tgt);

  // `num_inliers_threshold_override` >= 0 overrides config_.num_inliers_threshold_
  // for this call only (used by bootstrap reloc when a looser threshold is needed).
  RegOutput coarseToFineAlignment(const pcl::PointCloud<PointType> &src,
                                  const pcl::PointCloud<PointType> &tgt,
                                  const int num_inliers_threshold_override = -1);

  RegOutput performLoopClosure(const PoseGraphNode &query_keyframe,
                               const std::vector<PoseGraphNode> &keyframes);

  RegOutput performLoopClosure(const std::vector<PoseGraphNode> &keyframes,
                               const size_t query_idx,
                               const size_t match_idx);

  // Inter-session candidate selection: radius-filter `prior_keyframes` against
  // `query_frame.pose_corrected_` only (no time-diff check — prior timestamps
  // are from a previous run and not comparable). Returns up to
  // `num_max_candidates` pairs where `first` is the query's own idx_ and
  // `second` is the prior-keyframe index in the input vector. Pass a positive
  // `radius` to override `config_.loop_detection_radius_` (used by bootstrap
  // reloc where a larger search radius is needed).
  LoopIdxPairs fetchInterSessionLoopCandidates(
      const PoseGraphNode &query_frame,
      const std::vector<PoseGraphNode> &prior_keyframes,
      const size_t num_max_candidates = 3,
      const double radius             = -1.0);

  // Inter-session registration: stitches a submap from `query_keyframes`
  // around `query_idx` and from `match_keyframes` around `match_idx`, then
  // runs the same coarse-to-fine (or GICP-only) alignment as intra-session
  // loop closure. Returns a RegOutput; the caller is responsible for adding
  // a cross-prefix BetweenFactor on success.
  // `voxel_res_override` > 0 pre-voxelizes both submaps at a custom resolution
  // instead of `config_.voxel_res_` — used by bootstrap reloc when FPFH needs
  // a different density than steady-state loop closure.
  // `num_inliers_threshold_override` >= 0 overrides
  // `config_.num_inliers_threshold_` for the coarse stage (bootstrap reloc
  // often needs a looser threshold than steady-state LCs).
  RegOutput performInterSessionLoopClosure(
      const std::vector<PoseGraphNode> &query_keyframes,
      const std::vector<PoseGraphNode> &match_keyframes,
      const size_t query_idx,
      const size_t match_idx,
      const double voxel_res_override            = -1.0,
      const int num_inliers_threshold_override   = -1);

  pcl::PointCloud<PointType> getSourceCloud();
  pcl::PointCloud<PointType> getTargetCloud();
  pcl::PointCloud<PointType> getCoarseAlignedCloud();
  pcl::PointCloud<PointType> getFinalAlignedCloud();
  pcl::PointCloud<PointType> getDebugCloud();
};
}  // namespace kiss_matcher
#endif  // KISS_MATCHER_LOOP_CLOSURE_H
