#!/usr/bin/env python3
import os

import kiss_matcher
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from kiss_matcher.io_utils import read_bin, read_pcd, read_ply, write_pcd
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster


def load_cloud(path):
    if path.endswith(".bin"):
        return read_bin(path)
    if path.endswith(".pcd"):
        return read_pcd(path)
    if path.endswith(".ply"):
        return read_ply(path)
    raise ValueError(f"Unsupported file format: {path}. Use .bin, .pcd, or .ply")


def remove_nan(points):
    return points[np.isfinite(points).any(axis=1)]


def rotation_matrix_to_quaternion(R):
    # Returns (x, y, z, w)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


class KissMatcherNode(Node):
    def __init__(self):
        super().__init__("run_kiss_matcher")

        self.declare_parameter("src_path", "")
        self.declare_parameter("tgt_path", "")
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("source_frame", "source")
        self.declare_parameter("target_frame", "target")
        self.declare_parameter("publish_rate", 1.0)
        self.declare_parameter("save_transformed_cloud", False)
        self.declare_parameter("transformed_cloud_path", "")
        self.declare_parameter("save_tf", False)
        self.declare_parameter("tf_path", "")

        self.src_path = self.get_parameter("src_path").value
        self.tgt_path = self.get_parameter("tgt_path").value
        self.resolution = float(self.get_parameter("resolution").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.source_frame = self.get_parameter("source_frame").value
        self.target_frame = self.get_parameter("target_frame").value
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.save_cloud_flag = bool(self.get_parameter("save_transformed_cloud").value)
        self.transformed_cloud_path = self.get_parameter("transformed_cloud_path").value
        self.save_tf_flag = bool(self.get_parameter("save_tf").value)
        self.tf_path = self.get_parameter("tf_path").value

        if not self.src_path or not self.tgt_path:
            raise RuntimeError("src_path and tgt_path parameters must be set")

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.src_pub = self.create_publisher(PointCloud2, "src_cloud", latched)
        self.tgt_pub = self.create_publisher(PointCloud2, "tgt_cloud", latched)
        self.transformed_pub = self.create_publisher(
            PointCloud2, "transformed_src_cloud", latched
        )
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        self.run_registration()
        self.publish_once()
        self.timer = self.create_timer(1.0 / max(self.publish_rate, 1e-3), self.publish_once)

    def run_registration(self):
        self.get_logger().info(f"Loading source: {self.src_path}")
        src = remove_nan(load_cloud(self.src_path))
        self.get_logger().info(f"Loading target: {self.tgt_path}")
        tgt = remove_nan(load_cloud(self.tgt_path))

        self.get_logger().info(
            f"src: {src.shape}, tgt: {tgt.shape}, resolution: {self.resolution}"
        )

        config = kiss_matcher.KISSMatcherConfig(self.resolution)
        matcher = kiss_matcher.KISSMatcher(config)
        result = matcher.estimate(src, tgt)
        matcher.print()

        num_final_inliers = matcher.get_num_final_inliers()
        if num_final_inliers < 5:
            self.get_logger().warn(
                f"Registration might have failed (final inliers: {num_final_inliers})"
            )
        else:
            self.get_logger().info(
                f"Registration likely succeeded (final inliers: {num_final_inliers})"
            )

        R = np.array(result.rotation)
        t = np.array(result.translation).reshape(3)
        transformed_src = (R @ src.T).T + t

        self.src = src
        self.tgt = tgt
        self.transformed_src = transformed_src
        self.rotation = R
        self.translation = t

        if self.save_cloud_flag:
            out = self.transformed_cloud_path or "transformed_src.pcd"
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            write_pcd(transformed_src, out)
            self.get_logger().info(f"Saved transformed cloud to {out}")

        if self.save_tf_flag:
            out = self.tf_path or "transform.txt"
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            np.savetxt(out, T, fmt="%.9f")
            self.get_logger().info(f"Saved 4x4 transform to {out}")

        self.publish_static_tf()

    def publish_static_tf(self):
        qx, qy, qz, qw = rotation_matrix_to_quaternion(self.rotation)
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target_frame
        msg.child_frame_id = self.source_frame
        msg.transform.translation.x = float(self.translation[0])
        msg.transform.translation.y = float(self.translation[1])
        msg.transform.translation.z = float(self.translation[2])
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(msg)

    def _to_msg(self, points, frame):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame
        return point_cloud2.create_cloud_xyz32(header, points.astype(np.float32).tolist())

    def publish_once(self):
        self.src_pub.publish(self._to_msg(self.src, self.source_frame))
        self.tgt_pub.publish(self._to_msg(self.tgt, self.target_frame))
        self.transformed_pub.publish(self._to_msg(self.transformed_src, self.target_frame))


def main():
    rclpy.init()
    node = KissMatcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
