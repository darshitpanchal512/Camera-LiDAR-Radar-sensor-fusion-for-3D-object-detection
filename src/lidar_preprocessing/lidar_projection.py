import rclpy
from rclpy.node import Node

import numpy as np
np.float = float
import cv2
import yaml
import os

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from tf2_ros import TransformException

import tf_transformations

from ament_index_python.packages import get_package_share_directory


class LidarImageFusion(Node):

    def __init__(self):
        super().__init__('lidar_image_fusion')

        # -----------------------------
        # Subscribers
        # -----------------------------
        self.image_sub = self.create_subscription(
            Image,
            '/blackfly_s/cam0/image_rectified',
            self.image_callback,
            10)

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/filtered_points',
            self.lidar_callback,
            10)

        # -----------------------------
        # CV Bridge
        # -----------------------------
        self.bridge = CvBridge()

        # -----------------------------
        # TF
        # -----------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -----------------------------
        # Data storage
        # -----------------------------
        self.latest_image = None
        self.latest_points = None

        # Time Sync & Fused Publisher (ADD THIS HERE)
        # -----------------------------
        self.latest_image_header = None
        self.fusion_pub = self.create_publisher(Image, '/sensor_fusion/projected_image', 10)
        # -----------------------------
        # Load calibration YAML
        # -----------------------------
        self.load_camera_calibration()
    # =========================================================
    # Load YAML from Native Workspace Safely
    # =========================================================
    # =========================================================
    # Load YAML from Native Workspace Safely
    # =========================================================
    def load_camera_calibration(self):
        # Path to the camera calibration YAML. Override with:
        #   export CAMERA_CALIB_YAML=/path/to/calibration.yaml
        yaml_path = os.environ.get(
            'CAMERA_CALIB_YAML',
            './calibration/camera_front_center_autoware_camera_calibration.yaml')

        self.get_logger().info(f"Loading calibration from: {yaml_path}")

        try:
            with open(yaml_path, 'r') as f:
                yaml_text = f.read()
            
            #  THE ULTIMATE CLEANUP: Neutralize all OpenCV custom metadata
            yaml_text = yaml_text.replace("%YAML:1.0", "#%YAML:1.0")
            yaml_text = yaml_text.replace("!!opencv-matrix", "")

            # Since it's now clean, standard vanilla YAML, safe_load works perfectly
            calib = yaml.safe_load(yaml_text)
                
            # OpenCV stores the 3x3 matrix as a flat list of 9 numbers inside 'data'
            matrix_data = calib['CameraMat']['data']
            
            # Extracting the exact values based on standard 3x3 Intrinsic layout
            self.fx = matrix_data[0]  # Row 1, Col 1
            self.cx = matrix_data[2]  # Row 1, Col 3
            self.fy = matrix_data[4]  # Row 2, Col 2
            self.cy = matrix_data[5]  # Row 2, Col 3

            self.get_logger().info(
                f"SUCCESS: Loaded intrinsics fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Failed to load YAML: {e}")
            # Fallback to the hardcoded true values if the file read fails
            self.fx, self.cx = 880.5002, 926.3907
            self.fy, self.cy = 878.7256, 578.7613
    # =========================================================
    # Image callback
    # =========================================================
    def image_callback(self, msg):
        # Save the header timestamp for time synchronization
        self.latest_image_header = msg.header
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process()
    # =========================================================
    # LiDAR callback
    # =========================================================
    def lidar_callback(self, msg):
        raw = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

        # Extract each field individually and stack into plain (N, 3) float64 array
        points_list = [(p[0], p[1], p[2]) for p in raw]

        if len(points_list) == 0:
            self.latest_points = None
            return

        self.latest_points = np.array(points_list, dtype=np.float64)

    # =========================================================
    # TF lookup
    # =========================================================
    def get_transform(self):

        try:
            tf = self.tf_buffer.lookup_transform(
                'cam0',
                'velodyne',
                rclpy.time.Time()
            )

            trans = tf.transform.translation
            rot = tf.transform.rotation

            T = tf_transformations.quaternion_matrix([
                rot.x, rot.y, rot.z, rot.w
            ])

            T[0, 3] = trans.x
            T[1, 3] = trans.y
            T[2, 3] = trans.z

            return T

        except TransformException as ex:
            self.get_logger().warn(f"TF error: {ex}")
            return None

    # =========================================================
    # Main fusion pipeline
    # =========================================================
    def process(self):

        if self.latest_image is None or self.latest_points is None:
            return

        img = self.latest_image.copy()
        points = self.latest_points
        points = np.atleast_2d(points)

        # Get TF transform
        T = self.get_transform()
        if T is None:
            return

        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        pts_h = np.concatenate([points, ones], axis=1)

        # Transform LiDAR -> camera
        pts_cam = (T @ pts_h.T).T

        X = pts_cam[:, 0]
        Y = pts_cam[:, 1]
        Z = pts_cam[:, 2]

        # Keep points in front of camera
        mask = Z > 0
        X, Y, Z = X[mask], Y[mask], Z[mask]

        # -----------------------------
        # Projection using intrinsics
        # -----------------------------
        u = (self.fx * X / Z) + self.cx
        v = (self.fy * Y / Z) + self.cy

        h, w = img.shape[:2]

        valid = (
            (u >= 0) & (u < w) &
            (v >= 0) & (v < h)
        )

        u = u[valid].astype(np.int32)
        v = v[valid].astype(np.int32)

        # Draw points
        for i in range(len(u)):
            cv2.circle(img, (u[i], v[i]), 1, (0, 255, 0), -1)

    #    cv2.imshow("LiDAR projected on rectified image", img)
    #    cv2.waitKey(1)
        # Convert the painted OpenCV image back into a ROS 2 message
        try:
            if self.latest_image_header is not None:
                fusion_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                # Keep the timestamps perfectly synced
                fusion_msg.header = self.latest_image_header 
                
                # Broadcast the finished painting to the ROS 2 network!
                self.fusion_pub.publish(fusion_msg)
        except Exception as e:
            self.get_logger().error(f"CvBridge Conversion Failed: {e}")
# =========================================================
# Main
# =========================================================
def main():
    rclpy.init()
    node = LidarImageFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
# -----------------------------
        