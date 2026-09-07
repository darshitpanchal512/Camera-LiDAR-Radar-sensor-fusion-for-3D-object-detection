import os
import rclpy
from rclpy.node import Node

import numpy as np
np.float = float          # old-numpy shim, same as your other nodes
import cv2

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from tf2_ros import TransformException
import tf_transformations

# --- free any leftover GPU memory before we load YOLO (helps the 2GB GPU) ---
import torch
torch.cuda.empty_cache()
from ultralytics import YOLO

# =========================================================
# Settings
# =========================================================
YOLO_MODEL     = os.environ.get("YOLO_ENGINE_PATH", "./yolo26n.engine")   # override with YOLO_ENGINE_PATH env var
CONF_THRESH    = 0.4
WANTED_CLASSES = [0, 2]    # COCO ids -> 0 = person, 2 = car
MARGIN_PX      = 5         # slightly enlarge each box to catch edge points


class FrustumExtractor(Node):
    """
    Same 'use the latest frame' style as your working lidar_projection.py.
    We do NOT wait for a perfectly time-matched pair (that was why the old
    node never did anything). Instead:
      - we always remember the most recent LiDAR scan
      - every time a fresh CAMERA frame arrives, we run YOLO on it, project
        the latest LiDAR points, and keep the points that fall inside the
        person/car boxes. Those points are the 'frustum'.
    """

    def __init__(self):
        super().__init__('frustum_extractor_node')

        self.get_logger().info(f"Loading YOLO detection model: {YOLO_MODEL}")
        self.model = YOLO(YOLO_MODEL, task='detect')

        self.bridge = CvBridge()

        # ---- TF (same lookup direction as lidar_projection.py) ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- your real camera intrinsics ----
        self.fx, self.cx = 880.5002, 926.3907
        self.fy, self.cy = 878.7256, 578.7613

        # ---- latest-frame cache ----
        self.latest_points = None     # numpy (N,3) in velodyne frame

        # ---- subscribers ----
        self.image_sub = self.create_subscription(
            Image, '/blackfly_s/cam0/image_rectified', self.image_cb, 10)
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/filtered_points', self.lidar_cb, 10)

        # ---- publisher (merged frustum cloud, for RViz) ----
        self.frustum_pub = self.create_publisher(
            PointCloud2, '/yolo/frustum_points', 10)

        self.get_logger().info(
            "Frustum Extractor online (latest-frame mode). "
            "Will process on every fresh camera frame.")

    # =========================================================
    # LiDAR: just remember the most recent scan
    # =========================================================
    def lidar_cb(self, msg):
        raw = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts = [(p[0], p[1], p[2]) for p in raw]
        if len(pts) == 0:
            self.latest_points = None
            return
        self.latest_points = np.array(pts, dtype=np.float64)

    # =========================================================
    # Camera: this drives the work (fresh image + latest LiDAR)
    # =========================================================
    def image_cb(self, msg):
        if self.latest_points is None:
            self.get_logger().warn(
                "Got a camera frame but no LiDAR scan yet -- waiting...",
                throttle_duration_sec=3.0)
            return

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = img.shape[:2]
        points = self.latest_points

        # ---- TF velodyne -> cam0 ----
        T = self.get_transform()
        if T is None:
            return

        # ---- project ALL points into the camera ----
        ones = np.ones((points.shape[0], 1))
        pts_h = np.concatenate([points, ones], axis=1)
        pts_cam = (T @ pts_h.T).T

        X = pts_cam[:, 0]
        Y = pts_cam[:, 1]
        Z = pts_cam[:, 2]

        front = Z > 0                     # keep points in front of the camera
        Xf, Yf, Zf = X[front], Y[front], Z[front]
        pts_front = points[front]         # original velodyne XYZ (for output)

        if len(pts_front) == 0:
            return

        u = (self.fx * Xf / Zf) + self.cx
        v = (self.fy * Yf / Zf) + self.cy

        # ---- run YOLO on the fresh camera frame ----
        results = self.model(img, verbose=False,
                             conf=CONF_THRESH, classes=WANTED_CLASSES)

        frustum_points = []   # velodyne-frame XYZ of points inside any box
        n_objects = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])

            x1 -= MARGIN_PX; y1 -= MARGIN_PX
            x2 += MARGIN_PX; y2 += MARGIN_PX

            inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            obj_pts = pts_front[inside]
            if len(obj_pts) == 0:
                continue

            n_objects += 1
            frustum_points.extend(obj_pts.tolist())
            name = 'person' if cls_id == 0 else 'car'
            self.get_logger().info(f"  {name}: {len(obj_pts)} points in frustum")

        if len(frustum_points) == 0:
            self.get_logger().warn(
                "Boxes found, but no LiDAR points landed inside them this frame.",
                throttle_duration_sec=3.0)
            return

        # ---- publish the merged frustum cloud (in velodyne frame) ----
        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = 'velodyne'
        cloud = pc2.create_cloud_xyz32(header, frustum_points)
        self.frustum_pub.publish(cloud)
        self.get_logger().info(
            f"Published frustum cloud: {len(frustum_points)} points "
            f"from {n_objects} object(s)")

    # =========================================================
    # TF lookup (identical direction to lidar_projection.py)
    # =========================================================
    def get_transform(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'cam0', 'velodyne', rclpy.time.Time())
            trans = tf.transform.translation
            rot = tf.transform.rotation
            T = tf_transformations.quaternion_matrix(
                [rot.x, rot.y, rot.z, rot.w])
            T[0, 3] = trans.x
            T[1, 3] = trans.y
            T[2, 3] = trans.z
            return T
        except TransformException as ex:
            self.get_logger().warn(f"TF error: {ex}", throttle_duration_sec=3.0)
            return None


def main():
    rclpy.init()
    node = FrustumExtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
