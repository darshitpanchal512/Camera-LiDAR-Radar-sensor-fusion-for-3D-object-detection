# frustum_saver_node.py
# Grabs ONE car frustum from your bag and saves it as an .npz file,
# already transformed into the camera frame (the form the model wants), WITH intensity.
#
# RUN (inside the container) alongside the bag + ground filter, NO other YOLO node:
#   Terminal 1:  ros2 bag play <your_bag.db3> -l --rate 0.1
#   Terminal 2:  ros2 run ground_filter_pkg filter_node
#   Terminal 3:  source /opt/ros/humble/setup.bash ; python3 frustum_saver_node.py
# Wait for "SAVED a car frustum ..." then Ctrl+C everything.
#
# Configuration via environment variables (defaults in brackets):
#   YOLO_ENGINE_PATH        [./yolo26n.engine]
#   SAMPLE_FRUSTUM_PATH     [./sample_frustum.npz]

import os
import rclpy
from rclpy.node import Node
import numpy as np
np.float = float
import cv2

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from tf2_ros import TransformException
import tf_transformations

import torch
torch.cuda.empty_cache()
from ultralytics import YOLO

YOLO_MODEL  = os.environ.get("YOLO_ENGINE_PATH", "./yolo26n.engine")
CONF_THRESH = 0.4
CAR_CLASS   = 2            # COCO: car
SAVE_PATH   = os.environ.get("SAMPLE_FRUSTUM_PATH", "./sample_frustum.npz")
MIN_POINTS  = 10


class FrustumSaver(Node):
    def __init__(self):
        super().__init__('frustum_saver')

        self.get_logger().info(f"Loading YOLO: {YOLO_MODEL}")
        self.model = YOLO(YOLO_MODEL, task='detect')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # your real camera intrinsics
        self.fx, self.cx = 880.5002, 926.3907
        self.fy, self.cy = 878.7256, 578.7613

        self.latest_points = None     # (N,4) velodyne frame [x,y,z,intensity]
        self.saved = False

        self.create_subscription(Image, '/blackfly_s/cam0/image_rectified', self.image_cb, 10)
        self.create_subscription(PointCloud2, '/filtered_points', self.lidar_cb, 10)

        self.get_logger().info("Frustum Saver online. Waiting for a CAR detection...")

    def lidar_cb(self, msg):
        names = [f.name for f in msg.fields]
        if "intensity" in names:
            raw = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
            pts = [(p[0], p[1], p[2], p[3]) for p in raw]
        else:
            raw = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            pts = [(p[0], p[1], p[2], 0.0) for p in raw]
        self.latest_points = np.array(pts, dtype=np.float64) if len(pts) else None

    def get_transform(self):
        try:
            tf = self.tf_buffer.lookup_transform('cam0', 'velodyne', rclpy.time.Time())
            t = tf.transform.translation
            r = tf.transform.rotation
            T = tf_transformations.quaternion_matrix([r.x, r.y, r.z, r.w])
            T[0, 3] = t.x; T[1, 3] = t.y; T[2, 3] = t.z
            return T
        except TransformException as ex:
            self.get_logger().warn(f"TF error: {ex}", throttle_duration_sec=3.0)
            return None

    def image_cb(self, msg):
        if self.saved or self.latest_points is None:
            return

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(img, verbose=False, conf=CONF_THRESH, classes=[CAR_CLASS])
        if len(results[0].boxes) == 0:
            self.get_logger().info("No car in view yet...", throttle_duration_sec=3.0)
            return

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))                 # most confident car
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()

        T = self.get_transform()
        if T is None:
            return

        pts = self.latest_points
        xyz = pts[:, :3]
        inten = pts[:, 3]
        xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        cam = (T @ xyz_hom.T).T[:, :3]               # -> camera frame (x-right, y-down, z-forward)
        X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]

        front = Z > 0
        u = np.full_like(X, -1.0)
        v = np.full_like(X, -1.0)
        u[front] = self.fx * X[front] / Z[front] + self.cx
        v[front] = self.fy * Y[front] / Z[front] + self.cy

        inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2) & front
        cam_pts = np.column_stack([cam[inside], inten[inside]])   # (M,4) [camX,camY,camZ,intensity]

        if cam_pts.shape[0] < MIN_POINTS:
            self.get_logger().warn(
                f"Car found but only {cam_pts.shape[0]} LiDAR points inside the box - "
                "waiting for a clearer frame...", throttle_duration_sec=3.0)
            return

        box = np.array([x1, x2, y1, y2], dtype=np.float64)
        K = np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        np.savez(SAVE_PATH, cam_points=cam_pts, box=box, K=K)

        self.saved = True
        self.get_logger().info(f"SAVED a car frustum: {cam_pts.shape[0]} points -> {SAVE_PATH}")
        self.get_logger().info(f"  2D box [x1,x2,y1,y2] = [{x1:.0f}, {x2:.0f}, {y1:.0f}, {y2:.0f}]")
        self.get_logger().info(f"  intensity range = [{inten[inside].min():.3f}, {inten[inside].max():.3f}]")
        self.get_logger().info("Done. Ctrl+C this, then run:  python3 /root/decode_frustum.py")


def main():
    rclpy.init()
    node = FrustumSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
