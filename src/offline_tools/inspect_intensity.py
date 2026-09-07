#!/usr/bin/env python3
"""
inspect_intensity.py
Reads ONE message from a PointCloud2 topic and prints the real
min / max / sample of the intensity field, plus the field layout.

Usage (inside container, with the bag playing in another terminal):
    python3 inspect_intensity.py /velodyne/points_raw
    python3 inspect_intensity.py /filtered_points
"""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class IntensityInspector(Node):
    def __init__(self, topic):
        super().__init__('intensity_inspector')
        self.topic = topic
        self.got_one = False

        # BEST_EFFORT matches most sensor drivers; if nothing arrives,
        # try flipping this to RELIABLE.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(
            PointCloud2, topic, self.cb, qos)
        self.get_logger().info(f'Listening on {topic} ... (waiting for one message)')

    def cb(self, msg):
        if self.got_one:
            return
        self.got_one = True

        # ---- 1. Print the field layout ----
        print('\n========== FIELD LAYOUT ==========')
        print(f'point_step = {msg.point_step} bytes')
        for f in msg.fields:
            # datatype codes: 1=int8 2=uint8 3=int16 4=uint16
            #                 5=int32 6=uint32 7=float32 8=float64
            dt_names = {1:'INT8',2:'UINT8',3:'INT16',4:'UINT16',
                        5:'INT32',6:'UINT32',7:'FLOAT32',8:'FLOAT64'}
            dt = dt_names.get(f.datatype, f'code{f.datatype}')
            print(f'  name={f.name:12s} offset={f.offset:2d} datatype={dt} count={f.count}')

        # ---- 2. Check intensity exists ----
        field_names = [f.name for f in msg.fields]
        if 'intensity' not in field_names:
            print('\n*** No "intensity" field on this topic! ***')
            print(f'Available fields: {field_names}')
            rclpy.shutdown()
            return

        # ---- 3. Read the real intensity numbers ----
        pts = point_cloud2.read_points(
            msg, field_names=['intensity'], skip_nans=True)
        intensity = np.array([p[0] for p in pts], dtype=np.float64)

        print('\n========== INTENSITY VALUES ==========')
        print(f'  count   = {intensity.size}')
        print(f'  min     = {intensity.min():.3f}')
        print(f'  max     = {intensity.max():.3f}')
        print(f'  mean    = {intensity.mean():.3f}')
        print(f'  median  = {np.median(intensity):.3f}')
        print(f'  first 20 values: {np.round(intensity[:20], 2).tolist()}')

        # ---- 4. Interpret ----
        print('\n========== INTERPRETATION ==========')
        mx = intensity.max()
        if mx <= 1.5:
            print('  Range looks like 0-1  -> already KITTI-normalised.')
        elif mx <= 255 + 1:
            print('  Range looks like 0-255 -> native VLP-16 scale. Divide by 255.')
        elif mx <= 2000:
            print('  Range tops out ~1898  -> something multiplied the native')
            print('  0-255 value. 1898 / 255 = 7.44, so a x7.44 scaling happened')
            print('  upstream. Either divide by ~1898 OR remove the upstream scaling.')
        else:
            print(f'  Unexpected max ({mx:.1f}). Investigate the driver/config.')

        rclpy.shutdown()


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 inspect_intensity.py <topic>')
        print('  e.g. python3 inspect_intensity.py /velodyne/points_raw')
        sys.exit(1)
    topic = sys.argv[1]
    rclpy.init()
    node = IntensityInspector(topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
