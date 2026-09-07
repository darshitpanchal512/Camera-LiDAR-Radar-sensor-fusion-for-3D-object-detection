# One detected object — its class, its 2D box on the image, its point cloud.
# All fields for a single object travel together as one atomic unit.

string class_name                   # "car", "person", ...
float32 confidence                  # detector confidence for the 2D detection
float32[] box_2d                    # [x1, y1, x2, y2] in image pixels
sensor_msgs/PointCloud2 points      # the frustum points (XYZI, in camera frame)

