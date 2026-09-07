# fusion_msgs

Custom ROS 2 message definitions used to carry one frame's worth of detected-object frustums from the extractor to the inference node — each with its label, its confidence, its image box, and its own point cloud, all in one atomic delivery.

Full lifecycle and design rationale: [`docs/04_car_inference.md#the-custom-messages`](../../docs/04_car_inference.md#the-custom-messages).

## The two messages

- **`FrustumDetection.msg`** — one detected object.
- **`FrustumDetectionArray.msg`** — one frame's worth of detections, with a shared header.

## Design

Composed of standard ROS messages (`sensor_msgs/PointCloud2`, `std_msgs/Header`) plus the per-object fields we needed (`class_name`, `confidence`, `box_2d`). We did not reinvent point clouds — we wrapped them.

## Build

Standard ROS 2 message package. Depends on `std_msgs` and `sensor_msgs`. Generate with `colcon build --packages-select fusion_msgs`, then:

```python
from fusion_msgs.msg import FrustumDetection, FrustumDetectionArray
```
