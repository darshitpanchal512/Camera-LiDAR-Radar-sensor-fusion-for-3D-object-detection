# One frame's worth of detected objects — an array of FrustumDetections
# tied together by a shared header (timestamp + frame_id).

std_msgs/Header header              # timestamp + frame_id (typically 'cam0')
FrustumDetection[] detections       # zero or more objects for this frame
