import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class GroundFilter(Node):
    def __init__(self):
        super().__init__('ground_filter')
       
        self.subscription = self.create_subscription(
            PointCloud2, 
            '/velodyne/points_raw', 
            self.listener_callback, 
            10)
        

        self.publisher = self.create_publisher(PointCloud2, '/filtered_points', 10)
        
       
        self.z_threshold = -1.5 
        

    def listener_callback(self, msg):
        
        points = pc2.read_points(msg, skip_nans=True)
        
        filtered_points = [p for p in points if p[2] > self.z_threshold]
        
        filtered_msg = pc2.create_cloud(msg.header, msg.fields, filtered_points)
        
        self.publisher.publish(filtered_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GroundFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()