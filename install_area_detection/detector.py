import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters

class InstallAreaDetection(Node):
    def __init__(self):
        super().__init__('install_area_detection_node')
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('depth_topic', '/depth')
        self.declare_parameter('rgb_topic', '/image_rect')
        self.declare_parameter('vis', True)
        self.declare_parameter('threshold.top', 1500)
        self.declare_parameter('threshold.bottom', 1100)
        
        self.declare_parameter('roi_coordinates.left_top', [545, 237, 640, 327])
        self.declare_parameter('roi_coordinates.right_top', [740, 237, 835, 327])
        self.declare_parameter('roi_coordinates.left_bottom', [550, 415, 635, 491])
        self.declare_parameter('roi_coordinates.right_bottom', [745, 415, 830, 491])

        # Get parameters
        depth_topic = self.get_parameter('depth_topic').value
        rgb_topic = self.get_parameter('rgb_topic').value
        vis = self.get_parameter('vis').value
        self.top_threshold = self.get_parameter('threshold.top').value
        self.bottom_threshold = self.get_parameter('threshold.bottom').value
        
        self.rois = {
            'left_top': self.get_parameter('roi_coordinates.left_top').value,
            'right_top': self.get_parameter('roi_coordinates.right_top').value,
            'left_bottom': self.get_parameter('roi_coordinates.left_bottom').value,
            'right_bottom': self.get_parameter('roi_coordinates.right_bottom').value,
        }

        # Create subscribers
        if vis:
            self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
            self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
            
            # Use ApproximateTimeSynchronizer for syncing depth and RGB
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.depth_sub, self.rgb_sub], 
                queue_size=10, 
                slop=0.1
            )
            self.ts.registerCallback(self.detect_with_vis)
            self.get_logger().info(f'Node Started (VIS MODE). Listening to {rgb_topic} and {depth_topic}')
        else:
            self.create_subscription(Image, depth_topic, self.detect_without_vis, 10)
            self.get_logger().info(f'Node Started (HEADLESS MODE). Listening to {depth_topic}')

    def _get_cv_depth(self, depth_msg: Image) -> np.ndarray:
        """
        Convert ROS Depth Image to CV2 format
        
        Arguments:
            depth_msg: sensor_msgs/Image
        
        Returns:
            cv_depth: numpy array of depth values in mm
        """
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            # Transform depth to mm if needed
            if depth_msg.encoding == '32FC1':
                cv_depth = cv_depth * 1000.0
            return cv_depth
        except Exception as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
            return None

    def _analyze_roi(self, 
                     cv_depth: np.ndarray[float], 
                     name: str, 
                     coords: list[int]) -> tuple[str, float, tuple, tuple]:
        """
        Analyze a single ROI and determine its state
        
        Arguments:
            cv_depth: numpy array of depth values in mm
            name: str, name of the ROI
            coords: list of 4 ints, [x1, y1, x2, y2]
        
        Returns:
            state: str, "INSTALLED", "EMPTY", or "Unknown"
            avg_depth: float, average depth in mm
            color: tuple, BGR color for visualization
            rect_coords: tuple, (x1, y1, x2, y2) for drawing            
        """
        x1, y1, x2, y2 = coords
        
        # Determine threshold based on ROI position
        threshold = self.top_threshold if 'top' in name else self.bottom_threshold
        
        # Avoid out-of-bounds (Safety check)
        h, w = cv_depth.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        roi_depth = cv_depth[y1:y2, x1:x2]
        valid_pixels = roi_depth[roi_depth > 0]
        
        # Determine state
        if valid_pixels.size == 0:
            avg_depth = 9999.0
            state = "Unknown"
            color = (0, 255, 255) # Yellow
        else:
            avg_depth = np.mean(valid_pixels)
            if avg_depth < threshold:
                state = "INSTALLED"
                color = (0, 255, 0) # Green
            else:
                state = "EMPTY"
                color = (0, 0, 255) # Red
                
        return state, avg_depth, color, (x1, y1, x2, y2)

    def detect_with_vis(self, depth_msg: Image, rgb_msg: Image):
        """
        Visualization Mode: Draw rectangles on RGB image
        """
        cv_depth = self._get_cv_depth(depth_msg)
        if cv_depth is None: return

        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge RGB Error: {e}')
            return
        
        for name, coords in self.rois.items():
            state, avg_depth, color, (x1, y1, x2, y2) = self._analyze_roi(cv_depth, name, coords)
            
            cv2.rectangle(cv_rgb, (x1, y1), (x2, y2), color, 2)
            
            label = f"{state} ({avg_depth:.0f}mm)"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(cv_rgb, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
            cv2.putText(cv_rgb, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        cv2.imshow("Detection Result", cv_rgb)
        cv2.waitKey(1)
        
    def detect_without_vis(self, depth_msg: Image):
        """

        """
        cv_depth = self._get_cv_depth(depth_msg)
        if cv_depth is None: return
        
        for name, coords in self.rois.items():
            state, avg_depth, _, _ = self._analyze_roi(cv_depth, name, coords)
            
            self.get_logger().info(f'ROI: {name}, State: {state}, Avg Depth: {avg_depth:.0f}mm')

def main(args=None):
    rclpy.init(args=args)
    node = InstallAreaDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
