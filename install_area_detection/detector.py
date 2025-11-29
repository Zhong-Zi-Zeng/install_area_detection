from typing import Union

import rclpy
import cv2
import numpy as np
import message_filters
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class InstallAreaDetection(Node):
    def __init__(self):
        super().__init__("install_area_detection_node")
        self.bridge = CvBridge()

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        depth_topic = self.get_parameter("depth_topic").value
        rgb_topic = self.get_parameter("rgb_topic").value
        self.vis = self.get_parameter("vis").value
        self.top_threshold = self.get_parameter("threshold.top").value
        self.bottom_threshold = self.get_parameter("threshold.bottom").value
        self.bracket_height = self.get_parameter("bracket.height").value
        self.bracket_width = self.get_parameter("bracket.width").value
        self.shrink_ratio = self.get_parameter("shrink_ratio").value

        # April Tag Detector Initialization
        self.april_detection_dict = cv2.aruco.Dictionary_get(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        self.april_detection_parameters = cv2.aruco.DetectorParameters_create()
        self.april_detection_parameters.minMarkerPerimeterRate = 0.01
        self.april_detection_parameters.cornerRefinementWinSize = 5
        self.april_detection_parameters.cornerRefinementMaxIterations = 30
        self.april_detection_parameters.cornerRefinementMinAccuracy = 0.1
        self.april_detection_parameters.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )

        # Create subscribers
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)

        # Use ApproximateTimeSynchronizer for syncing depth and RGB
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.rgb_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.detect)
        self.get_logger().info(
            f"InstallAreaDetection Node Started. Listening to {rgb_topic} and {depth_topic}"
        )

    def _declare_parameters(self):
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("rgb_topic", "/image_rect")
        self.declare_parameter("vis", True)
        self.declare_parameter("threshold.top", 1500)
        self.declare_parameter("threshold.bottom", 1100)
        self.declare_parameter("bracket.height", 460)
        self.declare_parameter("bracket.width", 500)
        self.declare_parameter("shrink_ratio", 0.3)

    def _get_cv_depth(self, depth_msg: Image) -> np.ndarray:
        """
        Convert ROS Depth Image to CV2 format

        Arguments:
            depth_msg: sensor_msgs/Image

        Returns:
            cv_depth: numpy array of depth values in mm
        """
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            # Transform depth to mm if needed
            if depth_msg.encoding == "32FC1":
                cv_depth = cv_depth * 1000.0
            return cv_depth
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return None

    def _analyze_roi(self, corners_dict: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Analyze ROIs based on detected corners and return polygons for each ROI

        Arguments:
            corners_dict:
                dict of detected corner positions. Keys are 'tl', 'tr', 'bl', 'br'. Each value is a numpy array of shape (2,).
            shrink_ratio:
                ratio to shrink the ROI boxes. Larger value means smaller boxes. Default is 0.3.

        Returns:
            rois: list of ROI polygons as numpy arrays of shape (4, 1, 2)
        """
        # Define source points for perspective transform
        W, H = self.bracket_width, self.bracket_height
        src_pts = np.array(
            [
                [0, 0],  # top-left
                [W, 0],  # top-right
                [0, H],  # bottom-left
                [W, H],  # bottom-right
            ],
            dtype=np.float32,
        )

        # Define destination points from detected corners
        dst_pts = np.array(
            [
                corners_dict["tl"],
                corners_dict["tr"],
                corners_dict["bl"],
                corners_dict["br"],
            ],
            dtype=np.float32,
        )

        # Calculate perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Calculate padded box dimensions
        half_w, half_h = W // 2, H // 2
        pad_w = int(half_w * self.shrink_ratio)
        pad_h = int(half_h * self.shrink_ratio)

        # Define ideal boxes with padding
        ideal_boxes_dict = {
            "top-left": [0 + pad_w, 0 + pad_h, half_w - pad_w, half_h - pad_h],
            "top-right": [half_w + pad_w, 0 + pad_h, W - pad_w, half_h - pad_h],
            "buttom-left": [0 + pad_w, half_h + pad_h, half_w - pad_w, H - pad_h],
            "buttom-right": [half_w + pad_w, half_h + pad_h, W - pad_w, H - pad_h],
        }

        # Project ideal boxes to image plane
        rois_polygons_dict = {}

        for key, box in ideal_boxes_dict.items():
            bx1, by1, bx2, by2 = box

            box_pts = np.array(
                [
                    [[bx1, by1]],  # top-left
                    [[bx2, by1]],  # top-right
                    [[bx2, by2]],  # bottom-right
                    [[bx1, by2]],  # bottom-left
                ],
                dtype=np.float32,
            )

            projected_pts = cv2.perspectiveTransform(box_pts, M)
            rois_polygons_dict[key] = projected_pts.astype(int)

        return rois_polygons_dict

    def detect(self, depth_msg: Image, rgb_msg: Image):
        # Get RGB Image
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge RGB Error: {e}")
            return

        # Get Depth Image
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            # Transform depth to mm if needed
            if depth_msg.encoding == "32FC1":
                cv_depth = cv_depth * 1000.0
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        corners = self._detect_apriltag(cv_rgb)

        if corners is None:
            return

        # Check depth in each ROI and determine status
        rois_polygons_dict = self._analyze_roi(corners)

        # Determine status and publish 3D position of uninstall area
        uninstall_position = []

        for key, roi in rois_polygons_dict.items():
            mask = np.zeros(cv_depth.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [roi], color=1)

            roi_depth_values = cv_depth[mask == 1]
            roi_depth_values = roi_depth_values[roi_depth_values > 0]
            
            # If no depth value in ROI, skip
            if roi_depth_values.size == 0:
                continue
                
            threshold = self.top_threshold if 'top' in key else self.bottom_threshold
            
            if np.mean(roi_depth_values) > threshold:
                status = "Unstalled"
                color = [0, 0, 255]
            else:
                status = "Installed" 
                color = [0, 255, 0]
            
            if self.vis:
                cv2.polylines(cv_rgb, [roi], True, color, 2, cv2.LINE_AA)

                text = f"{status}, {np.mean(roi_depth_values):.2f} mm"
                font_scale = 0.6
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX

                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                x, y = roi[0][0][0], roi[0][0][1]
                cv2.rectangle(cv_rgb, (x, y - text_h - 10), (x + text_w, y), color, -1)
                cv2.putText(cv_rgb, text, (x, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)           

        cv2.imshow("Detection Result", cv_rgb)
        cv2.waitKey(1)

    def _detect_apriltag(self, cv_rgb: np.ndarray) -> Union[dict[np.ndarray], None]:
        """
        Detect AprilTags in the RGB image

        Arguments:
            cv_rgb: numpy array of the RGB image

        Returns:
            corners:
                left-top, right-top, right-bottom, left-bottom corne positions
                as a dict of numpy arrays with shape (2,)
        """
        gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.april_detection_dict, parameters=self.april_detection_parameters
        )

        # If exactly 4 markers are detected
        if len(corners) != 4:
            return

        # Resort corners based on marker positions. left-top, right-top, right-bottom, left-bottom
        centers = []
        for c in corners:
            centers.append(np.mean(c[0], axis=0))  # (x, y)

        pts = np.array(centers)

        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        if self.vis:
            cv2.aruco.drawDetectedMarkers(cv_rgb, corners, ids)

        return {"tl": tl, "tr": tr, "bl": bl, "br": br}


def main(args=None):
    rclpy.init(args=args)
    node = InstallAreaDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
