from typing import Union, Tuple

import rclpy
import cv2
import numpy as np
import message_filters
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseArray, Pose
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
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
        camera_info_topic = self.get_parameter("camera_info_topic").value
        pose_topic = self.get_parameter("pose_topic").value

        self.vis = self.get_parameter("vis").value
        self.top_threshold = self.get_parameter("threshold.top").value
        self.bottom_threshold = self.get_parameter("threshold.bottom").value
        self.bracket_height = self.get_parameter("bracket.height").value
        self.bracket_width = self.get_parameter("bracket.width").value
        self.shrink_ratio = self.get_parameter("shrink_ratio").value

        # Initialize camera intrinsics
        self.camera_intrinsics = None

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
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )

        # Create TF listener and pose publisher
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_pub = self.create_publisher(PoseArray, pose_topic, 10)

        # Use ApproximateTimeSynchronizer for syncing depth and RGB
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.rgb_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.detect)
        self.get_logger().info(
            f"InstallAreaDetection Node Started. Listening to {rgb_topic} and {depth_topic}"
        )

    def camera_info_callback(self, msg: CameraInfo):
        """
        Callback function to store camera intrinsics
        """
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }
            self.get_logger().info(
                f"Camera Intrinsics Received: {self.camera_intrinsics}")

    def detect(self, depth_msg: Image, rgb_msg: Image):

        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg, desired_encoding="bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough")

            if depth_msg.encoding == "32FC1":
                cv_depth = cv_depth * 1000.0

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                "World", rgb_msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF Error: {e}", throttle_duration_sec=1.0)
            return

        corners = self._detect_apriltag(cv_rgb)

        if corners is None:
            return

        # Determine status and publish 3D position of uninstall area
        pose_array_msg = PoseArray()
        pose_array_msg.header.frame_id = "World"
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()

        tags_3d = self._get_tag_3d_positions(corners, cv_depth)
        rois_polygons_dict, roi_ratios = self._analyze_roi_and_ratios(corners)

        for key, roi in rois_polygons_dict.items():
            rx, ry = roi_ratios[key]
            theoretical_3d_pt = self._bilinear_interpolation(tags_3d, rx, ry)

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

                tx, ty, tz = theoretical_3d_pt
                p_cam = PointStamped()
                p_cam.header.frame_id = rgb_msg.header.frame_id
                p_cam.point.x = float(tx)
                p_cam.point.y = float(ty)
                p_cam.point.z = float(tz)

                p_world = do_transform_point(p_cam, transform)

                pose = Pose()
                pose.position = p_world.point
                pose.orientation.w = 1.0

                pose_array_msg.poses.append(pose)
            else:
                status = "Installed"

            if self.vis:
                color = [0, 0, 255] if status == "Unstalled" else [0, 255, 0]

                cv2.polylines(cv_rgb, [roi], True, color, 2, cv2.LINE_AA)

                text = f"{status}, {np.mean(roi_depth_values):.2f} mm"
                font_scale = 0.6
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX

                (text_w, text_h), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness)

                x, y = roi[0][0][0], roi[0][0][1]
                cv2.rectangle(cv_rgb, (x, y - text_h - 10),
                              (x + text_w, y), color, -1)
                cv2.putText(cv_rgb, text, (x, y - 5), font, font_scale,
                            (255, 255, 255), thickness, cv2.LINE_AA)

        self.target_pub.publish(pose_array_msg)

        cv2.imshow("Detection Result", cv_rgb)
        cv2.waitKey(1)

    def _get_tag_3d_positions(self, corners: dict, cv_depth: np.ndarray) -> Union[dict, None]:
        """
        Calculate 3D position of each tag

        Arguments:
            corners:
                left-top, right-top, right-bottom, left-bottom corne positions
                as a dict of numpy arrays with shape (2,)
            cv_depth: depth image

        Returns:
            tags_3d: dict of numpy arrays with shape (3,)
        """

        tags_3d = {}
        for key, uv in corners.items():
            u, v = int(uv[0]), int(uv[1])

            # Check if pixel is valid
            if u < 0 or u >= cv_depth.shape[1] or v < 0 or v >= cv_depth.shape[0]:
                return None

            d = cv_depth[v, u]
            if d <= 0:
                return None  # 無效深度

            # Convert pixel coordinates to camera coordinates
            tags_3d[key] = np.array(self._pixel_to_camera(u, v, d))

        return tags_3d

    def _bilinear_interpolation(self, tags_3d: dict, rx: float, ry: float) -> np.ndarray:
        """
        Bilinear interpolation of 3D position

        Arguments:
            tags_3d: dict of numpy arrays with shape (3,)
            rx: x ratio
            ry: y ratio

        Returns:
            final_pt: numpy array with shape (3,)
        """
        tl = tags_3d['tl']
        tr = tags_3d['tr']
        bl = tags_3d['bl']
        br = tags_3d['br']

        top_pt = tl * (1 - rx) + tr * rx
        bot_pt = bl * (1 - rx) + br * rx

        final_pt = top_pt * (1 - ry) + bot_pt * ry

        return final_pt

    def _pixel_to_camera(self, u, v, depth_mm) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to camera coordinates

        Arguments:
            u: x pixel coordinate
            v: y pixel coordinate
            depth_mm: depth in millimeters

        Returns:
            x: x coordinate in meters
            y: y coordinate in meters
            z: z coordinate in meters
        """
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        z_m = depth_mm / 1000.0

        x_m = (u - cx) * z_m / fx
        y_m = (v - cy) * z_m / fy

        return (x_m, y_m, z_m)

    def _declare_parameters(self):
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("rgb_topic", "/image_rect")
        self.declare_parameter("camera_info_topic", "/camera_info_rect")
        self.declare_parameter("pose_topic", "/install_area/target_pose")
        self.declare_parameter("vis", True)
        self.declare_parameter("threshold.top", 1500)
        self.declare_parameter("threshold.bottom", 1100)
        self.declare_parameter("bracket.height", 460)
        self.declare_parameter("bracket.width", 500)
        self.declare_parameter("shrink_ratio", 0.3)

    def _analyze_roi_and_ratios(self, corners_dict: dict[str, np.ndarray]) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compute projected ROI polygons and relative geometric ratios for 3D interpolation

        Arguments:
            corners_dict:
                dict of detected corner positions. Keys are 'tl', 'tr', 'bl', 'br'. Each value is a numpy array of shape (2,).
            shrink_ratio:
                ratio to shrink the ROI boxes. Larger value means smaller boxes. Default is 0.3.

        Returns:
            rois: list of ROI polygons as numpy arrays of shape (4, 1, 2)
            ratios: list of relative geometric ratios with shape (4, 2)
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
        ratios = {}

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

            # Calculate ratios
            cx_ideal = (bx1 + bx2) / 2.0
            cy_ideal = (by1 + by2) / 2.0

            rx = cx_ideal / W
            ry = cy_ideal / H

            ratios[key] = (rx, ry)

        return rois_polygons_dict, ratios

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
        if ids is None or len(corners) != 4:
            return

        ids = ids.flatten()

        id_map = {
            0: 'tl',
            1: 'tr',
            2: 'bl',
            3: 'br'
        }

        detected_corners = {}

        # Calculate center of each marker
        for i, tag_id in enumerate(ids):
            if tag_id in id_map:
                center = np.mean(corners[i][0], axis=0)
                position_name = id_map[tag_id]
                detected_corners[position_name] = center

        if self.vis:
            cv2.aruco.drawDetectedMarkers(cv_rgb, corners, ids)

        return detected_corners


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
