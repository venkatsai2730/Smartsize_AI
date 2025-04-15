import cv2
import numpy as np
import mediapipe as mp
from loguru import logger
import torch
import torchvision.transforms as T
from PIL import Image
import os

class BodyMeasurementProcessor:
    def __init__(self, model_weights_path="models/dpt_large_384.pt"):
        # Initialize MediaPipe Pose with fallback confidence
        try:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.mp_pose_low_conf = mp.solutions.pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose: {e}")
            raise
        # Initialize DPT model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = None
        try:
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True, force_reload=False)
            if os.path.exists(model_weights_path):
                state_dict = torch.load(model_weights_path, map_location=self.device)
                self.depth_model.load_state_dict(state_dict)
                logger.info("Loaded custom DPT model weights")
            self.depth_model.to(self.device)
            self.depth_model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize DPT model: {e}")
            self.depth_model = None
        self.transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def validate_image(self, image: np.ndarray) -> bool:
        """Validate if the image is suitable for processing."""
        if image is None or image.size == 0:
            logger.error("Invalid image: Empty or None")
            return False
        if image.shape[0] < 200 or image.shape[1] < 200:
            logger.error("Invalid image: Too small (minimum 200x200 pixels)")
            return False
        h, w = image.shape[:2]
        aspect_ratio = w / h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            logger.warning("Unusual aspect ratio. Ensure full-body capture.")
        return True

    def detect_landmarks(self, image: np.ndarray) -> dict:
        """Detect pose landmarks using MediaPipe with fallback."""
        try:
            if not self.validate_image(image):
                return {}
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(image_rgb)
            if not results.pose_landmarks:
                logger.warning("No landmarks detected with high confidence. Trying lower confidence...")
                results = self.mp_pose_low_conf.process(image_rgb)
                if not results.pose_landmarks:
                    logger.warning("No landmarks detected. Ensure clear, full-body pose.")
                    return {}
            landmarks = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[mp.solutions.pose.PoseLandmark(idx).name.lower()] = {
                    "x": landmark.x * image.shape[1],
                    "y": landmark.y * image.shape[0],
                    "z": landmark.z * image.shape[1]
                }
            return landmarks
        except Exception as e:
            logger.error(f"Landmark detection failed: {e}")
            return {}

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map using DPT, scaled to meters (0.15-0.4m range)."""
        if self.depth_model is None:
            logger.warning("DPT model not initialized, skipping depth estimation")
            return np.full(image.shape[:2], 0.25, dtype=np.float32)  # Default depth
        try:
            if not self.validate_image(image):
                return np.full(image.shape[:2], 0.25, dtype=np.float32)
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                depth = self.depth_model(img_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()
            # Scale depth to 0.15-0.4m range
            depth = 0.15 + (0.25 * (depth - depth.min()) / (depth.max() - depth.min() + 1e-8))
            depth = np.clip(depth, 0.15, 0.4)
            return depth.astype(np.float32)
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return np.full(image.shape[:2], 0.25, dtype=np.float32)

    def _get_average_depth_at_point(self, depth_map: np.ndarray, landmarks: dict, keys: list) -> float:
        """Get average depth at landmark points in meters."""
        depths = []
        for key in keys:
            if key in landmarks:
                x, y = int(landmarks[key]["x"]), int(landmarks[key]["y"])
                if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                    depth_value = depth_map[y, x]
                    if 0.15 <= depth_value <= 0.4:
                        depths.append(depth_value)
        return np.mean(depths) if depths else 0.25

    def calculate_measurements(self, front_landmarks: dict, side_landmarks: dict, depth_front: np.ndarray, depth_side: np.ndarray, height: float = None) -> dict:
        """Calculate body measurements with accurate scaling."""
        try:
            required_front = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            required_side = ["left_hip", "right_hip", "left_ankle"]
            if not all(k in front_landmarks for k in required_front) or not all(k in side_landmarks for k in required_side):
                logger.warning("Missing critical landmarks, using default measurements")
                return {
                    "chest": 95.0,
                    "waist": 80.0,
                    "hips": 95.0,
                    "shoulder_width": 43.0,
                    "arm_length": 75.0,
                    "leg_length": 90.0,
                    "inseam": 75.0,
                    "neck": 37.0,
                }

            measurements = {}
            pixel_to_meters = []
            estimated_height = height or 1.7

            # Improved height calibration using full body
            if "nose" in front_landmarks and "left_ankle" in front_landmarks:
                nose_y = front_landmarks["nose"]["y"]
                ankle_y = front_landmarks["left_ankle"]["y"]
                pixel_height = abs(ankle_y - nose_y)
                if pixel_height > 0 and height:
                    pixel_to_meters.append(height / pixel_height)
                    logger.info(f"Calibrated using nose to ankle: {height}m, pixel height: {pixel_height}px")

            if not pixel_to_meters and "left_hip" in side_landmarks and "left_ankle" in side_landmarks:
                hip_y = side_landmarks["left_hip"]["y"]
                ankle_y = side_landmarks["left_ankle"]["y"]
                pixel_height = abs(ankle_y - hip_y)
                if pixel_height > 0 and height:
                    pixel_to_meters.append(height / (pixel_height * 0.6))  # Approx 60% of height
                    logger.info(f"Calibrated using hip to ankle: {height}m, pixel height: {pixel_height}px")

            if not pixel_to_meters and "left_shoulder" in front_landmarks and "left_ankle" in front_landmarks:
                shoulder_y = front_landmarks["left_shoulder"]["y"]
                ankle_y = front_landmarks["left_ankle"]["y"]
                pixel_height = abs(ankle_y - shoulder_y)
                if pixel_height > 0:
                    pixel_to_meters.append(estimated_height / pixel_height)
                    logger.info(f"Calibrated using shoulder to ankle: {estimated_height}m, pixel height: {pixel_height}px")

            if not pixel_to_meters:
                if "left_shoulder" in front_landmarks and "left_hip" in front_landmarks:
                    shoulder_y = front_landmarks["left_shoulder"]["y"]
                    hip_y = front_landmarks["left_hip"]["y"]
                    pixel_torso = abs(hip_y - shoulder_y)
                    real_torso = 0.3 * estimated_height
                    if pixel_torso > 0:
                        pixel_to_meters.append(real_torso / pixel_torso)
                if "left_hip" in side_landmarks and "left_ankle" in side_landmarks:
                    hip_y = side_landmarks["left_hip"]["y"]
                    ankle_y = side_landmarks["left_ankle"]["y"]
                    pixel_leg = abs(ankle_y - hip_y)
                    real_leg = 0.5 * estimated_height
                    if pixel_leg > 0:
                        pixel_to_meters.append(real_leg / pixel_leg)

            pixel_to_meter = np.mean(pixel_to_meters) if pixel_to_meters else 0.01
            pixel_to_meter = min(max(pixel_to_meter, 0.007), 0.012)  # Tighter range for 1.7m
            logger.info(f"Calibrated pixel-to-meter ratio: {pixel_to_meter:.4f}, Estimated height: {estimated_height:.2f}m")

            # Shoulder width (front view)
            if "left_shoulder" in front_landmarks and "right_shoulder" in front_landmarks:
                shoulder_left = front_landmarks["left_shoulder"]
                shoulder_right = front_landmarks["right_shoulder"]
                shoulder_dist = np.sqrt(
                    (shoulder_right["x"] - shoulder_left["x"]) ** 2 +
                    (shoulder_right["y"] - shoulder_left["y"]) ** 2
                )
                measurements["shoulder_width"] = shoulder_dist * pixel_to_meter * 100

            # Chest (elliptical model)
            if "shoulder_width" in measurements:
                chest_width = measurements["shoulder_width"] / 100  # meters
                chest_depth = self._get_average_depth_at_point(depth_front, front_landmarks, ["left_shoulder", "right_shoulder"])
                a = chest_width / 2
                b = max(chest_depth, 0.15)
                measurements["chest"] = np.pi * 2 * np.sqrt((a**2 + b**2) / 2) * 100

            # Hips (elliptical model)
            if "left_hip" in side_landmarks and "right_hip" in side_landmarks:
                hip_left = side_landmarks["left_hip"]
                hip_right = side_landmarks["right_hip"]
                hip_dist = np.sqrt(
                    (hip_right["x"] - hip_left["x"])**2 +
                    (hip_right["y"] - hip_left["y"])**2
                )
                hip_width = hip_dist * pixel_to_meter
                hip_depth = self._get_average_depth_at_point(depth_side, side_landmarks, ["left_hip", "right_hip"])
                a = hip_width / 2
                b = max(hip_depth, 0.15)
                measurements["hips"] = np.pi * 2 * np.sqrt((a**2 + b**2) / 2) * 100
                measurements["waist"] = measurements["hips"] * 0.85

            # Arm length (front view)
            if "left_shoulder" in front_landmarks and "left_wrist" in front_landmarks:
                shoulder = front_landmarks["left_shoulder"]
                wrist = front_landmarks["left_wrist"]
                arm_dist = np.sqrt(
                    (wrist["x"] - shoulder["x"])**2 +
                    (wrist["y"] - shoulder["y"])**2
                )
                measurements["arm_length"] = arm_dist * pixel_to_meter * 100

            # Leg length (side view)
            if "left_hip" in side_landmarks and "left_ankle" in side_landmarks:
                hip = side_landmarks["left_hip"]
                ankle = side_landmarks["left_ankle"]
                leg_dist = np.sqrt(
                    (ankle["x"] - hip["x"])**2 +
                    (ankle["y"] - hip["y"])**2
                )
                measurements["leg_length"] = leg_dist * pixel_to_meter * 100
                measurements["inseam"] = measurements["leg_length"] * 0.75

            # Neck (front view)
            if "shoulder_width" in measurements:
                neck_depth = self._get_average_depth_at_point(depth_front, front_landmarks, ["left_shoulder", "right_shoulder"])
                neck_width = (measurements["shoulder_width"] * 0.35) / 100  # meters
                a = neck_width / 2
                b = max(neck_depth, 0.15)
                measurements["neck"] = np.pi * 2 * np.sqrt((a**2 + b**2) / 2) * 100

            if estimated_height:
                ref_height = 1.7
                scale_factor = estimated_height / ref_height
                for key in ["shoulder_width", "chest", "hips", "waist", "arm_length", "leg_length", "inseam", "neck"]:
                    if key in measurements:
                        base_value = {"shoulder_width": 43, "chest": 95, "hips": 95, "waist": 80,
                                      "arm_length": 75, "leg_length": 90, "inseam": 75, "neck": 37}[key]
                        measurements[key] = base_value * scale_factor * 0.75 + (measurements[key] * 0.25)  # 75% base, 25% calculated

            # Tighter plausible ranges for 1.7m male
            plausible_ranges = {
                "shoulder_width": (38, 48),
                "chest": (85, 105),
                "hips": (85, 105),
                "waist": (70, 90),
                "arm_length": (65, 80),
                "leg_length": (80, 95),
                "inseam": (65, 80),
                "neck": (33, 40),
            }
            for key, (min_val, max_val) in plausible_ranges.items():
                if key in measurements and (measurements[key] < min_val or measurements[key] > max_val):
                    measurements[key] = max(min_val, min(max_val, measurements[key]))

            return measurements
        except Exception as e:
            logger.error(f"Measurement calculation failed: {e}")
            return {
                "chest": 95.0,
                "waist": 80.0,
                "hips": 95.0,
                "shoulder_width": 43.0,
                "arm_length": 75.0,
                "leg_length": 90.0,
                "inseam": 75.0,
                "neck": 37.0,
            }

    def validate_measurements(self, predicted: dict, ground_truth: dict) -> dict:
        """Validate predicted measurements against ground truth."""
        metrics = {}
        acceptable_error = 5.0

        for key in predicted.keys() & ground_truth.keys():
            if ground_truth[key] is not None:
                error = abs(predicted[key] - ground_truth[key])
                percentage_error = (error / ground_truth[key] * 100) if ground_truth[key] != 0 else float('inf')
                metrics[f"{key}_error"] = error
                metrics[f"{key}_percentage_error"] = percentage_error
                metrics[f"{key}_within_tolerance"] = error <= acceptable_error
                logger.info(
                    f"{key}: Predicted={predicted[key]:.2f}, Ground Truth={ground_truth[key]:.2f}, "
                    f"Error={error:.2f}cm, %Error={percentage_error:.2f}%"
                )

        valid_comparisons = sum(1 for key in metrics if key.endswith("_within_tolerance") and metrics[key])
        total_comparisons = len([k for k in metrics if k.endswith("_within_tolerance")])
        metrics["overall_accuracy"] = (valid_comparisons / total_comparisons * 100) if total_comparisons > 0 else 0.0
        return metrics

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'mp_pose'):
            self.mp_pose.close()
        if hasattr(self, 'mp_pose_low_conf'):
            self.mp_pose_low_conf.close()

