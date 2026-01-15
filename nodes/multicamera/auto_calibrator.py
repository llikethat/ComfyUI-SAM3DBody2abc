"""
Camera Auto-Calibrator Node

Automatically calibrates camera positions from synchronized footage.
Uses person keypoints visible in both cameras to solve relative pose.

No manual measurement required - just provide mesh sequences from both cameras.
"""

import numpy as np
import cv2
import os
import importlib.util
from typing import Dict, Tuple, Optional, List

# Get the directory containing this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_utils_dir = os.path.join(_current_dir, "utils")

# Function to load module from absolute path
def _load_util_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None

# Load utils modules
_camera_module = _load_util_module("camera", os.path.join(_utils_dir, "camera.py"))

if _camera_module:
    Camera = _camera_module.Camera
    compute_baseline = _camera_module.compute_baseline
    compute_angle_between_cameras = _camera_module.compute_angle_between_cameras
else:
    raise ImportError(f"Failed to load camera module from {_utils_dir}")

# Try to import logger
try:
    _lib_dir = os.path.dirname(_current_dir)
    _lib_dir = os.path.dirname(_lib_dir)
    _logger_module = _load_util_module("logger", os.path.join(_lib_dir, "lib", "logger.py"))
    if _logger_module:
        log = _logger_module.get_logger("CameraAutoCalibrator")
    else:
        raise ImportError()
except:
    class FallbackLogger:
        def info(self, msg): print(f"[Camera Auto-Calibrator] {msg}")
        def warning(self, msg): print(f"[Camera Auto-Calibrator] WARNING: {msg}")
        def error(self, msg): print(f"[Camera Auto-Calibrator] ERROR: {msg}")
        def debug(self, msg): pass
    log = FallbackLogger()


# Joint indices for calibration (using stable body joints)
CALIBRATION_JOINTS = {
    "head": 0,        # Nose
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_ankle": 15,
    "right_ankle": 16,
}

# For height estimation
HEAD_JOINT = 0       # Nose
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class CameraAutoCalibrator:
    """
    Automatically calibrate camera positions from synchronized footage.
    
    Uses person keypoints visible in both cameras to solve for relative
    camera positions. No manual measurement required.
    
    Requirements:
    - Same person visible in both camera views
    - Synchronized frames (same moment in time)
    - Known person height (for scale)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence_a": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from Camera A (will be reference/origin)"
                }),
                "mesh_sequence_b": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from Camera B"
                }),
            },
            "optional": {
                "person_height_m": ("FLOAT", {
                    "default": 1.75,
                    "min": 0.5,
                    "max": 2.5,
                    "step": 0.01,
                    "tooltip": "Height of person in meters (for scale calibration)"
                }),
                "focal_length_mm": ("FLOAT", {
                    "default": 35.0,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 1.0,
                    "tooltip": "Camera focal length in mm (assumed same for both)"
                }),
                "sensor_width_mm": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Camera sensor width in mm (36 = full frame)"
                }),
                "num_calibration_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of frames to use for calibration (averaged)"
                }),
                "min_joint_confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Minimum confidence for joint detection"
                }),
            }
        }
    
    RETURN_TYPES = ("CALIBRATION_DATA", "STRING")
    RETURN_NAMES = ("calibration", "calibration_info")
    FUNCTION = "calibrate"
    CATEGORY = "SAM3DBody2abc/MultiCamera"
    
    def calibrate(
        self,
        mesh_sequence_a: Dict,
        mesh_sequence_b: Dict,
        person_height_m: float = 1.75,
        focal_length_mm: float = 35.0,
        sensor_width_mm: float = 36.0,
        num_calibration_frames: int = 10,
        min_joint_confidence: float = 0.5,
    ) -> Tuple[Dict, str]:
        """
        Auto-calibrate camera positions from person keypoints.
        """
        
        log.info("=" * 60)
        log.info("CAMERA AUTO-CALIBRATION")
        log.info("=" * 60)
        log.info(f"Person height: {person_height_m}m")
        log.info(f"Focal length: {focal_length_mm}mm")
        log.info(f"Calibration frames: {num_calibration_frames}")
        
        # Get frames
        frames_a = mesh_sequence_a.get("frames", {})
        frames_b = mesh_sequence_b.get("frames", {})
        
        frame_indices_a = sorted(frames_a.keys())
        frame_indices_b = sorted(frames_b.keys())
        
        num_frames = min(len(frame_indices_a), len(frame_indices_b))
        
        if num_frames < 1:
            raise ValueError("No frames available for calibration")
        
        log.info(f"Available frames: {num_frames}")
        
        # Get image resolution from first frame
        first_frame_a = frames_a[frame_indices_a[0]]
        img_size = first_frame_a.get("image_size", [512, 512])
        if isinstance(img_size, (list, tuple)):
            img_w, img_h = img_size[0], img_size[1]
        else:
            img_w, img_h = 512, 512
        
        log.info(f"Image resolution: {img_w}x{img_h}")
        
        # Compute focal length in pixels
        focal_px = focal_length_mm * img_w / sensor_width_mm
        
        # Build camera intrinsic matrix
        K = np.array([
            [focal_px, 0, img_w / 2],
            [0, focal_px, img_h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        log.info(f"Focal length: {focal_px:.1f}px")
        
        # Collect keypoint correspondences from multiple frames
        all_points_a = []
        all_points_b = []
        
        # Sample frames evenly
        step = max(1, num_frames // num_calibration_frames)
        sample_indices = list(range(0, num_frames, step))[:num_calibration_frames]
        
        log.info(f"Sampling {len(sample_indices)} frames for calibration")
        
        for idx in sample_indices:
            frame_a = frames_a[frame_indices_a[idx]]
            frame_b = frames_b[frame_indices_b[idx]]
            
            # Get 2D keypoints
            kp_a = frame_a.get("pred_keypoints_2d")
            kp_b = frame_b.get("pred_keypoints_2d")
            
            if kp_a is None or kp_b is None:
                continue
            
            # Convert to numpy
            if hasattr(kp_a, 'numpy'):
                kp_a = kp_a.numpy()
            if hasattr(kp_b, 'numpy'):
                kp_b = kp_b.numpy()
            
            kp_a = np.array(kp_a)
            kp_b = np.array(kp_b)
            
            # Match corresponding joints
            for joint_name, joint_idx in CALIBRATION_JOINTS.items():
                if joint_idx >= len(kp_a) or joint_idx >= len(kp_b):
                    continue
                
                pt_a = kp_a[joint_idx]
                pt_b = kp_b[joint_idx]
                
                # Check bounds
                if pt_a[0] < 0 or pt_a[0] >= img_w or pt_a[1] < 0 or pt_a[1] >= img_h:
                    continue
                if pt_b[0] < 0 or pt_b[0] >= img_w or pt_b[1] < 0 or pt_b[1] >= img_h:
                    continue
                
                all_points_a.append(pt_a[:2])
                all_points_b.append(pt_b[:2])
        
        if len(all_points_a) < 8:
            raise ValueError(f"Not enough point correspondences ({len(all_points_a)}). Need at least 8.")
        
        log.info(f"Collected {len(all_points_a)} point correspondences")
        
        points_a = np.array(all_points_a, dtype=np.float64)
        points_b = np.array(all_points_b, dtype=np.float64)
        
        # Compute Essential Matrix using RANSAC
        E, mask = cv2.findEssentialMat(
            points_a, points_b, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            raise ValueError("Failed to compute Essential Matrix")
        
        inliers = mask.ravel().sum()
        log.info(f"Essential matrix inliers: {inliers}/{len(points_a)}")
        
        # Recover pose (rotation and translation)
        _, R, t, pose_mask = cv2.recoverPose(E, points_a, points_b, K)
        
        log.info(f"Recovered pose from {pose_mask.ravel().sum()} points")
        
        # t is unit vector, need to determine scale from person height
        # Triangulate head and ankle to get person height in arbitrary units
        
        # Get points for height estimation (use inlier frames)
        height_points_a = []
        height_points_b = []
        
        for idx in sample_indices[:5]:  # Use first few frames
            frame_a = frames_a[frame_indices_a[idx]]
            frame_b = frames_b[frame_indices_b[idx]]
            
            kp_a = frame_a.get("pred_keypoints_2d")
            kp_b = frame_b.get("pred_keypoints_2d")
            
            if kp_a is None or kp_b is None:
                continue
            
            if hasattr(kp_a, 'numpy'):
                kp_a = kp_a.numpy()
            if hasattr(kp_b, 'numpy'):
                kp_b = kp_b.numpy()
            
            kp_a = np.array(kp_a)
            kp_b = np.array(kp_b)
            
            # Get head and ankle points
            if HEAD_JOINT < len(kp_a) and LEFT_ANKLE < len(kp_a):
                height_points_a.append({
                    "head": kp_a[HEAD_JOINT][:2],
                    "ankle": kp_a[LEFT_ANKLE][:2]
                })
                height_points_b.append({
                    "head": kp_b[HEAD_JOINT][:2],
                    "ankle": kp_b[LEFT_ANKLE][:2]
                })
        
        # Collect pelvis points for viewing angle computation
        pelvis_points_a = []
        pelvis_points_b = []
        PELVIS_JOINT = 11  # Left hip as pelvis proxy
        
        for idx in sample_indices:
            frame_a = frames_a[frame_indices_a[idx]]
            frame_b = frames_b[frame_indices_b[idx]]
            
            kp_a = frame_a.get("pred_keypoints_2d")
            kp_b = frame_b.get("pred_keypoints_2d")
            
            if kp_a is None or kp_b is None:
                continue
            
            if hasattr(kp_a, 'numpy'):
                kp_a = kp_a.numpy()
            if hasattr(kp_b, 'numpy'):
                kp_b = kp_b.numpy()
            
            kp_a = np.array(kp_a)
            kp_b = np.array(kp_b)
            
            if PELVIS_JOINT < len(kp_a) and PELVIS_JOINT < len(kp_b):
                pelvis_points_a.append(kp_a[PELVIS_JOINT][:2])
                pelvis_points_b.append(kp_b[PELVIS_JOINT][:2])
        
        # Triangulate to get scale
        scale = self._estimate_scale(
            height_points_a, height_points_b,
            K, R, t, person_height_m
        )
        
        log.info(f"Estimated scale factor: {scale:.4f}")
        
        # Apply scale to translation
        t_scaled = t.flatten() * scale
        
        # Triangulate person center for viewing angle computation
        person_3d = self._triangulate_person_center(
            pelvis_points_a, pelvis_points_b,
            K, R, t, scale
        )
        
        log.info(f"Person 3D position: [{person_3d[0]:.2f}, {person_3d[1]:.2f}, {person_3d[2]:.2f}]")
        
        # Compute viewing angles
        camera_a_pos = np.array([0.0, 0.0, 0.0])
        camera_b_pos = t_scaled
        
        viewing_angles = self._compute_viewing_angles(
            person_3d, camera_a_pos, camera_b_pos, R
        )
        
        log.info(f"Camera A viewing angle: {viewing_angles['camera_a']['viewing_angle_deg']:.1f}°")
        log.info(f"Camera B viewing angle: {viewing_angles['camera_b']['viewing_angle_deg']:.1f}°")
        log.info(f"Camera A distance: {viewing_angles['camera_a']['distance_m']:.2f}m")
        log.info(f"Camera B distance: {viewing_angles['camera_b']['distance_m']:.2f}m")
        
        # Convert rotation matrix to Euler angles
        rotation_euler = self._rotation_matrix_to_euler(R)
        
        log.info(f"Camera B position: [{t_scaled[0]:.3f}, {t_scaled[1]:.3f}, {t_scaled[2]:.3f}]")
        log.info(f"Camera B rotation: [{rotation_euler[0]:.1f}°, {rotation_euler[1]:.1f}°, {rotation_euler[2]:.1f}°]")
        
        # Build calibration data
        calibration = {
            "version": "1.0",
            "name": "Auto-Calibration",
            "source": "auto",
            "method": "essential_matrix",
            "coordinate_system": {
                "up": "Y",
                "forward": "-Z",
                "unit": "meters"
            },
            "cameras": {
                "camera_A": {
                    "name": "Camera A (Reference)",
                    "position": [0.0, 0.0, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                    "rotation_order": "XYZ",
                    "focal_length_mm": focal_length_mm,
                    "sensor_width_mm": sensor_width_mm,
                    "sensor_height_mm": sensor_width_mm * img_h / img_w,
                    "resolution": [img_w, img_h],
                    "principal_point": [img_w / 2, img_h / 2],
                    "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0}
                },
                "camera_B": {
                    "name": "Camera B",
                    "position": t_scaled.tolist(),
                    "rotation_euler": rotation_euler,
                    "rotation_order": "XYZ",
                    "focal_length_mm": focal_length_mm,
                    "sensor_width_mm": sensor_width_mm,
                    "sensor_height_mm": sensor_width_mm * img_h / img_w,
                    "resolution": [img_w, img_h],
                    "principal_point": [img_w / 2, img_h / 2],
                    "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0}
                }
            },
            "calibration_info": {
                "num_correspondences": len(all_points_a),
                "num_inliers": int(inliers),
                "num_frames_used": len(sample_indices),
                "person_height_m": person_height_m,
                "scale_factor": scale,
            }
        }
        
        # Create Camera objects
        camera_a = Camera.from_dict(calibration["cameras"]["camera_A"], "camera_A")
        camera_b = Camera.from_dict(calibration["cameras"]["camera_B"], "camera_B")
        
        calibration["camera_objects"] = {
            "camera_A": camera_a,
            "camera_B": camera_b
        }
        
        # Compute geometry
        baseline = compute_baseline(camera_a, camera_b)
        angle = compute_angle_between_cameras(camera_a, camera_b)
        
        calibration["geometry"] = {
            "baseline_m": baseline,
            "angle_between_views_deg": angle
        }
        
        # Add viewing angles
        calibration["viewing_angles"] = viewing_angles
        
        # Generate info string
        info_lines = [
            "=== AUTO-CALIBRATION RESULTS ===",
            f"Method: Essential Matrix + Person Height",
            f"Point correspondences: {len(all_points_a)}",
            f"Inliers: {inliers}",
            f"Frames used: {len(sample_indices)}",
            "",
            "=== CAMERA A (Reference) ===",
            f"Position: (0.0, 0.0, 0.0) m",
            f"Rotation: (0.0°, 0.0°, 0.0°)",
            f"Distance to person: {viewing_angles['camera_a']['distance_m']:.2f}m",
            f"Viewing angle to person: {viewing_angles['camera_a']['viewing_angle_deg']:.1f}°",
            f"  Azimuth (horizontal): {viewing_angles['camera_a']['azimuth_deg']:.1f}°",
            f"  Elevation (vertical): {viewing_angles['camera_a']['elevation_deg']:.1f}°",
            "",
            "=== CAMERA B ===",
            f"Position: ({t_scaled[0]:.3f}, {t_scaled[1]:.3f}, {t_scaled[2]:.3f}) m",
            f"Rotation: ({rotation_euler[0]:.1f}°, {rotation_euler[1]:.1f}°, {rotation_euler[2]:.1f}°)",
            f"Distance to person: {viewing_angles['camera_b']['distance_m']:.2f}m",
            f"Viewing angle to person: {viewing_angles['camera_b']['viewing_angle_deg']:.1f}°",
            f"  Azimuth (horizontal): {viewing_angles['camera_b']['azimuth_deg']:.1f}°",
            f"  Elevation (vertical): {viewing_angles['camera_b']['elevation_deg']:.1f}°",
            "",
            "=== PERSON POSITION ===",
            f"3D Position: ({person_3d[0]:.2f}, {person_3d[1]:.2f}, {person_3d[2]:.2f}) m",
            "",
            "=== GEOMETRY ===",
            f"Baseline: {baseline:.2f}m",
            f"Angle between views: {angle:.1f}°",
            f"Person height used: {person_height_m}m",
        ]
        
        if angle < 30:
            info_lines.append("⚠️ Warning: Small angle - depth accuracy may be limited")
        elif angle > 150:
            info_lines.append("⚠️ Warning: Cameras nearly opposite - occlusion issues possible")
        else:
            info_lines.append("✓ Good camera geometry for triangulation")
        
        info = "\n".join(info_lines)
        
        log.info(f"Baseline: {baseline:.2f}m, Angle: {angle:.1f}°")
        log.info("=" * 60)
        
        return (calibration, info)
    
    def _estimate_scale(
        self,
        height_points_a: List[Dict],
        height_points_b: List[Dict],
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        person_height_m: float
    ) -> float:
        """
        Estimate scale factor using person height.
        
        Triangulates head and ankle joints to compute height in arbitrary units,
        then scales to match known person height.
        """
        if len(height_points_a) == 0:
            log.warning("No height points available, using default scale")
            return 5.0  # Default assumption: camera ~5m from subject
        
        # Build projection matrices
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        
        heights = []
        
        for pts_a, pts_b in zip(height_points_a, height_points_b):
            try:
                # Triangulate head
                head_a = pts_a["head"].reshape(2, 1)
                head_b = pts_b["head"].reshape(2, 1)
                head_4d = cv2.triangulatePoints(P1, P2, head_a, head_b)
                head_3d = (head_4d[:3] / head_4d[3]).flatten()
                
                # Triangulate ankle
                ankle_a = pts_a["ankle"].reshape(2, 1)
                ankle_b = pts_b["ankle"].reshape(2, 1)
                ankle_4d = cv2.triangulatePoints(P1, P2, ankle_a, ankle_b)
                ankle_3d = (ankle_4d[:3] / ankle_4d[3]).flatten()
                
                # Compute height (vertical distance)
                height = np.abs(head_3d[1] - ankle_3d[1])  # Y is up
                
                if height > 0.1:  # Sanity check
                    heights.append(height)
            except:
                continue
        
        if len(heights) == 0:
            log.warning("Could not triangulate height, using default scale")
            return 5.0
        
        # Average height in arbitrary units
        avg_height = np.median(heights)
        
        # Scale factor to convert to meters
        scale = person_height_m / avg_height
        
        log.info(f"Triangulated height: {avg_height:.4f} units")
        log.info(f"Target height: {person_height_m}m")
        
        return scale
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> List[float]:
        """
        Convert rotation matrix to Euler angles (XYZ order, degrees).
        """
        # Extract Euler angles (XYZ order)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return [np.degrees(x), np.degrees(y), np.degrees(z)]
    
    def _compute_viewing_angles(
        self,
        person_3d: np.ndarray,
        camera_a_pos: np.ndarray,
        camera_b_pos: np.ndarray,
        R: np.ndarray
    ) -> Dict:
        """
        Compute the viewing angle from each camera to the person.
        
        Returns angles in degrees:
        - camera_a_to_person: Angle from Camera A's forward direction to person
        - camera_b_to_person: Angle from Camera B's forward direction to person
        - person_facing_a: Approximate angle person is facing relative to Camera A
        - person_facing_b: Approximate angle person is facing relative to Camera B
        
        Angles:
        - 0° = looking directly at camera (frontal view)
        - 90° = side view (profile)
        - 180° = back view
        """
        # Camera A is at origin, looking down -Z axis
        cam_a_forward = np.array([0, 0, -1])
        
        # Camera B forward direction (rotated)
        cam_b_forward = R @ np.array([0, 0, -1])
        
        # Vector from Camera A to person
        vec_a_to_person = person_3d - camera_a_pos
        vec_a_to_person_norm = vec_a_to_person / (np.linalg.norm(vec_a_to_person) + 1e-8)
        
        # Vector from Camera B to person
        vec_b_to_person = person_3d - camera_b_pos
        vec_b_to_person_norm = vec_b_to_person / (np.linalg.norm(vec_b_to_person) + 1e-8)
        
        # Angle from Camera A forward to person direction
        # This tells us how much the camera needs to "look" towards the person
        dot_a = np.clip(np.dot(cam_a_forward, vec_a_to_person_norm), -1, 1)
        angle_a_to_person = np.degrees(np.arccos(dot_a))
        
        # Angle from Camera B forward to person direction
        dot_b = np.clip(np.dot(cam_b_forward, vec_b_to_person_norm), -1, 1)
        angle_b_to_person = np.degrees(np.arccos(dot_b))
        
        # Compute horizontal viewing angle (azimuth) for each camera
        # This is the angle in the XZ plane (ignoring vertical component)
        
        # Camera A horizontal angle to person
        vec_a_xz = np.array([vec_a_to_person[0], 0, vec_a_to_person[2]])
        vec_a_xz_norm = vec_a_xz / (np.linalg.norm(vec_a_xz) + 1e-8)
        forward_xz = np.array([0, 0, -1])
        
        # Use atan2 for signed angle
        cross_a = np.cross(forward_xz, vec_a_xz_norm)
        dot_a_xz = np.dot(forward_xz, vec_a_xz_norm)
        azimuth_a = np.degrees(np.arctan2(cross_a[1], dot_a_xz))  # Y component of cross gives sign
        
        # Camera B horizontal angle to person (in Camera B's frame)
        vec_b_local = R.T @ vec_b_to_person  # Transform to Camera B's local frame
        vec_b_xz = np.array([vec_b_local[0], 0, vec_b_local[2]])
        vec_b_xz_norm = vec_b_xz / (np.linalg.norm(vec_b_xz) + 1e-8)
        
        cross_b = np.cross(forward_xz, vec_b_xz_norm)
        dot_b_xz = np.dot(forward_xz, vec_b_xz_norm)
        azimuth_b = np.degrees(np.arctan2(cross_b[1], dot_b_xz))
        
        # Vertical angle (elevation) - how much above/below camera center
        elevation_a = np.degrees(np.arcsin(np.clip(vec_a_to_person_norm[1], -1, 1)))
        elevation_b = np.degrees(np.arcsin(np.clip(vec_b_to_person_norm[1], -1, 1)))
        
        # Distance from each camera to person
        distance_a = np.linalg.norm(vec_a_to_person)
        distance_b = np.linalg.norm(vec_b_to_person)
        
        return {
            "camera_a": {
                "viewing_angle_deg": angle_a_to_person,
                "azimuth_deg": azimuth_a,  # Horizontal angle (+ = person to right)
                "elevation_deg": elevation_a,  # Vertical angle (+ = person above)
                "distance_m": distance_a,
            },
            "camera_b": {
                "viewing_angle_deg": angle_b_to_person,
                "azimuth_deg": azimuth_b,
                "elevation_deg": elevation_b,
                "distance_m": distance_b,
            },
            "person_position_3d": person_3d.tolist(),
        }
    
    def _triangulate_person_center(
        self,
        pelvis_points_a: List[np.ndarray],
        pelvis_points_b: List[np.ndarray],
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """
        Triangulate the person's center (pelvis) position in 3D.
        """
        if len(pelvis_points_a) == 0:
            return np.array([0, 0, -5])  # Default: 5m in front (Maya convention: -Z is forward)
        
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        
        positions = []
        
        for pt_a, pt_b in zip(pelvis_points_a, pelvis_points_b):
            try:
                pt_a_2d = pt_a.reshape(2, 1)
                pt_b_2d = pt_b.reshape(2, 1)
                
                point_4d = cv2.triangulatePoints(P1, P2, pt_a_2d, pt_b_2d)
                point_3d = (point_4d[:3] / point_4d[3]).flatten()
                
                # Apply scale
                point_3d = point_3d * scale
                
                positions.append(point_3d)
            except:
                continue
        
        if len(positions) == 0:
            return np.array([0, 0, -5])  # Default: 5m in front (Maya convention)
        
        result = np.median(positions, axis=0)
        
        # Convert from OpenCV convention (Z-forward positive) to 
        # Maya/Blender convention (Z-forward negative, camera looks down -Z)
        # This ensures viewing angle calculations work correctly
        result[2] = -result[2]
        
        return result


# Node registration
NODE_CLASS_MAPPINGS = {
    "CameraAutoCalibrator": CameraAutoCalibrator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraAutoCalibrator": "📷 Camera Auto-Calibrator",
}
