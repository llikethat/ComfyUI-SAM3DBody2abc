"""
Multi-Camera Triangulator Node

Triangulates 3D positions from 2 camera views.
Produces jitter-free depth through geometric calculation.
"""

import numpy as np
import torch
import copy
import os
import sys
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

# Load utils modules using absolute paths
_camera_module = _load_util_module("camera", os.path.join(_utils_dir, "camera.py"))
_triangulation_module = _load_util_module("triangulation", os.path.join(_utils_dir, "triangulation.py"))
_visualization_module = _load_util_module("visualization", os.path.join(_utils_dir, "visualization.py"))

if _camera_module:
    Camera = _camera_module.Camera
    convert_coordinate_system = _camera_module.convert_coordinate_system
else:
    raise ImportError(f"Failed to load camera module from {_utils_dir}")

if _triangulation_module:
    triangulate_point_from_cameras = _triangulation_module.triangulate_point_from_cameras
    compute_reprojection_error = _triangulation_module.compute_reprojection_error
    estimate_triangulation_quality = _triangulation_module.estimate_triangulation_quality
else:
    raise ImportError(f"Failed to load triangulation module from {_utils_dir}")

if _visualization_module:
    create_multicamera_debug_view = _visualization_module.create_multicamera_debug_view
    create_topview_with_cameras = _visualization_module.create_topview_with_cameras
    create_error_graph = _visualization_module.create_error_graph
else:
    raise ImportError(f"Failed to load visualization module from {_utils_dir}")

# Try to import logger
try:
    _lib_dir = os.path.dirname(_current_dir)
    _lib_dir = os.path.dirname(_lib_dir)  # Go up to main package
    _logger_module = _load_util_module("logger", os.path.join(_lib_dir, "lib", "logger.py"))
    if _logger_module:
        log = _logger_module.get_logger("MultiCameraTriangulator")
    else:
        raise ImportError()
except:
    class FallbackLogger:
        def info(self, msg): print(f"[MultiCamera Triangulator] {msg}")
        def warning(self, msg): print(f"[MultiCamera Triangulator] WARNING: {msg}")
        def error(self, msg): print(f"[MultiCamera Triangulator] ERROR: {msg}")
        def debug(self, msg): pass
        def progress(self, c, t, task="", interval=10):
            if c == 0 or c == t - 1 or (c + 1) % interval == 0:
                print(f"[MultiCamera Triangulator] {task}: {c + 1}/{t}")
    log = FallbackLogger()


# Joint indices for different tracking modes (SMPL-X 127-joint skeleton)
JOINT_INDICES = {
    "Pelvis": 11,  # Left hip as pelvis proxy (center between 11 and 12)
    "Head": 15,
    "Left Ankle": 7,
    "Right Ankle": 8,
    "Left Wrist": 20,
    "Right Wrist": 21,
    "Left Knee": 4,
    "Right Knee": 5,
}

# For 17-joint keypoints_2d from SAM3DBody (pred_keypoints_2d)
# Based on COCO-like skeleton
SIMPLE_JOINT_INDICES = {
    "Pelvis": 11,  # Left hip as pelvis proxy
    "Head": 0,     # Nose
    "Left Ankle": 15,
    "Right Ankle": 16,
    "Left Wrist": 9,
    "Right Wrist": 10,
    "Left Knee": 13,
    "Right Knee": 14,
}


class MultiCameraTriangulator:
    """
    Triangulate 3D positions from 2 camera views.
    
    Produces jitter-free depth through geometric calculation,
    eliminating the noise inherent in monocular depth estimation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_list": ("CAMERA_LIST", {
                    "tooltip": "Camera list from Camera Accumulator (minimum 2 cameras)"
                }),
                "calibration": ("CALIBRATION_DATA", {
                    "tooltip": "Camera calibration data from Auto-Calibrator or CalibrationLoader"
                }),
            },
            "optional": {
                "joint_to_track": (["Pelvis", "Head", "Left Ankle", "Right Ankle", "All Joints"], {
                    "default": "Pelvis",
                    "tooltip": "Which joint(s) to triangulate for trajectory"
                }),
                "merge_mode": (["Z only (depth)", "Full XYZ"], {
                    "default": "Z only (depth)",
                    "tooltip": "Z only: Replace depth in Camera A sequence. Full XYZ: Use triangulated position."
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Minimum confidence for joint detection"
                }),
                "error_threshold_m": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Error threshold in meters (for quality warnings)"
                }),
                "debug_visualization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate debug visualization images"
                }),
            }
        }
    
    RETURN_TYPES = ("TRAJECTORY_3D", "MESH_SEQUENCE", "IMAGE", "STRING")
    RETURN_NAMES = ("trajectory_3d", "merged_sequence", "debug_view", "debug_info")
    FUNCTION = "triangulate"
    CATEGORY = "SAM3DBody2abc/MultiCamera"
    
    def triangulate(
        self,
        camera_list: Dict,
        calibration: Dict,
        joint_to_track: str = "Pelvis",
        merge_mode: str = "Z only (depth)",
        confidence_threshold: float = 0.5,
        error_threshold_m: float = 0.05,
        debug_visualization: bool = True,
    ) -> Tuple[Dict, Dict, torch.Tensor, str]:
        """
        Triangulate 3D positions from N camera views.
        
        Accepts CAMERA_LIST from Camera Accumulator.
        Uses all available cameras for each joint at each frame.
        """
        
        # Validate inputs
        num_cameras = camera_list.get("num_cameras", 0)
        cameras_data = camera_list.get("cameras", [])
        
        if num_cameras < 2:
            raise ValueError(
                f"Need at least 2 cameras for triangulation, got {num_cameras}. "
                f"Chain Camera Accumulator nodes to add more cameras."
            )
        
        log.info("=" * 60)
        log.info(f"MULTI-CAMERA TRIANGULATION ({num_cameras} cameras)")
        log.info("=" * 60)
        
        # Get calibrated Camera objects (ordered list)
        cal_camera_list = calibration.get("camera_list", [])
        
        # Fallback: build from camera_objects dict if camera_list not present
        if not cal_camera_list:
            cam_objs = calibration.get("camera_objects", {})
            for key in sorted(cam_objs.keys()):
                cal_camera_list.append(cam_objs[key])
        
        if len(cal_camera_list) < 2:
            raise ValueError("Calibration data has fewer than 2 cameras")
        
        # Match camera_list entries to calibrated cameras
        # Use as many as we have calibration for
        num_usable = min(num_cameras, len(cal_camera_list))
        
        if num_usable < num_cameras:
            log.warning(
                f"Camera list has {num_cameras} cameras but calibration has "
                f"{len(cal_camera_list)}. Using {num_usable} cameras."
            )
        
        for i in range(num_usable):
            cam = cal_camera_list[i]
            label = cameras_data[i]["label"] if i < len(cameras_data) else f"cam_{i}"
            log.info(f"  {label}: position={cam.position}, calibrated=✓")
        
        # Get mesh sequences and images from camera_list
        mesh_sequences = [c["mesh_sequence"] for c in cameras_data[:num_usable]]
        all_images = [c.get("images") for c in cameras_data[:num_usable]]
        
        # Get frame counts
        all_frames = [ms.get("frames", {}) for ms in mesh_sequences]
        frame_counts = [len(f) for f in all_frames]
        
        num_frames = min(frame_counts)
        if len(set(frame_counts)) > 1:
            log.warning(f"Frame count mismatch: {frame_counts}, using min={num_frames}")
        
        log.info(f"Processing {num_frames} frames across {num_usable} cameras")
        
        fps = mesh_sequences[0].get("fps", 24.0)
        
        # Backward compat aliases for debug visualization
        camera_a = cal_camera_list[0]
        camera_b = cal_camera_list[1]
        
        images_a = all_images[0] if len(all_images) > 0 else None
        images_b = all_images[1] if len(all_images) > 1 else None
        
        # Determine which joints to track
        if joint_to_track == "All Joints":
            joints_to_track = list(JOINT_INDICES.keys())
        else:
            joints_to_track = [joint_to_track]
        
        log.info(f"Tracking joints: {joints_to_track}")
        
        # Initialize trajectory output
        trajectory_3d = {
            "frames": num_frames,
            "fps": fps,
            "num_cameras": num_usable,
            "coordinate_system": "Y-up",
            "unit": "meters",
            "primary_camera": {
                "index": 0,
                "id": cameras[0]["id"],
                "label": cameras[0]["label"],
                "note": "Use this camera's video for Maya image plane overlay"
            },
            "joints": {},
            "trajectory": {
                "positions": [],
                "errors": [],
                "visibility": [],
            },
            "quality": {
                "average_error": 0.0,
                "max_error": 0.0,
                "frames_with_all_cameras": 0,
                "frames_with_partial_cameras": 0,
                "frames_with_both_cameras": 0,
                "frames_with_single_camera": 0,
                "frames_failed": 0,
            }
        }
        
        # Initialize joint tracking
        for joint_name in joints_to_track:
            trajectory_3d["joints"][joint_name.lower()] = {
                "positions": [],
                "errors": [],
                "visibility": [],
            }
        
        # Get sorted frame indices for each camera
        all_frame_indices = [sorted(f.keys()) for f in all_frames]
        
        # Determine keypoint source from first camera's first frame
        first_frame = all_frames[0][all_frame_indices[0][0]]
        use_simple_skeleton = "pred_keypoints_2d" in first_frame and first_frame["pred_keypoints_2d"] is not None
        
        if use_simple_skeleton:
            joint_map = SIMPLE_JOINT_INDICES
            kp_key = "pred_keypoints_2d"
            log.info("Using pred_keypoints_2d (18-joint)")
        else:
            joint_map = JOINT_INDICES
            kp_key = "joint_coords"
            log.info("Using joint_coords (127-joint)")
        
        # Track statistics
        total_error = 0.0
        max_error = 0.0
        valid_frames = 0
        
        # Process each frame
        for i in range(num_frames):
            log.progress(i, num_frames, "Triangulating", interval=20)
            
            # Get frame data from all cameras
            frame_data = []
            for cam_idx in range(num_usable):
                idx = all_frame_indices[cam_idx][i]
                frame_data.append(all_frames[cam_idx][idx])
            
            # Process each joint
            for joint_name in joints_to_track:
                joint_key = joint_name.lower()
                joint_idx = joint_map.get(joint_name, 0)
                
                # Get 2D coordinates from ALL cameras
                all_coords = []
                for cam_idx in range(num_usable):
                    coords = self._get_joint_2d(
                        frame_data[cam_idx], joint_idx, kp_key, 
                        cal_camera_list[cam_idx]
                    )
                    all_coords.append(coords)
                
                # Triangulate using all available cameras
                point_3d, error, num_cams = triangulate_point_from_cameras(
                    cal_camera_list[:num_usable],
                    all_coords,
                    confidences=None
                )
                
                # Store results
                if point_3d is not None:
                    trajectory_3d["joints"][joint_key]["positions"].append(point_3d.tolist())
                    trajectory_3d["joints"][joint_key]["errors"].append(error)
                    trajectory_3d["joints"][joint_key]["visibility"].append(num_cams)
                    
                    if error != float('inf'):
                        total_error += error
                        max_error = max(max_error, error)
                        valid_frames += 1
                    
                    if num_cams == num_usable:
                        trajectory_3d["quality"]["frames_with_all_cameras"] += 1
                    elif num_cams >= 2:
                        trajectory_3d["quality"]["frames_with_partial_cameras"] += 1
                    
                    # Keep backward compat stats
                    if num_cams >= 2:
                        trajectory_3d["quality"]["frames_with_both_cameras"] += 1
                    elif num_cams == 1:
                        trajectory_3d["quality"]["frames_with_single_camera"] += 1
                    else:
                        trajectory_3d["quality"]["frames_failed"] += 1
                else:
                    trajectory_3d["joints"][joint_key]["positions"].append(None)
                    trajectory_3d["joints"][joint_key]["errors"].append(float('inf'))
                    trajectory_3d["joints"][joint_key]["visibility"].append(0)
                    trajectory_3d["quality"]["frames_failed"] += 1
        
        # Calculate quality metrics
        if valid_frames > 0:
            trajectory_3d["quality"]["average_error"] = total_error / valid_frames
        trajectory_3d["quality"]["max_error"] = max_error
        
        # Set primary trajectory
        primary_joint = joints_to_track[0].lower()
        trajectory_3d["trajectory"]["positions"] = trajectory_3d["joints"][primary_joint]["positions"]
        trajectory_3d["trajectory"]["errors"] = trajectory_3d["joints"][primary_joint]["errors"]
        trajectory_3d["trajectory"]["visibility"] = trajectory_3d["joints"][primary_joint]["visibility"]
        
        # Calculate trajectory statistics
        positions = [p for p in trajectory_3d["trajectory"]["positions"] if p is not None]
        if len(positions) >= 2:
            positions_arr = np.array(positions)
            displacement = positions_arr[-1] - positions_arr[0]
            
            distances = np.sqrt(np.sum(np.diff(positions_arr, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            
            duration = num_frames / fps
            avg_speed = total_distance / duration if duration > 0 else 0
            
            trajectory_3d["trajectory"]["displacement"] = displacement.tolist()
            trajectory_3d["trajectory"]["total_distance"] = float(total_distance)
            trajectory_3d["trajectory"]["average_speed"] = float(avg_speed)
        
        log.info(f"Triangulation complete!")
        log.info(f"  Cameras used: {num_usable}")
        log.info(f"  Valid frames: {valid_frames}/{num_frames}")
        log.info(f"  Average error: {trajectory_3d['quality']['average_error']:.4f}m")
        log.info(f"  Max error: {trajectory_3d['quality']['max_error']:.4f}m")
        if num_usable > 2:
            log.info(f"  Frames with all {num_usable} cameras: {trajectory_3d['quality']['frames_with_all_cameras']}")
            log.info(f"  Frames with partial cameras: {trajectory_3d['quality']['frames_with_partial_cameras']}")
        
        # Merge triangulated data back into first camera's mesh_sequence
        merged_sequence = self._merge_trajectory(
            mesh_sequences[0], trajectory_3d, merge_mode, primary_joint
        )
        
        # Update merge metadata for N cameras
        merged_sequence["triangulation"]["method"] = f"{num_usable}_camera"
        
        # Generate debug visualization (uses first two cameras for backward compat)
        if debug_visualization and images_a is not None and images_b is not None:
            debug_view = self._create_debug_view(
                images_a, images_b, camera_a, camera_b,
                trajectory_3d, primary_joint
            )
        else:
            debug_view = self._create_simple_debug_view(
                camera_a, camera_b, trajectory_3d, primary_joint
            )
        
        # Generate info string
        debug_info = self._generate_info_string(
            trajectory_3d, calibration, error_threshold_m
        )
        
        log.info("=" * 60)
        
        return (trajectory_3d, merged_sequence, debug_view, debug_info)
    
    def _get_joint_2d(
        self,
        frame: Dict,
        joint_idx: int,
        kp_key: str,
        camera: Camera
    ) -> Optional[Tuple[float, float]]:
        """Get 2D joint coordinates from frame data."""
        
        if kp_key == "pred_keypoints_2d":
            kp2d = frame.get("pred_keypoints_2d")
            if kp2d is None:
                return None
            
            if hasattr(kp2d, 'numpy'):
                kp2d = kp2d.numpy()
            kp2d = np.array(kp2d)
            
            if joint_idx >= len(kp2d):
                return None
            
            return (float(kp2d[joint_idx, 0]), float(kp2d[joint_idx, 1]))
        
        else:
            # joint_coords is 3D - need to project to 2D
            joint_coords = frame.get("joint_coords")
            if joint_coords is None:
                return None
            
            if hasattr(joint_coords, 'numpy'):
                joint_coords = joint_coords.numpy()
            joint_coords = np.array(joint_coords)
            
            if joint_idx >= len(joint_coords):
                return None
            
            cam_t = frame.get("pred_cam_t")
            if cam_t is None:
                return None
            
            if hasattr(cam_t, 'numpy'):
                cam_t = cam_t.numpy()
            cam_t = np.array(cam_t)
            
            joint_3d = joint_coords[joint_idx] + np.array([cam_t[0], cam_t[1], 0])
            
            focal = frame.get("focal_length", 500)
            img_size = frame.get("image_size", [512, 512])
            
            tz = cam_t[2]
            if abs(tz) < 0.001:
                return None
            
            x = focal * joint_3d[0] / tz + img_size[0] / 2
            y = focal * joint_3d[1] / tz + img_size[1] / 2
            
            return (float(x), float(y))
    
    def _merge_trajectory(
        self,
        mesh_sequence: Dict,
        trajectory_3d: Dict,
        merge_mode: str,
        joint_name: str
    ) -> Dict:
        """Merge triangulated trajectory back into mesh sequence."""
        
        merged = copy.deepcopy(mesh_sequence)
        
        frames = merged.get("frames", {})
        positions = trajectory_3d["joints"].get(joint_name, {}).get("positions", [])
        
        frame_indices = sorted(frames.keys())
        
        for i, idx in enumerate(frame_indices):
            if i >= len(positions) or positions[i] is None:
                continue
            
            pos = positions[i]
            
            if merge_mode == "Z only (depth)":
                frames[idx]["tracked_depth"] = pos[2]
                frames[idx]["triangulated_depth"] = pos[2]
            else:
                frames[idx]["tracked_position_3d"] = pos
                frames[idx]["tracked_depth"] = pos[2]
                frames[idx]["triangulated_position"] = pos
        
        merged["triangulation"] = {
            "method": "two_camera",
            "merge_mode": merge_mode,
            "tracked_joint": joint_name,
            "quality": trajectory_3d["quality"]
        }
        
        log.info(f"Merged triangulated data ({merge_mode})")
        
        return merged
    
    def _create_debug_view(
        self,
        images_a: torch.Tensor,
        images_b: torch.Tensor,
        camera_a: Camera,
        camera_b: Camera,
        trajectory_3d: Dict,
        joint_name: str
    ) -> torch.Tensor:
        """Create debug visualization with camera views."""
        
        if isinstance(images_a, torch.Tensor):
            images_a_np = images_a.cpu().numpy()
        else:
            images_a_np = np.array(images_a)
        
        if isinstance(images_b, torch.Tensor):
            images_b_np = images_b.cpu().numpy()
        else:
            images_b_np = np.array(images_b)
        
        mid_frame = len(trajectory_3d["trajectory"]["positions"]) // 2
        
        debug_img = create_multicamera_debug_view(
            images_a_np, images_b_np,
            camera_a, camera_b,
            trajectory_3d,
            frame_idx=mid_frame,
            joint_name=joint_name,
            output_size=(1024, 768)
        )
        
        if debug_img.dtype == np.uint8:
            debug_img = debug_img.astype(np.float32) / 255.0
        
        debug_tensor = torch.from_numpy(debug_img).unsqueeze(0)
        
        return debug_tensor
    
    def _create_simple_debug_view(
        self,
        camera_a: Camera,
        camera_b: Camera,
        trajectory_3d: Dict,
        joint_name: str
    ) -> torch.Tensor:
        """Create simple debug view without camera frames."""
        
        mid_frame = len(trajectory_3d["trajectory"]["positions"]) // 2
        
        topview = create_topview_with_cameras(
            camera_a, camera_b, trajectory_3d,
            current_frame=mid_frame,
            joint_name=joint_name,
            size=(512, 512)
        )
        
        if topview.dtype == np.uint8:
            topview = topview.astype(np.float32) / 255.0
        
        return torch.from_numpy(topview).unsqueeze(0)
    
    def _generate_info_string(
        self,
        trajectory_3d: Dict,
        calibration: Dict,
        error_threshold: float
    ) -> str:
        """Generate info string for display."""
        
        quality = trajectory_3d["quality"]
        traj = trajectory_3d.get("trajectory", {})
        primary_cam = trajectory_3d.get("primary_camera", {})
        
        lines = [
            f"=== TRIANGULATION RESULTS ({trajectory_3d.get('num_cameras', 2)} cameras) ===",
            f"Frames: {trajectory_3d['frames']}",
            f"FPS: {trajectory_3d['fps']}",
            "",
            "=== PRIMARY CAMERA (for Maya image plane) ===",
            f"Camera: {primary_cam.get('label', 'Camera A')} ({primary_cam.get('id', 'cam_0')})",
            f"Index: {primary_cam.get('index', 0)}",
            "",
            "=== QUALITY ===",
            f"Average error: {quality['average_error']:.4f}m",
            f"Max error: {quality['max_error']:.4f}m",
            f"Frames with 2+ cameras: {quality['frames_with_both_cameras']}",
            f"Frames with single camera: {quality['frames_with_single_camera']}",
            f"Frames failed: {quality['frames_failed']}",
        ]
        
        # Add N-camera specific stats if present
        if quality.get('frames_with_all_cameras', 0) > 0:
            lines.append(f"Frames with ALL cameras: {quality['frames_with_all_cameras']}")
            lines.append(f"Frames with partial cameras: {quality.get('frames_with_partial_cameras', 0)}")
        
        if quality['average_error'] <= error_threshold:
            lines.append(f"✓ Quality: GOOD (error < {error_threshold}m)")
        elif quality['average_error'] <= error_threshold * 2:
            lines.append(f"⚠ Quality: MODERATE (error < {error_threshold * 2}m)")
        else:
            lines.append(f"✗ Quality: POOR (error > {error_threshold * 2}m)")
        
        if "displacement" in traj:
            lines.extend([
                "",
                "=== TRAJECTORY ===",
                f"Displacement: X={traj['displacement'][0]:.3f}m, Y={traj['displacement'][1]:.3f}m, Z={traj['displacement'][2]:.3f}m",
                f"Total distance: {traj.get('total_distance', 0):.3f}m",
                f"Average speed: {traj.get('average_speed', 0):.3f} m/s",
            ])
        
        lines.extend([
            "",
            "=== CAMERA GEOMETRY ===",
            f"Baseline: {calibration['geometry']['baseline_m']:.2f}m",
            f"Angle between views: {calibration['geometry']['angle_between_views_deg']:.1f}°",
        ])
        
        return "\n".join(lines)


# Node registration
NODE_CLASS_MAPPINGS = {
    "MultiCameraTriangulator": MultiCameraTriangulator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiCameraTriangulator": "🔺 Multi-Camera Triangulator",
}
