"""
Visualization utilities for multi-camera triangulation debugging.

Generates:
- Multi-view comparison images
- Top-down trajectory views with camera positions
- Error visualization graphs
"""

import numpy as np
import cv2
import os
import importlib.util
from typing import List, Dict, Optional, Tuple

# Get the directory containing this file
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load module from absolute path
def _load_util_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None

# Load camera module
_camera_module = _load_util_module("camera", os.path.join(_current_dir, "camera.py"))
if _camera_module:
    Camera = _camera_module.Camera
else:
    raise ImportError(f"Failed to load camera module from {_current_dir}")


def create_multicamera_debug_view(
    images_a: np.ndarray,
    images_b: np.ndarray,
    camera_a: Camera,
    camera_b: Camera,
    trajectory_3d: Dict,
    frame_idx: int = 0,
    joint_name: str = "pelvis",
    output_size: Tuple[int, int] = (1024, 768)
) -> np.ndarray:
    """
    Create a multi-camera debug visualization.
    
    Layout:
    ┌─────────────┬─────────────┐
    │  Camera A   │  Camera B   │
    │  (frame)    │  (frame)    │
    ├─────────────┴─────────────┤
    │       Top View (X-Z)      │
    │   with camera positions   │
    └───────────────────────────┘
    
    Args:
        images_a: Video frames from camera A [N, H, W, 3]
        images_b: Video frames from camera B [N, H, W, 3]
        camera_a: Camera A object
        camera_b: Camera B object
        trajectory_3d: Triangulated trajectory data
        frame_idx: Current frame to display
        joint_name: Joint being tracked
        output_size: Output image size (width, height)
    
    Returns:
        Debug visualization image [H, W, 3] uint8
    """
    out_w, out_h = output_size
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # Dark background
    
    # Layout dimensions
    cam_view_h = out_h // 2
    cam_view_w = out_w // 2
    top_view_h = out_h // 2
    top_view_w = out_w
    
    # Get frames
    if frame_idx < len(images_a):
        frame_a = images_a[frame_idx].copy()
        if frame_a.dtype == np.float32 or frame_a.dtype == np.float64:
            frame_a = (frame_a * 255).astype(np.uint8)
    else:
        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    
    if frame_idx < len(images_b):
        frame_b = images_b[frame_idx].copy()
        if frame_b.dtype == np.float32 or frame_b.dtype == np.float64:
            frame_b = (frame_b * 255).astype(np.uint8)
    else:
        frame_b = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize frames to fit
    frame_a_resized = cv2.resize(frame_a, (cam_view_w - 10, cam_view_h - 40))
    frame_b_resized = cv2.resize(frame_b, (cam_view_w - 10, cam_view_h - 40))
    
    # Place camera views
    canvas[30:30 + frame_a_resized.shape[0], 5:5 + frame_a_resized.shape[1]] = frame_a_resized
    canvas[30:30 + frame_b_resized.shape[0], cam_view_w + 5:cam_view_w + 5 + frame_b_resized.shape[1]] = frame_b_resized
    
    # Draw labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Camera A: {camera_a.name}", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f"Camera B: {camera_b.name}", (cam_view_w + 10, 20), font, 0.5, (255, 255, 255), 1)
    
    # Draw triangulated point on each view
    if trajectory_3d and "joints" in trajectory_3d:
        joint_data = trajectory_3d["joints"].get(joint_name, {})
        positions = joint_data.get("positions", [])
        
        if frame_idx < len(positions) and positions[frame_idx] is not None:
            point_3d = np.array(positions[frame_idx])
            
            # Project to camera A
            px_a, py_a, visible_a = camera_a.project_point(point_3d)
            if visible_a:
                # Scale to resized frame
                scale_x_a = frame_a_resized.shape[1] / frame_a.shape[1]
                scale_y_a = frame_a_resized.shape[0] / frame_a.shape[0]
                draw_x_a = int(5 + px_a * scale_x_a)
                draw_y_a = int(30 + py_a * scale_y_a)
                cv2.circle(canvas, (draw_x_a, draw_y_a), 8, (0, 255, 0), 2)
                cv2.circle(canvas, (draw_x_a, draw_y_a), 3, (0, 255, 0), -1)
            
            # Project to camera B
            px_b, py_b, visible_b = camera_b.project_point(point_3d)
            if visible_b:
                scale_x_b = frame_b_resized.shape[1] / frame_b.shape[1]
                scale_y_b = frame_b_resized.shape[0] / frame_b.shape[0]
                draw_x_b = int(cam_view_w + 5 + px_b * scale_x_b)
                draw_y_b = int(30 + py_b * scale_y_b)
                cv2.circle(canvas, (draw_x_b, draw_y_b), 8, (0, 255, 0), 2)
                cv2.circle(canvas, (draw_x_b, draw_y_b), 3, (0, 255, 0), -1)
    
    # Draw top view
    top_view = create_topview_with_cameras(
        camera_a, camera_b, trajectory_3d, frame_idx, joint_name,
        size=(top_view_w - 20, top_view_h - 20)
    )
    canvas[cam_view_h + 10:cam_view_h + 10 + top_view.shape[0], 10:10 + top_view.shape[1]] = top_view
    
    # Frame info
    num_frames = trajectory_3d.get("frames", 0) if trajectory_3d else 0
    error = 0.0
    if trajectory_3d and "joints" in trajectory_3d:
        errors = trajectory_3d["joints"].get(joint_name, {}).get("errors", [])
        if frame_idx < len(errors):
            error = errors[frame_idx]
    
    info_text = f"Frame: {frame_idx + 1}/{num_frames}  |  Error: {error:.3f}m  |  Joint: {joint_name}"
    cv2.putText(canvas, info_text, (10, out_h - 10), font, 0.5, (200, 200, 200), 1)
    
    return canvas


def create_topview_with_cameras(
    camera_a: Camera,
    camera_b: Camera,
    trajectory_3d: Optional[Dict],
    current_frame: int = 0,
    joint_name: str = "pelvis",
    size: Tuple[int, int] = (512, 384)
) -> np.ndarray:
    """
    Create top-down view showing camera positions and trajectory.
    
    Args:
        camera_a: First camera
        camera_b: Second camera
        trajectory_3d: Triangulated trajectory data
        current_frame: Current frame index (highlighted)
        joint_name: Joint to visualize
        size: Output size (width, height)
    
    Returns:
        Top view image [H, W, 3] uint8
    """
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)
    
    # Collect all points to determine bounds
    points = [camera_a.position, camera_b.position]
    
    if trajectory_3d and "joints" in trajectory_3d:
        positions = trajectory_3d["joints"].get(joint_name, {}).get("positions", [])
        for pos in positions:
            if pos is not None:
                points.append(np.array(pos))
    
    if len(points) < 2:
        cv2.putText(canvas, "No data", (width // 3, height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        return canvas
    
    points = np.array(points)
    
    # Get X-Z bounds (top view)
    x_coords = points[:, 0]
    z_coords = points[:, 2]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # Add padding
    padding = 0.15
    x_range = max(x_max - x_min, 1.0)
    z_range = max(z_max - z_min, 1.0)
    
    x_min -= x_range * padding
    x_max += x_range * padding
    z_min -= z_range * padding
    z_max += z_range * padding
    
    x_range = x_max - x_min
    z_range = z_max - z_min
    
    # Uniform scaling
    max_range = max(x_range, z_range)
    scale = min(width, height) * 0.8 / max_range
    
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2
    
    def world_to_image(x, z):
        """Convert world X-Z to image coordinates."""
        img_x = int(width / 2 + (x - x_center) * scale)
        img_y = int(height / 2 - (z - z_center) * scale)  # Flip Z for top-down
        return img_x, img_y
    
    # Draw grid
    grid_color = (60, 60, 60)
    for i in range(5):
        x = x_min + i * x_range / 4
        pt1 = world_to_image(x, z_min)
        pt2 = world_to_image(x, z_max)
        cv2.line(canvas, pt1, pt2, grid_color, 1)
        
        z = z_min + i * z_range / 4
        pt1 = world_to_image(x_min, z)
        pt2 = world_to_image(x_max, z)
        cv2.line(canvas, pt1, pt2, grid_color, 1)
    
    # Draw cameras
    def draw_camera(camera: Camera, color: Tuple[int, int, int], label: str):
        pos = world_to_image(camera.position[0], camera.position[2])
        
        # Camera body (circle)
        cv2.circle(canvas, pos, 10, color, -1)
        cv2.circle(canvas, pos, 10, (255, 255, 255), 1)
        
        # View direction
        view_dir = camera.get_view_direction()
        end_x = camera.position[0] + view_dir[0] * max_range * 0.2
        end_z = camera.position[2] + view_dir[2] * max_range * 0.2
        end_pos = world_to_image(end_x, end_z)
        cv2.arrowedLine(canvas, pos, end_pos, color, 2, tipLength=0.3)
        
        # Label
        cv2.putText(canvas, label, (pos[0] - 10, pos[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    draw_camera(camera_a, (255, 150, 100), "A")  # Orange
    draw_camera(camera_b, (100, 150, 255), "B")  # Blue
    
    # Draw trajectory
    if trajectory_3d and "joints" in trajectory_3d:
        positions = trajectory_3d["joints"].get(joint_name, {}).get("positions", [])
        errors = trajectory_3d["joints"].get(joint_name, {}).get("errors", [])
        
        # Get error range for coloring
        valid_errors = [e for e in errors if e is not None and e != float('inf')]
        if valid_errors:
            err_min, err_max = min(valid_errors), max(valid_errors)
            err_range = max(err_max - err_min, 0.001)
        else:
            err_min, err_max, err_range = 0, 1, 1
        
        # Draw path
        prev_pt = None
        for i, pos in enumerate(positions):
            if pos is None:
                prev_pt = None
                continue
            
            pt = world_to_image(pos[0], pos[2])
            
            # Color by error (green = low, red = high)
            if i < len(errors) and errors[i] is not None and errors[i] != float('inf'):
                err_norm = (errors[i] - err_min) / err_range
                r = int(255 * err_norm)
                g = int(255 * (1 - err_norm))
                color = (0, g, r)
            else:
                color = (0, 200, 200)  # Cyan for unknown
            
            if prev_pt is not None:
                cv2.line(canvas, prev_pt, pt, color, 2)
            
            prev_pt = pt
        
        # Draw current frame marker
        if current_frame < len(positions) and positions[current_frame] is not None:
            pos = positions[current_frame]
            pt = world_to_image(pos[0], pos[2])
            cv2.circle(canvas, pt, 8, (0, 255, 255), 2)  # Yellow outline
            cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
        
        # Draw start/end markers
        for i, pos in enumerate(positions):
            if pos is not None:
                pt = world_to_image(pos[0], pos[2])
                cv2.circle(canvas, pt, 6, (0, 255, 0), 2)  # Green start
                cv2.putText(canvas, "S", (pt[0] + 8, pt[1] + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                break
        
        for i in range(len(positions) - 1, -1, -1):
            if positions[i] is not None:
                pt = world_to_image(positions[i][0], positions[i][2])
                cv2.circle(canvas, pt, 6, (0, 0, 255), 2)  # Red end
                cv2.putText(canvas, "E", (pt[0] + 8, pt[1] + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                break
    
    # Labels
    cv2.putText(canvas, "TOP VIEW (X-Z)", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "X", (width - 20, height // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(canvas, "Z", (width // 2, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return canvas


def create_error_graph(
    trajectory_3d: Dict,
    joint_name: str = "pelvis",
    size: Tuple[int, int] = (512, 128),
    error_threshold: float = 0.05
) -> np.ndarray:
    """
    Create a graph showing triangulation error over time.
    
    Args:
        trajectory_3d: Triangulated trajectory data
        joint_name: Joint to visualize
        size: Output size (width, height)
        error_threshold: Threshold for "good" error (meters)
    
    Returns:
        Error graph image [H, W, 3] uint8
    """
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    
    if not trajectory_3d or "joints" not in trajectory_3d:
        return canvas
    
    errors = trajectory_3d["joints"].get(joint_name, {}).get("errors", [])
    
    if not errors:
        return canvas
    
    # Filter valid errors
    valid_errors = [(i, e) for i, e in enumerate(errors) if e is not None and e != float('inf')]
    
    if not valid_errors:
        return canvas
    
    # Get bounds
    max_error = max(e for _, e in valid_errors)
    max_error = max(max_error, error_threshold * 2)  # Ensure threshold is visible
    
    num_frames = len(errors)
    
    # Margins
    margin_left = 50
    margin_right = 10
    margin_top = 20
    margin_bottom = 25
    
    graph_w = width - margin_left - margin_right
    graph_h = height - margin_top - margin_bottom
    
    # Draw threshold line
    threshold_y = int(margin_top + graph_h * (1 - error_threshold / max_error))
    cv2.line(canvas, (margin_left, threshold_y), (width - margin_right, threshold_y),
             (0, 100, 0), 1)
    
    # Draw axes
    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom),
             (100, 100, 100), 1)
    cv2.line(canvas, (margin_left, height - margin_bottom),
             (width - margin_right, height - margin_bottom), (100, 100, 100), 1)
    
    # Draw error curve
    prev_pt = None
    for frame_idx, error in valid_errors:
        x = int(margin_left + (frame_idx / max(num_frames - 1, 1)) * graph_w)
        y = int(margin_top + graph_h * (1 - error / max_error))
        
        # Color by error level
        if error <= error_threshold:
            color = (0, 255, 0)  # Green
        elif error <= error_threshold * 2:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        if prev_pt is not None:
            cv2.line(canvas, prev_pt, (x, y), color, 1)
        
        prev_pt = (x, y)
    
    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Error", (5, height // 2), font, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, f"{max_error:.2f}m", (5, margin_top + 10), font, 0.3, (150, 150, 150), 1)
    cv2.putText(canvas, "0", (5, height - margin_bottom), font, 0.3, (150, 150, 150), 1)
    cv2.putText(canvas, f"Frames (1-{num_frames})", (width // 2 - 30, height - 5),
               font, 0.35, (150, 150, 150), 1)
    
    return canvas
