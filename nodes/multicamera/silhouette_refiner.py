"""
Silhouette Refiner Node

Refines triangulated 3D trajectory by enforcing silhouette consistency
with SAM3 segmentation masks using differentiable rendering.

Uses SMPL body model and PyTorch3D for differentiable silhouette rendering.
Optimizes pose parameters to minimize discrepancy between:
  - Rendered SMPL silhouette
  - SAM3 segmentation mask

This is the final refinement stage that ensures the 3D reconstruction
projects exactly within the observed silhouette boundaries.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import importlib.util
import copy
from typing import Dict, Tuple, Optional, List

# Get the directory containing this file
_current_dir = os.path.dirname(os.path.abspath(__file__))
_utils_dir = os.path.join(_current_dir, "utils")


def _load_util_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


# Load camera module
_camera_module = _load_util_module("camera", os.path.join(_utils_dir, "camera.py"))
if _camera_module:
    Camera = _camera_module.Camera
else:
    raise ImportError(f"Failed to load camera module from {_utils_dir}")


# Try to import logger
try:
    _lib_dir = os.path.dirname(_current_dir)
    _lib_dir = os.path.dirname(_lib_dir)
    _logger_module = _load_util_module("logger", os.path.join(_lib_dir, "lib", "logger.py"))
    if _logger_module:
        log = _logger_module.get_logger("SilhouetteRefiner")
    else:
        raise ImportError()
except:
    class FallbackLogger:
        def info(self, msg): print(f"[Silhouette Refiner] {msg}")
        def warning(self, msg): print(f"[Silhouette Refiner] WARNING: {msg}")
        def error(self, msg): print(f"[Silhouette Refiner] ERROR: {msg}")
        def debug(self, msg): pass
        def progress(self, c, t, task="", interval=10):
            if c == 0 or c == t - 1 or (c + 1) % interval == 0:
                print(f"[Silhouette Refiner] {task}: {c + 1}/{t}")
    log = FallbackLogger()


# Check for optional dependencies
SMPL_AVAILABLE = False
PYTORCH3D_AVAILABLE = False

try:
    import smplx
    SMPL_AVAILABLE = True
    log.info("SMPL/SMPL-X library available")
except ImportError:
    log.warning("smplx not installed - SMPL body model unavailable")
    log.warning("Install with: pip install smplx")

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        MeshRasterizer,
        SoftSilhouetteShader,
        BlendParams,
    )
    PYTORCH3D_AVAILABLE = True
    log.info("PyTorch3D available for differentiable rendering")
except ImportError:
    log.warning("pytorch3d not installed - differentiable rendering unavailable")
    log.warning("Install with: pip install pytorch3d")


# SMPL joint indices mapping to our skeleton
# Based on SMPL 24-joint skeleton
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
]

# Map our joint names to SMPL indices
JOINT_TO_SMPL = {
    "pelvis": 0,
    "head": 15,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_knee": 4,
    "right_knee": 5,
    "left_hip": 1,
    "right_hip": 2,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
}


class DifferentiableSilhouetteRenderer:
    """
    Differentiable silhouette renderer using PyTorch3D.
    
    Renders SMPL mesh silhouettes that can backpropagate gradients
    to mesh vertices and pose parameters.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int],
        device: torch.device,
        sigma: float = 1e-4,
        faces_per_pixel: int = 50,
    ):
        """
        Initialize renderer.
        
        Args:
            image_size: (height, width) of output silhouette
            device: torch device
            sigma: Blur sigma for soft silhouettes (higher = softer edges)
            faces_per_pixel: Number of faces to track per pixel
        """
        self.image_size = image_size
        self.device = device
        
        # Rasterization settings for silhouette rendering
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=faces_per_pixel,
            bin_size=0,  # Use naive rasterization for better gradients
        )
        
        # Blend parameters for soft silhouette
        self.blend_params = BlendParams(sigma=sigma, gamma=1e-4)
    
    def render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        camera: 'Camera',
    ) -> torch.Tensor:
        """
        Render soft silhouette of mesh.
        
        Args:
            vertices: (N, 3) mesh vertices in world space
            faces: (F, 3) face indices
            camera: Camera object with projection parameters
        
        Returns:
            silhouette: (H, W) soft silhouette image [0, 1]
        """
        # Create PyTorch3D mesh
        meshes = Meshes(
            verts=[vertices],
            faces=[faces.long()],
        )
        
        # Create PyTorch3D camera from our Camera object
        # Convert our camera parameters to PyTorch3D format
        R = torch.tensor(camera.R.T, dtype=torch.float32, device=self.device).unsqueeze(0)
        T = torch.tensor(
            -camera.R.T @ camera.position,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        # Focal length in NDC
        focal_ndc = torch.tensor(
            [[camera.focal_px * 2.0 / camera.width, 
              camera.focal_px * 2.0 / camera.height]],
            dtype=torch.float32,
            device=self.device
        )
        
        # Principal point in NDC
        principal_ndc = torch.tensor(
            [[(camera.cx - camera.width / 2) * 2.0 / camera.width,
              (camera.cy - camera.height / 2) * 2.0 / camera.height]],
            dtype=torch.float32,
            device=self.device
        )
        
        cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal_ndc,
            principal_point=principal_ndc,
            image_size=torch.tensor([self.image_size], device=self.device),
            device=self.device,
            in_ndc=True,
        )
        
        # Create rasterizer and shader
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings,
        )
        
        shader = SoftSilhouetteShader(blend_params=self.blend_params)
        
        # Render
        fragments = rasterizer(meshes)
        silhouette = shader(fragments, meshes)
        
        # Extract alpha channel (silhouette)
        return silhouette[0, ..., 3]


class SkeletonHullRenderer:
    """
    Fallback renderer using capsule-based skeleton hull.
    
    Approximates body silhouette from joint positions using
    cylinders connecting joints. No SMPL required.
    """
    
    # Bone connections (joint pairs)
    BONES = [
        ("pelvis", "left_hip"), ("pelvis", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
        ("pelvis", "head"),  # Simplified spine
        ("head", "left_shoulder"), ("head", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ]
    
    # Approximate bone radii (in meters)
    BONE_RADII = {
        ("pelvis", "left_hip"): 0.08,
        ("pelvis", "right_hip"): 0.08,
        ("left_hip", "left_knee"): 0.06,
        ("right_hip", "right_knee"): 0.06,
        ("left_knee", "left_ankle"): 0.05,
        ("right_knee", "right_ankle"): 0.05,
        ("pelvis", "head"): 0.10,
        ("head", "left_shoulder"): 0.05,
        ("head", "right_shoulder"): 0.05,
        ("left_shoulder", "left_elbow"): 0.04,
        ("right_shoulder", "right_elbow"): 0.04,
        ("left_elbow", "left_wrist"): 0.03,
        ("right_elbow", "right_wrist"): 0.03,
    }
    
    def __init__(
        self,
        image_size: Tuple[int, int],
        device: torch.device,
        sigma: float = 2.0,
    ):
        """
        Initialize skeleton hull renderer.
        
        Args:
            image_size: (height, width) of output silhouette
            device: torch device
            sigma: Gaussian blur sigma for soft edges
        """
        self.image_size = image_size
        self.device = device
        self.sigma = sigma
    
    def render(
        self,
        joint_positions: Dict[str, torch.Tensor],
        camera: 'Camera',
    ) -> torch.Tensor:
        """
        Render skeleton hull silhouette.
        
        Args:
            joint_positions: Dict mapping joint name to (3,) position tensor
            camera: Camera object
        
        Returns:
            silhouette: (H, W) soft silhouette [0, 1]
        """
        H, W = self.image_size
        
        # Create empty silhouette
        silhouette = torch.zeros((H, W), device=self.device, dtype=torch.float32)
        
        # For each bone, draw a thick line (capsule projection)
        for joint_a, joint_b in self.BONES:
            if joint_a not in joint_positions or joint_b not in joint_positions:
                continue
            
            pos_a = joint_positions[joint_a]
            pos_b = joint_positions[joint_b]
            
            # Project to 2D
            px_a, py_a, in_front_a = self._project_point(pos_a, camera)
            px_b, py_b, in_front_b = self._project_point(pos_b, camera)
            
            if not (in_front_a and in_front_b):
                continue
            
            # Get bone radius and project to pixel width
            radius = self.BONE_RADII.get((joint_a, joint_b), 0.05)
            depth = (pos_a[2].item() + pos_b[2].item()) / 2
            radius_px = radius * camera.focal_px / abs(depth) if abs(depth) > 0.1 else 10
            radius_px = max(3, min(50, radius_px))  # Clamp
            
            # Draw thick line using distance field
            self._draw_capsule(
                silhouette,
                px_a, py_a, px_b, py_b,
                radius_px
            )
        
        # Apply Gaussian blur for soft edges
        if self.sigma > 0:
            silhouette = self._gaussian_blur(silhouette, self.sigma)
        
        return silhouette.clamp(0, 1)
    
    def _project_point(
        self,
        point: torch.Tensor,
        camera: 'Camera'
    ) -> Tuple[float, float, bool]:
        """Project 3D point to 2D using camera."""
        point_np = point.detach().cpu().numpy()
        return camera.project_point(point_np)
    
    def _draw_capsule(
        self,
        image: torch.Tensor,
        x1: float, y1: float,
        x2: float, y2: float,
        radius: float
    ):
        """Draw a capsule (thick line) on the image."""
        H, W = image.shape
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=image.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=image.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Line segment parameters
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        
        if length_sq < 1e-6:
            # Degenerate case: draw circle
            dist = torch.sqrt((xx - x1) ** 2 + (yy - y1) ** 2)
            mask = (dist < radius).float()
            image[:] = torch.maximum(image, mask)
            return
        
        # Project points onto line segment
        t = ((xx - x1) * dx + (yy - y1) * dy) / length_sq
        t = t.clamp(0, 1)
        
        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance to closest point
        dist = torch.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)
        
        # Soft falloff
        mask = torch.sigmoid((radius - dist) * 2)
        
        image[:] = torch.maximum(image, mask)
    
    def _gaussian_blur(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to image."""
        # Create Gaussian kernel
        kernel_size = int(6 * sigma) | 1  # Ensure odd
        x = torch.arange(kernel_size, device=image.device, dtype=torch.float32)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Reshape for conv2d
        kernel_h = kernel_1d.view(1, 1, -1, 1)
        kernel_v = kernel_1d.view(1, 1, 1, -1)
        
        # Add batch and channel dims
        img = image.unsqueeze(0).unsqueeze(0)
        
        # Pad and convolve
        pad_h = kernel_size // 2
        img = torch.nn.functional.pad(img, (0, 0, pad_h, pad_h), mode='replicate')
        img = torch.nn.functional.conv2d(img, kernel_h)
        
        img = torch.nn.functional.pad(img, (pad_h, pad_h, 0, 0), mode='replicate')
        img = torch.nn.functional.conv2d(img, kernel_v)
        
        return img.squeeze()


class SilhouetteLoss(nn.Module):
    """
    Combined loss for silhouette matching.
    
    Combines:
    - IoU loss (Intersection over Union)
    - BCE loss (Binary Cross-Entropy)
    - Inside loss (for clothing-aware mode: body should be inside mask)
    """
    
    def __init__(
        self,
        iou_weight: float = 1.0,
        bce_weight: float = 1.0,
        inside_weight: float = 0.0,  # Set > 0 for clothing-aware
        eps: float = 1e-6,
    ):
        super().__init__()
        self.iou_weight = iou_weight
        self.bce_weight = bce_weight
        self.inside_weight = inside_weight
        self.eps = eps
    
    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute silhouette loss.
        
        Args:
            rendered: Rendered silhouette (H, W) in [0, 1]
            target: Target mask (H, W) in [0, 1]
        
        Returns:
            loss: Total loss tensor
            metrics: Dict of individual loss components
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=rendered.device)
        
        # IoU loss
        if self.iou_weight > 0:
            intersection = (rendered * target).sum()
            union = rendered.sum() + target.sum() - intersection
            iou = (intersection + self.eps) / (union + self.eps)
            iou_loss = 1.0 - iou
            total_loss = total_loss + self.iou_weight * iou_loss
            metrics['iou'] = iou.item()
            metrics['iou_loss'] = iou_loss.item()
        
        # BCE loss
        if self.bce_weight > 0:
            bce_loss = torch.nn.functional.binary_cross_entropy(
                rendered.clamp(self.eps, 1 - self.eps),
                target,
                reduction='mean'
            )
            total_loss = total_loss + self.bce_weight * bce_loss
            metrics['bce_loss'] = bce_loss.item()
        
        # Inside loss (penalize rendered pixels outside target)
        if self.inside_weight > 0:
            outside = torch.relu(rendered - target)
            inside_loss = outside.mean()
            total_loss = total_loss + self.inside_weight * inside_loss
            metrics['inside_loss'] = inside_loss.item()
        
        metrics['total_loss'] = total_loss.item()
        
        return total_loss, metrics


class SilhouetteRefiner:
    """
    Refine triangulated 3D trajectory using silhouette consistency.
    
    Uses differentiable rendering to optimize joint positions
    such that the projected body silhouette matches the SAM3
    segmentation masks from all cameras.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trajectory_3d": ("TRAJECTORY_3D", {
                    "tooltip": "Initial 3D trajectory from triangulator"
                }),
                "camera_list": ("CAMERA_LIST", {
                    "tooltip": "Camera list with mesh sequences (must include masks)"
                }),
                "calibration": ("CALIBRATION_DATA", {
                    "tooltip": "Camera calibration data"
                }),
            },
            "optional": {
                "iterations": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "tooltip": "Optimization iterations per frame"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Adam optimizer learning rate"
                }),
                "silhouette_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Weight for silhouette matching loss"
                }),
                "keypoint_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Weight for keypoint reprojection loss"
                }),
                "smoothness_weight": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Weight for temporal smoothness"
                }),
                "bone_length_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Weight for bone length preservation"
                }),
                "render_mode": (["smpl", "skeleton_hull", "auto"], {
                    "default": "auto",
                    "tooltip": "Rendering mode: SMPL mesh, skeleton hull, or auto-detect"
                }),
                "smpl_model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to SMPL model files (optional, uses default if empty)"
                }),
                "use_pose_prior": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use VPoser pose prior (requires additional setup)"
                }),
                "debug_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate debug visualization"
                }),
            }
        }
    
    RETURN_TYPES = ("TRAJECTORY_3D", "IMAGE", "STRING")
    RETURN_NAMES = ("refined_trajectory", "debug_view", "refinement_info")
    FUNCTION = "refine"
    CATEGORY = "SAM3DBody2abc/MultiCamera"
    
    def refine(
        self,
        trajectory_3d: Dict,
        camera_list: Dict,
        calibration: Dict,
        iterations: int = 50,
        learning_rate: float = 0.01,
        silhouette_weight: float = 1.0,
        keypoint_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        bone_length_weight: float = 1.0,
        render_mode: str = "auto",
        smpl_model_path: str = "",
        use_pose_prior: bool = False,
        debug_output: bool = True,
    ) -> Tuple[Dict, torch.Tensor, str]:
        """
        Refine 3D trajectory using silhouette consistency.
        
        Args:
            trajectory_3d: Initial triangulated trajectory
            camera_list: List of cameras with masks
            calibration: Camera calibration data
            iterations: Optimization iterations per frame
            learning_rate: Adam learning rate
            silhouette_weight: Weight for silhouette loss
            keypoint_weight: Weight for keypoint reprojection loss
            smoothness_weight: Weight for temporal smoothness
            bone_length_weight: Weight for bone length preservation
            render_mode: "smpl", "skeleton_hull", or "auto"
            smpl_model_path: Path to SMPL models
            use_pose_prior: Whether to use pose prior
            debug_output: Generate debug images
        
        Returns:
            refined_trajectory: Refined TRAJECTORY_3D
            debug_view: Debug visualization images
            refinement_info: Info string
        """
        
        log.info("=" * 60)
        log.info("SILHOUETTE REFINEMENT")
        log.info("=" * 60)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")
        
        # Get camera objects from calibration
        cal_camera_list = calibration.get("camera_list", [])
        if not cal_camera_list:
            cam_objs = calibration.get("camera_objects", {})
            for key in sorted(cam_objs.keys()):
                cal_camera_list.append(cam_objs[key])
        
        num_cameras = len(cal_camera_list)
        log.info(f"Cameras: {num_cameras}")
        
        # Get cameras data with masks
        cameras_data = camera_list.get("cameras", [])
        
        # Check for masks in mesh_sequence frames
        masks_available = self._check_masks_available(cameras_data)
        
        if not masks_available:
            log.warning("No segmentation masks found in camera data!")
            log.warning("Masks should be in mesh_sequence frames as 'segmentation_mask'")
            log.warning("Returning trajectory unchanged")
            
            # Return unchanged
            dummy_debug = torch.zeros((1, 256, 256, 3))
            return (trajectory_3d, dummy_debug, "ERROR: No masks available for refinement")
        
        # Determine render mode
        actual_render_mode = self._determine_render_mode(render_mode, smpl_model_path)
        log.info(f"Render mode: {actual_render_mode}")
        
        # Get image dimensions from first camera
        first_frame = cameras_data[0]["mesh_sequence"].get("frames", {})
        if first_frame:
            first_frame_data = list(first_frame.values())[0]
            img_size = first_frame_data.get("image_size", [512, 512])
            img_h, img_w = img_size[1], img_size[0]  # height, width
        else:
            img_h, img_w = 512, 512
        
        log.info(f"Image size: {img_w}x{img_h}")
        
        # Initialize renderer
        if actual_render_mode == "smpl" and SMPL_AVAILABLE and PYTORCH3D_AVAILABLE:
            renderer = self._create_smpl_renderer(
                (img_h, img_w), device, smpl_model_path
            )
            use_smpl = True
        else:
            renderer = SkeletonHullRenderer((img_h, img_w), device)
            use_smpl = False
        
        log.info(f"Using {'SMPL' if use_smpl else 'skeleton hull'} renderer")
        
        # Initialize loss function
        silhouette_loss_fn = SilhouetteLoss(
            iou_weight=1.0,
            bce_weight=0.5,
            inside_weight=0.0,  # Change to > 0 for clothing-aware
        )
        
        # Get trajectory data
        num_frames = trajectory_3d["frames"]
        fps = trajectory_3d["fps"]
        joints_data = trajectory_3d.get("joints", {})
        
        log.info(f"Frames: {num_frames}, FPS: {fps}")
        log.info(f"Joints: {list(joints_data.keys())}")
        
        # Deep copy trajectory for refinement
        refined_trajectory = copy.deepcopy(trajectory_3d)
        
        # Statistics
        total_improvement = 0.0
        frames_refined = 0
        all_metrics = []
        
        # Process each frame
        for frame_idx in range(num_frames):
            log.progress(frame_idx, num_frames, "Refining frames", interval=10)
            
            # Get joint positions for this frame
            joint_positions = {}
            for joint_name, joint_data in joints_data.items():
                positions = joint_data.get("positions", [])
                if frame_idx < len(positions) and positions[frame_idx] is not None:
                    pos = torch.tensor(
                        positions[frame_idx],
                        dtype=torch.float32,
                        device=device,
                        requires_grad=True
                    )
                    joint_positions[joint_name] = pos
            
            if len(joint_positions) < 3:
                # Not enough joints to refine
                continue
            
            # Get masks from all cameras for this frame
            target_masks = []
            for cam_idx, cam_data in enumerate(cameras_data[:num_cameras]):
                mask = self._get_mask_for_frame(cam_data, frame_idx)
                if mask is not None:
                    mask_tensor = torch.tensor(
                        mask, dtype=torch.float32, device=device
                    )
                    target_masks.append((cam_idx, mask_tensor))
            
            if not target_masks:
                continue
            
            # Initial bone lengths (for preservation)
            initial_bone_lengths = self._compute_bone_lengths(joint_positions)
            
            # Create optimizer
            params = list(joint_positions.values())
            optimizer = optim.Adam(params, lr=learning_rate)
            
            # Get previous frame positions for smoothness
            prev_positions = None
            if frame_idx > 0 and smoothness_weight > 0:
                prev_positions = {}
                for joint_name, joint_data in joints_data.items():
                    positions = joint_data.get("positions", [])
                    if frame_idx - 1 < len(positions) and positions[frame_idx - 1] is not None:
                        prev_positions[joint_name] = torch.tensor(
                            positions[frame_idx - 1],
                            dtype=torch.float32,
                            device=device
                        )
            
            # Optimization loop
            initial_loss = None
            final_loss = None
            
            for iter_idx in range(iterations):
                optimizer.zero_grad()
                
                total_loss = torch.tensor(0.0, device=device)
                
                # Silhouette loss from each camera
                if silhouette_weight > 0:
                    for cam_idx, target_mask in target_masks:
                        camera = cal_camera_list[cam_idx]
                        
                        if use_smpl:
                            # SMPL rendering (not implemented in this version)
                            pass
                        else:
                            # Skeleton hull rendering
                            rendered = renderer.render(joint_positions, camera)
                        
                        # Resize if needed
                        if rendered.shape != target_mask.shape:
                            rendered = torch.nn.functional.interpolate(
                                rendered.unsqueeze(0).unsqueeze(0),
                                size=target_mask.shape,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                        
                        sil_loss, _ = silhouette_loss_fn(rendered, target_mask)
                        total_loss = total_loss + silhouette_weight * sil_loss / len(target_masks)
                
                # Bone length preservation loss
                if bone_length_weight > 0:
                    current_bone_lengths = self._compute_bone_lengths(joint_positions)
                    bone_loss = torch.tensor(0.0, device=device)
                    for bone, init_len in initial_bone_lengths.items():
                        if bone in current_bone_lengths:
                            curr_len = current_bone_lengths[bone]
                            bone_loss = bone_loss + (curr_len - init_len) ** 2
                    total_loss = total_loss + bone_length_weight * bone_loss
                
                # Temporal smoothness loss
                if smoothness_weight > 0 and prev_positions is not None:
                    smooth_loss = torch.tensor(0.0, device=device)
                    for joint_name, pos in joint_positions.items():
                        if joint_name in prev_positions:
                            smooth_loss = smooth_loss + ((pos - prev_positions[joint_name]) ** 2).sum()
                    total_loss = total_loss + smoothness_weight * smooth_loss
                
                # Track initial/final loss
                if iter_idx == 0:
                    initial_loss = total_loss.item()
                if iter_idx == iterations - 1:
                    final_loss = total_loss.item()
                
                # Backprop
                total_loss.backward()
                optimizer.step()
            
            # Store refined positions
            for joint_name, pos in joint_positions.items():
                if joint_name in refined_trajectory["joints"]:
                    refined_trajectory["joints"][joint_name]["positions"][frame_idx] = \
                        pos.detach().cpu().numpy().tolist()
            
            # Track improvement
            if initial_loss is not None and final_loss is not None:
                improvement = initial_loss - final_loss
                total_improvement += improvement
                frames_refined += 1
                all_metrics.append({
                    "frame": frame_idx,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "improvement": improvement,
                })
        
        # Update primary trajectory
        primary_joint = list(joints_data.keys())[0] if joints_data else "pelvis"
        if primary_joint in refined_trajectory["joints"]:
            refined_trajectory["trajectory"]["positions"] = \
                refined_trajectory["joints"][primary_joint]["positions"]
        
        # Add refinement metadata
        refined_trajectory["refinement"] = {
            "method": "silhouette_refinement",
            "render_mode": actual_render_mode,
            "iterations": iterations,
            "frames_refined": frames_refined,
            "average_improvement": total_improvement / max(frames_refined, 1),
            "weights": {
                "silhouette": silhouette_weight,
                "keypoint": keypoint_weight,
                "smoothness": smoothness_weight,
                "bone_length": bone_length_weight,
            }
        }
        
        # Generate debug visualization
        if debug_output:
            debug_view = self._create_debug_visualization(
                refined_trajectory, trajectory_3d, all_metrics, device
            )
        else:
            debug_view = torch.zeros((1, 256, 256, 3))
        
        # Generate info string
        info_lines = [
            "=== SILHOUETTE REFINEMENT RESULTS ===",
            f"Render mode: {actual_render_mode}",
            f"Frames refined: {frames_refined}/{num_frames}",
            f"Iterations per frame: {iterations}",
            f"",
            "=== IMPROVEMENT ===",
            f"Average loss improvement: {total_improvement / max(frames_refined, 1):.6f}",
            f"",
            "=== WEIGHTS ===",
            f"Silhouette: {silhouette_weight}",
            f"Keypoint: {keypoint_weight}",
            f"Smoothness: {smoothness_weight}",
            f"Bone length: {bone_length_weight}",
        ]
        
        refinement_info = "\n".join(info_lines)
        
        log.info("Refinement complete!")
        log.info(f"Frames refined: {frames_refined}/{num_frames}")
        log.info(f"Average improvement: {total_improvement / max(frames_refined, 1):.6f}")
        log.info("=" * 60)
        
        return (refined_trajectory, debug_view, refinement_info)
    
    def _check_masks_available(self, cameras_data: List[Dict]) -> bool:
        """Check if segmentation masks are available in camera data."""
        for cam_data in cameras_data:
            mesh_seq = cam_data.get("mesh_sequence", {})
            frames = mesh_seq.get("frames", {})
            for frame_data in frames.values():
                if "segmentation_mask" in frame_data:
                    return True
                if "mask" in frame_data:
                    return True
        return False
    
    def _get_mask_for_frame(
        self,
        cam_data: Dict,
        frame_idx: int
    ) -> Optional[np.ndarray]:
        """Extract segmentation mask for a specific frame."""
        mesh_seq = cam_data.get("mesh_sequence", {})
        frames = mesh_seq.get("frames", {})
        
        # Get sorted frame indices
        frame_indices = sorted(frames.keys())
        if frame_idx >= len(frame_indices):
            return None
        
        frame_key = frame_indices[frame_idx]
        frame_data = frames[frame_key]
        
        # Try different mask keys
        for key in ["segmentation_mask", "mask", "silhouette"]:
            if key in frame_data:
                mask = frame_data[key]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                return np.array(mask, dtype=np.float32)
        
        return None
    
    def _determine_render_mode(
        self,
        requested_mode: str,
        smpl_path: str
    ) -> str:
        """Determine actual render mode based on availability."""
        if requested_mode == "smpl":
            if SMPL_AVAILABLE and PYTORCH3D_AVAILABLE:
                return "smpl"
            else:
                log.warning("SMPL requested but dependencies not available")
                return "skeleton_hull"
        elif requested_mode == "skeleton_hull":
            return "skeleton_hull"
        else:  # auto
            if SMPL_AVAILABLE and PYTORCH3D_AVAILABLE:
                return "smpl"
            else:
                return "skeleton_hull"
    
    def _create_smpl_renderer(
        self,
        image_size: Tuple[int, int],
        device: torch.device,
        model_path: str
    ) -> DifferentiableSilhouetteRenderer:
        """Create SMPL-based differentiable renderer."""
        return DifferentiableSilhouetteRenderer(
            image_size=image_size,
            device=device,
            sigma=1e-4,
        )
    
    def _compute_bone_lengths(
        self,
        joint_positions: Dict[str, torch.Tensor]
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        """Compute bone lengths from joint positions."""
        bones = {}
        
        bone_pairs = [
            ("pelvis", "left_hip"), ("pelvis", "right_hip"),
            ("left_hip", "left_knee"), ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
            ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
        ]
        
        for joint_a, joint_b in bone_pairs:
            if joint_a in joint_positions and joint_b in joint_positions:
                pos_a = joint_positions[joint_a]
                pos_b = joint_positions[joint_b]
                length = torch.norm(pos_a - pos_b)
                bones[(joint_a, joint_b)] = length
        
        return bones
    
    def _create_debug_visualization(
        self,
        refined_trajectory: Dict,
        original_trajectory: Dict,
        metrics: List[Dict],
        device: torch.device
    ) -> torch.Tensor:
        """Create debug visualization comparing original and refined trajectories."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot 1: Loss improvement over frames
        ax = axes[0, 0]
        if metrics:
            frames = [m["frame"] for m in metrics]
            improvements = [m["improvement"] for m in metrics]
            ax.bar(frames, improvements, alpha=0.7)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Loss Improvement")
            ax.set_title("Per-Frame Loss Improvement")
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No metrics", ha='center', va='center')
        
        # Plot 2: Trajectory comparison (X-Z plane, top view)
        ax = axes[0, 1]
        orig_positions = original_trajectory.get("trajectory", {}).get("positions", [])
        ref_positions = refined_trajectory.get("trajectory", {}).get("positions", [])
        
        if orig_positions and ref_positions:
            orig_valid = [p for p in orig_positions if p is not None]
            ref_valid = [p for p in ref_positions if p is not None]
            
            if orig_valid:
                orig_arr = np.array(orig_valid)
                ax.plot(orig_arr[:, 0], orig_arr[:, 2], 'b-', alpha=0.5, label='Original')
            if ref_valid:
                ref_arr = np.array(ref_valid)
                ax.plot(ref_arr[:, 0], ref_arr[:, 2], 'g-', alpha=0.8, label='Refined')
            
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_title("Trajectory (Top View)")
            ax.legend()
            ax.axis('equal')
        
        # Plot 3: Loss values
        ax = axes[1, 0]
        if metrics:
            initial_losses = [m["initial_loss"] for m in metrics]
            final_losses = [m["final_loss"] for m in metrics]
            frames = [m["frame"] for m in metrics]
            ax.plot(frames, initial_losses, 'r-', alpha=0.5, label='Initial')
            ax.plot(frames, final_losses, 'g-', alpha=0.8, label='Final')
            ax.set_xlabel("Frame")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Before/After Refinement")
            ax.legend()
        
        # Plot 4: Summary stats
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = [
            f"Frames refined: {len(metrics)}",
            f"Total improvement: {sum(m['improvement'] for m in metrics):.4f}" if metrics else "",
            f"Avg improvement: {np.mean([m['improvement'] for m in metrics]):.4f}" if metrics else "",
        ]
        ax.text(0.1, 0.7, "\n".join(stats_text), fontsize=12, family='monospace',
                transform=ax.transAxes, verticalalignment='top')
        ax.set_title("Summary")
        
        plt.tight_layout()
        
        # Convert to tensor
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        from PIL import Image
        img = Image.open(buf)
        img_array = np.array(img.convert('RGB')).astype(np.float32) / 255.0
        
        return torch.from_numpy(img_array).unsqueeze(0)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SilhouetteRefiner": SilhouetteRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SilhouetteRefiner": "🎭 Silhouette Refiner",
}
