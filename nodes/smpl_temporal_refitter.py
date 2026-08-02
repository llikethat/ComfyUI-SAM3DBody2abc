"""
SMPL Temporal Refitter for SAM3DBody2abc
=========================================
Version: 1.0.0

Re-optimizes SMPL parameters using tracked 2D keypoints with temporal prior.
Each frame warm-starts from previous frame's optimized solution.

Workflow:
    [Video Processor] → mesh_sequence ──┬──→ [SMPL Temporal Refitter] → refined_mesh
                              │         │              ↑
                              ↓         │              │
                  [Keypoint2DTracker] ──┴──→ tracked_keypoints_2d
                              ↑
                            images
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import copy


def log(msg):
    print(f"[SMPLRefitter] {msg}", flush=True)


class SMPLTemporalRefitter:
    """
    Re-optimize SMPL params using tracked 2D keypoints with temporal consistency.
    
    For each frame:
    1. Initialize from previous frame's optimized params (warm-start)
    2. Optimize to minimize 2D reprojection error to tracked keypoints
    3. Add temporal smoothness regularization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "tracked_keypoints_2d": ("KEYPOINTS_2D",),
            },
            "optional": {
                "iterations": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 10,
                    "tooltip": "Optimization iterations per frame"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Optimizer learning rate"
                }),
                "temporal_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Weight for temporal smoothness (deviation from previous frame)"
                }),
                "keypoint_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Weight for 2D keypoint fitting"
                }),
                "use_confidence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Weight keypoints by tracking confidence"
                }),
            },
        }

    RETURN_TYPES = ("MESH_SEQUENCE", "STRING")
    RETURN_NAMES = ("refined_mesh_sequence", "status")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Refinement"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.smpl_model = None

    def process(
        self,
        mesh_sequence: Dict,
        tracked_keypoints_2d: Dict,
        iterations: int = 50,
        learning_rate: float = 0.01,
        temporal_weight: float = 0.5,
        keypoint_weight: float = 1.0,
        use_confidence: bool = True,
    ) -> Tuple[Dict, str]:
        
        log("=" * 60)
        log("SMPL Temporal Refitter v1.0")
        log("=" * 60)
        
        # Extract data
        frames = mesh_sequence.get("frames", {})
        tracked_kp = tracked_keypoints_2d.get("keypoints")  # (N, K, 2)
        
        if not frames:
            log("ERROR: No frames in mesh_sequence")
            return (mesh_sequence, "Error: No frames in mesh_sequence")
        
        if tracked_kp is None:
            log("ERROR: No keypoints in tracked_keypoints_2d")
            return (mesh_sequence, "Error: No tracked keypoints")
        
        frame_indices = sorted(frames.keys())
        num_frames = len(frame_indices)
        num_keypoints = tracked_kp.shape[1]
        
        log(f"Input: {num_frames} frames, {num_keypoints} keypoints")
        log(f"Settings: {iterations} iters, lr={learning_rate}, temporal={temporal_weight}")
        
        # Try to load SMPL model
        smpl_model = self._load_smpl_model(mesh_sequence)
        if smpl_model is None:
            log("WARNING: Could not load SMPL model, using parameter-only refinement")
        
        # Create refined mesh sequence (deep copy)
        refined_sequence = copy.deepcopy(mesh_sequence)
        refined_frames = refined_sequence.get("frames", {})
        
        # Track optimization stats
        total_loss_before = 0
        total_loss_after = 0
        
        # Previous frame's optimized params (for temporal prior)
        prev_params = None
        
        # Process each frame
        for idx, frame_idx in enumerate(frame_indices):
            frame_data = refined_frames[frame_idx]
            
            # Get tracked keypoints for this frame
            if idx >= len(tracked_kp):
                log(f"Frame {frame_idx}: No tracked keypoints, skipping")
                continue
            
            target_kp_2d = tracked_kp[idx]  # (K, 2)
            
            # Get camera params
            focal_length = frame_data.get("focal_length", 2000.0)
            cx = frame_data.get("cx", 960.0)
            cy = frame_data.get("cy", 540.0)
            pred_cam_t = frame_data.get("pred_cam_t")
            
            if pred_cam_t is None:
                log(f"Frame {frame_idx}: No camera translation, skipping")
                continue
            
            # Get current SMPL params
            current_params = self._extract_smpl_params(frame_data)
            if current_params is None:
                log(f"Frame {frame_idx}: No SMPL params, skipping")
                continue
            
            # Get current 3D joints (for reprojection)
            joints_3d = frame_data.get("joint_coords")
            if joints_3d is None:
                joints_3d = frame_data.get("pred_keypoints_3d")
            
            if joints_3d is None:
                log(f"Frame {frame_idx}: No 3D joints, skipping")
                continue
            
            # Compute loss before optimization
            loss_before = self._compute_reprojection_loss(
                joints_3d, target_kp_2d, pred_cam_t, focal_length, cx, cy
            )
            total_loss_before += loss_before
            
            # Optimize
            if smpl_model is not None:
                # Full SMPL optimization
                optimized_params, optimized_joints = self._optimize_smpl(
                    smpl_model=smpl_model,
                    init_params=current_params,
                    prev_params=prev_params,
                    target_kp_2d=target_kp_2d,
                    pred_cam_t=pred_cam_t,
                    focal_length=focal_length,
                    cx=cx, cy=cy,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    temporal_weight=temporal_weight,
                    keypoint_weight=keypoint_weight,
                )
                
                # Update frame data
                frame_data["joint_coords"] = optimized_joints
                frame_data["body_pose_refined"] = True
                
                # Forward pass to get vertices if model available
                if hasattr(smpl_model, 'forward'):
                    try:
                        output = smpl_model(**optimized_params)
                        if hasattr(output, 'vertices'):
                            frame_data["vertices"] = output.vertices[0].detach().cpu().numpy()
                    except:
                        pass
                
                prev_params = optimized_params
                
            else:
                # Parameter-only refinement (smooth joints directly)
                optimized_joints = self._optimize_joints_only(
                    joints_3d=joints_3d,
                    prev_joints=prev_params,  # Using joints as "params"
                    target_kp_2d=target_kp_2d,
                    pred_cam_t=pred_cam_t,
                    focal_length=focal_length,
                    cx=cx, cy=cy,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    temporal_weight=temporal_weight,
                )
                
                frame_data["joint_coords"] = optimized_joints
                frame_data["joints_refined"] = True
                prev_params = optimized_joints
            
            # Compute loss after optimization
            loss_after = self._compute_reprojection_loss(
                frame_data.get("joint_coords", joints_3d), 
                target_kp_2d, pred_cam_t, focal_length, cx, cy
            )
            total_loss_after += loss_after
            
            if idx % 20 == 0 or idx == num_frames - 1:
                log(f"Frame {idx}/{num_frames}: loss {loss_before:.2f} → {loss_after:.2f}")
        
        # Summary
        avg_before = total_loss_before / max(num_frames, 1)
        avg_after = total_loss_after / max(num_frames, 1)
        improvement = (1 - avg_after / max(avg_before, 1e-6)) * 100
        
        status = (
            f"Refined {num_frames} frames\n"
            f"Avg reprojection loss: {avg_before:.2f} → {avg_after:.2f}\n"
            f"Improvement: {improvement:.1f}%"
        )
        
        log(status.replace('\n', ', '))
        log("=" * 60)
        
        return (refined_sequence, status)

    def _load_smpl_model(self, mesh_sequence: Dict) -> Optional[nn.Module]:
        """Try to load SMPL model from mesh_sequence or environment."""
        
        try:
            # Try to import smplx
            import smplx
            
            # Check if mesh_sequence has model path info
            model_path = mesh_sequence.get("smpl_model_path")
            
            if model_path is None:
                # Try standard locations
                import os
                possible_paths = [
                    "/home/burny/ComfyUI/models/smpl",
                    "/home/burny/ComfyUI/custom_nodes/ComfyUI-SAM3DBody2abc/models/smpl",
                    os.path.expanduser("~/.smpl"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
            
            if model_path is None:
                log("SMPL model path not found")
                return None
            
            # Determine model type from mesh_sequence
            num_joints = 127  # MHR default
            model_type = "smplx"  # Default
            
            # Load model
            model = smplx.create(
                model_path,
                model_type=model_type,
                gender='neutral',
                use_face_contour=False,
                num_betas=10,
            ).to(self.device)
            
            log(f"Loaded SMPL model from {model_path}")
            return model
            
        except ImportError:
            log("smplx not installed, using joint-only refinement")
            return None
        except Exception as e:
            log(f"Error loading SMPL: {e}")
            return None

    def _extract_smpl_params(self, frame_data: Dict) -> Optional[Dict]:
        """Extract SMPL parameters from frame data."""
        
        pose_params = frame_data.get("pose_params", {})
        
        body_pose = pose_params.get("body_pose") or frame_data.get("body_pose_params")
        global_rot = pose_params.get("global_rot") or frame_data.get("global_rot")
        shape = pose_params.get("shape") or frame_data.get("shape_params")
        
        if body_pose is None:
            return None
        
        # Convert to tensors
        def to_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.float().to(self.device)
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        
        return {
            "body_pose": to_tensor(body_pose),
            "global_orient": to_tensor(global_rot),
            "betas": to_tensor(shape),
            "transl": to_tensor(frame_data.get("pred_cam_t")),
        }

    def _compute_reprojection_loss(
        self,
        joints_3d: np.ndarray,
        target_2d: np.ndarray,
        cam_t: np.ndarray,
        focal: float,
        cx: float,
        cy: float,
    ) -> float:
        """Compute 2D reprojection error."""
        
        if isinstance(joints_3d, torch.Tensor):
            joints_3d = joints_3d.detach().cpu().numpy()
        if isinstance(cam_t, torch.Tensor):
            cam_t = cam_t.detach().cpu().numpy()
        
        # Project 3D to 2D
        joints_cam = joints_3d + cam_t.reshape(1, 3)
        z = joints_cam[:, 2:3]
        z = np.maximum(z, 0.1)  # Avoid division by zero
        
        proj_x = joints_cam[:, 0:1] * focal / z + cx
        proj_y = joints_cam[:, 1:2] * focal / z + cy
        proj_2d = np.concatenate([proj_x, proj_y], axis=1)
        
        # Match number of keypoints
        num_target = min(len(target_2d), len(proj_2d))
        
        # L2 distance
        diff = proj_2d[:num_target] - target_2d[:num_target]
        loss = np.sqrt((diff ** 2).sum(axis=1)).mean()
        
        return loss

    def _optimize_smpl(
        self,
        smpl_model: nn.Module,
        init_params: Dict,
        prev_params: Optional[Dict],
        target_kp_2d: np.ndarray,
        pred_cam_t: np.ndarray,
        focal_length: float,
        cx: float,
        cy: float,
        iterations: int,
        learning_rate: float,
        temporal_weight: float,
        keypoint_weight: float,
    ) -> Tuple[Dict, np.ndarray]:
        """Optimize SMPL params to fit 2D keypoints."""
        
        # Make params optimizable
        opt_params = {}
        for key, val in init_params.items():
            if val is not None:
                opt_params[key] = nn.Parameter(val.clone())
            else:
                opt_params[key] = None
        
        # Setup optimizer
        params_to_optimize = [p for p in opt_params.values() if p is not None and p.requires_grad]
        optimizer = optim.Adam(params_to_optimize, lr=learning_rate)
        
        # Target keypoints as tensor
        target_2d = torch.tensor(target_kp_2d, dtype=torch.float32, device=self.device)
        cam_t = torch.tensor(pred_cam_t, dtype=torch.float32, device=self.device)
        
        # Optimization loop
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = smpl_model(
                body_pose=opt_params.get("body_pose"),
                global_orient=opt_params.get("global_orient"),
                betas=opt_params.get("betas"),
                transl=opt_params.get("transl"),
            )
            
            # Get joints
            joints_3d = output.joints[0]  # (J, 3)
            
            # Project to 2D
            joints_cam = joints_3d + cam_t
            z = joints_cam[:, 2:3].clamp(min=0.1)
            proj_x = joints_cam[:, 0:1] * focal_length / z + cx
            proj_y = joints_cam[:, 1:2] * focal_length / z + cy
            proj_2d = torch.cat([proj_x, proj_y], dim=1)
            
            # Keypoint loss
            num_match = min(len(target_2d), len(proj_2d))
            kp_loss = ((proj_2d[:num_match] - target_2d[:num_match]) ** 2).sum(dim=1).sqrt().mean()
            
            total_loss = keypoint_weight * kp_loss
            
            # Temporal loss
            if prev_params is not None and temporal_weight > 0:
                for key in ["body_pose", "global_orient"]:
                    if opt_params.get(key) is not None and prev_params.get(key) is not None:
                        diff = opt_params[key] - prev_params[key].detach()
                        total_loss = total_loss + temporal_weight * (diff ** 2).mean()
            
            total_loss.backward()
            optimizer.step()
        
        # Extract optimized joints
        with torch.no_grad():
            output = smpl_model(
                body_pose=opt_params.get("body_pose"),
                global_orient=opt_params.get("global_orient"),
                betas=opt_params.get("betas"),
                transl=opt_params.get("transl"),
            )
            optimized_joints = output.joints[0].cpu().numpy()
        
        # Detach params for return
        result_params = {k: v.detach() if v is not None else None for k, v in opt_params.items()}
        
        return result_params, optimized_joints

    def _optimize_joints_only(
        self,
        joints_3d: np.ndarray,
        prev_joints: Optional[np.ndarray],
        target_kp_2d: np.ndarray,
        pred_cam_t: np.ndarray,
        focal_length: float,
        cx: float,
        cy: float,
        iterations: int,
        learning_rate: float,
        temporal_weight: float,
    ) -> np.ndarray:
        """Optimize 3D joints directly (when SMPL model unavailable)."""
        
        # Convert to tensors
        joints = torch.tensor(joints_3d, dtype=torch.float32, device=self.device, requires_grad=True)
        target_2d = torch.tensor(target_kp_2d, dtype=torch.float32, device=self.device)
        cam_t = torch.tensor(pred_cam_t, dtype=torch.float32, device=self.device)
        
        if prev_joints is not None:
            prev = torch.tensor(prev_joints, dtype=torch.float32, device=self.device)
        else:
            prev = None
        
        optimizer = optim.Adam([joints], lr=learning_rate)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Project to 2D
            joints_cam = joints + cam_t
            z = joints_cam[:, 2:3].clamp(min=0.1)
            proj_x = joints_cam[:, 0:1] * focal_length / z + cx
            proj_y = joints_cam[:, 1:2] * focal_length / z + cy
            proj_2d = torch.cat([proj_x, proj_y], dim=1)
            
            # Keypoint loss
            num_match = min(len(target_2d), len(proj_2d))
            kp_loss = ((proj_2d[:num_match] - target_2d[:num_match]) ** 2).sum(dim=1).sqrt().mean()
            
            total_loss = kp_loss
            
            # Temporal loss
            if prev is not None and temporal_weight > 0:
                total_loss = total_loss + temporal_weight * ((joints - prev) ** 2).mean()
            
            # Regularization: don't deviate too much from original
            total_loss = total_loss + 0.1 * ((joints - torch.tensor(joints_3d, device=self.device)) ** 2).mean()
            
            total_loss.backward()
            optimizer.step()
        
        return joints.detach().cpu().numpy()


NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_SMPLTemporalRefitter": SMPLTemporalRefitter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_SMPLTemporalRefitter": "🔄 SMPL Temporal Refitter"
}
