"""
GroundLink Physics-Based Foot Contact Solver
============================================

MIT Licensed physics-based foot contact detection using GroundLink neural network.
PRIMARY foot contact solver with TAPNet and Heuristic fallbacks.

References:
- Paper: "GroundLink: A Dataset Unifying Human Body Movement and Ground Reaction Dynamics"
- SIGGRAPH Asia 2023, Han et al.
- License: MIT (commercial use permitted)
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.spatial.transform import Rotation

torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


# =============================================================================
# GroundLink Neural Network (Exact Architecture Match)
# =============================================================================

def build_groundlink_model():
    """Build DeepNetwork matching GroundLink's exact architecture."""
    torch = _ensure_torch()
    import torch.nn as nn
    
    n_joints = 23  # GroundLink topology
    cnn_features = [3 * n_joints, 128, 128, 256, 256]  # [69, 128, 128, 256, 256]
    features_out = 6
    cnn_kernel = 7
    cnn_dropout = 0.0
    fc_depth = 3
    fc_dropout = 0.2
    
    class TransposeModule(nn.Module):
        def __init__(self, d1, d2):
            super().__init__()
            self.d1, self.d2 = d1, d2
        def forward(self, x):
            return x.transpose(self.d1, self.d2)
    
    layers = []
    
    # Preprocess: Flatten + Transpose
    layers.append(nn.Flatten(start_dim=2, end_dim=-1))  # 0: N x F x J x 3 -> N x F x 69
    layers.append(TransposeModule(-2, -1))  # 1: N x F x 69 -> N x 69 x F
    
    # CNN layers: Dropout, Conv1d, ELU for each feature transition
    for c_in, c_out in zip(cnn_features[:-1], cnn_features[1:]):
        layers.append(nn.Dropout(p=cnn_dropout))
        layers.append(nn.Conv1d(c_in, c_out, cnn_kernel, padding=cnn_kernel//2, padding_mode='replicate'))
        layers.append(nn.ELU())
    
    # Transpose back
    layers.append(TransposeModule(-2, -1))  # N x 256 x F -> N x F x 256
    
    # FC layers
    for _ in range(fc_depth - 1):
        layers.append(nn.Dropout(p=fc_dropout))
        layers.append(nn.Linear(cnn_features[-1], cnn_features[-1]))
        layers.append(nn.ELU())
    
    # Final output
    layers.append(nn.Dropout(p=fc_dropout))
    layers.append(nn.Linear(cnn_features[-1], 2 * features_out, bias=False))
    layers.append(nn.Unflatten(-1, (2, features_out)))
    
    return nn.Sequential(*layers)


class GroundLinkNet:
    """GroundLink neural network for GRF/CoP prediction."""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        torch = _ensure_torch()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_groundlink_model()
        self._checkpoint_loaded = False
        self._load_error = None
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        
        self.model.eval()
        self.model.to(self.device)
    
    def _load_checkpoint(self, path: str):
        """Load pretrained weights."""
        torch = _ensure_torch()
        
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model', checkpoint)
            self.model.load_state_dict(state_dict)
            self._checkpoint_loaded = True
            self._load_error = None
            print(f"[GroundLink] ✓ Loaded checkpoint from {path}")
        except Exception as e:
            self._load_error = str(e)
            self._checkpoint_loaded = False
            print(f"[GroundLink] ✗ Failed to load checkpoint: {e}")
    
    @property
    def is_ready(self) -> bool:
        return self._checkpoint_loaded
    
    @property
    def load_error(self) -> Optional[str]:
        return self._load_error
    
    def predict(self, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict GRF and CoP from pose sequence."""
        torch = _ensure_torch()
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).float()
        
        if poses.dim() == 3:
            poses = poses.unsqueeze(0)
        
        poses = poses.to(self.device)
        
        with torch.no_grad():
            output = self.model(poses)
        
        output = output.squeeze(0).cpu().numpy()
        
        cop = output[..., :3]
        grf = output[..., 3:]
        
        return grf, cop
    
    def predict_contacts(self, poses: np.ndarray, grf_threshold: float = 0.1, 
                         smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict foot contacts from poses."""
        grf, cop = self.predict(poses)
        
        # Vertical GRF for contact detection (Z-up)
        grf_vertical = np.abs(grf[..., 2])
        contacts = (grf_vertical > grf_threshold).astype(np.float32)
        
        if smooth_window > 1:
            contacts[:, 0] = medfilt(contacts[:, 0], smooth_window)
            contacts[:, 1] = medfilt(contacts[:, 1], smooth_window)
        
        return contacts.astype(bool), grf, cop


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GroundLinkConfig:
    checkpoint_path: str = ""
    grf_threshold: float = 0.1
    smooth_window: int = 5
    pin_feet: bool = True
    blend_frames: int = 3
    up_axis: str = 'y'
    fallback_to_heuristic: bool = True
    fallback_to_tapnet: bool = True
    height_threshold: float = 0.05
    velocity_threshold: float = 0.02
    left_foot_idx: int = 10
    right_foot_idx: int = 11
    left_ankle_idx: int = 7
    right_ankle_idx: int = 8


# =============================================================================
# Pose Conversion
# =============================================================================

def convert_sam3d_to_groundlink(frames: List[Dict], up_axis: str = 'y') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Convert SAM3DBody output to GroundLink input format."""
    T = len(frames)
    
    # Try rotation-based first
    poses, root_trans = _convert_rotations(frames, up_axis)
    if poses is not None:
        return poses, root_trans, 'rotations'
    
    # Fallback to positions
    poses, root_trans = _convert_positions(frames, up_axis)
    if poses is not None:
        return poses, root_trans, 'positions'
    
    return None, None, 'failed'


def _convert_rotations(frames: List[Dict], up_axis: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert using SMPL rotation parameters."""
    T = len(frames)
    
    global_orients = []
    body_poses = []
    translations = []
    
    for frame in frames:
        smpl = frame.get('smpl_params', {})
        
        go = smpl.get('global_orient') or frame.get('pred_global_orient')
        if go is None:
            return None, None
        global_orients.append(np.array(go).flatten()[:3])
        
        bp = smpl.get('body_pose') or frame.get('pred_body_pose')
        if bp is None:
            return None, None
        bp = np.array(bp).flatten()
        if len(bp) < 63:
            bp = np.pad(bp, (0, 63 - len(bp)))
        body_poses.append(bp[:63].reshape(21, 3))
        
        tr = smpl.get('transl') or frame.get('pred_cam_t')
        if tr is None:
            tr = np.zeros(3)
        translations.append(np.array(tr).flatten()[:3])
    
    global_orients = np.stack(global_orients)
    body_poses = np.stack(body_poses)
    translations = np.stack(translations)
    
    # Build poses tensor [T, 23, 3]
    poses = np.zeros((T, 23, 3))
    poses[:, 0, :] = translations  # Pelvis translation
    poses[:, 1, :] = global_orients  # Pelvis rotation
    poses[:, 2:23, :] = body_poses  # Body joints
    
    return poses, translations


def _convert_positions(frames: List[Dict], up_axis: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert using joint positions (fallback)."""
    T = len(frames)
    
    all_joints = []
    translations = []
    
    for frame in frames:
        joints = frame.get('pred_joints_3d') or frame.get('pred_joints') or frame.get('joints_3d')
        if joints is None:
            return None, None
        
        joints = np.array(joints)
        if joints.ndim == 1:
            joints = joints.reshape(-1, 3)
        
        if len(joints) < 22:
            return None, None
        
        all_joints.append(joints[:22])
        
        tr = frame.get('pred_cam_t') or frame.get('smpl_params', {}).get('transl')
        if tr is None:
            tr = joints[0]
        translations.append(np.array(tr).flatten()[:3])
    
    all_joints = np.stack(all_joints)
    translations = np.stack(translations)
    
    # Center around pelvis
    pelvis_pos = all_joints[:, 0, :]
    local_joints = all_joints - pelvis_pos[:, np.newaxis, :]
    
    poses = np.zeros((T, 23, 3))
    poses[:, 0, :] = np.zeros((T, 3))  # Local pelvis at origin
    poses[:, 1:23, :] = local_joints[:, :22, :]
    
    return poses, translations


# =============================================================================
# Contact Enforcer
# =============================================================================

class GroundLinkContactEnforcer:
    """Physics-based foot contact enforcement with fallback chain."""
    
    def __init__(self, config: Optional[GroundLinkConfig] = None):
        self.config = config or GroundLinkConfig()
        self.model = None
        self._method_used = "none"
        self._debug_info = {}
        self._load_model()
    
    def _load_model(self):
        """Load GroundLink model."""
        checkpoint = self.config.checkpoint_path
        
        if not checkpoint or not os.path.exists(checkpoint):
            candidates = [
                Path(__file__).parent / "groundlink_checkpoints" / "pretrained_s7_noshape.tar",
                Path(__file__).parent / "groundlink_checkpoints" / "pretrained_s4_noshape.tar",
                Path(__file__).parent.parent / "models" / "groundlink" / "pretrained_s7_noshape.tar",
                Path.home() / ".cache" / "groundlink" / "pretrained_s7_noshape.tar",
            ]
            for c in candidates:
                if c.exists():
                    checkpoint = str(c)
                    break
        
        if checkpoint and os.path.exists(checkpoint):
            self.model = GroundLinkNet(checkpoint)
            if not self.model.is_ready:
                print(f"[GroundLink] Model load failed, will use fallback")
                self.model = None
    
    def process(self, mesh_sequence: Dict, images: Optional[List] = None, verbose: bool = True) -> Dict:
        """Process with fallback chain: GroundLink → TAPNet → Heuristic."""
        frames_data = mesh_sequence.get("frames", {})
        
        if isinstance(frames_data, dict):
            frame_keys = sorted(frames_data.keys())
            frames = [frames_data[k] for k in frame_keys]
        else:
            frames = frames_data
            frame_keys = None
        
        if not frames:
            return mesh_sequence
        
        # Initialize debug info
        self._debug_info = {
            "frames": len(frames),
            "methods_tried": [],
            "method_used": "none",
            "groundlink_error": None,
            "tapnet_error": None,
            "foot_positions": [],
            "pin_events": [],
        }
        
        contacts = None
        grf = None
        cop = None
        conversion_mode = "none"
        
        # === TRY 1: GroundLink (Physics-based) ===
        if self.model is not None and self.model.is_ready:
            self._debug_info["methods_tried"].append("groundlink")
            
            poses, root_trans, conversion_mode = convert_sam3d_to_groundlink(
                frames, up_axis=self.config.up_axis
            )
            
            if poses is not None:
                try:
                    contacts, grf, cop = self.model.predict_contacts(
                        poses,
                        grf_threshold=self.config.grf_threshold,
                        smooth_window=self.config.smooth_window
                    )
                    self._method_used = "groundlink"
                    
                    if verbose:
                        print(f"[GroundLink] ✓ Physics prediction: L={contacts[:, 0].sum():.0f} R={contacts[:, 1].sum():.0f} frames ({conversion_mode} mode)")
                        
                except Exception as e:
                    self._debug_info["groundlink_error"] = str(e)
                    if verbose:
                        print(f"[GroundLink] ✗ Physics prediction failed: {e}")
            else:
                self._debug_info["groundlink_error"] = "Could not extract pose data"
                if verbose:
                    print(f"[GroundLink] ✗ Could not extract pose data for physics model")
        else:
            error = self.model.load_error if self.model else "Model not loaded"
            self._debug_info["groundlink_error"] = error
            if verbose:
                print(f"[GroundLink] ✗ Physics model not available: {error}")
        
        # === TRY 2: TAPNet (Visual tracking) ===
        if contacts is None and self.config.fallback_to_tapnet and images is not None:
            self._debug_info["methods_tried"].append("tapnet")
            
            try:
                contacts = self._tapnet_contacts(frames, images)
                if contacts is not None:
                    self._method_used = "tapnet"
                    if verbose:
                        print(f"[GroundLink] ✓ TAPNet fallback: L={contacts[:, 0].sum():.0f} R={contacts[:, 1].sum():.0f} frames")
            except Exception as e:
                self._debug_info["tapnet_error"] = str(e)
                if verbose:
                    print(f"[GroundLink] ✗ TAPNet fallback failed: {e}")
        elif contacts is None and self.config.fallback_to_tapnet:
            self._debug_info["tapnet_error"] = "No images provided"
            if verbose:
                print(f"[GroundLink] → TAPNet skipped (no images provided)")
        
        # === TRY 3: Heuristic (Height + Velocity) ===
        if contacts is None and self.config.fallback_to_heuristic:
            self._debug_info["methods_tried"].append("heuristic")
            
            root_trans = self._extract_translations(frames)
            contacts = self._heuristic_contacts(frames, root_trans)
            
            if contacts is not None:
                self._method_used = "heuristic"
                if verbose:
                    print(f"[GroundLink] ✓ Heuristic fallback: L={contacts[:, 0].sum():.0f} R={contacts[:, 1].sum():.0f} frames")
            else:
                if verbose:
                    print(f"[GroundLink] ✗ Heuristic fallback failed")
        
        self._debug_info["method_used"] = self._method_used
        
        if contacts is None:
            if verbose:
                print(f"[GroundLink] ✗ All methods failed, returning original sequence")
            return mesh_sequence
        
        # === ENFORCE CONTACTS ===
        root_trans = self._extract_translations(frames)
        
        if self.config.pin_feet and root_trans is not None:
            adjusted_trans, pin_events = self._enforce_contacts_with_debug(frames, root_trans, contacts)
            self._debug_info["pin_events"] = pin_events
        else:
            adjusted_trans = root_trans
        
        # Build result
        result = mesh_sequence.copy()
        result = self._update_translations(result, frames, frame_keys, adjusted_trans)
        
        result["foot_contacts"] = {
            "method": self._method_used,
            "conversion_mode": conversion_mode,
            "contacts": contacts.tolist(),
            "grf": grf.tolist() if grf is not None else None,
            "cop": cop.tolist() if cop is not None else None,
            "debug": self._debug_info,
        }
        
        return result
    
    def _extract_translations(self, frames: List[Dict]) -> Optional[np.ndarray]:
        """Extract root translations from frames."""
        translations = []
        for frame in frames:
            tr = frame.get('pred_cam_t') or frame.get('smpl_params', {}).get('transl')
            if tr is None:
                return None
            translations.append(np.array(tr).flatten()[:3])
        return np.stack(translations)
    
    def _tapnet_contacts(self, frames: List[Dict], images: List) -> Optional[np.ndarray]:
        """Use TAPNet for contact detection (placeholder - integrate with foot_tracker.py)."""
        # TODO: Integrate with existing foot_tracker.py
        return None
    
    def _heuristic_contacts(self, frames: List[Dict], root_trans: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Heuristic contact detection using height + velocity."""
        cfg = self.config
        T = len(frames)
        
        left_foot = []
        right_foot = []
        
        for i, frame in enumerate(frames):
            joints = frame.get('pred_joints_3d') or frame.get('pred_joints')
            if joints is None:
                return None
            
            joints = np.array(joints)
            if joints.ndim == 1:
                joints = joints.reshape(-1, 3)
            
            if len(joints) <= max(cfg.left_foot_idx, cfg.right_foot_idx):
                return None
            
            trans = root_trans[i] if root_trans is not None else np.zeros(3)
            left_foot.append(joints[cfg.left_foot_idx] + trans)
            right_foot.append(joints[cfg.right_foot_idx] + trans)
        
        left_foot = np.stack(left_foot)
        right_foot = np.stack(right_foot)
        
        # Store for debug visualization
        self._debug_info["foot_positions"] = {
            "left": left_foot.tolist(),
            "right": right_foot.tolist(),
        }
        
        up_idx = 1 if cfg.up_axis == 'y' else 2
        
        left_h = left_foot[:, up_idx]
        right_h = right_foot[:, up_idx]
        ground = min(np.percentile(left_h, 5), np.percentile(right_h, 5))
        
        left_on_ground = left_h < (ground + cfg.height_threshold)
        right_on_ground = right_h < (ground + cfg.height_threshold)
        
        left_vel = np.linalg.norm(np.diff(left_foot, axis=0, prepend=left_foot[:1]), axis=1)
        right_vel = np.linalg.norm(np.diff(right_foot, axis=0, prepend=right_foot[:1]), axis=1)
        
        left_stationary = left_vel < cfg.velocity_threshold
        right_stationary = right_vel < cfg.velocity_threshold
        
        left_contact = left_on_ground & left_stationary
        right_contact = right_on_ground & right_stationary
        
        if cfg.smooth_window > 1:
            left_contact = medfilt(left_contact.astype(float), cfg.smooth_window) > 0.5
            right_contact = medfilt(right_contact.astype(float), cfg.smooth_window) > 0.5
        
        return np.stack([left_contact, right_contact], axis=1)
    
    def _enforce_contacts_with_debug(self, frames: List[Dict], root_trans: np.ndarray, 
                                      contacts: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Adjust root translation to enforce contacts, with debug info."""
        cfg = self.config
        T = len(frames)
        
        adjusted = root_trans.copy()
        pin_events = []
        
        # Extract foot positions
        foot_positions = [[], []]
        for frame in frames:
            joints = np.array(frame.get('pred_joints_3d') or frame.get('pred_joints'))
            if joints.ndim == 1:
                joints = joints.reshape(-1, 3)
            foot_positions[0].append(joints[cfg.left_foot_idx])
            foot_positions[1].append(joints[cfg.right_foot_idx])
        
        foot_positions = [np.stack(fp) for fp in foot_positions]
        pin_positions = [None, None]
        foot_names = ["left", "right"]
        
        for t in range(T):
            for foot_idx in range(2):
                if contacts[t, foot_idx]:
                    foot_world = foot_positions[foot_idx][t] + adjusted[t]
                    
                    if pin_positions[foot_idx] is None:
                        # Start new pin
                        pin_positions[foot_idx] = foot_world.copy()
                        pin_events.append({
                            "frame": t,
                            "foot": foot_names[foot_idx],
                            "event": "pin_start",
                            "position": foot_world.tolist(),
                        })
                    else:
                        # Enforce existing pin
                        adjustment = pin_positions[foot_idx] - foot_world
                        if cfg.up_axis == 'y':
                            adjustment[1] = 0  # Don't adjust vertical
                        else:
                            adjustment[2] = 0
                        adjusted[t] += adjustment * 0.5
                else:
                    if pin_positions[foot_idx] is not None:
                        # Release pin
                        pin_events.append({
                            "frame": t,
                            "foot": foot_names[foot_idx],
                            "event": "pin_end",
                            "position": pin_positions[foot_idx].tolist(),
                        })
                    pin_positions[foot_idx] = None
        
        # Smooth
        adjusted = gaussian_filter1d(adjusted, sigma=1.0, axis=0)
        
        return adjusted, pin_events
    
    def _update_translations(self, result: Dict, frames: List[Dict], 
                            frame_keys: Optional[List], adjusted_trans: Optional[np.ndarray]) -> Dict:
        """Update mesh sequence with adjusted translations."""
        if adjusted_trans is None:
            return result
        
        updated_frames = []
        for i, frame in enumerate(frames):
            new_frame = frame.copy()
            
            if "pred_cam_t" in new_frame:
                new_frame["pred_cam_t"] = adjusted_trans[i].tolist()
            
            if "smpl_params" in new_frame:
                new_frame["smpl_params"] = new_frame["smpl_params"].copy()
                new_frame["smpl_params"]["transl"] = adjusted_trans[i].tolist()
            
            updated_frames.append(new_frame)
        
        if frame_keys is not None:
            result["frames"] = {k: v for k, v in zip(frame_keys, updated_frames)}
        else:
            result["frames"] = updated_frames
        
        return result


# =============================================================================
# ComfyUI Nodes
# =============================================================================

class GroundLinkSolverNode:
    """
    Physics-based foot contact solver using GroundLink.
    
    Fallback chain: GroundLink (physics) → TAPNet (visual) → Heuristic
    FULLY AUTOMATED - no manual intervention required.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "images": ("IMAGE",),
                "checkpoint_path": ("STRING", {"default": ""}),
                "grf_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                "smooth_window": ("INT", {"default": 5, "min": 1, "max": 15, "step": 2}),
                "pin_feet": ("BOOLEAN", {"default": True}),
                "up_axis": (["y", "z"], {"default": "y"}),
                "fallback_to_heuristic": ("BOOLEAN", {"default": True}),
                "fallback_to_tapnet": ("BOOLEAN", {"default": True}),
                "log_level": (["Normal", "Verbose", "Silent"], {"default": "Normal"}),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING", "FOOT_CONTACTS")
    RETURN_NAMES = ("mesh_sequence", "status", "foot_contacts")
    FUNCTION = "solve"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def solve(self, mesh_sequence: Dict, images=None, checkpoint_path: str = "", 
              grf_threshold: float = 0.1, smooth_window: int = 5, pin_feet: bool = True,
              up_axis: str = "y", fallback_to_heuristic: bool = True, 
              fallback_to_tapnet: bool = True, log_level: str = "Normal") -> Tuple[Dict, str, Dict]:
        
        verbose = log_level != "Silent"
        
        config = GroundLinkConfig(
            checkpoint_path=checkpoint_path,
            grf_threshold=grf_threshold,
            smooth_window=smooth_window,
            pin_feet=pin_feet,
            up_axis=up_axis,
            fallback_to_heuristic=fallback_to_heuristic,
            fallback_to_tapnet=fallback_to_tapnet,
        )
        
        enforcer = GroundLinkContactEnforcer(config)
        
        # Convert images if provided
        image_list = None
        if images is not None:
            try:
                image_list = [img for img in images]
            except:
                pass
        
        result = enforcer.process(mesh_sequence, images=image_list, verbose=verbose)
        
        foot_contacts = result.get("foot_contacts", {})
        method = foot_contacts.get("method", "none")
        contacts = foot_contacts.get("contacts", [])
        debug = foot_contacts.get("debug", {})
        
        # Build status message
        status_lines = []
        status_lines.append(f"Method: {method}")
        
        if contacts:
            arr = np.array(contacts)
            status_lines.append(f"Contacts: L={arr[:, 0].sum():.0f} R={arr[:, 1].sum():.0f} frames")
        
        methods_tried = debug.get("methods_tried", [])
        if "groundlink" in methods_tried and debug.get("groundlink_error"):
            status_lines.append(f"GroundLink: {debug['groundlink_error']}")
        if "tapnet" in methods_tried and debug.get("tapnet_error"):
            status_lines.append(f"TAPNet: {debug['tapnet_error']}")
        
        pin_events = debug.get("pin_events", [])
        if pin_events:
            status_lines.append(f"Pin events: {len(pin_events)}")
        
        status = " | ".join(status_lines)
        
        return (result, status, foot_contacts)


class GroundLinkContactVisualizer:
    """Visualize and debug foot contact detection."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foot_contacts": ("FOOT_CONTACTS",),
            },
            "optional": {
                "output_format": (["summary", "timeline", "debug", "json"], {"default": "summary"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "SAM3DBody2abc/Visualization"
    
    def visualize(self, foot_contacts: Dict, output_format: str = "summary") -> Tuple[str]:
        import json
        
        contacts = foot_contacts.get("contacts", [])
        method = foot_contacts.get("method", "unknown")
        debug = foot_contacts.get("debug", {})
        
        if output_format == "json":
            return (json.dumps(foot_contacts, indent=2),)
        
        if not contacts:
            return ("No contact data",)
        
        arr = np.array(contacts)
        T = len(arr)
        
        if output_format == "summary":
            lines = [
                "=" * 50,
                "FOOT CONTACT SUMMARY",
                "=" * 50,
                f"Method used: {method}",
                f"Total frames: {T}",
                f"Left foot contacts: {arr[:, 0].sum():.0f} frames ({100*arr[:, 0].mean():.1f}%)",
                f"Right foot contacts: {arr[:, 1].sum():.0f} frames ({100*arr[:, 1].mean():.1f}%)",
                "",
                "Methods tried: " + " → ".join(debug.get("methods_tried", [])),
            ]
            
            if debug.get("groundlink_error"):
                lines.append(f"GroundLink error: {debug['groundlink_error']}")
            if debug.get("tapnet_error"):
                lines.append(f"TAPNet error: {debug['tapnet_error']}")
            
            return ("\n".join(lines),)
        
        elif output_format == "timeline":
            lines = [
                "CONTACT TIMELINE",
                "-" * 50,
                "Frame | Left | Right | Events",
                "-" * 50
            ]
            
            pin_events = {e["frame"]: e for e in debug.get("pin_events", [])}
            
            for t in range(T):
                if t == 0 or (arr[t] != arr[t-1]).any() or t == T-1 or t in pin_events:
                    l = "████" if arr[t, 0] else "····"
                    r = "████" if arr[t, 1] else "····"
                    event = ""
                    if t in pin_events:
                        e = pin_events[t]
                        event = f" ← {e['foot']} {e['event']}"
                    lines.append(f"{t:5d} | {l} | {r} |{event}")
            
            return ("\n".join(lines),)
        
        else:  # debug
            lines = [
                "=" * 50,
                "DEBUG INFO",
                "=" * 50,
                f"Frames: {debug.get('frames', T)}",
                f"Method: {method}",
                f"Methods tried: {debug.get('methods_tried', [])}",
                "",
                "Errors:",
                f"  GroundLink: {debug.get('groundlink_error', 'None')}",
                f"  TAPNet: {debug.get('tapnet_error', 'None')}",
                "",
                f"Pin events: {len(debug.get('pin_events', []))}",
            ]
            
            for event in debug.get("pin_events", [])[:20]:  # Show first 20
                lines.append(f"  Frame {event['frame']}: {event['foot']} {event['event']}")
            
            if len(debug.get("pin_events", [])) > 20:
                lines.append(f"  ... and {len(debug['pin_events']) - 20} more")
            
            return ("\n".join(lines),)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "GroundLinkSolver": GroundLinkSolverNode,
    "GroundLinkContactVisualizer": GroundLinkContactVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundLinkSolver": "⚡ GroundLink Foot Contact (Physics)",
    "GroundLinkContactVisualizer": "📊 GroundLink Contact Visualizer",
}
