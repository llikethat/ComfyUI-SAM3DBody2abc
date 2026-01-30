"""
GroundLink Physics-Based Foot Contact Solver
============================================

MIT Licensed physics-based foot contact detection using GroundLink neural network.
PRIMARY foot contact solver with TAPNet and Heuristic fallbacks.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

torch = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


# =============================================================================
# GroundLink Neural Network
# =============================================================================

def build_groundlink_model():
    """Build DeepNetwork matching GroundLink's exact architecture."""
    torch = _ensure_torch()
    import torch.nn as nn
    
    n_joints = 23
    cnn_features = [3 * n_joints, 128, 128, 256, 256]
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
    layers.append(nn.Flatten(start_dim=2, end_dim=-1))
    layers.append(TransposeModule(-2, -1))
    
    for c_in, c_out in zip(cnn_features[:-1], cnn_features[1:]):
        layers.append(nn.Dropout(p=cnn_dropout))
        layers.append(nn.Conv1d(c_in, c_out, cnn_kernel, padding=cnn_kernel//2, padding_mode='replicate'))
        layers.append(nn.ELU())
    
    layers.append(TransposeModule(-2, -1))
    
    for _ in range(fc_depth - 1):
        layers.append(nn.Dropout(p=fc_dropout))
        layers.append(nn.Linear(cnn_features[-1], cnn_features[-1]))
        layers.append(nn.ELU())
    
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
        grf, cop = self.predict(poses)
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


# =============================================================================
# Debug Helper
# =============================================================================

def debug_frame_structure(frame: Dict, frame_idx: int = 0) -> str:
    """Generate debug info about frame structure."""
    lines = [f"Frame {frame_idx} structure:"]
    
    def describe(obj, prefix=""):
        if isinstance(obj, dict):
            return f"dict with keys: {list(obj.keys())}"
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 0:
                return f"list[{len(obj)}] of {type(obj[0]).__name__}"
            return f"empty list"
        elif isinstance(obj, np.ndarray):
            return f"ndarray shape={obj.shape} dtype={obj.dtype}"
        else:
            return f"{type(obj).__name__}"
    
    for key, value in frame.items():
        lines.append(f"  {key}: {describe(value)}")
        if isinstance(value, dict):
            for k2, v2 in value.items():
                lines.append(f"    {k2}: {describe(v2)}")
    
    return "\n".join(lines)


# =============================================================================
# Pose Conversion - More Flexible
# =============================================================================

def extract_joints_flexible(frame: Dict) -> Optional[np.ndarray]:
    """Extract 3D joints from frame using multiple possible keys."""
    # Try various possible keys for joint data
    joint_keys = [
        'pred_joints_3d',
        'pred_joints',
        'joints_3d',
        'joints',
        'keypoints_3d',
        'keypoints',
        'joint_coords',
    ]
    
    for key in joint_keys:
        if key in frame:
            joints = frame[key]
            if joints is not None:
                joints = np.array(joints)
                if joints.ndim == 1:
                    # Flatten array, try to reshape
                    if len(joints) >= 66:  # At least 22 joints * 3
                        joints = joints[:66].reshape(-1, 3)
                    else:
                        continue
                if joints.ndim == 2 and joints.shape[1] == 3 and joints.shape[0] >= 12:
                    return joints
    
    # Try nested in smpl_params
    smpl = frame.get('smpl_params', {})
    if isinstance(smpl, dict):
        for key in ['joints', 'joints_3d', 'pred_joints']:
            if key in smpl:
                joints = np.array(smpl[key])
                if joints.ndim == 1 and len(joints) >= 66:
                    joints = joints[:66].reshape(-1, 3)
                if joints.ndim == 2 and joints.shape[1] == 3 and joints.shape[0] >= 12:
                    return joints
    
    # Try mesh_data format (from SAM3DBody)
    mesh_data = frame.get('mesh_data', {})
    if isinstance(mesh_data, dict):
        for key in ['joint_coords', 'joints', 'joints_3d']:
            if key in mesh_data:
                joints = np.array(mesh_data[key])
                if joints.ndim == 1 and len(joints) >= 66:
                    joints = joints[:66].reshape(-1, 3)
                if joints.ndim == 2 and joints.shape[1] == 3 and joints.shape[0] >= 12:
                    return joints
    
    return None


def extract_translation_flexible(frame: Dict) -> Optional[np.ndarray]:
    """Extract translation from frame using multiple possible keys."""
    trans_keys = [
        'pred_cam_t',
        'transl',
        'translation',
        'trans',
        'root_trans',
        'pelvis_trans',
    ]
    
    for key in trans_keys:
        if key in frame:
            tr = frame[key]
            if tr is not None:
                tr = np.array(tr).flatten()
                if len(tr) >= 3:
                    return tr[:3]
    
    # Try nested
    smpl = frame.get('smpl_params', {})
    if isinstance(smpl, dict):
        for key in trans_keys:
            if key in smpl:
                tr = np.array(smpl[key]).flatten()
                if len(tr) >= 3:
                    return tr[:3]
    
    return None


def convert_to_groundlink_poses(frames: List[Dict], up_axis: str = 'y', verbose: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Convert frames to GroundLink format with flexible extraction."""
    T = len(frames)
    
    if T == 0:
        return None, None, "no_frames"
    
    # Debug first frame
    if verbose:
        print(f"[GroundLink] {debug_frame_structure(frames[0], 0)}")
    
    # Extract joints for all frames
    all_joints = []
    all_trans = []
    
    for i, frame in enumerate(frames):
        joints = extract_joints_flexible(frame)
        if joints is None:
            if verbose and i == 0:
                print(f"[GroundLink] Could not extract joints from frame {i}")
            return None, None, "no_joints"
        
        trans = extract_translation_flexible(frame)
        if trans is None:
            trans = joints[0] if len(joints) > 0 else np.zeros(3)
        
        all_joints.append(joints)
        all_trans.append(trans)
    
    all_joints = np.stack(all_joints)  # [T, J, 3]
    all_trans = np.stack(all_trans)    # [T, 3]
    
    if verbose:
        print(f"[GroundLink] Extracted joints: {all_joints.shape}, trans: {all_trans.shape}")
    
    # Ensure we have at least 22 joints
    n_joints = all_joints.shape[1]
    if n_joints < 22:
        if verbose:
            print(f"[GroundLink] Only {n_joints} joints, need at least 22")
        return None, None, f"insufficient_joints_{n_joints}"
    
    # Build GroundLink format [T, 23, 3]
    # Slot 0: pelvis translation (local)
    # Slots 1-22: joint positions relative to pelvis
    
    pelvis = all_joints[:, 0, :]
    local_joints = all_joints[:, :22, :] - pelvis[:, np.newaxis, :]
    
    poses = np.zeros((T, 23, 3))
    poses[:, 0, :] = np.zeros((T, 3))  # Local pelvis at origin
    poses[:, 1:23, :] = local_joints
    
    return poses, all_trans, "positions"


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
        checkpoint = self.config.checkpoint_path
        
        if not checkpoint or not os.path.exists(checkpoint):
            candidates = [
                Path(__file__).parent / "groundlink_checkpoints" / "pretrained_s7_noshape.tar",
                Path(__file__).parent / "groundlink_checkpoints" / "pretrained_s4_noshape.tar",
            ]
            for c in candidates:
                if c.exists():
                    checkpoint = str(c)
                    break
        
        if checkpoint and os.path.exists(checkpoint):
            self.model = GroundLinkNet(checkpoint)
            if not self.model.is_ready:
                self.model = None
    
    def _extract_frames(self, mesh_sequence: Dict) -> Tuple[List[Dict], Optional[List]]:
        """Extract frames from mesh_sequence in various formats."""
        frames_data = mesh_sequence.get("frames", mesh_sequence)
        
        if isinstance(frames_data, dict):
            # Could be {0: frame, 1: frame, ...} or {"frame_0": frame, ...}
            if all(isinstance(k, int) or k.isdigit() for k in frames_data.keys()):
                frame_keys = sorted(frames_data.keys(), key=lambda x: int(x) if isinstance(x, str) else x)
                frames = [frames_data[k] for k in frame_keys]
                return frames, frame_keys
            else:
                # Maybe the whole dict is a single frame?
                # Check if it has joint data
                if extract_joints_flexible(frames_data) is not None:
                    return [frames_data], None
                # Otherwise assume it's keyed frames
                frame_keys = sorted(frames_data.keys())
                frames = [frames_data[k] for k in frame_keys]
                return frames, frame_keys
        elif isinstance(frames_data, list):
            return frames_data, None
        else:
            return [], None
    
    def process(self, mesh_sequence: Dict, images: Optional[List] = None, verbose: bool = True) -> Dict:
        """Process with fallback chain: GroundLink → TAPNet → Heuristic."""
        
        frames, frame_keys = self._extract_frames(mesh_sequence)
        
        if not frames:
            if verbose:
                print(f"[GroundLink] No frames found in mesh_sequence")
                print(f"[GroundLink] mesh_sequence keys: {list(mesh_sequence.keys()) if isinstance(mesh_sequence, dict) else type(mesh_sequence)}")
            return mesh_sequence
        
        if verbose:
            print(f"[GroundLink] Processing {len(frames)} frames")
        
        self._debug_info = {
            "frames": len(frames),
            "methods_tried": [],
            "method_used": "none",
            "groundlink_error": None,
            "tapnet_error": None,
            "pin_events": [],
        }
        
        contacts = None
        grf = None
        cop = None
        conversion_mode = "none"
        
        # Check if groundlink is enabled
        use_groundlink = getattr(self, '_use_groundlink', True)
        
        # === TRY 1: GroundLink ===
        if use_groundlink and self.model is not None and self.model.is_ready:
            self._debug_info["methods_tried"].append("groundlink")
            
            poses, root_trans, conversion_mode = convert_to_groundlink_poses(
                frames, up_axis=self.config.up_axis, verbose=verbose
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
                        print(f"[GroundLink] ✓ Physics: L={contacts[:, 0].sum():.0f} R={contacts[:, 1].sum():.0f} frames")
                        
                except Exception as e:
                    self._debug_info["groundlink_error"] = str(e)
                    if verbose:
                        print(f"[GroundLink] ✗ Physics failed: {e}")
            else:
                self._debug_info["groundlink_error"] = f"Pose extraction failed: {conversion_mode}"
                if verbose:
                    print(f"[GroundLink] ✗ Could not extract pose data: {conversion_mode}")
        elif not use_groundlink:
            self._debug_info["groundlink_error"] = "Disabled by user"
            if verbose:
                print(f"[GroundLink] → GroundLink disabled by user")
        else:
            error = self.model.load_error if self.model else "Model not loaded"
            self._debug_info["groundlink_error"] = error
            if verbose:
                print(f"[GroundLink] ✗ Physics model unavailable: {error}")
        
        # === TRY 2: TAPNet ===
        if contacts is None and self.config.fallback_to_tapnet:
            self._debug_info["methods_tried"].append("tapnet")
            if images is not None:
                try:
                    contacts = self._run_tapnet(images, mesh_sequence, verbose)
                    if contacts is not None:
                        self._method_used = "tapnet"
                        if verbose:
                            print(f"[GroundLink] ✓ TAPNet: L={contacts[:, 0].sum():.0f} R={contacts[:, 1].sum():.0f} frames")
                    else:
                        self._debug_info["tapnet_error"] = "TAPNet returned no contacts"
                        if verbose:
                            print(f"[GroundLink] ✗ TAPNet returned no contacts")
                except Exception as e:
                    self._debug_info["tapnet_error"] = str(e)
                    if verbose:
                        print(f"[GroundLink] ✗ TAPNet failed: {e}")
            else:
                self._debug_info["tapnet_error"] = "No images provided"
                if verbose:
                    print(f"[GroundLink] → TAPNet skipped (no images)")
        
        # === TRY 3: Heuristic ===
        if contacts is None and self.config.fallback_to_heuristic:
            self._debug_info["methods_tried"].append("heuristic")
            
            contacts, root_trans = self._heuristic_contacts(frames, verbose)
            
            if contacts is not None:
                self._method_used = "heuristic"
                if verbose:
                    print(f"[GroundLink] ✓ Heuristic: L={contacts[:, 0].sum():.0f} R={contacts[:, 1].sum():.0f} frames")
            else:
                if verbose:
                    print(f"[GroundLink] ✗ Heuristic failed")
        
        self._debug_info["method_used"] = self._method_used
        
        if contacts is None:
            if verbose:
                print(f"[GroundLink] ✗ All methods failed")
            return mesh_sequence
        
        # === ENFORCE ===
        if self.config.pin_feet and root_trans is not None:
            adjusted_trans, pin_events = self._enforce_contacts(frames, root_trans, contacts)
            self._debug_info["pin_events"] = pin_events
        else:
            adjusted_trans = root_trans
        
        result = mesh_sequence.copy()
        result = self._update_frames(result, frames, frame_keys, adjusted_trans)
        
        result["foot_contacts"] = {
            "method": self._method_used,
            "conversion_mode": conversion_mode,
            "contacts": contacts.tolist(),
            "grf": grf.tolist() if grf is not None else None,
            "cop": cop.tolist() if cop is not None else None,
            "debug": self._debug_info,
        }
        
        return result
    
    def _run_tapnet(self, images: List, mesh_sequence: Dict, verbose: bool) -> Optional[np.ndarray]:
        """Run TAPNet/FootTracker for contact detection."""
        try:
            # Import FootTrackerCore from foot_tracker
            from .foot_tracker import FootTrackerCore
        except ImportError:
            try:
                # Try alternate import path
                import sys
                import os
                module_dir = os.path.dirname(os.path.abspath(__file__))
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                from foot_tracker import FootTrackerCore
            except ImportError:
                if verbose:
                    print(f"[GroundLink] FootTrackerCore not available")
                return None
        
        torch = _ensure_torch()
        
        # Convert images to tensor
        if isinstance(images, list):
            images_np = np.stack([np.array(img) for img in images])
        elif isinstance(images, np.ndarray):
            images_np = images
        elif hasattr(images, 'cpu'):  # torch tensor
            images_np = images.cpu().numpy()
        else:
            if verbose:
                print(f"[GroundLink] Unknown image format: {type(images)}")
            return None
        
        # Ensure proper shape [N, H, W, C]
        if images_np.ndim == 3:
            images_np = images_np[np.newaxis, ...]
        
        if verbose:
            print(f"[GroundLink] TAPNet input: {images_np.shape}")
        
        # Create tracker and run
        tracker = FootTrackerCore()
        
        images_tensor = torch.from_numpy(images_np)
        if images_tensor.dtype != torch.float32:
            images_tensor = images_tensor.float()
        if images_tensor.max() > 1.0:
            images_tensor = images_tensor / 255.0
        
        result = tracker.track_feet(
            images=images_tensor,
            mesh_sequence=mesh_sequence,
            use_tapir=True,
            output_debug_video=False,
        )
        
        if result is None:
            return None
        
        _, result_seq, _ = result
        contact_states = result_seq.get("foot_contact", {}).get("contact_states", [])
        
        if not contact_states:
            return None
        
        # Convert contact states to binary array [T, 2]
        T = len(contact_states)
        contacts = np.zeros((T, 2), dtype=bool)
        
        for i, state in enumerate(contact_states):
            if state in ["left", "both"]:
                contacts[i, 0] = True
            if state in ["right", "both"]:
                contacts[i, 1] = True
        
        return contacts
    
    def _heuristic_contacts(self, frames: List[Dict], verbose: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Heuristic contact detection."""
        cfg = self.config
        T = len(frames)
        
        left_foot = []
        right_foot = []
        translations = []
        
        for frame in frames:
            joints = extract_joints_flexible(frame)
            if joints is None:
                if verbose:
                    print(f"[GroundLink] Heuristic: no joints in frame")
                return None, None
            
            trans = extract_translation_flexible(frame)
            if trans is None:
                trans = np.zeros(3)
            
            n_joints = len(joints)
            left_idx = min(cfg.left_foot_idx, n_joints - 1)
            right_idx = min(cfg.right_foot_idx, n_joints - 1)
            
            left_foot.append(joints[left_idx] + trans)
            right_foot.append(joints[right_idx] + trans)
            translations.append(trans)
        
        left_foot = np.stack(left_foot)
        right_foot = np.stack(right_foot)
        translations = np.stack(translations)
        
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
        
        return np.stack([left_contact, right_contact], axis=1), translations
    
    def _enforce_contacts(self, frames: List[Dict], root_trans: np.ndarray, 
                          contacts: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Adjust root translation to pin feet during contacts."""
        cfg = self.config
        T = len(frames)
        
        adjusted = root_trans.copy()
        pin_events = []
        
        foot_positions = [[], []]
        for frame in frames:
            joints = extract_joints_flexible(frame)
            n_joints = len(joints)
            left_idx = min(cfg.left_foot_idx, n_joints - 1)
            right_idx = min(cfg.right_foot_idx, n_joints - 1)
            foot_positions[0].append(joints[left_idx])
            foot_positions[1].append(joints[right_idx])
        
        foot_positions = [np.stack(fp) for fp in foot_positions]
        pin_positions = [None, None]
        foot_names = ["left", "right"]
        
        for t in range(T):
            for foot_idx in range(2):
                if contacts[t, foot_idx]:
                    foot_world = foot_positions[foot_idx][t] + adjusted[t]
                    
                    if pin_positions[foot_idx] is None:
                        pin_positions[foot_idx] = foot_world.copy()
                        pin_events.append({
                            "frame": t, "foot": foot_names[foot_idx],
                            "event": "pin_start", "position": foot_world.tolist(),
                        })
                    else:
                        adjustment = pin_positions[foot_idx] - foot_world
                        if cfg.up_axis == 'y':
                            adjustment[1] = 0
                        else:
                            adjustment[2] = 0
                        adjusted[t] += adjustment * 0.5
                else:
                    if pin_positions[foot_idx] is not None:
                        pin_events.append({
                            "frame": t, "foot": foot_names[foot_idx],
                            "event": "pin_end", "position": pin_positions[foot_idx].tolist(),
                        })
                    pin_positions[foot_idx] = None
        
        adjusted = gaussian_filter1d(adjusted, sigma=1.0, axis=0)
        return adjusted, pin_events
    
    def _update_frames(self, result: Dict, frames: List[Dict], 
                       frame_keys: Optional[List], adjusted_trans: Optional[np.ndarray]) -> Dict:
        """Update frames with adjusted translations."""
        if adjusted_trans is None:
            return result
        
        updated_frames = []
        for i, frame in enumerate(frames):
            new_frame = frame.copy()
            
            if "pred_cam_t" in new_frame:
                new_frame["pred_cam_t"] = adjusted_trans[i].tolist()
            if "smpl_params" in new_frame and isinstance(new_frame["smpl_params"], dict):
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
    """Physics-based foot contact solver with fallback chain."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "images": ("IMAGE",),
                "use_groundlink": ("BOOLEAN", {"default": True, "tooltip": "Use GroundLink physics-based detection (primary)"}),
                "use_tapnet": ("BOOLEAN", {"default": True, "tooltip": "Use TAPNet visual tracking (fallback or primary if GroundLink disabled)"}),
                "use_heuristic": ("BOOLEAN", {"default": True, "tooltip": "Use heuristic height+velocity (final fallback)"}),
                "checkpoint_path": ("STRING", {"default": ""}),
                "grf_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                "smooth_window": ("INT", {"default": 5, "min": 1, "max": 15, "step": 2}),
                "pin_feet": ("BOOLEAN", {"default": True}),
                "up_axis": (["y", "z"], {"default": "y"}),
                "log_level": (["Normal", "Verbose", "Silent"], {"default": "Verbose"}),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING", "FOOT_CONTACTS")
    RETURN_NAMES = ("mesh_sequence", "status", "foot_contacts")
    FUNCTION = "solve"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def solve(self, mesh_sequence: Dict, images=None, 
              use_groundlink: bool = True, use_tapnet: bool = True, use_heuristic: bool = True,
              checkpoint_path: str = "", grf_threshold: float = 0.1, smooth_window: int = 5, 
              pin_feet: bool = True, up_axis: str = "y", 
              log_level: str = "Verbose") -> Tuple[Dict, str, Dict]:
        
        verbose = log_level != "Silent"
        
        config = GroundLinkConfig(
            checkpoint_path=checkpoint_path,
            grf_threshold=grf_threshold,
            smooth_window=smooth_window,
            pin_feet=pin_feet,
            up_axis=up_axis,
            fallback_to_heuristic=use_heuristic,
            fallback_to_tapnet=use_tapnet,
        )
        
        # Pass use_groundlink flag to enforcer
        enforcer = GroundLinkContactEnforcer(config)
        enforcer._use_groundlink = use_groundlink
        
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
        
        status_parts = [f"Method: {method}"]
        if contacts:
            arr = np.array(contacts)
            status_parts.append(f"L={arr[:, 0].sum():.0f} R={arr[:, 1].sum():.0f}")
        
        pin_events = debug.get("pin_events", [])
        if pin_events:
            status_parts.append(f"Pins: {len(pin_events)}")
        
        status = " | ".join(status_parts)
        
        return (result, status, foot_contacts)


class GroundLinkContactVisualizer:
    """Visualize foot contact detection results."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"foot_contacts": ("FOOT_CONTACTS",)},
            "optional": {"output_format": (["summary", "timeline", "debug", "json"], {"default": "summary"})}
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
                f"Method: {method}",
                f"Frames: {T}",
                f"Left contacts: {arr[:, 0].sum():.0f} ({100*arr[:, 0].mean():.1f}%)",
                f"Right contacts: {arr[:, 1].sum():.0f} ({100*arr[:, 1].mean():.1f}%)",
                "",
                f"Methods tried: {' → '.join(debug.get('methods_tried', []))}",
            ]
            
            if debug.get("groundlink_error"):
                lines.append(f"GroundLink: {debug['groundlink_error']}")
            if debug.get("tapnet_error"):
                lines.append(f"TAPNet: {debug['tapnet_error']}")
            
            pin_events = debug.get("pin_events", [])
            if pin_events:
                lines.append(f"\nPin events: {len(pin_events)}")
            
            return ("\n".join(lines),)
        
        elif output_format == "timeline":
            lines = ["CONTACT TIMELINE", "-" * 50, "Frame | L | R | Event", "-" * 50]
            
            pin_dict = {}
            for e in debug.get("pin_events", []):
                pin_dict[e["frame"]] = e
            
            for t in range(T):
                show = t == 0 or t == T-1 or (arr[t] != arr[t-1]).any() or t in pin_dict
                if show:
                    l = "████" if arr[t, 0] else "····"
                    r = "████" if arr[t, 1] else "····"
                    event = ""
                    if t in pin_dict:
                        e = pin_dict[t]
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
            
            for e in debug.get("pin_events", [])[:20]:
                lines.append(f"  Frame {e['frame']}: {e['foot']} {e['event']}")
            
            if len(debug.get("pin_events", [])) > 20:
                lines.append(f"  ... +{len(debug['pin_events']) - 20} more")
            
            return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "GroundLinkSolver": GroundLinkSolverNode,
    "GroundLinkContactVisualizer": GroundLinkContactVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroundLinkSolver": "⚡ GroundLink Foot Contact (Physics)",
    "GroundLinkContactVisualizer": "📊 GroundLink Contact Visualizer",
}
