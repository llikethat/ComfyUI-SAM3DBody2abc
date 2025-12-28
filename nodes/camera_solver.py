"""
Camera Solver Node for SAM3DBody2abc

Comprehensive camera solving with automatic shot type detection and quality modes.

Shot Types:
- Auto: Automatically detect shot type
- Static: No camera motion (identity matrices)
- Nodal: Rotation only (tripod pan/tilt)
- Parallax: Translation present (handheld/dolly/crane)
- Hybrid: Multiple transitions in one shot

Quality Modes:
- Fast: KLT/LightGlue only
- Balanced: With LoFTR fallback
- Best: LoFTR always

Feature Matchers:
- KLT: Fast sparse optical flow (CPU)
- ORB: Feature descriptor matching (CPU)
- LightGlue: AI-based matching (GPU) - Apache 2.0
- LoFTR: Detector-free matching (GPU) - Apache 2.0

External Import:
- JSON files from PFTrack, 3DEqualizer, Maya, SynthEyes, Nuke
"""

import numpy as np
import torch
import cv2
import json
import os
import math
from typing import Dict, Tuple, Any, Optional, List

# Check for COLMAP availability
COLMAP_AVAILABLE = False
try:
    import sys
    _custom_nodes = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _custom_nodes not in sys.path:
        sys.path.insert(0, _custom_nodes)
    # Try to import from ComfyUI-COLMAP
    from ComfyUI_COLMAP.colmap_solver import run_colmap_reconstruction
    COLMAP_AVAILABLE = True
    print("[CameraSolver] COLMAP integration available")
except ImportError:
    print("[CameraSolver] COLMAP not found - parallax solving will use Essential Matrix fallback")
except Exception as e:
    print(f"[CameraSolver] COLMAP import error: {e}")


class CameraSolver:
    """
    Comprehensive camera solver with automatic shot detection and quality modes.
    
    Masking Priority:
    1. SAM3 masks via foreground_masks input (best quality, recommended)
    2. YOLO auto-detection fallback if auto_mask_people=True
    3. No masking if neither provided
    
    Solving Methods:
    - Rotation Only: Homography decomposition (tripod/nodal shots)
    - Full 6-DOF: COLMAP or Essential Matrix fallback (handheld/dolly)
    """
    
    SOLVING_METHODS = ["Auto (Recommended)", "Rotation Only (Tripod)", "Full 6-DOF (Handheld)"]
    QUALITY_MODES = ["Fast", "Balanced", "Best"]
    TRANSLATION_SOLVERS = ["COLMAP", "Essential Matrix"]
    COLMAP_QUALITY = ["Low", "Medium", "High"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "solving_method": (cls.SOLVING_METHODS, {
                    "default": "Auto (Recommended)",
                    "tooltip": "Auto: detect shot type automatically. Rotation Only: tripod pan/tilt. Full 6-DOF: handheld with translation."
                }),
            },
            "optional": {
                # === Masking (SAM3 preferred, YOLO fallback) ===
                "foreground_masks": ("MASK", {
                    "tooltip": "Masks from SAM3 segmentation (recommended). Excludes foreground from tracking."
                }),
                "auto_mask_people": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "YOLO fallback: auto-detect people if no SAM3 masks provided"
                }),
                
                # === Quality ===
                "quality_mode": (cls.QUALITY_MODES, {
                    "default": "Balanced",
                    "tooltip": "Fast: KLT/LightGlue. Balanced: with LoFTR fallback. Best: LoFTR always."
                }),
                
                # === Intrinsics (for internal solve, can be overridden by MoGe2 in export) ===
                "focal_length_mm": ("FLOAT", {
                    "default": 35.0,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 0.1,
                    "tooltip": "Focal length in mm (for solving). Can be overridden by MoGe2 intrinsics in export."
                }),
                "sensor_width_mm": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Camera sensor width in mm (36mm = full frame)"
                }),
                
                # === Smoothing ===
                "smoothing": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 21,
                    "step": 2,
                    "tooltip": "Gaussian smoothing window (0=none, 5=light, 9=medium, 15=heavy)"
                }),
                
                # === Translation Solver Options (for Full 6-DOF) ===
                "translation_solver": (cls.TRANSLATION_SOLVERS, {
                    "default": "COLMAP",
                    "tooltip": "COLMAP: full bundle adjustment (accurate). Essential Matrix: pairwise (faster, may drift)."
                }),
                "colmap_quality": (cls.COLMAP_QUALITY, {
                    "default": "Medium",
                    "tooltip": "COLMAP reconstruction quality. Low: fast. Medium: balanced. High: best accuracy."
                }),
                
                # === Advanced ===
                "match_threshold": ("INT", {
                    "default": 500,
                    "min": 100,
                    "max": 2000,
                    "tooltip": "Minimum feature matches before quality fallback"
                }),
                "stitch_overlap": ("INT", {
                    "default": 10,
                    "min": 5,
                    "max": 50,
                    "tooltip": "Frame overlap for hybrid shot stitching"
                }),
                "transition_frames": ("STRING", {
                    "default": "",
                    "tooltip": "Manual shot transitions (e.g., '50,120') or empty for auto-detect"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_EXTRINSICS", "IMAGE", "STRING")
    RETURN_NAMES = ("camera_extrinsics", "debug_vis", "status")
    FUNCTION = "solve"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def __init__(self):
        self.yolo_model = None
        self.lightglue_matcher = None
        self.loftr_matcher = None
        self.superpoint = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==================== Model Loading ====================
    
    def load_yolo(self):
        """Load YOLO for person detection."""
        if self.yolo_model is not None:
            return True
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            print("[CameraSolver] YOLO loaded")
            return True
        except Exception as e:
            print(f"[CameraSolver] YOLO failed: {e}")
            return False
    
    def load_lightglue(self):
        """Load LightGlue matcher (Apache 2.0 license)."""
        if self.lightglue_matcher is not None:
            return True
        try:
            from lightglue import LightGlue, SuperPoint
            
            self.superpoint = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.lightglue_matcher = LightGlue(features='superpoint').eval().to(self.device)
            print(f"[CameraSolver] LightGlue loaded on {self.device}")
            return True
        except ImportError:
            print("[CameraSolver] LightGlue not installed. Install with: pip install lightglue")
            return False
        except Exception as e:
            print(f"[CameraSolver] LightGlue failed: {e}")
            return False
    
    def load_loftr(self):
        """Load LoFTR matcher (Apache 2.0 license)."""
        if self.loftr_matcher is not None:
            return True
        try:
            from kornia.feature import LoFTR
            self.loftr_matcher = LoFTR(pretrained='outdoor').eval().to(self.device)
            print(f"[CameraSolver] LoFTR loaded on {self.device}")
            return True
        except ImportError:
            print("[CameraSolver] LoFTR not installed. Install with: pip install kornia")
            return False
        except Exception as e:
            print(f"[CameraSolver] LoFTR failed: {e}")
            return False
    
    # ==================== Person Detection ====================
    
    def detect_people(self, frames: np.ndarray, confidence: float = 0.5, expansion: int = 20) -> Optional[np.ndarray]:
        """Detect people and create foreground masks."""
        if not self.load_yolo():
            return None
        
        num_frames, H, W = frames.shape[:3]
        masks = np.zeros((num_frames, H, W), dtype=np.float32)
        
        for i in range(num_frames):
            frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            results = self.yolo_model(frame_bgr, verbose=False, classes=[0])
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if box.conf[0] >= confidence:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            x1 = max(0, x1 - expansion)
                            y1 = max(0, y1 - expansion)
                            x2 = min(W, x2 + expansion)
                            y2 = min(H, y2 + expansion)
                            masks[i, y1:y2, x1:x2] = 1.0
        
        return masks
    
    # ==================== Feature Matching ====================
    
    def run_klt(self, frame0: np.ndarray, frame1: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """KLT optical flow tracking (CPU)."""
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        
        detection_mask = None
        if mask is not None:
            detection_mask = ((1.0 - mask) * 255).astype(np.uint8)
        
        pts0 = cv2.goodFeaturesToTrack(
            gray0, maxCorners=1000, qualityLevel=0.01, 
            minDistance=10, blockSize=7, mask=detection_mask
        )
        
        if pts0 is None or len(pts0) < 20:
            return np.array([]), np.array([])
        
        pts0 = pts0.reshape(-1, 2)
        
        pts1, status, _ = cv2.calcOpticalFlowPyrLK(
            gray0, gray1, pts0.astype(np.float32), None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        good = status.flatten() == 1
        return pts0[good], pts1[good]
    
    def run_orb(self, frame0: np.ndarray, frame1: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """ORB feature matching (CPU fallback)."""
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        
        orb = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        detection_mask = None
        if mask is not None:
            detection_mask = ((1.0 - mask) * 255).astype(np.uint8)
        
        kp0, desc0 = orb.detectAndCompute(gray0, detection_mask)
        kp1, desc1 = orb.detectAndCompute(gray1, detection_mask)
        
        if desc0 is None or desc1 is None or len(kp0) < 20 or len(kp1) < 20:
            return np.array([]), np.array([])
        
        matches = bf.match(desc0, desc1)
        matches = sorted(matches, key=lambda x: x.distance)[:500]
        
        if len(matches) < 20:
            return np.array([]), np.array([])
        
        pts0 = np.array([kp0[m.queryIdx].pt for m in matches])
        pts1 = np.array([kp1[m.trainIdx].pt for m in matches])
        
        return pts0, pts1
    
    def run_lightglue(self, frame0: np.ndarray, frame1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LightGlue matching (GPU with CPU fallback)."""
        if not self.load_lightglue():
            print("[CameraSolver] LightGlue unavailable, falling back to ORB")
            return self.run_orb(frame0, frame1)
        
        try:
            # Convert to grayscale tensor
            gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            
            img0 = torch.from_numpy(gray0).float()[None, None] / 255.0
            img1 = torch.from_numpy(gray1).float()[None, None] / 255.0
            
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            
            with torch.no_grad():
                feats0 = self.superpoint.extract(img0)
                feats1 = self.superpoint.extract(img1)
                matches01 = self.lightglue_matcher({'image0': feats0, 'image1': feats1})
            
            kpts0 = feats0['keypoints'][0].cpu().numpy()
            kpts1 = feats1['keypoints'][0].cpu().numpy()
            matches = matches01['matches'][0].cpu().numpy()
            
            valid = matches > -1
            pts0 = kpts0[valid]
            pts1 = kpts1[matches[valid]]
            
            return pts0, pts1
            
        except Exception as e:
            print(f"[CameraSolver] LightGlue error: {e}, falling back to ORB")
            return self.run_orb(frame0, frame1)
    
    def run_loftr(self, frame0: np.ndarray, frame1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LoFTR matching (GPU with CPU fallback)."""
        if not self.load_loftr():
            print("[CameraSolver] LoFTR unavailable, falling back to ORB")
            return self.run_orb(frame0, frame1)
        
        try:
            # Convert to grayscale
            gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            
            # Resize for LoFTR (works best at lower resolution)
            h, w = gray0.shape
            scale = min(640 / max(h, w), 1.0)
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                gray0_resized = cv2.resize(gray0, (new_w, new_h))
                gray1_resized = cv2.resize(gray1, (new_w, new_h))
            else:
                gray0_resized, gray1_resized = gray0, gray1
            
            img0 = torch.from_numpy(gray0_resized).float()[None, None] / 255.0
            img1 = torch.from_numpy(gray1_resized).float()[None, None] / 255.0
            
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            
            with torch.no_grad():
                result = self.loftr_matcher({'image0': img0, 'image1': img1})
            
            pts0 = result['keypoints0'].cpu().numpy()
            pts1 = result['keypoints1'].cpu().numpy()
            
            # Scale back to original resolution
            if scale < 1.0:
                pts0 = pts0 / scale
                pts1 = pts1 / scale
            
            return pts0, pts1
            
        except Exception as e:
            print(f"[CameraSolver] LoFTR error: {e}, falling back to ORB")
            return self.run_orb(frame0, frame1)
    
    # ==================== Shot Type Detection ====================
    
    def detect_shot_type(self, frames: np.ndarray, masks: Optional[np.ndarray] = None) -> Tuple[str, List[int]]:
        """
        Automatically detect shot type and transition frames.
        
        Returns: (shot_type, transition_frames)
        """
        num_frames = len(frames)
        sample_interval = max(1, num_frames // 20)
        
        inlier_ratios = []
        motion_magnitudes = []
        frame_indices = []
        
        for i in range(0, num_frames - sample_interval, sample_interval):
            mask0 = masks[i] if masks is not None else None
            pts0, pts1 = self.run_klt(frames[i], frames[i + sample_interval], mask0)
            
            if len(pts0) < 20:
                continue
            
            motion = np.mean(np.linalg.norm(pts1 - pts0, axis=1))
            motion_magnitudes.append(motion)
            
            H, inliers = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
            ratio = np.sum(inliers) / len(pts0) if inliers is not None else 0.0
            inlier_ratios.append(ratio)
            frame_indices.append(i)
        
        if not inlier_ratios:
            return "Nodal", []
        
        avg_motion = np.mean(motion_magnitudes)
        avg_inlier = np.mean(inlier_ratios)
        
        # Static detection
        if avg_motion < 2.0:
            print(f"[CameraSolver] Detected: Static (avg motion: {avg_motion:.2f}px)")
            return "Static", []
        
        # Find transitions
        transitions = []
        for i in range(1, len(inlier_ratios)):
            if inlier_ratios[i-1] > 0.80 and inlier_ratios[i] < 0.65:
                transitions.append(frame_indices[i])
            elif inlier_ratios[i-1] < 0.65 and inlier_ratios[i] > 0.80:
                transitions.append(frame_indices[i])
        
        if transitions:
            print(f"[CameraSolver] Detected: Hybrid with transitions at {transitions}")
            return "Hybrid", transitions
        elif avg_inlier > 0.85:
            print(f"[CameraSolver] Detected: Nodal (avg inlier: {avg_inlier:.2f})")
            return "Nodal", []
        else:
            print(f"[CameraSolver] Detected: Parallax (avg inlier: {avg_inlier:.2f})")
            return "Parallax", []
    
    def classify_segment(self, frames: np.ndarray, masks: Optional[np.ndarray] = None) -> str:
        """Classify a video segment as nodal or parallax."""
        if len(frames) < 3:
            return "nodal"
        
        mid = len(frames) // 2
        mask0 = masks[0] if masks is not None else None
        pts0, pts1 = self.run_klt(frames[0], frames[mid], mask0)
        
        if len(pts0) < 20:
            return "nodal"
        
        H, inliers = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
        ratio = np.sum(inliers) / len(pts0) if inliers is not None else 0.0
        return "nodal" if ratio > 0.80 else "parallax"
    
    # ==================== Camera Solving ====================
    
    def rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (pan, tilt, roll)."""
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            tilt = math.atan2(-R[2, 0], sy)
            pan = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])
        else:
            tilt = math.atan2(-R[2, 0], sy)
            pan = math.atan2(-R[0, 1], R[1, 1])
            roll = 0
        
        return (pan, tilt, roll)
    
    def solve_rotation_only(
        self, 
        frames: np.ndarray, 
        masks: Optional[np.ndarray],
        quality: str,
        focal_px: float,
        match_threshold: int
    ) -> List[Dict]:
        """Solve rotation-only camera (nodal pan/tilt)."""
        num_frames = len(frames)
        H, W = frames.shape[1:3]
        
        K = np.array([
            [focal_px, 0, W / 2],
            [0, focal_px, H / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        K_inv = np.linalg.inv(K)
        
        rotations = [{"frame": 0, "pan": 0.0, "tilt": 0.0, "roll": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}]
        cumulative_R = np.eye(3)
        
        for i in range(1, num_frames):
            mask0 = masks[i-1] if masks is not None else None
            
            # Get matches based on quality
            if quality == "Best":
                pts0, pts1 = self.run_loftr(frames[i-1], frames[i])
            elif quality == "Balanced":
                pts0, pts1 = self.run_klt(frames[i-1], frames[i], mask0)
                if len(pts0) < match_threshold:
                    pts0, pts1 = self.run_loftr(frames[i-1], frames[i])
            else:  # Fast
                pts0, pts1 = self.run_klt(frames[i-1], frames[i], mask0)
            
            if len(pts0) < 20:
                rotations.append(rotations[-1].copy())
                rotations[-1]["frame"] = i
                continue
            
            H_mat, inliers = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
            
            if H_mat is None:
                rotations.append(rotations[-1].copy())
                rotations[-1]["frame"] = i
                continue
            
            # R = K^-1 * H * K
            R_delta = K_inv @ H_mat @ K
            U, S, Vt = np.linalg.svd(R_delta)
            R_delta = U @ Vt
            if np.linalg.det(R_delta) < 0:
                R_delta = -R_delta
            
            cumulative_R = R_delta @ cumulative_R
            pan, tilt, roll = self.rotation_matrix_to_euler(cumulative_R)
            
            rotations.append({
                "frame": i,
                "pan": pan,
                "tilt": tilt,
                "roll": roll,
                "tx": 0.0,
                "ty": 0.0,
                "tz": 0.0,
            })
        
        return rotations
    
    def solve_with_translation(
        self,
        frames: np.ndarray,
        masks: Optional[np.ndarray],
        quality: str,
        focal_px: float,
        match_threshold: int
    ) -> List[Dict]:
        """Solve camera with translation using COLMAP or Essential Matrix."""
        num_frames = len(frames)
        H, W = frames.shape[1:3]
        
        K = np.array([
            [focal_px, 0, W / 2],
            [0, focal_px, H / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Collect all matches
        all_matches = []
        for i in range(1, num_frames):
            if quality == "Best":
                pts0, pts1 = self.run_loftr(frames[i-1], frames[i])
            elif quality == "Balanced":
                pts0, pts1 = self.run_lightglue(frames[i-1], frames[i])
                if len(pts0) < match_threshold:
                    pts0, pts1 = self.run_loftr(frames[i-1], frames[i])
            else:  # Fast
                pts0, pts1 = self.run_lightglue(frames[i-1], frames[i])
            
            all_matches.append((pts0, pts1))
        
        # Try COLMAP if available
        if COLMAP_AVAILABLE:
            try:
                print("[CameraSolver] Running COLMAP reconstruction...")
                colmap_result = run_colmap_reconstruction(frames, all_matches, focal_px)
                if colmap_result is not None:
                    return self._convert_colmap_result(colmap_result, num_frames)
            except Exception as e:
                print(f"[CameraSolver] COLMAP failed: {e}, using Essential Matrix")
        
        # Essential Matrix fallback
        return self._solve_essential_matrix(all_matches, K, num_frames)
    
    def _solve_essential_matrix(
        self,
        all_matches: List[Tuple[np.ndarray, np.ndarray]],
        K: np.ndarray,
        num_frames: int
    ) -> List[Dict]:
        """Fallback solver using Essential Matrix."""
        rotations = [{"frame": 0, "pan": 0.0, "tilt": 0.0, "roll": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0}]
        
        cumulative_R = np.eye(3)
        cumulative_t = np.zeros(3)
        
        for i, (pts0, pts1) in enumerate(all_matches):
            if len(pts0) < 20:
                rotations.append(rotations[-1].copy())
                rotations[-1]["frame"] = i + 1
                continue
            
            E, mask = cv2.findEssentialMat(pts0, pts1, K, cv2.RANSAC, 0.999, 1.0)
            
            if E is None:
                rotations.append(rotations[-1].copy())
                rotations[-1]["frame"] = i + 1
                continue
            
            _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K, mask)
            
            cumulative_t = cumulative_R @ t.flatten() + cumulative_t
            cumulative_R = R @ cumulative_R
            
            pan, tilt, roll = self.rotation_matrix_to_euler(cumulative_R)
            
            rotations.append({
                "frame": i + 1,
                "pan": pan,
                "tilt": tilt,
                "roll": roll,
                "tx": float(cumulative_t[0]),
                "ty": float(cumulative_t[1]),
                "tz": float(cumulative_t[2]),
            })
        
        return rotations
    
    def _convert_colmap_result(self, colmap_result: Dict, num_frames: int) -> List[Dict]:
        """Convert COLMAP result to our format."""
        rotations = []
        
        for i in range(num_frames):
            if i in colmap_result:
                cam = colmap_result[i]
                rotations.append({
                    "frame": i,
                    "pan": cam.get("pan", 0.0),
                    "tilt": cam.get("tilt", 0.0),
                    "roll": cam.get("roll", 0.0),
                    "tx": cam.get("tx", 0.0),
                    "ty": cam.get("ty", 0.0),
                    "tz": cam.get("tz", 0.0),
                })
            elif rotations:
                rotations.append(rotations[-1].copy())
                rotations[-1]["frame"] = i
            else:
                rotations.append({"frame": i, "pan": 0, "tilt": 0, "roll": 0, "tx": 0, "ty": 0, "tz": 0})
        
        return rotations
    
    # ==================== Hybrid Solving ====================
    
    def solve_hybrid(
        self,
        frames: np.ndarray,
        masks: Optional[np.ndarray],
        quality: str,
        focal_px: float,
        match_threshold: int,
        transitions: List[int],
        stitch_overlap: int
    ) -> List[Dict]:
        """Solve hybrid shots with multiple transitions."""
        num_frames = len(frames)
        
        # Build segments
        boundaries = [0] + sorted(transitions) + [num_frames]
        segments = []
        
        for i in range(len(boundaries) - 1):
            start = max(0, boundaries[i] - stitch_overlap if i > 0 else 0)
            end = min(num_frames, boundaries[i+1] + stitch_overlap if i < len(boundaries) - 2 else num_frames)
            
            seg_frames = frames[start:end]
            seg_masks = masks[start:end] if masks is not None else None
            seg_type = self.classify_segment(seg_frames, seg_masks)
            
            segments.append({
                "start": start,
                "end": end,
                "type": seg_type,
                "frames": seg_frames,
                "masks": seg_masks
            })
        
        # Solve each segment
        all_rotations = []
        for seg in segments:
            print(f"[CameraSolver] Solving {seg['start']}-{seg['end']} as {seg['type']}")
            
            if seg["type"] == "nodal":
                seg_rots = self.solve_rotation_only(
                    seg["frames"], seg["masks"], quality, focal_px, match_threshold
                )
            else:
                seg_rots = self.solve_with_translation(
                    seg["frames"], seg["masks"], quality, focal_px, match_threshold
                )
            
            for r in seg_rots:
                r["frame"] += seg["start"]
            
            all_rotations.append(seg_rots)
        
        return self._stitch_segments(all_rotations, stitch_overlap, num_frames)
    
    def _stitch_segments(
        self,
        segments: List[List[Dict]],
        overlap: int,
        total_frames: int
    ) -> List[Dict]:
        """Stitch camera segments with blending."""
        if len(segments) == 1:
            return segments[0]
        
        result = [None] * total_frames
        
        for seg in segments:
            for rot in seg:
                idx = rot["frame"]
                if 0 <= idx < total_frames:
                    if result[idx] is None:
                        result[idx] = rot
                    else:
                        # Blend overlapping frames
                        for key in ["pan", "tilt", "roll", "tx", "ty", "tz"]:
                            result[idx][key] = (result[idx][key] + rot[key]) / 2
        
        # Fill gaps
        for i in range(total_frames):
            if result[i] is None:
                if i > 0 and result[i-1]:
                    result[i] = result[i-1].copy()
                    result[i]["frame"] = i
                else:
                    result[i] = {"frame": i, "pan": 0, "tilt": 0, "roll": 0, "tx": 0, "ty": 0, "tz": 0}
        
        return result
    
    # ==================== Smoothing ====================
    
    def smooth_rotations(self, rotations: List[Dict], window: int) -> List[Dict]:
        """Apply Gaussian smoothing."""
        if window <= 1 or len(rotations) < 3:
            return rotations
        
        n = len(rotations)
        half = window // 2
        
        sigma = half / 2.0
        weights = [math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-half, half + 1)]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        def smooth(values):
            result = []
            for i in range(len(values)):
                total, total_w = 0.0, 0.0
                for j, w in enumerate(weights):
                    idx = i + j - half
                    if 0 <= idx < len(values):
                        total += values[idx] * w
                        total_w += w
                result.append(total / total_w if total_w > 0 else values[i])
            return result
        
        pans = smooth([r["pan"] for r in rotations])
        tilts = smooth([r["tilt"] for r in rotations])
        rolls = smooth([r["roll"] for r in rotations])
        txs = smooth([r["tx"] for r in rotations])
        tys = smooth([r["ty"] for r in rotations])
        tzs = smooth([r["tz"] for r in rotations])
        
        result = []
        for i in range(n):
            result.append({
                "frame": rotations[i]["frame"],
                "pan": pans[i],
                "tilt": tilts[i],
                "roll": rolls[i],
                "tx": txs[i],
                "ty": tys[i],
                "tz": tzs[i],
            })
        
        return result
    
    # ==================== Debug Visualization ====================
    
    def create_debug_vis(self, frames: np.ndarray, rotations: List[Dict], shot_type: str) -> np.ndarray:
        """Create debug visualization."""
        debug_frames = []
        
        for i, frame in enumerate(frames):
            debug = frame.copy()
            
            if i < len(rotations):
                rot = rotations[i]
                pan_deg = np.degrees(rot["pan"])
                tilt_deg = np.degrees(rot["tilt"])
                
                cv2.putText(debug, f"{shot_type}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(debug, f"Pan: {pan_deg:.1f} Tilt: {tilt_deg:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if abs(rot["tx"]) > 0.001 or abs(rot["ty"]) > 0.001 or abs(rot["tz"]) > 0.001:
                    cv2.putText(debug, f"T: ({rot['tx']:.2f}, {rot['ty']:.2f}, {rot['tz']:.2f})",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
            
            debug_frames.append(debug)
        
        return np.stack(debug_frames, axis=0)
    
    # ==================== Main Entry ====================
    
    def solve(
        self,
        images: torch.Tensor,
        solving_method: str = "Auto (Recommended)",
        foreground_masks: Optional[torch.Tensor] = None,
        auto_mask_people: bool = True,
        quality_mode: str = "Balanced",
        focal_length_mm: float = 35.0,
        sensor_width_mm: float = 36.0,
        smoothing: int = 5,
        translation_solver: str = "COLMAP",
        colmap_quality: str = "Medium",
        match_threshold: int = 500,
        stitch_overlap: int = 10,
        transition_frames: str = "",
    ) -> Tuple[Dict, torch.Tensor, str]:
        """
        Main camera solving entry point.
        
        Masking priority:
        1. SAM3 masks (foreground_masks input) - best quality
        2. YOLO auto-detection (if auto_mask_people=True) - fallback
        3. No masking - uses all features including people
        """
        
        print(f"[CameraSolver] Method: {solving_method}, Quality: {quality_mode}")
        
        frames = (images.cpu().numpy() * 255).astype(np.uint8)
        num_frames, H, W = frames.shape[:3]
        
        focal_px = focal_length_mm * W / sensor_width_mm
        print(f"[CameraSolver] Focal: {focal_length_mm}mm = {focal_px:.1f}px")
        
        # Resolution-adaptive threshold
        pixels = W * H
        adaptive_threshold = int(match_threshold * (pixels / 2_073_600) ** 0.5)
        
        # Process masks - SAM3 preferred, YOLO fallback
        masks = None
        mask_source = "none"
        if foreground_masks is not None:
            masks = foreground_masks.cpu().numpy()
            mask_source = "SAM3"
            print("[CameraSolver] Using SAM3 masks (recommended)")
        elif auto_mask_people:
            masks = self.detect_people(frames)
            if masks is not None:
                mask_source = "YOLO"
                print("[CameraSolver] Using YOLO auto-detection (SAM3 masks preferred)")
            else:
                print("[CameraSolver] YOLO failed, no masking")
        else:
            print("[CameraSolver] No masking - features may include foreground")
        
        # Map solving method to internal shot types
        shot_type = "Auto"
        if "Rotation Only" in solving_method:
            shot_type = "Nodal"
        elif "Full 6-DOF" in solving_method:
            shot_type = "Parallax"
        
        # Detect shot type if Auto
        transitions = []
        if shot_type == "Auto":
            shot_type, transitions = self.detect_shot_type(frames, masks)
        elif transition_frames:
            try:
                transitions = [int(x.strip()) for x in transition_frames.split(",") if x.strip()]
                if transitions:
                    shot_type = "Hybrid"
            except:
                transitions = []
        
        # Store translation solver preference for parallax solving
        self._translation_solver = translation_solver
        self._colmap_quality = colmap_quality
        
        # Solve based on detected/selected shot type
        if shot_type == "Static":
            rotations = [{"frame": i, "pan": 0, "tilt": 0, "roll": 0, "tx": 0, "ty": 0, "tz": 0} 
                        for i in range(num_frames)]
            status = "Static: No camera motion"
            solving_method_used = "none"
            
        elif shot_type == "Nodal":
            rotations = self.solve_rotation_only(frames, masks, quality_mode, focal_px, adaptive_threshold)
            status = f"Rotation Only: Homography ({quality_mode})"
            solving_method_used = "rotation_only"
            
        elif shot_type == "Parallax":
            rotations = self.solve_with_translation(frames, masks, quality_mode, focal_px, adaptive_threshold)
            solver_used = "COLMAP" if (COLMAP_AVAILABLE and translation_solver == "COLMAP") else "Essential Matrix"
            status = f"Full 6-DOF: {solver_used} ({quality_mode})"
            solving_method_used = "full_6dof"
            
        elif shot_type == "Hybrid":
            if not transitions:
                _, transitions = self.detect_shot_type(frames, masks)
            rotations = self.solve_hybrid(frames, masks, quality_mode, focal_px, adaptive_threshold, transitions, stitch_overlap)
            status = f"Hybrid: Transitions at {transitions}"
            solving_method_used = "hybrid"
        else:
            rotations = [{"frame": i, "pan": 0, "tilt": 0, "roll": 0, "tx": 0, "ty": 0, "tz": 0}
                        for i in range(num_frames)]
            status = "Unknown"
            solving_method_used = "none"
        
        # Smoothing
        if smoothing > 1:
            rotations = self.smooth_rotations(rotations, smoothing)
            print(f"[CameraSolver] Applied Gaussian smoothing (window={smoothing})")
        
        # Add degree values for convenience
        for rot in rotations:
            rot["pan_deg"] = np.degrees(rot["pan"])
            rot["tilt_deg"] = np.degrees(rot["tilt"])
            rot["roll_deg"] = np.degrees(rot["roll"])
        
        # Check if translation data is present
        has_translation = any(
            abs(r.get("tx", 0)) > 0.001 or abs(r.get("ty", 0)) > 0.001 or abs(r.get("tz", 0)) > 0.001
            for r in rotations
        )
        
        # Build CAMERA_EXTRINSICS output
        camera_extrinsics = {
            "num_frames": num_frames,
            "image_width": W,
            "image_height": H,
            "source": "CameraSolver",
            "solving_method": solving_method_used,
            "coordinate_system": "Y-up",
            "units": "radians",
            "has_translation": has_translation,
            "mask_source": mask_source,
            "quality_mode": quality_mode,
            "smoothing_applied": smoothing if smoothing > 1 else 0,
            "rotations": rotations
        }
        
        # Debug visualization
        debug_vis = self.create_debug_vis(frames, rotations, shot_type)
        debug_tensor = torch.from_numpy(debug_vis).float() / 255.0
        
        # Final status
        final = rotations[-1] if rotations else {"pan_deg": 0, "tilt_deg": 0}
        print(f"[CameraSolver] Final: pan={final['pan_deg']:.2f}°, tilt={final['tilt_deg']:.2f}°")
        status += f" | {num_frames} frames | Masks: {mask_source}"
        
        return (camera_extrinsics, debug_tensor, status)


class CameraDataFromJSON:
    """
    Load camera extrinsics from external tracking applications.
    
    Supports exports from:
    - PFTrack
    - 3DEqualizer
    - SynthEyes
    - Maya
    - Nuke
    - After Effects
    
    Accepts both naming conventions:
    - Cinema terms: pan, tilt, roll (preferred internal format)
    - Axis terms: rx, ry, rz (converted based on coordinate system)
    
    JSON Format:
    {
        "fps": 24,
        "image_width": 1920,
        "image_height": 1080,
        "sensor_width_mm": 36.0,
        "units": "degrees",  // or "radians"
        "coordinate_system": "maya",  // Y-up: ry=pan, rx=tilt. Z-up: rz=pan, rx=tilt
        "frames": [
            {"frame": 0, "pan": 0, "tilt": 0, "roll": 0, "tx": 0, "ty": 0, "tz": 0},
            // OR using axis notation:
            {"frame": 0, "rx": 0, "ry": 0, "rz": 0, "tx": 0, "ty": 0, "tz": 0},
            ...
        ]
    }
    """
    
    COORD_SYSTEMS = ["maya", "blender", "nuke", "3dequalizer", "pftrack", "houdini", "Y-up", "Z-up"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "json_input": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "coordinate_system": (cls.COORD_SYSTEMS, {
                    "default": "maya",
                    "tooltip": "Source coordinate system. Y-up: ry=pan. Z-up: rz=pan."
                }),
                "smoothing": ("INT", {"default": 0, "min": 0, "max": 21, "step": 2}),
                "frame_offset": ("INT", {"default": 0, "min": -1000, "max": 1000}),
                "image_width": ("INT", {"default": 1920, "min": 1, "max": 8192}),
                "image_height": ("INT", {"default": 1080, "min": 1, "max": 8192}),
                "sensor_width_mm": ("FLOAT", {"default": 36.0, "min": 1.0, "max": 100.0}),
                "scale_translation": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0}),
            }
        }
    
    RETURN_TYPES = ("CAMERA_EXTRINSICS",)
    RETURN_NAMES = ("camera_extrinsics",)
    FUNCTION = "load_camera_data"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def _is_y_up(self, coord_sys: str) -> bool:
        """Check if coordinate system is Y-up."""
        y_up_systems = ["maya", "blender", "Y-up", "3dequalizer"]
        return coord_sys.lower() in [s.lower() for s in y_up_systems]
    
    def _convert_axis_to_cinema(self, rx: float, ry: float, rz: float, coord_sys: str) -> Tuple[float, float, float]:
        """
        Convert axis notation (rx, ry, rz) to cinema terms (pan, tilt, roll).
        
        For Y-up systems (Maya, Blender):
            pan (horizontal) = ry (rotation around Y)
            tilt (vertical) = rx (rotation around X)
            roll (dutch) = rz (rotation around Z)
        
        For Z-up systems (Houdini, some engines):
            pan (horizontal) = rz (rotation around Z)
            tilt (vertical) = rx (rotation around X)
            roll (dutch) = ry (rotation around Y)
        """
        if self._is_y_up(coord_sys):
            pan = ry    # Horizontal rotation around Y
            tilt = rx   # Vertical rotation around X
            roll = rz   # Roll around Z (view axis)
        else:
            # Z-up
            pan = rz    # Horizontal rotation around Z
            tilt = rx   # Vertical rotation around X
            roll = ry   # Roll around Y
        
        return pan, tilt, roll
    
    def load_camera_data(
        self,
        json_input: str,
        coordinate_system: str = "maya",
        smoothing: int = 0,
        frame_offset: int = 0,
        image_width: int = 1920,
        image_height: int = 1080,
        sensor_width_mm: float = 36.0,
        scale_translation: float = 1.0,
    ) -> Tuple[Dict]:
        
        print(f"[CameraJSON] Loading camera extrinsics (coord system: {coordinate_system})...")
        
        data = None
        json_input = json_input.strip()
        
        # Try loading from file path
        if os.path.exists(json_input):
            try:
                with open(json_input, 'r') as f:
                    data = json.load(f)
                print(f"[CameraJSON] Loaded from file: {json_input}")
            except Exception as e:
                print(f"[CameraJSON] File read error: {e}")
        
        # Try parsing as JSON string
        if data is None:
            try:
                data = json.loads(json_input)
            except:
                return self._empty_result(image_width, image_height, sensor_width_mm)
        
        if not data:
            return self._empty_result(image_width, image_height, sensor_width_mm)
        
        # Get metadata from JSON or use defaults
        units = data.get("units", "degrees")
        img_w = data.get("image_width", image_width)
        img_h = data.get("image_height", image_height)
        sensor_w = data.get("sensor_width_mm", sensor_width_mm)
        json_coord_sys = data.get("coordinate_system", coordinate_system)
        
        frames_data = data.get("frames", [])
        if not frames_data:
            return self._empty_result(img_w, img_h, sensor_w)
        
        rotations = []
        for fd in frames_data:
            frame_idx = fd.get("frame", len(rotations)) + frame_offset
            
            # Check which notation is used and convert to cinema terms
            if "pan" in fd:
                # Already in cinema terms
                pan = fd.get("pan", 0.0)
                tilt = fd.get("tilt", 0.0)
                roll = fd.get("roll", 0.0)
            elif "rx" in fd or "ry" in fd or "rz" in fd:
                # Axis notation - convert based on coordinate system
                rx = fd.get("rx", 0.0)
                ry = fd.get("ry", 0.0)
                rz = fd.get("rz", 0.0)
                pan, tilt, roll = self._convert_axis_to_cinema(rx, ry, rz, json_coord_sys)
            else:
                # No rotation data
                pan, tilt, roll = 0.0, 0.0, 0.0
            
            # Convert to radians if needed
            if units == "degrees":
                pan_rad = np.radians(pan)
                tilt_rad = np.radians(tilt)
                roll_rad = np.radians(roll)
            else:
                pan_rad, tilt_rad, roll_rad = pan, tilt, roll
            
            # Translation with scaling
            tx = fd.get("tx", 0.0) * scale_translation
            ty = fd.get("ty", 0.0) * scale_translation
            tz = fd.get("tz", 0.0) * scale_translation
            
            rotations.append({
                "frame": frame_idx,
                "pan": pan_rad,
                "tilt": tilt_rad,
                "roll": roll_rad,
                "pan_deg": np.degrees(pan_rad),
                "tilt_deg": np.degrees(tilt_rad),
                "roll_deg": np.degrees(roll_rad),
                "tx": tx,
                "ty": ty,
                "tz": tz,
            })
        
        # Sort by frame
        rotations.sort(key=lambda x: x["frame"])
        
        # Apply smoothing if requested
        if smoothing > 1 and len(rotations) > smoothing:
            rotations = self._smooth(rotations, smoothing)
            print(f"[CameraJSON] Applied Gaussian smoothing (window={smoothing})")
        
        # Check for translation
        has_trans = any(
            abs(r.get("tx", 0)) > 0.001 or 
            abs(r.get("ty", 0)) > 0.001 or 
            abs(r.get("tz", 0)) > 0.001 
            for r in rotations
        )
        
        # Build CAMERA_EXTRINSICS output
        camera_extrinsics = {
            "num_frames": len(rotations),
            "image_width": img_w,
            "image_height": img_h,
            "source": "JSON",
            "solving_method": "external",
            "coordinate_system": "Y-up" if self._is_y_up(json_coord_sys) else "Z-up",
            "original_coordinate_system": json_coord_sys,
            "units": "radians",
            "has_translation": has_trans,
            "rotations": rotations
        }
        
        final = rotations[-1] if rotations else {"pan_deg": 0, "tilt_deg": 0}
        print(f"[CameraJSON] Loaded {len(rotations)} frames, final: pan={final['pan_deg']:.2f}°, tilt={final['tilt_deg']:.2f}°")
        
        return (camera_extrinsics,)
    
    def _smooth(self, rotations, window):
        n = len(rotations)
        half = window // 2
        sigma = half / 2.0
        weights = [math.exp(-(i*i)/(2*sigma*sigma)) for i in range(-half, half+1)]
        ws = sum(weights)
        weights = [w/ws for w in weights]
        
        def sm(vals):
            res = []
            for i in range(len(vals)):
                t, tw = 0.0, 0.0
                for j, w in enumerate(weights):
                    idx = i + j - half
                    if 0 <= idx < len(vals):
                        t += vals[idx] * w
                        tw += w
                res.append(t/tw if tw > 0 else vals[i])
            return res
        
        pans = sm([r["pan"] for r in rotations])
        tilts = sm([r["tilt"] for r in rotations])
        rolls = sm([r["roll"] for r in rotations])
        txs = sm([r.get("tx", 0) for r in rotations])
        tys = sm([r.get("ty", 0) for r in rotations])
        tzs = sm([r.get("tz", 0) for r in rotations])
        
        result = []
        for i in range(n):
            result.append({
                "frame": rotations[i]["frame"],
                "pan": pans[i],
                "tilt": tilts[i],
                "roll": rolls[i],
                "pan_deg": np.degrees(pans[i]),
                "tilt_deg": np.degrees(tilts[i]),
                "roll_deg": np.degrees(rolls[i]),
                "tx": txs[i],
                "ty": tys[i],
                "tz": tzs[i],
            })
        return result
    
    def _empty_result(self, w, h, sensor):
        return ({
            "num_frames": 1,
            "image_width": w,
            "image_height": h,
            "source": "JSON",
            "solving_method": "external",
            "coordinate_system": "Y-up",
            "units": "radians",
            "has_translation": False,
            "rotations": [{
                "frame": 0,
                "pan": 0, "tilt": 0, "roll": 0,
                "pan_deg": 0, "tilt_deg": 0, "roll_deg": 0,
                "tx": 0, "ty": 0, "tz": 0
            }]
        },)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CameraSolver": CameraSolver,
    "CameraDataFromJSON": CameraDataFromJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraSolver": "📷 Camera Solver",
    "CameraDataFromJSON": "📷 Camera Extrinsics from JSON",
}
