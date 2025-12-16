"""
Camera Rotation Solver Node for SAM3DBody2abc

Estimates camera rotation (pan/tilt) from video frames using optical flow
on the background (excluding foreground people).

This solves the fundamental problem: pred_cam_t from SAM3DBody tells us
WHERE the body appears on screen, but not WHY (body movement vs camera rotation).

By analyzing background motion, we can determine the actual camera rotation.

Pipeline:
1. Auto-detect all people using YOLO (or use provided masks)
2. Invert masks to get background
3. Compute optical flow (RAFT) between consecutive frames
4. Extract flow only from background regions
5. Estimate homography from background flow
6. Decompose homography to get camera rotation

Requirements:
- torchvision >= 0.14 (for RAFT)
- opencv-python
- ultralytics (for YOLO auto-masking, optional)
"""

import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Any, Optional, List


class CameraRotationSolver:
    """
    Estimates camera rotation from video frames using background optical flow.
    Automatically masks all people using YOLO, or accepts manual masks.
    """
    
    TRACKING_METHODS = ["KLT (Persistent)", "CoTracker (AI)", "ORB (Feature-Based)", "RAFT (Dense Flow)"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "foreground_masks": ("MASK",),
                "sam3_masks": ("SAM3_VIDEO_MASKS",),
                "tracking_method": (cls.TRACKING_METHODS, {
                    "default": "KLT (Persistent)",
                    "tooltip": "KLT: Professional persistent tracking (fast, CPU). CoTracker: AI-based, handles occlusion (GPU). ORB: Sparse features. RAFT: Dense flow."
                }),
                "auto_mask_people": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically detect and mask all people using YOLO (recommended). Disable if providing manual masks."
                }),
                "detection_confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "YOLO person detection confidence threshold"
                }),
                "mask_expansion": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Expand detected person masks by this many pixels (helps exclude motion blur)"
                }),
                "focal_length_px": ("FLOAT", {
                    "default": 1000.0,
                    "min": 100.0,
                    "max": 5000.0,
                    "tooltip": "Focal length in pixels (from SAM3DBody or estimated). Used for proper rotation decomposition."
                }),
                "flow_threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Minimum flow magnitude to consider (filters noise) - only for RAFT"
                }),
                "ransac_threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "RANSAC reprojection threshold for homography estimation"
                }),
                "smoothing": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 15,
                    "tooltip": "Temporal smoothing window for rotation values (0=none)"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_ROTATION_DATA", "MASK", "IMAGE")
    RETURN_NAMES = ("camera_rotations", "debug_masks", "debug_tracking")
    FUNCTION = "solve_camera_rotation"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def __init__(self):
        self.raft_model = None
        self.yolo_model = None
        self.device = None
        self.debug_points = []  # Store tracked points for visualization
    
    def load_raft(self):
        """Load RAFT model from torchvision."""
        if self.raft_model is not None:
            return
        
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[CameraSolver] Loading RAFT model on {self.device}...")
            
            weights = Raft_Large_Weights.DEFAULT
            self.raft_model = raft_large(weights=weights, progress=True)
            self.raft_model = self.raft_model.to(self.device)
            self.raft_model.eval()
            
            # Get transforms
            self.raft_transforms = weights.transforms()
            
            print(f"[CameraSolver] RAFT model loaded successfully")
            
        except ImportError as e:
            print(f"[CameraSolver] Error: torchvision >= 0.14 required for RAFT. {e}")
            raise
    
    def load_yolo(self):
        """Load YOLO model for person detection."""
        if self.yolo_model is not None:
            return True
        
        try:
            from ultralytics import YOLO
            
            print(f"[CameraSolver] Loading YOLOv8 model for person detection...")
            
            # Use YOLOv8 nano for speed, or small for better accuracy
            self.yolo_model = YOLO('yolov8n.pt')  # Auto-downloads ~6MB model
            
            print(f"[CameraSolver] YOLOv8 model loaded successfully")
            return True
            
        except ImportError:
            print(f"[CameraSolver] Warning: ultralytics not installed. Run: pip install ultralytics")
            print(f"[CameraSolver] Auto person masking disabled, will use provided masks or full frame")
            return False
        except Exception as e:
            print(f"[CameraSolver] Warning: Failed to load YOLO model: {e}")
            return False
    
    def detect_people_yolo(
        self, 
        frames: np.ndarray, 
        confidence: float = 0.5,
        mask_expansion: int = 20
    ) -> np.ndarray:
        """
        Detect all people in video frames using YOLO.
        
        Args:
            frames: Video frames (N, H, W, C) uint8 RGB
            confidence: Detection confidence threshold
            mask_expansion: Pixels to expand each detection mask
            
        Returns:
            foreground_masks: (N, H, W) float32, 1.0 where people detected
        """
        if not self.load_yolo():
            print(f"[CameraSolver] YOLO model failed to load!")
            return None
        
        num_frames, img_height, img_width, channels = frames.shape
        foreground_masks = np.zeros((num_frames, img_height, img_width), dtype=np.float32)
        
        total_detections = 0
        
        print(f"[CameraSolver] Detecting people in {num_frames} frames (confidence={confidence})...")
        print(f"[CameraSolver] Frame shape: {frames.shape}, dtype: {frames.dtype}, range: [{frames.min()}-{frames.max()}]")
        
        for i in range(num_frames):
            frame_rgb = frames[i]
            
            # Convert RGB to BGR for YOLO/OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Run YOLO detection
            results = self.yolo_model(frame_bgr, verbose=False, conf=confidence)
            
            frame_detections = 0
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    cls_id = int(box.cls)
                    conf_val = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Log first frame detections for debugging
                    if i == 0:
                        print(f"[CameraSolver] Frame 0: Detected class {cls_id} with conf {conf_val:.2f} at [{x1},{y1},{x2},{y2}]")
                    
                    # Class 0 is 'person' in COCO
                    if cls_id == 0:
                        # Expand the box
                        x1_exp = max(0, x1 - mask_expansion)
                        y1_exp = max(0, y1 - mask_expansion)
                        x2_exp = min(img_width, x2 + mask_expansion)
                        y2_exp = min(img_height, y2 + mask_expansion)
                        
                        # Fill mask
                        foreground_masks[i, y1_exp:y2_exp, x1_exp:x2_exp] = 1.0
                        frame_detections += 1
                        total_detections += 1
            
            if i == 0:
                print(f"[CameraSolver] Frame 0: {frame_detections} people detected")
            
            if (i + 1) % 20 == 0:
                print(f"[CameraSolver] Processed {i + 1}/{num_frames} frames ({frame_detections} people in this frame)...")
        
        avg_detections = total_detections / num_frames if num_frames > 0 else 0
        print(f"[CameraSolver] Person detection complete: {total_detections} total, ~{avg_detections:.1f} per frame")
        print(f"[CameraSolver] Mask sum: {foreground_masks.sum():.0f} pixels marked as foreground")
        
        return foreground_masks
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames using RAFT.
        
        Args:
            frame1: First frame (H, W, 3) uint8
            frame2: Second frame (H, W, 3) uint8
            
        Returns:
            flow: Optical flow (H, W, 2) - dx, dy per pixel
        """
        self.load_raft()
        
        # Convert to torch tensors
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float()  # (3, H, W)
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float()
        
        # Apply transforms
        img1, img2 = self.raft_transforms(img1, img2)
        
        # Add batch dimension
        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)
        
        # Compute flow
        with torch.no_grad():
            flow_predictions = self.raft_model(img1, img2)
            # RAFT returns list of predictions, take the last (most refined)
            flow = flow_predictions[-1]
        
        # Convert to numpy (H, W, 2)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        
        return flow
    
    def estimate_homography_feature_based(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        mask: Optional[np.ndarray],
        ransac_threshold: float = 3.0
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Estimate homography using sparse feature matching (ORB).
        More robust than dense optical flow for fast camera motion.
        
        Args:
            frame1: First frame (H, W, 3) uint8 RGB
            frame2: Second frame (H, W, 3) uint8 RGB
            mask: Background mask (H, W) - 255 for background, 0 for foreground
            ransac_threshold: RANSAC reprojection threshold
            
        Returns:
            homography: 3x3 homography matrix
            debug_info: Dict with matched points info
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Convert mask to uint8 for OpenCV
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = None
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=2000)
        
        # Detect keypoints and compute descriptors
        kp1, desc1 = orb.detectAndCompute(gray1, mask_uint8)
        kp2, desc2 = orb.detectAndCompute(gray2, mask_uint8)
        
        debug_info = {
            'total_points': 0,
            'points_after_mask': 0,
            'points_after_flow': 0,
            'src_points': None,
            'dst_points': None,
            'inliers': None
        }
        
        if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
            print(f"[CameraSolver] Not enough features: frame1={len(kp1) if kp1 else 0}, frame2={len(kp2) if kp2 else 0}")
            return None, debug_info
        
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test (Lowe's ratio)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        debug_info['total_points'] = len(kp1)
        debug_info['points_after_mask'] = len(good_matches)
        
        if len(good_matches) < 10:
            print(f"[CameraSolver] Not enough good matches: {len(good_matches)}")
            return None, debug_info
        
        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        debug_info['points_after_flow'] = len(src_pts)
        debug_info['src_points'] = src_pts.copy()
        debug_info['dst_points'] = dst_pts.copy()
        
        # Estimate homography with RANSAC
        homography, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        
        if homography is None:
            print(f"[CameraSolver] Homography estimation failed")
            return None, debug_info
        
        debug_info['inliers'] = inliers
        
        inlier_count = np.sum(inliers) if inliers is not None else 0
        inlier_ratio = inlier_count / len(inliers) if inliers is not None and len(inliers) > 0 else 0
        
        if inlier_ratio < 0.2:
            print(f"[CameraSolver] Warning: Low inlier ratio ({inlier_ratio:.2f}) - {inlier_count}/{len(good_matches)} inliers")
        else:
            print(f"[CameraSolver] Good match: {inlier_count}/{len(good_matches)} inliers ({inlier_ratio:.1%})")
        
        return homography, debug_info
    
    def solve_rotation_klt_persistent(
        self,
        frames: np.ndarray,
        bg_masks: Optional[np.ndarray],
        focal_length_px: float,
        ransac_threshold: float = 3.0
    ) -> Tuple[List[Tuple[float, float, float]], List[np.ndarray], List[Dict]]:
        """
        Professional-style camera rotation solving using persistent KLT tracking.
        
        This mimics how PFTrack/SynthEyes/3DEqualizer work:
        1. Detect good features in frame 0
        2. Track them persistently across all frames using KLT (Lucas-Kanade)
        3. Estimate rotation relative to frame 0 (no drift accumulation)
        4. Use Essential Matrix decomposition for pure rotation
        
        Args:
            frames: Video frames (N, H, W, 3) uint8 RGB
            bg_masks: Background masks (N, H, W) float, 1.0 = background
            focal_length_px: Focal length in pixels
            ransac_threshold: RANSAC threshold
            
        Returns:
            rotations: List of (pan, tilt, roll) tuples per frame
            debug_frames: List of debug visualization frames
            track_info: List of tracking info dicts per frame
        """
        num_frames, img_height, img_width = frames.shape[:3]
        cx, cy = img_width / 2, img_height / 2
        
        # Camera intrinsic matrix
        K = np.array([
            [focal_length_px, 0, cx],
            [0, focal_length_px, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Convert first frame to grayscale
        gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        
        # Create mask for feature detection (background only)
        detection_mask = None
        if bg_masks is not None:
            detection_mask = (bg_masks[0] * 255).astype(np.uint8)
        
        # Detect good features to track in frame 0
        # Parameters tuned for camera tracking
        feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7,
            mask=detection_mask
        )
        
        pts0 = cv2.goodFeaturesToTrack(gray0, **feature_params)
        
        if pts0 is None or len(pts0) < 20:
            print(f"[CameraSolver] KLT: Not enough features detected in frame 0 ({len(pts0) if pts0 else 0})")
            # Return identity rotations
            rotations = [(0.0, 0.0, 0.0)] * num_frames
            debug_frames = [frames[i].copy() for i in range(num_frames)]
            return rotations, debug_frames, []
        
        pts0 = pts0.reshape(-1, 2)
        original_pts0 = pts0.copy()  # Keep original frame 0 points
        print(f"[CameraSolver] KLT: Detected {len(pts0)} features in frame 0")
        
        # KLT tracking parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Track features across all frames
        rotations = [(0.0, 0.0, 0.0)]  # Frame 0 is reference
        debug_frames = []
        track_info_list = []
        
        # Create debug frame for frame 0
        debug_frame0 = frames[0].copy()
        for pt in pts0:
            cv2.circle(debug_frame0, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
        cv2.putText(debug_frame0, f"Frame 0: {len(pts0)} reference points", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        debug_frames.append(debug_frame0)
        
        # Current tracked points and their correspondence to original points
        current_pts = pts0.copy()
        valid_mask = np.ones(len(pts0), dtype=bool)  # Track which original points are still valid
        prev_gray = gray0.copy()
        
        for i in range(1, num_frames):
            # Convert current frame to grayscale
            gray_i = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Track from previous frame to current frame
            pts_to_track = current_pts[valid_mask].reshape(-1, 1, 2).astype(np.float32)
            
            if len(pts_to_track) < 8:
                print(f"[CameraSolver] KLT Frame {i}: Too few points to track ({len(pts_to_track)})")
                rotations.append(rotations[-1])
                debug_frames.append(frames[i].copy())
                prev_gray = gray_i.copy()
                continue
            
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray_i, pts_to_track, None, **lk_params
            )
            
            if next_pts is None:
                print(f"[CameraSolver] KLT: Tracking failed at frame {i}")
                rotations.append(rotations[-1])
                debug_frames.append(frames[i].copy())
                prev_gray = gray_i.copy()
                continue
            
            next_pts = next_pts.reshape(-1, 2)
            status = status.reshape(-1)
            
            # Update valid mask - points that were valid and still tracked
            valid_indices = np.where(valid_mask)[0]
            new_valid_mask = valid_mask.copy()
            
            for j, idx in enumerate(valid_indices):
                if status[j] != 1:
                    new_valid_mask[idx] = False
                else:
                    # Also check if point is in foreground (mask it out)
                    pt = next_pts[j]
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or x >= img_width or y < 0 or y >= img_height:
                        new_valid_mask[idx] = False
                    elif bg_masks is not None and bg_masks[i, y, x] < 0.5:
                        new_valid_mask[idx] = False
            
            # Update current points for valid tracks
            new_current_pts = current_pts.copy()
            valid_idx = 0
            for j, idx in enumerate(valid_indices):
                if status[j] == 1:
                    new_current_pts[idx] = next_pts[valid_idx]
                valid_idx += 1
            
            current_pts = new_current_pts
            valid_mask = new_valid_mask
            
            # Get corresponding points in frame 0 and frame i
            pts0_good = original_pts0[valid_mask]
            pts_i_good = current_pts[valid_mask]
            
            track_info = {
                'total_tracks': len(original_pts0),
                'active_tracks': np.sum(valid_mask),
                'src_points': pts0_good.copy(),
                'dst_points': pts_i_good.copy(),
                'inliers': None
            }
            
            if len(pts0_good) < 8:
                print(f"[CameraSolver] KLT Frame {i}: Not enough tracks ({len(pts0_good)})")
                rotations.append(rotations[-1])
                debug_frame = frames[i].copy()
                cv2.putText(debug_frame, f"Frame {i}: {len(pts0_good)} tracks (need 8+)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                debug_frames.append(debug_frame)
                track_info_list.append(track_info)
                prev_gray = gray_i.copy()
                continue
            
            # Estimate Essential Matrix (better for pure rotation than Homography)
            E, inliers_mask = cv2.findEssentialMat(
                pts0_good, pts_i_good, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=ransac_threshold
            )
            
            if E is None:
                print(f"[CameraSolver] KLT Frame {i}: Essential matrix estimation failed")
                rotations.append(rotations[-1])
                debug_frame = frames[i].copy()
                debug_frames.append(debug_frame)
                track_info_list.append(track_info)
                prev_gray = gray_i.copy()
                continue
            
            inliers_mask = inliers_mask.ravel() if inliers_mask is not None else np.ones(len(pts0_good))
            track_info['inliers'] = inliers_mask.reshape(-1, 1)
            
            inlier_count = np.sum(inliers_mask)
            inlier_ratio = inlier_count / len(pts0_good)
            
            # Recover rotation from Essential Matrix
            _, R, t, mask_pose = cv2.recoverPose(E, pts0_good, pts_i_good, K, mask=inliers_mask.copy().reshape(-1, 1).astype(np.uint8))
            
            # Extract Euler angles from rotation matrix
            pan, tilt, roll = self.rotation_matrix_to_euler(R)
            
            # Log progress
            if i % 10 == 0 or inlier_ratio > 0.5:
                print(f"[CameraSolver] KLT Frame {i}: {inlier_count}/{len(pts0_good)} inliers ({inlier_ratio:.1%}), pan={np.degrees(pan):.2f}°, tilt={np.degrees(tilt):.2f}°")
            
            rotations.append((pan, tilt, roll))
            track_info_list.append(track_info)
            
            # Create debug visualization
            debug_frame = self.create_debug_tracking_image(
                frames[i],
                pts0_good,
                pts_i_good,
                inliers_mask.reshape(-1, 1),
                bg_masks[i] if bg_masks is not None else None
            )
            cv2.putText(debug_frame, f"KLT Frame {i}: {inlier_count}/{len(pts0_good)} ({inlier_ratio:.0%})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            debug_frames.append(debug_frame)
            
            prev_gray = gray_i.copy()
        
        print(f"[CameraSolver] KLT tracking complete!")
        return rotations, debug_frames, track_info_list
    
    def rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pan, tilt, roll).
        
        Uses YXZ convention (common for camera rotations):
        - Pan (Y): Horizontal rotation
        - Tilt (X): Vertical rotation  
        - Roll (Z): Rotation around view axis
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            (pan, tilt, roll) in radians
        """
        # Handle gimbal lock cases
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            tilt = np.arctan2(-R[2, 0], sy)  # X rotation
            pan = np.arctan2(R[1, 0], R[0, 0])  # Y rotation
            roll = np.arctan2(R[2, 1], R[2, 2])  # Z rotation
        else:
            tilt = np.arctan2(-R[2, 0], sy)
            pan = np.arctan2(-R[0, 1], R[1, 1])
            roll = 0
        
        return (pan, tilt, roll)
    
    def load_cotracker(self):
        """Load CoTracker model (downloads automatically on first use)."""
        if hasattr(self, 'cotracker_model') and self.cotracker_model is not None:
            return True
        
        try:
            import torch
            print(f"[CameraSolver] Loading CoTracker model...")
            
            # Try to load CoTracker
            try:
                self.cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
                self.cotracker_model = self.cotracker_model.to(self.device)
                self.cotracker_model.eval()
                print(f"[CameraSolver] CoTracker loaded successfully on {self.device}")
                return True
            except Exception as e:
                print(f"[CameraSolver] CoTracker v2 failed, trying v1: {e}")
                try:
                    self.cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8")
                    self.cotracker_model = self.cotracker_model.to(self.device)
                    self.cotracker_model.eval()
                    print(f"[CameraSolver] CoTracker v1 loaded successfully")
                    return True
                except Exception as e2:
                    print(f"[CameraSolver] CoTracker loading failed: {e2}")
                    print(f"[CameraSolver] Install with: pip install cotracker")
                    self.cotracker_model = None
                    return False
                    
        except ImportError:
            print(f"[CameraSolver] CoTracker not available. Install with: pip install cotracker")
            self.cotracker_model = None
            return False
    
    def solve_rotation_cotracker(
        self,
        frames: np.ndarray,
        bg_masks: Optional[np.ndarray],
        focal_length_px: float,
        ransac_threshold: float = 3.0
    ) -> Tuple[List[Tuple[float, float, float]], List[np.ndarray], List[Dict]]:
        """
        AI-based camera rotation solving using Meta's CoTracker.
        
        CoTracker advantages over KLT:
        - Handles occlusion (tracks through obstacles)
        - GPU accelerated
        - More robust on motion blur
        - State-of-the-art point tracking
        
        Args:
            frames: Video frames (N, H, W, 3) uint8 RGB
            bg_masks: Background masks (N, H, W) float, 1.0 = background
            focal_length_px: Focal length in pixels
            ransac_threshold: RANSAC threshold
            
        Returns:
            rotations: List of (pan, tilt, roll) tuples per frame
            debug_frames: List of debug visualization frames
            track_info: List of tracking info dicts per frame
        """
        if not self.load_cotracker():
            print(f"[CameraSolver] CoTracker not available, falling back to KLT")
            return self.solve_rotation_klt_persistent(frames, bg_masks, focal_length_px, ransac_threshold)
        
        num_frames, img_height, img_width = frames.shape[:3]
        cx, cy = img_width / 2, img_height / 2
        
        # Camera intrinsic matrix
        K = np.array([
            [focal_length_px, 0, cx],
            [0, focal_length_px, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Prepare video tensor for CoTracker (B, T, C, H, W)
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float()
        video_tensor = video_tensor.to(self.device)
        
        # Generate query points in frame 0 (background only)
        # CoTracker expects queries as (B, N, 3) where 3 = (frame_idx, x, y)
        gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        
        detection_mask = None
        if bg_masks is not None:
            detection_mask = (bg_masks[0] * 255).astype(np.uint8)
        
        # Detect good features in frame 0
        feature_params = dict(
            maxCorners=200,  # Fewer points for CoTracker (it's more compute-intensive)
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7,
            mask=detection_mask
        )
        
        pts0 = cv2.goodFeaturesToTrack(gray0, **feature_params)
        
        if pts0 is None or len(pts0) < 20:
            print(f"[CameraSolver] CoTracker: Not enough features in frame 0 ({len(pts0) if pts0 else 0})")
            rotations = [(0.0, 0.0, 0.0)] * num_frames
            debug_frames = [frames[i].copy() for i in range(num_frames)]
            return rotations, debug_frames, []
        
        pts0 = pts0.reshape(-1, 2)
        print(f"[CameraSolver] CoTracker: Tracking {len(pts0)} points across {num_frames} frames...")
        
        # Create queries tensor (B, N, 3) - all points start at frame 0
        queries = torch.zeros((1, len(pts0), 3), device=self.device)
        queries[0, :, 0] = 0  # frame index = 0
        queries[0, :, 1] = torch.from_numpy(pts0[:, 0]).to(self.device)  # x
        queries[0, :, 2] = torch.from_numpy(pts0[:, 1]).to(self.device)  # y
        
        # Run CoTracker
        with torch.no_grad():
            try:
                # CoTracker returns (tracks, visibilities)
                # tracks: (B, T, N, 2) - x, y positions
                # visibilities: (B, T, N) - visibility scores
                pred_tracks, pred_visibility = self.cotracker_model(video_tensor, queries=queries)
                
                pred_tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)
                pred_visibility = pred_visibility[0].cpu().numpy()  # (T, N)
                
                print(f"[CameraSolver] CoTracker: Tracking complete, shape={pred_tracks.shape}")
                
            except Exception as e:
                print(f"[CameraSolver] CoTracker inference failed: {e}")
                print(f"[CameraSolver] Falling back to KLT")
                return self.solve_rotation_klt_persistent(frames, bg_masks, focal_length_px, ransac_threshold)
        
        # Process tracks to compute rotation per frame
        rotations = [(0.0, 0.0, 0.0)]  # Frame 0 is reference
        debug_frames = []
        track_info_list = []
        
        # Create debug frame for frame 0
        debug_frame0 = frames[0].copy()
        for pt in pts0:
            cv2.circle(debug_frame0, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
        cv2.putText(debug_frame0, f"CoTracker Frame 0: {len(pts0)} query points", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        debug_frames.append(debug_frame0)
        
        # Reference points (frame 0)
        pts_ref = pred_tracks[0]  # (N, 2)
        
        for i in range(1, num_frames):
            # Get tracked points for this frame
            pts_i = pred_tracks[i]  # (N, 2)
            vis_i = pred_visibility[i]  # (N,)
            
            # Filter by visibility and background mask
            valid_mask = vis_i > 0.5
            
            if bg_masks is not None:
                for j in range(len(pts_i)):
                    if valid_mask[j]:
                        x, y = int(pts_i[j, 0]), int(pts_i[j, 1])
                        if 0 <= x < img_width and 0 <= y < img_height:
                            if bg_masks[i, y, x] < 0.5:  # In foreground
                                valid_mask[j] = False
                        else:
                            valid_mask[j] = False
            
            pts_ref_good = pts_ref[valid_mask]
            pts_i_good = pts_i[valid_mask]
            
            track_info = {
                'total_tracks': len(pts0),
                'active_tracks': np.sum(valid_mask),
                'src_points': pts_ref_good.copy(),
                'dst_points': pts_i_good.copy(),
                'inliers': None
            }
            
            if len(pts_ref_good) < 8:
                print(f"[CameraSolver] CoTracker Frame {i}: Not enough visible tracks ({len(pts_ref_good)})")
                rotations.append(rotations[-1])
                debug_frame = frames[i].copy()
                cv2.putText(debug_frame, f"Frame {i}: {len(pts_ref_good)} tracks (need 8+)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                debug_frames.append(debug_frame)
                track_info_list.append(track_info)
                continue
            
            # Estimate Essential Matrix
            E, inliers_mask = cv2.findEssentialMat(
                pts_ref_good, pts_i_good, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=ransac_threshold
            )
            
            if E is None:
                print(f"[CameraSolver] CoTracker Frame {i}: Essential matrix failed")
                rotations.append(rotations[-1])
                debug_frame = frames[i].copy()
                debug_frames.append(debug_frame)
                track_info_list.append(track_info)
                continue
            
            inliers_mask = inliers_mask.ravel() if inliers_mask is not None else np.ones(len(pts_ref_good))
            track_info['inliers'] = inliers_mask.reshape(-1, 1)
            
            inlier_count = np.sum(inliers_mask)
            inlier_ratio = inlier_count / len(pts_ref_good)
            
            # Recover rotation
            _, R, t, _ = cv2.recoverPose(E, pts_ref_good, pts_i_good, K, 
                                          mask=inliers_mask.copy().reshape(-1, 1).astype(np.uint8))
            
            pan, tilt, roll = self.rotation_matrix_to_euler(R)
            
            if i % 10 == 0:
                print(f"[CameraSolver] CoTracker Frame {i}: {inlier_count}/{len(pts_ref_good)} inliers ({inlier_ratio:.1%}), pan={np.degrees(pan):.2f}°, tilt={np.degrees(tilt):.2f}°")
            
            rotations.append((pan, tilt, roll))
            track_info_list.append(track_info)
            
            # Create debug visualization
            debug_frame = self.create_debug_tracking_image(
                frames[i],
                pts_ref_good,
                pts_i_good,
                inliers_mask.reshape(-1, 1),
                bg_masks[i] if bg_masks is not None else None
            )
            cv2.putText(debug_frame, f"CoTracker Frame {i}: {inlier_count}/{len(pts_ref_good)} ({inlier_ratio:.0%})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            debug_frames.append(debug_frame)
        
        print(f"[CameraSolver] CoTracker solve complete!")
        return rotations, debug_frames, track_info_list
    
    def estimate_homography_from_flow(
        self, 
        flow: np.ndarray, 
        mask: Optional[np.ndarray], 
        flow_threshold: float,
        ransac_threshold: float,
        return_debug: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Estimate homography from optical flow on background regions.
        
        Args:
            flow: Optical flow (H, W, 2)
            mask: Background mask (H, W) - 1 for background, 0 for foreground
            flow_threshold: Minimum flow magnitude to use
            ransac_threshold: RANSAC threshold
            return_debug: Whether to return debug info
            
        Returns:
            homography: 3x3 homography matrix, or None if estimation failed
            debug_info: Dict with src_points, dst_points, inliers (if return_debug)
        """
        flow_height, flow_width = flow.shape[:2]
        
        # Create grid of points
        y_coords, x_coords = np.mgrid[0:flow_height:8, 0:flow_width:8]  # Sample every 8 pixels
        
        # Flatten
        src_points = np.column_stack([x_coords.ravel(), y_coords.ravel()]).astype(np.float32)
        
        # Get flow at these points
        flow_x = flow[::8, ::8, 0].ravel()
        flow_y = flow[::8, ::8, 1].ravel()
        
        # Destination points
        dst_points = src_points + np.column_stack([flow_x, flow_y])
        
        # Track how many points at each stage
        total_points = len(src_points)
        
        # Apply mask if provided
        if mask is not None:
            mask_sampled = mask[::8, ::8].ravel()
            valid = mask_sampled > 0.5
            points_after_mask = np.sum(valid)
            src_points = src_points[valid]
            dst_points = dst_points[valid]
            flow_x = flow_x[valid]
            flow_y = flow_y[valid]
        else:
            points_after_mask = total_points
        
        # Filter by flow magnitude
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)
        valid = flow_mag > flow_threshold
        points_after_flow = np.sum(valid)
        src_points = src_points[valid]
        dst_points = dst_points[valid]
        
        debug_info = {
            'total_points': total_points,
            'points_after_mask': points_after_mask,
            'points_after_flow': points_after_flow,
            'src_points': src_points.copy() if len(src_points) > 0 else None,
            'dst_points': dst_points.copy() if len(dst_points) > 0 else None,
            'inliers': None
        }
        
        if len(src_points) < 10:
            print(f"[CameraSolver] Warning: Not enough background points ({len(src_points)}) - total:{total_points}, after_mask:{points_after_mask}, after_flow:{points_after_flow}")
            return None, debug_info if return_debug else None
        
        # Estimate homography with RANSAC
        homography, inliers = cv2.findHomography(
            src_points, 
            dst_points, 
            cv2.RANSAC, 
            ransac_threshold
        )
        
        if homography is None:
            print(f"[CameraSolver] Warning: Homography estimation failed")
            return None, debug_info if return_debug else None
        
        debug_info['inliers'] = inliers
        
        inlier_ratio = np.sum(inliers) / len(inliers) if inliers is not None else 0
        if inlier_ratio < 0.3:
            print(f"[CameraSolver] Warning: Low inlier ratio ({inlier_ratio:.2f}) - {np.sum(inliers)}/{len(inliers)} inliers")
        
        return homography, debug_info if return_debug else None
    
    def create_debug_tracking_image(
        self,
        frame: np.ndarray,
        src_points: Optional[np.ndarray],
        dst_points: Optional[np.ndarray],
        inliers: Optional[np.ndarray],
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create debug visualization showing tracked points and flow.
        
        Args:
            frame: Original frame (H, W, 3) uint8
            src_points: Source points (N, 2)
            dst_points: Destination points (N, 2)
            inliers: Boolean array of inliers (N,)
            mask: Background mask to visualize (H, W)
            
        Returns:
            debug_frame: Frame with visualization (H, W, 3) uint8
        """
        debug_frame = frame.copy()
        
        # Frame is RGB, OpenCV uses BGR for colors
        # We'll work in RGB and use RGB color tuples
        
        # Overlay mask as semi-transparent blue (background regions)
        if mask is not None:
            mask_overlay = np.zeros_like(debug_frame)
            mask_overlay[:, :, 2] = (mask * 100).astype(np.uint8)  # Blue channel (index 2 in RGB) for background
            debug_frame = cv2.addWeighted(debug_frame, 0.7, mask_overlay, 0.3, 0)
        
        if src_points is None or dst_points is None:
            # No points to draw - add text showing frame info
            cv2.putText(debug_frame, "No valid points", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Red in RGB
            return debug_frame
        
        # Draw points and flow vectors
        # Colors in RGB: Green=(0,255,0), Red=(255,0,0)
        for i in range(len(src_points)):
            src = tuple(src_points[i].astype(int))
            dst = tuple(dst_points[i].astype(int))
            
            if inliers is not None and i < len(inliers):
                is_inlier = inliers[i][0] if len(inliers[i]) > 0 else inliers[i]
            else:
                is_inlier = True
            
            if is_inlier:
                # Inlier: green point, green line (RGB)
                color = (0, 255, 0)
            else:
                # Outlier: red point, red line (RGB)
                color = (255, 0, 0)
            
            # Draw flow vector
            cv2.arrowedLine(debug_frame, src, dst, color, 1, tipLength=0.3)
            cv2.circle(debug_frame, src, 3, color, -1)
        
        # Add stats text (white in RGB)
        num_inliers = np.sum(inliers) if inliers is not None else 0
        total = len(src_points)
        ratio = num_inliers / total if total > 0 else 0
        
        cv2.putText(debug_frame, f"Points: {total}, Inliers: {num_inliers} ({ratio:.1%})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_frame
    
    def decompose_homography_to_rotation(
        self, 
        homography: np.ndarray, 
        focal_length: float,
        image_width: int,
        image_height: int
    ) -> Tuple[float, float, float]:
        """
        Decompose homography to extract camera rotation.
        
        For a pure rotation, H = K * R * K^-1
        where K is the camera intrinsic matrix.
        
        Args:
            homography: 3x3 homography matrix
            focal_length: Focal length in pixels
            image_width, image_height: Image dimensions
            
        Returns:
            (pan, tilt, roll): Rotation angles in radians
        """
        # Camera intrinsic matrix
        cx, cy = image_width / 2, image_height / 2
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        K_inv = np.linalg.inv(K)
        
        # For pure rotation: H = K * R * K^-1
        # Therefore: R = K^-1 * H * K
        R = K_inv @ homography @ K
        
        # The result may not be a perfect rotation matrix due to noise
        # Use SVD to find the closest rotation matrix
        U, S, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_clean) < 0:
            R_clean = -R_clean
        
        # Extract Euler angles (assuming Y-up convention for Maya)
        # R = Ry(pan) * Rx(tilt) * Rz(roll)
        
        # Tilt (rotation around X axis)
        tilt = np.arctan2(-R_clean[1, 2], R_clean[2, 2])
        
        # Pan (rotation around Y axis)
        pan = np.arctan2(R_clean[0, 2], np.sqrt(R_clean[1, 2]**2 + R_clean[2, 2]**2))
        
        # Roll (rotation around Z axis)
        roll = np.arctan2(-R_clean[0, 1], R_clean[0, 0])
        
        return pan, tilt, roll
    
    def smooth_rotations(self, rotations: List[Tuple[float, float, float]], window: int) -> List[Tuple[float, float, float]]:
        """Apply moving average smoothing to rotation values."""
        if window <= 1 or len(rotations) < window:
            return rotations
        
        pans = [r[0] for r in rotations]
        tilts = [r[1] for r in rotations]
        rolls = [r[2] for r in rotations]
        
        def smooth_array(values):
            result = []
            half = window // 2
            for i in range(len(values)):
                start = max(0, i - half)
                end = min(len(values), i + half + 1)
                avg = sum(values[start:end]) / (end - start)
                result.append(avg)
            return result
        
        smoothed_pans = smooth_array(pans)
        smoothed_tilts = smooth_array(tilts)
        smoothed_rolls = smooth_array(rolls)
        
        return list(zip(smoothed_pans, smoothed_tilts, smoothed_rolls))
    
    def solve_camera_rotation(
        self,
        images: torch.Tensor,
        foreground_masks: Optional[torch.Tensor] = None,
        sam3_masks: Optional[Any] = None,
        tracking_method: str = "ORB (Feature-Based)",
        auto_mask_people: bool = True,
        detection_confidence: float = 0.5,
        mask_expansion: int = 20,
        focal_length_px: float = 1000.0,
        flow_threshold: float = 1.0,
        ransac_threshold: float = 3.0,
        smoothing: int = 5,
    ) -> Tuple[Dict]:
        """
        Solve for camera rotation from video frames.
        
        Args:
            images: Video frames (N, H, W, C)
            foreground_masks: Foreground masks to exclude (N, H, W)
            sam3_masks: SAM3 video masks (alternative to foreground_masks)
            tracking_method: "ORB (Feature-Based)" or "RAFT (Dense Flow)"
            auto_mask_people: Automatically detect and mask all people using YOLO
            detection_confidence: YOLO detection confidence threshold
            mask_expansion: Pixels to expand detected masks
            focal_length_px: Focal length in pixels
            flow_threshold: Minimum flow magnitude (RAFT only)
            ransac_threshold: RANSAC threshold
            smoothing: Temporal smoothing window
            
        Returns:
            camera_rotations: Dict with per-frame rotation data
        """
        use_klt = "KLT" in tracking_method
        use_orb = "ORB" in tracking_method
        use_raft = "RAFT" in tracking_method
        
        print(f"[CameraSolver] Starting camera rotation solve...")
        print(f"[CameraSolver] Tracking method: {tracking_method}")
        print(f"[CameraSolver] Input: {images.shape[0]} frames")
        
        # Convert images to numpy
        frames = (images.cpu().numpy() * 255).astype(np.uint8)
        num_frames, img_height, img_width, C = frames.shape
        
        print(f"[CameraSolver] Frame size: {img_width}x{img_height}")
        print(f"[CameraSolver] Input frames: shape={frames.shape}, dtype={frames.dtype}, range=[{frames.min()}-{frames.max()}]")
        print(f"[CameraSolver] Focal length: {focal_length_px}px")
        
        # Determine foreground masks
        # Priority: provided masks > YOLO auto-detection > no masks
        bg_masks = None
        fg_masks_for_debug = None
        
        if foreground_masks is not None:
            # Use provided foreground masks
            fg = foreground_masks.cpu().numpy()
            fg_masks_for_debug = fg.copy()
            bg_masks = 1.0 - fg
            print(f"[CameraSolver] Using provided foreground masks: shape={fg.shape}")
            
        elif sam3_masks is not None:
            # Try to use SAM3 masks
            bg_masks = self._process_sam3_masks(sam3_masks, num_frames, img_height, img_width)
            if bg_masks is not None:
                fg_masks_for_debug = 1.0 - bg_masks
                print(f"[CameraSolver] Using SAM3 masks successfully")
            else:
                print(f"[CameraSolver] SAM3 mask processing failed!")
                # Fall back to YOLO if enabled
                if auto_mask_people:
                    print(f"[CameraSolver] Falling back to YOLO auto-masking...")
                    fg_masks_for_debug = self.detect_people_yolo(frames, detection_confidence, mask_expansion)
                    if fg_masks_for_debug is not None:
                        bg_masks = 1.0 - fg_masks_for_debug
                        print(f"[CameraSolver] YOLO fallback successful")
                    else:
                        print(f"[CameraSolver] YOLO also failed - using full frame")
                else:
                    print(f"[CameraSolver] auto_mask_people disabled - using full frame")
            
        elif auto_mask_people:
            # Auto-detect all people using YOLO
            print(f"[CameraSolver] Auto-masking people with YOLO...")
            fg_masks_for_debug = self.detect_people_yolo(frames, detection_confidence, mask_expansion)
            if fg_masks_for_debug is not None:
                bg_masks = 1.0 - fg_masks_for_debug
                print(f"[CameraSolver] YOLO auto-masking complete")
            else:
                print(f"[CameraSolver] YOLO unavailable - using full frame")
        else:
            print(f"[CameraSolver] No masks - using full frame (may include foreground motion)")
        
        # ==== KLT PERSISTENT TRACKING (Professional Method) ====
        if use_klt:
            print(f"[CameraSolver] Using KLT persistent tracking (professional method)")
            rotations, debug_tracking_frames, track_info = self.solve_rotation_klt_persistent(
                frames, bg_masks, focal_length_px, ransac_threshold
            )
            
            # Apply smoothing
            if smoothing > 1:
                rotations = self.smooth_rotations(rotations, smoothing)
                print(f"[CameraSolver] Applied smoothing (window={smoothing})")
            
            # Build output
            camera_rotation_data = {
                "num_frames": num_frames,
                "image_width": img_width,
                "image_height": img_height,
                "focal_length_px": focal_length_px,
                "tracking_method": "KLT (Persistent)",
                "rotations": [
                    {
                        "frame": i,
                        "pan": rot[0],
                        "tilt": rot[1],
                        "roll": rot[2],
                        "pan_deg": np.degrees(rot[0]),
                        "tilt_deg": np.degrees(rot[1]),
                        "roll_deg": np.degrees(rot[2]),
                    }
                    for i, rot in enumerate(rotations)
                ]
            }
            
            # Summary
            final_rot = rotations[-1] if rotations else (0, 0, 0)
            print(f"[CameraSolver] Solve complete!")
            print(f"[CameraSolver] Total rotation: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°, roll={np.degrees(final_rot[2]):.2f}°")
            
            # Convert debug outputs
            if fg_masks_for_debug is not None:
                debug_masks_tensor = torch.from_numpy(fg_masks_for_debug).float()
            else:
                debug_masks_tensor = torch.zeros((num_frames, img_height, img_width), dtype=torch.float32)
            
            debug_tracking_array = np.stack(debug_tracking_frames, axis=0)
            debug_tracking_tensor = torch.from_numpy(debug_tracking_array).float() / 255.0
            
            return (camera_rotation_data, debug_masks_tensor, debug_tracking_tensor)
        
        # ==== COTRACKER AI-BASED TRACKING ====
        use_cotracker = "CoTracker" in tracking_method
        if use_cotracker:
            print(f"[CameraSolver] Using CoTracker AI-based tracking (handles occlusion)")
            rotations, debug_tracking_frames, track_info = self.solve_rotation_cotracker(
                frames, bg_masks, focal_length_px, ransac_threshold
            )
            
            # Apply smoothing
            if smoothing > 1:
                rotations = self.smooth_rotations(rotations, smoothing)
                print(f"[CameraSolver] Applied smoothing (window={smoothing})")
            
            # Build output
            camera_rotation_data = {
                "num_frames": num_frames,
                "image_width": img_width,
                "image_height": img_height,
                "focal_length_px": focal_length_px,
                "tracking_method": "CoTracker (AI)",
                "rotations": [
                    {
                        "frame": i,
                        "pan": rot[0],
                        "tilt": rot[1],
                        "roll": rot[2],
                        "pan_deg": np.degrees(rot[0]),
                        "tilt_deg": np.degrees(rot[1]),
                        "roll_deg": np.degrees(rot[2]),
                    }
                    for i, rot in enumerate(rotations)
                ]
            }
            
            # Summary
            final_rot = rotations[-1] if rotations else (0, 0, 0)
            print(f"[CameraSolver] Solve complete!")
            print(f"[CameraSolver] Total rotation: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°, roll={np.degrees(final_rot[2]):.2f}°")
            
            # Convert debug outputs
            if fg_masks_for_debug is not None:
                debug_masks_tensor = torch.from_numpy(fg_masks_for_debug).float()
            else:
                debug_masks_tensor = torch.zeros((num_frames, img_height, img_width), dtype=torch.float32)
            
            debug_tracking_array = np.stack(debug_tracking_frames, axis=0)
            debug_tracking_tensor = torch.from_numpy(debug_tracking_array).float() / 255.0
            
            return (camera_rotation_data, debug_masks_tensor, debug_tracking_tensor)
        
        # ==== ORB / RAFT FRAME-TO-FRAME TRACKING ====
        # Prepare debug outputs
        debug_tracking_frames = []
        
        # Compute per-frame rotations
        rotations = []
        cumulative_pan = 0.0
        cumulative_tilt = 0.0
        cumulative_roll = 0.0
        
        # First frame has no rotation (reference)
        rotations.append((0.0, 0.0, 0.0))
        
        # Create debug image for frame 0 (no tracking, just show mask)
        if fg_masks_for_debug is not None:
            debug_frame0 = self.create_debug_tracking_image(
                frames[0], None, None, None, bg_masks[0] if bg_masks is not None else None
            )
        else:
            debug_frame0 = frames[0].copy()
            cv2.putText(debug_frame0, "Frame 0 (reference)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        debug_tracking_frames.append(debug_frame0)
        
        for i in range(1, num_frames):
            frame1 = frames[i - 1]
            frame2 = frames[i]
            
            # Get background mask for this frame pair
            mask = None
            if bg_masks is not None:
                # Use average of both frames' masks
                if len(bg_masks.shape) == 3:
                    mask = (bg_masks[i-1] + bg_masks[i]) / 2
                else:
                    mask = bg_masks[i] if i < len(bg_masks) else None
            
            # Estimate homography using selected method
            homography = None
            debug_info = None
            
            if use_orb:
                # Feature-based tracking (ORB) - more robust for fast motion
                try:
                    homography, debug_info = self.estimate_homography_feature_based(
                        frame1, frame2, mask, ransac_threshold
                    )
                except Exception as e:
                    print(f"[CameraSolver] Frame {i}: ORB matching failed - {e}")
            else:
                # Dense optical flow (RAFT) - better for slow/detailed motion
                try:
                    flow = self.compute_optical_flow(frame1, frame2)
                    homography, debug_info = self.estimate_homography_from_flow(
                        flow, mask, flow_threshold, ransac_threshold, return_debug=True
                    )
                except Exception as e:
                    print(f"[CameraSolver] Frame {i}: RAFT flow failed - {e}")
            
            # Create debug tracking image
            if debug_info and debug_info.get('src_points') is not None:
                debug_frame = self.create_debug_tracking_image(
                    frame2,
                    debug_info['src_points'],
                    debug_info['dst_points'],
                    debug_info['inliers'],
                    mask
                )
            else:
                debug_frame = frame2.copy()
                cv2.putText(debug_frame, f"Frame {i}: No matches", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if mask is not None:
                    mask_overlay = np.zeros_like(debug_frame)
                    mask_overlay[:, :, 2] = (mask * 100).astype(np.uint8)
                    debug_frame = cv2.addWeighted(debug_frame, 0.7, mask_overlay, 0.3, 0)
            
            # Add frame number
            cv2.putText(debug_frame, f"Frame {i}", (10, img_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            debug_tracking_frames.append(debug_frame)
            
            if homography is None:
                # No valid homography, assume no rotation
                rotations.append((cumulative_pan, cumulative_tilt, cumulative_roll))
                continue
            
            # Decompose to rotation
            delta_pan, delta_tilt, delta_roll = self.decompose_homography_to_rotation(
                homography, focal_length_px, img_width, img_height
            )
            
            # Sanity check: reject huge rotations (likely errors)
            max_delta = np.radians(10)  # Max 10 degrees per frame
            if abs(delta_pan) > max_delta or abs(delta_tilt) > max_delta or abs(delta_roll) > max_delta:
                print(f"[CameraSolver] Frame {i}: Rejected large rotation delta (pan={np.degrees(delta_pan):.1f}°, tilt={np.degrees(delta_tilt):.1f}°, roll={np.degrees(delta_roll):.1f}°)")
                rotations.append((cumulative_pan, cumulative_tilt, cumulative_roll))
                continue
            
            # Accumulate rotation
            cumulative_pan += delta_pan
            cumulative_tilt += delta_tilt
            cumulative_roll += delta_roll
            
            rotations.append((cumulative_pan, cumulative_tilt, cumulative_roll))
            
            if i % 10 == 0:
                print(f"[CameraSolver] Frame {i}/{num_frames}: pan={np.degrees(cumulative_pan):.2f}°, tilt={np.degrees(cumulative_tilt):.2f}°")
        
        # Apply smoothing
        if smoothing > 1:
            rotations = self.smooth_rotations(rotations, smoothing)
            print(f"[CameraSolver] Applied smoothing (window={smoothing})")
        
        # Build output
        camera_rotation_data = {
            "num_frames": num_frames,
            "image_width": img_width,
            "image_height": img_height,
            "focal_length_px": focal_length_px,
            "rotations": [
                {
                    "frame": i,
                    "pan": rot[0],      # radians
                    "tilt": rot[1],     # radians
                    "roll": rot[2],     # radians
                    "pan_deg": np.degrees(rot[0]),
                    "tilt_deg": np.degrees(rot[1]),
                    "roll_deg": np.degrees(rot[2]),
                }
                for i, rot in enumerate(rotations)
            ]
        }
        
        # Summary
        final_rot = rotations[-1]
        print(f"[CameraSolver] Solve complete!")
        print(f"[CameraSolver] Total rotation: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°, roll={np.degrees(final_rot[2]):.2f}°")
        
        # Convert debug masks to tensor (N, H, W)
        if fg_masks_for_debug is not None:
            debug_masks_tensor = torch.from_numpy(fg_masks_for_debug).float()
            print(f"[CameraSolver] Debug masks: shape={debug_masks_tensor.shape}, sum={debug_masks_tensor.sum():.0f}, max={debug_masks_tensor.max():.2f}")
        else:
            # Return empty masks if none available
            debug_masks_tensor = torch.zeros((num_frames, img_height, img_width), dtype=torch.float32)
            print(f"[CameraSolver] Debug masks: EMPTY (no foreground masks generated)")
        
        # Convert debug tracking frames to tensor (N, H, W, C)
        print(f"[CameraSolver] Debug tracking frames: {len(debug_tracking_frames)} frames")
        if len(debug_tracking_frames) > 0:
            print(f"[CameraSolver] First frame shape: {debug_tracking_frames[0].shape}, dtype: {debug_tracking_frames[0].dtype}, range: [{debug_tracking_frames[0].min()}-{debug_tracking_frames[0].max()}]")
        
        debug_tracking_array = np.stack(debug_tracking_frames, axis=0)
        debug_tracking_tensor = torch.from_numpy(debug_tracking_array).float() / 255.0
        print(f"[CameraSolver] Debug tracking tensor: shape={debug_tracking_tensor.shape}, range=[{debug_tracking_tensor.min():.2f}-{debug_tracking_tensor.max():.2f}]")
        
        return (camera_rotation_data, debug_masks_tensor, debug_tracking_tensor)
    
    def _process_sam3_masks(self, sam3_masks, num_frames, img_height, img_width) -> np.ndarray:
        """Convert SAM3 masks to background masks."""
        print(f"[CameraSolver] Processing SAM3 masks: type={type(sam3_masks)}")
        
        try:
            # Handle different SAM3 mask formats
            if isinstance(sam3_masks, dict):
                print(f"[CameraSolver] SAM3 masks is dict with {len(sam3_masks)} keys")
                # Frame-indexed dict
                bg_masks = []
                for i in range(num_frames):
                    if i in sam3_masks:
                        fg = sam3_masks[i]
                        if isinstance(fg, torch.Tensor):
                            fg = fg.cpu().numpy()
                        print(f"[CameraSolver] Frame {i} mask shape: {fg.shape}") if i == 0 else None
                        
                        # Handle various shapes
                        if len(fg.shape) == 3:
                            fg = fg[0] if fg.shape[0] < fg.shape[1] else fg[:,:,0]  # Remove channel dim
                        
                        if fg.shape[0] != img_height or fg.shape[1] != img_width:
                            if fg.shape[0] > 0 and fg.shape[1] > 0:
                                fg = cv2.resize(fg.astype(np.float32), (img_width, img_height))
                            else:
                                print(f"[CameraSolver] Frame {i}: Invalid mask shape {fg.shape}")
                                fg = np.zeros((img_height, img_width), dtype=np.float32)
                        bg_masks.append(1.0 - fg)
                    else:
                        bg_masks.append(np.ones((img_height, img_width), dtype=np.float32))
                result = np.array(bg_masks)
                print(f"[CameraSolver] Processed dict masks: shape={result.shape}, sum={result.sum():.0f}")
                return result
            
            elif isinstance(sam3_masks, torch.Tensor):
                fg = sam3_masks.cpu().numpy()
                print(f"[CameraSolver] SAM3 masks tensor shape: {fg.shape}")
                
                # Handle various tensor shapes: (N,H,W), (N,1,H,W), (N,C,H,W)
                if len(fg.shape) == 4:
                    fg = fg[:, 0]  # Take first channel: (N,1,H,W) -> (N,H,W)
                elif len(fg.shape) == 2:
                    fg = fg[np.newaxis, ...]  # Single mask: (H,W) -> (1,H,W)
                
                # Check if we need to resize
                if fg.shape[1] != img_height or fg.shape[2] != img_width:
                    print(f"[CameraSolver] Resizing masks from {fg.shape[1]}x{fg.shape[2]} to {img_height}x{img_width}")
                    resized = []
                    for i in range(fg.shape[0]):
                        if fg[i].shape[0] > 0 and fg[i].shape[1] > 0:
                            resized.append(cv2.resize(fg[i].astype(np.float32), (img_width, img_height)))
                        else:
                            resized.append(np.zeros((img_height, img_width), dtype=np.float32))
                    fg = np.stack(resized, axis=0)
                
                result = 1.0 - fg
                print(f"[CameraSolver] Processed tensor masks: shape={result.shape}, fg_sum={fg.sum():.0f}")
                return result
            
            elif hasattr(sam3_masks, '__iter__'):
                # List or other iterable
                print(f"[CameraSolver] SAM3 masks is iterable with {len(sam3_masks)} items")
                bg_masks = []
                for i, mask in enumerate(sam3_masks):
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    if len(mask.shape) == 3:
                        mask = mask[0]
                    if mask.shape[0] != img_height or mask.shape[1] != img_width:
                        if mask.shape[0] > 0 and mask.shape[1] > 0:
                            mask = cv2.resize(mask.astype(np.float32), (img_width, img_height))
                        else:
                            mask = np.zeros((img_height, img_width), dtype=np.float32)
                    bg_masks.append(1.0 - mask)
                result = np.array(bg_masks)
                print(f"[CameraSolver] Processed iterable masks: shape={result.shape}")
                return result
            
            else:
                print(f"[CameraSolver] Unknown SAM3 mask format: {type(sam3_masks)}")
                return None
                
        except Exception as e:
            print(f"[CameraSolver] Error processing SAM3 masks: {e}")
            import traceback
            traceback.print_exc()
            return None


# Node registration
NODE_CLASS_MAPPINGS = {
    "CameraRotationSolver": CameraRotationSolver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraRotationSolver": "Camera Rotation Solver",
}
