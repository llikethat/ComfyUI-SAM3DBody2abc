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
    
    Depth-based methods available for more robust tracking:
    - DepthAnything: Uses depth to prioritize distant (background) features
    - DUSt3R: AI-based 3D reconstruction for camera poses
    - COLMAP: Traditional Structure from Motion
    - DepthCrafter: Video-native temporally consistent depth
    """
    
    TRACKING_METHODS = [
        "KLT (Persistent)", 
        "CoTracker (AI)", 
        "ORB (Feature-Based)", 
        "RAFT (Dense Flow)",
        "DepthAnything + KLT",
        "DUSt3R (3D Reconstruction)",
        "COLMAP (Structure from Motion)",
        "DepthCrafter (Video Depth)"
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "foreground_masks": ("MASK",),
                "sam3_masks": ("SAM3_VIDEO_MASKS",),
                "depth_maps": ("IMAGE", {
                    "tooltip": "Pre-computed depth maps from DepthAnything V2, MiDaS, or other depth nodes. Connect this for depth-weighted tracking."
                }),
                "tracking_method": (cls.TRACKING_METHODS, {
                    "default": "KLT (Persistent)",
                    "tooltip": "KLT: Fast CPU tracking. CoTracker: AI-based (GPU). DepthAnything+KLT: Depth-weighted (requires depth_maps input). DUSt3R: 3D reconstruction. COLMAP: Structure from Motion. DepthCrafter: Video depth."
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
                "cotracker_coords": ("TRACKING_COORDS", {
                    "tooltip": "Optional: Tracking coordinates from CoTracker node (comfyui_cotracker_node). Format: (N, T, P, 2) where N=batch, T=frames, P=points, 2=xy"
                }),
                "cotracker_visibility": ("TRACKING_VISIBILITY", {
                    "tooltip": "Optional: Visibility mask from CoTracker node. Format: (N, T, P) boolean"
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable verbose debug output with detailed per-frame tracking info"
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
        # Depth models
        self.depth_anything_model = None
        self.depth_anything_transform = None
        self.duster_model = None
        self.depthcrafter_model = None
    
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
    
    def load_depth_anything(self):
        """Load DepthAnything V2 model for monocular depth estimation."""
        if self.depth_anything_model is not None:
            return True
        
        try:
            # Try to import depth_anything_v2
            print(f"[CameraSolver] Loading DepthAnything V2 model...")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Try different import methods
            try:
                # Method 1: HuggingFace transformers
                from transformers import pipeline
                self.depth_anything_model = pipeline(
                    task="depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.depth_anything_type = "pipeline"
                print(f"[CameraSolver] DepthAnything V2 loaded via HuggingFace pipeline")
                return True
            except Exception as e1:
                print(f"[CameraSolver] HuggingFace pipeline failed: {e1}")
                
                # Method 2: Try torch hub
                try:
                    self.depth_anything_model = torch.hub.load(
                        'LiheYoung/Depth-Anything', 
                        'depth_anything_vits14',
                        pretrained=True
                    )
                    self.depth_anything_model = self.depth_anything_model.to(self.device).eval()
                    self.depth_anything_type = "torch_hub"
                    print(f"[CameraSolver] DepthAnything loaded via torch hub")
                    return True
                except Exception as e2:
                    print(f"[CameraSolver] Torch hub failed: {e2}")
                    
                    # Method 3: Use MiDaS as fallback
                    try:
                        print(f"[CameraSolver] Falling back to MiDaS...")
                        self.depth_anything_model = torch.hub.load(
                            'intel-isl/MiDaS', 
                            'MiDaS_small',
                            pretrained=True
                        )
                        self.depth_anything_model = self.depth_anything_model.to(self.device).eval()
                        
                        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
                        self.depth_anything_transform = midas_transforms.small_transform
                        self.depth_anything_type = "midas"
                        print(f"[CameraSolver] MiDaS loaded as depth fallback")
                        return True
                    except Exception as e3:
                        print(f"[CameraSolver] MiDaS also failed: {e3}")
                        return False
                        
        except Exception as e:
            print(f"[CameraSolver] Failed to load depth model: {e}")
            return False
    
    def load_duster(self):
        """Load DUSt3R model for 3D reconstruction.
        Works with ComfyUI-dust3r custom node package.
        """
        if self.duster_model is not None:
            return True
        
        try:
            print(f"[CameraSolver] Loading DUSt3R model...")
            
            import sys
            import os
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Find ComfyUI-dust3r in custom_nodes
            possible_paths = []
            
            try:
                import folder_paths
                comfy_path = os.path.dirname(folder_paths.__file__)
                possible_paths.append(os.path.join(comfy_path, 'custom_nodes', 'ComfyUI-dust3r'))
                possible_paths.append(os.path.join(comfy_path, 'custom_nodes', 'ComfyUI-dust3r-main'))
            except ImportError:
                pass
            
            # Add common paths
            possible_paths.extend([
                '/workspace/ComfyUI/custom_nodes/ComfyUI-dust3r',
                '/workspace/ComfyUI/custom_nodes/ComfyUI-dust3r-main',
                os.path.expanduser('~/ComfyUI/custom_nodes/ComfyUI-dust3r'),
            ])
            
            dust3r_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    # Check if it has the dust3r subfolder
                    if os.path.exists(os.path.join(path, 'dust3r')):
                        dust3r_path = path
                        break
            
            if dust3r_path is None:
                print(f"[CameraSolver] ComfyUI-dust3r not found in custom_nodes")
                print(f"[CameraSolver] Install via ComfyUI Manager or clone from:")
                print(f"[CameraSolver]   https://github.com/chaojie/ComfyUI-dust3r")
                return False
            
            # Add to sys.path
            if dust3r_path not in sys.path:
                sys.path.insert(0, dust3r_path)
                print(f"[CameraSolver] Added {dust3r_path} to path")
            
            # Check for checkpoint
            checkpoint_dir = os.path.join(dust3r_path, 'checkpoints')
            checkpoint_path = os.path.join(checkpoint_dir, 'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
            
            if not os.path.exists(checkpoint_path):
                print(f"[CameraSolver] DUSt3R checkpoint not found")
                print(f"[CameraSolver] Download from: https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
                print(f"[CameraSolver] Place in: {checkpoint_dir}")
                return False
            
            # Import and load model with PyTorch 2.6+ compatibility
            try:
                # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
                print(f"[CameraSolver] Loading checkpoint from {checkpoint_path}...")
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Parse model args
                args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
                if 'landscape_only' not in args:
                    args = args[:-1] + ', landscape_only=False)'
                else:
                    args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
                
                print(f"[CameraSolver] Instantiating model: {args[:100]}...")
                
                # Import required modules
                from dust3r.model import AsymmetricCroCo3DStereo
                
                # Define inf for eval() - used in depth_mode and conf_mode
                inf = float('inf')
                
                # Create model instance
                net = eval(args)
                
                # Load weights
                s = net.load_state_dict(ckpt['model'], strict=False)
                print(f"[CameraSolver] Model loaded: {s}")
                
                self.duster_model = net.to(self.device).eval()
                self.duster_path = dust3r_path
                print(f"[CameraSolver] DUSt3R loaded successfully on {self.device}")
                return True
                
            except ImportError as e:
                print(f"[CameraSolver] Failed to import dust3r: {e}")
                print(f"[CameraSolver] Try: pip install roma huggingface-hub>=0.22")
                return False
                
        except Exception as e:
            print(f"[CameraSolver] Failed to load DUSt3R: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_depthcrafter(self):
        """Load DepthCrafter model for video depth estimation."""
        if self.depthcrafter_model is not None:
            return True
        
        try:
            print(f"[CameraSolver] Loading DepthCrafter model...")
            
            # DepthCrafter uses diffusers
            try:
                from diffusers import DiffusionPipeline
                
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                self.depthcrafter_model = DiffusionPipeline.from_pretrained(
                    "tencent/DepthCrafter",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.depthcrafter_model = self.depthcrafter_model.to(self.device)
                print(f"[CameraSolver] DepthCrafter loaded successfully")
                return True
            except Exception as e:
                print(f"[CameraSolver] DepthCrafter not available: {e}")
                print(f"[CameraSolver] Install with: pip install diffusers")
                return False
                
        except Exception as e:
            print(f"[CameraSolver] Failed to load DepthCrafter: {e}")
            return False
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single frame using loaded depth model.
        
        Args:
            frame: RGB image (H, W, C) uint8
            
        Returns:
            depth: Depth map (H, W) float32, higher = further
        """
        if self.depth_anything_model is None:
            return None
        
        h, w = frame.shape[:2]
        
        try:
            if self.depth_anything_type == "pipeline":
                # HuggingFace pipeline
                from PIL import Image
                img = Image.fromarray(frame)
                result = self.depth_anything_model(img)
                depth = np.array(result["depth"])
                depth = cv2.resize(depth, (w, h))
                return depth.astype(np.float32)
                
            elif self.depth_anything_type == "midas":
                # MiDaS
                input_batch = self.depth_anything_transform(frame).to(self.device)
                with torch.no_grad():
                    prediction = self.depth_anything_model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(h, w),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                depth = prediction.cpu().numpy()
                return depth.astype(np.float32)
                
            else:
                # torch_hub DepthAnything
                from PIL import Image
                img = Image.fromarray(frame)
                with torch.no_grad():
                    depth = self.depth_anything_model.infer_image(img)
                depth = cv2.resize(depth, (w, h))
                return depth.astype(np.float32)
                
        except Exception as e:
            print(f"[CameraSolver] Depth estimation failed: {e}")
            return None
    
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
        
        Uses camera convention:
        - Pan (yaw): Horizontal rotation around Y axis (left/right)
        - Tilt (pitch): Vertical rotation around X axis (up/down)
        - Roll: Rotation around Z axis (camera tilt sideways)
        
        For broadcast tripod shots, roll should be ~0.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            (pan, tilt, roll) in radians
        """
        # Use scipy if available for more robust decomposition
        try:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(R)
            # ZYX order: first pan (Y), then tilt (X), then roll (Z)
            # This gives us intrinsic rotations in camera order
            angles = r.as_euler('ZYX', degrees=False)
            roll, pan, tilt = angles[0], angles[1], angles[2]
            return (pan, tilt, roll)
        except ImportError:
            pass
        
        # Fallback: manual extraction using ZYX convention
        # R = Rz(roll) * Ry(pan) * Rx(tilt)
        
        # Check for gimbal lock (tilt near ±90°)
        if abs(R[2, 0]) < 0.9999:
            tilt = np.arcsin(-R[2, 0])
            pan = np.arctan2(R[2, 1], R[2, 2])
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock case
            roll = 0
            if R[2, 0] < 0:  # tilt = +90°
                tilt = np.pi / 2
                pan = np.arctan2(R[0, 1], R[0, 2])
            else:  # tilt = -90°
                tilt = -np.pi / 2
                pan = np.arctan2(-R[0, 1], -R[0, 2])
        
        return (pan, tilt, roll)
    
    def load_cotracker(self):
        """Load CoTracker model (downloads automatically on first use via torch.hub)."""
        if hasattr(self, 'cotracker_model') and self.cotracker_model is not None:
            return True
        
        try:
            import torch
            
            # Ensure device is set
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"[CameraSolver] Loading CoTracker model on {self.device} (auto-download on first use)...")
            
            # CoTracker loads via torch.hub - no pip install needed
            try:
                self.cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
                self.cotracker_model = self.cotracker_model.to(self.device)
                self.cotracker_model.eval()
                print(f"[CameraSolver] CoTracker v2 loaded successfully on {self.device}")
                return True
            except Exception as e:
                print(f"[CameraSolver] CoTracker v2 failed: {e}")
                print(f"[CameraSolver] Trying CoTracker v1...")
                try:
                    self.cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8")
                    self.cotracker_model = self.cotracker_model.to(self.device)
                    self.cotracker_model.eval()
                    print(f"[CameraSolver] CoTracker v1 loaded successfully on {self.device}")
                    return True
                except Exception as e2:
                    print(f"[CameraSolver] CoTracker loading failed: {e2}")
                    print(f"[CameraSolver] CoTracker requires internet connection for first download")
                    print(f"[CameraSolver] Falling back to KLT tracking...")
                    self.cotracker_model = None
                    return False
                    
        except Exception as e:
            print(f"[CameraSolver] CoTracker initialization error: {e}")
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
    
    def solve_rotation_from_cotracker_data(
        self,
        frames: np.ndarray,
        cotracker_coords: torch.Tensor,
        cotracker_visibility: Optional[torch.Tensor],
        bg_masks: Optional[np.ndarray],
        focal_length_px: float,
        ransac_threshold: float = 3.0,
        verbose_debug: bool = False
    ) -> Tuple[List[Tuple[float, float, float]], List[np.ndarray]]:
        """
        Solve camera rotation using external CoTracker tracking data from comfyui_cotracker_node.
        
        This allows using the CoTracker ComfyUI node (s9roll7/comfyui_cotracker_node) 
        for point tracking, then feeding the results here for camera rotation estimation.
        
        Args:
            frames: Video frames (N, H, W, 3) uint8 RGB
            cotracker_coords: Tracking coordinates from CoTracker node
                              Shape: (B, T, P, 2) where B=batch, T=frames, P=points, 2=xy
            cotracker_visibility: Optional visibility mask from CoTracker
                                  Shape: (B, T, P) boolean
            bg_masks: Background masks (N, H, W) float, 1.0 = background
            focal_length_px: Focal length in pixels
            ransac_threshold: RANSAC threshold
            verbose_debug: Print detailed per-frame info
            
        Returns:
            rotations: List of (pan, tilt, roll) tuples per frame
            debug_frames: List of debug visualization frames
        """
        num_frames, img_height, img_width = frames.shape[:3]
        cx, cy = img_width / 2, img_height / 2
        
        # Camera intrinsic matrix
        K = np.array([
            [focal_length_px, 0, cx],
            [0, focal_length_px, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Process CoTracker input
        # Expected shape: (B, T, P, 2) from comfyui_cotracker_node
        coords_np = cotracker_coords.cpu().numpy()
        
        # Handle different input shapes
        if len(coords_np.shape) == 4:
            # (B, T, P, 2) - standard format
            coords_np = coords_np[0]  # Remove batch dimension -> (T, P, 2)
        elif len(coords_np.shape) == 3:
            # (T, P, 2) - already correct
            pass
        else:
            print(f"[CameraSolver] CoTracker data unexpected shape: {coords_np.shape}")
            print(f"[CameraSolver] Expected (B, T, P, 2) or (T, P, 2)")
            return [(0.0, 0.0, 0.0)] * num_frames, [frames[i].copy() for i in range(num_frames)]
        
        track_frames, num_points, _ = coords_np.shape
        print(f"[CameraSolver] External CoTracker data: {num_points} points across {track_frames} frames")
        
        # Handle frame count mismatch
        if track_frames != num_frames:
            print(f"[CameraSolver] Warning: CoTracker has {track_frames} frames, video has {num_frames}")
            track_frames = min(track_frames, num_frames)
        
        # Process visibility if provided
        if cotracker_visibility is not None:
            vis_np = cotracker_visibility.cpu().numpy()
            if len(vis_np.shape) == 3:
                vis_np = vis_np[0]  # Remove batch dimension
        else:
            # Assume all visible
            vis_np = np.ones((track_frames, num_points), dtype=bool)
        
        # Reference points from frame 0
        pts_ref = coords_np[0]  # (P, 2)
        
        rotations = [(0.0, 0.0, 0.0)]  # Frame 0 is reference
        debug_frames = []
        
        # Create debug frame for frame 0
        debug_frame0 = frames[0].copy()
        for j, pt in enumerate(pts_ref):
            if vis_np[0, j] if j < vis_np.shape[1] else True:
                cv2.circle(debug_frame0, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
        cv2.putText(debug_frame0, f"CoTracker External Frame 0: {num_points} points", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        debug_frames.append(debug_frame0)
        
        for i in range(1, track_frames):
            # Get tracked points for this frame
            pts_i = coords_np[i]  # (P, 2)
            vis_i = vis_np[i] if i < vis_np.shape[0] else np.ones(num_points, dtype=bool)
            
            # Filter by visibility
            valid_mask = vis_i.astype(bool)
            
            # Also filter by background mask if available
            if bg_masks is not None and i < len(bg_masks):
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
            
            if len(pts_ref_good) < 8:
                if verbose_debug:
                    print(f"[CameraSolver] CoTracker External Frame {i}: Not enough valid points ({len(pts_ref_good)})")
                rotations.append(rotations[-1])
                debug_frame = frames[i].copy()
                cv2.putText(debug_frame, f"Frame {i}: {len(pts_ref_good)} points (need 8+)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                debug_frames.append(debug_frame)
                continue
            
            # Estimate Essential Matrix
            E, inliers_mask = cv2.findEssentialMat(
                pts_ref_good, pts_i_good, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=ransac_threshold
            )
            
            if E is None:
                if verbose_debug:
                    print(f"[CameraSolver] CoTracker External Frame {i}: Essential matrix failed")
                rotations.append(rotations[-1])
                debug_frame = frames[i].copy()
                debug_frames.append(debug_frame)
                continue
            
            inliers_mask = inliers_mask.ravel() if inliers_mask is not None else np.ones(len(pts_ref_good))
            inlier_count = np.sum(inliers_mask)
            inlier_ratio = inlier_count / len(pts_ref_good)
            
            # Recover rotation
            _, R, t, _ = cv2.recoverPose(E, pts_ref_good, pts_i_good, K, 
                                          mask=inliers_mask.copy().reshape(-1, 1).astype(np.uint8))
            
            pan, tilt, roll = self.rotation_matrix_to_euler(R)
            
            if verbose_debug or i % 10 == 0:
                print(f"[CameraSolver] CoTracker External Frame {i}: {inlier_count}/{len(pts_ref_good)} inliers ({inlier_ratio:.1%}), pan={np.degrees(pan):.2f}°, tilt={np.degrees(tilt):.2f}°")
            
            rotations.append((pan, tilt, roll))
            
            # Create debug visualization
            debug_frame = frames[i].copy()
            for j, (ref_pt, cur_pt) in enumerate(zip(pts_ref_good, pts_i_good)):
                color = (0, 255, 0) if inliers_mask[j] else (0, 0, 255)
                cv2.circle(debug_frame, (int(cur_pt[0]), int(cur_pt[1])), 4, color, -1)
                cv2.line(debug_frame, (int(ref_pt[0]), int(ref_pt[1])), 
                        (int(cur_pt[0]), int(cur_pt[1])), color, 1)
            
            cv2.putText(debug_frame, f"CoTracker External {i}: pan={np.degrees(pan):.1f}° ({inlier_count}/{len(pts_ref_good)})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            debug_frames.append(debug_frame)
        
        # Pad rotations if track_frames < num_frames
        while len(rotations) < num_frames:
            rotations.append(rotations[-1])
            debug_frames.append(frames[len(debug_frames)].copy())
        
        print(f"[CameraSolver] CoTracker External solve complete!")
        return rotations, debug_frames
    
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
        
        # Use shared Euler angle extraction for consistency
        return self.rotation_matrix_to_euler(R_clean)
    
    def reject_outliers(self, rotations: List[Tuple[float, float, float]], max_delta_degrees: float = 10.0) -> List[Tuple[float, float, float]]:
        """
        Reject outlier frames where rotation changes too drastically between consecutive frames.
        
        Outliers are detected when:
        - Pan or tilt changes by more than max_delta_degrees between frames
        - Values exceed reasonable ranges (e.g., tilt > 45° is suspicious)
        
        Outliers are replaced with linearly interpolated values from valid neighbors.
        """
        if len(rotations) < 3:
            return rotations
        
        import math
        max_delta_rad = math.radians(max_delta_degrees)
        max_absolute_rad = math.radians(45.0)  # Max reasonable camera tilt/pan
        
        pans = [r[0] for r in rotations]
        tilts = [r[1] for r in rotations]
        rolls = [r[2] for r in rotations]
        
        # Identify outlier frames
        outlier_frames = set()
        
        for i in range(1, len(rotations)):
            # Check for large frame-to-frame jumps
            pan_delta = abs(pans[i] - pans[i-1])
            tilt_delta = abs(tilts[i] - tilts[i-1])
            
            # Check for unreasonable absolute values
            is_absolute_outlier = abs(tilts[i]) > max_absolute_rad or abs(pans[i]) > max_absolute_rad
            
            # Check for large jumps
            is_delta_outlier = pan_delta > max_delta_rad or tilt_delta > max_delta_rad
            
            if is_absolute_outlier or is_delta_outlier:
                outlier_frames.add(i)
                if is_absolute_outlier:
                    print(f"[CameraSolver] Frame {i}: OUTLIER (absolute value too large: pan={math.degrees(pans[i]):.1f}°, tilt={math.degrees(tilts[i]):.1f}°)")
                else:
                    print(f"[CameraSolver] Frame {i}: OUTLIER (delta too large: Δpan={math.degrees(pan_delta):.1f}°, Δtilt={math.degrees(tilt_delta):.1f}°)")
        
        if not outlier_frames:
            print(f"[CameraSolver] No outliers detected (threshold={max_delta_degrees}°)")
            return rotations
        
        print(f"[CameraSolver] Detected {len(outlier_frames)} outlier frames, interpolating...")
        
        # Interpolate outliers from valid neighbors
        def interpolate_outliers(values, outliers):
            result = values.copy()
            for i in outliers:
                # Find nearest valid neighbors
                left_idx = i - 1
                while left_idx in outliers and left_idx > 0:
                    left_idx -= 1
                
                right_idx = i + 1
                while right_idx in outliers and right_idx < len(values) - 1:
                    right_idx += 1
                
                # Check if neighbors are valid
                left_valid = left_idx >= 0 and left_idx not in outliers
                right_valid = right_idx < len(values) and right_idx not in outliers
                
                if left_valid and right_valid:
                    # Linear interpolation
                    t = (i - left_idx) / (right_idx - left_idx)
                    result[i] = values[left_idx] + t * (values[right_idx] - values[left_idx])
                elif left_valid:
                    result[i] = values[left_idx]
                elif right_valid:
                    result[i] = values[right_idx]
                else:
                    result[i] = 0.0  # Fallback to zero
            
            return result
        
        fixed_pans = interpolate_outliers(pans, outlier_frames)
        fixed_tilts = interpolate_outliers(tilts, outlier_frames)
        fixed_rolls = interpolate_outliers(rolls, outlier_frames)
        
        # Ensure frame 0 stays at (0, 0, 0)
        fixed_pans[0] = 0.0
        fixed_tilts[0] = 0.0
        fixed_rolls[0] = 0.0
        
        print(f"[CameraSolver] Outlier rejection complete, {len(outlier_frames)} frames interpolated")
        
        return list(zip(fixed_pans, fixed_tilts, fixed_rolls))
    
    def smooth_rotations(self, rotations: List[Tuple[float, float, float]], window: int) -> List[Tuple[float, float, float]]:
        """
        Apply moving average smoothing to rotation values.
        Frame 0 is always preserved as (0,0,0) since it's the reference.
        """
        if window <= 1 or len(rotations) < window:
            return rotations
        
        pans = [r[0] for r in rotations]
        tilts = [r[1] for r in rotations]
        rolls = [r[2] for r in rotations]
        
        def smooth_array(values):
            result = [values[0]]  # Preserve frame 0
            half = window // 2
            for i in range(1, len(values)):  # Start from frame 1
                start = max(1, i - half)  # Don't include frame 0 in smoothing window
                end = min(len(values), i + half + 1)
                avg = sum(values[start:end]) / (end - start)
                result.append(avg)
            return result
        
        smoothed_pans = smooth_array(pans)
        smoothed_tilts = smooth_array(tilts)
        smoothed_rolls = smooth_array(rolls)
        
        # Ensure frame 0 is exactly (0, 0, 0)
        return [(0.0, 0.0, 0.0)] + list(zip(smoothed_pans[1:], smoothed_tilts[1:], smoothed_rolls[1:]))
    
    def solve_camera_rotation(
        self,
        images: torch.Tensor,
        foreground_masks: Optional[torch.Tensor] = None,
        sam3_masks: Optional[Any] = None,
        depth_maps: Optional[torch.Tensor] = None,
        tracking_method: str = "ORB (Feature-Based)",
        auto_mask_people: bool = True,
        detection_confidence: float = 0.5,
        mask_expansion: int = 20,
        focal_length_px: float = 1000.0,
        flow_threshold: float = 1.0,
        ransac_threshold: float = 3.0,
        smoothing: int = 5,
        cotracker_coords: Optional[torch.Tensor] = None,
        cotracker_visibility: Optional[torch.Tensor] = None,
        verbose_debug: bool = False,
    ) -> Tuple[Dict]:
        """
        Solve for camera rotation from video frames.
        
        Args:
            images: Video frames (N, H, W, C)
            foreground_masks: Foreground masks to exclude (N, H, W)
            sam3_masks: SAM3 video masks (alternative to foreground_masks)
            depth_maps: Pre-computed depth maps from external nodes (N, H, W, C) or (N, H, W)
            tracking_method: "ORB (Feature-Based)" or "RAFT (Dense Flow)"
            auto_mask_people: Automatically detect and mask all people using YOLO
            detection_confidence: YOLO detection confidence threshold
            mask_expansion: Pixels to expand detected masks
            focal_length_px: Focal length in pixels
            flow_threshold: Minimum flow magnitude (RAFT only)
            ransac_threshold: RANSAC threshold
            smoothing: Temporal smoothing window
            cotracker_coords: Optional tracking coordinates from CoTracker node (N, T, P, 2)
            cotracker_visibility: Optional visibility mask from CoTracker node (N, T, P)
            verbose_debug: Enable verbose per-frame debug output
            
        Returns:
            camera_rotations: Dict with per-frame rotation data
        """
        # Check tracking method - order matters! Check specific methods before generic ones
        use_depth_klt = "DepthAnything" in tracking_method and "KLT" in tracking_method  # Only for "DepthAnything + KLT"
        use_depthcrafter = "DepthCrafter" in tracking_method
        use_duster = "DUSt3R" in tracking_method
        use_cotracker = "CoTracker" in tracking_method
        use_klt = "KLT" in tracking_method and not use_depth_klt  # Regular KLT only if not depth-weighted
        use_orb = "ORB" in tracking_method
        use_raft = "RAFT" in tracking_method
        
        print(f"[CameraSolver] Starting camera rotation solve...")
        print(f"[CameraSolver] Tracking method: {tracking_method}")
        print(f"[CameraSolver] Input: {images.shape[0]} frames")
        if verbose_debug:
            print(f"[CameraSolver] Verbose debug enabled")
        
        # Check for CoTracker external tracking data
        if cotracker_coords is not None:
            print(f"[CameraSolver] External CoTracker coords provided: shape={cotracker_coords.shape}")
            if cotracker_visibility is not None:
                print(f"[CameraSolver] External CoTracker visibility provided: shape={cotracker_visibility.shape}")
        
        # Process external depth maps if provided
        external_depths = None
        if depth_maps is not None:
            print(f"[CameraSolver] External depth maps provided: shape={depth_maps.shape}")
            depth_np = depth_maps.cpu().numpy()
            # Handle different shapes: (N, H, W, C), (N, H, W), (N, C, H, W)
            if len(depth_np.shape) == 4:
                if depth_np.shape[-1] in [1, 3]:  # (N, H, W, C)
                    depth_np = depth_np.mean(axis=-1)  # Convert to grayscale
                elif depth_np.shape[1] in [1, 3]:  # (N, C, H, W)
                    depth_np = depth_np.mean(axis=1)  # Convert to grayscale
            # Normalize to 0-1 range
            if depth_np.max() > 1.0:
                depth_np = depth_np / 255.0
            external_depths = depth_np.astype(np.float32)
            print(f"[CameraSolver] Processed depth maps: shape={external_depths.shape}, range=[{external_depths.min():.3f}-{external_depths.max():.3f}]")
        
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
            
            # Reject outliers before smoothing
            rotations = self.reject_outliers(rotations, max_delta_degrees=10.0)
            
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
        if use_cotracker:
            # Check if external CoTracker data is provided
            if cotracker_coords is not None:
                print(f"[CameraSolver] Using external CoTracker tracking data")
                rotations, debug_tracking_frames = self.solve_rotation_from_cotracker_data(
                    frames, cotracker_coords, cotracker_visibility, 
                    bg_masks, focal_length_px, ransac_threshold, verbose_debug
                )
            else:
                print(f"[CameraSolver] Using internal CoTracker AI-based tracking (handles occlusion)")
                rotations, debug_tracking_frames, track_info = self.solve_rotation_cotracker(
                    frames, bg_masks, focal_length_px, ransac_threshold
                )
            
            # Reject outliers before smoothing
            rotations = self.reject_outliers(rotations, max_delta_degrees=10.0)
            
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
                "tracking_method": "CoTracker (AI)" + (" - External" if cotracker_coords is not None else ""),
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
        
        # ==== DEPTH-BASED TRACKING METHODS ====
        
        # DepthAnything + KLT: Use depth to weight background features
        if use_depth_klt:
            print(f"[CameraSolver] Using DepthAnything + KLT (depth-weighted tracking)")
            if external_depths is not None:
                print(f"[CameraSolver] Using externally provided depth maps")
            rotations, debug_tracking_frames = self.solve_rotation_depth_klt(
                frames, bg_masks, focal_length_px, ransac_threshold, external_depths
            )
            
            # Reject outliers and smooth
            rotations = self.reject_outliers(rotations, max_delta_degrees=10.0)
            if smoothing > 1:
                rotations = self.smooth_rotations(rotations, smoothing)
                print(f"[CameraSolver] Applied smoothing (window={smoothing})")
            
            # Build output
            camera_rotation_data = {
                "num_frames": num_frames,
                "image_width": img_width,
                "image_height": img_height,
                "focal_length_px": focal_length_px,
                "tracking_method": "DepthAnything + KLT",
                "rotations": [
                    {"frame": i, "pan": rot[0], "tilt": rot[1], "roll": rot[2],
                     "pan_deg": np.degrees(rot[0]), "tilt_deg": np.degrees(rot[1]), "roll_deg": np.degrees(rot[2])}
                    for i, rot in enumerate(rotations)
                ]
            }
            
            final_rot = rotations[-1] if rotations else (0, 0, 0)
            print(f"[CameraSolver] Solve complete! Total rotation: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°")
            
            debug_masks_tensor = torch.from_numpy(fg_masks_for_debug).float() if fg_masks_for_debug is not None else torch.zeros((num_frames, img_height, img_width), dtype=torch.float32)
            debug_tracking_array = np.stack(debug_tracking_frames, axis=0)
            debug_tracking_tensor = torch.from_numpy(debug_tracking_array).float() / 255.0
            
            return (camera_rotation_data, debug_masks_tensor, debug_tracking_tensor)
        
        # DUSt3R: AI-based 3D reconstruction
        if use_duster:
            print(f"[CameraSolver] Using DUSt3R (3D reconstruction for camera poses)")
            rotations, debug_tracking_frames = self.solve_rotation_duster(
                frames, bg_masks, focal_length_px
            )
            
            # Reject outliers and smooth
            rotations = self.reject_outliers(rotations, max_delta_degrees=10.0)
            if smoothing > 1:
                rotations = self.smooth_rotations(rotations, smoothing)
                print(f"[CameraSolver] Applied smoothing (window={smoothing})")
            
            # Build output
            camera_rotation_data = {
                "num_frames": num_frames,
                "image_width": img_width,
                "image_height": img_height,
                "focal_length_px": focal_length_px,
                "tracking_method": "DUSt3R (3D Reconstruction)",
                "rotations": [
                    {"frame": i, "pan": rot[0], "tilt": rot[1], "roll": rot[2],
                     "pan_deg": np.degrees(rot[0]), "tilt_deg": np.degrees(rot[1]), "roll_deg": np.degrees(rot[2])}
                    for i, rot in enumerate(rotations)
                ]
            }
            
            final_rot = rotations[-1] if rotations else (0, 0, 0)
            print(f"[CameraSolver] Solve complete! Total rotation: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°")
            
            debug_masks_tensor = torch.from_numpy(fg_masks_for_debug).float() if fg_masks_for_debug is not None else torch.zeros((num_frames, img_height, img_width), dtype=torch.float32)
            debug_tracking_array = np.stack(debug_tracking_frames, axis=0)
            debug_tracking_tensor = torch.from_numpy(debug_tracking_array).float() / 255.0
            
            return (camera_rotation_data, debug_masks_tensor, debug_tracking_tensor)
        
        # COLMAP: Structure from Motion
        use_colmap = "COLMAP" in tracking_method
        if use_colmap:
            print(f"[CameraSolver] Using COLMAP (Structure from Motion)")
            rotations, debug_tracking_frames = self.solve_rotation_colmap(
                frames, focal_length_px
            )
            
            # Reject outliers and smooth
            rotations = self.reject_outliers(rotations, max_delta_degrees=10.0)
            if smoothing > 1:
                rotations = self.smooth_rotations(rotations, smoothing)
                print(f"[CameraSolver] Applied smoothing (window={smoothing})")
            
            # Build output
            camera_rotation_data = {
                "num_frames": num_frames,
                "image_width": img_width,
                "image_height": img_height,
                "focal_length_px": focal_length_px,
                "tracking_method": "COLMAP (Structure from Motion)",
                "rotations": [
                    {"frame": i, "pan": rot[0], "tilt": rot[1], "roll": rot[2],
                     "pan_deg": np.degrees(rot[0]), "tilt_deg": np.degrees(rot[1]), "roll_deg": np.degrees(rot[2])}
                    for i, rot in enumerate(rotations)
                ]
            }
            
            final_rot = rotations[-1] if rotations else (0, 0, 0)
            print(f"[CameraSolver] Solve complete! Total rotation: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°")
            
            debug_masks_tensor = torch.from_numpy(fg_masks_for_debug).float() if fg_masks_for_debug is not None else torch.zeros((num_frames, img_height, img_width), dtype=torch.float32)
            debug_tracking_array = np.stack(debug_tracking_frames, axis=0)
            debug_tracking_tensor = torch.from_numpy(debug_tracking_array).float() / 255.0
            
            return (camera_rotation_data, debug_masks_tensor, debug_tracking_tensor)
        
        # DepthCrafter: Video-native depth
        if use_depthcrafter:
            print(f"[CameraSolver] Using DepthCrafter (Video-native depth)")
            rotations, debug_tracking_frames = self.solve_rotation_depthcrafter(
                frames, bg_masks, focal_length_px
            )
            
            # Reject outliers and smooth
            rotations = self.reject_outliers(rotations, max_delta_degrees=10.0)
            if smoothing > 1:
                rotations = self.smooth_rotations(rotations, smoothing)
                print(f"[CameraSolver] Applied smoothing (window={smoothing})")
            
            # Build output
            camera_rotation_data = {
                "num_frames": num_frames,
                "image_width": img_width,
                "image_height": img_height,
                "focal_length_px": focal_length_px,
                "tracking_method": "DepthCrafter (Video Depth)",
                "rotations": [
                    {"frame": i, "pan": rot[0], "tilt": rot[1], "roll": rot[2],
                     "pan_deg": np.degrees(rot[0]), "tilt_deg": np.degrees(rot[1]), "roll_deg": np.degrees(rot[2])}
                    for i, rot in enumerate(rotations)
                ]
            }
            
            final_rot = rotations[-1] if rotations else (0, 0, 0)
            print(f"[CameraSolver] Solve complete! Total rotation: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°")
            
            debug_masks_tensor = torch.from_numpy(fg_masks_for_debug).float() if fg_masks_for_debug is not None else torch.zeros((num_frames, img_height, img_width), dtype=torch.float32)
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
        
        # Reject outliers before smoothing
        rotations = self.reject_outliers(rotations, max_delta_degrees=10.0)
        
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
    
    # ==== DEPTH-BASED TRACKING IMPLEMENTATIONS ====
    
    def solve_rotation_depth_klt(self, frames, bg_masks, focal_length_px, ransac_threshold, external_depths=None):
        """
        Depth-weighted KLT tracking - filters to keep only distant (background) features.
        
        Uses same persistent tracking approach as regular KLT, but:
        1. Estimates depth for all potential features
        2. Filters to keep only features with HIGH depth (distant = background)
        3. Tracks these distant features persistently from frame 0
        
        Args:
            external_depths: Pre-computed depth maps (N, H, W) - if provided, skip internal depth estimation
        """
        num_frames, img_height, img_width = frames.shape[:3]
        cx, cy = img_width / 2, img_height / 2
        
        # Camera intrinsic matrix
        K = np.array([
            [focal_length_px, 0, cx],
            [0, focal_length_px, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Get depth maps
        if external_depths is not None:
            print(f"[CameraSolver] Depth-KLT: Using {len(external_depths)} external depth maps")
            depths = []
            for i in range(num_frames):
                if i < len(external_depths):
                    depth = external_depths[i]
                    if depth.shape[0] != img_height or depth.shape[1] != img_width:
                        depth = cv2.resize(depth, (img_width, img_height))
                    depths.append(depth.astype(np.float32))
                else:
                    depths.append(np.ones((img_height, img_width), dtype=np.float32))
        else:
            # Load depth model
            if not self.load_depth_anything():
                print(f"[CameraSolver] Depth-KLT: No depth available, falling back to regular KLT")
                return self.solve_rotation_klt_persistent(frames, bg_masks, focal_length_px, ransac_threshold)[:2]
            
            print(f"[CameraSolver] Depth-KLT: Estimating depth for {num_frames} frames...")
            depths = []
            for i in range(num_frames):
                depth = self.estimate_depth(frames[i])
                if depth is not None:
                    depths.append(depth)
                else:
                    depths.append(np.ones((img_height, img_width), dtype=np.float32))
                if i % 10 == 0:
                    print(f"[CameraSolver] Depth estimated for frame {i}/{num_frames}")
        
        # Convert first frame to grayscale
        gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        
        # Create mask for feature detection (background only from YOLO)
        detection_mask = None
        if bg_masks is not None:
            detection_mask = (bg_masks[0] * 255).astype(np.uint8)
        
        # Detect features in frame 0
        feature_params = dict(
            maxCorners=1000,  # Detect more, we'll filter by depth
            qualityLevel=0.005,
            minDistance=10,
            blockSize=7,
            mask=detection_mask
        )
        
        pts0 = cv2.goodFeaturesToTrack(gray0, **feature_params)
        
        if pts0 is None or len(pts0) < 50:
            print(f"[CameraSolver] Depth-KLT: Not enough features ({len(pts0) if pts0 else 0}), falling back to regular KLT")
            return self.solve_rotation_klt_persistent(frames, bg_masks, focal_length_px, ransac_threshold)[:2]
        
        pts0 = pts0.reshape(-1, 2)
        print(f"[CameraSolver] Depth-KLT: Detected {len(pts0)} candidate features")
        
        # Get depth at each feature location
        depth0 = depths[0]
        feature_depths = []
        for pt in pts0:
            x, y = int(pt[0]), int(pt[1])
            x = min(max(x, 0), img_width - 1)
            y = min(max(y, 0), img_height - 1)
            feature_depths.append(depth0[y, x])
        
        feature_depths = np.array(feature_depths)
        
        # Filter to keep only DISTANT features (high depth = far = background)
        # Keep top 40% by depth
        depth_threshold = np.percentile(feature_depths, 60)  # Top 40%
        distant_mask = feature_depths >= depth_threshold
        
        pts0_filtered = pts0[distant_mask]
        
        if len(pts0_filtered) < 30:
            print(f"[CameraSolver] Depth-KLT: Not enough distant features ({len(pts0_filtered)}), using all features")
            pts0_filtered = pts0
        else:
            print(f"[CameraSolver] Depth-KLT: Filtered to {len(pts0_filtered)} distant features (depth >= {depth_threshold:.3f})")
        
        original_pts0 = pts0_filtered.copy()
        
        # KLT tracking parameters
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Track features across all frames (PERSISTENT from frame 0)
        rotations = [(0.0, 0.0, 0.0)]  # Frame 0 is reference
        debug_frames = []
        
        # Create debug frame for frame 0
        debug_frame0 = frames[0].copy()
        for pt in pts0_filtered:
            cv2.circle(debug_frame0, tuple(pt.astype(int)), 4, (0, 255, 0), -1)
        cv2.putText(debug_frame0, f"Depth-KLT Frame 0: {len(pts0_filtered)} distant features", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        debug_frames.append(debug_frame0)
        
        # Current tracked points and their correspondence to original points
        current_pts = pts0_filtered.copy()
        valid_mask = np.ones(len(pts0_filtered), dtype=bool)
        prev_gray = gray0.copy()
        
        for i in range(1, num_frames):
            # Convert current frame to grayscale
            gray_i = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Track from previous frame to current frame
            pts_to_track = current_pts[valid_mask].reshape(-1, 1, 2).astype(np.float32)
            
            if len(pts_to_track) < 8:
                print(f"[CameraSolver] Depth-KLT Frame {i}: Too few points ({len(pts_to_track)})")
                rotations.append(rotations[-1])
                debug_frames.append(frames[i].copy())
                prev_gray = gray_i.copy()
                continue
            
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray_i, pts_to_track, None, **lk_params
            )
            
            if next_pts is None:
                rotations.append(rotations[-1])
                debug_frames.append(frames[i].copy())
                prev_gray = gray_i.copy()
                continue
            
            next_pts = next_pts.reshape(-1, 2)
            status = status.reshape(-1)
            
            # Update valid mask
            valid_indices = np.where(valid_mask)[0]
            new_valid_mask = valid_mask.copy()
            
            for j, idx in enumerate(valid_indices):
                if status[j] != 1:
                    new_valid_mask[idx] = False
                else:
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
            
            if len(pts0_good) < 8:
                print(f"[CameraSolver] Depth-KLT Frame {i}: Not enough tracks ({len(pts0_good)})")
                rotations.append(rotations[-1])
                debug_frame = frames[i].copy()
                cv2.putText(debug_frame, f"Depth-KLT Frame {i}: {len(pts0_good)} tracks (need 8+)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                debug_frames.append(debug_frame)
                prev_gray = gray_i.copy()
                continue
            
            # Estimate Essential Matrix (comparing frame 0 to frame i directly)
            E, inliers_mask = cv2.findEssentialMat(
                pts0_good, pts_i_good, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=ransac_threshold
            )
            
            if E is None:
                rotations.append(rotations[-1])
                debug_frames.append(frames[i].copy())
                prev_gray = gray_i.copy()
                continue
            
            inliers_mask = inliers_mask.ravel() if inliers_mask is not None else np.ones(len(pts0_good))
            inlier_count = np.sum(inliers_mask)
            inlier_ratio = inlier_count / len(pts0_good)
            
            # Recover rotation from Essential Matrix
            _, R, t, mask_pose = cv2.recoverPose(E, pts0_good, pts_i_good, K, 
                                                  mask=inliers_mask.copy().reshape(-1, 1).astype(np.uint8))
            
            # Extract Euler angles from rotation matrix
            pan, tilt, roll = self.rotation_matrix_to_euler(R)
            
            # Log progress
            print(f"[CameraSolver] Depth-KLT Frame {i}: {inlier_count}/{len(pts0_good)} inliers ({inlier_ratio:.1%}), pan={np.degrees(pan):.2f}°, tilt={np.degrees(tilt):.2f}°")
            
            rotations.append((pan, tilt, roll))
            
            # Create debug visualization
            debug_frame = frames[i].copy()
            for j, (old, new) in enumerate(zip(pts0_good, pts_i_good)):
                color = (0, 255, 0) if inliers_mask[j] else (0, 0, 255)
                cv2.circle(debug_frame, tuple(new.astype(int)), 4, color, -1)
                cv2.line(debug_frame, tuple(old.astype(int)), tuple(new.astype(int)), color, 1)
            
            cv2.putText(debug_frame, f"Depth-KLT Frame {i}: pan={np.degrees(pan):.1f}° ({inlier_count}/{len(pts0_good)} inliers)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            debug_frames.append(debug_frame)
            
            prev_gray = gray_i.copy()
        
        print(f"[CameraSolver] Depth-KLT tracking complete!")
        return rotations, debug_frames
    
    def solve_rotation_duster(self, frames, bg_masks, focal_length_px):
        """
        Use DUSt3R for 3D reconstruction and camera pose estimation.
        DUSt3R directly predicts relative camera poses between frame pairs.
        
        This method uses the ComfyUI-dust3r package which must be installed
        in ComfyUI/custom_nodes/ComfyUI-dust3r
        
        Args:
            frames: Video frames (N, H, W, 3)
            bg_masks: Background masks (N, H, W) - 1 for background, 0 for foreground/people
            focal_length_px: Focal length in pixels
        """
        num_frames = frames.shape[0]
        img_height, img_width = frames.shape[1:3]
        
        if not self.load_duster():
            print(f"[CameraSolver] DUSt3R not available, falling back to KLT")
            return self.solve_rotation_klt_persistent(frames, None, focal_length_px, 3.0)[:2]
        
        print(f"[CameraSolver] DUSt3R: Processing {num_frames} frames...")
        
        # Monkey-patch torch.linalg.inv BEFORE importing dust3r modules
        # This is needed because PyTorch 2.x doesn't support BFloat16 for linalg.inv
        original_linalg_inv = torch.linalg.inv
        def patched_linalg_inv(input, *, out=None):
            if input.dtype == torch.bfloat16:
                result = original_linalg_inv(input.float(), out=out)
                return result  # Keep as float32
            return original_linalg_inv(input, out=out)
        torch.linalg.inv = patched_linalg_inv
        print(f"[CameraSolver] DUSt3R: Patched torch.linalg.inv for BFloat16 compatibility")
        
        try:
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            from dust3r.utils.device import to_numpy
            import dust3r.utils.geometry as dust3r_geometry
            import dust3r.cloud_opt.init_im_poses as init_im_poses_module
            import dust3r.cloud_opt.pair_viewer as pair_viewer_module
            import dust3r.cloud_opt.base_opt as base_opt_module
            import tempfile
            import os
            from PIL import Image
            
            # Try to import losses module too
            try:
                import dust3r.losses as losses_module
            except ImportError:
                losses_module = None
            
            # Patch inv in ALL modules that import it directly
            # Multiple modules do "from geometry import inv" so they each have their own reference
            def patched_inv(mat):
                if mat.dtype == torch.bfloat16:
                    return original_linalg_inv(mat.float())
                return original_linalg_inv(mat)
            
            original_geom_inv = dust3r_geometry.inv
            dust3r_geometry.inv = patched_inv
            
            # Patch all cloud_opt modules that import inv directly
            modules_to_patch = [init_im_poses_module, pair_viewer_module, base_opt_module]
            if losses_module:
                modules_to_patch.append(losses_module)
            
            for module in modules_to_patch:
                if hasattr(module, 'inv'):
                    setattr(module, 'inv', patched_inv)
            
            print(f"[CameraSolver] DUSt3R: Patched inv in geometry + {len(modules_to_patch)} other modules")
            
            # Save frames to temp directory for DUSt3R
            with tempfile.TemporaryDirectory() as tmpdir:
                image_paths = []
                
                # Check if we have background masks
                use_masks = bg_masks is not None and len(bg_masks) == num_frames
                if use_masks:
                    print(f"[CameraSolver] DUSt3R: Applying background masks (masking out people)")
                
                for i, frame in enumerate(frames):
                    # Apply background mask if available (black out people/foreground)
                    if use_masks:
                        mask = bg_masks[i]
                        # Expand mask to 3 channels if needed
                        if mask.ndim == 2:
                            mask_3d = mask[:, :, np.newaxis]
                        else:
                            mask_3d = mask
                        # Apply mask: keep background (mask=1), black out foreground (mask=0)
                        masked_frame = (frame * mask_3d).astype(np.uint8)
                    else:
                        masked_frame = frame
                    
                    path = os.path.join(tmpdir, f"frame_{i:04d}.png")
                    img = Image.fromarray(masked_frame)
                    img.save(path)
                    image_paths.append(path)
                
                print(f"[CameraSolver] DUSt3R: Loading {len(image_paths)} images...")
                
                # Load images for DUSt3R (resizes to 512)
                images = load_images(image_paths, size=512)
                
                print(f"[CameraSolver] DUSt3R: Creating image pairs...")
                
                # Create pairs - use 'complete' for small sequences, 'swin' for longer
                if num_frames <= 10:
                    scene_graph = 'complete'
                else:
                    scene_graph = 'swin-5'  # sliding window of 5
                
                pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
                print(f"[CameraSolver] DUSt3R: Created {len(pairs)} pairs using '{scene_graph}' graph")
                
                # Run inference in float32 mode (disable BFloat16/autocast)
                print(f"[CameraSolver] DUSt3R: Running inference (float32 mode)...")
                
                # Ensure model is in float32
                self.duster_model = self.duster_model.float()
                
                # Disable autocast to prevent BFloat16
                with torch.cuda.amp.autocast(enabled=False):
                    output = inference(pairs, self.duster_model, self.device, batch_size=1)
                
                print(f"[CameraSolver] DUSt3R: Inference complete")
                
                # Global alignment to get camera poses
                print(f"[CameraSolver] DUSt3R: Running global alignment...")
                mode = GlobalAlignerMode.PointCloudOptimizer if num_frames > 2 else GlobalAlignerMode.PairViewer
                
                # Create scene (don't convert to float32 as it breaks gradients)
                # The patched inv functions handle BFloat16 conversion
                with torch.cuda.amp.autocast(enabled=False):
                    scene = global_aligner(output, device=self.device, mode=mode)
                
                # For PointCloudOptimizer, use MST initialization only (skip gradient optimization)
                # dust3r's inference uses @torch.no_grad(), so outputs don't have gradients
                # MST initialization uses PnP RANSAC which doesn't need gradients
                if mode == GlobalAlignerMode.PointCloudOptimizer:
                    print(f"[CameraSolver] DUSt3R: Running MST initialization (no gradient refinement)")
                    # Use the already-imported and patched init_im_poses module
                    with torch.no_grad():
                        init_im_poses_module.init_minimum_spanning_tree(scene, niter_PnP=100)
                    print(f"[CameraSolver] DUSt3R: MST initialization complete")
                
                # Extract camera poses (cam-to-world 4x4 matrices)
                poses = scene.get_im_poses()  # Tensor of shape (N, 4, 4)
                poses_np = to_numpy(poses)
                
                print(f"[CameraSolver] DUSt3R: Extracted {len(poses_np)} camera poses")
                
                # Convert to rotations relative to frame 0
                rotations = [(0.0, 0.0, 0.0)]
                pose0_inv = np.linalg.inv(poses_np[0])
                
                for i in range(1, len(poses_np)):
                    # Get relative pose from frame 0 to frame i
                    pose_rel = pose0_inv @ poses_np[i]
                    R = pose_rel[:3, :3]
                    
                    # Convert rotation matrix to Euler angles using our consistent method
                    pan, tilt, roll = self.rotation_matrix_to_euler(R)
                    
                    if i % 10 == 0 or i == len(poses_np) - 1:
                        print(f"[CameraSolver] DUSt3R Frame {i}: pan={np.degrees(pan):.2f}°, tilt={np.degrees(tilt):.2f}°, roll={np.degrees(roll):.2f}°")
                    
                    rotations.append((pan, tilt, roll))
                
                print(f"[CameraSolver] DUSt3R: Pose extraction complete!")
                final_rot = rotations[-1]
                print(f"[CameraSolver] DUSt3R Final: pan={np.degrees(final_rot[0]):.2f}°, tilt={np.degrees(final_rot[1]):.2f}°")
                
                # Debug frames with pose visualization
                debug_tracking_frames = []
                for i, frame in enumerate(frames):
                    debug_frame = frame.copy()
                    rot = rotations[i] if i < len(rotations) else (0, 0, 0)
                    cv2.putText(debug_frame, f"DUSt3R Frame {i}: pan={np.degrees(rot[0]):.1f}°, tilt={np.degrees(rot[1]):.1f}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    debug_tracking_frames.append(debug_frame)
                
                return rotations, debug_tracking_frames
                
        except Exception as e:
            print(f"[CameraSolver] DUSt3R failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"[CameraSolver] Falling back to KLT")
            return self.solve_rotation_klt_persistent(frames, None, focal_length_px, 3.0)[:2]
        finally:
            # Always restore patched functions
            torch.linalg.inv = original_linalg_inv
            print(f"[CameraSolver] DUSt3R: Restored torch.linalg.inv")
    
    def solve_rotation_colmap(self, frames, focal_length_px):
        """
        Use COLMAP for Structure from Motion camera pose estimation.
        COLMAP is a traditional but robust SfM pipeline.
        """
        num_frames = frames.shape[0]
        img_height, img_width = frames.shape[1:3]
        
        try:
            import subprocess
            import tempfile
            import os
            
            # Check if COLMAP is installed
            result = subprocess.run(['colmap', '-h'], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("COLMAP not found")
                
        except Exception as e:
            print(f"[CameraSolver] COLMAP not available: {e}")
            print(f"[CameraSolver] Install COLMAP: https://colmap.github.io/install.html")
            print(f"[CameraSolver] Falling back to KLT")
            return self.solve_rotation_klt_persistent(frames, None, focal_length_px, 3.0)[:2]
        
        print(f"[CameraSolver] COLMAP: Processing {num_frames} frames...")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                images_dir = os.path.join(tmpdir, "images")
                os.makedirs(images_dir)
                
                # Save frames
                for i, frame in enumerate(frames):
                    path = os.path.join(images_dir, f"frame_{i:04d}.jpg")
                    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                database_path = os.path.join(tmpdir, "database.db")
                sparse_dir = os.path.join(tmpdir, "sparse")
                os.makedirs(sparse_dir)
                
                # Feature extraction
                print(f"[CameraSolver] COLMAP: Extracting features...")
                subprocess.run([
                    'colmap', 'feature_extractor',
                    '--database_path', database_path,
                    '--image_path', images_dir,
                    '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_model', 'PINHOLE',
                    '--ImageReader.camera_params', f'{focal_length_px},{focal_length_px},{img_width/2},{img_height/2}'
                ], capture_output=True, check=True)
                
                # Feature matching
                print(f"[CameraSolver] COLMAP: Matching features...")
                subprocess.run([
                    'colmap', 'sequential_matcher',
                    '--database_path', database_path
                ], capture_output=True, check=True)
                
                # Sparse reconstruction
                print(f"[CameraSolver] COLMAP: Running sparse reconstruction...")
                subprocess.run([
                    'colmap', 'mapper',
                    '--database_path', database_path,
                    '--image_path', images_dir,
                    '--output_path', sparse_dir
                ], capture_output=True, check=True)
                
                # Read camera poses from reconstruction
                # COLMAP outputs to sparse/0/ directory
                model_dir = os.path.join(sparse_dir, '0')
                if not os.path.exists(model_dir):
                    raise RuntimeError("COLMAP reconstruction failed")
                
                # Read images.txt for poses
                images_txt = os.path.join(model_dir, 'images.txt')
                poses = {}
                with open(images_txt, 'r') as f:
                    lines = f.readlines()
                    i = 0
                    while i < len(lines):
                        line = lines[i].strip()
                        if line.startswith('#') or not line:
                            i += 1
                            continue
                        parts = line.split()
                        if len(parts) >= 10:
                            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            name = parts[9]
                            # Extract frame number from name
                            frame_num = int(name.split('_')[1].split('.')[0])
                            
                            # Convert quaternion to rotation matrix
                            R = self._quat_to_rotation(qw, qx, qy, qz)
                            poses[frame_num] = R
                        i += 2  # Skip the points line
                
                # Convert to relative rotations
                rotations = [(0.0, 0.0, 0.0)]
                R0_inv = np.linalg.inv(poses.get(0, np.eye(3)))
                
                for i in range(1, num_frames):
                    if i in poses:
                        R_rel = R0_inv @ poses[i]
                        rvec, _ = cv2.Rodrigues(R_rel)
                        pan = rvec[1, 0]
                        tilt = rvec[0, 0]
                        roll = rvec[2, 0]
                        rotations.append((pan, tilt, roll))
                    else:
                        # Frame not in reconstruction, use previous
                        rotations.append(rotations[-1])
                
                print(f"[CameraSolver] COLMAP: Extracted {len(rotations)} camera poses")
                
                # Debug frames
                debug_tracking_frames = []
                for i, frame in enumerate(frames):
                    debug_frame = frame.copy()
                    rot = rotations[i] if i < len(rotations) else (0, 0, 0)
                    in_recon = "✓" if i in poses else "✗"
                    cv2.putText(debug_frame, f"COLMAP Frame {i} {in_recon}: pan={np.degrees(rot[0]):.1f}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    debug_tracking_frames.append(debug_frame)
                
                return rotations, debug_tracking_frames
                
        except Exception as e:
            print(f"[CameraSolver] COLMAP failed: {e}")
            print(f"[CameraSolver] Falling back to KLT")
            return self.solve_rotation_klt_persistent(frames, None, focal_length_px, 3.0)[:2]
    
    def _quat_to_rotation(self, qw, qx, qy, qz):
        """Convert quaternion to rotation matrix."""
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        return R
    
    def solve_rotation_depthcrafter(self, frames, bg_masks, focal_length_px):
        """
        Use DepthCrafter for video-native temporally consistent depth estimation.
        Track camera motion from depth changes in the background.
        """
        num_frames = frames.shape[0]
        img_height, img_width = frames.shape[1:3]
        
        if not self.load_depthcrafter():
            print(f"[CameraSolver] DepthCrafter not available, trying DepthAnything...")
            if not self.load_depth_anything():
                print(f"[CameraSolver] No depth model available, falling back to KLT")
                return self.solve_rotation_klt_persistent(frames, bg_masks, focal_length_px, 3.0)[:2]
            # Fall through to use DepthAnything in video mode
        
        print(f"[CameraSolver] DepthCrafter: Processing {num_frames} frames...")
        
        try:
            # Estimate depth for all frames
            depths = []
            for i in range(num_frames):
                if self.depthcrafter_model is not None:
                    # Use DepthCrafter (video-native)
                    from PIL import Image
                    img = Image.fromarray(frames[i])
                    result = self.depthcrafter_model(img)
                    depth = np.array(result.images[0])
                    depth = cv2.resize(depth, (img_width, img_height))
                else:
                    # Fallback to DepthAnything
                    depth = self.estimate_depth(frames[i])
                    if depth is None:
                        depth = np.ones((img_height, img_width), dtype=np.float32)
                
                depths.append(depth.astype(np.float32))
                
                if i % 10 == 0:
                    print(f"[CameraSolver] DepthCrafter: Depth estimated for frame {i}/{num_frames}")
            
            # Compute camera motion from depth flow
            # Idea: Track how depth gradients shift between frames
            rotations = [(0.0, 0.0, 0.0)]
            cumulative_pan, cumulative_tilt, cumulative_roll = 0.0, 0.0, 0.0
            
            for i in range(1, num_frames):
                depth1 = depths[i-1]
                depth2 = depths[i]
                
                # Apply background mask
                if bg_masks is not None:
                    mask = bg_masks[i] > 0.5
                else:
                    mask = np.ones_like(depth1, dtype=bool)
                
                # Compute depth gradients
                grad_x1 = cv2.Sobel(depth1, cv2.CV_32F, 1, 0, ksize=5)
                grad_y1 = cv2.Sobel(depth1, cv2.CV_32F, 0, 1, ksize=5)
                grad_x2 = cv2.Sobel(depth2, cv2.CV_32F, 1, 0, ksize=5)
                grad_y2 = cv2.Sobel(depth2, cv2.CV_32F, 0, 1, ksize=5)
                
                # Compute gradient shift (camera motion causes systematic gradient shift)
                dx = (grad_x2 - grad_x1)[mask].mean() if mask.sum() > 100 else 0
                dy = (grad_y2 - grad_y1)[mask].mean() if mask.sum() > 100 else 0
                
                # Convert to rotation (approximate)
                # Scale factor depends on focal length and depth
                avg_depth = depth1[mask].mean() if mask.sum() > 100 else 1.0
                scale = 0.001 / max(avg_depth, 0.1)  # Empirical scale
                
                delta_pan = -dx * scale
                delta_tilt = -dy * scale
                
                cumulative_pan += delta_pan
                cumulative_tilt += delta_tilt
                
                rotations.append((cumulative_pan, cumulative_tilt, cumulative_roll))
                
                if i % 10 == 0:
                    print(f"[CameraSolver] DepthCrafter Frame {i}: pan={np.degrees(cumulative_pan):.2f}°, tilt={np.degrees(cumulative_tilt):.2f}°")
            
            print(f"[CameraSolver] DepthCrafter tracking complete!")
            
            # Debug frames with depth visualization
            debug_tracking_frames = []
            for i, frame in enumerate(frames):
                debug_frame = frame.copy()
                
                # Overlay depth visualization
                depth_vis = (depths[i] - depths[i].min()) / (depths[i].max() - depths[i].min() + 1e-6)
                depth_color = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
                depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                
                # Blend with original
                alpha = 0.3
                debug_frame = cv2.addWeighted(debug_frame, 1-alpha, depth_color, alpha, 0)
                
                rot = rotations[i] if i < len(rotations) else (0, 0, 0)
                cv2.putText(debug_frame, f"DepthCrafter Frame {i}: pan={np.degrees(rot[0]):.1f}°", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                debug_tracking_frames.append(debug_frame)
            
            return rotations, debug_tracking_frames
            
        except Exception as e:
            print(f"[CameraSolver] DepthCrafter failed: {e}")
            print(f"[CameraSolver] Falling back to KLT")
            return self.solve_rotation_klt_persistent(frames, bg_masks, focal_length_px, 3.0)[:2]
    
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


class ManualCameraData:
    """
    Create camera rotation data manually from external tracking software.
    
    Use this when you have solved the camera in Maya, 3DEqualizer, SynthEyes, 
    PFTrack, or any other tracking application and want to use those values
    to place the SAM3D body in world space.
    """
    
    INTERPOLATION_MODES = ["Linear", "Ease In/Out", "Hold After End", "Constant"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "num_frames": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Total number of frames in your video"
                }),
                "total_pan_degrees": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 0.1,
                    "tooltip": "Total camera pan (Y rotation) in degrees. Positive = pan right."
                }),
                "total_tilt_degrees": ("FLOAT", {
                    "default": 0.0,
                    "min": -90.0,
                    "max": 90.0,
                    "step": 0.1,
                    "tooltip": "Total camera tilt (X rotation) in degrees. Positive = tilt up."
                }),
                "total_roll_degrees": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 0.1,
                    "tooltip": "Total camera roll (Z rotation) in degrees."
                }),
            },
            "optional": {
                "motion_start_frame": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Frame where camera motion starts (0-indexed). Frame 0 is always the reference."
                }),
                "motion_end_frame": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Frame where camera motion ends (-1 = last frame). After this, rotation stays constant."
                }),
                "interpolation": (cls.INTERPOLATION_MODES, {
                    "default": "Hold After End",
                    "tooltip": "How to interpolate rotation between start and end frames"
                }),
                "image_width": ("INT", {
                    "default": 1280,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Image width in pixels (for metadata)"
                }),
                "image_height": ("INT", {
                    "default": 720,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "Image height in pixels (for metadata)"
                }),
                "focal_length_px": ("FLOAT", {
                    "default": 1000.0,
                    "min": 100.0,
                    "max": 5000.0,
                    "tooltip": "Focal length in pixels (for metadata)"
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_ROTATION_DATA",)
    RETURN_NAMES = ("camera_rotations",)
    FUNCTION = "create_camera_data"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def create_camera_data(
        self,
        num_frames: int,
        total_pan_degrees: float,
        total_tilt_degrees: float,
        total_roll_degrees: float,
        motion_start_frame: int = 1,
        motion_end_frame: int = -1,
        interpolation: str = "Hold After End",
        image_width: int = 1280,
        image_height: int = 720,
        focal_length_px: float = 1000.0,
    ) -> Tuple[Dict]:
        """Create camera rotation data from manual input."""
        
        print(f"[ManualCameraData] Creating camera data for {num_frames} frames")
        print(f"[ManualCameraData] Total rotation: pan={total_pan_degrees}°, tilt={total_tilt_degrees}°, roll={total_roll_degrees}°")
        
        # Handle motion_end_frame = -1 (use last frame)
        if motion_end_frame < 0:
            motion_end_frame = num_frames - 1
        
        # Clamp to valid range
        motion_start_frame = max(0, min(motion_start_frame, num_frames - 1))
        motion_end_frame = max(motion_start_frame, min(motion_end_frame, num_frames - 1))
        
        print(f"[ManualCameraData] Motion range: frame {motion_start_frame} to {motion_end_frame}")
        print(f"[ManualCameraData] Interpolation: {interpolation}")
        
        # Convert to radians
        total_pan = np.radians(total_pan_degrees)
        total_tilt = np.radians(total_tilt_degrees)
        total_roll = np.radians(total_roll_degrees)
        
        # Generate per-frame rotations
        rotations = []
        
        for i in range(num_frames):
            if i <= motion_start_frame:
                # Before motion starts - no rotation
                t = 0.0
            elif i >= motion_end_frame:
                # After motion ends - full rotation (hold)
                t = 1.0
            else:
                # During motion - interpolate
                motion_duration = motion_end_frame - motion_start_frame
                if motion_duration > 0:
                    t = (i - motion_start_frame) / motion_duration
                else:
                    t = 1.0
                
                # Apply easing if requested
                if interpolation == "Ease In/Out":
                    # Smoothstep easing
                    t = t * t * (3 - 2 * t)
                elif interpolation == "Constant":
                    # No interpolation - instant change at start
                    t = 1.0 if i > motion_start_frame else 0.0
            
            pan = total_pan * t
            tilt = total_tilt * t
            roll = total_roll * t
            
            rotations.append({
                "frame": i,
                "pan": pan,
                "tilt": tilt,
                "roll": roll,
                "pan_deg": np.degrees(pan),
                "tilt_deg": np.degrees(tilt),
                "roll_deg": np.degrees(roll),
            })
        
        # Build output data
        camera_rotation_data = {
            "num_frames": num_frames,
            "image_width": image_width,
            "image_height": image_height,
            "focal_length_px": focal_length_px,
            "tracking_method": "Manual Input",
            "motion_start_frame": motion_start_frame,
            "motion_end_frame": motion_end_frame,
            "rotations": rotations
        }
        
        # Summary
        final_rot = rotations[-1]
        print(f"[ManualCameraData] Created {num_frames} frames of camera data")
        print(f"[ManualCameraData] Final rotation: pan={final_rot['pan_deg']:.2f}°, tilt={final_rot['tilt_deg']:.2f}°, roll={final_rot['roll_deg']:.2f}°")
        
        return (camera_rotation_data,)


class CameraDataFromJSON:
    """
    Load camera rotation data from a JSON file or string.
    
    Use this to import camera solve data exported from external applications.
    
    Expected JSON format:
    {
        "frames": [
            {"frame": 0, "pan": 0.0, "tilt": 0.0, "roll": 0.0},
            {"frame": 1, "pan": 0.5, "tilt": 0.1, "roll": 0.0},
            ...
        ]
    }
    
    Rotation values should be in DEGREES.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "json_data": ("STRING", {
                    "default": '{"frames": [{"frame": 0, "pan": 0, "tilt": 0, "roll": 0}]}',
                    "multiline": True,
                    "tooltip": "JSON string with camera rotation data, or path to JSON file"
                }),
            },
            "optional": {
                "image_width": ("INT", {
                    "default": 1280,
                    "min": 1,
                    "max": 8192,
                }),
                "image_height": ("INT", {
                    "default": 720,
                    "min": 1,
                    "max": 8192,
                }),
                "focal_length_px": ("FLOAT", {
                    "default": 1000.0,
                    "min": 100.0,
                    "max": 5000.0,
                }),
            }
        }
    
    RETURN_TYPES = ("CAMERA_ROTATION_DATA",)
    RETURN_NAMES = ("camera_rotations",)
    FUNCTION = "load_camera_data"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def load_camera_data(
        self,
        json_data: str,
        image_width: int = 1280,
        image_height: int = 720,
        focal_length_px: float = 1000.0,
    ) -> Tuple[Dict]:
        """Load camera rotation data from JSON."""
        import json
        import os
        
        print(f"[CameraDataFromJSON] Loading camera data...")
        
        # Try to load as file path first
        data = None
        if os.path.exists(json_data.strip()):
            try:
                with open(json_data.strip(), 'r') as f:
                    data = json.load(f)
                print(f"[CameraDataFromJSON] Loaded from file: {json_data.strip()}")
            except Exception as e:
                print(f"[CameraDataFromJSON] Failed to load file: {e}")
        
        # Try to parse as JSON string
        if data is None:
            try:
                data = json.loads(json_data)
                print(f"[CameraDataFromJSON] Parsed JSON string")
            except Exception as e:
                print(f"[CameraDataFromJSON] Failed to parse JSON: {e}")
                # Return empty data
                return ({
                    "num_frames": 1,
                    "image_width": image_width,
                    "image_height": image_height,
                    "focal_length_px": focal_length_px,
                    "tracking_method": "JSON Import (Error)",
                    "rotations": [{"frame": 0, "pan": 0, "tilt": 0, "roll": 0, "pan_deg": 0, "tilt_deg": 0, "roll_deg": 0}]
                },)
        
        # Parse frames
        frames_data = data.get("frames", [])
        if not frames_data:
            print(f"[CameraDataFromJSON] No frames found in JSON")
            return ({
                "num_frames": 1,
                "image_width": image_width,
                "image_height": image_height,
                "focal_length_px": focal_length_px,
                "tracking_method": "JSON Import (Empty)",
                "rotations": [{"frame": 0, "pan": 0, "tilt": 0, "roll": 0, "pan_deg": 0, "tilt_deg": 0, "roll_deg": 0}]
            },)
        
        # Convert to our format
        rotations = []
        for frame_data in frames_data:
            frame_idx = frame_data.get("frame", len(rotations))
            pan_deg = frame_data.get("pan", 0.0)
            tilt_deg = frame_data.get("tilt", 0.0)
            roll_deg = frame_data.get("roll", 0.0)
            
            rotations.append({
                "frame": frame_idx,
                "pan": np.radians(pan_deg),
                "tilt": np.radians(tilt_deg),
                "roll": np.radians(roll_deg),
                "pan_deg": pan_deg,
                "tilt_deg": tilt_deg,
                "roll_deg": roll_deg,
            })
        
        # Sort by frame number
        rotations.sort(key=lambda x: x["frame"])
        
        num_frames = len(rotations)
        
        # Build output
        camera_rotation_data = {
            "num_frames": num_frames,
            "image_width": data.get("image_width", image_width),
            "image_height": data.get("image_height", image_height),
            "focal_length_px": data.get("focal_length_px", focal_length_px),
            "tracking_method": "JSON Import",
            "rotations": rotations
        }
        
        final_rot = rotations[-1] if rotations else {"pan_deg": 0, "tilt_deg": 0, "roll_deg": 0}
        print(f"[CameraDataFromJSON] Loaded {num_frames} frames")
        print(f"[CameraDataFromJSON] Final rotation: pan={final_rot['pan_deg']:.2f}°, tilt={final_rot['tilt_deg']:.2f}°")
        
        return (camera_rotation_data,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CameraRotationSolver": CameraRotationSolver,
    "ManualCameraData": ManualCameraData,
    "CameraDataFromJSON": CameraDataFromJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraRotationSolver": "Camera Rotation Solver",
    "ManualCameraData": "Manual Camera Data",
    "CameraDataFromJSON": "Camera Data from JSON",
}
