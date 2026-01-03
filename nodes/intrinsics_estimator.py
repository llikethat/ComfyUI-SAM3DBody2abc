"""
Unified Camera Intrinsics Estimator for SAM3DBody2abc v5.0

This node provides a single, consistent source of camera intrinsics for the entire
v5.0 stabilization pipeline. It consolidates multiple intrinsics sources with
clear priority ordering:

Priority Order:
    1. User Input (manual focal length) - Most trusted
    2. EXIF Metadata (from video/image files) - Good for phone/DSLR
    3. MoGe2 Estimation - AI-based estimation (fallback)
    4. Heuristic - Simple W×1.0 assumption (last resort)

Usage in v5.0 Pipeline:
    Video → IntrinsicsEstimator → INTRINSICS
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              CameraSolverV2    VideoStabilizer    SAM3DBody
                    ↓                 ↓                 ↓
              Camera Solve      Stabilization     Pose Estimation

The INTRINSICS output is used consistently across all pipeline stages.

Version: 5.0.0
Author: SAM3DBody2abc Project
"""

import numpy as np
import torch
import os
import json
import struct
from typing import Dict, Tuple, Any, Optional, List, Union
from pathlib import Path

# Check for optional dependencies
MOGE_AVAILABLE = False
EXIFTOOL_AVAILABLE = False
PIL_AVAILABLE = False

try:
    from moge.model.v2 import MoGeModel
    MOGE_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    PIL_AVAILABLE = True
except ImportError:
    pass


# Common camera sensor sizes (width in mm)
SENSOR_DATABASE = {
    # Full Frame
    "full_frame": 36.0,
    "35mm": 36.0,
    
    # APS-C
    "aps-c_canon": 22.3,
    "aps-c_nikon": 23.5,
    "aps-c_sony": 23.5,
    "aps-c": 23.5,
    
    # Micro Four Thirds
    "m43": 17.3,
    "micro_four_thirds": 17.3,
    
    # Medium Format
    "medium_format_fuji": 43.8,
    "medium_format_hasselblad": 43.8,
    
    # Smartphones (approximate)
    "iphone_14_pro": 9.8,
    "iphone_15_pro": 9.8,
    "pixel_7_pro": 8.2,
    "samsung_s23": 8.0,
    "smartphone_1inch": 13.2,
    "smartphone": 6.17,  # Common 1/2.55" sensor
    
    # Action cameras
    "gopro": 6.17,
    
    # Cinema
    "super35": 24.89,
    "red_8k": 40.96,
}


class IntrinsicsEstimator:
    """
    Unified camera intrinsics estimation for SAM3DBody2abc v5.0 pipeline.
    
    This node consolidates intrinsics from multiple sources with priority:
    1. User manual input (if focal_length_mm > 0)
    2. EXIF metadata (if source_path provided and contains EXIF)
    3. MoGe2 AI estimation (if available)
    4. Heuristic fallback (focal = image_width)
    
    Output INTRINSICS format:
    {
        "focal_px": float,          # Focal length in pixels
        "focal_mm": float,          # Focal length in mm
        "sensor_width_mm": float,   # Sensor width assumption
        "cx": float,                # Principal point X (pixels)
        "cy": float,                # Principal point Y (pixels)
        "width": int,               # Image width
        "height": int,              # Image height
        "fov_x_deg": float,         # Horizontal FOV in degrees
        "fov_y_deg": float,         # Vertical FOV in degrees
        "source": str,              # "user" | "exif" | "moge2" | "heuristic"
        "confidence": float,        # 0.0-1.0 confidence score
        "k_matrix": [[3x3]],        # Camera intrinsic matrix K
    }
    """
    
    SENSOR_PRESETS = ["custom"] + list(SENSOR_DATABASE.keys())
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images/video frames for intrinsics estimation"
                }),
            },
            "optional": {
                # === User Override (Highest Priority) ===
                "focal_length_mm": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Manual focal length in mm. Set >0 to override all other sources."
                }),
                
                # === Sensor Configuration ===
                "sensor_preset": (cls.SENSOR_PRESETS, {
                    "default": "full_frame",
                    "tooltip": "Camera sensor preset for focal length conversion"
                }),
                "sensor_width_mm": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Custom sensor width in mm (used if sensor_preset='custom')"
                }),
                
                # === EXIF Source (Second Priority) ===
                "source_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to original video/image file for EXIF extraction"
                }),
                
                # === MoGe2 Estimation (Third Priority) ===
                "use_moge2": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use MoGe2 AI estimation if no user input or EXIF available"
                }),
                "moge2_sample_frames": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Number of frames to sample for MoGe2 estimation"
                }),
                
                # === Per-Frame Options ===
                "per_frame_estimation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Estimate intrinsics per frame (for zoom detection). Slower but handles variable focal length."
                }),
                
                # === Debug ===
                "verbose": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print detailed intrinsics information"
                }),
            }
        }
    
    RETURN_TYPES = ("INTRINSICS", "FLOAT", "STRING")
    RETURN_NAMES = ("intrinsics", "focal_length_mm", "status")
    FUNCTION = "estimate"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def __init__(self):
        self.moge_model = None
        self.moge_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def estimate(
        self,
        images: torch.Tensor,
        focal_length_mm: float = 0.0,
        sensor_preset: str = "full_frame",
        sensor_width_mm: float = 36.0,
        source_path: str = "",
        use_moge2: bool = True,
        moge2_sample_frames: int = 5,
        per_frame_estimation: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict, float, str]:
        """
        Estimate camera intrinsics with priority ordering.
        
        Returns:
            intrinsics: INTRINSICS dict for pipeline
            focal_length_mm: Primary focal length in mm
            status: Human-readable status string
        """
        
        # Get image dimensions
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        
        # Resolve sensor width
        if sensor_preset != "custom" and sensor_preset in SENSOR_DATABASE:
            sensor_width = SENSOR_DATABASE[sensor_preset]
        else:
            sensor_width = sensor_width_mm
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[IntrinsicsEstimator] Analyzing {num_frames} frames ({W}x{H})")
            print(f"[IntrinsicsEstimator] Sensor: {sensor_preset} ({sensor_width:.1f}mm)")
            print(f"{'='*60}")
        
        # Priority 1: User manual input
        if focal_length_mm > 0:
            intrinsics = self._from_user_input(
                focal_length_mm, sensor_width, W, H, verbose
            )
            return (intrinsics, intrinsics["focal_mm"], f"User input: {focal_length_mm:.1f}mm")
        
        # Priority 2: EXIF metadata
        if source_path and os.path.exists(source_path):
            exif_intrinsics = self._from_exif(source_path, sensor_width, W, H, verbose)
            if exif_intrinsics is not None:
                return (exif_intrinsics, exif_intrinsics["focal_mm"], 
                       f"EXIF: {exif_intrinsics['focal_mm']:.1f}mm")
        
        # Priority 3: MoGe2 estimation
        if use_moge2 and MOGE_AVAILABLE:
            moge_intrinsics = self._from_moge2(
                images, sensor_width, moge2_sample_frames, per_frame_estimation, verbose
            )
            if moge_intrinsics is not None:
                return (moge_intrinsics, moge_intrinsics["focal_mm"],
                       f"MoGe2: {moge_intrinsics['focal_mm']:.1f}mm")
        
        # Priority 4: Heuristic fallback
        intrinsics = self._from_heuristic(sensor_width, W, H, verbose)
        return (intrinsics, intrinsics["focal_mm"], 
               f"Heuristic: {intrinsics['focal_mm']:.1f}mm (fallback)")
    
    def _build_intrinsics(
        self,
        focal_px: float,
        sensor_width_mm: float,
        width: int,
        height: int,
        source: str,
        confidence: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        per_frame_data: Optional[List[Dict]] = None,
    ) -> Dict:
        """Build standardized INTRINSICS output dict."""
        
        # Default principal point to image center
        if cx is None:
            cx = width / 2.0
        if cy is None:
            cy = height / 2.0
        
        # Convert to mm
        focal_mm = focal_px * sensor_width_mm / width
        
        # Compute FOV
        fov_x_deg = 2 * np.degrees(np.arctan(width / (2 * focal_px)))
        fov_y_deg = 2 * np.degrees(np.arctan(height / (2 * focal_px)))
        
        # Build K matrix
        k_matrix = [
            [focal_px, 0.0, cx],
            [0.0, focal_px, cy],
            [0.0, 0.0, 1.0]
        ]
        
        intrinsics = {
            # Core parameters
            "focal_px": float(focal_px),
            "focal_mm": float(focal_mm),
            "sensor_width_mm": float(sensor_width_mm),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(width),
            "height": int(height),
            
            # Derived values
            "fov_x_deg": float(fov_x_deg),
            "fov_y_deg": float(fov_y_deg),
            "aspect_ratio": float(width / height),
            
            # Metadata
            "source": source,
            "confidence": float(confidence),
            "k_matrix": k_matrix,
            
            # Per-frame data (for zoom handling)
            "per_frame": per_frame_data,
            "is_variable": per_frame_data is not None and len(per_frame_data) > 1,
        }
        
        return intrinsics
    
    def _from_user_input(
        self,
        focal_mm: float,
        sensor_width_mm: float,
        width: int,
        height: int,
        verbose: bool,
    ) -> Dict:
        """Build intrinsics from user-provided focal length."""
        
        # Convert mm to pixels
        focal_px = focal_mm * width / sensor_width_mm
        
        if verbose:
            print(f"[IntrinsicsEstimator] Priority 1: USER INPUT")
            print(f"  Focal: {focal_mm:.1f}mm = {focal_px:.1f}px")
            print(f"  Confidence: 1.0 (user specified)")
        
        return self._build_intrinsics(
            focal_px=focal_px,
            sensor_width_mm=sensor_width_mm,
            width=width,
            height=height,
            source="user",
            confidence=1.0,
        )
    
    def _from_exif(
        self,
        source_path: str,
        sensor_width_mm: float,
        width: int,
        height: int,
        verbose: bool,
    ) -> Optional[Dict]:
        """Extract intrinsics from EXIF metadata."""
        
        if not PIL_AVAILABLE:
            if verbose:
                print(f"[IntrinsicsEstimator] Priority 2: EXIF - PIL not available")
            return None
        
        try:
            # Try to extract EXIF from image file
            path = Path(source_path)
            
            # For video files, we might need to extract a frame first
            video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
            if path.suffix.lower() in video_extensions:
                if verbose:
                    print(f"[IntrinsicsEstimator] Priority 2: EXIF - Video file, checking sidecar...")
                # Look for sidecar JSON with metadata
                sidecar_path = path.with_suffix('.json')
                if sidecar_path.exists():
                    return self._from_sidecar_json(sidecar_path, sensor_width_mm, width, height, verbose)
                return None
            
            # Image file - extract EXIF directly
            with Image.open(source_path) as img:
                exif_data = img._getexif()
                
                if exif_data is None:
                    if verbose:
                        print(f"[IntrinsicsEstimator] Priority 2: EXIF - No EXIF data found")
                    return None
                
                # Decode EXIF tags
                exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
                
                # Look for focal length
                focal_mm = None
                focal_35mm = None
                
                if 'FocalLength' in exif:
                    focal_mm = float(exif['FocalLength'])
                
                if 'FocalLengthIn35mmFilm' in exif:
                    focal_35mm = float(exif['FocalLengthIn35mmFilm'])
                
                if focal_mm is None and focal_35mm is None:
                    if verbose:
                        print(f"[IntrinsicsEstimator] Priority 2: EXIF - No focal length in EXIF")
                    return None
                
                # Prefer 35mm equivalent if available (more standardized)
                if focal_35mm:
                    # Convert 35mm equivalent back to pixels
                    # 35mm sensor is 36mm wide
                    focal_px = focal_35mm * width / 36.0
                    effective_sensor = 36.0
                    focal_used = focal_35mm
                elif focal_mm:
                    # Use actual focal length with specified sensor
                    focal_px = focal_mm * width / sensor_width_mm
                    effective_sensor = sensor_width_mm
                    focal_used = focal_mm
                
                if verbose:
                    print(f"[IntrinsicsEstimator] Priority 2: EXIF")
                    print(f"  Focal: {focal_used:.1f}mm = {focal_px:.1f}px")
                    if focal_35mm:
                        print(f"  (35mm equivalent: {focal_35mm:.1f}mm)")
                    print(f"  Confidence: 0.9 (EXIF metadata)")
                
                return self._build_intrinsics(
                    focal_px=focal_px,
                    sensor_width_mm=effective_sensor,
                    width=width,
                    height=height,
                    source="exif",
                    confidence=0.9,
                )
        
        except Exception as e:
            if verbose:
                print(f"[IntrinsicsEstimator] Priority 2: EXIF - Error: {e}")
            return None
    
    def _from_sidecar_json(
        self,
        json_path: Path,
        sensor_width_mm: float,
        width: int,
        height: int,
        verbose: bool,
    ) -> Optional[Dict]:
        """Extract intrinsics from sidecar JSON metadata file."""
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            focal_mm = metadata.get('focal_length_mm') or metadata.get('focal_length')
            if focal_mm:
                focal_px = float(focal_mm) * width / sensor_width_mm
                
                if verbose:
                    print(f"[IntrinsicsEstimator] Priority 2: EXIF (sidecar JSON)")
                    print(f"  Focal: {focal_mm:.1f}mm = {focal_px:.1f}px")
                    print(f"  Confidence: 0.85 (sidecar metadata)")
                
                return self._build_intrinsics(
                    focal_px=focal_px,
                    sensor_width_mm=sensor_width_mm,
                    width=width,
                    height=height,
                    source="exif",
                    confidence=0.85,
                )
        except Exception as e:
            if verbose:
                print(f"[IntrinsicsEstimator] Sidecar JSON error: {e}")
        return None
    
    def _load_moge_model(self) -> bool:
        """Load MoGe2 model if available."""
        if not MOGE_AVAILABLE:
            return False
        
        if self.moge_model is not None:
            return True
        
        try:
            model_name = "Ruicheng/moge-2-vitl-normal"
            print(f"[IntrinsicsEstimator] Loading MoGe2 model...")
            self.moge_model = MoGeModel.from_pretrained(model_name).to(self.device)
            self.moge_model.eval()
            self.moge_model_name = model_name
            print(f"[IntrinsicsEstimator] MoGe2 loaded on {self.device}")
            return True
        except Exception as e:
            print(f"[IntrinsicsEstimator] MoGe2 load error: {e}")
            return False
    
    def _from_moge2(
        self,
        images: torch.Tensor,
        sensor_width_mm: float,
        sample_frames: int,
        per_frame: bool,
        verbose: bool,
    ) -> Optional[Dict]:
        """Estimate intrinsics using MoGe2 model."""
        
        if not self._load_moge_model():
            return None
        
        num_frames = images.shape[0]
        H, W = images.shape[1], images.shape[2]
        
        # Determine which frames to process
        if per_frame:
            frame_indices = list(range(num_frames))
        else:
            if num_frames <= sample_frames:
                frame_indices = list(range(num_frames))
            else:
                step = num_frames / sample_frames
                frame_indices = [int(i * step) for i in range(sample_frames)]
        
        if verbose:
            print(f"[IntrinsicsEstimator] Priority 3: MoGe2")
            print(f"  Processing {len(frame_indices)} frames...")
        
        all_focal_px = []
        per_frame_data = [] if per_frame else None
        
        with torch.no_grad():
            for idx in frame_indices:
                frame = images[idx]  # (H, W, C)
                frame_tensor = frame.permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                try:
                    output = self.moge_model.infer(frame_tensor)
                    intrinsics_matrix = output["intrinsics"]
                    
                    if hasattr(intrinsics_matrix, 'cpu'):
                        intrinsics_matrix = intrinsics_matrix.cpu().numpy()
                    if intrinsics_matrix.ndim == 3:
                        intrinsics_matrix = intrinsics_matrix[0]
                    
                    # Extract focal length (normalized)
                    fx_norm = intrinsics_matrix[0, 0]
                    fy_norm = intrinsics_matrix[1, 1]
                    
                    # Denormalize
                    fx_px = fx_norm * W
                    fy_px = fy_norm * H
                    focal_px = (fx_px + fy_px) / 2  # Average
                    
                    all_focal_px.append(focal_px)
                    
                    if per_frame:
                        per_frame_data.append({
                            "frame": idx,
                            "focal_px": float(focal_px),
                            "focal_mm": float(focal_px * sensor_width_mm / W),
                        })
                
                except Exception as e:
                    if verbose:
                        print(f"  Frame {idx} error: {e}")
                    continue
        
        if not all_focal_px:
            return None
        
        # Use median for robustness
        focal_px = float(np.median(all_focal_px))
        focal_std = float(np.std(all_focal_px)) if len(all_focal_px) > 1 else 0.0
        
        # Confidence based on consistency
        if focal_std / focal_px < 0.05:
            confidence = 0.8  # Very consistent
        elif focal_std / focal_px < 0.1:
            confidence = 0.7  # Reasonably consistent
        else:
            confidence = 0.6  # Variable (might be zoom)
        
        focal_mm = focal_px * sensor_width_mm / W
        
        if verbose:
            print(f"  Focal: {focal_mm:.1f}mm = {focal_px:.1f}px")
            print(f"  Std dev: {focal_std:.1f}px ({100*focal_std/focal_px:.1f}%)")
            print(f"  Confidence: {confidence}")
            if per_frame and len(per_frame_data) > 1:
                focals = [d["focal_mm"] for d in per_frame_data]
                print(f"  Range: {min(focals):.1f}mm - {max(focals):.1f}mm")
        
        return self._build_intrinsics(
            focal_px=focal_px,
            sensor_width_mm=sensor_width_mm,
            width=W,
            height=H,
            source="moge2",
            confidence=confidence,
            per_frame_data=per_frame_data if per_frame else None,
        )
    
    def _from_heuristic(
        self,
        sensor_width_mm: float,
        width: int,
        height: int,
        verbose: bool,
    ) -> Dict:
        """Fallback heuristic: assume focal length = image width (≈53° FOV)."""
        
        focal_px = float(width)
        
        if verbose:
            print(f"[IntrinsicsEstimator] Priority 4: HEURISTIC FALLBACK")
            print(f"  Focal: {focal_px * sensor_width_mm / width:.1f}mm = {focal_px:.1f}px")
            print(f"  Assumption: focal = image_width (standard ~53° FOV)")
            print(f"  Confidence: 0.3 (heuristic)")
        
        return self._build_intrinsics(
            focal_px=focal_px,
            sensor_width_mm=sensor_width_mm,
            width=width,
            height=height,
            source="heuristic",
            confidence=0.3,
        )


class IntrinsicsInfo:
    """
    Display and validate camera intrinsics.
    
    Utility node to inspect INTRINSICS data and validate against expected values.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "intrinsics": ("INTRINSICS",),
            },
            "optional": {
                "expected_focal_mm": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 500.0,
                    "tooltip": "Expected focal length for validation (0 = skip validation)"
                }),
                "tolerance_percent": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "tooltip": "Acceptable deviation from expected value (%)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("info", "validation_passed")
    FUNCTION = "display"
    CATEGORY = "SAM3DBody2abc/Camera"
    OUTPUT_NODE = True
    
    def display(
        self,
        intrinsics: Dict,
        expected_focal_mm: float = 0.0,
        tolerance_percent: float = 10.0,
    ) -> Tuple[str, bool]:
        """Display intrinsics information and optionally validate."""
        
        lines = [
            "=" * 50,
            "CAMERA INTRINSICS",
            "=" * 50,
            "",
            f"Source: {intrinsics.get('source', 'unknown').upper()}",
            f"Confidence: {intrinsics.get('confidence', 0):.0%}",
            "",
            "--- Focal Length ---",
            f"  {intrinsics.get('focal_mm', 0):.1f} mm",
            f"  {intrinsics.get('focal_px', 0):.1f} px",
            "",
            "--- Field of View ---",
            f"  Horizontal: {intrinsics.get('fov_x_deg', 0):.1f}°",
            f"  Vertical: {intrinsics.get('fov_y_deg', 0):.1f}°",
            "",
            "--- Image ---",
            f"  Resolution: {intrinsics.get('width', 0)} x {intrinsics.get('height', 0)}",
            f"  Aspect: {intrinsics.get('aspect_ratio', 0):.3f}",
            "",
            "--- Principal Point ---",
            f"  cx: {intrinsics.get('cx', 0):.1f} px",
            f"  cy: {intrinsics.get('cy', 0):.1f} px",
            "",
            f"--- Sensor ---",
            f"  Width: {intrinsics.get('sensor_width_mm', 0):.1f} mm",
        ]
        
        # Variable focal length info
        if intrinsics.get('is_variable'):
            per_frame = intrinsics.get('per_frame', [])
            if per_frame:
                focals = [d['focal_mm'] for d in per_frame]
                lines.extend([
                    "",
                    "--- Variable Focal (Zoom Detected) ---",
                    f"  Range: {min(focals):.1f} - {max(focals):.1f} mm",
                    f"  Frames: {len(per_frame)}",
                ])
        
        # Validation
        validation_passed = True
        if expected_focal_mm > 0:
            actual = intrinsics.get('focal_mm', 0)
            diff_percent = abs(actual - expected_focal_mm) / expected_focal_mm * 100
            
            if diff_percent <= tolerance_percent:
                status = "✓ PASSED"
                validation_passed = True
            else:
                status = "✗ FAILED"
                validation_passed = False
            
            lines.extend([
                "",
                "--- Validation ---",
                f"  Expected: {expected_focal_mm:.1f} mm",
                f"  Actual: {actual:.1f} mm",
                f"  Deviation: {diff_percent:.1f}%",
                f"  Tolerance: ±{tolerance_percent:.1f}%",
                f"  Status: {status}",
            ])
        
        lines.append("")
        lines.append("=" * 50)
        
        info_text = "\n".join(lines)
        print(info_text)
        
        return (info_text, validation_passed)


class IntrinsicsFromJSON:
    """
    Load camera intrinsics from JSON file.
    
    Supports loading intrinsics exported from:
    - Previous IntrinsicsEstimator runs
    - External camera calibration tools
    - Manual JSON configuration
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to JSON file containing intrinsics data"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Images to validate dimensions against"
                }),
            }
        }
    
    RETURN_TYPES = ("INTRINSICS", "STRING")
    RETURN_NAMES = ("intrinsics", "status")
    FUNCTION = "load"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def load(
        self,
        json_path: str,
        images: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict, str]:
        """Load intrinsics from JSON file."""
        
        if not os.path.exists(json_path):
            raise ValueError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        required = ['focal_px', 'width', 'height']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field in JSON: {field}")
        
        # Validate against images if provided
        if images is not None:
            H, W = images.shape[1], images.shape[2]
            if data['width'] != W or data['height'] != H:
                print(f"[WARNING] Image dimensions ({W}x{H}) don't match JSON ({data['width']}x{data['height']})")
        
        # Ensure all expected fields exist
        intrinsics = {
            "focal_px": data.get("focal_px", 1000.0),
            "focal_mm": data.get("focal_mm", 35.0),
            "sensor_width_mm": data.get("sensor_width_mm", 36.0),
            "cx": data.get("cx", data.get("width", 1920) / 2),
            "cy": data.get("cy", data.get("height", 1080) / 2),
            "width": data.get("width", 1920),
            "height": data.get("height", 1080),
            "fov_x_deg": data.get("fov_x_deg", 50.0),
            "fov_y_deg": data.get("fov_y_deg", 30.0),
            "aspect_ratio": data.get("aspect_ratio", 16/9),
            "source": "json",
            "confidence": data.get("confidence", 0.95),
            "k_matrix": data.get("k_matrix"),
            "per_frame": data.get("per_frame"),
            "is_variable": data.get("is_variable", False),
        }
        
        status = f"Loaded from JSON: {intrinsics['focal_mm']:.1f}mm"
        print(f"[IntrinsicsFromJSON] {status}")
        
        return (intrinsics, status)


class IntrinsicsToJSON:
    """
    Save camera intrinsics to JSON file.
    
    Useful for:
    - Saving estimated intrinsics for reuse
    - Sharing calibration data
    - Debugging and inspection
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "intrinsics": ("INTRINSICS",),
                "output_path": ("STRING", {
                    "default": "intrinsics.json",
                    "tooltip": "Path for output JSON file"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "SAM3DBody2abc/Camera"
    OUTPUT_NODE = True
    
    def save(
        self,
        intrinsics: Dict,
        output_path: str,
    ) -> Tuple[str]:
        """Save intrinsics to JSON file."""
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        output_data = {}
        for key, value in intrinsics.items():
            if isinstance(value, np.ndarray):
                output_data[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                output_data[key] = float(value)
            else:
                output_data[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[IntrinsicsToJSON] Saved to {output_path}")
        
        return (output_path,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "IntrinsicsEstimator": IntrinsicsEstimator,
    "IntrinsicsInfo": IntrinsicsInfo,
    "IntrinsicsFromJSON": IntrinsicsFromJSON,
    "IntrinsicsToJSON": IntrinsicsToJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntrinsicsEstimator": "📷 Intrinsics Estimator (v5.0)",
    "IntrinsicsInfo": "📷 Intrinsics Info",
    "IntrinsicsFromJSON": "📷 Intrinsics from JSON",
    "IntrinsicsToJSON": "📷 Intrinsics to JSON",
}
