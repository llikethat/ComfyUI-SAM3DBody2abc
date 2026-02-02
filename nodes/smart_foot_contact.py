"""
Smart Foot Contact - Physics + Visual Feedback Loop
====================================================

Unified foot contact detection and correction using:
1. GroundLink/Heuristic for initial contact estimation
2. TAPNet for 2D foot tracking (ground truth from video)
3. Reprojection error feedback to validate and correct

Key Innovation:
- Video is ground truth - we measure discrepancy between SAM3DBody and video
- Contact detection based on 2D foot velocity (not 3D guesswork)
- Corrections computed mathematically to match video projection
- Temporal smoothing prevents visible snapping

Author: SAM3DBody2abc
Version: 1.0.0
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import gc

# Lazy imports
torch = None
scipy_gaussian = None
scipy_medfilt = None


def _ensure_imports():
    """Lazy import heavy dependencies."""
    global torch, scipy_gaussian, scipy_medfilt
    if torch is None:
        import torch as _torch
        torch = _torch
    if scipy_gaussian is None:
        from scipy.ndimage import gaussian_filter1d
        scipy_gaussian = gaussian_filter1d
    if scipy_medfilt is None:
        from scipy.signal import medfilt
        scipy_medfilt = medfilt


# =============================================================================
# Logging System
# =============================================================================

class LogLevel(Enum):
    SILENT = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class SmartFootContactLogger:
    """Structured logger with levels and timing."""
    
    def __init__(self, level: LogLevel = LogLevel.NORMAL, prefix: str = "SmartFootContact"):
        self.level = level
        self.prefix = prefix
        self.timings: Dict[str, float] = {}
        self._timer_stack: List[Tuple[str, float]] = []
        self.frame_logs: List[Dict] = []  # Per-frame debug info
    
    def _should_log(self, min_level: LogLevel) -> bool:
        return self.level.value >= min_level.value
    
    def info(self, msg: str):
        if self._should_log(LogLevel.NORMAL):
            print(f"[{self.prefix}] {msg}")
    
    def verbose(self, msg: str):
        if self._should_log(LogLevel.VERBOSE):
            print(f"[{self.prefix}] {msg}")
    
    def debug(self, msg: str):
        if self._should_log(LogLevel.DEBUG):
            print(f"[{self.prefix}] DEBUG: {msg}")
    
    def warning(self, msg: str):
        print(f"[{self.prefix}] ⚠ WARNING: {msg}")
    
    def error(self, msg: str):
        print(f"[{self.prefix}] ✗ ERROR: {msg}")
    
    def progress(self, current: int, total: int, task: str = "", interval: int = 10):
        if self._should_log(LogLevel.NORMAL):
            if current == 0 or current == total - 1 or (current + 1) % interval == 0:
                pct = 100 * (current + 1) / total
                print(f"[{self.prefix}] {task}: {current + 1}/{total} ({pct:.0f}%)")
    
    def start_timer(self, name: str):
        self._timer_stack.append((name, time.time()))
    
    def end_timer(self, name: str = None):
        if self._timer_stack:
            timer_name, start_time = self._timer_stack.pop()
            if name and name != timer_name:
                self.warning(f"Timer mismatch: expected {name}, got {timer_name}")
            elapsed = time.time() - start_time
            self.timings[timer_name] = elapsed
            if self._should_log(LogLevel.VERBOSE):
                print(f"[{self.prefix}] ⏱ {timer_name}: {elapsed:.3f}s")
            return elapsed
        return 0.0
    
    def log_frame(self, frame_idx: int, data: Dict):
        """Store per-frame debug data."""
        self.frame_logs.append({"frame": frame_idx, **data})
    
    def get_summary(self) -> str:
        """Get timing summary."""
        lines = ["=== TIMING SUMMARY ==="]
        for name, elapsed in self.timings.items():
            lines.append(f"  {name}: {elapsed:.3f}s")
        total = sum(self.timings.values())
        lines.append(f"  TOTAL: {total:.3f}s")
        return "\n".join(lines)
    
    def get_frame_log(self, frame_idx: int) -> Optional[Dict]:
        """Get debug data for specific frame."""
        for log in self.frame_logs:
            if log["frame"] == frame_idx:
                return log
        return None


# =============================================================================
# Contact Segment Detection
# =============================================================================

@dataclass
class ContactSegment:
    """Represents a contiguous foot contact period."""
    foot: str  # "left" or "right"
    start_frame: int
    end_frame: int
    pin_position_2d: np.ndarray  # Median 2D position during contact
    pin_position_3d: Optional[np.ndarray] = None  # Back-projected 3D position
    confidence: float = 1.0
    source: str = "video"  # "video", "groundlink", "heuristic"
    
    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1
    
    def contains_frame(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame
    
    def get_blend_weight(self, frame: int, blend_in: int = 3, blend_out: int = 3) -> float:
        """Get smooth blend weight for frame (0 at edges, 1 in middle)."""
        if not self.contains_frame(frame):
            return 0.0
        
        frames_from_start = frame - self.start_frame
        frames_to_end = self.end_frame - frame
        
        if frames_from_start < blend_in:
            return frames_from_start / blend_in
        elif frames_to_end < blend_out:
            return frames_to_end / blend_out
        else:
            return 1.0


class ContactSegmentDetector:
    """
    Detects contiguous contact segments from boolean contact array.
    
    Handles:
    - Minimum duration filtering
    - Gap bridging (merge segments separated by few frames)
    - Finding most stable frame within segment
    """
    
    def __init__(
        self,
        min_duration: int = 3,
        max_gap_to_bridge: int = 2,
        logger: Optional[SmartFootContactLogger] = None
    ):
        self.min_duration = min_duration
        self.max_gap_to_bridge = max_gap_to_bridge
        self.log = logger or SmartFootContactLogger(LogLevel.SILENT)
    
    def detect(
        self,
        contact_mask: np.ndarray,
        foot_name: str,
        foot_positions_2d: np.ndarray,
        foot_velocities_2d: Optional[np.ndarray] = None
    ) -> List[ContactSegment]:
        """
        Detect contact segments from boolean mask.
        
        Args:
            contact_mask: (T,) boolean array
            foot_name: "left" or "right"
            foot_positions_2d: (T, 2) array of 2D positions
            foot_velocities_2d: (T,) array of velocities (optional, for finding stable frame)
        
        Returns:
            List of ContactSegment objects
        """
        T = len(contact_mask)
        
        # Step 1: Bridge small gaps
        bridged = self._bridge_gaps(contact_mask)
        
        # Step 2: Find contiguous segments
        raw_segments = self._find_runs(bridged)
        
        # Step 3: Filter by minimum duration
        segments = []
        for start, end in raw_segments:
            duration = end - start + 1
            if duration >= self.min_duration:
                # Compute pin position (median for robustness)
                positions_in_segment = foot_positions_2d[start:end+1]
                pin_pos = np.median(positions_in_segment, axis=0)
                
                # Compute confidence based on velocity stability
                if foot_velocities_2d is not None:
                    vels_in_segment = foot_velocities_2d[start:end+1]
                    mean_vel = np.mean(vels_in_segment)
                    # Lower velocity = higher confidence
                    confidence = 1.0 / (1.0 + mean_vel * 10)
                else:
                    confidence = 1.0
                
                segment = ContactSegment(
                    foot=foot_name,
                    start_frame=start,
                    end_frame=end,
                    pin_position_2d=pin_pos,
                    confidence=confidence,
                    source="video"
                )
                segments.append(segment)
                
                self.log.debug(
                    f"{foot_name} contact: frames {start}-{end} "
                    f"(duration={duration}, confidence={confidence:.2f})"
                )
        
        return segments
    
    def _bridge_gaps(self, mask: np.ndarray) -> np.ndarray:
        """Bridge small gaps in contact mask."""
        if self.max_gap_to_bridge <= 0:
            return mask.copy()
        
        bridged = mask.copy()
        T = len(mask)
        
        in_gap = False
        gap_start = 0
        
        for t in range(T):
            if mask[t]:
                if in_gap:
                    # End of gap - check if we should bridge
                    gap_length = t - gap_start
                    if gap_length <= self.max_gap_to_bridge:
                        # Bridge the gap
                        bridged[gap_start:t] = True
                    in_gap = False
            else:
                if not in_gap and t > 0 and mask[t-1]:
                    # Start of gap
                    in_gap = True
                    gap_start = t
        
        return bridged
    
    def _find_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous runs of True values."""
        runs = []
        T = len(mask)
        
        in_run = False
        start = 0
        
        for t in range(T):
            if mask[t] and not in_run:
                in_run = True
                start = t
            elif not mask[t] and in_run:
                in_run = False
                runs.append((start, t - 1))
        
        # Handle run at end
        if in_run:
            runs.append((start, T - 1))
        
        return runs


# =============================================================================
# Temporal Filters
# =============================================================================

class TemporalFilterType(Enum):
    NONE = "none"
    EMA = "ema"
    GAUSSIAN = "gaussian"
    SAVGOL = "savgol"
    BIDIRECTIONAL_EMA = "bidirectional_ema"


@dataclass
class TemporalFilterConfig:
    """Configuration for temporal smoothing."""
    filter_type: TemporalFilterType = TemporalFilterType.EMA
    ema_alpha: float = 0.2  # Higher = less smoothing
    gaussian_sigma: float = 2.0
    savgol_window: int = 7
    savgol_order: int = 2
    
    @classmethod
    def from_string(cls, filter_name: str, **kwargs) -> "TemporalFilterConfig":
        """Create config from filter name string."""
        try:
            filter_type = TemporalFilterType(filter_name.lower())
        except ValueError:
            filter_type = TemporalFilterType.EMA
        
        return cls(filter_type=filter_type, **kwargs)


class TemporalFilter:
    """
    Applies temporal smoothing to correction signals.
    
    Supports multiple filter types with consistent interface.
    """
    
    def __init__(self, config: TemporalFilterConfig):
        self.config = config
    
    def apply(self, signal: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Apply temporal filter to signal.
        
        Args:
            signal: Input array (T, ...) or (..., T, ...)
            axis: Time axis
        
        Returns:
            Filtered signal with same shape
        """
        _ensure_imports()
        
        cfg = self.config
        
        if cfg.filter_type == TemporalFilterType.NONE:
            return signal.copy()
        
        elif cfg.filter_type == TemporalFilterType.EMA:
            return self._apply_ema(signal, cfg.ema_alpha, axis)
        
        elif cfg.filter_type == TemporalFilterType.GAUSSIAN:
            return scipy_gaussian(signal, sigma=cfg.gaussian_sigma, axis=axis)
        
        elif cfg.filter_type == TemporalFilterType.SAVGOL:
            from scipy.signal import savgol_filter
            return savgol_filter(
                signal, 
                window_length=cfg.savgol_window, 
                polyorder=cfg.savgol_order,
                axis=axis
            )
        
        elif cfg.filter_type == TemporalFilterType.BIDIRECTIONAL_EMA:
            # Forward EMA
            forward = self._apply_ema(signal, cfg.ema_alpha, axis)
            # Backward EMA
            backward = self._apply_ema(signal[::-1] if axis == 0 else np.flip(signal, axis), 
                                       cfg.ema_alpha, axis)
            backward = backward[::-1] if axis == 0 else np.flip(backward, axis)
            # Average
            return (forward + backward) / 2
        
        else:
            return signal.copy()
    
    def _apply_ema(self, signal: np.ndarray, alpha: float, axis: int) -> np.ndarray:
        """Apply exponential moving average."""
        result = np.zeros_like(signal)
        T = signal.shape[axis]
        
        # Initialize with first value
        if axis == 0:
            result[0] = signal[0]
            for t in range(1, T):
                result[t] = alpha * signal[t] + (1 - alpha) * result[t-1]
        else:
            # Handle other axes by moving time axis to front
            signal_moved = np.moveaxis(signal, axis, 0)
            result_moved = np.zeros_like(signal_moved)
            result_moved[0] = signal_moved[0]
            for t in range(1, T):
                result_moved[t] = alpha * signal_moved[t] + (1 - alpha) * result_moved[t-1]
            result = np.moveaxis(result_moved, 0, axis)
        
        return result


# =============================================================================
# Memory Management for Long Videos
# =============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunked processing of long videos."""
    enabled: bool = True
    chunk_size: int = 500  # Frames per chunk
    overlap: int = 30  # Overlap between chunks for smooth stitching
    
    def get_chunks(self, total_frames: int) -> List[Tuple[int, int]]:
        """Get list of (start, end) frame indices for chunks."""
        if not self.enabled or total_frames <= self.chunk_size:
            return [(0, total_frames)]
        
        chunks = []
        start = 0
        while start < total_frames:
            end = min(start + self.chunk_size, total_frames)
            chunks.append((start, end))
            start = end - self.overlap
            if start >= total_frames - self.overlap:
                break
        
        return chunks


# =============================================================================
# TAPNet Integration
# =============================================================================

class TAPNetTracker:
    """
    Wrapper for TAPNet point tracking.
    
    Handles:
    - Lazy loading of TAPNet (Apache 2.0 license - commercial OK)
    - GPU memory management
    - Batched processing for long videos
    - Automatic fallback to optical flow if TAPNet unavailable
    
    Note: CoTracker is NOT used due to CC-BY-NC (non-commercial) license.
    """
    
    def __init__(self, logger: SmartFootContactLogger):
        self.log = logger
        self._model = None
        self._device = None
        self._model_type = None  # "tapnet" or "fallback"
    
    def _ensure_model(self):
        """Lazy load TAPNet model with optical flow fallback."""
        if self._model is not None:
            return
        
        _ensure_imports()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try TAPNet (Apache 2.0 - commercial friendly)
        if self._try_load_tapnet():
            return
        
        # Fallback to optical flow (OpenCV - Apache 2.0)
        self.log.warning("TAPNet not available - using optical flow fallback")
        self.log.warning("For better tracking, install: pip install tapnet")
        self._model = "fallback"
        self._model_type = "fallback"
    
    def _try_load_tapnet(self) -> bool:
        """Try to load TAPNet/TAPIR model (Apache 2.0 license)."""
        try:
            from tapnet.torch import tapir_model
            import os
            
            self.log.info(f"Loading TAPNet on {self._device}")
            
            # First, try local checkpoint paths (same as foot_tracker.py)
            checkpoint_paths = [
                "/workspace/ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt",
                "/workspace/models/tapir/bootstapir_checkpoint_v2.pt",
                os.path.expanduser("~/models/tapir/bootstapir_checkpoint_v2.pt"),
                "bootstapir_checkpoint_v2.pt",
                # Also check torch hub cache
                os.path.expanduser("~/.cache/torch/hub/checkpoints/bootstapir_checkpoint_v2.pt"),
                "/root/.cache/torch/hub/checkpoints/bootstapir_checkpoint_v2.pt",
            ]
            
            checkpoint_path = None
            for path in checkpoint_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    self.log.verbose(f"Found checkpoint: {path}")
                    break
            
            if checkpoint_path is not None:
                # Load from local file (matching foot_tracker.py approach)
                try:
                    # Use pyramid_level=1 to match the checkpoint architecture
                    model = tapir_model.TAPIR(pyramid_level=1)
                    model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
                    model = model.to(self._device)
                    model.eval()
                    
                    self._model = model
                    self._model_type = "tapnet"
                    self.log.info(f"TAPNet loaded from {checkpoint_path} (Apache 2.0 license)")
                    return True
                except Exception as e:
                    self.log.debug(f"Local checkpoint failed: {e}")
            
            # If no local checkpoint, try to download
            # Note: The downloaded checkpoint may have different architecture
            self.log.verbose("No local checkpoint found, trying to download...")
            
            try:
                # Try downloading the original TAPIR checkpoint (not BootsTAPIR)
                # which has a simpler architecture
                model = tapir_model.TAPIR(pyramid_level=0)
                
                checkpoint_url = "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt"
                checkpoint = torch.hub.load_state_dict_from_url(
                    checkpoint_url,
                    map_location=self._device
                )
                
                # Try to load with strict=False
                model.load_state_dict(checkpoint, strict=False)
                model = model.to(self._device)
                model.eval()
                
                self._model = model
                self._model_type = "tapnet"
                self.log.info("TAPNet loaded from URL (Apache 2.0 license)")
                return True
                
            except Exception as e:
                self.log.debug(f"Download checkpoint failed: {e}")
            
            return False
            
        except ImportError:
            self.log.verbose("TAPNet not installed (pip install tapnet)")
            return False
        except Exception as e:
            self.log.debug(f"TAPNet loading failed: {e}")
            return False
    
    def track_points(
        self,
        video: np.ndarray,
        query_points: np.ndarray,
        chunk_config: Optional[ChunkConfig] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track points through video.
        
        Args:
            video: (T, H, W, 3) video frames
            query_points: (N, 3) array of [frame, y, x] for each query point
            chunk_config: Optional chunking config for long videos
        
        Returns:
            tracks: (T, N, 2) array of [x, y] positions
            visibles: (T, N) boolean array of visibility
        """
        self._ensure_model()
        _ensure_imports()
        
        T, H, W, _ = video.shape
        N = len(query_points)
        
        if self._model_type == "fallback":
            self.log.info("Using optical flow tracking (OpenCV)")
            return self._track_optical_flow(video, query_points)
        
        self.log.info(f"Using TAPNet tracking")
        
        # Process in chunks if video is long
        if chunk_config and chunk_config.enabled and T > chunk_config.chunk_size:
            tracks, visibles = self._track_chunked(video, query_points, chunk_config)
        else:
            tracks, visibles = self._track_tapnet(video, query_points)
        
        return tracks, visibles
    
    def _track_tapnet(
        self,
        video: np.ndarray,
        query_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track points using TAPNet/TAPIR (Apache 2.0 license)."""
        T, H, W, _ = video.shape
        N = len(query_points)
        
        # Resize video for memory efficiency if too large
        # TAPIR works well at lower resolutions
        max_size = 512  # Maximum dimension
        scale = 1.0
        if max(H, W) > max_size:
            scale = max_size / max(H, W)
            new_H = int(H * scale)
            new_W = int(W * scale)
            # Make sure dimensions are multiples of 8 for TAPIR
            new_H = (new_H // 8) * 8
            new_W = (new_W // 8) * 8
            
            self.log.verbose(f"Resizing video from {W}x{H} to {new_W}x{new_H} for memory efficiency")
            
            import cv2
            video_resized = np.zeros((T, new_H, new_W, 3), dtype=video.dtype)
            for t in range(T):
                video_resized[t] = cv2.resize(video[t], (new_W, new_H))
            video = video_resized
            
            # Scale query points
            query_points = query_points.copy()
            query_points[:, 1] *= scale  # y
            query_points[:, 2] *= scale  # x
            
            H, W = new_H, new_W
        
        # Clear GPU memory before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Convert to tensor
        video_tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2) / 255.0
        video_tensor = video_tensor.unsqueeze(0).to(self._device)  # (1, T, C, H, W)
        
        # Convert query points
        query_tensor = torch.from_numpy(query_points).float().unsqueeze(0).to(self._device)
        
        try:
            with torch.no_grad():
                outputs = self._model(video_tensor, query_tensor)
            
            tracks = outputs['tracks'][0].cpu().numpy()  # (T, N, 2)
            visibles = outputs['occlusion'][0].cpu().numpy() < 0.5  # (T, N)
            
        finally:
            # Clean up GPU memory
            del video_tensor, query_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Scale tracks back to original resolution
        if scale != 1.0:
            tracks /= scale
        
        return tracks, visibles
    
    def _track_chunked(
        self,
        video: np.ndarray,
        query_points: np.ndarray,
        chunk_config: ChunkConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track points with chunked processing for memory efficiency."""
        T, H, W, _ = video.shape
        N = len(query_points)
        
        tracks = np.zeros((T, N, 2), dtype=np.float32)
        visibles = np.zeros((T, N), dtype=bool)
        weights = np.zeros((T, N), dtype=np.float32)  # For blending overlaps
        
        chunks = chunk_config.get_chunks(T)
        self.log.verbose(f"Processing {len(chunks)} chunks")
        
        for chunk_idx, (start, end) in enumerate(chunks):
            self.log.progress(chunk_idx, len(chunks), "Tracking chunks")
            
            # Adjust query points for this chunk
            chunk_queries = query_points.copy()
            chunk_queries[:, 0] = np.clip(chunk_queries[:, 0] - start, 0, end - start - 1)
            
            # Track this chunk using TAPNet
            chunk_video = video[start:end]
            chunk_tracks, chunk_visibles = self._track_tapnet(chunk_video, chunk_queries)
            
            # Blend into result
            chunk_len = end - start
            for t_local in range(chunk_len):
                t_global = start + t_local
                
                # Compute blend weight (ramp up/down in overlap regions)
                if chunk_idx > 0 and t_local < chunk_config.overlap:
                    weight = t_local / chunk_config.overlap
                elif chunk_idx < len(chunks) - 1 and t_local >= chunk_len - chunk_config.overlap:
                    weight = (chunk_len - t_local) / chunk_config.overlap
                else:
                    weight = 1.0
                
                tracks[t_global] += chunk_tracks[t_local] * weight
                visibles[t_global] |= chunk_visibles[t_local]
                weights[t_global] += weight
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Normalize by weights
        weights = np.maximum(weights, 1e-6)
        tracks /= weights[:, :, np.newaxis]
        
        return tracks, visibles
    
    def _track_optical_flow(
        self,
        video: np.ndarray,
        query_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback tracking using OpenCV optical flow."""
        import cv2
        
        T, H, W, _ = video.shape
        N = len(query_points)
        
        tracks = np.zeros((T, N, 2), dtype=np.float32)
        visibles = np.ones((T, N), dtype=bool)
        
        # Initialize tracks at query frames
        for i, (frame_idx, y, x) in enumerate(query_points):
            tracks[int(frame_idx), i] = [x, y]
        
        # Forward tracking
        for t in range(1, T):
            prev_gray = cv2.cvtColor(video[t-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(video[t], cv2.COLOR_RGB2GRAY)
            
            prev_pts = tracks[t-1].reshape(-1, 1, 2).astype(np.float32)
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None,
                winSize=(21, 21), maxLevel=3
            )
            
            if curr_pts is not None:
                tracks[t] = curr_pts.reshape(-1, 2)
                visibles[t] = status.flatten() == 1
        
        # Backward tracking for frames before query
        for i, (frame_idx, y, x) in enumerate(query_points):
            frame_idx = int(frame_idx)
            for t in range(frame_idx - 1, -1, -1):
                prev_gray = cv2.cvtColor(video[t+1], cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(video[t], cv2.COLOR_RGB2GRAY)
                
                prev_pts = tracks[t+1, i:i+1].reshape(-1, 1, 2).astype(np.float32)
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_pts, None
                )
                
                if curr_pts is not None and status[0] == 1:
                    tracks[t, i] = curr_pts.reshape(2)
                else:
                    visibles[t, i] = False
        
        return tracks, visibles


# =============================================================================
# Main Smart Foot Contact Processor
# =============================================================================

@dataclass
class SmartFootContactConfig:
    """Configuration for smart foot contact processing."""
    # Contact detection
    velocity_threshold: float = 3.0  # Pixels per frame for "stationary"
    min_contact_duration: int = 3  # Minimum frames for valid contact
    max_gap_to_bridge: int = 2  # Bridge gaps this small
    
    # Correction
    blend_in_frames: int = 3  # Ease-in frames at contact start
    blend_out_frames: int = 3  # Ease-out frames at contact end
    max_correction_px: float = 50.0  # Maximum correction per frame (pixels)
    
    # Temporal filtering
    filter_type: str = "ema"  # none, ema, gaussian, savgol, bidirectional_ema
    ema_alpha: float = 0.2
    gaussian_sigma: float = 2.0
    
    # Memory management
    enable_chunking: bool = True
    chunk_size: int = 24  # Reduced from 500 - TAPNet is memory hungry
    chunk_overlap: int = 6  # Reduced proportionally
    max_tracking_resolution: int = 512  # Max dimension for tracking (saves memory)
    
    # Foot indices (MHR skeleton)
    left_foot_idx: int = 10
    right_foot_idx: int = 11
    
    # Logging
    log_level: str = "verbose"  # silent, normal, verbose, debug


class SmartFootContactProcessor:
    """
    Main processor for smart foot contact with visual feedback.
    
    Pipeline:
    1. Extract foot positions from SAM3DBody
    2. Track feet in video using TAPNet
    3. Detect contacts based on 2D velocity
    4. Compute corrections to match video
    5. Apply temporal smoothing
    6. Update translations
    """
    
    def __init__(self, config: SmartFootContactConfig):
        self.config = config
        self.log = SmartFootContactLogger(
            LogLevel[config.log_level.upper()],
            prefix="SmartFootContact"
        )
        self.tapnet = TAPNetTracker(self.log)
        self.segment_detector = ContactSegmentDetector(
            min_duration=config.min_contact_duration,
            max_gap_to_bridge=config.max_gap_to_bridge,
            logger=self.log
        )
        self.temporal_filter = TemporalFilter(
            TemporalFilterConfig.from_string(
                config.filter_type,
                ema_alpha=config.ema_alpha,
                gaussian_sigma=config.gaussian_sigma
            )
        )
    
    def process(
        self,
        mesh_sequence: Dict,
        video_frames: np.ndarray,
        intrinsics: Optional[Dict] = None
    ) -> Dict:
        """
        Process mesh sequence with smart foot contact correction.
        
        Args:
            mesh_sequence: SAM3DBody mesh sequence
            video_frames: (T, H, W, 3) video frames as numpy array
            intrinsics: Camera intrinsics (focal_length, cx, cy)
        
        Returns:
            Updated mesh_sequence with corrected translations
        """
        _ensure_imports()
        
        self.log.info("=" * 60)
        self.log.info("SMART FOOT CONTACT PROCESSING")
        self.log.info("=" * 60)
        self.log.start_timer("total_processing")
        
        cfg = self.config
        
        # Get frames data
        frames_dict = mesh_sequence.get("frames", {})
        frame_keys = sorted(frames_dict.keys())
        frames = [frames_dict[k] for k in frame_keys]
        T = len(frames)
        
        self.log.info(f"Processing {T} frames")
        
        # Get video dimensions
        if video_frames is not None:
            _, H, W, _ = video_frames.shape
        else:
            # Fallback from mesh sequence
            first_frame = frames[0]
            img_size = first_frame.get("image_size", [512, 512])
            W, H = img_size[0], img_size[1]
        
        # Get intrinsics
        if intrinsics is None:
            intrinsics = self._extract_intrinsics(frames[0], W, H)
        
        self.log.verbose(f"Image size: {W}x{H}")
        self.log.verbose(f"Focal length: {intrinsics['focal_length']:.1f}px")
        
        # Step 1: Extract SAM3D foot positions and translations
        self.log.start_timer("extract_sam3d_data")
        sam3d_data = self._extract_sam3d_data(frames, cfg)
        self.log.end_timer("extract_sam3d_data")
        
        # Step 2: Track feet in video using TAPNet
        self.log.start_timer("tapnet_tracking")
        if video_frames is not None:
            foot_tracks_2d, track_visibility = self._track_feet_in_video(
                video_frames, sam3d_data, cfg
            )
        else:
            self.log.warning("No video frames provided - using SAM3D projections only")
            foot_tracks_2d = sam3d_data["feet_2d_projected"]
            track_visibility = np.ones((T, 2), dtype=bool)
        self.log.end_timer("tapnet_tracking")
        
        # Step 3: Compute 2D velocities and detect contacts
        self.log.start_timer("contact_detection")
        contacts, contact_segments = self._detect_contacts(
            foot_tracks_2d, track_visibility, cfg
        )
        self.log.end_timer("contact_detection")
        
        # Step 4: Compute reprojection errors and corrections
        self.log.start_timer("compute_corrections")
        corrections = self._compute_corrections(
            sam3d_data, foot_tracks_2d, contact_segments, intrinsics, cfg
        )
        self.log.end_timer("compute_corrections")
        
        # Step 5: Apply temporal smoothing
        self.log.start_timer("temporal_smoothing")
        smoothed_corrections = self.temporal_filter.apply(corrections, axis=0)
        self.log.end_timer("temporal_smoothing")
        
        # Step 6: Apply corrections to translations
        adjusted_translations = sam3d_data["translations"] + smoothed_corrections
        
        # Final smoothing pass on adjusted translations
        adjusted_translations = scipy_gaussian(adjusted_translations, sigma=1.0, axis=0)
        
        # Step 7: Update mesh sequence
        result = self._update_mesh_sequence(
            mesh_sequence, frame_keys, adjusted_translations, contacts, contact_segments
        )
        
        self.log.end_timer("total_processing")
        
        # Log summary
        self._log_summary(contact_segments, corrections)
        
        return result
    
    def _extract_intrinsics(self, frame: Dict, W: int, H: int) -> Dict:
        """Extract camera intrinsics from frame data."""
        focal = frame.get("focal_length", frame.get("focal_length_sam3d", W))
        cx = frame.get("cx", W / 2)
        cy = frame.get("cy", H / 2)
        
        return {
            "focal_length": float(focal) if focal else W,
            "cx": float(cx) if cx else W / 2,
            "cy": float(cy) if cy else H / 2,
            "width": W,
            "height": H
        }
    
    def _extract_sam3d_data(self, frames: List[Dict], cfg: SmartFootContactConfig) -> Dict:
        """Extract foot positions and translations from SAM3D frames."""
        T = len(frames)
        
        feet_3d = np.zeros((T, 2, 3), dtype=np.float32)  # [T, left/right, xyz]
        translations = np.zeros((T, 3), dtype=np.float32)
        feet_2d_projected = np.zeros((T, 2, 2), dtype=np.float32)  # [T, left/right, xy]
        
        for t, frame in enumerate(frames):
            # Get joint coordinates
            joints = frame.get("joint_coords")
            if joints is None:
                joints = frame.get("pred_joints_3d", frame.get("pred_keypoints_3d"))
            
            if joints is not None:
                if hasattr(joints, 'numpy'):
                    joints = joints.numpy()
                joints = np.array(joints)
                
                n_joints = len(joints)
                left_idx = min(cfg.left_foot_idx, n_joints - 1)
                right_idx = min(cfg.right_foot_idx, n_joints - 1)
                
                feet_3d[t, 0] = joints[left_idx]
                feet_3d[t, 1] = joints[right_idx]
            
            # Get translation
            trans = frame.get("pred_cam_t")
            if trans is not None:
                if hasattr(trans, 'numpy'):
                    trans = trans.numpy()
                translations[t] = np.array(trans)
            
            # Get 2D keypoints for projection reference
            kp2d = frame.get("pred_keypoints_2d")
            if kp2d is not None:
                if hasattr(kp2d, 'numpy'):
                    kp2d = kp2d.numpy()
                kp2d = np.array(kp2d)
                
                # Map to foot indices (may be different for 2D keypoints)
                # Using ankle indices for COCO-style keypoints
                left_2d_idx = 15 if len(kp2d) > 15 else len(kp2d) - 2
                right_2d_idx = 16 if len(kp2d) > 16 else len(kp2d) - 1
                
                feet_2d_projected[t, 0] = kp2d[left_2d_idx, :2]
                feet_2d_projected[t, 1] = kp2d[right_2d_idx, :2]
        
        return {
            "feet_3d": feet_3d,
            "translations": translations,
            "feet_2d_projected": feet_2d_projected,
        }
    
    def _track_feet_in_video(
        self,
        video: np.ndarray,
        sam3d_data: Dict,
        cfg: SmartFootContactConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Track feet in video using TAPNet."""
        T = len(video)
        
        # Initialize query points from SAM3D 2D projections at frame 0
        initial_feet_2d = sam3d_data["feet_2d_projected"][0]  # [2, 2] for left/right
        
        # Format: [frame, y, x]
        query_points = np.array([
            [0, initial_feet_2d[0, 1], initial_feet_2d[0, 0]],  # Left foot
            [0, initial_feet_2d[1, 1], initial_feet_2d[1, 0]],  # Right foot
        ], dtype=np.float32)
        
        # Track
        chunk_config = ChunkConfig(
            enabled=cfg.enable_chunking,
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap
        ) if cfg.enable_chunking else None
        
        tracks, visibles = self.tapnet.track_points(video, query_points, chunk_config)
        
        # tracks is [T, 2, 2] - [time, point, xy]
        return tracks, visibles
    
    def _detect_contacts(
        self,
        foot_tracks_2d: np.ndarray,
        visibility: np.ndarray,
        cfg: SmartFootContactConfig
    ) -> Tuple[np.ndarray, List[ContactSegment]]:
        """Detect foot contacts based on 2D velocity."""
        T = foot_tracks_2d.shape[0]
        
        # Compute velocities
        velocities = np.zeros((T, 2), dtype=np.float32)
        velocities[1:] = np.linalg.norm(np.diff(foot_tracks_2d, axis=0), axis=2)
        velocities[0] = velocities[1]  # Copy for first frame
        
        # Smooth velocities
        velocities = scipy_gaussian(velocities, sigma=1.0, axis=0)
        
        # Detect stationary (low velocity = contact)
        contacts = np.zeros((T, 2), dtype=bool)
        contacts[:, 0] = (velocities[:, 0] < cfg.velocity_threshold) & visibility[:, 0]
        contacts[:, 1] = (velocities[:, 1] < cfg.velocity_threshold) & visibility[:, 1]
        
        self.log.verbose(f"Raw contacts - Left: {contacts[:, 0].sum()}, Right: {contacts[:, 1].sum()} frames")
        
        # Detect segments
        all_segments = []
        
        # Left foot
        left_segments = self.segment_detector.detect(
            contacts[:, 0], "left", 
            foot_tracks_2d[:, 0], velocities[:, 0]
        )
        all_segments.extend(left_segments)
        
        # Right foot
        right_segments = self.segment_detector.detect(
            contacts[:, 1], "right",
            foot_tracks_2d[:, 1], velocities[:, 1]
        )
        all_segments.extend(right_segments)
        
        self.log.info(f"Detected {len(left_segments)} left, {len(right_segments)} right contact segments")
        
        return contacts, all_segments
    
    def _compute_corrections(
        self,
        sam3d_data: Dict,
        foot_tracks_2d: np.ndarray,
        contact_segments: List[ContactSegment],
        intrinsics: Dict,
        cfg: SmartFootContactConfig
    ) -> np.ndarray:
        """Compute translation corrections to match video."""
        T = sam3d_data["translations"].shape[0]
        corrections = np.zeros((T, 3), dtype=np.float32)
        
        focal = intrinsics["focal_length"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        
        for segment in contact_segments:
            foot_idx = 0 if segment.foot == "left" else 1
            
            for t in range(segment.start_frame, segment.end_frame + 1):
                # SAM3D projected foot position
                sam3d_foot_2d = sam3d_data["feet_2d_projected"][t, foot_idx]
                
                # Video tracked foot position (pin position)
                video_foot_2d = segment.pin_position_2d
                
                # 2D error
                error_2d = video_foot_2d - sam3d_foot_2d
                
                # Clamp error
                error_norm = np.linalg.norm(error_2d)
                if error_norm > cfg.max_correction_px:
                    error_2d = error_2d * (cfg.max_correction_px / error_norm)
                
                # Back-project to 3D correction (approximate)
                # Assuming depth from SAM3D translation
                depth = sam3d_data["translations"][t, 2]
                if abs(depth) < 0.1:
                    depth = 1.0  # Default depth
                
                correction_3d = np.array([
                    error_2d[0] * depth / focal,
                    -error_2d[1] * depth / focal,  # Y is inverted
                    0.0  # Don't adjust depth
                ], dtype=np.float32)
                
                # Apply blend weight
                blend = segment.get_blend_weight(
                    t, cfg.blend_in_frames, cfg.blend_out_frames
                )
                
                corrections[t] += correction_3d * blend
                
                # Log per-frame debug info
                self.log.log_frame(t, {
                    "foot": segment.foot,
                    "error_2d": error_2d.tolist(),
                    "error_norm": float(error_norm),
                    "correction_3d": correction_3d.tolist(),
                    "blend": blend
                })
        
        return corrections
    
    def _update_mesh_sequence(
        self,
        mesh_sequence: Dict,
        frame_keys: List,
        adjusted_translations: np.ndarray,
        contacts: np.ndarray,
        contact_segments: List[ContactSegment]
    ) -> Dict:
        """Update mesh sequence with corrected translations."""
        import copy
        result = copy.deepcopy(mesh_sequence)
        
        frames = result.get("frames", {})
        
        for i, key in enumerate(frame_keys):
            if key in frames:
                frames[key]["pred_cam_t"] = adjusted_translations[i].tolist()
                frames[key]["foot_contact"] = {
                    "left": bool(contacts[i, 0]),
                    "right": bool(contacts[i, 1])
                }
                
                # Also update smpl_params if present
                if "smpl_params" in frames[key] and isinstance(frames[key]["smpl_params"], dict):
                    frames[key]["smpl_params"]["transl"] = adjusted_translations[i].tolist()
        
        # Add metadata
        result["smart_foot_contact"] = {
            "method": "visual_feedback_loop",
            "config": {
                "velocity_threshold": self.config.velocity_threshold,
                "min_contact_duration": self.config.min_contact_duration,
                "filter_type": self.config.filter_type,
            },
            "segments": [
                {
                    "foot": seg.foot,
                    "start": seg.start_frame,
                    "end": seg.end_frame,
                    "duration": seg.duration,
                    "confidence": seg.confidence
                }
                for seg in contact_segments
            ],
            "timings": self.log.timings
        }
        
        return result
    
    def _log_summary(self, contact_segments: List[ContactSegment], corrections: np.ndarray):
        """Log processing summary."""
        self.log.info("=" * 60)
        self.log.info("PROCESSING SUMMARY")
        self.log.info("=" * 60)
        
        left_segs = [s for s in contact_segments if s.foot == "left"]
        right_segs = [s for s in contact_segments if s.foot == "right"]
        
        self.log.info(f"Contact segments: {len(left_segs)} left, {len(right_segs)} right")
        
        if contact_segments:
            total_contact_frames = sum(s.duration for s in contact_segments)
            self.log.info(f"Total contact frames: {total_contact_frames}")
            
            avg_confidence = np.mean([s.confidence for s in contact_segments])
            self.log.info(f"Average confidence: {avg_confidence:.2f}")
        
        correction_magnitude = np.linalg.norm(corrections, axis=1)
        self.log.info(f"Correction magnitude - Mean: {correction_magnitude.mean():.4f}, Max: {correction_magnitude.max():.4f}")
        
        self.log.info(self.log.get_summary())


# =============================================================================
# ComfyUI Node
# =============================================================================

class SmartFootContactNode:
    """
    Smart Foot Contact - Physics + Visual Feedback Loop
    
    Combines GroundLink/heuristic contact detection with TAPNet visual tracking
    to create accurate foot contact correction using video as ground truth.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Mesh sequence from SAM3DBody"
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames for visual tracking (highly recommended)"
                }),
                "velocity_threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Pixels/frame below which foot is considered stationary"
                }),
                "min_contact_duration": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Minimum frames for valid contact segment"
                }),
                "blend_frames": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 15,
                    "tooltip": "Frames to ease in/out of corrections"
                }),
                "temporal_filter": (["ema", "gaussian", "savgol", "bidirectional_ema", "none"], {
                    "default": "ema",
                    "tooltip": "Temporal smoothing filter type"
                }),
                "ema_alpha": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "EMA smoothing factor (lower = smoother)"
                }),
                "gaussian_sigma": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Gaussian filter sigma"
                }),
                "enable_chunking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable chunked processing for long videos (saves memory)"
                }),
                "chunk_size": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 100,
                    "step": 4,
                    "tooltip": "Frames per chunk when chunking enabled (lower = less memory)"
                }),
                "log_level": (["normal", "verbose", "debug", "silent"], {
                    "default": "verbose",
                    "tooltip": "Logging verbosity level"
                }),
            }
        }
    
    RETURN_TYPES = ("MESH_SEQUENCE", "STRING", "FOOT_CONTACTS")
    RETURN_NAMES = ("mesh_sequence", "debug_info", "foot_contacts")
    FUNCTION = "process"
    CATEGORY = "SAM3DBody2abc/Processing"
    
    def process(
        self,
        mesh_sequence: Dict,
        images=None,
        velocity_threshold: float = 3.0,
        min_contact_duration: int = 3,
        blend_frames: int = 3,
        temporal_filter: str = "ema",
        ema_alpha: float = 0.2,
        gaussian_sigma: float = 2.0,
        enable_chunking: bool = True,
        chunk_size: int = 24,
        log_level: str = "verbose",
    ) -> Tuple[Dict, str, Dict]:
        """Process mesh sequence with smart foot contact correction."""
        
        # Build config
        config = SmartFootContactConfig(
            velocity_threshold=velocity_threshold,
            min_contact_duration=min_contact_duration,
            blend_in_frames=blend_frames,
            blend_out_frames=blend_frames,
            filter_type=temporal_filter,
            ema_alpha=ema_alpha,
            gaussian_sigma=gaussian_sigma,
            enable_chunking=enable_chunking,
            chunk_size=chunk_size,
            log_level=log_level,
        )
        
        # Convert images to numpy if provided
        video_frames = None
        if images is not None:
            _ensure_imports()
            if torch.is_tensor(images):
                video_frames = (images.cpu().numpy() * 255).astype(np.uint8)
            else:
                video_frames = np.array(images)
                if video_frames.max() <= 1.0:
                    video_frames = (video_frames * 255).astype(np.uint8)
        
        # Process
        processor = SmartFootContactProcessor(config)
        result = processor.process(mesh_sequence, video_frames)
        
        # Build debug info string
        metadata = result.get("smart_foot_contact", {})
        segments = metadata.get("segments", [])
        timings = metadata.get("timings", {})
        
        debug_lines = [
            "=== SMART FOOT CONTACT RESULTS ===",
            f"Method: Visual Feedback Loop",
            f"Filter: {temporal_filter}",
            "",
            f"Contact Segments: {len(segments)}",
        ]
        
        for seg in segments[:10]:  # Limit output
            debug_lines.append(
                f"  {seg['foot']}: frames {seg['start']}-{seg['end']} "
                f"(dur={seg['duration']}, conf={seg['confidence']:.2f})"
            )
        
        if len(segments) > 10:
            debug_lines.append(f"  ... and {len(segments) - 10} more")
        
        debug_lines.extend(["", "=== TIMING ==="])
        for name, elapsed in timings.items():
            debug_lines.append(f"  {name}: {elapsed:.3f}s")
        
        debug_info = "\n".join(debug_lines)
        
        # Build foot_contacts output (compatible with existing visualizer)
        contacts_array = []
        frames_dict = result.get("frames", {})
        for key in sorted(frames_dict.keys()):
            fc = frames_dict[key].get("foot_contact", {"left": False, "right": False})
            contacts_array.append([fc["left"], fc["right"]])
        
        foot_contacts = {
            "contacts": contacts_array,
            "method": "smart_visual_feedback",
            "segments": segments,
            "debug": {
                "timings": timings,
                "config": {
                    "velocity_threshold": velocity_threshold,
                    "min_contact_duration": min_contact_duration,
                    "temporal_filter": temporal_filter,
                }
            }
        }
        
        return (result, debug_info, foot_contacts)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "SmartFootContact": SmartFootContactNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartFootContact": "🦶✨ Smart Foot Contact (Visual Feedback)",
}
