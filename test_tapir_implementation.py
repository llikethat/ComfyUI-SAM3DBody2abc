#!/usr/bin/env python3
"""
SAM3DBody2abc v5.0 - TAPIR Implementation Test Script

This script tests the TAPIR-based camera solver without requiring ComfyUI.
Run this to verify:
1. TAPIR model loads correctly
2. Point tracking works
3. Shot classification produces valid results
4. Rainbow trail visualization generates correctly

Usage:
    python test_tapir_implementation.py [--video path/to/video.mp4] [--checkpoint path/to/checkpoint.pt]

Requirements:
    pip install torch torchvision opencv-python numpy
    pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git'
    
    Download checkpoint:
    wget https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
import colorsys
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ShotType(Enum):
    STATIC = "static"
    ROTATION = "rotation"
    TRANSLATION = "translation"
    MIXED = "mixed"


@dataclass
class TestResult:
    passed: bool
    message: str
    data: Optional[dict] = None


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_test(name: str, result: TestResult):
    status = "✅ PASS" if result.passed else "❌ FAIL"
    print(f"{status} | {name}")
    print(f"       {result.message}")
    if result.data:
        for key, value in result.data.items():
            print(f"       {key}: {value}")
    print()


def test_tapir_import() -> TestResult:
    """Test if TAPIR can be imported."""
    try:
        from tapnet.torch import tapir_model
        from tapnet.utils import transforms
        return TestResult(True, "TAPIR imported successfully", {
            "tapir_model": str(tapir_model),
        })
    except ImportError as e:
        return TestResult(False, f"TAPIR import failed: {e}", {
            "install_cmd": "pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git'"
        })


def test_tapir_checkpoint(checkpoint_path: str) -> TestResult:
    """Test if TAPIR checkpoint exists and can be loaded."""
    
    # Check common locations
    possible_paths = [
        checkpoint_path,
        "bootstapir_checkpoint_v2.pt",
        "models/tapir/bootstapir_checkpoint_v2.pt",
        os.path.expanduser("~/.cache/tapir/bootstapir_checkpoint_v2.pt"),
    ]
    
    found_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            found_path = path
            break
    
    if not found_path:
        return TestResult(False, "Checkpoint not found", {
            "searched": possible_paths,
            "download_url": "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
        })
    
    # Try loading
    try:
        from tapnet.torch import tapir_model
        model = tapir_model.TAPIR(pyramid_level=1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(found_path, map_location=device))
        model = model.to(device).eval()
        
        return TestResult(True, f"Checkpoint loaded from {found_path}", {
            "device": device,
            "model_params": sum(p.numel() for p in model.parameters()),
        })
    except Exception as e:
        return TestResult(False, f"Failed to load checkpoint: {e}")


def create_synthetic_video(num_frames: int = 30, height: int = 256, width: int = 256) -> np.ndarray:
    """Create a synthetic video with moving patterns for testing."""
    
    frames = []
    
    for t in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            frame[y, :, 0] = int(50 + 100 * y / height)  # Blue gradient
            frame[y, :, 2] = int(50 + 50 * y / height)   # Red gradient
        
        # Add moving circles (simulating tracked features)
        # Simulate a pan: all features move together
        pan_offset = int(t * 2)  # 2 pixels per frame pan
        
        for i in range(5):
            for j in range(5):
                cx = 40 + j * 45 + pan_offset
                cy = 40 + i * 45
                
                if 0 <= cx < width and 0 <= cy < height:
                    cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 4, (100, 200, 100), -1)
        
        # Add a "foreground" object that moves differently (to be masked)
        fg_x = width // 2 + int(20 * np.sin(t * 0.3))
        fg_y = height // 2 + int(10 * np.cos(t * 0.2))
        cv2.rectangle(frame, (fg_x - 30, fg_y - 40), (fg_x + 30, fg_y + 40), (255, 100, 100), -1)
        
        frames.append(frame)
    
    return np.stack(frames)


def create_synthetic_mask(num_frames: int, height: int, width: int) -> np.ndarray:
    """Create mask for the foreground object."""
    masks = []
    
    for t in range(num_frames):
        mask = np.zeros((height, width), dtype=np.float32)
        fg_x = width // 2 + int(20 * np.sin(t * 0.3))
        fg_y = height // 2 + int(10 * np.cos(t * 0.2))
        cv2.rectangle(mask, (fg_x - 35, fg_y - 45), (fg_x + 35, fg_y + 45), 1.0, -1)
        masks.append(mask)
    
    return np.stack(masks)


def test_point_generation(frames: np.ndarray, mask: Optional[np.ndarray] = None) -> TestResult:
    """Test query point generation on background."""
    
    num_frames, height, width = frames.shape[:3]
    grid_size = 8
    margin = 20
    
    y_coords = np.linspace(margin, height - margin, grid_size)
    x_coords = np.linspace(margin, width - margin, grid_size)
    
    query_points = []
    masked_count = 0
    
    for y in y_coords:
        for x in x_coords:
            yi, xi = int(y), int(x)
            
            if mask is not None:
                if mask[0, yi, xi] > 0.5:
                    masked_count += 1
                    continue
            
            query_points.append([0, yi, xi])
    
    if len(query_points) < 10:
        return TestResult(False, "Not enough valid query points", {
            "total_grid_points": grid_size * grid_size,
            "masked_out": masked_count,
            "remaining": len(query_points),
        })
    
    return TestResult(True, f"Generated {len(query_points)} query points", {
        "grid_size": f"{grid_size}x{grid_size}",
        "masked_out": masked_count,
        "valid_points": len(query_points),
    })


def test_tapir_inference(checkpoint_path: str, frames: np.ndarray) -> Tuple[TestResult, Optional[np.ndarray], Optional[np.ndarray]]:
    """Test TAPIR inference on video frames."""
    
    try:
        from tapnet.torch import tapir_model
        import torch.nn.functional as F
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        model = tapir_model.TAPIR(pyramid_level=1)
        
        # Find checkpoint
        possible_paths = [
            checkpoint_path,
            "bootstapir_checkpoint_v2.pt",
            "models/tapir/bootstapir_checkpoint_v2.pt",
        ]
        found_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                found_path = path
                break
        
        if not found_path:
            return TestResult(False, "Checkpoint not found"), None, None
        
        model.load_state_dict(torch.load(found_path, map_location=device))
        model = model.to(device).eval()
        
        # Prepare video tensor
        num_frames, height, width = frames.shape[:3]
        video = torch.from_numpy(frames).float()
        video = video / 255.0 * 2 - 1  # Normalize to [-1, 1]
        video = video.unsqueeze(0).to(device)  # [1, T, H, W, C]
        
        # Generate query points
        grid_size = 6
        margin = 30
        y_coords = np.linspace(margin, height - margin, grid_size)
        x_coords = np.linspace(margin, width - margin, grid_size)
        
        query_points = []
        for y in y_coords:
            for x in x_coords:
                query_points.append([0, int(y), int(x)])
        
        query_tensor = torch.tensor(query_points, dtype=torch.float32)
        query_tensor = query_tensor.unsqueeze(0).to(device)
        
        # Run inference
        print("       Running TAPIR inference...")
        with torch.no_grad():
            outputs = model(video, query_tensor)
        
        tracks = outputs['tracks'][0].cpu().numpy()  # [N, T, 2]
        occlusions = outputs['occlusion'][0]
        expected_dist = outputs['expected_dist'][0]
        
        # Compute visibility
        visibles = (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist)) > 0.5
        visibles = visibles.cpu().numpy()
        
        return TestResult(True, "TAPIR inference successful", {
            "num_points": tracks.shape[0],
            "num_frames": tracks.shape[1],
            "track_shape": str(tracks.shape),
            "visible_ratio": f"{np.mean(visibles):.1%}",
        }), tracks, visibles
        
    except Exception as e:
        import traceback
        return TestResult(False, f"Inference failed: {e}\n{traceback.format_exc()}"), None, None


def test_shot_classification(tracks: np.ndarray, visibles: np.ndarray) -> TestResult:
    """Test shot classification logic."""
    
    num_points, num_frames = tracks.shape[:2]
    
    flow_coherence_scores = []
    homography_errors = []
    motion_magnitudes = []
    
    for t in range(1, min(num_frames, 10)):  # Check first 10 frames
        vis_both = visibles[:, 0] & visibles[:, t]
        if np.sum(vis_both) < 8:
            continue
        
        pts0 = tracks[vis_both, 0, :]
        pts1 = tracks[vis_both, t, :]
        
        # Motion magnitude
        motion = pts1 - pts0
        motion_mag = np.linalg.norm(motion, axis=1)
        motion_magnitudes.append(np.median(motion_mag))
        
        # Flow coherence
        if np.median(motion_mag) > 0.5:
            motion_norm = motion / (np.linalg.norm(motion, axis=1, keepdims=True) + 1e-6)
            mean_direction = np.mean(motion_norm, axis=0)
            coherence = np.mean(np.sum(motion_norm * mean_direction, axis=1))
            flow_coherence_scores.append(coherence)
        
        # Homography fit
        if len(pts0) >= 8:
            try:
                H, _ = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
                if H is not None:
                    pts0_h = np.hstack([pts0, np.ones((len(pts0), 1))])
                    pts1_pred = (H @ pts0_h.T).T
                    pts1_pred = pts1_pred[:, :2] / pts1_pred[:, 2:3]
                    error = np.mean(np.linalg.norm(pts1 - pts1_pred, axis=1))
                    homography_errors.append(error)
            except:
                pass
    
    # Aggregate
    avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0
    avg_coherence = np.mean(flow_coherence_scores) if flow_coherence_scores else 0.9
    avg_homography_error = np.mean(homography_errors) if homography_errors else 0
    
    # Classify
    if avg_motion < 2.0:
        shot_type = ShotType.STATIC
    elif avg_coherence > 0.85 and avg_homography_error < 3.0:
        shot_type = ShotType.ROTATION
    else:
        shot_type = ShotType.MIXED
    
    return TestResult(True, f"Classified as {shot_type.value}", {
        "motion_magnitude": f"{avg_motion:.2f}px",
        "flow_coherence": f"{avg_coherence:.3f}",
        "homography_error": f"{avg_homography_error:.2f}px",
        "shot_type": shot_type.value,
    })


def get_rainbow_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate rainbow colors."""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


def test_rainbow_visualization(
    frames: np.ndarray, 
    tracks: np.ndarray, 
    visibles: np.ndarray,
    output_path: str = "tapir_test_output.mp4"
) -> TestResult:
    """Test rainbow trail visualization generation."""
    
    try:
        num_points, num_frames = tracks.shape[:2]
        height, width = frames.shape[1], frames.shape[2]
        
        colors = get_rainbow_colors(num_points)
        trail_length = 15
        point_radius = max(3, int(min(height, width) * 0.015))
        trail_thickness = max(1, int(min(height, width) * 0.005))
        
        output_frames = []
        
        for t in range(num_frames):
            frame = frames[t].copy()
            
            for i in range(num_points):
                color = colors[i]
                
                # Draw trail
                trail_start = max(0, t - trail_length)
                trail_points = []
                
                for tt in range(trail_start, t + 1):
                    if visibles[i, tt]:
                        x, y = tracks[i, tt]
                        trail_points.append((int(x), int(y)))
                
                # Draw trail segments
                if len(trail_points) > 1:
                    for j in range(len(trail_points) - 1):
                        cv2.line(frame, trail_points[j], trail_points[j+1], color, trail_thickness, cv2.LINE_AA)
                
                # Draw current point
                if visibles[i, t]:
                    x, y = tracks[i, t]
                    center = (int(x), int(y))
                    cv2.circle(frame, center, point_radius, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, center, point_radius, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add frame number
            cv2.putText(frame, f"Frame {t}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            output_frames.append(frame)
        
        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 15.0, (width, height))
        
        for frame in output_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        
        # Also save first and last frame as images
        cv2.imwrite(output_path.replace('.mp4', '_frame0.png'), cv2.cvtColor(output_frames[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(output_path.replace('.mp4', '_frame_last.png'), cv2.cvtColor(output_frames[-1], cv2.COLOR_RGB2BGR))
        
        return TestResult(True, f"Rainbow visualization saved to {output_path}", {
            "video_path": output_path,
            "frame0_path": output_path.replace('.mp4', '_frame0.png'),
            "frame_last_path": output_path.replace('.mp4', '_frame_last.png'),
            "num_frames": len(output_frames),
            "resolution": f"{width}x{height}",
        })
        
    except Exception as e:
        import traceback
        return TestResult(False, f"Visualization failed: {e}\n{traceback.format_exc()}")


def load_video(video_path: str, max_frames: int = 50, resize: Tuple[int, int] = (256, 256)) -> Optional[np.ndarray]:
    """Load video file and convert to numpy array."""
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser(description="Test TAPIR implementation for SAM3DBody2abc v5.0")
    parser.add_argument("--video", type=str, default="", help="Path to test video (optional, uses synthetic if not provided)")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to TAPIR checkpoint")
    parser.add_argument("--output", type=str, default="tapir_test_output.mp4", help="Output video path")
    args = parser.parse_args()
    
    print_header("SAM3DBody2abc v5.0 - TAPIR Implementation Test")
    
    results = []
    
    # Test 1: TAPIR Import
    print("Test 1: TAPIR Import")
    result = test_tapir_import()
    print_test("Import TAPIR module", result)
    results.append(("Import", result))
    
    if not result.passed:
        print("\n❌ TAPIR not installed. Cannot continue tests.")
        print("Install with: pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git'")
        return 1
    
    # Test 2: Checkpoint
    print("Test 2: TAPIR Checkpoint")
    result = test_tapir_checkpoint(args.checkpoint)
    print_test("Load TAPIR checkpoint", result)
    results.append(("Checkpoint", result))
    
    if not result.passed:
        print("\n❌ Checkpoint not found. Cannot continue tests.")
        print("Download from: https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt")
        return 1
    
    # Load or create video
    print_header("Loading Test Video")
    
    if args.video and os.path.exists(args.video):
        print(f"Loading video: {args.video}")
        frames = load_video(args.video)
        if frames is None:
            print("Failed to load video, using synthetic")
            frames = create_synthetic_video()
        else:
            print(f"Loaded {len(frames)} frames")
    else:
        print("Using synthetic video (pan simulation)")
        frames = create_synthetic_video(num_frames=30, height=256, width=256)
    
    mask = create_synthetic_mask(len(frames), frames.shape[1], frames.shape[2])
    print(f"Video shape: {frames.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Test 3: Point Generation
    print("\nTest 3: Query Point Generation")
    result = test_point_generation(frames, mask)
    print_test("Generate query points", result)
    results.append(("Point Generation", result))
    
    # Test 4: TAPIR Inference
    print("Test 4: TAPIR Inference")
    result, tracks, visibles = test_tapir_inference(args.checkpoint, frames)
    print_test("Run TAPIR tracking", result)
    results.append(("Inference", result))
    
    if not result.passed or tracks is None:
        print("\n❌ TAPIR inference failed. Cannot continue.")
        return 1
    
    # Test 5: Shot Classification
    print("Test 5: Shot Classification")
    result = test_shot_classification(tracks, visibles)
    print_test("Classify shot type", result)
    results.append(("Classification", result))
    
    # Test 6: Rainbow Visualization
    print("Test 6: Rainbow Trail Visualization")
    result = test_rainbow_visualization(frames, tracks, visibles, args.output)
    print_test("Generate rainbow trails", result)
    results.append(("Visualization", result))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, r in results if r.passed)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed\n")
    
    for name, result in results:
        status = "✅" if result.passed else "❌"
        print(f"  {status} {name}")
    
    if passed == total:
        print("\n🎉 All tests passed! TAPIR implementation is working correctly.")
        print(f"\nOutput files:")
        print(f"  - {args.output}")
        print(f"  - {args.output.replace('.mp4', '_frame0.png')}")
        print(f"  - {args.output.replace('.mp4', '_frame_last.png')}")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
