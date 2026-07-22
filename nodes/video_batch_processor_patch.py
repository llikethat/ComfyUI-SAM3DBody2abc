"""
Video Batch Processor Patch for Tracked Keypoints
=================================================

This patch modifies SAM3DBodyBatchProcessor to accept pre-tracked 2D keypoints
from Keypoint2DTracker and use them instead of per-frame detection.

Installation:
1. Add tracked_keypoints_2d input to INPUT_TYPES
2. Modify process_batch to use tracked keypoints
3. Store tracked keypoints in output frames

FULL INTEGRATION NOTE:
---------------------
For true temporal consistency, the SMPL fitting optimization inside 
SAM3DBodyEstimator.process_one_image() would need to use tracked 2D keypoints
as the fitting target instead of per-frame detected ones.

This would require modifications to:
- SAM3DBodyEstimator to accept external 2D keypoints
- The HMR/SMPL optimization loop to use tracked keypoints

Current patch: Stores tracked 2D in output, useful for overlay/visualization.
"""

# === Changes to INPUT_TYPES ===

# Add to optional inputs:
"""
"tracked_keypoints_2d": ("KEYPOINTS_2D", {
    "tooltip": "Pre-tracked 2D keypoints from Keypoint2DTracker (enables temporal consistency)"
}),
"""


# === Changes to process_batch method ===

def process_batch_with_tracking(
    self,
    model,
    images,
    mask=None,
    tracked_keypoints_2d=None,  # NEW PARAMETER
    # ... other params ...
):
    """
    Modified process_batch that uses tracked 2D keypoints.
    
    If tracked_keypoints_2d is provided:
    - Skip per-frame 2D detection (use tracked instead)
    - Store tracked keypoints in frame output
    - 3D fitting still happens per-frame (until estimator is modified)
    """
    
    # Extract tracked keypoints array
    tracked_kp = None
    if tracked_keypoints_2d is not None:
        tracked_kp = tracked_keypoints_2d.get("keypoints")  # (N, 70, 2)
        num_tracked_frames = tracked_keypoints_2d.get("num_frames", 0)
        print(f"[SAM3DBody2abc] Using tracked 2D keypoints: {num_tracked_frames} frames")
    
    # ... existing processing loop ...
    
    for idx, frame_idx in enumerate(frame_indices):
        # ... existing frame processing ...
        
        # After getting outputs from model:
        for sorted_idx, orig_idx, output in persons_to_process:
            
            # === NEW: Override pred_keypoints_2d with tracked version ===
            if tracked_kp is not None and idx < len(tracked_kp):
                output["pred_keypoints_2d"] = tracked_kp[idx]  # Use tracked
                output["keypoints_source"] = "tracked"  # Mark source
            else:
                output["keypoints_source"] = "per_frame"
            
            # ... rest of existing processing ...
            
            # Store in mesh_data
            mesh_data = {
                # ... existing fields ...
                "pred_keypoints_2d": output.get("pred_keypoints_2d"),  # Tracked or per-frame
                "keypoints_source": output.get("keypoints_source", "per_frame"),
            }


# === Full modified INPUT_TYPES ===

MODIFIED_INPUT_TYPES = {
    "required": {
        "model": ("SAM3D_MODEL",),
        "images": ("IMAGE",),
    },
    "optional": {
        "mask": ("MASK",),
        "tracked_keypoints_2d": ("KEYPOINTS_2D", {  # NEW
            "tooltip": "Pre-tracked 2D keypoints from Keypoint2DTracker"
        }),
        "bbox_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0}),
        "inference_type": (["full", "body", "hand"], {"default": "full"}),
        "start_frame": ("INT", {"default": 0, "min": 0}),
        "end_frame": ("INT", {"default": -1, "min": -1}),
        "skip_frames": ("INT", {"default": 1, "min": 1, "max": 30}),
        "fov": ("FLOAT", {"default": 55.0, "min": 0.0, "max": 150.0}),
        "focal_length_mm": ("FLOAT", {"default": 0.0}),
        "sensor_width_mm": ("FLOAT", {"default": 0.0}),
        "focal_length_px": ("FLOAT", {"default": 0.0}),
        "auto_calibrate": ("BOOLEAN", {"default": False}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
        "person_index": ("INT", {"default": 0, "min": -1, "max": 10}),
    }
}


# === Architecture for full integration ===

"""
ARCHITECTURE FOR FULL TEMPORAL CONSISTENCY
==========================================

Current flow (per-frame):
    Frame N → SAM3D detect 2D → Fit SMPL → 3D mesh
    Frame N+1 → SAM3D detect 2D → Fit SMPL → 3D mesh
    Result: Jittery 2D detection causes jittery 3D mesh

Ideal flow (tracked):
    Frame 0 → SAM3D detect 2D (70 keypoints)
    Frame 0-N → TAPIR tracks those 70 points
    
    For each frame:
        Use TRACKED 2D keypoints → Fit SMPL → 3D mesh
    
    Result: Consistent 2D → Consistent 3D mesh

To implement ideal flow, SAM3DBodyEstimator needs:
    
    def process_one_image(
        self,
        image,
        bboxes=None,
        masks=None,
        external_keypoints_2d=None,  # NEW: Use these instead of detecting
        ...
    ):
        if external_keypoints_2d is not None:
            # Skip 2D detection
            keypoints_2d = external_keypoints_2d
        else:
            # Detect 2D keypoints as usual
            keypoints_2d = self.detect_2d_keypoints(image, bboxes)
        
        # Use keypoints_2d for SMPL fitting
        smpl_params = self.fit_smpl(keypoints_2d, ...)
        
        return {
            "pred_keypoints_2d": keypoints_2d,
            "pred_keypoints_3d": smpl.get_joints_3d(smpl_params),
            "pred_vertices": smpl.get_vertices(smpl_params),
            ...
        }

This would require understanding SAM3D's internal architecture - specifically
how it separates 2D detection from SMPL fitting.
"""
