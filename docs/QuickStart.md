# SAM3DBody2abc Quick Start Guide

## Installation

1. Clone or download to `ComfyUI/custom_nodes/`
2. Install requirements: `pip install -r requirements.txt`
3. Restart ComfyUI

## Basic Workflow: Video to FBX

### Step 1: Load Video
Use "Load Video" node to import your video.

### Step 2: Process with SAM3DBody
Connect video frames to SAM3DBody node.

### Step 3: Accumulate Frames
Use **Mesh Accumulator** to collect all frames into a MESH_SEQUENCE.

### Step 4: Analyze Motion
Connect MESH_SEQUENCE to **Motion Analyzer**.

**Important settings:**
- `skeleton_mode`: "Simple Skeleton" for basic tracking, "Full Skeleton" for hands
- `depth_source`: "Auto" works best for most videos
- `subject_height_m`: Set to 0 for auto-estimation

### Step 5: (Optional) Smooth Trajectory
If trajectory is noisy, use **Trajectory Smoother**.

**Recommended settings:**
- `method`: "Savitzky-Golay (Best)" for most cases
- `strength`: Start at 0.5, increase if still noisy
- For Joint-Guided: Connect mesh_sequence and select reference joint

### Step 6: Export FBX
Use **Export Animated FBX** node.

**Key settings:**
- `world_translation`: "Body World (Compensated)" for walking/moving subjects
- `skeleton_mode`: "Rotations (Recommended)" for best Maya compatibility
- `include_camera`: True if you want camera motion

---

## Understanding Depth Sources

| Source | Best For | Notes |
|--------|----------|-------|
| Auto | Most cases | Uses tracked if available |
| SAM3DBody Only | Static camera | Uses pred_cam_t[2] |
| Tracked Depth | Moving camera | Requires Character Trajectory |

---

## Understanding World Translation

| Option | Use Case |
|--------|----------|
| None (Body at Origin) | Body stays at 0,0,0 - good for retargeting |
| Body World (Raw) | Includes camera movement effects |
| Body World (Compensated) | True world movement, camera removed |
| Depth Only (Z-axis) | Only depth positioning |

---

## Troubleshooting

### Noisy Trajectory
1. Try Trajectory Smoother with strength 0.5-0.8
2. Use Joint-Guided method with stable joint (Pelvis or L_Hip)

### Character Drifting
1. Use Camera Solver V2 to extract camera motion
2. Use "Body World (Compensated)" in export

### Wrong Scale in Maya
1. Check subject_height_m in Motion Analyzer
2. Verify scale_factor in FBX custom attributes

---

## FBX Custom Attributes

All analysis data is stored in FBX as custom attributes on the `SAM3DBody_Metadata` locator:

- `sam3dbody2abc_version`: Package version
- `depth_source`: Which depth method was used
- `trajectory_smoothing_method`: Smoothing applied
- `trajectory_jitter_reduction_pct`: How much noise was removed
- `scale_factor`: Conversion factor
- `subject_height_m`: Detected/specified height

Access in Maya:
```python
import maya.cmds as cmds
cmds.getAttr("SAM3DBody_Metadata.sam3dbody2abc_version")
```

---

*SAM3DBody2abc v4.8.8*
