# SAM3DBody2abc Node Reference

Complete reference for all nodes in SAM3DBody2abc v5.0.1

---

## Table of Contents

1. [Video Processing](#video-processing)
2. [Camera Solving](#camera-solving)
3. [MegaSAM (Optional)](#megasam-optional)
4. [Motion Analysis](#motion-analysis)
5. [Multi-Camera](#multi-camera)
6. [Export](#export)
7. [Debug & Visualization](#debug--visualization)
8. [Datatype Reference](#datatype-reference)

---

## Video Processing

### 🎬 Video Batch Processor

Core node for extracting 3D body data from video frames.

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `images` | IMAGE | Video frames [N, H, W, 3] |
| `sam3dbody_model` | SAM3DBODY_MODEL | Loaded SAM3DBody model |
| `masks` | MASK | Optional foreground masks |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `mesh_sequence` | MESH_SEQUENCE | Per-frame mesh and skeleton data |
| `status` | STRING | Processing status |

---

### 🏃 Character Trajectory Tracker

Tracks character position through video using TAPIR point tracking + depth.

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `images` | IMAGE | Video frames |
| `mesh_sequence` | MESH_SEQUENCE | From Video Batch Processor |
| `depth_maps` | IMAGE | Optional depth maps |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `mesh_sequence` | MESH_SEQUENCE | Updated with tracked_depth |
| `trajectory_vis` | IMAGE | Visualization of tracking |

---

## Camera Solving

### 📷 Camera Solver

Standard camera solver for pan/tilt detection from background motion.

**Best for:** Static tripod shots, pan/tilt movements, simple rotations

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `images` | IMAGE | Video frames |
| `solving_method` | COMBO | Static/Auto/Rotation Only/Full 6-DOF |
| `foreground_masks` | MASK | Optional masks to exclude foreground |
| `quality_mode` | COMBO | Fast/Balanced/Best |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `camera_extrinsics` | CAMERA_EXTRINSICS | Per-frame camera rotation/translation |
| `debug_vis` | IMAGE | Feature matching visualization |
| `status` | STRING | Solving status |

---

### 📷 Camera Extrinsics from JSON

Load camera data from external tracking applications.

**Supported Formats:** PFTrack, 3DEqualizer, SynthEyes, Maya, Nuke, After Effects

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `json_path` | STRING | Path to camera JSON file |
| `coordinate_system` | COMBO | Source coordinate system |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `camera_extrinsics` | CAMERA_EXTRINSICS | Loaded camera data |

---

## MegaSAM (Optional)

High-quality camera solving for complex shots. Requires separate installation.

### 🎬 MegaSAM Camera Solver

Full 6-DOF camera solving using DROID-SLAM + Consistent Video Depth.

**Best for:** Dolly shots, crane movements, handheld footage, dynamic scenes

**Requirements:**
- lietorch, kornia, torch-scatter
- Compiled DROID base module
- Checkpoints in `ComfyUI/models/megasam/`

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `images` | IMAGE | Video frames |
| `coordinate_system` | COMBO | Maya (Y-up) / Blender (Z-up) / OpenCV |
| `focal_length_px` | FLOAT | Initial focal length (0 = auto) |
| `optimize_focal` | BOOLEAN | Optimize focal during tracking |
| `model_path` | STRING | Custom checkpoint directory |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `camera_extrinsics` | CAMERA_EXTRINSICS | Full 6-DOF camera data |
| `depth_maps` | IMAGE | Dense depth maps [N, H, W, 1] |
| `motion_masks` | IMAGE | Motion probability [N, H, W, 1] |
| `status` | STRING | Processing status |

**Checkpoint Paths:**
```
ComfyUI/models/megasam/
├── megasam_final.pth       # Main MegaSAM weights
├── depth_anything_vitl14.pth  # Depth estimation
└── raft-things.pth         # Optical flow
```

---

### 📁 MegaSAM Data Loader

Load pre-computed MegaSAM results from .npz files.

**Use when:** You've run MegaSAM offline and want to use results in ComfyUI

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `npz_path` | STRING | Path to `{scene}_sgd_cvd_hr.npz` file |
| `coordinate_system` | COMBO | Target coordinate system |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `camera_extrinsics` | CAMERA_EXTRINSICS | Loaded camera data |
| `depth_maps` | IMAGE | Depth visualization |
| `images` | IMAGE | Original images from file |
| `status` | STRING | Load status |

**NPZ File Format:**
```python
{
    "cam_c2w": np.float32,   # [N, 4, 4] camera-to-world matrices
    "depths": np.float16,    # [N, H, W] depth maps
    "intrinsic": np.float32, # [3, 3] K matrix
    "images": np.uint8,      # [N, H, W, 3] optional
}
```

---

## Motion Analysis

### 📊 Motion Analyzer

Analyze character motion: speed, direction, foot contacts, trajectory.

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `mesh_sequence` | MESH_SEQUENCE | From Video Batch Processor |
| `camera_extrinsics` | CAMERA_EXTRINSICS | Optional camera data |
| `depth_source` | COMBO | Auto/SAM3DBody Only/Tracked Depth |
| `fps` | FLOAT | Frames per second |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `subject_motion` | SUBJECT_MOTION | Analyzed motion data |
| `trajectory_vis` | IMAGE | Trajectory visualization |
| `trajectory_topview` | IMAGE | Top-down trajectory view |
| `foot_contact_vis` | IMAGE | Foot contact detection |
| `status` | STRING | Analysis summary |

---

### 📏 Scale Info Display

Display scale and measurement information.

---

## Multi-Camera

### 🎯 Camera Auto-Calibrator

Automatically compute relative camera positions from person keypoints.

**Use when:** You have 2 synchronized camera views of the same person

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `mesh_sequence_a` | MESH_SEQUENCE | Camera A data |
| `mesh_sequence_b` | MESH_SEQUENCE | Camera B data |
| `person_height_m` | FLOAT | Known person height for scale |
| `focal_length_px` | FLOAT | Camera focal length |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `calibration` | CALIBRATION_DATA | Camera calibration result |
| `debug_vis` | IMAGE | Calibration visualization |
| `status` | STRING | Calibration summary with viewing angles |

---

### 🔺 Multi-Camera Triangulator

Triangulate 3D positions from 2 camera views.

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `mesh_sequence_a` | MESH_SEQUENCE | Camera A data |
| `mesh_sequence_b` | MESH_SEQUENCE | Camera B data |
| `calibration` | CALIBRATION_DATA | From Auto-Calibrator |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `mesh_sequence` | MESH_SEQUENCE | Triangulated 3D data |
| `debug_vis` | IMAGE | Triangulation visualization |

---

## Export

### 📦 Export Animated FBX

Export animated FBX for Maya/Blender.

**Inputs:**
| Name | Type | Description |
|------|------|-------------|
| `mesh_sequence` | MESH_SEQUENCE | Body data |
| `subject_motion` | SUBJECT_MOTION | Motion analysis |
| `camera_extrinsics` | CAMERA_EXTRINSICS | Optional camera data |
| `camera_intrinsics` | CAMERA_INTRINSICS | Optional intrinsics |
| `output_path` | STRING | Output file path |
| `fps` | INT | Frame rate |
| `world_translation_mode` | COMBO | None/Root Locator/Baked |
| `include_skeleton` | BOOLEAN | Include skeleton |
| `include_mesh` | BOOLEAN | Include mesh |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `fbx_path` | STRING | Path to exported FBX |
| `status` | STRING | Export summary |

---

### 📦 Export FBX from JSON

Export FBX from saved JSON data.

---

## Debug & Visualization

### 🔍 Verify Overlay

Overlay mesh on single frame for verification.

### 🔍 Verify Overlay (Sequence)

Overlay mesh on video sequence.

### 🎥 FBX Animation Viewer

Preview FBX animation in ComfyUI.

### 🔄 Temporal Smoothing

Reduce jitter in single-camera depth estimates.

**Methods:** Gaussian, EMA (bidirectional), Savitzky-Golay

---

## Datatype Reference

### CAMERA_EXTRINSICS

Per-frame camera rotation and translation data.

```python
{
    "num_frames": int,
    "image_width": int,
    "image_height": int,
    "source": str,  # "CameraSolver", "MegaSAM", "JSON", etc.
    "solving_method": str,
    "coordinate_system": str,  # "maya", "blender", "opencv"
    "units": str,  # "radians" or "degrees"
    "has_translation": bool,
    "rotations": [
        {
            "frame": int,
            "pan": float,      # Y-rotation (radians)
            "tilt": float,     # X-rotation (radians)
            "roll": float,     # Z-rotation (radians)
            "pan_deg": float,  # Convenience
            "tilt_deg": float,
            "roll_deg": float,
            "tx": float,       # Translation X
            "ty": float,       # Translation Y
            "tz": float,       # Translation Z
        },
        ...
    ],
    # Optional MegaSAM-specific:
    "focal_length_px": float,
    "principal_point": [float, float],
}
```

### MESH_SEQUENCE

Per-frame mesh and skeleton data.

```python
{
    "vertices": list,      # Per-frame vertex positions
    "faces": np.array,     # Face indices (constant)
    "joint_coords": list,  # Per-frame joint positions
    "joint_parents": list, # Skeleton hierarchy
    "pred_cam_t": list,    # SAM3DBody camera parameters
    "tracked_depth": list, # Optional tracked depth
    # ... additional fields
}
```

### SUBJECT_MOTION

Analyzed motion data from Motion Analyzer.

```python
{
    "body_world_3d": list,             # Per-frame world position
    "body_world_3d_compensated": list, # Camera-compensated position
    "depth_estimate": list,            # Depth per frame
    "speed_mps": list,                 # Speed in m/s
    "direction": list,                 # Movement direction
    "foot_contacts": list,             # Left/right foot ground contact
    # ... additional fields
}
```

### CALIBRATION_DATA

Multi-camera calibration from Auto-Calibrator.

```python
{
    "R": np.array,         # [3, 3] Relative rotation
    "t": np.array,         # [3] Relative translation
    "scale": float,        # Scale factor
    "K": np.array,         # [3, 3] Intrinsic matrix
    "viewing_angles": {
        "camera_a": {
            "viewing_angle_deg": float,
            "azimuth_deg": float,
            "elevation_deg": float,
            "distance_m": float,
        },
        "camera_b": { ... },
    },
}
```

---

## Coordinate Systems

| System | Up | Forward | Right |
|--------|-----|---------|-------|
| **Maya** | +Y | -Z | +X |
| **Blender** | +Z | +Y | +X |
| **OpenCV** | -Y | +Z | +X |

MegaSAM outputs in OpenCV convention and converts to Maya/Blender automatically.

---

*Last updated: January 2025 (v5.0.1)*
