# SAM3DBody2abc Node Reference

## Version 5.4.0

This document provides detailed information about all nodes in the SAM3DBody2abc extension.

---

## Table of Contents

1. [Processing Nodes](#processing-nodes)
2. [Analysis Nodes](#analysis-nodes)
3. [Export Nodes](#export-nodes)
4. [Camera Nodes](#camera-nodes)
5. [Multi-Camera Nodes](#multi-camera-nodes)
6. [Utility Nodes](#utility-nodes)
7. [External Intrinsics](#external-intrinsics)

---

## Processing Nodes

### SAM3DBody2abc Mesh Accumulator

Accumulates per-frame SAM3DBody outputs into a MESH_SEQUENCE.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| sam3dbody_mesh | SAM3DBODY_MESH | Per-frame mesh from SAM3DBody |
| frame_index | INT | Current frame index |
| mesh_sequence | MESH_SEQUENCE | Existing sequence (optional) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | Accumulated sequence |
| frame_count | INT | Number of frames |

---

### SAM3DBody2abc Video Batch Processor

Processes video through SAM3DBody frame-by-frame with optional external intrinsics.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| model | SAM3D_MODEL | From Load SAM3DBody Model node |
| images | IMAGE | Video frames |
| masks | MASK | Per-frame masks (optional) |
| fps | FLOAT | Source video FPS |
| external_intrinsics | CAMERA_INTRINSICS | From MoGe2 Intrinsics (optional) |
| intrinsics_json | INTRINSICS | From JSON/IntrinsicsEstimator (optional) |

**Intrinsics Priority:**
1. `external_intrinsics` (CAMERA_INTRINSICS) - MoGe2 or external source
2. `intrinsics_json` (INTRINSICS) - From JSON file or estimator
3. SAM3DBody internal estimation - Default fallback

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | Complete sequence with intrinsics |
| debug_images | IMAGE | Processed frames |
| frame_count | INT | Number of frames |
| status | STRING | Processing summary |
| fps | FLOAT | Output FPS |
| focal_length_px | FLOAT | Average focal length used |

**Stored in mesh_sequence:**
- `focal_length_px`: Active focal length (external or SAM3DBody)
- `focal_length_sam3d`: Original SAM3DBody estimation
- `intrinsics_source`: "SAM3DBody", "MoGe2", "json", or "external"
- `external_intrinsics`: Full normalized intrinsics if provided

---

## Analysis Nodes

### Motion Analyzer

Analyzes subject motion and generates trajectory data.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | From Accumulator |
| images | IMAGE | Original video (optional, for overlay) |
| camera_extrinsics | CAMERA_EXTRINSICS | For camera-compensated trajectory (optional) |
| skeleton_mode | STRING | "Simple Skeleton" (18-joint) or "Full Skeleton" (127-joint) |
| subject_height_m | FLOAT | Subject height in meters (0 = auto) |
| reference_frame | INT | Frame for height estimation |
| depth_source | STRING | "Auto", "SAM3DBody Only", or "Tracked Depth Only" |
| reference_joint_idx | INT | Joint to highlight (-1 = default pelvis) |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| subject_motion | SUBJECT_MOTION | Motion analysis data |
| scale_info | SCALE_INFO | Height/scale information |
| debug_overlay | IMAGE | Video with joint markers |
| trajectory_topview | IMAGE | Top-down trajectory view |
| debug_info | STRING | Analysis summary |

**Custom Fields Stored:**
- `depth_source`: Which depth method was used
- `skeleton_mode`: Simple or Full
- `scale_factor`: Computed scale

---

### Trajectory Smoother

Reduces noise/jitter in trajectory data.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| subject_motion | SUBJECT_MOTION | From Motion Analyzer |
| method | STRING | Smoothing algorithm |
| strength | FLOAT | 0.0-1.0 smoothing intensity |
| mesh_sequence | MESH_SEQUENCE | For Joint-Guided method (optional) |
| skeleton_format | STRING | "SMPL-H" or "COCO" |
| reference_joint_smplh | STRING | Reference joint for SMPL-H |
| reference_joint_coco | STRING | Reference joint for COCO |

**Smoothing Methods:**
| Method | Description |
|--------|-------------|
| Savitzky-Golay (Best) | Preserves shape, best for motion capture |
| Gaussian | General-purpose smoothing |
| Moving Average | Simple but effective |
| Spline | Fits smooth curve |
| Kalman | Physics-based (constant velocity model) |
| Joint-Guided (NEW) | Uses stable 2D joint detection to guide smoothing |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| subject_motion | SUBJECT_MOTION | Smoothed motion data |
| comparison_image | IMAGE | Before/after visualization |
| stats | STRING | Jitter reduction statistics |
| reference_joint_index | INT | Selected joint index |

**Custom Fields Stored:**
- `smoothing_applied.method`: Which method was used
- `smoothing_applied.strength`: Smoothing strength
- `smoothing_applied.jitter_reduction_pct`: Percentage reduction
- `smoothing_applied.reference_joint_name`: Joint used for Joint-Guided

---

### Character Trajectory Tracker

Tracks character position using TAPIR point tracking.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| images | IMAGE | Video frames |
| mesh_sequence | MESH_SEQUENCE | From Accumulator |
| tracking_mode | STRING | "Pelvis", "Average Body", etc. |
| smoothing_method | STRING | Smoothing algorithm |
| smoothing_window | INT | Window size |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | With tracked_depth added |
| trajectory_3d | LIST | 3D positions |

**Custom Fields Stored (via subject_motion):**
- `character_trajectory_settings.tracking_mode`
- `character_trajectory_settings.smoothing_method`
- `character_trajectory_settings.smoothing_window`

---

## Export Nodes

### Export Animated FBX

Exports MESH_SEQUENCE to animated FBX file.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | From Accumulator |
| filename | STRING | Output filename |
| camera_extrinsics | CAMERA_EXTRINSICS | Camera motion (optional) |
| fps | FLOAT | Export framerate (0 = use source) |
| skeleton_mode | STRING | "Rotations" or "Positions" |
| world_translation | STRING | Body positioning mode |
| camera_motion | STRING | Camera export mode |
| extrinsics_smoothing | STRING | Smoothing for camera |
| use_depth_positioning | BOOLEAN | Enable depth-based Z |
| depth_mode | STRING | "Position" or "Scale" |
| subject_motion | SUBJECT_MOTION | Motion analysis (optional) |
| scale_info | SCALE_INFO | Scale data (optional) |

**World Translation Options:**
| Option | Description |
|--------|-------------|
| None (Body at Origin) | Body stays at 0,0,0 |
| Body World (Raw) | Uses pred_cam_t directly |
| Body World (Compensated) | Camera effects removed |
| Depth Only (Z-axis) | Only Z from depth |

**Custom Fields in FBX:**
| Field | Source |
|-------|--------|
| sam3dbody2abc_version | Package version |
| export_timestamp | Export time (IST) |
| world_translation | Position mode |
| camera_motion | Camera mode |
| extrinsics_smoothing | Smoothing method |
| extrinsics_smoothing_strength | Strength value |
| depth_positioning_enabled | True/False |
| depth_mode | Position/Scale |
| skeleton_export_mode | Rotations/Positions |
| trajectory_smoothing_method | From Trajectory Smoother |
| trajectory_jitter_reduction_pct | Reduction achieved |
| depth_source | From Motion Analyzer |
| camera_solving_method | From Camera Solver |
| camera_translational_solver | From Camera Solver |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| output_path | STRING | Path to FBX file |
| status | STRING | Export summary |
| frame_count | INT | Frames exported |
| fps | FLOAT | Export framerate |

---

## Camera Nodes

### Camera Solver V2

Solves camera rotation from video features.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| images | IMAGE | Video frames |
| foreground_masks | MASK | Person mask (optional) |
| solving_method | STRING | Algorithm selection |
| translational_solver | STRING | For translation |
| coordinate_system | STRING | Maya/Blender/OpenCV |

**Solving Methods:**
| Method | Description |
|--------|-------------|
| Essential + Bundle | Most accurate |
| Homography | Fast, rotation-only |
| Feature Flow | Simple averaging |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| camera_extrinsics | CAMERA_EXTRINSICS | 6-DOF camera motion |
| status | STRING | Solving summary |

**Custom Fields Stored:**
- `solving_method`: Which method was used
- `translational_solver`: Translation method
- `coordinate_system`: Output coordinate system

---

### Intrinsics from SAM3DBody

Estimates camera intrinsics from SAM3DBody focal length.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | From Accumulator |
| sensor_width_mm | FLOAT | Camera sensor width |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| camera_intrinsics | CAMERA_INTRINSICS | Focal length, principal point |

---

## Multi-Camera Nodes

### Multi-Camera Auto Calibrator

Automatically calibrates two cameras from synchronized video.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| subject_motion_a | SUBJECT_MOTION | Camera A motion |
| subject_motion_b | SUBJECT_MOTION | Camera B motion |
| focal_length_mm | FLOAT | Camera focal length |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| calibration | CALIBRATION | Camera calibration data |
| visualization | IMAGE | Debug visualization |
| status | STRING | Calibration summary |

---

### Multi-Camera Triangulator

Triangulates 3D positions from multiple camera views.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| subject_motion_a | SUBJECT_MOTION | Camera A |
| subject_motion_b | SUBJECT_MOTION | Camera B |
| calibration | CALIBRATION | From Auto Calibrator |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| triangulated_motion | TRIANGULATED_MOTION | True 3D positions |
| topview | IMAGE | Top-down visualization |

---

## Skeleton Formats

### SMPL-H (127 joints) - joint_coords

```
0: Pelvis      1: L_Hip       2: R_Hip       3: Spine1
4: L_Knee      5: R_Knee      6: Spine2      7: L_Ankle
8: R_Ankle     9: Spine3     10: L_Foot     11: R_Foot
12: Neck      13: L_Collar   14: R_Collar   15: Head
16: L_Shoulder 17: R_Shoulder 18: L_Elbow   19: R_Elbow
20: L_Wrist   21: R_Wrist    22-126: Hands/Face
```

### COCO (17 joints) - keypoints_2d/3d

```
0: Nose        1: L_Eye       2: R_Eye       3: L_Ear
4: R_Ear       5: L_Shoulder  6: R_Shoulder  7: L_Elbow
8: R_Elbow     9: L_Wrist    10: R_Wrist    11: L_Hip
12: R_Hip     13: L_Knee     14: R_Knee     15: L_Ankle
16: R_Ankle
```

---

## Typical Workflow

```
Video → SAM3DBody → Mesh Accumulator → Motion Analyzer
                                            ↓
                    Character Trajectory (optional)
                                            ↓
                    Camera Solver V2 (optional)
                                            ↓
                    Trajectory Smoother (optional)
                                            ↓
                    Export Animated FBX → .fbx file
```

---

## Changelog

### v5.4.0
- **NEW**: Direct Meta SAM-3D-Body integration (no third-party wrappers)
- **NEW**: Automatic model download with HuggingFace token
- **NEW**: External intrinsics support in Video Batch Processor
  - `external_intrinsics` input (CAMERA_INTRINSICS type)
  - `intrinsics_json` input (INTRINSICS type)
  - Priority: external > json > SAM3DBody
- **NEW**: Format-agnostic intrinsics normalization
- **FIXED**: Double transformation issue with root_locator
- **FIXED**: pred_cam_t offset calculation
- **REMOVED**: align_mesh_to_skeleton option
- **REMOVED**: Third-party SAM3DBody wrapper dependencies

### v4.8.8
- Added Joint-Guided smoothing method to Trajectory Smoother
- Added reference joint selection (SMPL-H/COCO)
- Added reference_joint_idx input to Motion Analyzer
- Added depth_source to subject_motion for FBX metadata
- Extended FBX custom fields with export settings
- Fixed trajectory top-view X-axis orientation

### v4.8.7
- Fixed trajectory top-view horizontal flip
- Fixed viewing angle calculation (170° → 30-35°)
- Fixed version display in FBX custom fields

---

## External Intrinsics

The Video Batch Processor supports external camera intrinsics to override SAM3DBody's internal estimation.

### Use Cases

- **Pre-calibrated cameras**: Use known lens parameters from camera calibration
- **Professional footage**: Known focal length from camera metadata
- **Zoom lenses**: Per-frame variable focal length
- **Multi-camera setups**: Consistent intrinsics across cameras
- **Better accuracy**: MoGe2 often provides more accurate intrinsics than SAM3DBody

### Supported Input Formats

The `normalize_intrinsics()` function accepts multiple formats and converts them internally:

#### Format 1: Simple Dict (Single Focal Length)
```json
{
  "focal_length": 1108.5,
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720
}
```

#### Format 2: INTRINSICS Type (from IntrinsicsFromJSON)
```json
{
  "focal_px": 1108.5,
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720,
  "focal_mm": 35.0,
  "sensor_width_mm": 36.0
}
```

#### Format 3: Per-Frame (Zoom Lenses)
```json
{
  "per_frame": true,
  "frames": [
    {"focal_length": 1100.0},
    {"focal_length": 1105.0},
    {"focal_length": 1110.0}
  ],
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720
}
```

#### Format 4: MoGe2 Format (CAMERA_INTRINSICS)
```json
{
  "focal_length": 1108.5,
  "per_frame_focal": [1100.0, 1105.0, 1110.0],
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720,
  "sensor_width_mm": 36.0
}
```

#### Format 5: Calibration Format (OpenCV-style)
```json
{
  "fx": 1108.5,
  "fy": 1108.5,
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720
}
```

### Normalized Output Format

All input formats are converted to:
```json
{
  "focal_length": 1108.5,
  "per_frame_focal": [1108.5, 1108.5, ...],
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720,
  "source": "external"
}
```

### Key Name Aliases

The normalizer recognizes these equivalent key names:

| Canonical | Aliases |
|-----------|---------|
| `focal_length` | `focal_length_px`, `focal_px`, `focal`, `fx` |
| `cx` | `principal_point_x`, `principal_x` |
| `cy` | `principal_point_y`, `principal_y` |
| `width` | `image_width` |
| `height` | `image_height` |

### Verifying Intrinsics

Use the **Verify Overlay** node to compare intrinsics:

1. Connect both `mesh_sequence` and `camera_intrinsics` to Verify Overlay
2. Set `intrinsics_source` to **"Compare Both"**
3. Enable `show_mesh` to see wireframe comparison

The overlay shows:
- **Green wireframe**: SAM3DBody intrinsics projection
- **Orange wireframe**: External/MoGe2 intrinsics projection
- **Text overlay**: Focal length values and difference

---

*Generated for SAM3DBody2abc v5.4.0*
