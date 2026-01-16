# FBX Custom Fields Reference

## Overview

SAM3DBody2abc embeds metadata as custom attributes in exported FBX files. These are accessible in Maya, Blender, and other 3D software as extra attributes on the `SAM3DBody_Metadata` locator.

---

## Build Information

| Field | Type | Description |
|-------|------|-------------|
| `sam3dbody2abc_version` | string | Package version (e.g., "4.8.8") |
| `export_timestamp` | string | Export time in IST |
| `filename` | string | Base filename used for export |
| `output_path` | string | Full path to exported file |

---

## Export Settings

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `world_translation` | string | Export Node | Body positioning mode |
| `camera_motion` | string | Export Node | Camera export mode |
| `export_fps` | float | Export Node | Framerate used |
| `frame_count` | int | Export Node | Total frames |
| `skeleton_export_mode` | string | Export Node | "Rotations" or "Positions" |
| `up_axis` | string | Export Node | "Y" or "Z" |
| `flip_x` | bool | Export Node | X-axis flip status |
| `include_mesh` | bool | Export Node | Mesh included |
| `include_skeleton` | bool | Export Node | Skeleton included |
| `include_camera` | bool | Export Node | Camera included |

---

## Depth & Positioning

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `depth_source` | string | Motion Analyzer | "Auto", "SAM3DBody Only", or "Tracked Depth Only" |
| `depth_positioning_enabled` | bool | Export Node | Depth positioning used |
| `depth_mode` | string | Export Node | "position" (recommended), "scale", "both", or "off" |
| `depth_min_m` | float | Motion Analyzer | Minimum depth in meters |
| `depth_max_m` | float | Motion Analyzer | Maximum depth in meters |

### Depth Mode Explanation

| Mode | What Happens | Best For |
|------|--------------|----------|
| **Position (Recommended)** | Character moves in Z axis | 3D scenes with lighting/shadows |
| Scale Only | Mesh scales (no Z movement) | 2D compositing over video |
| Both | Position + Scale | Special cases |
| Off | No depth handling | Legacy/flat look |

**Why Position is Recommended:** Scale mode places the mesh at Z=0 and scales it - this breaks lighting/shadow interactions in 3D scenes. Position mode moves the character to the correct Z depth, ensuring proper shadow casting and light interaction.

---

## Camera Smoothing

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `extrinsics_smoothing` | string | Export Node | "Kalman Filter", "Savitzky-Golay", etc. |
| `extrinsics_smoothing_strength` | float | Export Node | Smoothing intensity (0-1) |

---

## Trajectory Smoothing

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `trajectory_smoothing_method` | string | Trajectory Smoother | Method used |
| `trajectory_smoothing_strength` | float | Trajectory Smoother | Intensity (0-1) |
| `trajectory_jitter_reduction_pct` | float | Trajectory Smoother | Noise reduction percentage |
| `trajectory_reference_joint` | string | Trajectory Smoother | Joint used for Joint-Guided |
| `trajectory_skeleton_format` | string | Trajectory Smoother | "SMPL-H" or "COCO" |

---

## Character Trajectory

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `char_traj_tracking_mode` | string | Character Trajectory | "Pelvis", "Average Body", etc. |
| `char_traj_smoothing_method` | string | Character Trajectory | Smoothing algorithm |
| `char_traj_smoothing_window` | int | Character Trajectory | Window size |

---

## Camera Solving

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `camera_solving_method` | string | Camera Solver V2 | "Essential + Bundle", etc. |
| `camera_translational_solver` | string | Camera Solver V2 | Translation method |
| `camera_coordinate_system` | string | Camera Solver V2 | "maya", "blender", "opencv" |

---

## Scale & Height

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `subject_height_m` | float | Motion Analyzer | Subject height in meters |
| `scale_factor` | float | Motion Analyzer | Conversion factor |
| `mesh_height_units` | float | Motion Analyzer | Raw mesh height |
| `estimated_height_units` | float | Motion Analyzer | Estimated from skeleton |
| `leg_length_units` | float | Motion Analyzer | Leg segment length |
| `torso_head_units` | float | Motion Analyzer | Torso+head length |
| `height_source` | string | Motion Analyzer | "auto" or "user" |

---

## Motion Statistics (Raw)

| Field | Type | Description |
|-------|------|-------------|
| `total_distance_m_raw` | float | Total path distance |
| `avg_speed_ms_raw` | float | Average speed (m/s) |
| `avg_speed_kmh_raw` | float | Average speed (km/h) |
| `direction_angle_raw` | float | Movement direction (degrees) |
| `direction_desc_raw` | string | "Forward", "Left", etc. |
| `direction_vector_x_raw` | float | Normalized X direction |
| `direction_vector_y_raw` | float | Normalized Y direction |
| `direction_vector_z_raw` | float | Normalized Z direction |

---

## Motion Statistics (Compensated)

| Field | Type | Description |
|-------|------|-------------|
| `total_distance_m_comp` | float | Distance with camera effects removed |
| `avg_speed_ms_comp` | float | Speed with camera effects removed |
| `avg_speed_kmh_comp` | float | Speed in km/h (compensated) |
| `direction_angle_comp` | float | True direction (degrees) |
| `direction_desc_comp` | string | True direction description |
| `direction_vector_x_comp` | float | True X direction |
| `direction_vector_y_comp` | float | True Y direction |
| `direction_vector_z_comp` | float | True Z direction |

---

## Body World Displacement

| Field | Type | Description |
|-------|------|-------------|
| `body_world_disp_x` | float | Total X displacement |
| `body_world_disp_y` | float | Total Y displacement |
| `body_world_disp_z` | float | Total Z displacement |
| `body_world_total_distance` | float | Total 3D path length |
| `body_world_avg_speed_per_frame` | float | Per-frame speed |

---

## Foot Contact

| Field | Type | Description |
|-------|------|-------------|
| `grounded_frames` | int | Frames with feet on ground |
| `airborne_frames` | int | Frames with feet off ground |
| `grounded_percent` | float | Percentage grounded |

---

## Focal Length

| Field | Type | Description |
|-------|------|-------------|
| `focal_length_ref_px` | float | Reference focal length (pixels) |
| `focal_length_ref_mm` | float | Reference focal length (mm) |
| `focal_length_min_mm` | float | Minimum focal length |
| `focal_length_max_mm` | float | Maximum focal length |
| `focal_variation_percent` | float | Focal length variation |
| `sensor_width_mm` | float | Camera sensor width |
| `has_extrinsics_compensation` | bool | Camera compensation applied |

---

## Accessing in Maya (Python)

```python
import maya.cmds as cmds

# Get specific attribute
version = cmds.getAttr("SAM3DBody_Metadata.sam3dbody2abc_version")
print(f"Exported with v{version}")

# Get all custom attributes
attrs = cmds.listAttr("SAM3DBody_Metadata", userDefined=True)
for attr in attrs:
    value = cmds.getAttr(f"SAM3DBody_Metadata.{attr}")
    print(f"{attr}: {value}")
```

---

## Accessing in Blender (Python)

```python
import bpy

obj = bpy.data.objects.get("SAM3DBody_Metadata")
if obj:
    for key, value in obj.items():
        print(f"{key}: {value}")
```

---

*SAM3DBody2abc v4.8.8*
