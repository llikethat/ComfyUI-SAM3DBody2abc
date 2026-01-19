# ComfyUI-SAM3DBody2abc

**Video to Animated FBX Export for Maya/Blender**

Convert video of a person into animated 3D body mesh with skeleton, camera, and trajectory data exported as FBX.

## Features

- **3D Body Reconstruction**: Per-frame mesh and skeleton from SAM3DBody
- **Depth Tracking**: TAPIR + Depth Anything V2 for accurate Z-axis movement
- **Camera Solving**: Automatic pan/tilt detection from background motion
- **Motion Analysis**: Speed, direction, foot contact detection
- **FBX Export**: Maya-compatible with proper axis conversion and scale

## Workflow

```
Load Video → SAM3 Segmentation → Video Batch Processor
                                        ↓
                              Character Trajectory Tracker
                                        ↓
                              Camera Solver → Motion Analyzer
                                        ↓
                              Export Animated FBX
```

A complete workflow is included: `workflows/SAM3DBody2abc_Video_to_FBX.json`

## Nodes

| Node | Purpose |
|------|---------|
| 🎬 **Video Batch Processor** | Core pose estimation from SAM3DBody |
| 🏃 **Character Trajectory Tracker** | TAPIR + Depth Anything V2 depth tracking |
| 📷 **Camera Solver** | Pan/tilt detection from background |
| 📷 **Camera Extrinsics from JSON** | Import external camera data |
| 📊 **Motion Analyzer** | Speed, direction, foot contact analysis |
| 📏 **Scale Info Display** | Display scale information |
| 🔍 **Verify Overlay** | Debug visualization (single frame) |
| 🔍 **Verify Overlay (Sequence)** | Debug visualization (batch) |
| 📦 **Export Animated FBX** | Export to Maya/Blender FBX |
| 📦 **Export FBX from JSON** | Export from saved JSON data |
| 🎥 **FBX Animation Viewer** | Preview animation in ComfyUI |
| 🔄 **Temporal Smoothing** | Reduce trajectory jitter (single camera) |
| 🎯 **Camera Auto-Calibrator** | Auto-compute camera positions from person |
| 🔺 **Multi-Camera Triangulator** | Triangulate 3D from 2 camera views |

## Installation

### Required ComfyUI Custom Nodes

Install these custom nodes via ComfyUI Manager or manually:

| Custom Node | Purpose | Repository |
|-------------|---------|------------|
| **ComfyUI-SAM3DBody** | Core 3D body estimation | [PozzettiAndrea/ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) |
| **ComfyUI-SAM3** | Video segmentation (person masks) | [PozzettiAndrea/ComfyUI-SAM3](https://github.com/PozzettiAndrea/ComfyUI-SAM3) |
| **ComfyUI-VideoHelperSuite** | Video loading/saving | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |
| **ComfyUI-DepthAnythingV2** | Depth estimation | [kijai/ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2) |
| **ComfyUI-Custom-Scripts** | Utility nodes | [pythongosssss/ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) |
| **ComfyUI-Crystools** | Utility nodes | [crystian/ComfyUI-Crystools](https://github.com/crystian/ComfyUI-Crystools) |

### Python Requirements

```bash
pip install torch torchvision
pip install opencv-python numpy scipy
pip install einshape  # For TAPIR
pip install kornia    # For LoFTR feature matching
```

### Optional Dependencies (for better camera solving)

```bash
# LightGlue - Recommended for fast, accurate feature matching
git clone https://github.com/cvg/LightGlue.git && cd LightGlue && pip install -e .

# LoFTR - Alternative detector-free matching (included in kornia)
pip install kornia
```

### Model Downloads

Models are downloaded automatically on first use:

| Model | Size | Purpose |
|-------|------|---------|
| **TAPIR** (BootsTAPIR) | ~150MB | Point tracking for trajectory |
| **Depth Anything V2** | ~350MB | Depth estimation |
| **LightGlue + SuperPoint** | ~75MB | Feature matching (camera solve) |
| **LoFTR** | ~50MB | Detector-free matching (camera solve fallback) |

## Logging / Verbosity Control

Control console output verbosity via the `log_level` parameter in Export FBX node:

| Level | Output |
|-------|--------|
| `Silent` | No output |
| `Errors Only` | Errors only |
| `Warnings` | Errors + warnings |
| `Normal (Info)` | Key status messages (default) |
| `Verbose (Status)` | Info + progress updates |
| `Debug (All)` | Everything including diagnostics |

All log messages include timestamps: `[HH:MM:SS.mmm] [Module] message`

Environment variable `SAM3DBODY_LOG_LEVEL` can also be used as fallback.

## Export Options

### World Translation Modes
| Mode | Description |
|------|-------------|
| **None** | Body at origin |
| **Root Locator** | Animated locator drives position |
| **Baked** | Position baked into mesh vertices |

### Skeleton Modes
| Mode | Description |
|------|-------------|
| **Positions** | Joint locations animated |
| **Rotations** | Joint rotations for FK (recommended) |

### Depth Modes
| Mode | Description |
|------|-------------|
| **Scale** | Character scales with depth (recommended) |
| **Position** | Character moves in Z-axis |
| **Both** | Combined scale + Z movement |

### Camera Options
| Option | Description |
|--------|-------------|
| **Static** | Fixed camera, character moves |
| **Translation** | Camera translates to follow |
| **Rotation** | Camera rotates (pan/tilt) to follow |

## FBX Output Structure

```
FBX File:
├── Camera (with focal length, optional animation)
├── SAM3DBody_root_locator (animated position)
├── SAM3DBody_Trajectory (world position locator)
├── SAM3DBody_ScreenPosition (screen-space position)
├── SAM3DBody_WorldPosition (compensated trajectory)
├── SAM3DBody_CameraExtrinsics (camera rotation data)
├── Armature (skeleton with 127 joints)
│   └── Bones with animation
├── body (mesh)
│   └── Shape keys (per-frame deformation)
└── SAM3DBody_Metadata (custom attributes)
```

## GPU Requirements

- **Minimum**: 8GB VRAM (RTX 3070 or equivalent)
- **Recommended**: 12GB+ VRAM for higher resolution
- Falls back to CPU for some operations if GPU unavailable

## FBX Export Methods

The FBX export uses one of two methods:

### Method 1: Direct bpy Export (Recommended if Python 3.11)
- No external Blender installation needed
- Faster (no subprocess overhead)
- **Requires Python 3.11.x exactly** + bpy module

```bash
# Check your Python version first
python --version

# For Python 3.11.x:
pip install bpy

# For Python 3.10.x:
pip install bpy==4.0.0
```

**Note:** bpy has strict Python version requirements. It will NOT install on Python 3.9 or 3.12.

### Method 2: Blender Subprocess (Works with any Python)
- Used automatically when bpy module not available
- Requires Blender 4.2+ installed on system

```bash
# Install Blender on Linux/RunPod:
cd /workspace
wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
tar -xf blender-4.2.0-linux-x64.tar.xz
ln -sf /workspace/blender-4.2.0-linux-x64/blender /usr/local/bin/blender
```

The export will automatically detect which method is available and use the best option.

## Multi-Camera Triangulation

For jitter-free depth, use synchronized footage from 2 cameras:

### Workflow
```
Video A → SAM3DBody → mesh_sequence_a ─┬─→ 🎯 Auto-Calibrator → calibration ─┐
                                       │                                      │
Video B → SAM3DBody → mesh_sequence_b ─┼──────────────────────────────────────┼→ 🔺 Triangulator → mesh_sequence
                                       └──────────────────────────────────────┘
```

### Requirements
- Two synchronized videos of the same person
- Person visible in both cameras simultaneously
- Cameras at 60-120° angle for best accuracy (minimum 30°)

### What It Solves
Single-camera depth estimation is noisy because it guesses depth from body size. Multi-camera triangulation **calculates** depth geometrically by finding where viewing rays intersect - eliminating jitter.

### Maya Import
- Use **Video A** as the image plane (Camera A is the reference at origin)
- The exported mesh/skeleton aligns with Video A

## Licenses

### Main Package
MIT License

### Third-Party Components

| Component | License | Usage |
|-----------|---------|-------|
| **TAPIR** | Apache 2.0 | Point tracking for depth/trajectory |
| **Depth Anything V2** | Apache 2.0 | Monocular depth estimation |
| **LightGlue** | Apache 2.0 | Feature matching for camera solve |
| **LoFTR** | Apache 2.0 | Feature matching fallback |
| **SAM3DBody** | See original repo | 3D body estimation |

#### TAPIR License
```
Copyright 2023 Google DeepMind
Licensed under the Apache License, Version 2.0
https://github.com/google-deepmind/tapnet
```

#### Depth Anything V2 License
```
Copyright 2024 The University of Hong Kong
Licensed under the Apache License, Version 2.0
https://github.com/DepthAnything/Depth-Anything-V2
```

## Version History

### v4.8.6 (January 2025)
- **Multi-Camera System**: Jitter-free depth through geometric triangulation
  - 🎯 **Camera Auto-Calibrator**: Automatically computes camera positions from person keypoints
    - No manual measurement required
    - Uses person visible in both cameras to solve relative pose
    - Scales using known person height
    - Reports viewing angles, distance to person, camera geometry
  - 🔺 **Multi-Camera Triangulator**: Triangulates 3D positions from 2 camera views
    - Geometric intersection of viewing rays
    - Produces jitter-free depth
    - Outputs same MESH_SEQUENCE format for seamless pipeline integration
- **Viewing angle computation**: Distance, azimuth, elevation from each camera to person

### v4.8.1 (January 2025)
- 🔄 **Temporal Smoothing** node for single-camera jitter reduction
  - Smooths pred_cam_t (tx, ty, tz) and tracked_depth
  - Does NOT smooth joint_coords (pose data is accurate)
  - Methods: Gaussian, EMA (bidirectional), Savitzky-Golay
  - Optional vertex smoothing for shape keys
  - Reports jitter reduction percentage

### v4.7.8 (January 2025)
- Added `trajectory_topview` output to Motion Analyzer node:
  - Top-down view showing character path (X-Z plane)
  - Camera position indicator at top
  - Color-coded by depth (blue=close, red=far)
  - Start/End markers with direction arrow
  - Grid with scale reference (meters)
- Helps visualize circular/complex paths for debugging before Maya export

### v4.7.7 (January 2025)
- Added `depth_source` toggle to Motion Analyzer node:
  - **Auto (Tracked if available)** - Uses per-frame tracked_depth from Character Trajectory Tracker (recommended for circular/complex paths)
  - **SAM3DBody Only (pred_cam_t)** - Original behavior, uses SAM3DBody's depth estimate
  - **Tracked Depth Only** - Forces tracked_depth usage
- Fixes circular walk trajectories appearing as straight lines in Maya
- Per-frame depth now properly flows through the entire pipeline to FBX export

### v4.7.6 (January 2025)
- Added camera custom properties exported as Maya Extra Attributes:
  - `sensor_width_mm`, `sensor_height_mm`
  - `focal_length_mm`, `focal_length_px`
  - `image_width`, `image_height`, `aspect_ratio`
- These allow proper camera setup in Maya after FBX import

### v4.7.5 (January 2025)
- Fixed all fallback loggers to include `progress()` method
- Prevents AttributeError when logger import fails

### v4.7.4 (January 2025)
- Fixed double module names in logs (e.g., `[Motion Analyzer] [Motion Analyzer]`)
- Added progress logging to Verify Overlay (Sequence) node
- Added configurable `batch_size` parameter to Video Batch Processor (default: 10)
- Added "Static (No Motion)" option to Camera Solver - skips all feature detection for static shots

### v4.7.3 (January 2025)
- Added traceback to module loading errors for better debugging
- Removed all leading spaces from log messages
- Fixed potential import issues in camera_solver

### v4.7.2 (January 2025)
- Fixed circular reference in fallback loggers (camera_solver, motion_analyzer)
- All fallback loggers now properly use print() instead of undefined log reference

### v4.7.1 (January 2025)
- Updated README with complete ComfyUI custom node dependencies
- Added LoFTR and kornia to requirements documentation
- Fixed verify_overlay syntax errors from debug removal

### v4.7.0 (January 2025)
- **Code optimization**: Centralized logging system with verbosity levels
- **Node parameter**: `log_level` in Export FBX node controls verbosity
- **Timestamps**: All log messages include `[HH:MM:SS.mmm]` timestamps
- Removed 500+ debug print statements  
- Cleaner console output with progress indicators
- Removed unused functions and dead code (~10% code reduction)

### v4.6.10 (January 2025)
- Removed unused nodes (COLMAP Bridge, MoGe2 Intrinsics, Accumulator, Logger)
- Added `include_skeleton` option for camera-only export
- Added working workflow: `SAM3DBody2abc_Video_to_FBX.json`
- Code cleanup and package optimization

### v4.6.9 (January 2025)
- Fixed depth (tz) handling - properly uses tracked_depth from CharacterTrajectoryTracker
- Added depth_mode option: Scale (default), Position, Both, Legacy
- Character Trajectory Tracker now updates mesh_sequence with depth data

### v4.6.8 (January 2025)
- ZXY rotation order for camera objects (Maya compatibility)
- Fixed FBX scale settings (0.01 scale, FBX_SCALE_ALL)
- Proper Blender → Maya axis conversion

### v4.6.7 (January 2025)
- Per-frame body offset (fixes drift issue)
- Added ScreenPosition, WorldPosition, CameraExtrinsics locators

### v4.6.6 (January 2025)
- Fixed FBX export settings for Maya

### v4.6.5 (January 2025)
- Fixed Blender output capture (shows all DEBUG lines)

### v4.6.4 (January 2025)
- Added camera rotation debug output

### v4.6.3 (December 2024)
- Per-frame focal animation in FBX camera
- Trajectory interpretation notes in metadata

### v4.6.2 (December 2024)
- Focal length in mm, sensor_width_mm input

### v4.6.1 (December 2024)
- Camera compensation (focal + extrinsics)
- Raw & compensated trajectory output

### v4.6.0 (December 2024)
- Animated trajectory locator (SAM3DBody_Trajectory)

### v4.5.x (December 2024)
- Metadata improvements
- Maya custom properties support
- Version detection fixes

### v4.1.0 (December 2024)
- Bake Camera into Geometry mode
- Smoothing options (Kalman, Spline, Gaussian)

### v4.0.0 (December 2024)
- Complete camera solver rewrite
- Auto shot type detection
- Quality modes (Fast/Balanced/Best)
- LightGlue + LoFTR integration

### v3.x (November 2024)
- Initial release
- Basic camera rotation solver
- JSON import support

## Troubleshooting

### Character drifts in Maya
- Use **Static** camera mode for best results
- Verify overlay video matches before export
- Check depth tracking visualization

### Large FBX file size
- Shape keys (blend shapes) store per-frame vertex data
- ~10-15MB for 50 frames is normal
- Skeleton-only export (`include_mesh = OFF`) is smaller
- Future: Proper skeleton with skinning will enable smaller files

### Mesh not deforming
- Mesh deformation uses blend shapes (shape keys)
- Check if `include_mesh` is enabled
- Verify frame range in Maya matches export

### Maya import issues
- Ensure FBX 2020 or later
- Check Y-up axis setting
- Timeline should match frame count

## Credits

- [SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) by Andrea Pozzetti
- [TAPIR](https://github.com/google-deepmind/tapnet) by Google DeepMind
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) by University of Hong Kong
