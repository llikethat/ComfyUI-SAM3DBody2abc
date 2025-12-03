# SAM3DBody2abc

**Extension for ComfyUI-SAM3DBody that adds video batch processing and animated export to Alembic (.abc) and FBX formats.**

![Version](https://img.shields.io/badge/version-2.0.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Purpose

The existing [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) node exports **per-frame** STL/OBJ/PLY meshes and FBX skeletons. This extension adds:

- **Video batch processing** - Process entire videos through SAM3DBody automatically
- **Animated Alembic export** - Single .abc file with full animation timeline
- **Animated FBX skeleton export** - Single .fbx file with animated skeleton
- **Mesh overlay visualization** - Render 3D mesh/skeleton overlay on video frames

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎬 **Batch Processing** | Process video/image sequences through SAM3DBody in one go |
| 📦 **Animated Alembic** | Export animated geometry to single .abc file (not per-frame!) |
| 🦴 **Animated FBX** | Export animated skeleton to single .fbx file |
| 〰️ **Temporal Smoothing** | Reduce jitter between frames |
| 🎨 **Overlay Rendering** | Visualize mesh/skeleton on images |
| 📐 **FOV Control** | Manual FOV setting or auto-calibration with GeoCalib |
| 🔄 **VHS Compatible** | Works with Load Video (Upload) from VideoHelperSuite |

## 📋 Requirements

### Required
- [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) - SAM 3D Body model integration

### Optional
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) - For video input
- **Blender** - Required for FBX export (installed separately)
- **PyAlembic** - For native Alembic export (falls back to Blender if not available)

## 🔧 Installation

### Via ComfyUI Manager
1. Open ComfyUI Manager
2. Search for "SAM3DBody2abc"
3. Click Install
4. Restart ComfyUI

### Manual Installation
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/llikethat/ComfyUI-SAM3DBody2abc.git
cd SAM3DBody2abc
python install.py
```

## 🎬 Nodes Overview

### Video/Batch Processing

| Node | Description |
|------|-------------|
| **🎬 SAM3DBody Batch Processor** | Process video frames through SAM3DBody. Accepts images from VHS Load Video. |
| **📹 Process Image Sequence → SAM3DBody** | Process image sequence with optional temporal smoothing |

### Animated Export

| Node | Description |
|------|-------------|
| **📦 Export Animated Alembic (.abc)** | Export mesh sequence to single animated Alembic file |
| **🦴 Export Animated Skeleton FBX** | Export joint animation to single FBX file |
| **💾 Export Animated Mesh (All Formats)** | Combined export to multiple formats |

### Mesh Sequence Management

| Node | Description |
|------|-------------|
| **📋 Mesh Sequence Accumulator** | Collect meshes from per-frame SAM3DBody calls |
| **🔄 Convert SAM3DBody Mesh → Sequence** | Convert single mesh to sequence format |
| **👁️ Preview Mesh Sequence** | View sequence statistics |
| **〰️ Smooth Mesh Sequence** | Apply temporal smoothing to reduce jitter |
| **🗑️ Clear Mesh Sequence** | Clear accumulated data |

### Visualization

| Node | Description |
|------|-------------|
| **🎨 Render Mesh Overlay** | Render mesh wireframe/joints on single image |
| **🎨 Render Mesh Overlay (Batch)** | Render overlay on entire video |

## 📊 Workflows

### Basic: Video to Animated Alembic + FBX

```
┌──────────────────┐     ┌─────────────────────┐     ┌────────────────────────┐
│ Load Video (VHS) │────▶│ Load SAM 3D Body    │────▶│ SAM3DBody Batch        │
│                  │     │ Model               │     │ Processor 🎬           │
└──────────────────┘     └─────────────────────┘     └────────────────────────┘
                                                              │
                                    ┌─────────────────────────┼─────────────────────────┐
                                    ▼                         ▼                         ▼
                         ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
                         │ Export Animated  │      │ Export Animated  │      │ Render Mesh      │
                         │ Alembic 📦       │      │ FBX 🦴           │      │ Overlay Batch 🎨 │
                         └──────────────────┘      └──────────────────┘      └──────────────────┘
                                │                          │                          │
                                ▼                          ▼                          ▼
                          body.abc                   skeleton.fbx                overlay.mp4
```

### Per-Frame Processing with Accumulator

If you need more control and want to use the standard SAM3DBody "Process Image" node:

```
┌──────────────────┐     ┌─────────────────────┐     ┌────────────────────────┐
│ Load Image       │────▶│ Process Image       │────▶│ Mesh Sequence          │
│ (Loop)           │     │ (SAM3DBody)         │     │ Accumulator 📋         │
└──────────────────┘     └─────────────────────┘     └────────────────────────┘
        │                                                      │
        │ iterate over frames                                  │
        └──────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                                    ┌──────────────────┐
                                                    │ Export Animated  │
                                                    │ Alembic 📦       │
                                                    └──────────────────┘
```

## ⚙️ Export Details

### Alembic (.abc)
- Creates **single file** with animated vertex positions
- Compatible with: Blender, Maya, Houdini, Cinema 4D, Unreal Engine
- Optional joint positions as animated point cloud
- Configurable scale and axis orientation

### FBX Skeleton
- Creates **single file** with animated armature
- MHR (Momentum Human Rig) joint hierarchy (127 joints)
- Animated empties with reference armature using constraints
- Can be retargeted to other characters in Blender/Maya
- Requires Blender for export

### Export Methods
1. **Native Alembic** (fastest) - Requires PyAlembic
2. **Blender subprocess** - Uses Blender as export backend
3. **OBJ sequence fallback** - Always available

## 📐 FOV / Camera Calibration

The SAM3DBody model uses camera focal length for accurate 3D reconstruction. By default it uses a 55° FOV assumption, but you can improve accuracy by:

### Manual FOV Setting
Set the `fov` parameter in the Batch Processor based on your camera:

| Camera Type | Typical FOV |
|-------------|-------------|
| Smartphone (portrait) | 50-60° |
| Smartphone (wide) | 65-80° |
| Webcam | 55-70° |
| GoPro/Action cam | 90-120° |
| DSLR 50mm lens | 40-47° |
| DSLR 35mm lens | 55-65° |
| DSLR 24mm lens | 75-85° |

### Auto-Calibration with GeoCalib
Enable `auto_calibrate` to automatically estimate FOV using [GeoCalib](https://github.com/cvg/GeoCalib) (ECCV 2024):

```bash
# Install GeoCalib
pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
```

GeoCalib analyzes the first few frames to estimate:
- **Focal length** (→ FOV)
- **Gravity direction** (helps with orientation)
- **Lens distortion** (optional)

This provides more accurate 3D reconstruction and better overlay alignment.

## 🎛️ Tips

### Processing Speed
- Use `skip_frames` to process every Nth frame
- Lower video resolution before processing
- Use `temporal_smooth` to interpolate skipped frames

### Quality
- Set `det_thresh` lower (0.3-0.5) for difficult poses
- Use `full` detection mode for best mesh quality
- Enable temporal smoothing to reduce jitter

### Export Settings
- **Scale 1.0** = meters (Blender default)
- **Scale 100** = centimeters (Maya default)
- **Up Axis Y** = Blender, Maya
- **Up Axis Z** = Houdini, some game engines

## 🔍 Troubleshooting

### "No valid mesh data"
- Check SAM3DBody model is loaded correctly
- Try lower `det_thresh` value
- Ensure person is visible in frames

### "Blender not found" (FBX export)
- Install Blender: https://www.blender.org/download/
- Add Blender to system PATH
- Blender not loading - install the libraries using `apt install libsm6 libice6`

### Jittery animation
- Enable temporal smoothing
- Increase `smooth_window` size
- Process at higher resolution

## 📜 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Credits

- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) by Meta AI
- [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) by PozzettiAndrea
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) by Kosinkadink

## 📝 Changelog

### v2.0.8
- **GeoCalib Fix v2**: Fixed focal length extraction from GeoCalib Camera object (TensorWrapper)
- **Camera._data Access**: Now properly accesses underlying tensor data `camera._data[..., 2]` for fx
- **Better Debug Output**: Prints camera type and available attributes when extraction fails
- **Multiple Fallbacks**: Tries `_data`, tensor indexing, `.fx`, and `.f` properties

### v2.0.7
- **GeoCalib Fix**: Fixed "'list' object has no attribute 'shape'" error in auto-calibration
- **Robust Focal Extraction**: Now tries multiple methods to extract focal length from GeoCalib Camera object (camera.f, camera.fx, camera.K)
- **Better Debugging**: Added detailed logging when focal length extraction fails to help diagnose issues
- **Single Frame Calibration**: Simplified to use single frame instead of batch for more reliable results

### v2.0.6
- **FOV Parameter**: Added `fov` parameter to Batch Processor (default: 55°) for manual camera FOV setting
- **Auto-Calibration**: Added `auto_calibrate` option using GeoCalib (ECCV 2024) for automatic FOV estimation
- **Focal Length Override**: Custom FOV now properly overrides model's default focal length for better 3D accuracy
- **Documentation**: Added FOV/camera calibration guide with typical values for different cameras

### v2.0.5
- **Overlay Temporal Smoothing**: Added `temporal_smoothing` parameter to Overlay Batch node (0.0-1.0) to reduce frame-to-frame jitter using Gaussian filter
- **FBX Coordinate Fix**: Changed coordinate transform from `(X, -Z, Y)` to `(X, Z, -Y)` so person stands upright with positive Z in Blender

### v2.0.4
- **Overlay Projection Fix**: Fixed camera projection math - now correctly subtracts camera translation and flips Y for image coordinates
- **FBX Skeleton Simplified**: Removed empty parenting complexity - empties now animate with world positions directly for consistent animation

### v2.0.3
- **FBX Export Rewrite**: Changed from broken bone animation to animated empties approach
  - Each joint is an empty (null object) that follows animated position
  - Reference armature with bones constrained to follow empties
  - Proper FBX baking of animation
- **Overlay Debug Output**: Added detailed debug logging for first frame to diagnose projection issues
- **Improved Face Culling**: More permissive bounds checking for partially visible faces

### v2.0.2
- **Hardcoded MHR Joint Hierarchy**: Added anatomical fallback for 127-joint MHR skeleton when model extraction fails
- **Coordinate Transform Fix**: Fixed Blender coordinate conversion (x=x, y=-z, z=y)
- **Temporal Smoothing**: Added Gaussian kernel smoothing for exports

### v2.0.1
- **Overlay Renderer**: Switched to OpenCV-based rendering for better compatibility
- **Joint Hierarchy Extraction**: Added automatic extraction from SAM3DBody model

### v2.0.0
- Complete rewrite for better integration with existing SAM3DBody node
- Added animated Alembic export (full timeline)
- Added animated FBX skeleton export (full timeline)
- Added mesh overlay visualization
- Added temporal smoothing
- Added mesh sequence accumulator for per-frame workflows
