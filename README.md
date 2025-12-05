# SAM3DBody2abc

**Extension for ComfyUI-SAM3DBody that adds video batch processing and animated export to Alembic (.abc) and FBX formats.**

![Version](https://img.shields.io/badge/version-2.2.3-blue)
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

### Option 1: Manual FOV Setting
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

### Option 2: Focal Length + Sensor Size (NEW in v2.1.0) - Recommended for DSLR/Cinema
If you know your lens focal length and camera sensor size:

1. Set `focal_length_mm` to your lens focal length (e.g., 50, 35, 85)
2. Set `sensor_width_mm` to your camera's sensor width:

| Camera/Sensor Type | Sensor Width (mm) |
|-------------------|-------------------|
| Full Frame (35mm) | 36.0 |
| APS-C Canon | 22.3 |
| APS-C Nikon/Sony/Fuji | 23.5 |
| Micro Four Thirds | 17.3 |
| Super 35 (Cinema) | 24.89 |
| RED Komodo | 27.03 |
| ARRI Alexa | 28.17 |
| 1" Sensor | 13.2 |
| iPhone 15 Pro Main | 9.8 |

**Example**: 50mm lens on Full Frame → `focal_length_mm=50`, `sensor_width_mm=36`

The node calculates: `focal_px = focal_mm × (image_width / sensor_width)`

### Option 3: Direct Pixel Focal Length
If you already have focal length in pixels from metadata or calibration:
- Set `focal_length_px` directly
- This overrides the mm calculation

### Option 4: Auto-Calibration with GeoCalib
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

### Priority Order
When multiple options are set:
1. `auto_calibrate` (highest - uses GeoCalib)
2. `focal_length_px` (direct pixel value)
3. `focal_length_mm` + `sensor_width_mm` (DSLR/Cinema)
4. `fov` (lowest - simple FOV angle)

## 🎬 FBX Export Target Application

The FBX export node now supports different target applications:

| Target | Axis | Description |
|--------|------|-------------|
| `maya` | Y-up | Optimized for Maya - proper joint hierarchy with rotation animation |
| `blender` | Z-up | Optimized for Blender - armature with bone transforms |
| `houdini` | Y-up | Optimized for Houdini - joint hierarchy similar to Maya |

### Maya Import Tips
1. Import FBX with "Animation" checked
2. The skeleton will have proper joint rotations
3. Use the skeleton for retargeting with HumanIK or similar

### Why Rotation Animation?
Previous versions used location-based animation which caused issues in Maya. The new export:
- Calculates bone rotations from joint position changes
- Properly handles parent-child relationships
- Results in clean, retarget-ready skeletons

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

### v2.2.3 - Improved Meta Renderer Detection
- **Better Import**: Now searches multiple paths to find Meta's `visualize_sample_together`
- **pyrender Check**: Explicitly checks if pyrender is installed before attempting import
- **Helpful Messages**: Shows what's missing and how to fix it:
  - If pyrender missing: "Install with: pip install pyrender"
  - If vis_utils not found: "Ensure ComfyUI-SAM3DBody is installed in custom_nodes"

### v2.2.2 - Fixed Maya FBX Export (Proper Joints)
- **FIXED**: FBX now exports as proper Maya joints instead of locators
  - Removed constraint-based approach that was creating empty objects
  - Uses direct bone animation with proper armature export
  - Only exports ARMATURE type, no empties
- **Improved Export**: Added debug output showing exported bone count

### v2.2.1 - Meta Official Renderer Support
- **NEW: Meta Renderer Option** - Added `use_meta_renderer` toggle to batch overlay
  - When enabled (default), uses Meta's official `visualize_sample_together` function
  - Produces more accurate overlay matching Meta's web demo
  - Falls back to OpenCV renderer if Meta's renderer is not available
- **Better Compatibility**: Improved rendering to match SAM3DBody's official output

### v2.2.0 - Multi-Application FBX Export & GPU Optimization
- **NEW: Target Application Selection** - FBX export now supports:
  - `maya`: Optimized for Maya (Y-up, proper joint hierarchy with rotations)
  - `blender`: Optimized for Blender (Z-up, armature)
  - `houdini`: Optimized for Houdini (Y-up, joints)
- **Improved FBX Export**: Proper rotation-based animation instead of just locations
  - Creates armature with bones pointing toward children
  - Animates rotations calculated from joint position changes
  - Better compatibility with retargeting tools
- **GPU Optimization Improvements**:
  - Added `batch_size` parameter (hint for future parallel processing)
  - Enabled CUDA cuDNN benchmark mode
  - Added periodic VRAM cache clearing (every 50 frames)
  - Wrapped inference in `torch.no_grad()` for memory efficiency
  - Console now shows CUDA device name and VRAM available

### v2.1.0 - DSLR/Cinema Camera Support & Improved FBX Export
- **NEW: Focal Length + Sensor Size Input** - For DSLR/Cinema cameras:
  - `focal_length_mm`: Lens focal length in mm (e.g., 50, 35, 85)
  - `sensor_width_mm`: Camera sensor width in mm (e.g., 36 for Full Frame, 23.5 for APS-C)
  - Auto-calculates pixel focal length: `focal_px = focal_mm × (image_width / sensor_width)`
- **NEW: Direct Pixel Focal Length** - `focal_length_px` for pre-calculated values
- **Priority Order**: auto_calibrate > focal_length_px > (focal_length_mm + sensor_width_mm) > fov
- **Improved FBX Export**: 
  - Uses hierarchical empties with proper parent-child relationships
  - Local transform animation (relative to parent)
  - Bone visualization as edge meshes
  - Better compatibility with retargeting tools

### v2.0.9
- **Wireframe Option**: Added 'wireframe' to mesh_color options for edge-only rendering
- **Camera Smoothing**: Now applies temporal smoothing to camera position in addition to vertices (reduces jitter)
- **NaN Fix**: Added handling for NaN values in opacity and line_thickness parameters
- **Updated Workflow**: Workflow now saves overlay video as MP4 at 24fps and displays focal length
- **Audio Passthrough**: Video save node now includes audio from source video

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
