# SAM3DBody2abc

**Extension for ComfyUI-SAM3DBody that adds video batch processing and animated export to Alembic (.abc) and FBX formats.**

![Version](https://img.shields.io/badge/version-2.0.0-blue)
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
git clone https://github.com/your-username/SAM3DBody2abc.git
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
- SMPL joint hierarchy (24 joints)
- Can be retargeted to other characters in Blender/Maya
- Requires Blender for export

### Export Methods
1. **Native Alembic** (fastest) - Requires PyAlembic
2. **Blender subprocess** - Uses Blender as export backend
3. **OBJ sequence fallback** - Always available

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

### v2.0.0
- Complete rewrite for better integration with existing SAM3DBody node
- Added animated Alembic export (full timeline)
- Added animated FBX skeleton export (full timeline)
- Added mesh overlay visualization
- Added temporal smoothing
- Added mesh sequence accumulator for per-frame workflows
