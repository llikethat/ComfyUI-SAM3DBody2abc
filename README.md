# ComfyUI-SAM3DBody2abc

![Version](https://img.shields.io/badge/version-2.3.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Extension for [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) that adds:
- **Video batch processing** - Process entire videos at once
- **Animated Alembic export** - Export geometry animation as .abc files
- **Mesh overlay rendering** - Visualize 3D mesh on video frames

## 📦 Installation

1. Install [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) first
2. Clone this repository into `custom_nodes`:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI-SAM3DBody2abc
```
3. Install requirements:
```bash
pip install -r requirements.txt
```

## 🔧 Nodes

### Batch Processing
| Node | Description |
|------|-------------|
| **🎬 Batch Processor** | Process video frames with SAM3DBody |
| **📹 Sequence Process** | Process image sequences |

### Export
| Node | Description |
|------|-------------|
| **📦 Export Alembic** | Export animated mesh as .abc file |
| **📁 Export OBJ Sequence** | Export as numbered OBJ files |

### Overlay
| Node | Description |
|------|-------------|
| **🎨 Overlay** | Render mesh on single image |
| **🎨 Overlay Batch** | Render mesh on video frames |

### Mesh Management
| Node | Description |
|------|-------------|
| **📋 Accumulator** | Collect meshes into sequence |
| **〰️ Smooth** | Apply temporal smoothing |
| **🗑️ Clear** | Clear accumulated sequence |

## 📷 Camera Parameters

### Option 1: Auto-Calibrate (Recommended)
Set `auto_calibrate: True` to use GeoCalib for automatic FOV estimation.

### Option 2: Focal Length + Sensor Size
For DSLR/Cinema cameras:
- `focal_length_mm`: Your lens focal length (e.g., 50)
- `sensor_width_mm`: Your sensor width (see table below)

| Camera Type | Sensor Width |
|-------------|--------------|
| Full Frame | 36.0 mm |
| APS-C (Sony/Nikon) | 23.5 mm |
| APS-C (Canon) | 22.3 mm |
| Micro Four Thirds | 17.3 mm |
| Super 35 (Cinema) | 24.89 mm |

### Option 3: Direct FOV
Set `fov` in degrees (default: 55°).

## 🎨 Overlay Rendering

The overlay renderer supports two modes:

| Mode | Description |
|------|-------------|
| **solid** | Uses pyrender for accurate 3D rendering (requires pyrender) |
| **wireframe** | Fast OpenCV-based wireframe (fallback) |

### Requirements for Solid Rendering
```bash
pip install pyrender
# For headless servers:
pip install PyOpenGL==3.1.7
```

## 📝 Changelog

### v2.3.0 - Simplified & Fixed Overlay
- **REMOVED**: FBX export (was causing issues with Maya)
- **FIXED**: Overlay renderer now uses pyrender for accurate rendering
- **SIMPLIFIED**: Removed broken Meta renderer detection
- **NEW**: Solid and wireframe render modes

### v2.2.x - Previous versions
- Added DSLR camera support
- Various FBX export attempts (removed)

### v2.0.x - Initial release
- Batch video processing
- Alembic export
- Temporal smoothing

## 📋 Requirements

- ComfyUI-SAM3DBody
- numpy, torch, opencv-python
- trimesh (for mesh handling)
- pyrender (optional, for solid overlay)
- scipy (for temporal smoothing)

## 📄 License

MIT License - See LICENSE file
