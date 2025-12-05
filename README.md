# ComfyUI-SAM3DBody2abc

![Version](https://img.shields.io/badge/version-2.3.1-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Extension for [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) that adds:
- **Video batch processing** - Process entire videos at once
- **Animated Alembic export** - Export geometry animation as .abc files
- **Mesh overlay rendering** - Visualize 3D mesh on video frames
- **Temporal smoothing** - Reduce jitter in animations
- **DSLR/Cinema camera support** - Use real-world camera parameters

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
| **🎬 Batch Processor** | Process video frames with SAM3DBody model |
| **📹 Sequence Process** | Process image sequences frame by frame |

### Export
| Node | Description |
|------|-------------|
| **📦 Export Alembic** | Export animated mesh as .abc file (Maya, Houdini, Blender compatible) |
| **📁 Export OBJ Sequence** | Export as numbered OBJ files |

### Overlay Visualization
| Node | Description |
|------|-------------|
| **🎨 Overlay** | Render mesh on single image |
| **🎨 Overlay Batch** | Render mesh on video frames |

### Mesh Sequence Management
| Node | Description |
|------|-------------|
| **📋 Accumulator** | Collect meshes into sequence |
| **🔄 Mesh → Sequence** | Convert single mesh to sequence |
| **👁️ Preview** | Preview mesh sequence info |
| **〰️ Smooth** | Apply temporal smoothing to reduce jitter |
| **🗑️ Clear** | Clear accumulated sequence |

## 📷 Camera Calibration

The extension supports multiple ways to specify camera parameters, in order of priority:

### Priority 1: Auto-Calibrate (Recommended)
Set `auto_calibrate: True` to use GeoCalib for automatic FOV estimation from the image.

```bash
# Install GeoCalib
pip install -e 'git+https://github.com/cvg/GeoCalib#egg=geocalib'
```

### Priority 2: Direct Focal Length in Pixels
If you know the focal length in pixels, use `focal_length_px` directly.

### Priority 3: DSLR/Cinema Camera Parameters
For real cameras, specify physical parameters:
- `focal_length_mm`: Your lens focal length (e.g., 50)
- `sensor_width_mm`: Your sensor width (see table below)

The extension calculates: `focal_px = focal_mm × (image_width_px / sensor_width_mm)`

#### Common Sensor Sizes

| Camera Type | Sensor Width |
|-------------|--------------|
| **Full Frame** (Sony A7, Canon R5, Nikon Z) | 36.0 mm |
| **APS-C Sony/Nikon/Fuji** | 23.5 mm |
| **APS-C Canon** | 22.3 mm |
| **Micro Four Thirds** (Panasonic, Olympus) | 17.3 mm |
| **Super 35 Cinema** | 24.89 mm |
| **RED Komodo** | 27.03 mm |
| **ARRI Alexa Mini** | 28.17 mm |
| **Blackmagic Pocket 6K** | 23.10 mm |
| **iPhone 15 Pro (Main)** | 9.8 mm |
| **iPhone 15 Pro (Ultra Wide)** | 9.8 mm |

#### Example: Sony A7 IV with 50mm Lens
```
focal_length_mm: 50
sensor_width_mm: 36.0
```
For a 1920px wide image: `focal_px = 50 × (1920 / 36) = 2667px`

### Priority 4: Simple FOV
Set `fov` in degrees (default: 55°). Good for quick tests.

## 🎨 Overlay Rendering

The overlay renderer supports two modes:

| Mode | Description | Requirements |
|------|-------------|--------------|
| **solid** | 3D rendered mesh with lighting (Meta's approach) | pyrender, osmesa |
| **wireframe** | Fast wireframe using OpenCV | None (always works) |

### For Solid Rendering on Headless Servers
```bash
# Install OSMesa for software rendering
apt-get install libosmesa6-dev

# Install PyOpenGL
pip install PyOpenGL PyOpenGL_accelerate
```

## 〰️ Temporal Smoothing

Reduce jitter in mesh animations with Gaussian temporal smoothing:

| Parameter | Description |
|-----------|-------------|
| `smoothing_strength` | 0.0 (none) to 1.0 (maximum) |
| `smoothing_radius` | Number of frames to consider (1-10) |

Higher values = smoother but may lose quick movements.

## 📦 Alembic Export

The Alembic exporter creates industry-standard `.abc` files compatible with:
- **Maya** - File > Import
- **Houdini** - Alembic SOP
- **Blender** - File > Import > Alembic
- **Cinema 4D** - Merge Object
- **3ds Max** - Import Alembic

### Export Options
| Parameter | Description |
|-----------|-------------|
| `fps` | Animation frame rate (default: 30) |
| `include_uvs` | Include UV coordinates if available |
| `include_normals` | Include vertex normals |

## 🔄 Workflow Example

```
Load Video → SAM3DBody Batch Processor → Export Alembic
                    ↓
              Overlay Batch → Save Video
```

1. Load your video using VideoHelperSuite
2. Process with **SAM3DBody2abc Batch Processor**
3. Export mesh animation with **Export Alembic**
4. Render overlay with **Overlay Batch**
5. Save result with VideoHelperSuite

## 📝 Changelog

### v2.3.1 - Fixed Overlay Rendering
- **FIXED**: Pyrender now uses `osmesa` for headless servers (no display needed)
- **FIXED**: Wireframe projection matches Meta's camera model exactly
- **ADDED**: Debug output for first frame projection

### v2.3.0 - Simplified & Removed FBX
- **REMOVED**: FBX export (was causing issues with Maya joint hierarchy)
- **SIMPLIFIED**: Removed broken Meta renderer detection
- **IMPROVED**: Cleaner codebase focused on Alembic + overlay

### v2.2.x - DSLR Camera Support
- Added `focal_length_mm` and `sensor_width_mm` parameters
- Common sensor size database
- GPU optimization hints

### v2.1.x - FOV Calibration
- GeoCalib auto-calibration support
- Manual FOV override
- Improved camera parameter handling

### v2.0.x - Initial Release
- Video batch processing
- Animated Alembic export
- Mesh overlay rendering
- Temporal smoothing
- MHR 127-joint hierarchy support

## 📋 Requirements

### Required
- ComfyUI-SAM3DBody
- numpy
- torch
- opencv-python
- trimesh
- scipy

### Optional
- pyrender (for solid overlay rendering)
- PyOpenGL + OSMesa (for headless solid rendering)
- GeoCalib (for auto FOV calibration)
- ComfyUI-VideoHelperSuite (for video input/output)

## 🐛 Troubleshooting

### "Cannot connect to None" error
This means pyrender can't find a display. Solutions:
1. Use `render_mode: wireframe` (always works)
2. Install OSMesa: `apt-get install libosmesa6-dev`
3. Set environment: `export PYOPENGL_PLATFORM=osmesa`

### Mesh overlay doesn't align with person
1. Check your FOV/focal length settings
2. Try `auto_calibrate: True` if GeoCalib is installed
3. For DSLR footage, use correct `sensor_width_mm`

### Alembic export fails
1. Ensure Blender is installed (bundled with ComfyUI-SAM3DBody)
2. Check write permissions on output directory
3. Try OBJ sequence export as fallback

## 📄 License

MIT License - See LICENSE file

## 🙏 Credits

- [Meta SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) - Core mesh recovery model
- [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) - ComfyUI integration
- [GeoCalib](https://github.com/cvg/GeoCalib) - Camera calibration
