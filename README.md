# ComfyUI-SAM3DBody2abc

![Version](https://img.shields.io/badge/version-2.3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Extension for [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) that adds:
- **Video batch processing** - Process entire videos at once
- **Animated Alembic export** - Export geometry animation as .abc files
- **Mesh overlay rendering** - Visualize 3D mesh on video frames
- **Temporal smoothing** - Reduce jitter in animations
- **DSLR/Cinema camera support** - Use real-world camera parameters

## 📦 Installation

1. Install [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) first

2. Install system dependencies (for headless rendering):
```bash
apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx libglvnd-dev freeglut3-dev
```

3. Clone this repository into `custom_nodes`:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/llikethat/ComfyUI-SAM3DBody2abc
```

4. Install Python requirements:
```bash
cd ComfyUI-SAM3DBody2abc
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

| Camera Type | Sensor Width | Notes |
|-------------|--------------|-------|
| **Full Frame** (Sony A7, Canon R5, Nikon Z) | 36.0 mm | |
| **APS-C Sony/Nikon/Fuji** | 23.5 mm | |
| **APS-C Canon** | 22.3 mm | |
| **Micro Four Thirds** (Panasonic, Olympus) | 17.3 mm | |

**Cinema Cameras:**

| Camera | Sensor Width | Notes |
|--------|--------------|-------|
| **Sony Venice 2** (Full Frame 8.6K) | 36.2 mm | Full Frame mode |
| **Sony Venice 2** (Super 35 5.8K) | 24.9 mm | Super 35 mode |
| **ARRI Alexa 35** (Super 35 4.6K) | 28.25 mm | Super 35 mode |
| **ARRI Alexa 35** (Open Gate) | 34.98 mm | Large Format mode |
| **ARRI Alexa Mini** | 28.17 mm | |
| **RED Komodo** | 27.03 mm | Super 35 |
| **RED V-Raptor** | 40.96 mm | Vista Vision |
| **Blackmagic Pocket 6K** | 23.10 mm | Super 35 |
| **Blackmagic URSA Mini Pro 12K** | 27.03 mm | Super 35 |
| **Canon C70** | 26.2 mm | Super 35 DGO |
| **Super 35 (standard)** | 24.89 mm | Generic |

**Smartphone Cameras:**

| Device | Sensor Width | Notes |
|--------|--------------|-------|
| **iPhone 15 Pro** (Main 48MP) | 9.8 mm | 24mm equivalent |
| **iPhone 15 Pro** (Ultra Wide) | 9.8 mm | 13mm equivalent |
| **Samsung Galaxy S24 Ultra** (Main) | 9.56 mm | |

#### Example: Sony Venice 2 with 50mm Lens (Full Frame mode)
```
focal_length_mm: 50
sensor_width_mm: 36.2
```
For a 1920px wide image: `focal_px = 50 × (1920 / 36.2) = 2652px`

### Priority 4: Simple FOV
Set `fov` in degrees (default: 55°). Good for quick tests.

## 🎨 Overlay Rendering

The overlay renderer supports two modes:

| Mode | Description | Requirements |
|------|-------------|--------------|
| **solid** | 3D rendered mesh with lighting (Meta's approach) | pyrender, OSMesa |
| **wireframe** | Fast wireframe using OpenCV | None (always works) |

### For Solid Rendering on Headless Servers

Install OSMesa and required libraries:
```bash
# System packages
apt-get update && apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglvnd-dev \
    freeglut3-dev

# Python packages
pip install PyOpenGL PyOpenGL_accelerate
```

**Note:** The "No FOV estimator... Using the default FOV!" message is from SAM3DBody's internal logging and is expected - our extension correctly passes the calibrated focal length to the model.

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
| `world_space` | Apply 180° X rotation to match overlay (default: True) |
| `include_uvs` | Include UV coordinates if available |
| `include_normals` | Include vertex normals |

## 🎬 Maya Import Guide

### Camera Translation Values Explained

The `cam_t` values from SAM3DBody represent the camera position in meters:

```
cam_t = [X, Y, Z]
         ↑  ↑  ↑
         |  |  └── Depth (distance from camera to subject)
         |  └───── Vertical (camera height above pelvis)
         └──────── Horizontal (left/right offset)
```

Example: `cam_t = [-0.075, 1.535, 1.208]`
- Camera is 7.5cm to the left of center
- Camera is 1.53m above the mesh origin (pelvis)
- Subject is 1.21m from camera

### Alembic Import with Correct Alignment

**With `world_space: True` (default, recommended):**
The mesh is pre-transformed to match the overlay render. Just import and it should look correct.

**With `world_space: False`:**
You'll need to rotate the mesh -180° around X axis in Maya.

### Setting Up Matching Camera in Maya

1. **Create Camera:**
   ```python
   # Maya Python
   import maya.cmds as cmds
   
   cam = cmds.camera(name='sam3d_camera')[0]
   ```

2. **Set Focal Length:**
   ```python
   # focal_length from SAM3DBody is in PIXELS
   # Convert to Maya's mm assuming 36mm sensor (full frame)
   focal_px = 686.2  # from console output
   image_width = 640  # your image width
   sensor_width_mm = 36.0
   
   focal_mm = focal_px * sensor_width_mm / image_width
   cmds.setAttr(cam + '.focalLength', focal_mm)
   ```

3. **Set Film Back (sensor size):**
   ```python
   cmds.setAttr(cam + '.horizontalFilmAperture', 1.417)  # 36mm in inches
   cmds.setAttr(cam + '.verticalFilmAperture', 0.945)    # 24mm in inches
   ```

### Quick Maya Settings

| Setting | Value |
|---------|-------|
| World Space Export | `True` (recommended) |
| Scale | `1.0` (meters) or `100` (cm for Maya) |
| Up Axis | `Y` for Maya |
| Maya Film Back | 36mm × 24mm (Full Frame) |
| Focal Length Formula | `focal_px × 36 / image_width` mm |

### Static Camera Shot (Camera Fixed, Character Moving)

When your real-world camera is static and the character moves through frame:

**Console output example:**
```
Camera frame 0: [-0.017, 1.117, 3.179]
Camera frame 46: [-0.034, 1.177, 2.612]
Camera delta: [-0.017, 0.060, -0.567]
```

This means:
- Character moved ~57cm **closer** to camera (Z delta = -0.567)
- Character moved ~6cm **up** (Y delta = 0.060)
- Character moved ~1.7cm **right** in frame (X delta = -0.017)

**Maya Setup for Static Camera:**

```python
import maya.cmds as cmds

# Your values from console output:
focal_px = 686.2  # from [SAM3DBody2abc] Focal length: xxx
image_width = 640  # your video width in pixels
image_height = 360  # your video height in pixels

# Create camera at ORIGIN (static camera)
cam, cam_shape = cmds.camera(name='sam3d_camera')

# Convert focal length: pixels → mm (assuming 36mm sensor)
sensor_width_mm = 36.0
focal_mm = focal_px * sensor_width_mm / image_width
cmds.setAttr(cam_shape + '.focalLength', focal_mm)

# Set film back to match your video aspect ratio
aspect = image_width / image_height
sensor_height_mm = sensor_width_mm / aspect
cmds.setAttr(cam_shape + '.horizontalFilmAperture', sensor_width_mm / 25.4)  # mm to inches
cmds.setAttr(cam_shape + '.verticalFilmAperture', sensor_height_mm / 25.4)

# Camera stays at origin - mesh will move relative to it
cmds.setAttr(cam + '.translateX', 0)
cmds.setAttr(cam + '.translateY', 0)
cmds.setAttr(cam + '.translateZ', 0)

# Point camera down -Z axis (default in Maya)
cmds.setAttr(cam + '.rotateX', 0)
cmds.setAttr(cam + '.rotateY', 0)
cmds.setAttr(cam + '.rotateZ', 0)

print(f"Focal length: {focal_mm:.1f}mm")
print(f"Film back: {sensor_width_mm:.1f}mm x {sensor_height_mm:.1f}mm")
```

**For your specific values (640x360 video, focal=686.2px):**
```
Focal Length = 686.2 × 36 / 640 = 38.6mm
Film Back = 36mm × 20.25mm (16:9 aspect)
```

## 👥 Multi-Character Tracking with SAM3

For videos with multiple people (sports, group scenes), use [ComfyUI-SAM3](https://github.com/neverbiasu/ComfyUI-SAM3) to select specific characters before processing.

### Installation

```bash
# Install ComfyUI-SAM3 in your custom_nodes folder
cd ComfyUI/custom_nodes
git clone https://github.com/neverbiasu/ComfyUI-SAM3
cd ComfyUI-SAM3
pip install -r requirements.txt
```

### Workflow: Select Character → 3D Mesh → Export

```
[Load Video] → [LoadSAM3Model] → [SAM3VideoSegmentation]
                                   ↓ (click/text prompt)
                           [SAM3Propagate]
                                   ↓
                           [SAM3VideoOutput]
                                   ↓ MASK
[SAM3DBody2abc BatchProcessor] ← mask input
           ↓
[ExportAlembic] → character_1.abc
```

### Prompt Types

| Mode | How to Use | Example |
|------|------------|---------|
| **Text** | Describe the character | "player in red", "goalkeeper" |
| **Point** | Click on the character | Use SAM3CreatePoint nodes |
| **Box** | Draw box around character | Use SAM3CreateBox nodes |

### Multi-Character Export

For each character:
1. Create separate `SAM3VideoSegmentation` with different prompts
2. Each produces separate MASK output
3. Connect each mask to separate `BatchProcessor`
4. Export to separate Alembic files

**Will SAM3 track reliably?** SAM3's video tracking is generally robust, but:
- ✅ Works well for clear, distinct subjects
- ✅ Handles moderate occlusion
- ⚠️ Fast motion or motion blur can cause brief tracking loss
- ⚠️ Similar-looking subjects may require point prompts for disambiguation
- 💡 Use text prompts like "person on left" or click prompts for ambiguous cases

### Static Camera (Sports/Surveillance)

For fixed-camera shots with multiple moving characters:

```
BatchProcessor → ExportAlembic
                  ├── world_space: True
                  ├── static_camera: True  ← All characters at world positions
                  └── person_index: -1     ← All detected people
```

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

### v2.3.9 - ComfyUI-SAM3 Integration & Per-Frame Masks
- **ADDED**: Full support for per-frame masks from ComfyUI-SAM3
- **ADDED**: Auto-detection of per-frame vs single mask input
- **IMPROVED**: Mask bounding box computed per-frame for accurate tracking
- **DOCS**: Added workflow guide for multi-character segmentation with SAM3
- **NOTE**: Use [ComfyUI-SAM3](https://github.com/neverbiasu/ComfyUI-SAM3) for character selection

### v2.3.8 - Static Camera Mode for Sports/Surveillance
- **ADDED**: `static_camera` option in Alembic export
  - For wide shots where camera is fixed (sports, surveillance, multi-person scenes)
  - Places all characters at their absolute world positions
  - Camera at origin, characters move independently in world space
- **IMPROVED**: Tracking camera mode now clearly labeled in console output

### v2.3.7 - Multi-Character Support & Workflow Fixes
- **ADDED**: `person_index` parameter (-1=all, 0=first, 1=second, etc.)
- **ADDED**: `person_count` output showing max people detected across frames
- **ADDED**: Status message now shows character count
- **FIXED**: Maya Camera Script now pulls resolution from input images automatically
- **FIXED**: Workflow JSON updated with proper connections
- **ADDED**: `image_size` stored in mesh data for resolution tracking

### v2.3.6 - Maya Camera Script Node
- **ADDED**: New `🎥 Maya Camera Script` node - generates Python script with your solve values
- **ADDED**: Script includes focal length, film back, resolution, camera delta info
- **UPDATED**: Workflow now includes Maya Camera Script node with ShowText output
- Copy the generated script directly into Maya's Script Editor!

### v2.3.5 - World-Space Translation Fix
- **FIXED**: Character now translates in world-space when moving (uses cam_t delta)
- **ADDED**: Debug output showing camera delta between first and last frame
- **FIXED**: Offset transformation now correctly rotated with mesh

### v2.3.4 - Maya Alignment & World Space Export
- **ADDED**: `world_space` option in Alembic export (default: True)
- **ADDED**: Bakes 180° X rotation into mesh so it matches overlay in Maya
- **ADDED**: Maya import guide with camera setup instructions
- **ADDED**: Explanation of cam_t values (X, Y, Z in meters)

### v2.3.3 - Fixed Projection & Cleanup
- **FIXED**: OpenGL viewport Y-flip in wireframe projection - mesh now renders correctly
- **ADDED**: "filled" render mode - draws filled triangles with z-buffering (better than wireframe when pyrender unavailable)
- **FIXED**: "Found Blender" message no longer prints 9+ times (now cached)
- **REMOVED**: Joint hierarchy fallback message is normal (127 joints = MHR format)

### v2.3.2 - Bug Fixes & Cinema Cameras
- **FIXED**: Numpy array truth value error in overlay renderer
- **ADDED**: Sony Venice 2 sensor sizes (Full Frame & Super 35)
- **ADDED**: ARRI Alexa 35 sensor sizes (Super 35 & Open Gate)
- **ADDED**: More cinema cameras (RED V-Raptor, Canon C70, BMPCC 12K)
- **IMPROVED**: README with apt install commands for OSMesa

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

### System Dependencies (for headless rendering)
```bash
apt-get install -y libosmesa6-dev libgl1-mesa-glx libglvnd-dev freeglut3-dev
```

## 🐛 Troubleshooting

### "Cannot connect to None" or "OSMesaCreateContextAttribs" error
This means OSMesa is not properly installed. Solutions:
1. Install system packages: `apt-get install -y libosmesa6-dev libgl1-mesa-glx`
2. Use `render_mode: wireframe` (always works without OpenGL)

### "The truth value of an array..." error
Update to v2.3.2 which fixes this numpy array handling issue.

### Mesh overlay doesn't align with person
1. Check your FOV/focal length settings
2. Try `auto_calibrate: True` if GeoCalib is installed
3. For DSLR footage, use correct `sensor_width_mm`

### "No FOV estimator... Using the default FOV!" message
This is normal! It's from SAM3DBody's internal logging. Check the line above it - if you see `[SAM3DBody2abc] Focal length: XXX` with your calibrated value, the focal length IS being used correctly.

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
