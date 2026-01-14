# ComfyUI-SAM3DBody2abc

**Video to Animated FBX Export with Camera Solving**

Convert video input to animated 3D body mesh with automatic camera motion compensation.

## Features

### Camera Solver
Comprehensive camera solving with automatic shot type detection:

| Shot Type | Description | Solver Method |
|-----------|-------------|---------------|
| **Auto** | Automatically detect shot type | Homography analysis |
| **Static** | No camera motion | Identity matrices |
| **Nodal** | Rotation only (tripod) | Homography decomposition |
| **Parallax** | Translation (handheld/dolly) | COLMAP / Essential Matrix |
| **Hybrid** | Multiple transitions | Segment + stitch |

### Quality Modes

| Mode | Nodal Pipeline | Parallax Pipeline |
|------|----------------|-------------------|
| **Fast** | KLT only | LightGlue → COLMAP |
| **Balanced** | KLT → LoFTR fallback | LightGlue → LoFTR → COLMAP |
| **Best** | LoFTR always | LoFTR → COLMAP |

### External Camera Import
Import solved camera data from professional tracking applications:
- PFTrack
- 3DEqualizer
- SynthEyes
- Maya
- Nuke
- After Effects

## Installation

### Requirements
```bash
pip install torch torchvision
pip install opencv-python numpy
pip install ultralytics  # YOLO for person detection

# Optional but recommended for better matching:
# pip install lightglue   # Apache 2.0 license - auto-downloads on first use
# pip install kornia      # LoFTR - Apache 2.0 license - auto-downloads on first use
```

### For Parallax/Translation Solving
Install [ComfyUI-COLMAP](https://github.com/your-repo/ComfyUI-COLMAP) for full bundle adjustment:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI-COLMAP
```

Without COLMAP, parallax shots use Essential Matrix fallback (less accurate).

### For Better Camera Intrinsics (Optional)
Install MoGe2 for accurate monocular intrinsics estimation:
```bash
pip install moge
```
Models auto-download from HuggingFace on first use (~60ms per frame on RTX3090).

### Model Downloads
LightGlue and LoFTR models are downloaded automatically on first use:
- LightGlue SuperPoint: ~25MB
- LoFTR outdoor: ~50MB

## Nodes

### 📷 Camera Solver
Main camera solving node with automatic shot detection.

**Inputs:**
- `images`: Video frames (IMAGE)
- `shot_type`: Auto, Static, Nodal, Parallax, Hybrid
- `quality_mode`: Fast, Balanced, Best
- `foreground_masks`: Optional masks to exclude people
- `smoothing`: Gaussian smoothing window (0-21)
- `sensor_width_mm`: Camera sensor width (default: 36mm)
- `focal_length_mm`: Focal length (default: 35mm)
- `stitch_overlap`: Frames for hybrid stitching (default: 10)
- `transition_frames`: Manual transitions (e.g., "50,120")
- `match_threshold`: Min matches before fallback (default: 500)

**Outputs:**
- `camera_data`: CAMERA_DATA for FBX export
- `debug_vis`: Visualization frames
- `shot_info`: Detection/solve info string

### 📷 Camera Data from JSON
Import camera solve from external applications.

**JSON Format:**
```json
{
    "fps": 24,
    "image_width": 1920,
    "image_height": 1080,
    "sensor_width_mm": 36.0,
    "units": "degrees",
    "coordinate_system": "maya",
    "frames": [
        {"frame": 0, "pan": 0, "tilt": 0, "roll": 0, "tx": 0, "ty": 0, "tz": 0, "focal_length_mm": 35.0},
        {"frame": 1, "pan": 1.5, "tilt": 0.2, "roll": 0, "tx": 0.01, "ty": 0, "tz": 0.02, "focal_length_mm": 35.0}
    ]
}
```

## Workflow

```
Load Video → SAM3 Segmentation → SAM3DBody Processor
                                        ↓
                             Camera Solver (or JSON Import)
                                        ↓
                              Export Animated FBX
```

## GPU Requirements

- **Fast mode**: CPU only (KLT, ORB)
- **Balanced/Best**: GPU recommended for LightGlue/LoFTR
- Falls back to ORB (CPU) if GPU unavailable

## License

- Main code: MIT
- LightGlue: Apache 2.0
- LoFTR (kornia): Apache 2.0
- COLMAP: BSD-3

## Version History

### v4.1.0
- **Bake Camera into Geometry**: New export mode that applies inverse camera transforms to mesh/skeleton
  - Eliminates jitter by baking camera motion directly into vertex positions
  - Exports static camera with correct intrinsics
  - Three smoothing options before baking:
    - **Kalman Filter**: Optimal for sequential data, physics-based smoothing
    - **Spline Fitting**: Cubic splines for smooth continuous curves
    - **Gaussian**: Simple weighted averaging
- **MoGe2 Intrinsics Estimation**: Accurate focal length estimation from single images
  - Uses Microsoft's MoGe2 model for monocular geometry estimation
  - Much better than heuristics when camera metadata is unavailable
  - Can improve COLMAP's intrinsics for better accuracy
- New nodes:
  - `📷 MoGe2 Intrinsics Estimator`
  - `📷 Apply Intrinsics to Mesh`
  - `📷 Apply Intrinsics to Camera`

### v4.0.0
- Complete camera solver rewrite
- Auto shot type detection
- Quality modes (Fast/Balanced/Best)
- LightGlue + LoFTR integration
- Hybrid shot support with multi-transition
- Removed CoTracker (licensing)
- Removed RAFT (unnecessary)

### v3.x
- Initial camera rotation solver
- KLT, ORB, Nodal Pan methods
- JSON import support
