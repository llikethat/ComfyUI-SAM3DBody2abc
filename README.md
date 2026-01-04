# ComfyUI-SAM3DBody2abc

**Video to Animated FBX Export with Camera Solving**

Convert video input to animated 3D body mesh with automatic camera motion compensation.

## Features

### Camera Solver V2 (NEW in v5.0)
TAPIR-based temporal point tracking for accurate camera motion estimation:

| Feature | Description |
|---------|-------------|
| **TAPIR Tracking** | Temporally-consistent point tracking across all frames |
| **Shot Classification** | Automatic detection: Static, Rotation, Translation, Mixed |
| **Background Masking** | Tracks only background points (excludes foreground) |
| **Rainbow Trails** | Debug visualization showing point trajectories |
| **Homography Solver** | Rotation-only solving for pan/tilt shots |

### Camera Solver Legacy (v4.x)
Frame-pair feature matching methods:

| Shot Type | Description | Solver Method |
|-----------|-------------|---------------|
| **Auto** | Automatically detect shot type | Homography analysis |
| **Static** | No camera motion | Identity matrices |
| **Nodal** | Rotation only (tripod) | Homography decomposition |
| **Parallax** | Translation (handheld/dolly) | COLMAP / Essential Matrix |
| **Hybrid** | Multiple transitions | Segment + stitch |

### Quality Modes (Legacy Solver)

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

# v5.0 TAPIR dependencies (required for CameraSolverV2):
pip install tensorflow tensorflow-datasets  # Required by tapnet
pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git'

# Download TAPIR checkpoint (~250MB):
mkdir -p ComfyUI/models/tapir
wget -P ComfyUI/models/tapir https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt

# Optional for high-quality mesh rendering:
pip install pyrender trimesh PyOpenGL

# Optional for legacy solver matching:
# pip install lightglue   # Apache 2.0 license
# pip install kornia      # LoFTR - Apache 2.0 license
```

### Quick Install (Copy-Paste)
```bash
# Install all TAPIR dependencies at once
pip install tensorflow tensorflow-datasets einshape
pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git'

# Download checkpoint
mkdir -p ComfyUI/models/tapir && \
wget -O ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt \
  https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
```

### Verify Installation
```bash
# Test TAPIR import
python -c "from tapnet.torch import tapir_model; print('TAPIR OK')"

# Test checkpoint loading
python -c "
import torch
from tapnet.torch import tapir_model
model = tapir_model.TAPIR(pyramid_level=1)
ckpt = torch.load('ComfyUI/models/tapir/bootstapir_checkpoint_v2.pt', map_location='cpu')
model.load_state_dict(ckpt)
print('TAPIR checkpoint OK')
"
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

## Nodes

### 📷 Camera Solver V2 (NEW)
TAPIR-based camera solver with temporal consistency.

**Inputs:**
- `images`: Video frames (IMAGE)
- `foreground_mask`: Mask to exclude foreground (MASK)
- `intrinsics`: Camera intrinsics (INTRINSICS)
- `quality`: fast, balanced, best
- `force_shot_type`: auto, rotation, translation, mixed
- `grid_size`: Point sampling grid density (5-20)
- `debug_visualization`: Show rainbow trails

**Outputs:**
- `camera_matrices`: Per-frame 4x4 transformation matrices
- `debug_vis`: Rainbow trail visualization
- `shot_info`: Shot classification results
- `status`: Human-readable status

### 📷 Intrinsics from SAM3DBody (NEW)
Extract camera intrinsics from SAM3DBody mesh_data.

**Inputs:**
- `mesh_data`: SAM3D_OUTPUT from SAM3DBody Process
- `images`: Video frames for debug overlay
- `render_mode`: solid, wireframe, points
- `mesh_color`: skin, white, green, blue, or hex

**Outputs:**
- `intrinsics`: INTRINSICS for CameraSolverV2
- `debug_overlay`: Mesh overlay visualization
- `focal_length_mm`: Extracted focal length
- `status`: Status string

### 📷 Camera Solver (Legacy)
Frame-pair feature matching solver (v4.x method).

**Inputs:**
- `images`: Video frames (IMAGE)
- `shot_type`: Auto, Static, Nodal, Parallax, Hybrid
- `quality_mode`: Fast, Balanced, Best
- `foreground_masks`: Optional masks to exclude people
- `smoothing`: Gaussian smoothing window (0-21)

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

### v5.0 Pipeline (Recommended)
```
Load Video → SAM3 Segmentation → SAM3DBody Processor
                  ↓                      ↓
           Foreground Mask         mesh_data
                  ↓                      ↓
            CameraSolverV2 ← IntrinsicsFromSAM3DBody
                  ↓
           Export Animated FBX
```

### Legacy Pipeline (v4.x)
```
Load Video → SAM3 Segmentation → SAM3DBody Processor
                                        ↓
                             Camera Solver (Legacy)
                                        ↓
                              Export Animated FBX
```

## GPU Requirements

- **TAPIR (v5.0)**: GPU strongly recommended
- **Fast mode (Legacy)**: CPU only (KLT, ORB)
- **Balanced/Best (Legacy)**: GPU recommended for LightGlue/LoFTR
- Falls back to ORB (CPU) if GPU unavailable

## License

- Main code: MIT
- TAPIR: Apache 2.0
- LightGlue: Apache 2.0
- LoFTR (kornia): Apache 2.0
- COLMAP: BSD-3

## Version History

### v5.0.0 (Current)
- **NEW: CameraSolverV2** - TAPIR-based temporal point tracking
  - Temporally-consistent tracking across all frames
  - Automatic shot classification (static/rotation/translation/mixed)
  - Background-only tracking with foreground mask exclusion
  - Rainbow trail debug visualization
  - Homography-based rotation solving
- **NEW: IntrinsicsFromSAM3DBody** - Extract camera intrinsics from mesh_data
  - Solid mesh rendering with pyrender (fallback to OpenCV)
  - Multiple render modes: solid, wireframe, points
  - Skeleton overlay visualization
- **Legacy CameraSolver** - Previous v4.x solver preserved
  - KLT, LoFTR, LightGlue methods still available
  - Use for comparison or fallback
- Python 3.12 compatible TensorFlow blocker for TAPIR

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
