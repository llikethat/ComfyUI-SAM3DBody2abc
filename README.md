# SAM3DBody2abc v5.8.0

**Standalone** ComfyUI package for video-to-animated-FBX export using Meta's SAM-3D-Body.

🚀 **Zero manual setup!** Everything downloads automatically on first run.

## Features

- 🎯 **Fully Automatic Setup** - Downloads model & code on first run
- 📹 **Video Processing** - Batch process video frames with tracking
- 🦴 **Full Skeleton** - 70-joint MHR skeleton with rotations
- 📦 **FBX/Alembic Export** - Animated mesh + skeleton via Blender
- 🔧 **Direct Meta Integration** - No third-party wrapper dependencies
- 📷 **Multi-Camera Support** - N-camera triangulation for jitter-free 3D
- ⚡ **Physics-Based Foot Contact** - GroundLink neural network for accurate ground contact
- 🎭 **Silhouette Refinement** - SMPL-based differentiable rendering (optional)

## Installation

### Step 1: Extract Package

```bash
cd ComfyUI/custom_nodes
unzip ComfyUI-SAM3DBody2abc-v5.8.0.zip
```

### Step 2: Get HuggingFace Token

1. Go to https://huggingface.co/facebook/sam-3d-body-dinov3
2. Click **"Request access"** (free, usually approved within hours)
3. Go to https://huggingface.co/settings/tokens
4. Create a new token with **Read** permission
5. Copy the token (starts with `hf_...`)

### Step 3: Run!

1. Start ComfyUI
2. Add **"🔧 Load SAM3DBody Model (Direct)"** node
3. Paste your HuggingFace token in the `hf_token` field
4. Run the workflow

On first run, it automatically downloads:
- **SAM-3D-Body source code** (~100MB from GitHub)
- **Model weights** (~3GB from HuggingFace)

## Workflows

### Single Camera (Basic)

```
[Load Video] → [🔧 Load SAM3DBody Model] → [🎬 Video Batch Processor] → [📦 Export Animated FBX]
                      ↑
               (paste hf_token here)
```

### Single Camera with Foot Contact

```
[Load Video] → [🎬 Video Batch Processor] → [⚡ GroundLink Foot Contact] → [📦 Export Animated FBX]
```

### Multi-Camera Triangulation (Jitter-Free 3D)

```
Camera A: [Load Video] → [🎬 Video Batch Processor] → [📷 Camera Accumulator] ─┐
                                                                                 │ (chain)
Camera B: [Load Video] → [🎬 Video Batch Processor] → [📷 Camera Accumulator] ─┤
                                                                                 │ (chain)
Camera C: [Load Video] → [🎬 Video Batch Processor] → [📷 Camera Accumulator] ─┘
                                                                                 ↓
                                                                          CAMERA_LIST
                                                                          ↓         ↓
                                                          [🎯 Auto-Calibrator]  [📷 Calibration Loader]
                                                                          ↓         ↓
                                                                     CALIBRATION_DATA
                                                                                 ↓
                                                          [🔺 Multi-Camera Triangulator] ← CAMERA_LIST
                                                                                 ↓
                                                                          TRAJECTORY_3D
                                                                                 ↓
                                                          [🎭 Silhouette Refiner] (optional)
                                                                                 ↓
                                                                     REFINED_TRAJECTORY_3D
```

## Nodes

### Core Processing

| Node | Description |
|------|-------------|
| 🔧 Load SAM3DBody Model (Direct) | Load model (auto-downloads on first run) |
| 🎬 Video Batch Processor | Process video → MESH_SEQUENCE |
| 📦 Export Animated FBX | Export to FBX/Alembic |
| 📄 Export BVH | Export to BVH motion capture format |

### Foot Contact Detection

| Node | Description |
|------|-------------|
| ⚡ GroundLink Foot Contact (Physics) | **PRIMARY** - Neural network GRF prediction |
| 🦶 Foot Tracker (TAPNet) | Visual foot tracking fallback |
| 🦶 Foot Contact Enforcer | Heuristic height/velocity fallback |
| 📊 GroundLink Contact Visualizer | Debug visualization |

### Multi-Camera

| Node | Description |
|------|-------------|
| 📷 Camera Accumulator | Build CAMERA_LIST from multiple views |
| 🎯 Camera Auto-Calibrator | Auto-calibrate from person keypoints |
| 📷 Camera Calibration Loader | Load calibration from JSON |
| 🔺 Multi-Camera Triangulator | Triangulate 3D from N cameras |
| 🎭 Silhouette Refiner | Refine trajectory using mask constraints |

### Analysis & Debug

| Node | Description |
|------|-------------|
| 📊 Motion Analyzer | Motion statistics and trajectory |
| 📈 Trajectory Smoother | Reduce jitter in trajectories |
| 🔍 Verify Overlay | Debug visualization overlay |
| 📷 Camera Solver V2 | Camera motion estimation (TAPIR) |

### Intrinsics

| Node | Description |
|------|-------------|
| 📐 MoGe2 Intrinsics Estimator | AI-based focal length estimation |
| 📐 Intrinsics from SAM3DBody | Extract intrinsics from mesh sequence |
| 📐 Intrinsics from JSON | Load from calibration file |

## Requirements

### Required

- ComfyUI
- Python 3.10+
- PyTorch with CUDA (recommended)
- Git (for automatic sam-3d-body clone)
- HuggingFace account with model access

### Optional

- **Blender 3.6+** - For FBX export
- **smplx** - For SMPL body model in Silhouette Refiner
- **pytorch3d** - For differentiable rendering in Silhouette Refiner

## External Dependencies & Licenses

See [docs/Dependencies.md](docs/Dependencies.md) for full details.

| Package | License | Usage |
|---------|---------|-------|
| SAM-3D-Body | Meta License | Core body reconstruction |
| GroundLink | MIT | Physics-based foot contact |
| TAPNet/TAPIR | Apache 2.0 | Point tracking |
| LightGlue | Apache 2.0 | Feature matching |
| Kornia/LoFTR | Apache 2.0 | Feature matching |
| PyTorch3D | BSD | Differentiable rendering (optional) |
| SMPL/SMPL-X | MPI License | Body model (optional) |
| Ultralytics YOLO | AGPL-3.0 | Person detection |

⚠️ **Note**: Ultralytics YOLO uses AGPL-3.0 which has copyleft requirements. For commercial use without AGPL obligations, disable YOLO-based features or obtain a commercial license.

## Troubleshooting

### "Authentication failed" error

1. Make sure you requested access at https://huggingface.co/facebook/sam-3d-body-dinov3
2. Wait for approval (check your email)
3. Get a fresh token from https://huggingface.co/settings/tokens
4. Paste it in the `hf_token` field

### "Git not found"

Install git:
```bash
sudo apt install git  # Linux
brew install git      # Mac
```

### "Blender not found" (FBX export)

```bash
sudo apt install blender  # Linux
# Or download from https://www.blender.org/download/
```

### "BFloat16 not supported" (GroundLink)

Fixed in v5.7.0+. Update to the latest version. The checkpoint is automatically converted to Float32 for compatibility with older GPUs and CPU.

### GroundLink shows "L=0 R=0 frames"

This means no foot contacts were detected. Check:
1. Is the person walking/running in the video?
2. Are feet visible in the frame?
3. Try adjusting `grf_threshold` (lower = more sensitive)

## Changelog

### v5.8.0 (Current)
- **NEW**: Camera Accumulator now accepts per-camera intrinsics
  - `focal_length_mm` input (0 = auto-detect from mesh_sequence)
  - `sensor_width_mm` input (default: 36mm full-frame)
- **UPDATED**: Auto-Calibrator uses intrinsics from CAMERA_LIST
- **UPDATED**: Comprehensive documentation overhaul
  - Full version history
  - External dependencies and licenses
  - Updated node reference

### v5.7.0
- **NEW**: 🎭 Silhouette Refiner node
  - Refine triangulated trajectory using silhouette consistency
  - SMPL body model + PyTorch3D differentiable rendering
  - Skeleton hull fallback (no dependencies required)
  - Multi-camera silhouette constraints
- **FIX**: GroundLink BFloat16 compatibility
  - Converts checkpoint weights to Float32 automatically
  - Works on older GPUs and CPU

### v5.6.0
- **NEW**: 📷 Camera Accumulator node (serial chaining pattern)
  - Build CAMERA_LIST by chaining cameras
  - Supports 2 to unlimited cameras
- **UPDATED**: 🎯 Camera Auto-Calibrator accepts CAMERA_LIST
  - Pairwise calibration against reference camera
  - Supports N cameras (3, 4, 5+)
- **UPDATED**: 🔺 Multi-Camera Triangulator accepts CAMERA_LIST
  - N-camera triangulation using weighted least squares
  - Better accuracy with more camera views

### v5.5.0
- **NEW**: ⚡ GroundLink Physics-Based Foot Contact (PRIMARY solver)
  - Neural network predicts Ground Reaction Forces from poses
  - MIT License (commercial use OK)
  - Pretrained checkpoints included
- **NEW**: 📊 GroundLink Contact Visualizer

### v5.4.0
- **NEW**: 🔧 Direct SAM3DBody Integration
  - No longer requires third-party ComfyUI-SAM3DBody wrapper
  - Load Meta's SAM-3D-Body model directly
  - Full control over model loading and coordinate system
- **NEW**: 🦶 Foot Tracker (TAPNet) - Visual foot tracking
- **NEW**: Coordinate system documentation

### v5.2.0
- **NEW**: flip_ty option for newer SAM3DBody versions
- **NEW**: 📐 Body Shape Lock node
- **NEW**: 🔄 Pose Smoothing node
- **NEW**: 🦶 Foot Contact Enforcer node
- **NEW**: 📹 SLAM Camera Solver node
- **FIX**: Improved Blender auto-detection

## License

MIT License (this package)

External dependencies have their own licenses - see [docs/Dependencies.md](docs/Dependencies.md)

SAM-3D-Body model: [Meta License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE)
