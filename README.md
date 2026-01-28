# SAM3DBody2abc v5.4.0

**Standalone** ComfyUI package for video-to-animated-FBX export using Meta's SAM-3D-Body.

🚀 **Zero manual setup!** Everything downloads automatically on first run.

## Features

- 🎯 **Fully Automatic Setup** - Downloads model & code on first run
- 📹 **Video Processing** - Batch process video frames with tracking
- 🦴 **Full Skeleton** - 70-joint MHR skeleton with rotations
- 📦 **FBX/Alembic Export** - Animated mesh + skeleton via Blender
- 🔧 **Direct Meta Integration** - No third-party wrapper dependencies

## Installation

### Step 1: Extract Package

```bash
cd ComfyUI/custom_nodes
unzip ComfyUI-SAM3DBody2abc-v5.4.0.zip
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

## Usage

### Workflow

```
[Load Video] → [🔧 Load SAM3DBody Model] → [🎬 Video Batch Processor] → [📦 Export Animated FBX]
                      ↑
               (paste hf_token here)
```

### First Run Output

```
[SAM3DBody Loader] Cloning SAM-3D-Body to: .../custom_nodes/sam-3d-body
[SAM3DBody Loader] SAM-3D-Body cloned successfully!
[SAM3DBody Loader] ==================================================
[SAM3DBody Loader]   FIRST RUN: Downloading model weights...
[SAM3DBody Loader] ==================================================
[SAM3DBody Loader] Downloading model weights to: .../models/sam3dbody
[SAM3DBody Loader] Model weights downloaded successfully!
```

### Subsequent Runs

After first run, everything is cached - no token needed, instant startup.

## Nodes

| Node | Description |
|------|-------------|
| 🔧 Load SAM3DBody Model (Direct) | Load model (auto-downloads on first run) |
| 🎬 Video Batch Processor | Process video → MESH_SEQUENCE |
| 📦 Export Animated FBX | Export to FBX/Alembic |
| 🔍 Verify Overlay | Debug visualization |
| 📷 Camera Solver | Camera motion estimation |
| 📊 Motion Analyzer | Motion statistics |

## External Camera Intrinsics

The Video Batch Processor supports external camera intrinsics to override SAM3DBody's internal estimation. This is useful for:
- Pre-calibrated cameras with known lens parameters
- Professional footage with known focal length
- Zoom lenses with variable focal length
- Multi-camera setups

### Intrinsics Priority

```
1. external_intrinsics (CAMERA_INTRINSICS) - MoGe2 or external
2. intrinsics_json (INTRINSICS) - From JSON file
3. SAM3DBody internal estimation - Default fallback
```

### Input Types

**`external_intrinsics`** (type: `CAMERA_INTRINSICS`)
- Connect from: MoGe2 Intrinsics node
- Provides per-frame focal length estimation

**`intrinsics_json`** (type: `INTRINSICS`)  
- Connect from: IntrinsicsFromJSON, IntrinsicsEstimator, or IntrinsicsFromSAM3DBody nodes
- Load from calibration JSON files

### Supported JSON Formats

**Simple format (single focal length):**
```json
{
  "focal_px": 1108.5,
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720
}
```

**Per-frame format (zoom lenses):**
```json
{
  "focal_px": 1108.5,
  "per_frame": [
    {"focal_px": 1100.0},
    {"focal_px": 1105.0},
    {"focal_px": 1110.0}
  ],
  "width": 1280,
  "height": 720
}
```

**MoGe2 format:**
```json
{
  "focal_length": 1108.5,
  "per_frame_focal": [1100.0, 1105.0, 1110.0],
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720
}
```

**Calibration format:**
```json
{
  "fx": 1108.5,
  "fy": 1108.5,
  "cx": 640.0,
  "cy": 360.0,
  "width": 1280,
  "height": 720
}
```

### Verifying Intrinsics

Use the **Verify Overlay** node with `intrinsics_source: "Compare Both"` to visualize the difference between SAM3DBody estimation and external intrinsics:
- **Green wireframe**: SAM3DBody intrinsics
- **Orange wireframe**: External intrinsics

## Requirements

- ComfyUI
- Python 3.10+
- PyTorch with CUDA (recommended)
- Git (for automatic sam-3d-body clone)
- Blender 3.6+ (for FBX export)
- HuggingFace account with model access

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

## Changelog

### v5.4.0 (Current)
- **NEW**: Automatic download with HuggingFace token input
- **NEW**: Direct SAM-3D-Body integration  
- **REMOVED**: All third-party wrapper dependencies
- **REMOVED**: Need for huggingface-cli installation

## License

MIT License (this package)

SAM-3D-Body model: [Meta License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE)
