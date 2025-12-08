# SAM3DBody2abc - Video to Animated FBX Export

Export video sequences to animated FBX files using SAM3DBody.

## 🔧 Workflow

```
Load Video → 🎬 Batch Process → 📦 Export FBX Direct
```

Simple 3-node workflow:
1. Load video with `VHS_LoadVideo`
2. Process all frames with `🎬 Batch Process` (includes temporal smoothing)
3. Export with `📦 Export FBX Direct`

## 📦 Nodes

| Node | Description |
|------|-------------|
| **🎬 Batch Process** | Process video frames with SAM3DBody + temporal smoothing |
| **📋 Frame Accumulator** | Manually accumulate frames (for custom workflows) |
| **〰️ Apply Smoothing** | Apply additional temporal smoothing |
| **💾 Export JSON** | Export sequence to JSON (intermediate format) |
| **📦 Export FBX Direct** | Export sequence directly to FBX |
| **📦 Export FBX (from JSON)** | Convert JSON to FBX |
| **🗑️ Clear Sequences** | Clear accumulated frames |

## 📥 Installation

1. Copy `ComfyUI-SAM3DBody2abc` to `ComfyUI/custom_nodes/`
2. Requires **ComfyUI-SAM3DBody** (the base extension)
3. Requires **Blender** for FBX export (or use SAM3DBody's bundled Blender)

## 🎬 Usage

### Simple Video to FBX (Recommended)

1. Load video with `VHS_LoadVideo`
2. Connect to `🎬 Batch Process`
3. Export with `📦 Export FBX Direct`

### Batch Process Options

| Option | Default | Description |
|--------|---------|-------------|
| `bbox_threshold` | 0.8 | Detection confidence |
| `inference_type` | full | `full` (body+hands) or `body` (faster) |
| `smoothing_strength` | 0.5 | Temporal smoothing (0=none, 2=heavy) |
| `smoothing_radius` | 3 | Neighboring frames for smoothing |
| `start_frame` | 0 | First frame to process |
| `end_frame` | -1 | Last frame (-1 = all) |
| `skip_frames` | 1 | Process every Nth frame |

### Export FBX Options

| Option | Default | Description |
|--------|---------|-------------|
| `filename` | animation | Output filename |
| `fps` | 24.0 | Animation framerate |
| `include_mesh` | true | Include mesh (false = skeleton only) |

## ⚙️ Fixed Settings

- **Scale:** 1.0
- **Up axis:** Y
- **Coordinate system:** Blender/Maya compatible

## 📝 Version History

### v2.6.0
- Simplified architecture using SAM3DBody's native Process Image
- Removed batch processor (use native nodes)
- Removed BVH export (use FBX)
- Fixed scale at 1.0, up axis at Y
- Efficient frame-by-frame accumulation with temporal smoothing

## 📄 License

MIT License
