# SAM3DBody2abc v3.0.0 - Video to Animated FBX

Export video sequences to animated FBX files with proper skeleton hierarchy.

## 🔧 Workflow

```
Load Video → SAM3 Video Segmentation → SAM3 Propagate → SAM3 Extract Masks
                                                              ↓
Load SAM3DBody → 🎬 Video Batch Processor ←───────────────────┘
                            ↓
               📦 Export Animated FBX
```

Uses **SAM3** for person segmentation and **SAM3DBody** for body reconstruction.

## 📦 Nodes

| Node | Description |
|------|-------------|
| **🎬 Video Batch Processor** | Process video frames with SAM3 masks, collect skeleton data |
| **📋 Skeleton Accumulator** | Accumulate SKELETON outputs from SAM3DBody Process |
| **💾 Export Skeleton JSON** | Export skeleton sequence to JSON |
| **📦 Export Animated FBX** | Convert skeleton sequence to animated FBX |
| **📦 Export FBX from JSON** | Convert saved JSON to animated FBX |
| **🗑️ Clear Accumulator** | Clear accumulated data |

## 📥 Installation

1. Copy `ComfyUI-SAM3DBody2abc` to `ComfyUI/custom_nodes/`
2. Requires: **ComfyUI-SAM3** (for segmentation)
3. Requires: **ComfyUI-SAM3DBody** (for body reconstruction)
4. Requires: **Blender** (bundled with SAM3DBody or system install)

## ⚙️ Settings

**Video Batch Processor:**
| Option | Default | Description |
|--------|---------|-------------|
| `smoothing_strength` | 0.5 | Temporal smoothing (0=none, 2=heavy) |
| `skip_frames` | 1 | Process every Nth frame |
| `inference_type` | full | `full` (body+hands) or `body` (faster) |

**Export Animated FBX:**
| Option | Default | Description |
|--------|---------|-------------|
| `filename` | animation | Output filename |
| `fps` | 24.0 | Animation framerate |

## 🎯 Fixed Settings

- **Scale:** 1.0
- **Up axis:** Y
- **Coordinate system:** Blender/Maya compatible

## 📋 Output

The animated FBX contains:
- Armature with 127 joints (MHR skeleton)
- Joint hierarchy from SAM3DBody
- Keyframed joint positions for each frame

## 🔗 Integration with SAM3DBody Native Nodes

This extension works alongside SAM3DBody's native nodes:
- `SAM3DBodyProcess` → outputs SKELETON
- `SAM3DBodySaveSkeleton` → saves to JSON/BVH/FBX (single frame)
- `SAM3DBodyExportFBX` → exports mesh + skeleton (single frame)

Our extension adds **video/animation** support:
- Process multiple frames
- Temporal smoothing
- Export animated FBX with keyframes

## 📝 Version History

### v3.0.0 (Fresh Build)
- Clean architecture using SAM3DBody's native SKELETON output
- Video batch processor with temporal smoothing
- Animated FBX export with proper joint hierarchy
- Fixed scale=1.0, up axis=Y

## 📄 License

MIT License
