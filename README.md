# SAM3DBody2abc v3.0.0 - Video to Animated FBX

Export video sequences to animated FBX with mesh shape keys and skeleton.

## 🔧 Workflow

```
Load Video → SAM3 Video Segmentation → SAM3 Propagate → SAM3 Extract Masks
                                                              ↓
Load SAM3DBody → 🎬 Video Batch Processor ←───────────────────┘
                            ↓
               📦 Export Animated FBX
```

## 📦 Nodes

| Node | Description |
|------|-------------|
| **🎬 Video Batch Processor** | Process video with SAM3DBody, collect mesh_data per frame |
| **📋 Mesh Data Accumulator** | Manually accumulate mesh_data from SAM3DBody Process |
| **💾 Export Sequence JSON** | Save sequence to JSON |
| **📦 Export Animated FBX** | Export with mesh shape keys + skeleton keyframes |
| **📦 Export FBX from JSON** | Convert JSON to FBX |
| **🗑️ Clear Accumulator** | Clear data |

## 🔗 SAM3DBody Native Integration

Works with SAM3DBody Process node outputs:

| Output | Type | Description |
|--------|------|-------------|
| `mesh_data` | SAM3D_OUTPUT | vertices, faces, joint_coords (127 joints) |
| `skeleton` | SKELETON | joint_positions, joint_rotations, params |
| `debug_image` | IMAGE | Visualization |

Our nodes accept `mesh_data` (SAM3D_OUTPUT) to accumulate per-frame data.

## 📥 Installation

1. Copy `ComfyUI-SAM3DBody2abc` to `ComfyUI/custom_nodes/`
2. Requires:
   - **ComfyUI-SAM3** (segmentation)
   - **ComfyUI-SAM3DBody** (body reconstruction)
   - **Blender** (bundled with SAM3DBody or system)

## ⚙️ Options

**Video Batch Processor:**
| Option | Default | Description |
|--------|---------|-------------|
| `smoothing_strength` | 0.5 | Temporal smoothing (0=none) |
| `skip_frames` | 1 | Process every Nth frame |
| `inference_type` | full | `full` (body+hands) or `body` |

**Export Animated FBX:**
| Option | Default | Description |
|--------|---------|-------------|
| `include_mesh` | true | Include mesh with shape keys |
| `fps` | 24.0 | Animation framerate |

## 🎯 Fixed Settings

- **Scale:** 1.0
- **Up axis:** Y
- **Coordinate flip:** Same as SAM3DBody (-X, -Y, -Z)

## 📋 Output FBX Contains

- **Mesh** with shape keys (one per frame for vertex animation)
- **Armature** with 127 joints
- **Keyframed** joint positions per frame

## 📝 Workflows

1. `video_to_animated_fbx.json` - Full video pipeline with SAM3 segmentation
2. `manual_with_native_nodes.json` - Shows both native SAM3DBody export and our accumulator

## 📄 License

MIT License
