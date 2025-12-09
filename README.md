# SAM3DBody2abc v3.0.0 - Video to Animated FBX

Export video sequences to animated FBX with mesh shape keys and properly connected skeleton hierarchy.

## 🔧 Workflow

```
Load Video → SAM3 BBox Collector → SAM3 Grounding → Mask
                                                      ↓
Load SAM3DBody → 🎬 Video Batch Processor ←───────────┘
                            ↓
               📦 Export Animated FBX
                            ↓
               🎥 FBX Animation Viewer
```

## 📦 Nodes

| Node | Description |
|------|-------------|
| **🎬 Video Batch Processor** | Process video with SAM3DBody, collect mesh_data per frame |
| **📦 Export Animated FBX** | Export with mesh shape keys + skeleton keyframes |
| **🎥 FBX Animation Viewer** | Preview animated FBX (requires MotionCapture extension) |
| **📋 Mesh Data Accumulator** | Manually accumulate mesh_data from SAM3DBody Process |
| **💾 Export Sequence JSON** | Save sequence to JSON |
| **📦 Export FBX from JSON** | Convert JSON to FBX |
| **🗑️ Clear Accumulator** | Clear data |

## 🆕 Features in v3.0.0

### ✅ Properly Connected Skeleton Hierarchy
- Joints are connected using parent-child relationships from MHR model
- Root joint (pelvis/hip) correctly identified
- Bone tails point toward children for better visualization
- 127-joint skeleton with proper hierarchy

### ✅ Orientation Options
Choose which axis points up:
- **Y** (default) - Standard Y-up orientation
- **Z** - Blender default Z-up
- **-Y** - Inverted Y
- **-Z** - Inverted Z

### ✅ FBX Animation Viewer
Preview animated FBX files directly in ComfyUI. Works with ComfyUI-MotionCapture web extension for full playback controls.

## 🔗 SAM3DBody Integration

Works with SAM3DBody Process node outputs:

| Output | Type | Description |
|--------|------|-------------|
| `mesh_data` | SAM3D_OUTPUT | vertices, faces, joint_coords (127 joints) |
| `skeleton` | SKELETON | joint_positions, joint_rotations, params |
| `debug_image` | IMAGE | Visualization |

## 📥 Installation

1. Copy `ComfyUI-SAM3DBody2abc` to `ComfyUI/custom_nodes/`
2. Dependencies:
   - **ComfyUI-SAM3** (segmentation)
   - **ComfyUI-SAM3DBody** (body reconstruction)
   - **Blender** (bundled with SAM3DBody or system)
3. Optional:
   - **ComfyUI-MotionCapture** (for FBX viewer web extension)

## ⚙️ Options

### Video Batch Processor
| Option | Default | Description |
|--------|---------|-------------|
| `smoothing_strength` | 0.5 | Temporal smoothing (0=none) |
| `skip_frames` | 1 | Process every Nth frame |
| `inference_type` | full | `full` (body+hands) or `body` |

### Export Animated FBX
| Option | Default | Description |
|--------|---------|-------------|
| `include_mesh` | true | Include mesh with shape keys |
| `up_axis` | Y | Which axis points up (Y, Z, -Y, -Z) |
| `fps` | 24.0 | Animation framerate |

## 📋 Output FBX Contains

- **Mesh** with shape keys (one per frame for vertex animation)
- **Armature** with 127 joints in proper hierarchy
- **Keyframed** joint positions per frame
- **Parent-child** bone connections

## 🎬 Sample Workflow

Included: `workflows/animation_workflow.json`

This workflow demonstrates:
1. Load video with VHS_LoadVideo
2. Create bounding box with SAM3BBoxCollector
3. Segment person with SAM3Grounding
4. Process frames with Video Batch Processor
5. Export to animated FBX

## 🔧 Skeleton Hierarchy

The skeleton uses MHR's 127-joint model:
- **Root**: Pelvis/Hip (joint 0 or identified from joint_parents)
- **Hierarchy**: Established via `joint_parents` array
- **Bones**: Tails point toward first child for visualization
- **Animation**: Location keyframes on each joint

## 📄 License

MIT License
