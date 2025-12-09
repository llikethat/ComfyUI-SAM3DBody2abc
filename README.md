# SAM3DBody2abc v3.0.0 - Video to Animated FBX

Export video sequences to animated FBX with mesh shape keys and properly connected skeleton hierarchy.

## 🔧 Workflow

```
VHS_LoadVideo ──┬──→ SAM3BBoxCollector → SAM3VideoSegmentation
                │                                  ↓
                │                          SAM3Propagate
                │                                  ↓
                │                          SAM3VideoOutput → per-frame masks
                │                                                    ↓
                └──→ 🎬 Video Batch Processor ←──────────────────────┘
                              ↑
                    LoadSAM3DBodyModel
                              ↓
                   📦 Export Animated FBX
                              ↓
                   🎥 FBX Animation Viewer
```

Uses SAM3's built-in video propagation for accurate per-frame character tracking.

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

## 🎯 Character Tracking

Use SAM3's video segmentation nodes:

1. **SAM3BBoxCollector** - Draw bbox around character on first frame
2. **SAM3VideoSegmentation** - Initialize video tracking
3. **SAM3Propagate** - Propagate mask across all frames
4. **SAM3VideoOutput** - Get per-frame MASK output

This gives accurate per-frame masks that follow the character's movement.

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
| `include_camera` | true | Include camera with focal length from SAM3DBody |
| `up_axis` | Y | Which axis points up (Y, Z, -Y, -Z) |
| `fps` | 24.0 | Animation framerate |

## 📷 Camera Export

SAM3DBody estimates the camera focal length for each frame. The export includes:
- **Focal length** converted from pixels to mm
- **Camera position** based on subject depth
- **Per-frame animation** if focal length varies

### Sensor Width Options

| Camera Type | Sensor Width | Notes |
|-------------|--------------|-------|
| Full Frame | 36.0 mm | Default, matches most DSLR/mirrorless |
| APS-C (Canon) | 22.3 mm | Canon crop sensor |
| APS-C (Nikon/Sony) | 23.6 mm | Nikon/Sony crop sensor |
| Micro Four Thirds | 17.3 mm | Olympus/Panasonic |
| 1-inch | 13.2 mm | Compact cameras |
| iPhone/Smartphone | 5-7 mm | Varies by model |

### Focal Length Conversion
```
focal_mm = focal_px × (sensor_width / image_width)
Example: 1500px × (36mm / 1920px) = ~28mm
```

## 📋 Output FBX Contains

- **Mesh** with shape keys (one per frame for vertex animation)
- **Armature** with 127 joints in proper hierarchy
- **Keyframed** joint positions per frame
- **Parent-child** bone connections
- **Camera** with estimated focal length (optional)

## 🎬 Sample Workflow

Included: `workflows/animation_workflow.json`

Node sequence:
1. **VHS_LoadVideo** - Load video file
2. **SAM3BBoxCollector** - Draw bbox on first frame
3. **SAM3VideoSegmentation** - Initialize tracking
4. **SAM3Propagate** - Track mask across frames
5. **SAM3VideoOutput** - Get per-frame masks
6. **Video Batch Processor** - Process with SAM3DBody
7. **Export Animated FBX** - Export with mesh + skeleton + camera

## 🔧 Skeleton Hierarchy

The skeleton uses MHR's 127-joint model:
- **Root**: Pelvis/Hip (joint 0 or identified from joint_parents)
- **Hierarchy**: Established via `joint_parents` array
- **Bones**: Tails point toward first child for visualization
- **Animation**: Location keyframes on each joint

## 📄 License

MIT License
