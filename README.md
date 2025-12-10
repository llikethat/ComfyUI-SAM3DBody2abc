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

### ✅ Accurate Joint Animation
- Joints use empties (locators) instead of bones
- Direct world space positioning for exact joint locations
- No parent-child transform accumulation issues
- 127-joint skeleton matches mesh animation

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
| `output_format` | FBX | Export format: FBX (blend shapes) or ABC (Alembic vertex cache) |
| `up_axis` | Y | Which axis points up (Y, Z, -Y, -Z) |
| `include_mesh` | true | Include mesh with animation |
| `include_camera` | true | Include camera with focal length from SAM3DBody |
| `sensor_width` | 36.0 | Camera sensor width in mm |
| `fps` | 24.0 | Animation framerate |

## ⚠️ Known Limitations

### Mesh Flattening on Extreme Rotations
When the character spins or flips significantly, the mesh may lose volume and flatten. This is a fundamental limitation of SAM3DBody's single-view 3D reconstruction - it cannot maintain consistent depth when the viewing angle changes dramatically.

**Workarounds:**
- Use footage with minimal rotation
- Export skeleton-only for retargeting to a proper 3D character
- Use multiple camera angles (requires multi-view reconstruction)

### FBX File Size / Load Time
FBX shape keys store per-frame vertex data, which can result in large files and slow loading in Maya/other DCCs. Maya may also show hidden per-frame geometry from blend shape targets.

**Workarounds:**
- **Use Alembic (.abc) format** - cleaner vertex animation, no hidden geometry
- Set `include_mesh=false` to export skeleton-only (fast, small)
- Use `skip_frames` > 1 in Video Batch Processor to reduce frame count

## 📦 Output Formats

### FBX (Default)
- Uses blend shapes (shape keys) for mesh animation
- Contains: mesh, joint locators, camera
- Note: Maya may show hidden per-frame geometry from blend shape targets

### Alembic (.abc)
- Uses vertex cache for mesh animation  
- Cleaner playback in Maya
- Also exports `_skeleton.fbx` with joints and camera

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
- **Skeleton** with 127 joint locators (empties/nulls)
- **Keyframed** joint positions per frame
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

## 🔧 Skeleton (Joint Locators)

The skeleton uses empties (locators/nulls) instead of bones:
- **Direct world space animation** - joints animate at exact positions
- **No bone local space issues** - avoids parent-child transform accumulation
- **127 joint locators** organized under "Skeleton" parent
- **Exports as nulls** in Maya, locators in other DCCs
- **Easy retargeting** - can be used to drive a rigged character

## 📄 License

MIT License
