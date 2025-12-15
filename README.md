# ComfyUI-SAM3DBody2abc

Export SAM3DBody mesh sequences to animated FBX or Alembic files for use in Maya, Blender, and other 3D applications.

## Features

### SAM3 Video Mask Integration (v3.1.x)
- **Direct SAM3 Propagation Support** - Connect SAM3 Propagate node's `SAM3_VIDEO_MASKS` output directly
- **Multi-person Filtering** - When mask is provided, only the masked character is tracked (others ignored)
- **Per-frame Tracking** - Uses mask for each frame to follow moving characters

### Verification Overlay
- **🔍 Verify Overlay** - Project 3D mesh/skeleton back onto original image
- Helps verify the correct person is being tracked
- Shows joint positions, skeleton connections, and mesh wireframe
- Mesh aligned to match detected keypoints

### Export Formats
- **FBX** - Blend shapes for mesh animation, widely compatible
- **Alembic (.abc)** - Vertex cache for mesh animation, cleaner Maya workflow

### Skeleton Animation Modes
- **Rotations (Recommended)** - Uses true joint rotation matrices from MHR model
  - Proper bone rotations for retargeting to other characters
  - Standard animation workflow compatible
- **Positions (Legacy)** - Uses joint positions with location offsets
  - Shows exact joint positions
  - Limited retargeting capability

### World Translation Modes
- **None (Body at Origin)** - Character centered, camera can be static OR animated (pan/tilt)
- **Baked into Mesh/Joints** - World offset baked into positions, static camera
- **Baked into Camera** - Body at origin, camera animated (translation or rotation)
- **Root Locator** - Root empty carries translation, body/skeleton as children, static camera
- **Root Locator + Animated Camera** ⭐ - Character path visible AND camera follows (best for moving camera shots)
- **Separate Track** - Body at origin + separate locator showing world path

### Camera Motion Modes
The `camera_motion` option controls how the camera follows the character:

| Mode | Description | Best For |
|------|-------------|----------|
| **Translation (Default)** | Camera moves laterally to frame character | Dolly/crane/steadicam shots |
| **Rotation (Pan/Tilt)** | Camera pans/tilts to follow character | Tripod/handheld/locked-off shots |

**Rotation Math (Y-up / Maya):**
- From SAM3DBody projection: `x_2d = focal * tx / tz + center`
- Camera rotation: `pan_angle = atan2(tx, tz)`, `tilt_angle = atan2(ty, tz)`
- Using `atan2` is safer and standard in 3D graphics
- This ensures the 3D camera view matches the 2D overlay projection exactly!

**Applies to these world_translation modes:**
| World Translation | Camera Motion Effect |
|-------------------|---------------------|
| **None (Body at Origin)** | ✅ Camera rotates to show body at correct screen position |
| **Baked into Camera** | ✅ Camera animated with translation OR rotation |
| **Root Locator + Animated Camera** | ✅ Camera follows root with local animation |
| Baked into Mesh/Joints | ❌ Camera is static |
| Root Locator | ❌ Camera is static |
| Separate Track | ❌ Camera is static |

**Note**: "Root Locator + Animated Camera" is the recommended mode for shots where both character and camera move. The camera is parented to the root locator, so they move together while maintaining the screen-space relationship.

### Export Options
- **FPS Passthrough** - Source FPS flows from video loader through to export
- **Frame Offset** - Start animation at frame 1 (for Maya) or frame 0
- **Flip X** - Mirror animation on X axis (applies to mesh, skeleton, and root locator)
- **Up Axis** - Y, Z, -Y, -Z (configurable per DCC)

### Camera Export
- Focal length conversion from SAM3DBody pixel values to mm
- Configurable sensor width (Full Frame 36mm, APS-C 23.6mm, etc.)

## Installation

1. Clone or download to your ComfyUI `custom_nodes` directory
2. Requires Blender installed and accessible via PATH
3. Requires ComfyUI-SAM3DBody for mesh data

## Usage

### Recommended Workflow with SAM3

```
Video Loader (fps output) ──────────────────────────────────┐
      │                                                     │
      ├── SAM3 Video Segmentation                          │
      │         │                                           │
      │   SAM3 Propagate                                    │
      │         │ (masks: SAM3_VIDEO_MASKS)                 │
      │         │                                           │
      │         ↓                                           │
      └──→ Video Batch Processor ←──── sam3_masks          │
                │         │ ←───────────── fps ─────────────┘
                │         │
                ↓         ↓
        mesh_sequence    fps
                │         │
                ↓         ↓
        Export Animated FBX (fps=0 uses source fps)
```

### Node Inputs

#### Video Batch Processor
| Input | Type | Description |
|-------|------|-------------|
| model | SAM3D_MODEL | Loaded SAM3DBody model |
| images | IMAGE | Video frames |
| sam3_masks | SAM3_VIDEO_MASKS | Masks from SAM3 Propagate (optional) |
| fps | FLOAT | Source video FPS (passed through to export) |
| object_id | INT | Which object to track (default: 1) |
| bbox_threshold | FLOAT | Detection confidence threshold |

#### Export Animated FBX
| Input | Type | Description |
|-------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | From Video Batch Processor |
| filename | STRING | Output filename |
| fps | FLOAT | 0 = use source fps from mesh_sequence |
| frame_offset | INT | Start frame (1 for Maya, 0 for Blender) |
| output_format | FBX/ABC | Export format |
| up_axis | Y/Z/-Y/-Z | Up axis for target application |
| skeleton_mode | Rotations/Positions | Animation method |
| world_translation | None/Baked/Camera/Root/Separate | How to handle world movement |
| flip_x | BOOLEAN | Mirror animation on X axis |
| include_mesh | BOOLEAN | Include mesh in export |
| include_camera | BOOLEAN | Include camera in export |
| camera_motion | Translation/Rotation | How camera follows character (see Camera Motion Modes) |
| camera_smoothing | INT | Smoothing window for camera animation (0=none, 3=light, 5=medium, 9=heavy) |
| sensor_width | FLOAT | Camera sensor width in mm (36mm = Full Frame) |

### Verifying Correct Person Tracking

Use the **Verify Overlay (Sequence)** node to check tracking:

1. Connect original images and `mesh_sequence` to the node
2. Enable `show_joints`, `show_skeleton`, and optionally `show_mesh`
3. Output shows:
   - **Yellow box** - Detection bounding box
   - **Red dots** - Joint positions (numbered)
   - **Cyan lines** - Skeleton connections
   - **Yellow wireframe** - Mesh (if enabled)

If joints don't align with your masked person, check:
- `object_id` matches the SAM3 object (usually 1)
- SAM3 Propagate is connected directly (not through SAM3 Video Output)

## Quick Reference

### For Maya Import
```
frame_offset: 1        (animation starts at frame 1)
up_axis: Y             (Maya default)
output_format: FBX
```

### For Blender Import
```
frame_offset: 0        (animation starts at frame 0)
up_axis: Z             (Blender default)
output_format: FBX
```

### If Animation Appears Flipped/Mirrored
```
flip_x: true
```

### For Retargeting
```
skeleton_mode: Rotations (Recommended)
world_translation: None (Body at Origin)
```

### For Scene Recreation
```
world_translation: Baked into Mesh/Joints
```

### For Tripod/Handheld Camera Shots (Camera Pans/Tilts) ⭐
When the original camera rotates to follow the subject:
```
world_translation: Root Locator + Animated Camera
camera_motion: Rotation (Pan/Tilt)
camera_smoothing: 5                            (reduce jitter, adjust as needed)
include_camera: true
up_axis: Y                                     (for Maya)
flip_x: true                                   (test both values)
```

**How it works:**
- Root locator carries world translation
- Camera rotates (pan around Y, tilt around X) to frame the body
- Looking through the camera shows body at correct screen position
- Smoothing reduces frame-to-frame jitter in camera movement

### For Moving Camera Shots (Tracking/Dolly/Zoom) ⭐

When both the character and camera move, use the hybrid mode:

```
world_translation: Root Locator + Animated Camera
include_camera: true
camera_motion: Rotation (Pan/Tilt)
camera_smoothing: 5                            (adjust 0-15 as needed)
```

**Camera Motion Options:**

| Option | Best For | How It Works |
|--------|----------|--------------|
| **Translation (Default)** | Dolly/crane/steadicam shots | Camera moves laterally to frame character |
| **Rotation (Pan/Tilt)** | Tripod/handheld shots | Camera rotates (pans/tilts) to follow character |

**How it works internally:**
```
Frame N: pred_cam_t = [tx, ty, tz]  (character position relative to camera)

Root Locator position = world_offset(tx, ty, tz)  → Character's world path

Camera (parented to root):
  
  ROTATION mode:
    - LOCAL rotation = pan/tilt angles from (tx, ty)
    - pan_angle  = atan(tx * 0.5)   → horizontal rotation
    - tilt_angle = atan(ty * 0.5)   → vertical rotation
    - Like a camera operator rotating to follow subject
  
  TRANSLATION mode:
    - LOCAL position offset = inverse of (tx, ty)
    - Camera moves laterally to show character at offset
    - Like dolly/crane movement

Result:
- Root + Body move through world space
- Camera follows root with LOCAL animation
- From camera view: character at correct screen position
- From world view: character path visible, camera follows
```

**What you get:**
- Root locator shows character's world path
- Camera is parented to root locator, follows character movement
- Camera has LOCAL animation to show character at correct screen position
- In camera view: character at same position as in original video
- In world view: both character and camera move together
- Focal length is animated if it changes

**This works WITHOUT external camera tracking!**

**With External Camera Tracking (VFX workflow)**
If you have camera tracking data from SynthEyes, PFTrack, 3DEqualizer, etc.:
```
world_translation: None (Body at Origin)
include_camera: false
```
Then in Maya:
1. Import your tracked camera
2. Import the FBX
3. Parent/constrain the character to your scene

**Note on Camera Rotation**: SAM3DBody estimates body pose relative to camera view. For rotating cameras, the "Root Locator + Animated Camera" mode provides a reasonable approximation. For precise matchmoving with heavy camera rotation, use external camera tracking.

## Technical Details

### Camera Projection (How 2D Overlay = 3D Camera View)

SAM3DBody outputs these camera parameters:
```
focal_length: float (in pixels)
pred_cam_t: [tx, ty, tz]
  - tx: horizontal offset (where body appears in frame, -1 to +1)
  - ty: vertical offset (where body appears in frame, -1 to +1)
  - tz: depth (camera-to-body distance)
```

**2D Projection (verify_overlay.py):**
```
screen_x = focal_px * (X_3d / depth) + image_center_x + tx * depth * scale
screen_y = focal_px * (Y_3d / depth) + image_center_y + ty * depth * scale
```

**3D Camera Setup (blender_animated_fbx.py):**
```
# Match the same projection:
focal_mm = focal_px * sensor_width / image_width
sensor_height = sensor_width / aspect_ratio
render_resolution = (image_width, image_height)
camera_distance = tz
camera_target_offset = (tx * tz * 0.5, ty * tz * 0.5, 0)
```

When you look through the exported camera with the video as background, the mesh should align.

### Data Flow
```
SAM3DBody Process
    ↓
mesh_data (SAM3D_OUTPUT)          
    - vertices: [N, 3]            
    - faces: [F, 3]               
    - joint_coords: [127, 3]      
    - joint_rotations: [127, 3, 3]
    - pred_cam_t: [3]
    - focal_length
    ↓                             
Mesh Data Accumulator
    ↓
MESH_SEQUENCE
    ↓
Export Animated FBX
    ↓
.fbx / .abc file
```

### Skeleton Hierarchy
The exported armature uses proper parent-child bone relationships from MHR's `joint_parents` array:
```
Skeleton (Armature)
└── joint_000 (root - pelvis)
    ├── joint_001 (spine)
    │   ├── joint_002 (chest)
    │   │   └── joint_003 (neck)
    │   │       └── joint_004 (head)
    │   └── ...
    ├── joint_010 (left hip)
    │   └── joint_011 (left knee)
    │       └── joint_012 (left ankle)
    └── ...
```

### Joint Rotation Data
SAM3DBody uses Meta's MHR (Momentum Human Rig) body model:
- 127 joints with 3x3 rotation matrices
- Global (world-space) rotations per joint
- Converted to local rotations using parent hierarchy
- Quaternion interpolation for smooth animation

### Shape Key Animation
Each frame creates a shape key with value keyframed:
- Frame N-1: value = 0 (fade in)
- Frame N: value = 1 (active)
- Frame N+1: value = 0 (fade out)
- Last frame stays at 1 (no fade out)

## Changelog

### v3.2.9
- **FIX**: Camera tilt direction corrected
  - ty > 0 (body lower in frame) now correctly tilts camera DOWN
  - Previously was inverting ty which caused geo to appear above video frame
  - Pan direction unchanged (was already working)

### v3.2.8
- **CRITICAL FIX**: Camera rotation now correctly matches geometry coordinate transform
  - Root cause: geometry uses `(x, -y, -z)` for Y-up but camera used raw `(tx, ty)`
  - Now camera applies same transform: `ty_cam = -ty` to match geometry
  - This fixes the "geo out of frame due to tilt" issue
- **NEW**: `camera_smoothing` parameter to reduce camera jitter
  - Values: 0=none, 3=light, 5=medium, 9=heavy, up to 15
  - Applies moving average smoothing to camera translation values
- Removed confusing `pan_sign`/`tilt_sign` variables - coordinate transform handles direction
- Better console output showing smoothing status

### v3.2.7
- **CRITICAL FIX**: Camera rotation now uses correct projection math
  - Previously: `angle = atan(tx * 0.5)` - arbitrary scale factor ✗
  - Now: `angle = atan2(tx, tz)` - matches SAM3DBody's projection model ✓
  - Using `atan2` for safety and 3D graphics standard practice
  - This ensures 3D camera view matches 2D overlay exactly!

### v3.2.6
- **FIX**: Corrected camera pan/tilt direction for Y-up (Maya)
  - Previously: tilt was inverted, causing mesh to appear too low/high
  - Now: ty > 0 (body below center) → camera tilts UP → origin appears LOWER ✓
  - Now: tx > 0 (body on right) → camera pans LEFT → origin appears RIGHT ✓
- All three rotation-enabled modes now use consistent, corrected math

### v3.2.5
- **NEW**: Camera rotation mode now works with "None (Body at Origin)"
  - Previously only worked with "Baked into Camera" and "Root Locator + Animated Camera"
  - Now you can keep body at origin while camera pans/tilts to frame it correctly
- **FIX**: Corrected pan/tilt axis signs for proper camera rotation direction
  - Character on RIGHT → camera pans RIGHT (negative Y rotation)
  - Character BELOW center → camera tilts DOWN (positive X rotation)
- **FIX**: Rotation mode camera starts at base position (no target offset)
  - Rotation handles framing, not translation
- **IMPROVED**: Separated translation vs rotation code paths for cleaner logic
- **IMPROVED**: Updated README with comprehensive camera motion documentation

### v3.2.2
- **FIX**: Camera up-axis now matches scene up-axis
  - Previously camera always used UP_Y regardless of setting
  - Now correctly uses UP_Y for Y/-Y axis, UP_Z for Z/-Z axis
  - Console shows: `Camera static at ..., up_axis=Y`

### v3.2.1
- **FIX**: Camera now uses SAME parameters as 2D overlay for accurate 3D matching
  - Uses `pred_cam_t` [tx, ty, tz] directly (not computed from bbox)
  - Sets sensor_height to match video aspect ratio
  - Sets render resolution to match video dimensions
  - Correct scale factor for target offset calculation
- Camera should now match the 2D overlay projection in Maya/Blender viewport

### v3.2.0
- **NEW**: `camera_motion` option with two modes:
  - **Translation (Default)**: Camera moves laterally to frame character (dolly/crane behavior)
  - **Rotation (Pan/Tilt)**: Camera pans/tilts to follow character (tripod/handheld behavior)
- Choose based on how the original camera moved in your footage
- Works with "Root Locator + Animated Camera" mode for best results
- Console shows: `Using PAN/TILT rotation` or `Using local TRANSLATION`

### v3.1.9
- **FIX**: Camera in "Root Locator + Animated Camera" now has LOCAL animation
  - Camera parented to root_locator for world movement
  - Camera LOCAL position animated with inverse offset
  - Result: Camera world X,Y stays near origin while body moves
  - In camera view: Character appears at correct screen position (not always centered)
  - In world view: Both character and camera move together

### v3.1.8
- **NEW**: "Root Locator + Animated Camera" mode for moving camera shots
  - Camera parented to root locator - both move together
  - Character path visible in world space
  - Screen-space relationship preserved in camera view
  - Works without external camera tracking!
- Updated documentation with workflow recommendations

### v3.1.7
- **NEW**: Animated focal length support
  - Detects if focal length changes across frames
  - Automatically animates camera lens if variation > 1px
  - Console shows: `Animating focal length: 1200px to 1800px`
- Improved camera alignment for off-center detections

### v3.1.6
- **NEW**: Camera alignment using bbox position for accurate Maya/Blender viewport match
  - Camera target offset computed from detection bbox vs image center
  - Exports actual image dimensions from video
  - When looking through camera with video background, mesh should align
- **IMPROVED**: Frame data now includes `image_size` and `bbox` for alignment calculations

### v3.1.5
- **CRITICAL FIX**: SAM3 Propagate frame-indexed dict format now correctly detected
  - Previously only using 1 mask frame, now correctly stacks all 150 frames
  - Console now shows: `Stacked 150 frames into shape (150, 720, 1280)`
- Improved mask format detection with cleaner debug output
- Fixed variable name inconsistencies in mask extraction

### v3.1.4
- **FIX**: Last blendshape stays at value 1 (was being set to 0)
- **FIX**: Multi-person filtering uses mask overlap to select correct detection
- **IMPROVED**: Mesh overlay alignment using centroid matching
- Better debug output for alignment diagnostics

### v3.1.3
- **NEW**: `frame_offset` parameter (default: 1 for Maya)
- **NEW**: `flip_x` applies to root locator translation
- **FIX**: Camera only animated when "Baked into Camera" selected
- **FIX**: All other world translation modes create static camera

### v3.1.1
- **NEW**: FPS passthrough from video loader to export
- **NEW**: `flip_x` option to mirror animation
- **IMPROVED**: Skeleton connections use `joint_parents` from MHR
- Removed unused `mask` input, cleaned up Video Batch Processor

### v3.1.0
- **NEW**: `SAM3_VIDEO_MASKS` input type for SAM3 Propagate connection
- **NEW**: 🔍 Verify Overlay node - project mesh/skeleton onto image
- **NEW**: Rotation-based skeleton animation using MHR joint rotation matrices
- **NEW**: Proper hierarchical bone structure using `joint_parents`
- `skeleton_mode` option: "Rotations" vs "Positions"
- Quaternion interpolation for smooth animation

### v3.0.0
- World translation modes (5 options)
- Camera "Baked into Camera" mode with animated position
- Skeleton uses armature bones instead of empties
- Up axis options (Y, Z, -Y, -Z)

### v2.0.0
- Alembic export support
- Camera export with focal length
- Multiple people support

### v1.0.0
- Initial release with FBX export

## Requirements

- ComfyUI
- ComfyUI-SAM3DBody
- Blender 3.6+ (system installation)

## Troubleshooting

### "Only 1 mask available" in console
You may have SAM3 Video Output node between SAM3 Propagate and Video Batch Processor. Connect SAM3 Propagate → masks directly to Video Batch Processor → sam3_masks.

### Wrong person being tracked
Check `object_id` matches the SAM3 tracked object. Use Verify Overlay to visualize tracking.

### Animation appears flipped
Enable `flip_x = true` in Export node.

### Keyframes start at wrong frame
Adjust `frame_offset` (1 for Maya, 0 for Blender).

### Character not aligned in camera view
1. Make sure `camera_motion` matches your footage:
   - Tripod/static camera → **Rotation (Pan/Tilt)**
   - Dolly/crane/steadicam → **Translation (Default)**
2. Check `up_axis` matches your application (Y for Maya, Z for Blender)
3. Try toggling `flip_x` if character appears mirrored
4. For "Baked into Camera" or "None (Body at Origin)" modes, the character should be at/near origin

### Camera pointing wrong direction
1. Verify `up_axis` is correct for your DCC:
   - Maya: `Y`
   - Blender: `Z`
2. Check console output for camera setup messages:
   - `Camera using ROTATION (Pan/Tilt)...` → rotation mode active
   - `Camera using TRANSLATION...` → translation mode active
   - `Camera static at...` → no animation

### Camera animation has jitter
1. Increase `camera_smoothing` value:
   - `0` = no smoothing (raw data)
   - `3` = light smoothing
   - `5` = medium smoothing (recommended starting point)
   - `9` = heavy smoothing
   - `15` = maximum smoothing
2. Console shows: `Applied camera smoothing (window=5)`
3. Higher values = smoother but may lose subtle camera movements

### Camera rotation seems inverted
The camera should rotate TOWARD the character:
- Character on RIGHT → camera pans RIGHT
- Character BELOW → camera tilts DOWN

If inverted, try toggling `flip_x`.

## License

MIT License
