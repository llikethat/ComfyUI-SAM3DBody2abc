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

| Output | Type | Description |
|--------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | Accumulated mesh data for export |
| debug_images | IMAGE | Debug visualization frames |
| frame_count | INT | Number of processed frames |
| status | STRING | Processing status message |
| fps | FLOAT | Source FPS (passthrough) |
| **focal_length_px** | FLOAT | **NEW** Focal length from SAM3DBody (for Camera Solver) |

#### Export Animated FBX
| Input | Type | Description |
|-------|------|-------------|
| mesh_sequence | MESH_SEQUENCE | From Video Batch Processor |
| filename | STRING | Output filename |
| **camera_rotations** | CAMERA_ROTATION_DATA | **NEW** From Camera Rotation Solver (optional) |
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

#### Camera Rotation Solver (v3.4.0)
| Input | Type | Description |
|-------|------|-------------|
| images | IMAGE | Video frames |
| foreground_masks | MASK | Foreground masks (optional, overrides YOLO) |
| sam3_masks | SAM3_VIDEO_MASKS | SAM3 masks (optional, overrides YOLO) |
| tracking_method | DROPDOWN | KLT (Persistent), CoTracker (AI), ORB, or RAFT |
| auto_mask_people | BOOLEAN | Auto-detect all people using YOLO (default: True) |
| detection_confidence | FLOAT | YOLO detection confidence (default: 0.5) |
| mask_expansion | INT | Expand masks by pixels (default: 20) |
| focal_length_px | FLOAT | Focal length in pixels (default: 1000) |
| flow_threshold | FLOAT | Min flow magnitude - RAFT only (default: 1.0) |
| ransac_threshold | FLOAT | RANSAC threshold (default: 3.0) |
| smoothing | INT | Temporal smoothing window (default: 5) |

| Output | Type | Description |
|--------|------|-------------|
| camera_rotations | CAMERA_ROTATION_DATA | Per-frame pan/tilt/roll values |
| debug_masks | MASK | Visualization of detected foreground |
| debug_tracking | IMAGE | Visualization of tracked points (green=inliers) |

**Tracking Methods:**

| Method | Speed | GPU | Occlusion | Best For |
|--------|-------|-----|-----------|----------|
| **KLT (Persistent)** | ⚡ Fast | ❌ CPU | ❌ Dies | Most footage (default) |
| **CoTracker (AI)** | 🐢 Slow | ✅ GPU | ✅ Handles | Difficult shots, crossing paths |
| ORB (Feature-Based) | ⚡ Fast | ❌ CPU | ❌ Per-frame | Backup option |
| RAFT (Dense Flow) | 🐢 Slow | ✅ GPU | ❌ Per-frame | Slow/detailed motion |

**Recommended: KLT (Persistent)** - Uses professional-style persistent tracking like PFTrack/SynthEyes:
- Detects features in frame 0, tracks them across entire video
- Estimates rotation relative to frame 0 (no drift!)
- Fast (3 seconds for 50 frames)

**Why this node exists:**

`pred_cam_t` from SAM3DBody tells us WHERE the body appears on screen, but it cannot distinguish between:
- Body moving right in the world
- Camera panning left
- Both happening together

This causes misalignment when reconstructing 3D camera motion. The Camera Rotation Solver analyzes **background motion** (excluding people) to determine **actual camera rotation**.

**How it works:**
1. Detects ALL people using YOLO (automatic, no setup needed)
2. Inverts masks to isolate background (bounding boxes are fine!)
3. Tracks background features persistently (KLT) or with AI (CoTracker)
4. Estimates rotation relative to frame 0 using Essential Matrix
5. Decomposes to pan/tilt/roll with consistent Euler angles

**Simplest usage (fully automatic):**
```
Video Frames → Camera Rotation Solver → camera_rotations
               (tracking_method=KLT, auto_mask_people=True)
```

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

### v3.5.20 - External Tracking Node Integration
Added support for external ComfyUI tracking nodes:

**New Inputs:**
- `cotracker_coords` - Accept tracking data from `comfyui_cotracker_node` (s9roll7)
- `cotracker_visibility` - Visibility mask from CoTracker
- `verbose_debug` - Enable detailed per-frame debug output

**Compatible External Nodes:**
| Node | GitHub | Purpose |
|------|--------|---------|
| ComfyUI-DepthCrafter-Nodes | akatz-ai | Temporally consistent video depth |
| comfyui_cotracker_node | s9roll7 | AI point tracking |
| ComfyUI-DUSt3R | logtd | 3D reconstruction |
| ComfyUI-dust3r | chaojie | 3D reconstruction |
| DepthAnything V2 | (various) | Per-frame depth estimation |

**Workflow Examples:**

1. **DepthCrafter + Camera Solver:**
```
Load Video → DepthCrafter → Camera Rotation Solver (depth_maps input)
```

2. **CoTracker + Camera Solver:**
```
Load Video → CoTrackerNode → Camera Rotation Solver (cotracker_coords input)
```

### v3.5.19 - Manual Camera Data Input
Added two new nodes for importing external camera data:

1. **Manual Camera Data** - Create camera rotation data by entering values directly
   - Input total pan/tilt/roll in degrees
   - Specify motion start/end frames
   - Choose interpolation: Linear, Ease In/Out, Hold After End, Constant
   - Perfect for when you've solved the camera in Maya/3DEqualizer/SynthEyes

2. **Camera Data from JSON** - Import camera data from JSON file or string
   - Load per-frame rotation data from external tracking software
   - JSON format: `{"frames": [{"frame": 0, "pan": 0, "tilt": 0, "roll": 0}, ...]}`
   - Rotation values in degrees

Example workflow:
```
Manual Camera Data (pan=3.686°, end_frame=24) → FBX Export (camera_rotations)
```

### v3.5.18 - Rewrite Depth-KLT with Proper Filtering
- **REWRITE**: Complete rewrite of Depth-KLT algorithm
  - Now uses **persistent tracking from frame 0** (like regular KLT) instead of frame-to-frame
  - **Hard depth filtering**: Keeps only top 40% features by depth (distant = background)
  - Uses same Essential Matrix approach as working regular KLT
  - No more accumulated delta drift
- Previous version had random weighted sampling which caused catastrophic failures (pan=-236°!)

### v3.5.17 - Fix DepthAnything+KLT Dispatch
- **BUGFIX**: Fixed tracking method dispatch - "DepthAnything + KLT" was incorrectly using regular KLT
  - "DepthAnything + KLT" contains "KLT", so it matched regular KLT first
  - Now correctly routes to depth-weighted tracking when selected

### v3.5.16 - External Depth Maps Input
- **NEW**: Added `depth_maps` input to Camera Rotation Solver
  - Connect output from DepthAnything V2, MiDaS, ZoeDepth, or any depth node
  - When connected, uses external depth instead of loading internal model
  - Supports various tensor shapes: (N, H, W), (N, H, W, C), (N, C, H, W)
- **Workflow**: Run DepthAnything V2 node → Connect to Camera Solver depth_maps input → Select "DepthAnything + KLT" method

### v3.5.15 - Depth-Based Camera Tracking Methods
Added 4 new depth-based tracking methods for more robust camera rotation estimation:

1. **DepthAnything + KLT**: Uses depth to weight features - prioritizes distant (background) features
   - Auto-downloads MiDaS as fallback if DepthAnything unavailable
   - Features weighted by depth for better background tracking

2. **DUSt3R (3D Reconstruction)**: AI-based 3D reconstruction for direct camera poses
   - Requires: `pip install dust3r` or clone from https://github.com/naver/dust3r
   - Directly predicts relative camera poses between frames

3. **COLMAP (Structure from Motion)**: Traditional but robust SfM pipeline
   - Requires: COLMAP installed (https://colmap.github.io/install.html)
   - Full sparse reconstruction for camera poses

4. **DepthCrafter (Video Depth)**: Video-native temporally consistent depth
   - Requires: `pip install diffusers`
   - Tracks camera motion from depth gradient changes

All methods fall back to KLT if dependencies unavailable.

### v3.5.14 - Revert to Static Body Offset (Stable)
- **Reverted** animated body offset - compensation formula was incorrect
- **Static body_offset** from frame 0 restored (this worked perfectly in v3.5.10)
- Camera Solver outlier rejection retained from v3.5.13
- **Recommended workflow for camera pan videos:**
  1. Export with **Static** camera mode → Perfect frame 1 alignment
  2. Manually animate camera in Maya based on solver's reported total rotation
- **Future**: Will implement depth-based camera tracking for automatic pan handling

### v3.5.13 - Camera Solver Outlier Rejection
- **Critical Fix**: Added outlier rejection to Camera Solver
  - Detects frames with unreasonable rotation jumps (>10° between frames)
  - Detects frames with excessive absolute rotation (>45°)
  - Replaces outliers with linearly interpolated values from valid neighbors
- Example issue fixed: Frame 2 with tilt=178.99° (flip) now gets interpolated
- Outlier rejection runs BEFORE smoothing for cleaner results

### v3.5.12 - Animated Body Offset with Rotation Compensation
- **Major Fix**: Body offset is now ANIMATED when camera rotation is solved
  - Static offset only correct at frame 0
  - With camera pan/tilt, body position that gives correct alignment CHANGES per frame
- **Rotation compensation**: `body_x = tx + depth × tan(pan)`, `body_y = -ty - depth × tan(tilt)`
- **Stronger smoothing**: Body offset uses minimum smoothing of 5 to reduce jitter
- Both camera and body offset values are smoothed together for consistency

### v3.5.11 - Fix Camera Rotation with Root Locator Mode
- **BUG FIX**: camera_follow_root mode now properly uses solved_camera_rotations
  - Previously: Always computed pan/tilt from pred_cam_t (double-counting issue)
  - Now: Uses solved rotations from Camera Solver when available
- body_offset is STATIC (from frame 0) - handles initial positioning
- Solved camera rotations handle frame-to-frame pan/tilt
- This properly separates body alignment from camera movement

### v3.5.10 - Add Static Camera Option
- Added **"Static"** option to camera_motion_mode
  - Camera stays completely fixed (no rotation, no translation animation)
  - Only body_offset positions the body correctly
  - Simplest and most predictable alignment behavior
- This is the recommended mode for accurate body alignment with "Root Locator + Animated Camera"

### v3.5.9 - Fix Double Offset (Camera Target + Body Offset)
- **CRITICAL FIX**: Removed target_offset when using body_offset (root mode)
  - Previously: camera looked at target_offset AND body was at body_offset → DOUBLE OFFSET
  - Now: camera looks at origin (0,0,0), body at body_offset → SINGLE CORRECT OFFSET
  - This should fix the ~0.28 horizontal offset issue!

### v3.5.8 - Debug Build for X Offset Investigation
- Added detailed 3D center analysis debug output
  - Mesh center vs Joints center vs Pelvis
  - Should reveal why X offset is ~0.26 too far left

### v3.5.7 - Vertical Offset Sign Fix
- **CRITICAL FIX**: Negated ty in body offset calculation
  - SAM3DBody: ty positive = body above image center
  - Maya camera rotated -90° around X flips the vertical axis
  - Now: `-ty` in Blender → `+ty` in Maya camera view (correct alignment!)
- Both Alembic and FBX exports now align correctly without manual adjustment

### v3.5.6 - Body Offset Fix for Camera Alignment
- **CRITICAL FIX**: Separated root_locator and body offset
  - root_locator now stays at (0, 0, 0) - no offset applied
  - Body offset (tx, ty) applied directly to mesh and skeleton
  - This fixes alignment because root_locator moves both camera AND body together
- **CRITICAL FIX**: Removed Y negation in projection formula
  - SAM3DBody's pred_keypoints_3d are already in image-aligned coordinates
  - Projection now matches pred_keypoints_2d (dx=0, dy=0)
- **IMPROVED**: Body offset uses correct Maya coordinate mapping
  - Blender (X, Y, 0) → Maya (X, 0, Y) for camera view alignment
  - Maya X = horizontal in camera view
  - Maya Z = vertical in camera view (camera rotated -90° around X)

### v3.5.5 - Projection Debug Improvement
- **FIXED**: Debug comparison now properly uses `pred_keypoints_3d` (70 joints)
  - Apples-to-apples comparison with `pred_keypoints_2d` (70 joints)
  - Previously was comparing to `joint_coords` (127 joints) - different joint set!
- **IMPROVED**: Better debug output showing which 3D data source is used
- **IMPROVED**: Handles various shapes of keypoints data (squeezing if needed)
- **ADDED**: Success/failure indicator for projection validation

### v3.5.4 - Blender-to-Maya Coordinate Mapping Fix
- **CRITICAL FIX**: Fixed coordinate axis mapping for Y-up export to Maya
  - When Blender exports with up_axis="Y":
    - Blender X → Maya X (horizontal)
    - Blender Y → Maya Z (depth)
    - Blender Z → Maya Y (vertical)
  - Previous: `Vector((world_x, world_y, 0))` → world_y went to Maya Z (wrong!)
  - Fixed: `Vector((world_x, 0, world_y))` → world_y goes to Maya Y (correct!)
- Root locator now positions body at correct vertical position in Maya
- Debug output updated to show coordinate mapping

### v3.5.3 - Root Locator Depth Scaling Fix
- **CRITICAL FIX**: Removed incorrect `* tz * 0.5` depth scaling from world offset
  - Previous formula: `world_y = ty * |tz| * 0.5` → gave Y=2.21 (way too high!)
  - New formula: `world_y = ty` → gives Y=0.81 (closer to correct)
  - tx/ty from pred_cam_t are already in world units - no depth scaling needed!
- Root locator now positions body much closer to correct vertical position
- Debug output updated to show new formula

### v3.5.1 - X-Axis Sign Fix & Debug Improvements
- **FIX**: Fixed X-axis sign in `get_world_offset_from_cam_t()`
  - tx negative (body LEFT of center) → world X negative (correct!)
  - Previously was being negated incorrectly
- **FIX**: Debug comparison now uses `pred_keypoints_3d` (70 joints) instead of `joint_coords` (127 joints)
  - Apples-to-apples comparison with `pred_keypoints_2d`
  - Now properly validates projection formula
- **ADDED**: `pred_keypoints_3d` stored in mesh_sequence for projection validation
- **IMPROVED**: Debug output now shows body position relative to center

### v3.5.0 - Y-Axis Projection Fix
- **CRITICAL FIX**: Fixed Y-axis projection for 3D to 2D conversion
  - 3D Y points UP, image Y points DOWN - now correctly negated
  - Mesh overlay now aligns correctly with video
  - Root locator Y position now correct
- **DEBUG**: Added projection comparison debug output
  - Shows ground truth vs calculated 2D positions
  - Helps verify projection accuracy
- **DEBUG**: Added root_locator calculation debug output
  - Shows pred_cam_t values and world offset calculation
- **ADDED**: `requirements.txt` for easy dependency installation
- **ADDED**: ultralytics dependency for YOLO person detection

**Test Results:**
Before fix: Average Y offset = +211px (body appeared too low)
After fix: Average Y offset ≈ 0px (body aligns correctly)

### v3.4.1 - Full Pipeline Integration
- **NEW**: Camera Rotation Solver → Export FBX integration
  - Connect `camera_rotations` output directly to Export Animated FBX
  - Solved rotations automatically override camera animation
  - Forces Rotation mode when solved rotations are provided
- **NEW**: Video Batch Processor outputs `focal_length_px`
  - Connect directly to Camera Rotation Solver's `focal_length_px` input
  - No more guessing focal length!
- **IMPROVED**: Console output shows when using solved rotations
  - `[Export] Using solved camera rotations from Camera Rotation Solver (50 frames)`
  - `[Blender] Camera uses SOLVED rotation over 50 frames`

**Complete Workflow:**
```
Video Loader ──┬──→ Video Batch Processor ──┬──→ mesh_sequence ──→ Export Animated FBX
               │                            │                            ↑
               │                            └──→ focal_length_px ────────┤
               │                                                         │
               └──→ Camera Rotation Solver ──→ camera_rotations ─────────┘
```

### v3.4.0 - Camera Solver Fixes & Improvements
- **FIX**: Frame 0 now always (0,0,0) - smoothing no longer corrupts reference frame
- **FIX**: CoTracker GPU device initialization (was showing `None`, now correctly uses `cuda`)
- **FIX**: Consistent Euler angle extraction across all tracking methods
  - Uses scipy.spatial.transform.Rotation when available (more robust)
  - Fallback to manual ZYX convention extraction
  - All methods (KLT, CoTracker, ORB, RAFT) now produce consistent rotation values
- **IMPROVED**: Roll values now correctly near-zero for level tripod shots
- **IMPROVED**: Smoothing preserves frame 0 as reference, only smooths frames 1+

**Tested Results on 50-frame sprint footage:**
| Method | Speed | Inliers | Quality |
|--------|-------|---------|---------|
| KLT (Persistent) | 2.95s ⚡ | 80-98% | ✅ Great |
| CoTracker (AI) | 123s | 95-99% | ✅ Excellent |

### v3.3.9
- **FIX**: Corrected CoTracker installation docs - no pip install needed
  - CoTracker auto-downloads via `torch.hub` on first use
  - Just select "CoTracker (AI)" and it will download automatically
  - Requires internet connection on first use only

### v3.3.8
- **NEW**: CoTracker (Meta AI) integration for state-of-the-art point tracking
  - Handles occlusion (tracks points through obstacles)
  - GPU accelerated using PyTorch
  - More robust on fast motion and motion blur
  - Auto-downloads model on first use via torch.hub (~200MB)
  - Falls back to KLT if CoTracker unavailable
- **NEW**: 4 tracking methods available:
  - "KLT (Persistent)" - Professional CPU-based (default, recommended)
  - "CoTracker (AI)" - Meta's AI tracker (GPU, handles occlusion)
  - "ORB (Feature-Based)" - Sparse matching per frame
  - "RAFT (Dense Flow)" - Dense optical flow
- **NOTE**: CoTracker downloads automatically - no pip install needed!

### v3.3.7
- **NEW**: KLT Persistent Tracking (Professional Method) - now DEFAULT
  - Mimics PFTrack/SynthEyes/3DEqualizer approach
  - Detects features in frame 0 and tracks them persistently
  - Estimates rotation relative to frame 0 (no drift accumulation!)
  - Uses Essential Matrix decomposition for pure rotation
  - Much more robust for broadcast sports footage
- **NEW**: `tracking_method` dropdown with 3 options:
  - "KLT (Persistent)" - Professional style, recommended (default)
  - "ORB (Feature-Based)" - Sparse feature matching per frame pair
  - "RAFT (Dense Flow)" - Dense optical flow
- **IMPROVED**: Essential Matrix decomposition instead of Homography for rotation

### v3.3.6
- **NEW**: ORB feature-based tracking method (default)
  - More robust for fast camera motion and motion blur
  - Uses sparse feature matching instead of dense optical flow
  - Should work much better on broadcast sports footage
- **NEW**: `tracking_method` parameter
  - "ORB (Feature-Based)" - recommended for most cases
  - "RAFT (Dense Flow)" - better for slow/detailed motion
- **IMPROVED**: Better debug visualization for ORB matches

### v3.3.5
- **FIX**: SAM3 mask processing - handles various tensor shapes correctly
- **FIX**: Falls back to YOLO when SAM3 mask processing fails
- **FIX**: YOLO detection debug logging scope issue
- **IMPROVED**: Extensive debug output for SAM3 mask format detection
- **IMPROVED**: Shows mask processing success/failure and fallback status

### v3.3.4
- **FIX**: RGB to BGR conversion for YOLO detection
- **IMPROVED**: Extensive debug logging to diagnose detection issues
  - Shows input frame statistics (shape, dtype, value range)
  - Shows per-frame detection counts
  - Shows frame 0 detection details (class, confidence, bbox)
  - Shows mask sum and debug tensor statistics
- **IMPROVED**: Fixed color channels in debug visualization (RGB consistency)

### v3.3.3
- **NEW**: Debug outputs in Camera Rotation Solver
  - `debug_masks` (MASK): Shows YOLO-detected foreground masks
  - `debug_tracking` (IMAGE): Shows tracked points with flow vectors
    - Green = inlier points (used for homography)
    - Red = outlier points (rejected)
    - Blue overlay = background region (used for tracking)
- **NEW**: Sanity check - rejects rotation deltas > 10° per frame
- **IMPROVED**: More detailed logging showing point counts at each stage

### v3.3.2
- **NEW**: YOLO auto-masking in Camera Rotation Solver
  - Automatically detects and masks ALL people in video
  - No manual BBox or SAM3 masks needed
  - Uses YOLOv8 nano (fast, ~6MB model)
  - Parameters: `auto_mask_people`, `detection_confidence`, `mask_expansion`
- Requires: ultralytics (`pip install ultralytics`)

### v3.3.1
- **FIX**: Camera Rotation Solver variable naming collision
  - `H` was used for both image height and homography matrix
  - Renamed to `img_height`, `img_width`, `homography` for clarity
  - Fixes "inhomogeneous shape" error

### v3.3.0
- **NEW**: Camera Rotation Solver node
  - Estimates actual camera rotation (pan/tilt/roll) from background motion
  - Uses RAFT optical flow (GPU accelerated) on background regions
  - Masks out foreground using SAM3 masks
  - Solves the fundamental issue: pred_cam_t can't distinguish body movement from camera rotation
  - Output can be used to improve 3D camera alignment
- Requires: torchvision >= 0.14 (for RAFT), opencv-python

### v3.2.11
- **REVERT**: Restored v3.2.9 camera rotation logic
  - v3.2.10 changes made alignment worse
  - Back to simpler approach that was closer to correct
- Still investigating remaining ~0.6 unit vertical offset

### v3.2.10
- **CRITICAL FIX**: Camera base rotation now correctly established for rotation mode
  - Bug: Static camera section pointed at OFFSET target, then rotation added MORE offset
  - Fix: For rotation mode, camera now points at ORIGIN first, then adds pan/tilt
  - Added debug output showing camera base rotation and frame 0 values
- **FIX**: Restored correct tilt sign (ty_cam = -ty)
  - SAM3DBody: ty > 0 = body LOWER in frame (image Y points down)
  - Maya Y-up: to show body lower, camera must tilt UP (negative X rotation)
  - Therefore tilt_angle = atan2(-ty, depth)

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

### Python Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy torch torchvision opencv-python ultralytics scipy
```

**Camera Rotation Solver specific:**
- `torchvision >= 0.14` - For RAFT optical flow
- `opencv-python` - For KLT tracking and image processing
- `ultralytics` - For YOLO person detection
- `scipy` - For rotation matrix decomposition

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

### Third-Party Licenses

This project uses the following open-source libraries:

| Library | License | Usage |
|---------|---------|-------|
| **NumPy** | BSD-3-Clause | Array operations |
| **PyTorch** | BSD-3-Clause | Deep learning framework |
| **TorchVision** | BSD-3-Clause | RAFT optical flow |
| **OpenCV** | Apache-2.0 | Image processing, KLT tracking |
| **Ultralytics (YOLOv8)** | AGPL-3.0 | Person detection |
| **SciPy** | BSD-3-Clause | Rotation matrix decomposition |
| **CoTracker** | CC-BY-NC-4.0 | AI point tracking (Meta Research) |
| **Blender** | GPL-2.0+ | FBX/Alembic export |

**Note on Ultralytics/YOLOv8**: The AGPL-3.0 license requires that if you distribute a modified version of this software, you must also distribute the source code. For commercial use, consider Ultralytics Enterprise license.

**Note on CoTracker**: CC-BY-NC-4.0 is for non-commercial use only. For commercial applications, contact Meta for licensing.

**Note on Blender**: This project calls Blender as an external tool via subprocess. The exported FBX/ABC files are not covered by GPL.
