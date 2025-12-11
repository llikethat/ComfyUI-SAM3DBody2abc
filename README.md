# ComfyUI-SAM3DBody2abc

Export SAM3DBody mesh sequences to animated FBX or Alembic files for use in Maya, Blender, and other 3D applications.

## Features

### Verification Overlay (v3.1.0)
- **🔍 Verify Overlay** - Project 3D mesh/skeleton back onto original image
- Helps verify the correct person is being tracked (not mixing with others)
- Shows joint positions and skeleton connections overlaid on the image
- Optional mesh wireframe visualization

### Export Formats
- **FBX** - Blend shapes for mesh animation, widely compatible
- **Alembic (.abc)** - Vertex cache for mesh animation, cleaner Maya workflow

### Skeleton Animation Modes (v3.1.0)
- **Rotations (Recommended)** - Uses true joint rotation matrices from MHR model
  - Proper bone rotations for retargeting to other characters
  - Standard animation workflow compatible
  - Better for editing in animation software
- **Positions (Legacy)** - Uses joint positions with location offsets
  - Shows exact joint positions
  - Limited retargeting capability

### World Translation Modes
- **None (Body at Origin)** - Character centered at origin, static camera
- **Baked into Mesh/Joints** - World offset baked into vertex and joint positions
- **Baked into Camera** - Body at origin, camera moves to preserve original framing
- **Root Locator** - Root empty carries translation, body/skeleton as children
- **Separate Track** - Body at origin + separate locator showing world path

### Camera Export
- Focal length conversion from SAM3DBody pixel values to mm
- Configurable sensor width (Full Frame 36mm, APS-C 23.6mm, etc.)
- Animated distance based on depth estimation

### Up Axis Options
- Y, Z, -Y, -Z (configurable per DCC application requirements)

## Installation

1. Clone or download to your ComfyUI `custom_nodes` directory
2. Requires Blender installed and accessible via PATH
3. Requires ComfyUI-SAM3DBody for mesh data

## Usage

### Basic Workflow
1. **Load Model** → SAM 3D Body: Load Model
2. **Process Frames** → SAM 3D Body: Process Image (loop over video frames)
3. **Accumulate** → Mesh Data Accumulator (collect frames into sequence)
   - Connect `mesh_data` output
   - Connect `skeleton` output (provides joint_parents hierarchy and rotations)
4. **Export** → Export Animated FBX

### Export Settings

#### Verifying Correct Person Tracking
Use the **🔍 Verify Overlay** node to check if SAM3DBody is tracking the correct person:

1. Connect your original image and `mesh_data` output to the Verify Overlay node
2. The output shows joints projected onto the image
3. If joints don't align with your masked person, the tracking may be mixed with another person

Options:
- `show_joints` - Draw joint positions as circles
- `show_skeleton` - Draw skeleton connections between joints
- `show_mesh` - Draw mesh wireframe (can be slow)
- Colors and sizes are customizable

#### Skeleton Mode
- **Rotations (Recommended)**: Uses the 127 joint rotation matrices output by MHR (Meta's body model). Produces proper bone rotations that can be retargeted to other character rigs and edited in animation software.
- **Positions (Legacy)**: Animates bones using location offsets from rest positions. Shows exact joint positions but limited for retargeting.

#### World Translation
Choose how character movement through space is handled:
- For retargeting: Use "None" or "Baked into Camera"
- For full scene recreation: Use "Baked into Mesh/Joints"
- For flexibility: Use "Root Locator" (translation on parent, animation on bones)

## Technical Details

### Data Flow
```
SAM3DBody Process
    ↓
mesh_data (SAM3D_OUTPUT)          skeleton (SKELETON)
    - vertices: [N, 3]                - joint_positions: [127, 3]
    - faces: [F, 3]                   - joint_rotations: [127, 3, 3]  ← rotations
    - joint_coords: [127, 3]          - joint_parents: [127]  ← hierarchy
    - joint_rotations: [127, 3, 3]    - pose_params, shape_params, etc.
    - camera: [3] pred_cam_t
    - focal_length
    ↓                                 ↓
    └──────────┬──────────────────────┘
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
    └── joint_015 (right hip)
        └── ...
```

### Joint Rotation Data
SAM3DBody uses Meta's MHR (Momentum Human Rig) body model internally. The `joint_rotations` output contains:
- 127 joints with 3x3 rotation matrices
- Global (world-space) rotations per joint
- Converted to local rotations using parent hierarchy
- Quaternion interpolation in Blender for smooth animation

For "Root Locator" mode:
```
root_locator (empty with translation keyframes)
└── Skeleton (Armature)
    └── (hierarchical bone structure as above)
```

## Changelog

### v3.1.0
- **NEW**: 🔍 Verify Overlay node - project mesh/skeleton onto image for verification
- **NEW**: Rotation-based skeleton animation using MHR joint rotation matrices
- **NEW**: Proper hierarchical bone structure using `joint_parents` from MHR
- Added `skeleton_mode` option: "Rotations (Recommended)" vs "Positions (Legacy)"
- Added optional `skeleton` input to accumulator (provides joint_parents and rotations)
- Bones are properly parented (child bones follow parent rotations)
- Global rotations converted to local for proper FK chain
- Bone tails point toward children for better visualization
- Quaternion interpolation in Blender for smooth animation
- Fixed numpy array boolean check error with pred_cam_t

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
- Blender 3.6+ (system installation or bundled with SAM3DBody)

## License

MIT License
