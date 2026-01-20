# SAM-3D-Body Coordinate System

This document explains the coordinate system used by Meta's SAM-3D-Body and how SAM3DBody2abc converts it for FBX export.

## SAM-3D-Body Output Coordinates

SAM-3D-Body uses a **weak perspective camera model** with image-aligned coordinates.

### Camera Translation (pred_cam_t)

```
pred_cam_t = [tx, ty, tz]
```

| Component | Meaning | Sign Convention |
|-----------|---------|-----------------|
| `tx` | Horizontal offset from image center | Positive = body to the RIGHT |
| `ty` | Vertical offset from image center | Positive = body BELOW center (IMAGE Y-down) |
| `tz` | Depth (camera distance from body) | Always positive |

### Screen Projection Formula

To project a 3D point to screen coordinates:

```python
screen_x = focal_length * tx / tz + image_width / 2
screen_y = focal_length * ty / tz + image_height / 2
```

### Mesh Vertices

- Body-relative coordinates (roughly centered at pelvis)
- Newer SAM-3D-Body versions may use ground-centered mesh
- Scale is in meters

### Skeleton Joints

- 70 joints (MHR70 format)
- Typically pelvis-centered
- Includes body, hands, feet, and face joints

## FBX Export Coordinates (Y-up World Space)

For FBX/Maya/Blender, we convert to Y-up world space:

```
world_x = tx              # Horizontal stays the same
world_y = -ty             # FLIP: image Y-down → world Y-up
world_z = 0 or tz         # Body at Z=0, camera at Z=tz
```

### Coordinate Transformation

```python
def cam_t_to_world(pred_cam_t, up_axis="Y"):
    tx, ty, tz = pred_cam_t
    
    if up_axis == "Y":
        # Y-up (Maya, Blender default)
        return (tx, -ty, 0)
    elif up_axis == "Z":
        # Z-up (some game engines)
        return (tx, 0, -ty)
```

## Mesh-to-Skeleton Alignment

### The Problem

Newer SAM-3D-Body versions use different reference points:
- **Mesh**: Ground-centered (feet at Y=0)
- **Skeleton**: Pelvis-centered (pelvis at origin)

This causes a vertical offset between mesh and skeleton.

### The Solution

SAM3DBody2abc calculates and applies the alignment offset:

```python
pelvis = joints[0]  # Pelvis is joint 0
mesh_center = mean(vertices)
offset = mesh_center - pelvis

# Apply offset to align mesh to skeleton
aligned_vertices = vertices - offset
```

### FBX Export Option

Enable `align_mesh_to_skeleton` (default: True) in the FBX Export node to automatically handle this.

## Debugging Coordinate Issues

### Check First Frame Output

The Video Batch Processor logs this information for the first frame:

```
Mesh center 3D: X=0.0123, Y=0.4567, Z=0.0089
Pelvis (joint 0) 3D: X=0.0001, Y=0.0002, Z=0.0001
Mesh vs Pelvis offset: dX=0.0122, dY=0.4565, dZ=0.0088
pred_cam_t: tx=0.1234, ty=-0.5678, tz=5.2345
```

### Expected Values

| Value | Typical Range | Notes |
|-------|---------------|-------|
| `tz` | 3.0 - 10.0 | Camera distance in meters |
| `tx` | -1.0 - 1.0 | Horizontal offset |
| `ty` | -1.0 - 1.0 | Vertical offset (IMAGE Y-down) |
| Mesh Y offset | 0.4 - 0.6 | If ground-centered mesh |

### Common Issues

1. **Character upside down**: Check `up_axis` setting, try "-Y" or "Z"
2. **Character offset vertically**: Enable `align_mesh_to_skeleton`
3. **Character moves wrong direction**: Check if `ty` negation is applied correctly

## MHR70 Joint Layout

Key joints for reference:

| Index | Name | Description |
|-------|------|-------------|
| 0 | Pelvis/Hips | Root joint, skeleton origin |
| 5 | Left Shoulder | Upper arm start |
| 6 | Right Shoulder | Upper arm start |
| 11 | Left Hip | Upper leg start |
| 12 | Right Hip | Upper leg start |
| 15 | Left Ankle | Foot reference |
| 16 | Right Ankle | Foot reference |
| 63 | Neck | Head/neck connection |

## Version History

### v5.4.0 (Current)
- Direct integration with Meta's SAM-3D-Body
- Documented coordinate system
- Fixed mesh-to-skeleton alignment

### v5.2.0
- Added `align_mesh_to_skeleton` option
- Improved coordinate handling for new SAM-3D-Body

### v5.1.x
- Initial support for changed coordinate system
- Added `flip_ty` option (now deprecated, automatic)
