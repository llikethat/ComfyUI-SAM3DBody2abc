# Camera Calibration Setup Guide

## Overview

This folder contains calibration files for multi-camera triangulation.
Each JSON file defines camera positions, rotations, and intrinsics.

## Quick Start

1. Copy one of the example files as a starting point
2. Measure your camera positions in meters
3. Measure camera rotations (or estimate from viewing direction)
4. Update focal length from your camera/lens specs
5. Load the file in CameraCalibrationLoader node

## Measuring Your Setup

### 1. Choose a World Origin

Pick a reference point in your capture space:
- **Center of performance area** (recommended)
- Position of Camera A
- A marked point on the floor

The origin is where X=0, Y=0, Z=0.

### 2. Measure Camera Positions

For each camera, measure from the origin:

| Axis | Direction | Measure |
|------|-----------|---------|
| **X** | Right (+) / Left (-) | Distance sideways |
| **Y** | Up (+) / Down (-) | Height from floor |
| **Z** | Back (+) / Forward (-) | Distance toward/away |

**Example:**
```
Camera is 3 meters to the right, 1.5m high, 4m behind origin
Position: [3.0, 1.5, 4.0]
```

### 3. Measure Camera Rotations

Rotations are in degrees, using Euler angles (XYZ order):

| Rotation | Axis | Positive Direction |
|----------|------|-------------------|
| **Pitch (X)** | Left-right axis | Looking up |
| **Yaw (Y)** | Up-down axis | Looking right |
| **Roll (Z)** | Forward axis | Tilting clockwise |

**Common examples:**
```
Looking straight ahead:      [0, 0, 0]
Looking 45° to the left:     [0, -45, 0]
Looking 90° to the right:    [0, 90, 0]
Looking down 15°:            [-15, 0, 0]
```

### 4. Camera Intrinsics

Get from your camera specs or EXIF data:

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `focal_length_mm` | Lens focal length | 35mm, 50mm, 85mm |
| `sensor_width_mm` | Sensor size | 36mm (full-frame), 23.5mm (APS-C) |
| `resolution` | Video resolution | [1920, 1080], [3840, 2160] |

## Calibration File Format

```json
{
  "version": "1.0",
  "name": "My Setup Name",
  
  "coordinate_system": {
    "up": "Y",
    "forward": "-Z",
    "unit": "meters"
  },
  
  "cameras": {
    "camera_A": {
      "name": "Front Camera",
      "position": [X, Y, Z],
      "rotation_euler": [pitch, yaw, roll],
      "rotation_order": "XYZ",
      "focal_length_mm": 35.0,
      "sensor_width_mm": 36.0,
      "sensor_height_mm": 24.0,
      "resolution": [1920, 1080],
      "principal_point": [960, 540],
      "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0}
    },
    "camera_B": {
      ...
    }
  }
}
```

## Common Setups

### 90° L-Shape (Best for Depth)

```
Top View:
                    
        📷 B (Side)
        │
        │ 90°
        │
        └────────── Subject ────────── 📷 A (Front)
```

- **Best for:** Circular walks, movements in X-Z plane
- **Advantages:** Excellent depth (Z) triangulation
- **File:** `two_camera_90deg.json`

### 180° Front/Back

```
Top View:

        📷 B (Back)
              │
         [Subject]
              │
        📷 A (Front)
```

- **Best for:** Movements in X-Y plane
- **Advantages:** Good lateral coverage, full body visibility
- **Disadvantages:** Limited depth precision
- **File:** `two_camera_180deg.json`

### 60° Converging

```
Top View:

              [Subject]
             /    60°   \
           /              \
         📷 A            📷 B
```

- **Best for:** General purpose tracking
- **Advantages:** Good overall accuracy, redundant coverage

## Tips for Best Results

### Camera Placement

1. **Wider angles = better depth accuracy**
   - 90° between cameras is optimal for depth
   - Angles < 30° will have poor depth precision

2. **Same height recommended**
   - Simplifies setup
   - Reduces occlusion issues

3. **Overlapping field of view**
   - Both cameras must see the subject at all times
   - Account for subject movement range

### Synchronization

- Videos must be frame-synchronized
- Use timecode, clap sync, or audio sync
- Same frame rate on all cameras

### Resolution Matching

- Same resolution on all cameras is recommended
- Different resolutions can work but may reduce accuracy

## Validation

After creating your calibration:

1. Load in CameraCalibrationLoader node
2. Check the calibration_info output for geometry details
3. Verify baseline distance matches your measurement
4. Check angle between views is as expected
5. Test with a simple scene before full capture

## Troubleshooting

### "Small angle between cameras" warning
- Cameras are too close to parallel
- Move cameras further apart in angle
- Depth accuracy will be limited

### High triangulation error
- Check camera positions are correct
- Verify focal lengths match actual lenses
- Ensure videos are properly synchronized

### Joints not visible in one camera
- Subject may be occluded
- Check camera placement covers full motion range
- Consider adding a third camera

## Example Workflow

1. Set up cameras at approximately 90° angle
2. Measure camera positions from center of capture area
3. Note camera heights and rotation angles
4. Record synchronized video from both cameras
5. Create calibration JSON file
6. Run both videos through SAM3DBody pipeline
7. Feed mesh sequences to MultiCameraTriangulator
8. Verify triangulation quality in debug output
9. Export jitter-free FBX

## Default Calibration Folder

Calibration files can be placed in:
```
/workspace/ComfyUI/models/calibrations/
```

The CameraCalibrationLoader will search this folder by default.
