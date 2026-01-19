# Changelog

All notable changes to ComfyUI-SAM3DBody2abc will be documented in this file.

## [5.3.0] - 2026-01-19

### Added
- **Direct bpy module support** for FBX export (no Blender subprocess needed)
  - New `lib/bpy_exporter.py` with complete export functionality (~900 lines)
  - Automatically detects if bpy is available and uses it
  - Falls back to Blender subprocess if bpy not installed
  - Supports all existing features: mesh animation, skeleton, camera, locators
  
- **comfy-env.toml** for isolated environment support
  - Defines `sam3dbody2abc` environment with bpy dependency
  - Compatible with ComfyUI's isolated execution system

### Improved
- **Better error messages** when no export method available
  - Shows current Python version
  - Explains bpy version requirements clearly
  - Provides installation commands for both bpy and Blender
  - Special message for Python 3.12+ users (bpy not supported)

- **Updated README** with accurate installation instructions
  - Documents both export methods
  - Clarifies Python version requirements for bpy

### Technical Notes
- bpy module requires **exact** Python version match:
  - `bpy 4.1+` → Python 3.11.x only
  - `bpy 4.0.0` → Python 3.10.x only
  - Python 3.12+ → **Not supported**, must use Blender subprocess

---

## [5.2.0] - Previous

### Added
- Mesh-to-skeleton alignment for SAM3DBody v5.2.0
  - Handles ground-centered mesh vs pelvis-centered skeleton
  - Automatic offset calculation and application

### Changed
- Depth-based positioning now default mode
- Scale factor integration from Motion Analyzer

---

## [5.1.8] - Previous

### Added
- `flip_ty` option to FBX Export for newer SAM3DBody versions
- 📐 Body Shape Lock node
- 🔄 Pose Smoothing node  
- 🦶 Foot Contact Enforcer node
- 📹 SLAM Camera Solver node

### Fixed
- Improved Blender auto-detection with more search paths
- Character Trajectory numpy array boolean evaluation error
- Handle dict-based frames with sorted keys
- Handle variable return signatures from load_sam_3d_body()
- Support path-based SAM3D_MODEL format

---

## Installation

### For Python 3.12+ users (like RunPod default):
```bash
# bpy is NOT available for Python 3.12+
# Install Blender instead:
cd /workspace
wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
tar -xf blender-4.2.0-linux-x64.tar.xz
ln -sf /workspace/blender-4.2.0-linux-x64/blender /usr/local/bin/blender
```

### For Python 3.11.x users:
```bash
pip install bpy
```

### For Python 3.10.x users:
```bash
pip install bpy==4.0.0
```
