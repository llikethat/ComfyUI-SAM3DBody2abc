#!/usr/bin/env python3
"""
Installation script for ComfyUI-SAM3DBody-Video-Alembic
Extension for animated export from SAM3DBody
"""

import subprocess
import sys
import os
import shutil


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_package(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def main():
    print("=" * 70)
    print("ComfyUI-SAM3DBody-Video-Alembic Installation")
    print("Extension for animated export from SAM3DBody")
    print("=" * 70)
    
    # Install requirements
    print("\n[1/5] Installing Python dependencies...")
    requirements = [
        ("numpy", "numpy>=1.20.0"),
        ("trimesh", "trimesh>=3.9.0"),
        ("scipy", "scipy>=1.7.0"),
    ]
    
    for check_name, install_name in requirements:
        if not check_package(check_name):
            print(f"  Installing {install_name}...")
            try:
                install_package(install_name)
                print(f"  ✓ {check_name} installed")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        else:
            print(f"  ✓ {check_name} already installed")
    
    # Check OpenCV
    print("\n[2/5] Checking OpenCV (for overlay rendering)...")
    if check_package("cv2"):
        print("  ✓ OpenCV installed")
    else:
        print("  Installing opencv-python...")
        try:
            install_package("opencv-python")
            print("  ✓ OpenCV installed")
        except:
            print("  ⚠ OpenCV not installed - overlay will use fallback renderer")
    
    # Check Alembic
    print("\n[3/5] Checking Alembic support...")
    if check_package("alembic"):
        print("  ✓ Native PyAlembic available")
    else:
        print("  ℹ Native Alembic not installed")
        print("    Export will use:")
        print("    - Blender subprocess (if Blender installed)")
        print("    - OBJ sequence fallback")
        print("")
        print("  To enable native Alembic:")
        print("    conda install -c conda-forge alembic")
    
    # Check Blender
    print("\n[4/5] Checking Blender (for FBX export)...")
    blender_paths = [
        shutil.which("blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    for v in ["4.2", "4.1", "4.0", "3.6"]:
        blender_paths.append(f"C:\\Program Files\\Blender Foundation\\Blender {v}\\blender.exe")
    
    blender_found = None
    for path in blender_paths:
        if path and os.path.exists(path):
            blender_found = path
            break
    
    if blender_found:
        print(f"  ✓ Blender found: {blender_found}")
    else:
        print("  ⚠ Blender not found")
        print("    FBX skeleton export will not be available")
        print("    Install Blender: https://www.blender.org/download/")
    
    # Check required custom nodes
    print("\n[5/5] Checking required ComfyUI custom nodes...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.dirname(script_dir)
    
    required = [
        ("ComfyUI-SAM3DBody", "https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody"),
    ]
    optional = [
        ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
    ]
    
    for name, url in required:
        path = os.path.join(custom_nodes_dir, name)
        if os.path.exists(path):
            print(f"  ✓ {name} found")
        else:
            print(f"  ✗ {name} NOT FOUND (REQUIRED)")
            print(f"    Install from: {url}")
    
    for name, url in optional:
        path = os.path.join(custom_nodes_dir, name)
        if os.path.exists(path):
            print(f"  ✓ {name} found")
        else:
            print(f"  ℹ {name} not found (optional, for video input)")
            print(f"    Install from: {url}")
    
    # Done
    print("\n" + "=" * 70)
    print("Installation complete!")
    print("=" * 70)
    print("""
Key Nodes Added:
  🎬 SAM3DBody Batch Processor - Process video through SAM3DBody
  📦 Export Animated Alembic   - Export full animation to .abc
  🦴 Export Animated FBX       - Export skeleton animation to .fbx
  🎨 Render Mesh Overlay       - Visualize mesh on images

Basic Workflow:
  1. Load Video (VHS) → Load SAM3DBody Model
  2. SAM3DBody Batch Processor
  3. Export Animated Alembic / Export Animated FBX
  4. (Optional) Render Mesh Overlay Batch for visualization

Restart ComfyUI to load the new nodes.
""")


if __name__ == "__main__":
    main()
