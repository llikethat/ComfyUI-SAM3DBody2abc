#!/usr/bin/env python3
"""
SAM3DBody2abc Installation Script

This script helps set up the dependencies for SAM3DBody2abc:
1. Clones Meta's SAM-3D-Body repository
2. Installs Python dependencies
3. Downloads model weights from HuggingFace (requires approval)
4. Verifies installation

Usage:
    python install.py              # Interactive installation
    python install.py --auto       # Automatic installation (skip prompts)
    python install.py --check      # Just verify installation
    python install.py --download   # Only download model weights
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path


def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def print_step(num, msg):
    print(f"\n[Step {num}] {msg}")
    print("-" * 50)


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"  ERROR: {result.stderr}")
        return False
    return True


def find_comfyui_path():
    """Try to find ComfyUI installation."""
    # Check if we're in custom_nodes
    current = Path(__file__).parent
    if current.parent.name == "custom_nodes":
        return current.parent.parent
    
    # Check common locations
    common_paths = [
        Path.home() / "ComfyUI",
        Path("/workspace/ComfyUI"),
        Path("/opt/ComfyUI"),
        Path.cwd() / "ComfyUI",
    ]
    
    for p in common_paths:
        if (p / "custom_nodes").exists():
            return p
    
    return None


def check_sam3d_source():
    """Check if sam_3d_body is importable."""
    try:
        import sam_3d_body
        return True, sam_3d_body.__file__
    except ImportError:
        return False, None


def check_model_files(model_path):
    """Check if model files exist."""
    model_path = Path(model_path)
    ckpt = model_path / "model.ckpt"
    mhr = model_path / "assets" / "mhr_model.pt"
    
    return {
        "model.ckpt": ckpt.exists(),
        "assets/mhr_model.pt": mhr.exists(),
    }


def clone_sam3d_body(target_dir):
    """Clone Meta's SAM-3D-Body repository."""
    target = Path(target_dir)
    
    if target.exists():
        print(f"  Directory already exists: {target}")
        return True
    
    target.parent.mkdir(parents=True, exist_ok=True)
    
    return run_cmd([
        "git", "clone",
        "https://github.com/facebookresearch/sam-3d-body.git",
        str(target)
    ])


def install_sam3d_deps(sam3d_path):
    """Install SAM-3D-Body dependencies."""
    sam3d_path = Path(sam3d_path)
    req_file = sam3d_path / "requirements.txt"
    
    if req_file.exists():
        return run_cmd([
            sys.executable, "-m", "pip", "install",
            "-r", str(req_file)
        ])
    else:
        print(f"  Warning: requirements.txt not found at {req_file}")
        return True


def download_model_weights(model_path):
    """Download model weights from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  Installing huggingface_hub...")
        run_cmd([sys.executable, "-m", "pip", "install", "huggingface-hub"])
        from huggingface_hub import snapshot_download
    
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    
    print(f"  Downloading to: {model_path}")
    print("  Note: You must have requested access at:")
    print("    https://huggingface.co/facebook/sam-3d-body-dinov3")
    print()
    
    try:
        snapshot_download(
            repo_id="facebook/sam-3d-body-dinov3",
            local_dir=str(model_path),
        )
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        print()
        print("  To download manually:")
        print("    1. Go to https://huggingface.co/facebook/sam-3d-body-dinov3")
        print("    2. Request access (usually approved within hours)")
        print("    3. Download files to:", model_path)
        return False


def setup_environment(sam3d_path):
    """Set up environment variable."""
    sam3d_path = str(Path(sam3d_path).absolute())
    
    # For current session
    os.environ["SAM3D_PATH"] = sam3d_path
    
    # Suggest permanent setup
    shell = os.environ.get("SHELL", "/bin/bash")
    rc_file = ".bashrc" if "bash" in shell else ".zshrc"
    
    print(f"  SAM3D_PATH={sam3d_path}")
    print()
    print(f"  To make this permanent, add to ~/{rc_file}:")
    print(f"    export SAM3D_PATH={sam3d_path}")
    
    return True


def verify_installation():
    """Verify the installation is working."""
    print_header("Verifying Installation")
    
    all_ok = True
    
    # Check sam_3d_body import
    found, path = check_sam3d_source()
    if found:
        print(f"  ✓ sam_3d_body found at: {path}")
    else:
        print("  ✗ sam_3d_body NOT found")
        print("    Set SAM3D_PATH environment variable or install via pip")
        all_ok = False
    
    # Check model files
    comfyui = find_comfyui_path()
    if comfyui:
        model_path = comfyui / "models" / "sam3dbody"
        files = check_model_files(model_path)
        
        for name, exists in files.items():
            if exists:
                print(f"  ✓ {name} found")
            else:
                print(f"  ✗ {name} NOT found at {model_path}")
                all_ok = False
    
    # Check Blender
    blender = shutil.which("blender")
    if blender:
        print(f"  ✓ Blender found: {blender}")
    else:
        print("  ⚠ Blender not found (needed for FBX export)")
        print("    Install: apt install blender  OR  download from blender.org")
    
    print()
    if all_ok:
        print("  ✓ Installation complete! Ready to use.")
    else:
        print("  ⚠ Some components missing. See above for details.")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="SAM3DBody2abc Installation")
    parser.add_argument("--auto", action="store_true", help="Automatic installation")
    parser.add_argument("--check", action="store_true", help="Only verify installation")
    parser.add_argument("--download", action="store_true", help="Only download model weights")
    parser.add_argument("--sam3d-path", type=str, help="Custom path for SAM-3D-Body source")
    parser.add_argument("--model-path", type=str, help="Custom path for model weights")
    args = parser.parse_args()
    
    print_header("SAM3DBody2abc Installation")
    
    # Find ComfyUI
    comfyui = find_comfyui_path()
    if comfyui:
        print(f"Found ComfyUI at: {comfyui}")
    else:
        print("Warning: Could not find ComfyUI installation")
        comfyui = Path.cwd()
    
    # Set default paths - sam-3d-body goes INSIDE our extension folder
    our_extension = Path(__file__).parent  # ComfyUI-SAM3DBody2abc folder
    sam3d_path = args.sam3d_path or str(our_extension / "sam-3d-body")
    model_path = args.model_path or str(comfyui / "models" / "sam3dbody")
    
    if args.check:
        return 0 if verify_installation() else 1
    
    if args.download:
        print_step(1, "Downloading Model Weights")
        success = download_model_weights(model_path)
        return 0 if success else 1
    
    # Full installation
    print_step(1, "Clone SAM-3D-Body Repository")
    print(f"  Target: {sam3d_path}")
    
    if not args.auto:
        response = input("  Continue? [Y/n]: ").strip().lower()
        if response == 'n':
            print("  Skipped.")
        else:
            clone_sam3d_body(sam3d_path)
    else:
        clone_sam3d_body(sam3d_path)
    
    print_step(2, "Install Dependencies")
    if Path(sam3d_path).exists():
        install_sam3d_deps(sam3d_path)
    
    print_step(3, "Set Up Environment")
    setup_environment(sam3d_path)
    
    print_step(4, "Download Model Weights")
    print(f"  Target: {model_path}")
    print()
    print("  IMPORTANT: Model download requires HuggingFace account and")
    print("  access approval for facebook/sam-3d-body-dinov3")
    print()
    
    if not args.auto:
        response = input("  Attempt download now? [y/N]: ").strip().lower()
        if response == 'y':
            download_model_weights(model_path)
        else:
            print("  Skipped. Download manually later with:")
            print(f"    python install.py --download --model-path {model_path}")
    
    print_step(5, "Verify Installation")
    verify_installation()
    
    print()
    print_header("Next Steps")
    print("1. If model download failed, request access at:")
    print("   https://huggingface.co/facebook/sam-3d-body-dinov3")
    print()
    print("2. Set SAM3D_PATH permanently:")
    print(f"   export SAM3D_PATH={sam3d_path}")
    print()
    print("3. Restart ComfyUI and look for the new node:")
    print("   '🔧 Load SAM3DBody Model (Direct)'")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
