#!/usr/bin/env python3
"""
SAM3DBody2abc Installation Script

This script is called by ComfyUI-Manager on install/update.
It checks if dependencies are already installed and skips if so.

Manual usage:
    python install.py              # Auto-install (skips if already done)
    python install.py --force      # Force reinstall
    python install.py --check      # Just verify installation
"""

import os
import sys
import subprocess
from pathlib import Path


def print_status(msg, status="info"):
    prefix = {"info": "ℹ️", "ok": "✓", "skip": "⏭️", "warn": "⚠️", "error": "✗"}
    print(f"  {prefix.get(status, '')} {msg}")


def is_already_installed():
    """Check if SAM3DBody2abc is already set up."""
    our_extension = Path(__file__).parent
    sam3d_path = our_extension / "sam-3d-body" / "sam_3d_body"
    
    # Check if sam-3d-body source exists
    if not sam3d_path.exists():
        return False, "sam-3d-body source not found"
    
    # Check core dependencies
    missing_deps = []
    core_deps = ["braceexpand", "omegaconf", "hydra", "roma", "yacs", "einops", "timm"]
    for dep in core_deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        return False, f"Missing dependencies: {', '.join(missing_deps)}"
    
    return True, "All components installed"


def install_dependencies():
    """Install required Python dependencies."""
    deps = [
        "braceexpand",
        "omegaconf", 
        "hydra-core",
        "pytorch-lightning",
        "einops",
        "timm",
        "roma",
        "yacs",
        "huggingface-hub",
    ]
    
    print_status("Installing Python dependencies...")
    
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            print_status(f"{dep} already installed", "skip")
        except ImportError:
            print_status(f"Installing {dep}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep, "--quiet"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print_status(f"Failed to install {dep}: {result.stderr}", "error")
            else:
                print_status(f"{dep} installed", "ok")


def clone_sam3d_body():
    """Clone SAM-3D-Body repository if not exists."""
    our_extension = Path(__file__).parent
    target = our_extension / "sam-3d-body"
    
    if (target / "sam_3d_body").exists():
        print_status("sam-3d-body already cloned", "skip")
        return True
    
    print_status("Cloning sam-3d-body repository...")
    
    result = subprocess.run(
        ["git", "clone", "https://github.com/facebookresearch/sam-3d-body.git", str(target)],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print_status("sam-3d-body cloned successfully", "ok")
        return True
    else:
        print_status(f"Clone failed: {result.stderr}", "error")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAM3DBody2abc Installation")
    parser.add_argument("--force", action="store_true", help="Force reinstall")
    parser.add_argument("--check", action="store_true", help="Only check installation")
    args = parser.parse_args()
    
    print()
    print("=" * 50)
    print("  SAM3DBody2abc - Installation Check")
    print("=" * 50)
    print()
    
    # Quick check if already installed
    if not args.force:
        installed, reason = is_already_installed()
        if installed:
            print_status("SAM3DBody2abc is already installed!", "ok")
            print_status("Use --force to reinstall", "info")
            print()
            return 0
        else:
            print_status(f"Need to install: {reason}", "info")
    
    if args.check:
        installed, reason = is_already_installed()
        print_status(f"Status: {reason}", "ok" if installed else "warn")
        return 0 if installed else 1
    
    # Install dependencies
    print()
    print("[1/2] Python Dependencies")
    print("-" * 30)
    install_dependencies()
    
    # Clone sam-3d-body
    print()
    print("[2/2] SAM-3D-Body Source")
    print("-" * 30)
    clone_sam3d_body()
    
    # Final check
    print()
    print("=" * 50)
    installed, reason = is_already_installed()
    if installed:
        print_status("Installation complete!", "ok")
        print()
        print("  Note: Model weights will be downloaded automatically")
        print("  when you first use the Load Model node.")
        print("  (Requires HuggingFace token with access to facebook/sam-3d-body-dinov3)")
        print()
    else:
        print_status(f"Installation incomplete: {reason}", "warn")
        print()
    
    return 0 if installed else 1


if __name__ == "__main__":
    sys.exit(main())
