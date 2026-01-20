"""
SAM3DBody Model Loader - Direct Integration

This node loads Meta's SAM-3D-Body model directly without requiring
any third-party ComfyUI wrappers. It outputs a SAM3D_MODEL that is
compatible with the Video Batch Processor node.

AUTOMATIC SETUP:
- Model weights are automatically downloaded from HuggingFace on first run
- SAM-3D-Body source code is automatically cloned if not found

REQUIREMENTS:
- HuggingFace account with access to facebook/sam-3d-body-dinov3
  (Request access at: https://huggingface.co/facebook/sam-3d-body-dinov3)
- Run: huggingface-cli login (once, to authenticate)
"""

import os
import sys
import subprocess
from typing import Dict, Any, Optional

# Try to get ComfyUI folder_paths
try:
    import folder_paths
    DEFAULT_MODEL_PATH = os.path.join(folder_paths.models_dir, "sam3dbody")
    CUSTOM_NODES_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except ImportError:
    DEFAULT_MODEL_PATH = os.path.expanduser("~/.cache/sam3dbody")
    CUSTOM_NODES_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import logger
try:
    from ..lib.logger import log, set_module
    set_module("SAM3DBody Loader")
except ImportError:
    class _FallbackLog:
        def info(self, msg): print(f"[SAM3DBody Loader] {msg}")
        def debug(self, msg): pass
        def warn(self, msg): print(f"[SAM3DBody Loader] WARN: {msg}")
        def error(self, msg): print(f"[SAM3DBody Loader] ERROR: {msg}")
    log = _FallbackLog()


def find_sam3d_path() -> Optional[str]:
    """
    Find the sam-3d-body source code path.
    
    Checks in order:
    1. SAM3D_PATH environment variable
    2. Inside our own extension folder (ComfyUI-SAM3DBody2abc/sam-3d-body)
    3. Common installation locations
    4. Already installed via pip
    """
    # Check if already importable
    try:
        import sam_3d_body
        return None  # Already in path, no need to add
    except ImportError:
        pass
    
    # Check env var
    env_path = os.environ.get("SAM3D_PATH")
    if env_path and os.path.exists(env_path):
        sam_module = os.path.join(env_path, "sam_3d_body")
        if os.path.exists(sam_module):
            return env_path
    
    # Check inside our extension folder FIRST (most common case)
    our_sam3d = os.path.join(CUSTOM_NODES_PATH, "sam-3d-body")
    if os.path.exists(os.path.join(our_sam3d, "sam_3d_body")):
        return our_sam3d
    
    # Check common locations
    common_paths = [
        os.path.expanduser("~/sam-3d-body"),
        os.path.expanduser("~/projects/sam-3d-body"),
        "/opt/sam-3d-body",
        "/workspace/sam-3d-body",
    ]
    
    # Check sibling to custom_nodes
    try:
        custom_nodes = os.path.dirname(CUSTOM_NODES_PATH)
        common_paths.append(os.path.join(custom_nodes, "sam-3d-body"))
    except:
        pass
    
    for path in common_paths:
        sam_module = os.path.join(path, "sam_3d_body")
        if os.path.exists(sam_module):
            return path
    
    return None


def clone_sam3d_body(target_dir: str) -> bool:
    """Clone Meta's SAM-3D-Body repository and install dependencies."""
    if os.path.exists(target_dir):
        log.info(f"SAM-3D-Body already exists at: {target_dir}")
        # Still need to ensure dependencies are installed
        install_sam3d_dependencies(target_dir)
        return True
    
    log.info(f"Cloning SAM-3D-Body to: {target_dir}")
    log.info("This is a one-time download (~100MB)...")
    
    try:
        result = subprocess.run(
            ["git", "clone", "https://github.com/facebookresearch/sam-3d-body.git", target_dir],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            log.info("SAM-3D-Body cloned successfully!")
            # Install dependencies
            install_sam3d_dependencies(target_dir)
            return True
        else:
            log.error(f"Git clone failed: {result.stderr}")
            return False
    except FileNotFoundError:
        log.error("Git not found. Please install git and try again.")
        return False
    except subprocess.TimeoutExpired:
        log.error("Git clone timed out. Please check your internet connection.")
        return False
    except Exception as e:
        log.error(f"Failed to clone: {e}")
        return False


def install_sam3d_dependencies(sam3d_path: str):
    """Install sam-3d-body's Python dependencies."""
    log.info("Installing SAM-3D-Body dependencies...")
    
    # Core dependencies that sam-3d-body needs
    core_deps = [
        "braceexpand",
        "omegaconf",
        "hydra-core",
        "pytorch-lightning",
        "einops",
        "timm",
        "scipy",
    ]
    
    try:
        # Install core dependencies first
        for dep in core_deps:
            try:
                __import__(dep.replace("-", "_"))
            except ImportError:
                log.info(f"Installing {dep}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep, "--quiet"],
                    capture_output=True,
                    timeout=120,
                )
        
        # Check for requirements.txt in sam-3d-body
        req_file = os.path.join(sam3d_path, "requirements.txt")
        if os.path.exists(req_file):
            log.info("Installing from requirements.txt...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"],
                capture_output=True,
                timeout=300,
            )
        
        log.info("Dependencies installed successfully!")
    except Exception as e:
        log.warn(f"Some dependencies may not have installed: {e}")
        log.warn("You may need to manually install: pip install braceexpand omegaconf hydra-core")


def download_model_weights(model_path: str, hf_token: str = "") -> bool:
    """Download model weights from HuggingFace."""
    log.info(f"Downloading model weights to: {model_path}")
    log.info("This is a one-time download (~3GB)...")
    
    # Get token from parameter or environment
    token = hf_token if hf_token else os.environ.get("HF_TOKEN")
    
    if not token:
        log.warn("No HuggingFace token provided!")
        log.warn("Get your token from: https://huggingface.co/settings/tokens")
        log.warn("Then enter it in the 'hf_token' field of the Load Model node")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.info("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"], 
                      capture_output=True)
        from huggingface_hub import snapshot_download
    
    os.makedirs(model_path, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id="facebook/sam-3d-body-dinov3",
            local_dir=model_path,
            local_dir_use_symlinks=False,
            token=token,
        )
        log.info("Model weights downloaded successfully!")
        return True
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg or "Access" in error_msg or "authorize" in error_msg.lower():
            log.error("")
            log.error("=" * 50)
            log.error("  AUTHENTICATION FAILED")
            log.error("=" * 50)
            log.error("")
            log.error("To fix this:")
            log.error("1. Go to: https://huggingface.co/facebook/sam-3d-body-dinov3")
            log.error("2. Click 'Request access' (free, usually approved in hours)")
            log.error("3. Go to: https://huggingface.co/settings/tokens")
            log.error("4. Create a token with 'Read' permission")
            log.error("5. Paste the token in the 'hf_token' field of this node")
            log.error("")
        else:
            log.error(f"Download failed: {e}")
        return False


def ensure_sam3d_available() -> Optional[str]:
    """
    Ensure SAM-3D-Body source code is available.
    Downloads automatically if not found.
    Returns the path to add to sys.path, or None if already importable.
    """
    # Check if already importable
    try:
        import sam_3d_body
        log.info(f"SAM-3D-Body found: {os.path.dirname(sam_3d_body.__file__)}")
        return None
    except ImportError:
        pass
    
    # Try to find existing installation
    found_path = find_sam3d_path()
    if found_path:
        log.info(f"Found SAM-3D-Body at: {found_path}")
        return found_path
    
    # Auto-clone to custom_nodes directory
    target_dir = os.path.join(CUSTOM_NODES_PATH, "sam-3d-body")
    
    if clone_sam3d_body(target_dir):
        return target_dir
    
    return None


class LoadSAM3DBodyModel:
    """
    Load SAM-3D-Body Model (Direct)
    
    This node loads Meta's SAM-3D-Body model directly from the official
    repository. Everything is downloaded automatically on first run.
    
    The output is a SAM3D_MODEL config dict that is compatible with
    the Video Batch Processor node.
    
    REQUIRED FILES (auto-downloaded):
    - model.ckpt (main model weights, ~3GB)
    - assets/mhr_model.pt (MHR parametric body model)
    
    Source: https://huggingface.co/facebook/sam-3d-body-dinov3
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": DEFAULT_MODEL_PATH,
                    "tooltip": "Path to model weights folder. Downloads automatically on first run."
                }),
            },
            "optional": {
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for inference"
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace token for downloading model. Get from: https://huggingface.co/settings/tokens"
                }),
            }
        }
    
    RETURN_TYPES = ("SAM3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3DBody2abc"
    
    def load_model(
        self,
        model_path: str,
        device: str = "cuda",
        hf_token: str = "",
    ) -> tuple:
        """
        Load SAM-3D-Body model configuration.
        
        Automatically downloads:
        - SAM-3D-Body source code (if not found)
        - Model weights from HuggingFace (if not found)
        
        Args:
            model_path: Path to model folder (auto-downloads if missing)
            device: cuda or cpu
            hf_token: HuggingFace token for downloading model weights
            
        Returns:
            Tuple containing SAM3D_MODEL config dict
        """
        import torch
        
        log.info("=" * 50)
        log.info("  Loading SAM-3D-Body Model")
        log.info("=" * 50)
        
        # Override device if CUDA not available
        if device == "cuda" and not torch.cuda.is_available():
            log.warn("CUDA not available, falling back to CPU")
            device = "cpu"
        
        # Step 1: Check/download model weights FIRST (before importing sam_3d_body)
        # This way users can see download progress even if dependencies are missing
        model_path = os.path.abspath(os.path.expanduser(model_path))
        ckpt_path = os.path.join(model_path, "model.ckpt")
        mhr_path = os.path.join(model_path, "assets", "mhr_model.pt")
        
        log.info(f"Model path: {model_path}")
        log.info(f"Checkpoint: {ckpt_path}")
        log.info(f"MHR model: {mhr_path}")
        
        # Check if files exist
        need_download = False
        if not os.path.exists(ckpt_path):
            log.info(f"Model checkpoint not found!")
            need_download = True
        else:
            log.info(f"Model checkpoint found: {os.path.getsize(ckpt_path) / 1e9:.2f} GB")
            
        if not os.path.exists(mhr_path):
            log.info(f"MHR model not found!")
            need_download = True
        else:
            log.info(f"MHR model found: {os.path.getsize(mhr_path) / 1e6:.1f} MB")
        
        if need_download:
            log.info("")
            log.info("=" * 50)
            log.info("  DOWNLOADING MODEL WEIGHTS...")
            log.info("=" * 50)
            log.info("")
            log.info(f"HF Token provided: {'Yes' if hf_token else 'No'}")
            
            success = download_model_weights(model_path, hf_token)
            
            if not success:
                # Re-check what's missing
                missing = []
                if not os.path.exists(ckpt_path):
                    missing.append("model.ckpt")
                if not os.path.exists(mhr_path):
                    missing.append("assets/mhr_model.pt")
                
                if missing:
                    raise FileNotFoundError(
                        f"Model download failed. Missing files in {model_path}:\n" +
                        "\n".join(f"  - {m}" for m in missing) +
                        "\n\n"
                        "To fix this:\n"
                        "1. Go to: https://huggingface.co/facebook/sam-3d-body-dinov3\n"
                        "2. Click 'Request access' (free, usually approved within hours)\n"
                        "3. Get token from: https://huggingface.co/settings/tokens\n"
                        "4. Paste token in 'hf_token' field of this node\n"
                        "5. Restart ComfyUI and try again\n\n"
                        f"Or manually download to: {model_path}"
                    )
        
        # Step 2: Ensure SAM-3D-Body source code is available (auto-clone if needed)
        log.info("")
        log.info("Checking SAM-3D-Body source code...")
        sam3d_path = ensure_sam3d_available()
        if sam3d_path and sam3d_path not in sys.path:
            log.info(f"Adding to path: {sam3d_path}")
            sys.path.insert(0, sam3d_path)
            # Install dependencies if this is a fresh clone
            install_sam3d_dependencies(sam3d_path)
        
        # Step 3: Verify sam_3d_body is importable
        log.info("")
        log.info("Importing sam_3d_body...")
        try:
            import sam_3d_body
            log.info(f"sam_3d_body module loaded from: {os.path.dirname(sam_3d_body.__file__)}")
        except ImportError as e:
            error_msg = str(e)
            log.warn(f"Import error: {error_msg}")
            
            # Check for specific missing dependencies
            if "braceexpand" in error_msg:
                log.info("Installing missing dependency: braceexpand")
                subprocess.run([sys.executable, "-m", "pip", "install", "braceexpand"], capture_output=True)
            if "omegaconf" in error_msg:
                log.info("Installing missing dependency: omegaconf")
                subprocess.run([sys.executable, "-m", "pip", "install", "omegaconf"], capture_output=True)
            if "hydra" in error_msg:
                log.info("Installing missing dependency: hydra-core")
                subprocess.run([sys.executable, "-m", "pip", "install", "hydra-core"], capture_output=True)
            if "einops" in error_msg:
                log.info("Installing missing dependency: einops")
                subprocess.run([sys.executable, "-m", "pip", "install", "einops"], capture_output=True)
            if "timm" in error_msg:
                log.info("Installing missing dependency: timm")
                subprocess.run([sys.executable, "-m", "pip", "install", "timm"], capture_output=True)
            
            # Try import again
            try:
                import sam_3d_body
                log.info(f"sam_3d_body module loaded after installing dependencies")
            except ImportError as e2:
                raise ImportError(
                    f"Could not import sam_3d_body.\n\n"
                    f"Missing dependency or automatic clone failed.\n"
                    f"Try running: pip install braceexpand omegaconf hydra-core einops timm\n\n"
                    f"If that doesn't work, manually run:\n"
                    f"  git clone https://github.com/facebookresearch/sam-3d-body.git\n"
                    f"  pip install -r sam-3d-body/requirements.txt\n"
                    f"  export SAM3D_PATH=/path/to/sam-3d-body\n\n"
                    f"Original error: {e2}"
                )
        
        # Return config dict (model loads lazily in video processor)
        config = {
            "model_path": model_path,
            "ckpt_path": ckpt_path,
            "mhr_path": mhr_path,
            "device": device,
            "_loader": "SAM3DBody2abc_Direct",
        }
        
        log.info("")
        log.info("=" * 50)
        log.info("  Model config ready!")
        log.info("=" * 50)
        log.info(f"  Weights: {model_path}")
        log.info(f"  Device: {device}")
        
        return (config,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_LoadModel": LoadSAM3DBodyModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_LoadModel": "🔧 Load SAM3DBody Model (Direct)",
}
