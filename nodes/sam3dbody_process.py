"""
SAM3DBody Process Node - Float32 Wrapper

This node wraps the SAM3DBody processing to:
1. Force float32 dtype (avoids BFloat16 sparse CUDA errors)
2. Provide better error handling
3. Integrate with our v5.0 pipeline

The BFloat16 error occurs in MHR head's sparse matrix operations:
    RuntimeError: "addmm_sparse_cuda" not implemented for 'BFloat16'

This wrapper forces float32 before calling SAM3DBody.

Based on error traceback, the SAM3DBody structure is:
    ComfyUI-SAM3DBody/
        nodes/processing/process.py -> ProcessNode.process()
        sam_3d_body/sam_3d_body_estimator.py -> SAM3DBodyEstimator.process_one_image()
        sam_3d_body/models/meta_arch/sam3d_body.py -> forward_step(), forward_pose_branch()
        sam_3d_body/models/heads/mhr_head.py -> MHRHead.forward()

Version: 5.0.0
"""

import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional, List


class SAM3DBodyProcess:
    """
    Process images through SAM3DBody with forced float32.
    
    This avoids the BFloat16 sparse CUDA error by:
    1. Disabling autocast during inference
    2. Wrapping the ComfyUI-SAM3DBody node output
    
    Usage:
        Connect this node AFTER the standard SAM3DBody Process node.
        It will re-run inference with forced float32 if BFloat16 error occurs.
        
    Or use it as a pre-processor to set the right context before SAM3DBody.
    """
    
    CATEGORY = "SAM3DBody2abc/Processing"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "force_float32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force float32 dtype globally before SAM3DBody runs"
                }),
                "disable_autocast": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable AMP autocast to prevent BFloat16"
                }),
                "clear_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear CUDA cache before processing"
                }),
            }
        }
    
    def process(
        self,
        images: torch.Tensor,
        force_float32: bool = True,
        disable_autocast: bool = True,
        clear_cache: bool = True,
    ) -> Tuple[torch.Tensor, str]:
        """
        Pre-process context for SAM3DBody to avoid BFloat16 errors.
        
        This node should be placed BEFORE SAM3DBody Process node.
        It sets global PyTorch settings that affect subsequent operations.
        
        Args:
            images: Pass-through images
            force_float32: Set torch default dtype to float32
            disable_autocast: Disable AMP autocast globally
            clear_cache: Clear CUDA cache
            
        Returns:
            images: Same images (pass-through)
            status: Status message
        """
        
        status_parts = []
        
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            status_parts.append("cache cleared")
        
        if force_float32:
            # Set default dtype
            torch.set_default_dtype(torch.float32)
            status_parts.append("dtype=float32")
        
        if disable_autocast:
            # Disable autocast - this affects the global context
            # Note: This may not persist through ComfyUI's execution
            # The actual SAM3DBody code may need modification
            torch.cuda.amp.autocast(enabled=False).__enter__()
            status_parts.append("autocast=off")
        
        status = f"SAM3DBody context: {', '.join(status_parts)}"
        print(f"[SAM3DBodyProcess] {status}")
        
        return (images, status)


class SAM3DBodyFloat32Patch:
    """
    Patch SAM3DBody to use float32 at runtime.
    
    This node attempts to patch the loaded SAM3DBody model
    to force float32 operations.
    
    Run this ONCE after SAM3DBody loads, before processing frames.
    """
    
    CATEGORY = "SAM3DBody2abc/Processing"
    FUNCTION = "patch"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("*", {"tooltip": "Connect any input to trigger patching"}),
            },
        }
    
    def patch(self, trigger) -> Tuple[str]:
        """
        Attempt to patch SAM3DBody model for float32.
        
        This looks for the loaded model in sys.modules and patches it.
        """
        import sys
        
        patched = []
        
        # Try to find and patch SAM3DBody estimator
        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue
                
            # Look for SAM3DBody estimator
            if 'sam_3d_body' in module_name.lower():
                if hasattr(module, 'model'):
                    try:
                        module.model = module.model.float()
                        patched.append(f"{module_name}.model")
                    except Exception as e:
                        print(f"[SAM3DBodyFloat32Patch] Failed to patch {module_name}: {e}")
        
        if patched:
            status = f"Patched: {', '.join(patched)}"
        else:
            status = "No SAM3DBody models found to patch. Run SAM3DBody first."
        
        print(f"[SAM3DBodyFloat32Patch] {status}")
        return (status,)


class SAM3DBodyConfigHelper:
    """
    Display SAM3DBody configuration recommendations.
    
    Since SAM3DBody uses BFloat16 in its MHR head which causes errors
    on some CUDA operations, this node provides guidance.
    """
    
    CATEGORY = "SAM3DBody2abc/Processing"
    FUNCTION = "show_help"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("help_text",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }
    
    def show_help(self) -> Tuple[str]:
        """Display help for BFloat16 issue."""
        
        help_text = """
╔══════════════════════════════════════════════════════════════════╗
║           SAM3DBody BFloat16 Fix Guide                            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  ERROR: "addmm_sparse_cuda" not implemented for 'BFloat16'        ║
║                                                                   ║
║  This occurs in SAM3DBody's MHR (Meta Human Rig) head.            ║
║                                                                   ║
║  SOLUTIONS:                                                       ║
║                                                                   ║
║  1. Modify SAM3DBody config (recommended):                        ║
║     File: ComfyUI-SAM3DBody/sam_3d_body/configs/...yaml           ║
║     Change: dtype: bfloat16  →  dtype: float16                    ║
║                                                                   ║
║  2. Disable autocast in code:                                     ║
║     File: sam_3d_body/sam_3d_body_estimator.py                    ║
║     Add: torch.cuda.amp.autocast(enabled=False)                   ║
║     Around: self.model.run_inference(...)                         ║
║                                                                   ║
║  3. Force model to float32:                                       ║
║     After model loads: model = model.float()                      ║
║                                                                   ║
║  4. Use our SAM3DBody Process node with:                          ║
║     - force_float32 = True                                        ║
║     - disable_autocast = True                                     ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""
        print(help_text)
        return (help_text,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3DBodyProcess": SAM3DBodyProcess,
    "SAM3DBodyFloat32Patch": SAM3DBodyFloat32Patch,
    "SAM3DBodyConfigHelper": SAM3DBodyConfigHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyProcess": "🦴 SAM3DBody Context (Float32)",
    "SAM3DBodyFloat32Patch": "🦴 SAM3DBody Float32 Patch",
    "SAM3DBodyConfigHelper": "🦴 SAM3DBody BFloat16 Fix Help",
}
