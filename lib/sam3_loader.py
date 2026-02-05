"""
SAM3 Library Loader
===================

Provides access to SAM3 model code, either from:
1. System-installed sam3 package
2. Bundled sam3 library in this package

This allows the package to work standalone without requiring
separate SAM3 installation.
"""

import os
import sys

# Check if sam3 is already installed
try:
    import sam3 as _external_sam3
    SAM3_AVAILABLE = True
    SAM3_SOURCE = "external"
except ImportError:
    SAM3_AVAILABLE = False
    SAM3_SOURCE = None

# If not available externally, try to use bundled version
if not SAM3_AVAILABLE:
    _lib_dir = os.path.dirname(os.path.abspath(__file__))
    _sam3_dir = os.path.join(_lib_dir, "sam3")
    
    if os.path.exists(_sam3_dir):
        # Add to path temporarily for imports
        if _lib_dir not in sys.path:
            sys.path.insert(0, _lib_dir)
        
        try:
            import sam3 as _bundled_sam3
            SAM3_AVAILABLE = True
            SAM3_SOURCE = "bundled"
        except ImportError as e:
            print(f"[SAM3DBody2abc] Could not load bundled SAM3: {e}")
            SAM3_AVAILABLE = False


def get_sam3():
    """Get the SAM3 module."""
    if SAM3_SOURCE == "external":
        import sam3
        return sam3
    elif SAM3_SOURCE == "bundled":
        _lib_dir = os.path.dirname(os.path.abspath(__file__))
        if _lib_dir not in sys.path:
            sys.path.insert(0, _lib_dir)
        import sam3
        return sam3
    else:
        raise ImportError(
            "SAM3 is not available. Please install with: pip install sam3 "
            "or ensure the bundled library is present."
        )


def build_video_predictor(checkpoint_path: str, **kwargs):
    """Build SAM3 video predictor from checkpoint."""
    sam3 = get_sam3()
    return sam3.model_builder.build_sam3_video_predictor(checkpoint_path, **kwargs)


# Report status
if SAM3_AVAILABLE:
    print(f"[SAM3DBody2abc] SAM3 library available ({SAM3_SOURCE})")
else:
    print("[SAM3DBody2abc] SAM3 library not available - will use fallback processing")
