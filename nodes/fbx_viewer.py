"""
FBX Animation Viewer Node for SAM3DBody2abc
Interactive animation playback for animated FBX files.

This node passes the FBX path to the UI for visualization.
Requires the web extension from ComfyUI-MotionCapture for full functionality,
or will display the path for external viewing.
"""

from typing import Dict, Any, Tuple

# Import logger
try:
    from ..lib.logger import log
except ImportError:
    class _FallbackLog:
        def info(self, msg): print(f"[FBX Viewer] {msg}")
    log = _FallbackLog()


class FBXAnimationViewer:
    """
    Display an interactive animation viewer for animated FBX files.
    
    Shows skeletal animation playback with play/pause controls, timeline scrubber,
    and adjustable playback speed.
    
    Note: Full visualization requires ComfyUI-MotionCapture web extension.
    Without it, this node displays the FBX path for external viewing.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "fbx_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Path to animated FBX file"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    FUNCTION = "view_animation"
    OUTPUT_NODE = True
    CATEGORY = "SAM3DBody2abc"

    def view_animation(self, fbx_path: str) -> Dict:
        """
        Display animated FBX playback in ComfyUI UI.

        Args:
            fbx_path: Absolute path to animated FBX file

        Returns:
            Dict with ui key for web extension
        """
        try:
            log.info(f" Displaying animation for: {fbx_path}")

            # The actual animation viewer is handled by the web extension
            # Return ui dict to send data to onExecuted callback
            return {
                "ui": {
                    "fbx_path": [fbx_path]
                },
                "result": (fbx_path,)
            }

        except Exception as e:
            error_msg = f"FBXAnimationViewer failed: {str(e)}"
            log.info(f" Error: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "ui": {
                    "fbx_path": [""]
                },
                "result": ("",)
            }
