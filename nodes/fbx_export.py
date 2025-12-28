"""
Animated FBX Export for SAM3DBody2abc
Exports MESH_SEQUENCE to animated FBX using Blender.

The exported FBX contains:
- Mesh with shape keys (vertex animation per frame)
- Armature with keyframed bone rotations (from MHR) or positions

Settings:
- skeleton_mode: "Rotations" uses true joint rotations from MHR model
                 "Positions" uses joint positions (legacy)
"""

import os
import json
import subprocess
import shutil
import glob
import tempfile
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
import folder_paths

BLENDER_TIMEOUT = 600

_current_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(os.path.dirname(_current_dir), "lib")
BLENDER_SCRIPT = os.path.join(_lib_dir, "blender_animated_fbx.py")

_BLENDER_PATH = None


def find_blender() -> Optional[str]:
    """Find Blender executable."""
    global _BLENDER_PATH
    
    if _BLENDER_PATH is not None:
        return _BLENDER_PATH
    
    locations = [
        shutil.which("blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    
    # Check SAM3DBody bundled Blender
    try:
        custom_nodes = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        patterns = [
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*-linux-x64", "blender"),
            os.path.join(custom_nodes, "ComfyUI-SAM3DBody", "lib", "blender", "*", "blender"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            locations.extend(matches)
    except Exception:
        pass
    
    # Windows
    for ver in ["4.2", "4.1", "4.0", "3.6"]:
        locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {ver}\\blender.exe")
    
    for loc in locations:
        if loc and os.path.exists(loc):
            _BLENDER_PATH = loc
            print(f"[FBX Export] Found Blender: {loc}")
            return loc
    
    return None


def to_list(obj):
    """Convert to JSON-serializable list."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_list(v) for v in obj]
    return obj


class ExportAnimatedFBX:
    """
    Export MESH_SEQUENCE to animated FBX or Alembic.
    
    Creates export with:
    - Mesh + shape keys (FBX) or vertex cache (ABC)
    - Skeleton with rotation animation (from MHR) or position animation
    - Camera with estimated focal length
    
    Output Formats:
    - FBX: Uses blend shapes for mesh animation (may show hidden per-frame geometry in Maya)
    - ABC: Uses vertex cache for mesh animation (better for Maya, cleaner playback)
    
    Skeleton Modes:
    - Rotations (Recommended): Uses true joint rotation matrices from MHR model.
      This produces proper bone rotations for retargeting and animation editing.
    - Positions (Legacy): Uses joint positions only. Bones animate via location offset.
      Limited for retargeting but shows exact joint positions.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "animation",
                }),
            },
            "optional": {
                "camera_rotations": ("CAMERA_DATA", {
                    "tooltip": "Camera rotation data from Camera Rotation Solver. Overrides internal camera rotation calculation."
                }),
                "fps": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 120.0,
                    "tooltip": "FPS for animation (0 = use source fps from mesh_sequence)"
                }),
                "frame_offset": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Start frame in output (1 = start from frame 1 for Maya, 0 = start from frame 0)"
                }),
                "output_format": (["FBX", "ABC (Alembic)"], {
                    "default": "FBX",
                    "tooltip": "FBX: blend shapes, ABC: vertex cache (better for Maya)"
                }),
                "up_axis": (["Y", "Z", "-Y", "-Z"], {
                    "default": "Y",
                    "tooltip": "Which axis points up in the output"
                }),
                "skeleton_mode": (["Rotations (Recommended)", "Positions (Legacy)"], {
                    "default": "Rotations (Recommended)",
                    "tooltip": "Rotations: proper bone rotations for retargeting. Positions: exact joint locations."
                }),
                "world_translation": (["None (Body at Origin)", "Baked into Mesh/Joints", "Baked into Camera", "Baked into Geometry (Static Camera)", "Root Locator", "Root Locator + Animated Camera", "Separate Track"], {
                    "default": "None (Body at Origin)",
                    "tooltip": "How to handle world translation. 'Baked into Geometry (Static Camera)' bakes camera motion into mesh/skeleton for stable exports."
                }),
                "flip_x": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Mirror/flip the animation on X axis (applies to mesh, skeleton, and root locator)"
                }),
                "include_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include mesh with animation"
                }),
                "include_camera": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include camera. Static for most modes, animated for 'Baked into Camera', follows root for 'Root Locator + Animated Camera'"
                }),
                "camera_motion": (["Translation (Default)", "Rotation (Pan/Tilt)", "Static"], {
                    "default": "Translation (Default)",
                    "tooltip": "How camera follows character. Translation: camera moves laterally. Rotation: camera pans/tilts (tripod-like). Static: camera stays fixed, body offset handles alignment. Only applies when world_translation is 'Root Locator + Animated Camera'."
                }),
                "camera_smoothing": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 15,
                    "step": 1,
                    "tooltip": "Smoothing window for camera animation to reduce jitter (0=none, 3=light, 5=medium, 9=heavy)"
                }),
                "bake_smoothing_method": (["Kalman Filter", "Spline Fitting", "Gaussian", "None"], {
                    "default": "Kalman Filter",
                    "tooltip": "Smoothing method when baking camera into geometry. Kalman: optimal for sequential data. Spline: smooth curves. Gaussian: simple averaging."
                }),
                "bake_smoothing_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Smoothing strength for geometry baking (0=minimal, 1=maximum). For Kalman: affects measurement noise. For Spline: affects curve smoothness."
                }),
                "sensor_width": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Camera sensor width in mm (Full Frame=36, APS-C=23.6, MFT=17.3)"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("output_path", "status", "frame_count", "fps")
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_fbx(
        self,
        mesh_sequence: Dict,
        filename: str = "animation",
        camera_rotations: Optional[Dict] = None,
        fps: float = 0.0,
        frame_offset: int = 1,
        output_format: str = "FBX",
        up_axis: str = "Y",
        skeleton_mode: str = "Rotations (Recommended)",
        world_translation: str = "None (Body at Origin)",
        flip_x: bool = False,
        include_mesh: bool = True,
        include_camera: bool = True,
        camera_motion: str = "Translation (Default)",
        camera_smoothing: int = 0,
        bake_smoothing_method: str = "Kalman Filter",
        bake_smoothing_strength: float = 0.5,
        sensor_width: float = 36.0,
        output_dir: str = "",
    ) -> Tuple[str, str, int, float]:
        """Export to animated FBX or Alembic."""
        
        # Get fps from mesh_sequence if not specified (0 means use source)
        if fps <= 0:
            fps = mesh_sequence.get("fps", 24.0)
            print(f"[Export] Using fps from source: {fps}")
        
        # Check if we have solved camera rotations from Camera Rotation Solver
        has_solved_rotations = camera_rotations is not None and "rotations" in camera_rotations
        if has_solved_rotations:
            print(f"[Export] Using solved camera rotations from Camera Rotation Solver ({len(camera_rotations['rotations'])} frames)")
            # Force rotation mode when using solved rotations
            use_camera_rotation = True
            camera_static = False
        elif camera_motion == "Static":
            use_camera_rotation = False
            camera_static = True
        else:
            use_camera_rotation = ("Rotation" in camera_motion)
            camera_static = False
        
        # Determine camera behavior based on world_translation mode
        # - "Baked into Camera": camera animated with inverse world offset
        # - "Root Locator + Animated Camera": camera parented to root, follows character
        # - "None" + rotation: camera rotates to frame body at origin
        animate_camera = (world_translation == "Baked into Camera")
        camera_follow_root = ("Root Locator + Animated Camera" in world_translation)
        
        # Camera rotation works with: None, Baked into Camera, Root Locator + Animated Camera
        # It does NOT work with: Baked into Mesh/Joints, Root Locator (no animation), Separate Track
        rotation_incompatible = any(x in world_translation for x in ["Baked into Mesh", "Separate"])
        rotation_incompatible = rotation_incompatible or (world_translation == "Root Locator")
        
        if use_camera_rotation and rotation_incompatible:
            print(f"[Export] Warning: Camera rotation mode not supported with '{world_translation}' - camera will be static.")
        
        if include_camera:
            if animate_camera:
                mode_str = "SOLVED rotation" if has_solved_rotations else ("rotation (pan/tilt)" if use_camera_rotation else "translation")
                print(f"[Export] Camera animated with {mode_str} (world_translation={world_translation})")
            elif camera_follow_root:
                mode_str = "SOLVED rotation" if has_solved_rotations else ("rotation (pan/tilt)" if use_camera_rotation else "translation")
                print(f"[Export] Camera follows root with {mode_str} (world_translation={world_translation})")
            elif use_camera_rotation and not rotation_incompatible:
                mode_str = "SOLVED rotation" if has_solved_rotations else "rotation (pan/tilt)"
                print(f"[Export] Camera {mode_str} to frame body at origin (world_translation={world_translation})")
            else:
                print(f"[Export] Camera will be static (world_translation={world_translation})")
        
        frames = mesh_sequence.get("frames", {})
        if not frames:
            return ("", "Error: No frames", 0, fps)
        
        blender_path = find_blender()
        if not blender_path:
            return ("", "Error: Blender not found", 0, fps)
        
        if not os.path.exists(BLENDER_SCRIPT):
            return ("", f"Error: Script not found: {BLENDER_SCRIPT}", 0, fps)
        
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        sorted_indices = sorted(frames.keys())
        
        # Determine output extension
        use_alembic = "ABC" in output_format
        ext = ".abc" if use_alembic else ".fbx"
        
        # Map world_translation option to mode
        translation_mode = "none"
        if "Baked into Geometry" in world_translation:
            translation_mode = "bake_to_geometry"
        elif "Baked into Mesh" in world_translation:
            translation_mode = "baked"
        elif "Baked into Camera" in world_translation:
            translation_mode = "camera"
        elif "Root" in world_translation:
            translation_mode = "root"
        elif "Separate" in world_translation:
            translation_mode = "separate"
        
        # Map skeleton_mode option
        use_rotations = "Rotations" in skeleton_mode
        
        # Check if rotation data is available (handle numpy arrays properly)
        first_frame = frames[sorted_indices[0]]
        joint_rots = first_frame.get("joint_rotations")
        
        # Debug: print what data we have
        print(f"[Export] First frame keys: {list(first_frame.keys())}")
        print(f"[Export] joint_rotations type: {type(joint_rots)}")
        if joint_rots is not None:
            if isinstance(joint_rots, np.ndarray):
                print(f"[Export] joint_rotations shape: {joint_rots.shape}, size: {joint_rots.size}")
            elif isinstance(joint_rots, list):
                print(f"[Export] joint_rotations is list with {len(joint_rots)} elements")
        
        has_rotations = joint_rots is not None
        if has_rotations and isinstance(joint_rots, np.ndarray):
            has_rotations = joint_rots.size > 0
        elif has_rotations and isinstance(joint_rots, list):
            has_rotations = len(joint_rots) > 0
        
        if use_rotations and not has_rotations:
            print("[Export] Warning: Rotation mode requested but no rotation data available. Falling back to positions.")
            print("[Export] Note: Make sure you're using a recent version of ComfyUI-SAM3DBody that outputs joint_rotations.")
            use_rotations = False
        else:
            print(f"[Export] Rotation data available: {has_rotations}, using rotations: {use_rotations}")
        
        # Build JSON for Blender
        joint_parents = mesh_sequence.get("joint_parents")
        print(f"[Export] joint_parents in mesh_sequence: {joint_parents is not None}")
        if joint_parents is not None:
            if hasattr(joint_parents, 'shape'):
                print(f"[Export] joint_parents shape: {joint_parents.shape}")
            elif isinstance(joint_parents, list):
                print(f"[Export] joint_parents length: {len(joint_parents)}")
        
        print(f"[Export] Camera motion mode: {'Static' if camera_static else ('Rotation (Pan/Tilt)' if use_camera_rotation else 'Translation')}")
        if camera_smoothing > 0:
            print(f"[Export] Camera smoothing: {camera_smoothing} frames")
        
        # Prepare solved camera rotations if available
        solved_rotations = None
        if has_solved_rotations:
            solved_rotations = []
            has_translation = camera_rotations.get("has_translation", False)
            for rot_data in camera_rotations["rotations"]:
                rot_entry = {
                    "frame": rot_data.get("frame", 0),
                    "pan": rot_data.get("pan", 0.0),
                    "tilt": rot_data.get("tilt", 0.0),
                    "roll": rot_data.get("roll", 0.0),
                }
                # Include translation if present
                if has_translation:
                    rot_entry["tx"] = rot_data.get("tx", 0.0)
                    rot_entry["ty"] = rot_data.get("ty", 0.0)
                    rot_entry["tz"] = rot_data.get("tz", 0.0)
                solved_rotations.append(rot_entry)
            
            final_rot = solved_rotations[-1]
            print(f"[Export] Solved rotations: {len(solved_rotations)} frames, final pan={np.degrees(final_rot['pan']):.2f}°, tilt={np.degrees(final_rot['tilt']):.2f}°")
            if has_translation:
                print(f"[Export] Camera translation data included (will compensate root position)")
        
        # Map bake smoothing method to internal name
        bake_method_map = {
            "Kalman Filter": "kalman",
            "Spline Fitting": "spline",
            "Gaussian": "gaussian",
            "None": "none",
        }
        bake_method = bake_method_map.get(bake_smoothing_method, "kalman")
        
        export_data = {
            "fps": fps,
            "frame_count": len(sorted_indices),
            "frame_offset": frame_offset,
            "faces": to_list(mesh_sequence.get("faces")),
            "joint_parents": to_list(joint_parents),
            "sensor_width": sensor_width,
            "world_translation_mode": translation_mode,
            "skeleton_mode": "rotations" if use_rotations else "positions",
            "flip_x": flip_x,
            "animate_camera": animate_camera,
            "camera_follow_root": camera_follow_root,
            "camera_use_rotation": use_camera_rotation,
            "camera_static": camera_static,  # New: disable all camera animation
            "camera_smoothing": camera_smoothing,
            "solved_camera_rotations": solved_rotations,  # From Camera Rotation Solver
            "bake_smoothing_method": bake_method,  # For geometry baking
            "bake_smoothing_strength": bake_smoothing_strength,
            "frames": [],
        }
        
        # DEBUG: Show root_locator and body_offset calculation for frame 0
        first_frame = frames[sorted_indices[0]]
        first_cam_t = first_frame.get("pred_cam_t")
        first_focal = first_frame.get("focal_length")
        first_image_size = first_frame.get("image_size")
        if first_cam_t is not None:
            first_cam_t = to_list(first_cam_t)
            if len(first_cam_t) >= 3:
                tx, ty, tz = first_cam_t[0], first_cam_t[1], first_cam_t[2]
                print(f"\n[Export DEBUG] ========== BODY ALIGNMENT (Frame 0) ==========")
                print(f"[Export DEBUG] pred_cam_t: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
                print(f"[Export DEBUG] focal_length: {first_focal}")
                print(f"[Export DEBUG] image_size: {first_image_size}")
                print(f"[Export DEBUG]")
                print(f"[Export DEBUG] NEW APPROACH (v3.5.7):")
                print(f"[Export DEBUG]   root_locator = (0, 0, 0)  ← Fixed at origin")
                print(f"[Export DEBUG]   body_offset in Blender = (tx, -ty, 0) = ({tx:.4f}, {-ty:.4f}, 0)")
                print(f"[Export DEBUG]   body_offset in Maya = (tx, 0, -ty) = ({tx:.4f}, 0, {-ty:.4f})")
                print(f"[Export DEBUG]   (ty negated due to camera rotation convention)")
                print(f"[Export DEBUG]")
                
                # What screen position does this correspond to?
                if first_focal and first_image_size:
                    focal = float(first_focal) if not isinstance(first_focal, (list, tuple)) else float(first_focal[0])
                    img_w, img_h = first_image_size[0], first_image_size[1]
                    cx, cy = img_w / 2, img_h / 2
                    # Screen position from pred_cam_t
                    screen_x = focal * tx / tz + cx
                    screen_y = focal * ty / tz + cy  # NO negation - SAM3DBody coords are image-aligned
                    print(f"[Export DEBUG] EXPECTED SCREEN POSITION:")
                    print(f"[Export DEBUG]   screen_x = {screen_x:.1f}px")
                    print(f"[Export DEBUG]   screen_y = {screen_y:.1f}px")
                    print(f"[Export DEBUG]   (Image center: {cx:.0f}, {cy:.0f})")
                print(f"[Export DEBUG] ================================================================\n")
        
        for idx in sorted_indices:
            frame = frames[idx]
            
            # Handle pred_cam_t vs camera field (can't use 'or' with numpy arrays)
            pred_cam_t = frame.get("pred_cam_t")
            if pred_cam_t is None:
                pred_cam_t = frame.get("camera")
            
            frame_data = {
                "frame_index": idx,
                "joint_coords": to_list(frame.get("joint_coords")),
                "pred_cam_t": to_list(pred_cam_t),
                "focal_length": frame.get("focal_length"),
                "bbox": to_list(frame.get("bbox")),  # For camera alignment
                "image_size": frame.get("image_size"),  # (width, height)
            }
            if include_mesh:
                frame_data["vertices"] = to_list(frame.get("vertices"))
            if use_rotations:
                frame_data["joint_rotations"] = to_list(frame.get("joint_rotations"))
            export_data["frames"].append(frame_data)
        
        # Write temp JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            json_path = f.name
        
        output_path = os.path.join(output_dir, f"{filename}{ext}")
        
        try:
            cmd = [
                blender_path,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                json_path,
                output_path,
                up_axis,
                "1" if include_mesh else "0",
                "1" if include_camera else "0",
            ]
            
            format_name = "Alembic" if use_alembic else "FBX"
            skel_mode_str = "rotations" if use_rotations else "positions"
            print(f"[Export] Exporting {len(sorted_indices)} frames as {format_name}")
            print(f"[Export] Settings: up={up_axis}, translation={translation_mode}, skeleton={skel_mode_str}, camera={include_camera}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT,
            )
            
            if result.returncode != 0:
                error = result.stderr[:500] if result.stderr else "Unknown error"
                print(f"[Export] Blender error: {error}")
                return ("", f"Blender error: {error}", 0, fps)
            
            if not os.path.exists(output_path):
                return ("", f"Error: {format_name} not created", 0, fps)
            
            status = f"Exported {len(sorted_indices)} frames as {format_name} (up={up_axis}, skeleton={skel_mode_str})"
            if not include_mesh:
                status += " skeleton only"
            if include_camera:
                status += " +camera"
            
            print(f"[Export] {status}")
            return (output_path, status, len(sorted_indices), fps)
            
        except subprocess.TimeoutExpired:
            return ("", "Error: Blender timed out", 0, fps)
        except Exception as e:
            return ("", f"Error: {str(e)}", 0, fps)
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


class ExportFBXFromJSON:
    """
    Export animated FBX or Alembic from saved JSON file.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "json_path": ("STRING", {"forceInput": True}),
                "output_format": (["FBX", "ABC (Alembic)"], {"default": "FBX"}),
                "up_axis": (["Y", "Z", "-Y", "-Z"], {"default": "Y"}),
                "skeleton_mode": (["Rotations (Recommended)", "Positions (Legacy)"], {
                    "default": "Rotations (Recommended)",
                }),
                "world_translation": (["None (Body at Origin)", "Baked into Mesh/Joints", "Baked into Camera", "Baked into Geometry (Static Camera)", "Root Locator", "Separate Track"], {
                    "default": "None (Body at Origin)",
                }),
            },
            "optional": {
                "filename": ("STRING", {"default": "animation"}),
                "include_mesh": ("BOOLEAN", {"default": True}),
                "include_camera": ("BOOLEAN", {"default": True}),
                "output_dir": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "status")
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_fbx(
        self,
        json_path: str,
        output_format: str = "FBX",
        up_axis: str = "Y",
        skeleton_mode: str = "Rotations (Recommended)",
        world_translation: str = "None (Body at Origin)",
        filename: str = "animation",
        include_mesh: bool = True,
        include_camera: bool = True,
        output_dir: str = "",
    ) -> Tuple[str, str]:
        """Convert JSON to FBX or Alembic."""
        
        if not os.path.exists(json_path):
            return ("", f"Error: JSON not found: {json_path}")
        
        blender_path = find_blender()
        if not blender_path:
            return ("", "Error: Blender not found")
        
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        
        # Determine output extension
        use_alembic = "ABC" in output_format
        ext = ".abc" if use_alembic else ".fbx"
        output_path = os.path.join(output_dir, f"{filename}{ext}")
        
        # Map world_translation option to mode
        translation_mode = "none"
        if "Baked into Geometry" in world_translation:
            translation_mode = "bake_to_geometry"
        elif "Baked into Mesh" in world_translation:
            translation_mode = "baked"
        elif "Baked into Camera" in world_translation:
            translation_mode = "camera"
        elif "Root" in world_translation:
            translation_mode = "root"
        elif "Separate" in world_translation:
            translation_mode = "separate"
        
        # Map skeleton_mode option
        use_rotations = "Rotations" in skeleton_mode
        
        # Read JSON and add modes
        with open(json_path, 'r') as f:
            data = json.load(f)
        data["world_translation_mode"] = translation_mode
        data["skeleton_mode"] = "rotations" if use_rotations else "positions"
        
        # Write to temp file with updated modes
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_json = f.name
        
        try:
            cmd = [
                blender_path,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                temp_json,
                output_path,
                up_axis,
                "1" if include_mesh else "0",
                "1" if include_camera else "0",
            ]
            
            format_name = "Alembic" if use_alembic else "FBX"
            skel_mode_str = "rotations" if use_rotations else "positions"
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)
            
            if result.returncode != 0:
                return ("", f"Blender error: {result.stderr[:500]}")
            
            if not os.path.exists(output_path):
                return ("", f"Error: {format_name} not created")
            
            return (output_path, f"Exported to {filename}{ext} (up={up_axis}, skeleton={skel_mode_str}, translation={translation_mode})")
            
        except Exception as e:
            return ("", f"Error: {str(e)}")
        finally:
            if os.path.exists(temp_json):
                os.unlink(temp_json)
