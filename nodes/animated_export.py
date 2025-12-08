"""
Animated Export Nodes for SAM3DBody
Export mesh sequences to animated Alembic (.abc) and FBX formats.
Unlike per-frame export, these create single files with full animation timeline.
"""

import os
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
import folder_paths

# Global cache for Blender path to avoid repeated searches
_BLENDER_PATH_CACHE = None
_BLENDER_SEARCHED = False


def find_blender() -> Optional[str]:
    """Find Blender executable (uses global cache). Can be imported by other modules."""
    global _BLENDER_PATH_CACHE, _BLENDER_SEARCHED
    
    if _BLENDER_SEARCHED:
        return _BLENDER_PATH_CACHE
    
    import shutil
    import glob
    
    locations = [
        shutil.which("blender"),
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    
    # ComfyUI SAM3DBody bundled Blender - try multiple approaches
    try:
        # Method 1: From this file's location (custom_nodes/ComfyUI-SAM3DBody2abc/nodes/)
        custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Method 2: From folder_paths
        try:
            custom_nodes_dir2 = os.path.join(folder_paths.base_path, "custom_nodes")
        except:
            custom_nodes_dir2 = None
        
        # Search patterns for SAM3DBody bundled Blender
        for base_dir in [custom_nodes_dir, custom_nodes_dir2, "/workspace/ComfyUI/custom_nodes"]:
            if base_dir is None:
                continue
            sam3d_blender_patterns = [
                os.path.join(base_dir, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*-linux-x64", "blender"),
                os.path.join(base_dir, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*", "blender"),
                os.path.join(base_dir, "ComfyUI-SAM3DBody", "lib", "blender", "*", "blender"),
            ]
            for pattern in sam3d_blender_patterns:
                matches = glob.glob(pattern)
                if matches:
                    locations.extend(matches)
                    break
            if len(locations) > 4:  # Found something beyond defaults
                break
    except Exception as e:
        print(f"[SAM3DBody2abc] Error searching for Blender: {e}")
    
    # Windows paths
    for version in ["4.2", "4.1", "4.0", "3.6"]:
        locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {version}\\blender.exe")
    
    _BLENDER_SEARCHED = True
    for loc in locations:
        if loc and os.path.exists(loc):
            _BLENDER_PATH_CACHE = loc
            print(f"[SAM3DBody2abc] Found Blender: {loc}")
            return loc
    
    _BLENDER_PATH_CACHE = None
    return None


class ExportAnimatedAlembic:
    """
    Export mesh sequence to animated Alembic (.abc) format.
    Creates a SINGLE file with animated vertices across ALL frames.
    
    This is different from per-frame export - the entire animation
    is contained in one .abc file that can be imported into Blender,
    Maya, Houdini, Cinema 4D, etc.
    
    Includes optional temporal smoothing to reduce jitter.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "body_animation",
                    "multiline": False
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 1.0
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Leave empty for ComfyUI output folder"
                }),
                "world_space": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply camera transform to match overlay render (recommended for Maya)"
                }),
                "static_camera": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "For static camera shots (sports, surveillance). Places all characters in absolute world positions."
                }),
                "include_joints": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include joint positions as point cloud"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Scale factor (1.0=meters, 100=centimeters)"
                }),
                "up_axis": (["Y", "Z"], {"default": "Y"}),
                "center_mesh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Center mesh at origin (ignores world_space)"
                }),
                "temporal_smoothing": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Temporal smoothing strength (0=none, 1=max smoothing)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("file_path", "status", "exported_frames")
    FUNCTION = "export_alembic"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def _apply_temporal_smoothing(
        self,
        frames: List[Dict],
        smoothing_strength: float,
    ) -> List[Dict]:
        """Apply Gaussian temporal smoothing to vertex positions."""
        if smoothing_strength <= 0:
            return frames
        
        num_frames = len(frames)
        if num_frames < 3:
            return frames
        
        # Determine kernel radius based on smoothing strength (1-5 frames)
        kernel_radius = int(np.ceil(smoothing_strength * 4)) + 1
        kernel_radius = min(kernel_radius, num_frames // 2)
        
        print(f"[ExportAnimatedAlembic] Applying temporal smoothing (strength={smoothing_strength}, radius={kernel_radius})")
        
        # Create Gaussian kernel
        sigma = max(kernel_radius / 2.0, 0.5)
        x = np.arange(-kernel_radius, kernel_radius + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # Stack all vertices
        all_verts = [np.array(f["vertices"]) for f in frames]
        num_verts = all_verts[0].shape[0]
        
        # Smooth each frame
        smoothed_verts = []
        for t in range(num_frames):
            weighted_sum = np.zeros((num_verts, 3), dtype=np.float64)
            weight_sum = 0.0
            
            for k in range(-kernel_radius, kernel_radius + 1):
                src_t = t + k
                if 0 <= src_t < num_frames:
                    weight = kernel[k + kernel_radius]
                    weighted_sum += all_verts[src_t] * weight
                    weight_sum += weight
            
            smoothed_verts.append(weighted_sum / weight_sum)
        
        # Create new frames with smoothed vertices
        smoothed_frames = []
        for i, frame in enumerate(frames):
            new_frame = dict(frame)
            new_frame["vertices"] = smoothed_verts[i]
            smoothed_frames.append(new_frame)
        
        print(f"[ExportAnimatedAlembic] Smoothing complete")
        return smoothed_frames
    
    def export_alembic(
        self,
        mesh_sequence: List[Dict],
        filename: str = "body_animation",
        fps: float = 24.0,
        output_dir: str = "",
        world_space: bool = True,
        static_camera: bool = False,
        include_joints: bool = True,
        scale: float = 1.0,
        up_axis: str = "Y",
        center_mesh: bool = False,
        temporal_smoothing: float = 0.5,
    ) -> Tuple[str, str, int]:
        """
        Export complete animation to Alembic file(s).
        
        If multiple people detected (person_index varies), exports SEPARATE files:
        - filename_person0.abc
        - filename_person1.abc
        - etc.
        
        If world_space=True, applies the same transforms as the overlay renderer:
        - 180° rotation around X axis
        - Camera-relative translation for character movement
        
        If static_camera=True (for sports/surveillance):
        - Places each character at their absolute world position
        - All characters share the same world coordinate system
        - Camera is effectively at origin
        """
        # Setup output path
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter valid frames
        valid_frames = [f for f in mesh_sequence if f.get("valid") and f.get("vertices") is not None]
        
        if not valid_frames:
            return ("", "Error: No valid mesh data", 0)
        
        # ========================================
        # GROUP FRAMES BY PERSON INDEX
        # ========================================
        person_frames = {}
        for frame in valid_frames:
            person_idx = frame.get("person_index", 0)
            if person_idx not in person_frames:
                person_frames[person_idx] = []
            person_frames[person_idx].append(frame)
        
        num_people = len(person_frames)
        print(f"[ExportAnimatedAlembic] Found {num_people} person(s) in mesh sequence")
        for p_idx, frames in person_frames.items():
            print(f"[ExportAnimatedAlembic]   Person {p_idx}: {len(frames)} frames")
        
        # ========================================
        # EXPORT EACH PERSON TO SEPARATE FILE
        # ========================================
        exported_files = []
        total_frames = 0
        
        for person_idx in sorted(person_frames.keys()):
            person_valid_frames = person_frames[person_idx]
            
            # Get bbox position for auto-ID (use first frame)
            bbox_x = None
            if person_valid_frames:
                first_frame = person_valid_frames[0]
                bbox_x = first_frame.get("person_bbox_x")
                if bbox_x is None and first_frame.get("bbox") is not None:
                    bbox = first_frame["bbox"]
                    bbox_x = (bbox[0] + bbox[2]) / 2
            
            # Generate filename for this person with position info
            if num_people > 1:
                if bbox_x is not None:
                    # Include position hint in filename (L=left, C=center, R=right)
                    # Get image width from first frame
                    img_size = person_valid_frames[0].get("image_size", (1920, 1080))
                    img_w = img_size[0] if img_size else 1920
                    
                    if bbox_x < img_w * 0.33:
                        pos_hint = "L"
                    elif bbox_x > img_w * 0.66:
                        pos_hint = "R"
                    else:
                        pos_hint = "C"
                    
                    person_filename = f"{filename}_person{person_idx}_{pos_hint}"
                else:
                    person_filename = f"{filename}_person{person_idx}"
            else:
                person_filename = filename
            
            output_path = os.path.join(output_dir, f"{person_filename}.abc")
            
            # Handle existing files
            counter = 1
            while os.path.exists(output_path):
                output_path = os.path.join(output_dir, f"{person_filename}_{counter:04d}.abc")
                counter += 1
            
            print(f"[ExportAnimatedAlembic] Exporting Person {person_idx} → {os.path.basename(output_path)}")
            
            # Apply world_space transform (same as overlay renderer)
            if world_space and not center_mesh:
                if static_camera:
                    # STATIC CAMERA MODE: For sports, surveillance, wide shots
                    # Camera is at origin, characters placed at absolute world positions
                    print(f"[ExportAnimatedAlembic] Person {person_idx}: Static camera mode")
                    
                    for i, frame in enumerate(person_valid_frames):
                        verts = np.array(frame["vertices"])
                        cam_t = frame.get("camera")
                        
                        if cam_t is not None:
                            if hasattr(cam_t, 'cpu'):
                                cam_t = cam_t.cpu().numpy()
                            cam_t = np.array(cam_t).flatten()
                            
                            # cam_t = [x, y, z] is camera position relative to person
                            # Meta flips X in their convention
                            # For world position: person is at (-cam_t) relative to camera
                            # After 180° X rotation: Y and Z are negated
                            # So final world position = [-cam_t[0], cam_t[1], cam_t[2]] after rotation
                            person_world_x = -cam_t[0]  # Flip X (Meta convention)
                            person_world_y = cam_t[1]   # Y stays (will be negated by rotation)
                            person_world_z = cam_t[2]   # Z stays (will be negated by rotation)
                        else:
                            person_world_x, person_world_y, person_world_z = 0.0, 0.0, 0.0
                        
                        # Step 1: Apply 180° rotation around X axis (negate Y and Z)
                        verts[:, 1] = -verts[:, 1]
                        verts[:, 2] = -verts[:, 2]
                        
                        # Step 2: Translate to world position
                        # After rotation, person offset becomes [x, -y, -z]
                        verts[:, 0] += person_world_x
                        verts[:, 1] += -person_world_y  # Negated due to rotation
                        verts[:, 2] += -person_world_z  # Negated due to rotation
                        
                        frame["vertices"] = verts
                        
                        if frame.get("joints") is not None:
                            joints = np.array(frame["joints"])
                            joints[:, 1] = -joints[:, 1]
                            joints[:, 2] = -joints[:, 2]
                            joints[:, 0] += person_world_x
                            joints[:, 1] += -person_world_y
                            joints[:, 2] += -person_world_z
                            frame["joints"] = joints
                        
                        if i == 0:
                            print(f"[ExportAnimatedAlembic] Person {person_idx} cam_t: {cam_t if cam_t is not None else 'None'}")
                            print(f"[ExportAnimatedAlembic] Person {person_idx} world pos: ({person_world_x:.2f}, {-person_world_y:.2f}, {-person_world_z:.2f})")
                else:
                    # TRACKING CAMERA MODE
                    print(f"[ExportAnimatedAlembic] Person {person_idx}: Tracking camera mode")
                    
                    ref_cam = person_valid_frames[0].get("camera")
                    if ref_cam is not None:
                        if hasattr(ref_cam, 'cpu'):
                            ref_cam = ref_cam.cpu().numpy()
                        ref_cam = np.array(ref_cam).flatten()
                    else:
                        ref_cam = np.array([0.0, 0.0, 0.0])
                    
                    for i, frame in enumerate(person_valid_frames):
                        verts = np.array(frame["vertices"])
                        cam_t = frame.get("camera")
                        offset = np.array([0.0, 0.0, 0.0])
                        
                        if cam_t is not None:
                            if hasattr(cam_t, 'cpu'):
                                cam_t = cam_t.cpu().numpy()
                            cam_t = np.array(cam_t).flatten()
                            camera_trans = np.array([-cam_t[0], cam_t[1], cam_t[2]])
                            ref_trans = np.array([-ref_cam[0], ref_cam[1], ref_cam[2]])
                            offset = camera_trans - ref_trans
                        
                        verts[:, 1] = -verts[:, 1]
                        verts[:, 2] = -verts[:, 2]
                        rotated_offset = np.array([offset[0], -offset[1], -offset[2]])
                        verts = verts - rotated_offset
                        frame["vertices"] = verts
                        
                        if frame.get("joints") is not None:
                            joints = np.array(frame["joints"])
                            joints[:, 1] = -joints[:, 1]
                            joints[:, 2] = -joints[:, 2]
                            joints = joints - rotated_offset
                            frame["joints"] = joints
            
            # Apply temporal smoothing
            if temporal_smoothing > 0:
                person_valid_frames = self._apply_temporal_smoothing(person_valid_frames, temporal_smoothing)
            
            # Export this person
            export_result = self._export_single_person(
                person_valid_frames, output_path, fps, include_joints, scale, up_axis, center_mesh
            )
            
            if export_result[0]:  # If file was created
                exported_files.append(export_result[0])
                total_frames += export_result[2]
        
        # ========================================
        # RETURN RESULTS
        # ========================================
        if not exported_files:
            return ("", "Error: No files exported", 0)
        
        # Return first file path, but include all in status
        primary_path = exported_files[0]
        
        if num_people > 1:
            file_list = ", ".join([os.path.basename(f) for f in exported_files])
            status = f"Exported {num_people} characters: {file_list}"
        else:
            status = f"Exported {total_frames} frames to {os.path.basename(primary_path)}"
        
        print(f"[ExportAnimatedAlembic] {status}")
        
        return (primary_path, status, total_frames)
    
    def _export_single_person(
        self,
        frames: List[Dict],
        output_path: str,
        fps: float,
        include_joints: bool,
        scale: float,
        up_axis: str,
        center_mesh: bool,
    ) -> Tuple[str, str, int]:
        """Export a single person's frames to Alembic."""
        # Try native Alembic export (PyAlembic - not SQLAlchemy alembic!)
        try:
            from alembic import Abc, AbcGeom
            import imath
            result = self._export_native_alembic(
                frames, output_path, fps, include_joints, scale, up_axis, center_mesh
            )
            return result
        except ImportError as e:
            print(f"[ExportAnimatedAlembic] PyAlembic not available ({e}), trying Blender...")
        except Exception as e:
            print(f"[ExportAnimatedAlembic] PyAlembic export failed: {e}, trying Blender...")
        
        # Try Blender export
        try:
            result = self._export_via_blender(
                frames, output_path, fps, include_joints, scale, up_axis, center_mesh
            )
            return result
        except Exception as e:
            print(f"[ExportAnimatedAlembic] Blender export failed: {e}")
        
        # Fallback to OBJ sequence with animation metadata
        return self._export_with_metadata(
            frames, output_path, fps, scale, up_axis, center_mesh
        )
    
    def _export_native_alembic(
        self,
        frames: List[Dict],
        output_path: str,
        fps: float,
        include_joints: bool,
        scale: float,
        up_axis: str,
        center_mesh: bool,
    ) -> Tuple[str, str, int]:
        """Export using native PyAlembic with constant topology."""
        import alembic
        from alembic import Abc, AbcGeom
        import imath
        
        print(f"[ExportAnimatedAlembic] Using native PyAlembic export")
        print(f"[ExportAnimatedAlembic] {len(frames)} frames at {fps} FPS")
        
        # Get reference topology from first frame (constant for all frames)
        ref_faces = frames[0].get("faces")
        if ref_faces is None:
            raise ValueError("No face data in mesh sequence - cannot export Alembic")
        
        ref_faces = np.array(ref_faces)
        num_faces = len(ref_faces)
        face_indices = imath.IntArray(ref_faces.flatten().tolist())
        face_counts = imath.IntArray([3] * num_faces)  # All triangles
        
        print(f"[ExportAnimatedAlembic] Topology: {len(frames[0]['vertices'])} verts, {num_faces} faces")
        
        # Create archive
        archive = Abc.OArchive(output_path)
        top = archive.getTop()
        
        # Time sampling - uniform samples at fps
        time_per_frame = 1.0 / fps
        time_sampling = Abc.TimeSampling(time_per_frame, 0.0)
        ts_idx = archive.addTimeSampling(time_sampling)
        
        # Create mesh object
        mesh_obj = AbcGeom.OPolyMesh(top, "body_mesh")
        mesh_schema = mesh_obj.getSchema()
        mesh_schema.setTimeSampling(ts_idx)
        
        # Create joints object if requested
        joints_schema = None
        if include_joints:
            joints_obj = AbcGeom.OPoints(top, "body_joints")
            joints_schema = joints_obj.getSchema()
            joints_schema.setTimeSampling(ts_idx)
        
        # Axis conversion matrix
        axis_transform = self._get_axis_transform(up_axis)
        
        # Compute global center if needed (from first frame)
        global_center = np.zeros(3)
        if center_mesh:
            global_center = np.mean(np.array(frames[0]["vertices"]), axis=0)
        
        # Write each frame
        for idx, frame_data in enumerate(frames):
            vertices = np.array(frame_data["vertices"], dtype=np.float32)
            
            # Apply transforms
            vertices = vertices * scale
            if center_mesh:
                vertices = vertices - global_center * scale
            if axis_transform is not None:
                vertices = vertices @ axis_transform.T
            
            # Convert to imath array
            vertex_array = imath.V3fArray(len(vertices))
            for i, v in enumerate(vertices):
                vertex_array[i] = imath.V3f(float(v[0]), float(v[1]), float(v[2]))
            
            # Create mesh sample with SAME topology for all frames
            mesh_sample = AbcGeom.OPolyMeshSchemaSample(
                vertex_array,
                face_indices,
                face_counts
            )
            mesh_schema.set(mesh_sample)
            
            # Write joints
            if joints_schema is not None:
                joints = frame_data.get("joints")
                if joints is not None:
                    joints = np.array(joints, dtype=np.float32) * scale
                    if center_mesh:
                        joints = joints - global_center * scale
                    if axis_transform is not None:
                        joints = joints @ axis_transform.T
                    
                    joint_array = imath.V3fArray(len(joints))
                    for i, j in enumerate(joints):
                        joint_array[i] = imath.V3f(float(j[0]), float(j[1]), float(j[2]))
                    
                    joints_sample = AbcGeom.OPointsSchemaSample(joint_array)
                    joints_schema.set(joints_sample)
            
            if (idx + 1) % 50 == 0:
                print(f"[ExportAnimatedAlembic] Written {idx + 1}/{len(frames)} frames...")
        
        # Archive is closed when it goes out of scope
        del archive
        
        print(f"[ExportAnimatedAlembic] Successfully exported {len(frames)} frames")
        return (output_path, f"Exported {len(frames)} frames via PyAlembic", len(frames))
    
    def _export_via_blender(
        self,
        frames: List[Dict],
        output_path: str,
        fps: float,
        include_joints: bool,
        scale: float,
        up_axis: str,
        center_mesh: bool,
    ) -> Tuple[str, str, int]:
        """Export using Blender subprocess."""
        import subprocess
        import tempfile
        
        # Find Blender
        blender_path = self._find_blender()
        if not blender_path:
            raise RuntimeError("Blender not found")
        
        # Prepare data
        export_data = {
            "frames": [],
            "output_path": output_path,
            "fps": fps,
            "scale": scale,
            "up_axis": up_axis,
            "center_mesh": center_mesh,
            "include_joints": include_joints,
        }
        
        for frame in frames:
            frame_export = {
                "vertices": np.array(frame["vertices"]).tolist(),
                "faces": np.array(frame.get("faces", [])).tolist() if frame.get("faces") is not None else None,
            }
            if include_joints and frame.get("joints") is not None:
                frame_export["joints"] = np.array(frame["joints"]).tolist()
            export_data["frames"].append(frame_export)
        
        # Write data to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            data_path = f.name
        
        # Create Blender script
        script = self._create_blender_alembic_script()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            result = subprocess.run(
                [blender_path, "--background", "--python", script_path, "--", data_path],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0 and os.path.exists(output_path):
                return (output_path, f"Exported {len(frames)} frames via Blender", len(frames))
            else:
                raise RuntimeError(f"Blender export failed: {result.stderr}")
        finally:
            os.unlink(data_path)
            os.unlink(script_path)
    
    def _export_with_metadata(
        self,
        frames: List[Dict],
        output_path: str,
        fps: float,
        scale: float,
        up_axis: str,
        center_mesh: bool,
    ) -> Tuple[str, str, int]:
        """Fallback: Export OBJ sequence with animation metadata."""
        
        base_path = output_path.replace('.abc', '')
        output_dir = os.path.dirname(output_path)
        base_name = os.path.basename(base_path)
        
        axis_transform = self._get_axis_transform(up_axis)
        
        # Export each frame as OBJ
        for idx, frame in enumerate(frames):
            vertices = np.array(frame["vertices"]) * scale
            if center_mesh:
                vertices = vertices - np.mean(vertices, axis=0)
            if axis_transform is not None:
                vertices = vertices @ axis_transform.T
            
            frame_path = os.path.join(output_dir, f"{base_name}_{idx:05d}.obj")
            
            with open(frame_path, 'w') as f:
                f.write(f"# Frame {idx}, FPS: {fps}\n")
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                faces = frame.get("faces")
                if faces is not None:
                    for face in np.array(faces):
                        f.write(f"f {' '.join(str(int(i)+1) for i in face)}\n")
        
        # Write animation metadata
        meta_path = os.path.join(output_dir, f"{base_name}_animation.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "fps": fps,
                "frame_count": len(frames),
                "scale": scale,
                "up_axis": up_axis,
                "format": "obj_sequence",
                "naming": f"{base_name}_XXXXX.obj"
            }, f, indent=2)
        
        first_file = os.path.join(output_dir, f"{base_name}_00000.obj")
        return (first_file, f"Exported {len(frames)} frames as OBJ sequence", len(frames))
    
    def _get_axis_transform(self, up_axis: str) -> Optional[np.ndarray]:
        """Get axis transformation matrix."""
        if up_axis == "Y":
            return None
        elif up_axis == "Z":
            return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        return None
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable (uses shared function)."""
        return find_blender()
    
    def _create_blender_alembic_script(self) -> str:
        """Create Blender Python script for animated Alembic export using mesh cache."""
        return '''
import bpy
import json
import sys
import os

argv = sys.argv
data_path = argv[argv.index("--") + 1]

with open(data_path, 'r') as f:
    data = json.load(f)

frames = data["frames"]
output_path = data["output_path"]
fps = data["fps"]
scale = data["scale"]
num_frames = len(frames)

print(f"[Blender] Exporting {num_frames} frames to Alembic...")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Set FPS and frame range
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = num_frames

# Write OBJ sequence to temp directory
temp_dir = os.path.dirname(data_path)
obj_dir = os.path.join(temp_dir, "obj_sequence")
os.makedirs(obj_dir, exist_ok=True)

print(f"[Blender] Writing {num_frames} OBJ files...")

# Get faces from first frame (constant topology)
first_faces = frames[0].get("faces", [])

for idx, frame_data in enumerate(frames):
    verts = frame_data["vertices"]
    faces = frame_data.get("faces", first_faces)
    
    obj_path = os.path.join(obj_dir, f"frame_{idx+1:06d}.obj")
    
    with open(obj_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0] * scale} {v[1] * scale} {v[2] * scale}\\n")
        for face in faces:
            f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\\n")
    
    if (idx + 1) % 50 == 0:
        print(f"[Blender] Written {idx + 1}/{num_frames} OBJ files...")

print(f"[Blender] Importing first frame...")

# Import first OBJ
first_obj = os.path.join(obj_dir, "frame_000001.obj")
bpy.ops.wm.obj_import(filepath=first_obj)

# Get imported object
obj = bpy.context.selected_objects[0]
obj.name = "body_mesh"
mesh = obj.data
mesh.name = "body_mesh"

print(f"[Blender] Mesh has {len(mesh.vertices)} vertices, {len(mesh.polygons)} faces")

# Add Mesh Cache modifier pointing to OBJ sequence
mod = obj.modifiers.new(name="MeshCache", type='MESH_SEQUENCE_CACHE')

# Unfortunately MESH_SEQUENCE_CACHE needs Alembic input, not OBJ
# Remove it and use shape keys instead
obj.modifiers.remove(mod)

print(f"[Blender] Creating shape keys from OBJ sequence...")

# Add basis shape key
obj.shape_key_add(name="Basis", from_mix=False)

# Create shape keys from OBJ files
for idx in range(num_frames):
    obj_path = os.path.join(obj_dir, f"frame_{idx+1:06d}.obj")
    
    # Read vertices
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    
    # Create shape key
    sk = obj.shape_key_add(name=f"frame_{idx+1:06d}", from_mix=False)
    for i, v in enumerate(verts):
        if i < len(sk.data):
            sk.data[i].co = v
    
    if (idx + 1) % 50 == 0:
        print(f"[Blender] Created {idx + 1}/{num_frames} shape keys...")

print(f"[Blender] Keyframing shape keys with CONSTANT interpolation...")

# Get all shape keys (excluding Basis)
shape_keys = [kb for kb in obj.data.shape_keys.key_blocks if kb.name != "Basis"]

# Set all to 0 initially
for sk in shape_keys:
    sk.value = 0.0

# Keyframe each shape key: on at its frame, off elsewhere
for idx, sk in enumerate(shape_keys):
    frame_num = idx + 1
    
    # Turn this one on
    sk.value = 1.0
    sk.keyframe_insert(data_path="value", frame=frame_num)
    
    # Turn it off at adjacent frames
    if frame_num > 1:
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=frame_num - 1)
    if frame_num < num_frames:
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=frame_num + 1)

# Set CONSTANT interpolation
if obj.data.shape_keys.animation_data and obj.data.shape_keys.animation_data.action:
    for fc in obj.data.shape_keys.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'

# Select object for export
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# Cleanup OBJ files
import shutil
shutil.rmtree(obj_dir, ignore_errors=True)

print(f"[Blender] Exporting to Alembic with evaluated mesh...")

# Export to Alembic
# use_mesh_modifiers=True ensures shape keys are evaluated
bpy.ops.wm.alembic_export(
    filepath=output_path,
    selected=True,
    start=1,
    end=num_frames,
    flatten=False,
    visible_objects_only=False,
    export_hair=False,
    export_particles=False,
    apply_subdiv=False,
    evaluation_mode='RENDER',  # Ensures modifiers/shape keys evaluated
)

print(f"SUCCESS: Exported {num_frames} frames to {output_path}")
'''




# FBX export: See fbx_export.py for animated FBX with skeleton
# BVH export: See bvh_export.py for skeleton animation export
# OBJ sequence export removed - use Alembic or FBX instead
