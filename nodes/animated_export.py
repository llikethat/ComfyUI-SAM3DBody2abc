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
                    "tooltip": "Center mesh at origin"
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
        include_joints: bool = True,
        scale: float = 1.0,
        up_axis: str = "Y",
        center_mesh: bool = False,
        temporal_smoothing: float = 0.5,
    ) -> Tuple[str, str, int]:
        """
        Export complete animation to single Alembic file.
        """
        # Setup output path
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{filename}.abc")
        
        # Handle existing files
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{filename}_{counter:04d}.abc")
            counter += 1
        
        # Filter valid frames
        valid_frames = [f for f in mesh_sequence if f.get("valid") and f.get("vertices") is not None]
        
        if not valid_frames:
            return (output_path, "Error: No valid mesh data", 0)
        
        # Apply temporal smoothing to reduce jitter
        if temporal_smoothing > 0:
            valid_frames = self._apply_temporal_smoothing(valid_frames, temporal_smoothing)
        
        # Try native Alembic export (PyAlembic - not SQLAlchemy alembic!)
        try:
            # Check for the correct PyAlembic package (not SQLAlchemy's alembic)
            from alembic import Abc, AbcGeom
            import imath
            print("[ExportAnimatedAlembic] PyAlembic found, using native export...")
            result = self._export_native_alembic(
                valid_frames, output_path, fps, include_joints, scale, up_axis, center_mesh
            )
            return result
        except ImportError as e:
            # Could be missing PyAlembic entirely, or have wrong package (SQLAlchemy)
            print(f"[ExportAnimatedAlembic] PyAlembic not available ({e}), trying Blender...")
        except Exception as e:
            print(f"[ExportAnimatedAlembic] PyAlembic export failed: {e}, trying Blender...")
        
        # Try Blender export
        try:
            result = self._export_via_blender(
                valid_frames, output_path, fps, include_joints, scale, up_axis, center_mesh
            )
            return result
        except Exception as e:
            print(f"[ExportAnimatedAlembic] Blender export failed: {e}")
        
        # Fallback to OBJ sequence with animation metadata
        return self._export_with_metadata(
            valid_frames, output_path, fps, scale, up_axis, center_mesh
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
        """Find Blender executable."""
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
                import folder_paths
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
                        print(f"[SAM3DBody2abc] Found Blender: {matches[0]}")
                        locations.extend(matches)
        except Exception as e:
            print(f"[SAM3DBody2abc] Error searching for Blender: {e}")
        
        # Windows paths
        for version in ["4.2", "4.1", "4.0", "3.6"]:
            locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {version}\\blender.exe")
        
        for loc in locations:
            if loc and os.path.exists(loc):
                return loc
        
        return None
    
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


class ExportAnimatedFBX:
    """
    Export animated skeleton to FBX format.
    Creates a SINGLE FBX file with animated skeleton across ALL frames.
    
    Uses the MHR (Momentum Human Rig) 127-joint skeleton hierarchy.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "filename": ("STRING", {
                    "default": "body_skeleton",
                    "multiline": False
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0
                }),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0}),
                "include_mesh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include mesh geometry with skeleton"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("file_path", "status", "exported_frames")
    FUNCTION = "export_fbx"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_fbx(
        self,
        mesh_sequence: List[Dict],
        filename: str = "body_skeleton",
        fps: float = 24.0,
        output_dir: str = "",
        scale: float = 1.0,
        include_mesh: bool = False,
    ) -> Tuple[str, str, int]:
        """Export animated skeleton to FBX."""
        
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{filename}.fbx")
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{filename}_{counter:04d}.fbx")
            counter += 1
        
        # Filter frames with joint data
        valid_frames = [
            f for f in mesh_sequence 
            if f.get("valid") and f.get("joints") is not None
        ]
        
        if not valid_frames:
            return (output_path, "Error: No valid joint data", 0)
        
        # Try Blender export
        try:
            result = self._export_via_blender(
                valid_frames, output_path, fps, scale, include_mesh
            )
            return result
        except Exception as e:
            print(f"[ExportAnimatedFBX] Blender export failed: {e}")
        
        # Fallback to JSON skeleton data
        return self._export_skeleton_json(valid_frames, output_path, fps, scale)
    
    def _export_via_blender(
        self,
        frames: List[Dict],
        output_path: str,
        fps: float,
        scale: float,
        include_mesh: bool,
    ) -> Tuple[str, str, int]:
        """Export using Blender."""
        import subprocess
        import tempfile
        
        blender_path = self._find_blender()
        if not blender_path:
            raise RuntimeError("Blender not found")
        
        # Get MHR joint hierarchy from first frame (127 joints)
        joint_parents = frames[0].get("joint_parents")
        if joint_parents is None:
            print("[ExportAnimatedFBX] WARNING: No joint_parents in data, trying to load from MHR model...")
            # Try to load from HuggingFace cache
            try:
                import glob
                hf_cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam-3d-body-dinov3")
                if os.path.exists(hf_cache_base):
                    pattern = os.path.join(hf_cache_base, "snapshots", "*", "assets", "mhr_model.pt")
                    matches = glob.glob(pattern)
                    if matches:
                        matches.sort(key=os.path.getmtime, reverse=True)
                        mhr_path = matches[0]
                        import torch
                        mhr_model = torch.jit.load(mhr_path, map_location='cpu')
                        parent_tensor = mhr_model.character_torch.skeleton.joint_parents
                        joint_parents = parent_tensor.cpu().numpy().astype(int).tolist()
                        print(f"[ExportAnimatedFBX] Loaded MHR hierarchy: {len(joint_parents)} joints")
            except Exception as e:
                print(f"[ExportAnimatedFBX] Failed to load MHR model: {e}")
        
        if joint_parents is None:
            # Use anatomically-correct MHR skeleton hierarchy
            num_joints = len(frames[0]["joints"])
            joint_parents = self._get_mhr_joint_parents(num_joints)
            print(f"[ExportAnimatedFBX] Using MHR anatomical hierarchy: {num_joints} joints")
        
        num_joints = len(joint_parents)
        joint_names = [f"Joint_{i:03d}" for i in range(num_joints)]
        
        print(f"[ExportAnimatedFBX] Exporting MHR skeleton with {num_joints} joints")
        root_idx = joint_parents.index(-1) if -1 in joint_parents else 0
        print(f"[ExportAnimatedFBX] Root joint at index: {root_idx}")
        
        # Prepare export data - Blender script handles coordinate transforms
        export_data = {
            "frames": [],
            "output_path": output_path,
            "fps": fps,
            "scale": scale,
            "include_mesh": include_mesh,
            "joint_names": joint_names,
            "joint_parents": joint_parents if isinstance(joint_parents, list) else list(joint_parents),
        }
        
        for frame in frames:
            joints = frame.get("joints")
            if joints is None:
                continue
            
            # Pass raw joint data - Blender script handles coordinate transform
            joints_array = np.array(joints)
            
            frame_export = {
                "joints": joints_array.tolist(),
            }
            if include_mesh and frame.get("vertices") is not None:
                verts = np.array(frame["vertices"])
                frame_export["vertices"] = verts.tolist()
                if frame.get("faces") is not None:
                    frame_export["faces"] = np.array(frame["faces"]).tolist()
            export_data["frames"].append(frame_export)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f)
            data_path = f.name
        
        script = self._create_blender_fbx_script()
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
            
            print(f"[ExportAnimatedFBX] Blender stdout: {result.stdout[-500:] if result.stdout else 'None'}")
            if result.stderr:
                print(f"[ExportAnimatedFBX] Blender stderr: {result.stderr[-500:]}")
            
            if result.returncode == 0 and os.path.exists(output_path):
                return (output_path, f"Exported {len(frames)} frames to FBX", len(frames))
            else:
                raise RuntimeError(f"Blender FBX export failed: {result.stderr}")
        finally:
            os.unlink(data_path)
            os.unlink(script_path)
    
    def _export_skeleton_json(
        self,
        frames: List[Dict],
        output_path: str,
        fps: float,
        scale: float,
    ) -> Tuple[str, str, int]:
        """Fallback: Export skeleton as JSON."""
        
        json_path = output_path.replace('.fbx', '_skeleton.json')
        
        skeleton_data = {
            "fps": fps,
            "scale": scale,
            "joint_names": self.JOINT_NAMES,
            "joint_parents": self.JOINT_PARENTS,
            "frames": []
        }
        
        for frame in frames:
            joints = np.array(frame["joints"]) * scale
            skeleton_data["frames"].append({
                "joints": joints.tolist(),
                "pose": np.array(frame.get("pose", [])).tolist() if frame.get("pose") is not None else None,
            })
        
        with open(json_path, 'w') as f:
            json.dump(skeleton_data, f, indent=2)
        
        return (json_path, f"Exported {len(frames)} frames as skeleton JSON (FBX requires Blender)", len(frames))
    
    def _get_mhr_joint_parents(self, num_joints: int) -> List[int]:
        """
        Get anatomically-correct joint parent hierarchy for MHR skeleton.
        
        MHR70 joint mapping (first 70 joints):
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        9: left_hip, 10: right_hip, 11: left_knee, 12: right_knee
        13: left_ankle, 14: right_ankle, 15-17: left foot, 18-20: right foot
        21-41: right hand (wrist at 41), 42-62: left hand (wrist at 62)
        63-68: additional arm points, 69: neck
        70+: additional keypoints (face, etc.)
        
        Hierarchy uses left_hip (9) as root connecting upper and lower body.
        """
        # Base hierarchy for first 70 joints (MHR70)
        mhr70_parents = [
            69,   # 0: nose -> neck
            0,    # 1: left_eye -> nose
            0,    # 2: right_eye -> nose
            1,    # 3: left_ear -> left_eye
            2,    # 4: right_ear -> right_eye
            69,   # 5: left_shoulder -> neck
            69,   # 6: right_shoulder -> neck
            5,    # 7: left_elbow -> left_shoulder
            6,    # 8: right_elbow -> right_shoulder
            -1,   # 9: left_hip -> ROOT
            9,    # 10: right_hip -> left_hip (pelvis connection)
            9,    # 11: left_knee -> left_hip
            10,   # 12: right_knee -> right_hip
            11,   # 13: left_ankle -> left_knee
            12,   # 14: right_ankle -> right_knee
            13,   # 15: left_big_toe -> left_ankle
            13,   # 16: left_small_toe -> left_ankle
            13,   # 17: left_heel -> left_ankle
            14,   # 18: right_big_toe -> right_ankle
            14,   # 19: right_small_toe -> right_ankle
            14,   # 20: right_heel -> right_ankle
            # Right hand finger chain (21-41)
            22,   # 21: right_thumb_tip -> 22
            23,   # 22: right_thumb_first -> 23
            24,   # 23: right_thumb_second -> 24
            41,   # 24: right_thumb_third -> wrist
            26,   # 25: right_index_tip -> 26
            27,   # 26: right_index_first -> 27
            28,   # 27: right_index_second -> 28
            41,   # 28: right_index_third -> wrist
            30,   # 29: right_middle_tip -> 30
            31,   # 30: right_middle_first -> 31
            32,   # 31: right_middle_second -> 32
            41,   # 32: right_middle_third -> wrist
            34,   # 33: right_ring_tip -> 34
            35,   # 34: right_ring_first -> 35
            36,   # 35: right_ring_second -> 36
            41,   # 36: right_ring_third -> wrist
            38,   # 37: right_pinky_tip -> 38
            39,   # 38: right_pinky_first -> 39
            40,   # 39: right_pinky_second -> 40
            41,   # 40: right_pinky_third -> wrist
            8,    # 41: right_wrist -> right_elbow
            # Left hand finger chain (42-62)
            43,   # 42: left_thumb_tip -> 43
            44,   # 43: left_thumb_first -> 44
            45,   # 44: left_thumb_second -> 45
            62,   # 45: left_thumb_third -> wrist
            47,   # 46: left_index_tip -> 47
            48,   # 47: left_index_first -> 48
            49,   # 48: left_index_second -> 49
            62,   # 49: left_index_third -> wrist
            51,   # 50: left_middle_tip -> 51
            52,   # 51: left_middle_first -> 52
            53,   # 52: left_middle_second -> 53
            62,   # 53: left_middle_third -> wrist
            55,   # 54: left_ring_tip -> 55
            56,   # 55: left_ring_first -> 56
            57,   # 56: left_ring_second -> 57
            62,   # 57: left_ring_third -> wrist
            59,   # 58: left_pinky_tip -> 59
            60,   # 59: left_pinky_first -> 60
            61,   # 60: left_pinky_second -> 61
            62,   # 61: left_pinky_third -> wrist
            7,    # 62: left_wrist -> left_elbow
            # Additional arm joints (63-68)
            7,    # 63: left_olecranon -> left_elbow
            8,    # 64: right_olecranon -> right_elbow
            7,    # 65: left_cubital_fossa -> left_elbow
            8,    # 66: right_cubital_fossa -> right_elbow
            5,    # 67: left_acromion -> left_shoulder
            6,    # 68: right_acromion -> right_shoulder
            # Neck connects upper body to hips
            9,    # 69: neck -> left_hip (connects upper/lower body)
        ]
        
        if num_joints <= 70:
            return mhr70_parents[:num_joints]
        
        # For 127 joints, additional joints (70-126) are likely face keypoints
        # Connect them to the nose (0) or neck (69)
        joint_parents = mhr70_parents.copy()
        for i in range(70, num_joints):
            joint_parents.append(0)  # Face keypoints -> nose
        
        return joint_parents
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable."""
        import shutil
        import glob
        
        locations = [
            shutil.which("blender"),
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
        ]
        
        # ComfyUI SAM3DBody bundled Blender
        try:
            comfy_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sam3d_blender_patterns = [
                os.path.join(comfy_base, "ComfyUI-SAM3DBody", "lib", "blender", "blender-*", "blender"),
                os.path.join(comfy_base, "ComfyUI-SAM3DBody", "lib", "blender", "*/blender"),
            ]
            for pattern in sam3d_blender_patterns:
                matches = glob.glob(pattern)
                locations.extend(matches)
        except:
            pass
        
        for version in ["4.2", "4.1", "4.0", "3.6"]:
            locations.append(f"C:\\Program Files\\Blender Foundation\\Blender {version}\\blender.exe")
        
        for loc in locations:
            if loc and os.path.exists(loc):
                return loc
        return None
    
    def _create_blender_fbx_script(self) -> str:
        """Create Blender script for animated FBX skeleton export."""
        return '''
import bpy
import json
import sys
import numpy as np
from mathutils import Vector

argv = sys.argv
data_path = argv[argv.index("--") + 1]

print(f"FBX export starting... loading {data_path}")

with open(data_path, 'r') as f:
    data = json.load(f)

frames = data["frames"]
output_path = data["output_path"]
fps = data["fps"]
scale = data["scale"]
joint_names = data["joint_names"]
joint_parents = data["joint_parents"]

num_joints = len(joint_names)
num_frames = len(frames)

print(f"FBX export starting... '{output_path}'")
print(f"Joints: {num_joints}, Frames: {num_frames}")

# Clean scene
for c in bpy.data.actions:
    bpy.data.actions.remove(c)
for c in bpy.data.armatures:
    bpy.data.armatures.remove(c)
for c in bpy.data.objects:
    bpy.data.objects.remove(c)
for c in bpy.data.meshes:
    bpy.data.meshes.remove(c)

# Create collection
collection = bpy.data.collections.new('SAM3D_Export')
bpy.context.scene.collection.children.link(collection)

# Set FPS and frame range
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = num_frames

# Find root joint(s)
root_indices = [i for i, p in enumerate(joint_parents) if p == -1]
print(f"Root joint(s): {root_indices}")

# Create empties for each joint (NOT parented - we'll animate world positions)
empties = {}
for i, name in enumerate(joint_names):
    bpy.ops.object.empty_add(type='SPHERE', radius=0.02 * scale)
    empty = bpy.context.active_object
    empty.name = name
    empty.empty_display_size = 0.02 * scale
    
    # Move to collection
    if empty.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(empty)
    collection.objects.link(empty)
    
    empties[name] = empty

# Animate empties with world positions (no parenting = simpler and correct)
for frame_idx, frame_data in enumerate(frames):
    bpy.context.scene.frame_set(frame_idx + 1)
    
    joints_raw = np.array(frame_data["joints"])
    
    # Apply coordinate correction: (x, y, z) -> (x, -z, y)
    # This converts from Y-up to Blender's Z-up coordinate system
    joints = np.zeros_like(joints_raw)
    joints[:, 0] = joints_raw[:, 0] * scale      # X stays X
    joints[:, 1] = -joints_raw[:, 2] * scale     # Y = -Z (original)
    joints[:, 2] = joints_raw[:, 1] * scale      # Z = Y (original, the height)
    
    for i, name in enumerate(joint_names):
        empty = empties[name]
        empty.location = Vector((joints[i, 0], joints[i, 1], joints[i, 2]))
        empty.keyframe_insert(data_path="location", frame=frame_idx + 1)
    
    if (frame_idx + 1) % 50 == 0:
        print(f"Animated frame {frame_idx + 1}/{num_frames}")

# Create armature from first frame for visual reference
print("Creating reference armature...")

# Get first frame joint positions (transformed)
first_joints_raw = np.array(frames[0]["joints"])
first_joints = np.zeros_like(first_joints_raw)
first_joints[:, 0] = first_joints_raw[:, 0] * scale
first_joints[:, 1] = -first_joints_raw[:, 2] * scale
first_joints[:, 2] = first_joints_raw[:, 1] * scale

bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
armature = bpy.data.armatures.get('Armature')
armature.name = 'SAM3D_Skeleton'
armature_obj = bpy.context.active_object
armature_obj.name = 'SAM3D_Skeleton'

# Move to collection
if armature_obj.name in bpy.context.scene.collection.objects:
    bpy.context.scene.collection.objects.unlink(armature_obj)
collection.objects.link(armature_obj)

edit_bones = armature.edit_bones

# Remove default bone
default_bone = edit_bones.get('Bone')
if default_bone:
    edit_bones.remove(default_bone)

# Create bones
bones_dict = {}
for i, name in enumerate(joint_names):
    bone = edit_bones.new(name)
    pos = Vector((first_joints[i, 0], first_joints[i, 1], first_joints[i, 2]))
    bone.head = pos
    
    # Find first child to determine tail direction
    children = [j for j, p in enumerate(joint_parents) if p == i]
    if children:
        child_pos = Vector((first_joints[children[0], 0], first_joints[children[0], 1], first_joints[children[0], 2]))
        direction = child_pos - pos
        if direction.length > 0.001:
            bone.tail = pos + direction.normalized() * min(direction.length * 0.5, 0.05 * scale)
        else:
            bone.tail = pos + Vector((0, 0, 0.03 * scale))
    else:
        # Leaf bone - point up (Z in Blender)
        bone.tail = pos + Vector((0, 0, 0.03 * scale))
    
    bones_dict[name] = bone

# Set bone parents
for i, parent_idx in enumerate(joint_parents):
    if parent_idx >= 0 and parent_idx < num_joints:
        bones_dict[joint_names[i]].parent = bones_dict[joint_names[parent_idx]]
        bones_dict[joint_names[i]].use_connect = False

bpy.ops.object.mode_set(mode='OBJECT')

# Add bone constraints to follow empties
bpy.ops.object.mode_set(mode='POSE')
for i, name in enumerate(joint_names):
    pose_bone = armature_obj.pose.bones[name]
    constraint = pose_bone.constraints.new('COPY_LOCATION')
    constraint.target = empties[name]
    constraint.name = "Follow_Empty"

bpy.ops.object.mode_set(mode='OBJECT')

# Select all objects for export
for obj in bpy.context.selected_objects:
    obj.select_set(False)

for obj in collection.objects:
    obj.select_set(True)

bpy.context.view_layer.objects.active = armature_obj

# Export FBX with baked animation
print("Exporting FBX...")
bpy.ops.export_scene.fbx(
    filepath=output_path,
    check_existing=False,
    use_selection=True,
    object_types={'ARMATURE', 'EMPTY'},
    add_leaf_bones=False,
    bake_anim=True,
    bake_anim_use_all_actions=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_bones=True,
    bake_anim_force_startend_keying=True,
)

print(f"SUCCESS: Exported {num_frames} frames to {output_path}")
'''


class ExportAnimatedMesh:
    """
    Combined export node - exports both Alembic geometry and FBX skeleton.
    Convenience node for complete animation export workflow.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "filename": ("STRING", {"default": "body_animation", "multiline": False}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "", "multiline": False}),
                "export_alembic": ("BOOLEAN", {"default": True}),
                "export_fbx": ("BOOLEAN", {"default": True}),
                "export_obj_sequence": ("BOOLEAN", {"default": False}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("alembic_path", "fbx_path", "status")
    FUNCTION = "export_all"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_all(
        self,
        mesh_sequence: List[Dict],
        filename: str = "body_animation",
        fps: float = 24.0,
        output_dir: str = "",
        export_alembic: bool = True,
        export_fbx: bool = True,
        export_obj_sequence: bool = False,
        scale: float = 1.0,
    ) -> Tuple[str, str, str]:
        """Export to multiple formats."""
        
        abc_path = ""
        fbx_path = ""
        status_parts = []
        
        if export_alembic:
            exporter = ExportAnimatedAlembic()
            abc_path, status, count = exporter.export_alembic(
                mesh_sequence, f"{filename}_mesh", fps,
                output_dir, scale=scale
            )
            status_parts.append(f"ABC: {status}")
        
        if export_fbx:
            exporter = ExportAnimatedFBX()
            fbx_path, status, count = exporter.export_fbx(
                mesh_sequence, f"{filename}_skeleton", fps,
                output_dir, scale=scale
            )
            status_parts.append(f"FBX: {status}")
        
        return (abc_path, fbx_path, " | ".join(status_parts))


class ExportOBJSequence:
    """
    Export mesh sequence as OBJ file sequence.
    
    Creates numbered OBJ files (frame_000001.obj, frame_000002.obj, etc.)
    that can be imported into any 3D software using Stop Motion OBJ addon
    or similar mesh sequence importers.
    
    This is the most reliable export format for animated meshes.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE",),
                "filename_prefix": ("STRING", {
                    "default": "body",
                    "multiline": False,
                    "tooltip": "Prefix for OBJ files (e.g., 'body' -> body_000001.obj)"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Leave empty for ComfyUI output folder"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 1000.0,
                    "step": 0.01,
                }),
                "flip_yz": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swap Y and Z axes (for Z-up software)"
                }),
                "write_mtl": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Write .mtl material files"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output_dir", "status", "frame_count")
    FUNCTION = "export_obj_sequence"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
    def export_obj_sequence(
        self,
        mesh_sequence: List[Dict],
        filename_prefix: str = "body",
        output_dir: str = "",
        scale: float = 1.0,
        flip_yz: bool = False,
        write_mtl: bool = False,
    ) -> Tuple[str, str, int]:
        """Export mesh sequence as numbered OBJ files."""
        
        # Setup output directory
        if not output_dir:
            output_dir = folder_paths.get_output_directory()
        
        # Create subdirectory for OBJ sequence
        seq_dir = os.path.join(output_dir, f"{filename_prefix}_obj_sequence")
        os.makedirs(seq_dir, exist_ok=True)
        
        # Filter valid frames
        valid_frames = [f for f in mesh_sequence if f.get("valid") and f.get("vertices") is not None]
        
        if not valid_frames:
            return (seq_dir, "Error: No valid mesh data", 0)
        
        # Get reference faces (constant topology)
        ref_faces = valid_frames[0].get("faces")
        if ref_faces is None:
            return (seq_dir, "Error: No face data", 0)
        ref_faces = np.array(ref_faces)
        
        print(f"[ExportOBJSequence] Exporting {len(valid_frames)} frames to {seq_dir}")
        
        for idx, frame_data in enumerate(valid_frames):
            vertices = np.array(frame_data["vertices"]) * scale
            faces = frame_data.get("faces")
            if faces is None:
                faces = ref_faces
            else:
                faces = np.array(faces)
            
            # Apply axis flip if needed
            if flip_yz:
                vertices = vertices[:, [0, 2, 1]]  # Swap Y and Z
                vertices[:, 2] = -vertices[:, 2]   # Flip new Z
            
            # Write OBJ file
            frame_num = idx + 1
            obj_path = os.path.join(seq_dir, f"{filename_prefix}_{frame_num:06d}.obj")
            
            with open(obj_path, 'w') as f:
                f.write(f"# SAM3DBody2abc OBJ Export - Frame {frame_num}\n")
                f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
                
                if write_mtl:
                    mtl_name = f"{filename_prefix}_{frame_num:06d}.mtl"
                    f.write(f"mtllib {mtl_name}\n")
                    f.write(f"usemtl body_material\n")
                
                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Write faces (1-indexed)
                for face in faces:
                    f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")
            
            # Write MTL if requested
            if write_mtl:
                mtl_path = os.path.join(seq_dir, f"{filename_prefix}_{frame_num:06d}.mtl")
                with open(mtl_path, 'w') as f:
                    f.write("# SAM3DBody2abc MTL\n")
                    f.write("newmtl body_material\n")
                    f.write("Kd 0.8 0.6 0.5\n")  # Skin-ish color
                    f.write("Ka 0.2 0.2 0.2\n")
                    f.write("Ks 0.3 0.3 0.3\n")
                    f.write("Ns 50.0\n")
            
            if (idx + 1) % 50 == 0:
                print(f"[ExportOBJSequence] Written {idx + 1}/{len(valid_frames)} files...")
        
        # Write info file for importers
        info_path = os.path.join(seq_dir, "sequence_info.json")
        import json
        with open(info_path, 'w') as f:
            json.dump({
                "prefix": filename_prefix,
                "start_frame": 1,
                "end_frame": len(valid_frames),
                "total_frames": len(valid_frames),
                "padding": 6,
                "extension": "obj",
                "pattern": f"{filename_prefix}_######.obj",
                "vertex_count": len(valid_frames[0]["vertices"]),
                "face_count": len(ref_faces),
            }, f, indent=2)
        
        status = f"Exported {len(valid_frames)} OBJ files"
        print(f"[ExportOBJSequence] {status}")
        
        return (seq_dir, status, len(valid_frames))
