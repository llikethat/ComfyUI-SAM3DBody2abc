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
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("file_path", "status", "exported_frames")
    FUNCTION = "export_alembic"
    CATEGORY = "SAM3DBody2abc/Export"
    OUTPUT_NODE = True
    
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
        
        # Try native Alembic export
        try:
            result = self._export_native_alembic(
                valid_frames, output_path, fps, include_joints, scale, up_axis, center_mesh
            )
            return result
        except ImportError:
            print("[ExportAnimatedAlembic] Native Alembic not available, trying Blender export...")
        
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
        """Export using native PyAlembic."""
        import alembic
        from alembic import Abc, AbcGeom
        import imath
        
        # Create archive
        archive = Abc.OArchive(output_path)
        top = archive.getTop()
        
        # Time sampling
        time_sampling = Abc.TimeSampling(1.0 / fps, 0.0)
        ts_idx = archive.addTimeSampling(time_sampling)
        
        # Create mesh object
        mesh_obj = AbcGeom.OPolyMesh(top, "body_mesh")
        mesh_schema = mesh_obj.getSchema()
        mesh_schema.setTimeSampling(ts_idx)
        
        # Create joints object if requested
        joints_obj = None
        if include_joints:
            joints_obj = AbcGeom.OPoints(top, "body_joints")
            joints_schema = joints_obj.getSchema()
            joints_schema.setTimeSampling(ts_idx)
        
        # Get reference faces from first frame
        ref_faces = frames[0].get("faces")
        if ref_faces is not None:
            ref_faces = np.array(ref_faces)
        
        # Axis conversion
        axis_transform = self._get_axis_transform(up_axis)
        
        # Write each frame
        for frame_data in frames:
            vertices = np.array(frame_data["vertices"]) * scale
            
            # Center if requested
            if center_mesh:
                vertices = vertices - np.mean(vertices, axis=0)
            
            # Apply axis transform
            if axis_transform is not None:
                vertices = vertices @ axis_transform.T
            
            # Convert to imath
            vertex_array = imath.V3fArray(len(vertices))
            for i, v in enumerate(vertices):
                vertex_array[i] = imath.V3f(float(v[0]), float(v[1]), float(v[2]))
            
            # Face data
            faces = frame_data.get("faces") or ref_faces
            if faces is not None:
                faces = np.array(faces)
                face_indices = faces.flatten().tolist()
                face_counts = [3] * len(faces)  # Assuming triangles
            else:
                face_indices = list(range(len(vertices)))
                face_counts = [3] * (len(vertices) // 3)
            
            # Create sample
            mesh_sample = AbcGeom.OPolyMeshSchemaSample(
                vertex_array,
                imath.IntArray(face_indices),
                imath.IntArray(face_counts)
            )
            mesh_schema.set(mesh_sample)
            
            # Write joints if available
            if include_joints and joints_obj is not None:
                joints = frame_data.get("joints")
                if joints is not None:
                    joints = np.array(joints) * scale
                    if center_mesh:
                        joints = joints - np.mean(np.array(frame_data["vertices"]) * scale, axis=0)
                    if axis_transform is not None:
                        joints = joints @ axis_transform.T
                    
                    joint_array = imath.V3fArray(len(joints))
                    for i, j in enumerate(joints):
                        joint_array[i] = imath.V3f(float(j[0]), float(j[1]), float(j[2]))
                    
                    joints_sample = AbcGeom.OPointsSchemaSample(joint_array)
                    joints_schema = joints_obj.getSchema()
                    joints_schema.set(joints_sample)
        
        return (output_path, f"Exported {len(frames)} frames to Alembic", len(frames))
    
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
        """Create Blender Python script for animated Alembic export."""
        return '''
import bpy
import json
import sys
import bmesh

argv = sys.argv
data_path = argv[argv.index("--") + 1]

with open(data_path, 'r') as f:
    data = json.load(f)

frames = data["frames"]
output_path = data["output_path"]
fps = data["fps"]
scale = data["scale"]

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Set FPS
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = len(frames)

# Create mesh from first frame
first_verts = [[v * scale for v in vert] for vert in frames[0]["vertices"]]
first_faces = frames[0].get("faces", [])

mesh = bpy.data.meshes.new("body_mesh")
obj = bpy.data.objects.new("body_mesh", mesh)
bpy.context.collection.objects.link(obj)

if first_faces:
    mesh.from_pydata(first_verts, [], first_faces)
else:
    mesh.from_pydata(first_verts, [], [])
mesh.update()

# Add basis shape key
obj.shape_key_add(name="Basis", from_mix=False)

# Add shape key for each frame and keyframe it
for idx, frame_data in enumerate(frames):
    sk = obj.shape_key_add(name=f"frame_{idx:05d}", from_mix=False)
    
    verts = frame_data["vertices"]
    for i, v in enumerate(verts):
        if i < len(sk.data):
            sk.data[i].co = [v[0] * scale, v[1] * scale, v[2] * scale]
    
    # Keyframe animation
    frame_num = idx + 1
    
    # Turn off all shape keys
    for key in obj.data.shape_keys.key_blocks:
        if key.name != "Basis":
            key.value = 0.0
            key.keyframe_insert(data_path="value", frame=frame_num)
    
    # Turn on this frame's shape key
    sk.value = 1.0
    sk.keyframe_insert(data_path="value", frame=frame_num)

# Select for export
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# Export to Alembic
bpy.ops.wm.alembic_export(
    filepath=output_path,
    selected=True,
    start=1,
    end=len(frames),
    flatten=False,
    export_hair=False,
    export_particles=False,
    apply_subdiv=False,
)

print(f"SUCCESS: Exported {len(frames)} frames to {output_path}")
'''


class ExportAnimatedFBX:
    """
    Export animated skeleton to FBX format.
    Creates a SINGLE FBX file with animated skeleton across ALL frames.
    
    The skeleton follows SMPL joint hierarchy (24 joints) and can be
    imported into any 3D software for retargeting to other characters.
    """
    
    # SMPL joint names
    JOINT_NAMES = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
    ]
    
    # SMPL skeleton hierarchy (parent indices)
    JOINT_PARENTS = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
    ]
    
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
        
        # Prepare data
        export_data = {
            "frames": [],
            "output_path": output_path,
            "fps": fps,
            "scale": scale,
            "include_mesh": include_mesh,
            "joint_names": self.JOINT_NAMES,
            "joint_parents": self.JOINT_PARENTS,
        }
        
        for frame in frames:
            frame_export = {
                "joints": np.array(frame["joints"]).tolist(),
            }
            if include_mesh and frame.get("vertices") is not None:
                frame_export["vertices"] = np.array(frame["vertices"]).tolist()
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
from mathutils import Vector, Matrix

argv = sys.argv
data_path = argv[argv.index("--") + 1]

with open(data_path, 'r') as f:
    data = json.load(f)

frames = data["frames"]
output_path = data["output_path"]
fps = data["fps"]
scale = data["scale"]
joint_names = data["joint_names"]
joint_parents = data["joint_parents"]

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Set FPS
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = len(frames)

# Create armature
bpy.ops.object.armature_add()
armature_obj = bpy.context.active_object
armature_obj.name = "body_skeleton"
armature = armature_obj.data
armature.name = "body_armature"

# Enter edit mode
bpy.ops.object.mode_set(mode='EDIT')

# Remove default bone
for bone in armature.edit_bones:
    armature.edit_bones.remove(bone)

# Create bones from first frame
first_joints = [[j * scale for j in joint] for joint in frames[0]["joints"]]

bones = []
for i, name in enumerate(joint_names):
    bone = armature.edit_bones.new(name)
    pos = Vector(first_joints[i])
    bone.head = pos
    
    # Find children to determine bone direction
    children = [j for j, p in enumerate(joint_parents) if p == i]
    if children:
        child_pos = Vector(first_joints[children[0]])
        bone.tail = child_pos
    else:
        parent_idx = joint_parents[i]
        if parent_idx >= 0:
            parent_pos = Vector(first_joints[parent_idx])
            direction = pos - parent_pos
            bone.tail = pos + direction.normalized() * 0.05 * scale
        else:
            bone.tail = pos + Vector((0, 0.1 * scale, 0))
    
    bones.append(bone)

# Set parents
for i, parent_idx in enumerate(joint_parents):
    if parent_idx >= 0:
        bones[i].parent = bones[parent_idx]

bpy.ops.object.mode_set(mode='OBJECT')

# Animate the armature
bpy.ops.object.mode_set(mode='POSE')

for frame_idx, frame_data in enumerate(frames):
    bpy.context.scene.frame_set(frame_idx + 1)
    
    joints = frame_data["joints"]
    
    for i, name in enumerate(joint_names):
        pose_bone = armature_obj.pose.bones[name]
        pos = Vector([j * scale for j in joints[i]])
        
        # Set bone location (relative to parent)
        if joint_parents[i] >= 0:
            parent_pos = Vector([j * scale for j in joints[joint_parents[i]]])
            local_pos = pos - parent_pos
        else:
            local_pos = pos
        
        pose_bone.location = local_pos
        pose_bone.keyframe_insert(data_path="location", frame=frame_idx + 1)

bpy.ops.object.mode_set(mode='OBJECT')

# Select armature for export
armature_obj.select_set(True)
bpy.context.view_layer.objects.active = armature_obj

# Export FBX
bpy.ops.export_scene.fbx(
    filepath=output_path,
    use_selection=True,
    object_types={'ARMATURE'},
    add_leaf_bones=False,
    bake_anim=True,
    bake_anim_use_all_actions=False,
    bake_anim_use_nla_strips=False,
)

print(f"SUCCESS: Exported {len(frames)} frames to {output_path}")
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
