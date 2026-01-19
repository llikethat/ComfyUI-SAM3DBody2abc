"""
BPY Exporter Module for SAM3DBody2abc

This module provides a direct Python interface to the Blender export functionality,
allowing it to be called without subprocess when running in an isolated environment
with bpy installed.

The module imports and wraps all functions from blender_animated_fbx.py, providing
a clean API for the FBX export node.

Usage:
    from lib.bpy_exporter import export_animated_fbx
    
    result = export_animated_fbx(
        export_data=data_dict,
        output_path="/path/to/output.fbx",
        up_axis="Y",
        include_mesh=True,
        include_camera=True
    )
"""

import os
import sys

# Add this directory to path so we can import the blender script
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


def export_animated_fbx(export_data: dict, output_path: str, up_axis: str = "Y",
                        include_mesh: bool = True, include_camera: bool = True) -> dict:
    """
    Export animated FBX using bpy directly.
    
    This function runs the same logic as blender_animated_fbx.py but without
    needing to spawn a subprocess. It requires bpy to be available in the
    Python environment.
    
    Args:
        export_data: Dictionary containing all export parameters:
            - fps: Frame rate
            - frames: List of frame data dicts
            - faces: Face indices for mesh
            - joint_parents: Joint hierarchy
            - sensor_width: Camera sensor width in mm
            - world_translation_mode: How to handle translation
            - skeleton_mode: "rotations" or "positions"
            - flip_x: Mirror on X axis
            - align_mesh_to_skeleton: Align mesh to skeleton origin
            - frame_offset: Starting frame number
            - include_skeleton: Include armature
            - animate_camera: Animate camera position
            - camera_follow_root: Parent camera to root
            - camera_use_rotation: Use rotation for camera
            - camera_static: Static camera
            - camera_compensation: Bake inverse extrinsics to root
            - use_depth_positioning: Use depth for positioning
            - depth_mode: "position", "scale", or "both"
            - scale_factor: Scale factor for world coordinates
            - camera_extrinsics: Camera rotation data
            - camera_intrinsics: Camera intrinsics from MoGe2
            - extrinsics_smoothing_method: Smoothing method
            - extrinsics_smoothing_strength: Smoothing strength
            - metadata: Metadata to embed in FBX
            - body_world_trajectory: World trajectory data
            - body_world_trajectory_compensated: Compensated trajectory
        output_path: Full path for output file (.fbx or .abc)
        up_axis: Up axis - "Y", "Z", "-Y", or "-Z"
        include_mesh: Include mesh in export
        include_camera: Include camera in export
    
    Returns:
        dict with:
            - status: "success" or "error"
            - message: Error message if failed
            - path: Output file path
            - frame_count: Number of frames exported
            - fps: Frame rate used
            - file_size_mb: File size in MB
    """
    import bpy
    import json
    import numpy as np
    from mathutils import Vector, Matrix, Euler, Quaternion
    import math
    from datetime import datetime
    
    # =========================================================================
    # EMBEDDED LOGGER
    # =========================================================================
    class LogLevel:
        SILENT = 0
        ERROR = 1
        WARN = 2
        INFO = 3
        STATUS = 4
        DEBUG = 5

    class Log:
        def __init__(self, level=LogLevel.INFO):
            self.level = level
            self.prefix = "BPY Export"
        
        def _ts(self):
            return datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        def error(self, msg): 
            if self.level >= LogLevel.ERROR: print(f"[{self._ts()}] [{self.prefix}] ERROR: {msg}")
        
        def warn(self, msg): 
            if self.level >= LogLevel.WARN: print(f"[{self._ts()}] [{self.prefix}] WARN: {msg}")
        
        def info(self, msg): 
            if self.level >= LogLevel.INFO: print(f"[{self._ts()}] [{self.prefix}] {msg}")
        
        def status(self, msg): 
            if self.level >= LogLevel.STATUS: print(f"[{self._ts()}] [{self.prefix}] {msg}")
        
        def debug(self, msg): 
            if self.level >= LogLevel.DEBUG: print(f"[{self._ts()}] [{self.prefix}] DEBUG: {msg}")
        
        def progress(self, current, total, task="", interval=10):
            if self.level < LogLevel.STATUS: return
            if current == 0 or current == total - 1 or (current + 1) % interval == 0:
                pct = (current + 1) / total * 100
                msg = f"{task}: {current + 1}/{total} ({pct:.0f}%)" if task else f"Progress: {current + 1}/{total}"
                print(f"[{self._ts()}] [{self.prefix}] {msg}")

    _log_level = os.environ.get("SAM3DBODY_LOG_LEVEL", "INFO").upper()
    _level_map = {"SILENT": 0, "ERROR": 1, "WARN": 2, "INFO": 3, "STATUS": 4, "DEBUG": 5}
    log = Log(level=_level_map.get(_log_level, LogLevel.INFO))
    
    # =========================================================================
    # EXTRACT PARAMETERS FROM export_data
    # =========================================================================
    fps = export_data.get("fps", 24.0)
    frames = export_data.get("frames", [])
    faces = export_data.get("faces")
    joint_parents = export_data.get("joint_parents")
    sensor_width = export_data.get("sensor_width", 36.0)
    world_translation_mode = export_data.get("world_translation_mode", "none")
    skeleton_mode = export_data.get("skeleton_mode", "rotations")
    flip_x = export_data.get("flip_x", False)
    align_mesh_to_skeleton = export_data.get("align_mesh_to_skeleton", True)
    frame_offset = export_data.get("frame_offset", 0)
    include_skeleton = export_data.get("include_skeleton", True)
    animate_camera = export_data.get("animate_camera", False)
    camera_follow_root = export_data.get("camera_follow_root", False)
    camera_use_rotation = export_data.get("camera_use_rotation", False)
    camera_static = export_data.get("camera_static", False)
    camera_compensation = export_data.get("camera_compensation", False)
    use_depth_positioning = export_data.get("use_depth_positioning", True)
    depth_mode = export_data.get("depth_mode", "position")
    scale_factor = export_data.get("scale_factor", 1.0)
    camera_extrinsics = export_data.get("camera_extrinsics") or export_data.get("solved_camera_rotations")
    camera_intrinsics = export_data.get("camera_intrinsics")
    extrinsics_smoothing_method = export_data.get("extrinsics_smoothing_method", "kalman")
    extrinsics_smoothing_strength = export_data.get("extrinsics_smoothing_strength", 0.5)
    metadata = export_data.get("metadata", {})
    body_world_trajectory = export_data.get("body_world_trajectory", [])
    body_world_trajectory_compensated = export_data.get("body_world_trajectory_compensated", [])
    
    # Detect output format
    output_format = "abc" if output_path.lower().endswith(".abc") else "fbx"
    
    log.info(f"Starting {output_format.upper()} export: {len(frames)} frames at {fps} fps")
    log.info(f"Output: {output_path}")
    log.info(f"Up axis: {up_axis}, Include mesh: {include_mesh}, Include camera: {include_camera}")
    
    if not frames:
        return {"status": "error", "message": "No frames to export", "path": "", "frame_count": 0, "fps": fps, "file_size_mb": 0}
    
    # Global settings
    FLIP_X = flip_x
    ALIGN_MESH_TO_SKELETON = align_mesh_to_skeleton
    DISABLE_VERTICAL_OFFSET = False
    FLIP_VERTICAL = False
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def smooth_array(values, window):
        """Moving average smoothing."""
        if window <= 1 or len(values) < window:
            return values
        result = []
        half = window // 2
        for i in range(len(values)):
            start = max(0, i - half)
            end = min(len(values), i + half + 1)
            avg = sum(values[start:end]) / (end - start)
            result.append(avg)
        return result

    def smooth_camera_data(camera_rotations, window=9):
        """Pre-smooth camera rotation/translation data."""
        if not camera_rotations or window <= 1:
            return camera_rotations
        
        n = len(camera_rotations)
        if n < window:
            window = max(3, n)
        
        pans = [r.get("pan", 0.0) for r in camera_rotations]
        tilts = [r.get("tilt", 0.0) for r in camera_rotations]
        rolls = [r.get("roll", 0.0) for r in camera_rotations]
        txs = [r.get("tx", 0.0) for r in camera_rotations]
        tys = [r.get("ty", 0.0) for r in camera_rotations]
        tzs = [r.get("tz", 0.0) for r in camera_rotations]
        
        half = window // 2
        weights = []
        for i in range(-half, half + 1):
            sigma = half / 2.0
            w = math.exp(-(i * i) / (2 * sigma * sigma))
            weights.append(w)
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        def gaussian_smooth(values):
            result = []
            for i in range(len(values)):
                total = 0.0
                total_weight = 0.0
                for j, w in enumerate(weights):
                    idx = i + j - half
                    if 0 <= idx < len(values):
                        total += values[idx] * w
                        total_weight += w
                result.append(total / total_weight if total_weight > 0 else values[i])
            return result
        
        smoothed_pans = gaussian_smooth(pans)
        smoothed_tilts = gaussian_smooth(tilts)
        smoothed_rolls = gaussian_smooth(rolls)
        smoothed_txs = gaussian_smooth(txs)
        smoothed_tys = gaussian_smooth(tys)
        smoothed_tzs = gaussian_smooth(tzs)
        
        result = []
        for i in range(n):
            result.append({
                "frame": camera_rotations[i].get("frame", i),
                "pan": smoothed_pans[i],
                "tilt": smoothed_tilts[i],
                "roll": smoothed_rolls[i],
                "tx": smoothed_txs[i],
                "ty": smoothed_tys[i],
                "tz": smoothed_tzs[i],
            })
        return result

    def clear_scene():
        """Remove all objects from scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        for block in bpy.data.meshes:
            bpy.data.meshes.remove(block)
        for block in bpy.data.armatures:
            bpy.data.armatures.remove(block)
        for block in bpy.data.cameras:
            bpy.data.cameras.remove(block)
        for block in bpy.data.actions:
            bpy.data.actions.remove(block)
        for block in bpy.data.materials:
            bpy.data.materials.remove(block)

    def get_transform_for_axis(up_axis, flip_x=False):
        """Get vertex transformation function for up axis."""
        x_mult = -1 if flip_x else 1
        
        if up_axis == "Y":
            return lambda v: (v[0] * x_mult, -v[2], v[1])
        elif up_axis == "Z":
            return lambda v: (v[0] * x_mult, v[1], v[2])
        elif up_axis == "-Y":
            return lambda v: (v[0] * x_mult, v[2], -v[1])
        elif up_axis == "-Z":
            return lambda v: (v[0] * x_mult, -v[1], -v[2])
        else:
            return lambda v: (v[0] * x_mult, -v[2], v[1])

    def get_rotation_transform_matrix(up_axis, flip_x=False):
        """Get 3x3 transformation matrix for rotations."""
        x_mult = -1 if flip_x else 1
        
        if up_axis == "Y":
            return Matrix(((x_mult, 0, 0), (0, -1, 0), (0, 0, -1)))
        elif up_axis == "Z":
            return Matrix(((x_mult, 0, 0), (0, 0, 1), (0, -1, 0)))
        elif up_axis == "-Y":
            return Matrix(((x_mult, 0, 0), (0, 1, 0), (0, 0, 1)))
        elif up_axis == "-Z":
            return Matrix(((x_mult, 0, 0), (0, 0, -1), (0, 1, 0)))
        else:
            return Matrix(((x_mult, 0, 0), (0, -1, 0), (0, 0, -1)))

    def transform_rotation_matrix(rot_3x3, up_axis):
        """Transform a 3x3 rotation matrix from MHR space to Blender space."""
        m = Matrix((
            (rot_3x3[0][0], rot_3x3[0][1], rot_3x3[0][2]),
            (rot_3x3[1][0], rot_3x3[1][1], rot_3x3[1][2]),
            (rot_3x3[2][0], rot_3x3[2][1], rot_3x3[2][2])
        ))
        
        T = get_rotation_transform_matrix(up_axis, FLIP_X)
        T_inv = T.inverted()
        transformed = T @ m @ T_inv
        
        return transformed

    def get_world_offset_from_cam_t(pred_cam_t, up_axis):
        """Get world offset for root_locator."""
        return Vector((0, 0, 0))

    def get_body_offset_from_cam_t(pred_cam_t, up_axis):
        """Get offset to apply to body mesh/skeleton for correct camera alignment."""
        if not pred_cam_t or len(pred_cam_t) < 3:
            return Vector((0, 0, 0))
        
        tx, ty, tz = pred_cam_t[0], pred_cam_t[1], pred_cam_t[2]
        ty_world = -ty
        
        if up_axis == "Y":
            return Vector((tx, ty_world, 0))
        elif up_axis == "Z":
            return Vector((tx, 0, ty_world))
        elif up_axis == "-Y":
            return Vector((tx, -ty_world, 0))
        elif up_axis == "-Z":
            return Vector((tx, 0, -ty_world))
        else:
            return Vector((tx, ty_world, 0))

    def create_animated_mesh(all_frames, faces, fps, transform_func, world_translation_mode="none", up_axis="Y", frame_offset=0):
        """Create mesh with per-vertex animation using shape keys."""
        first_verts = all_frames[0].get("vertices")
        if not first_verts:
            return None
        
        mesh_offset = [0, 0, 0]
        if ALIGN_MESH_TO_SKELETON:
            first_joints = all_frames[0].get("joint_coords")
            if first_joints is not None and len(first_joints) > 0:
                pelvis = np.array(first_joints[0])
                first_verts_np = np.array(first_verts)
                mesh_center = np.mean(first_verts_np, axis=0)
                offset_from_pelvis = mesh_center - pelvis
                if np.linalg.norm(offset_from_pelvis) > 0.1:
                    mesh_offset = offset_from_pelvis.tolist()
                    log.info(f"Mesh-to-skeleton alignment offset: [{mesh_offset[0]:.3f}, {mesh_offset[1]:.3f}, {mesh_offset[2]:.3f}]")
        
        first_world_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            first_cam_t = all_frames[0].get("pred_cam_t")
            first_world_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
        
        mesh = bpy.data.meshes.new("body_mesh")
        verts = []
        for v in first_verts:
            v_aligned = [v[0] - mesh_offset[0], v[1] - mesh_offset[1], v[2] - mesh_offset[2]]
            pos = Vector(transform_func(v_aligned))
            if world_translation_mode == "baked":
                pos += first_world_offset
            verts.append(pos)
        
        if faces:
            mesh.from_pydata(verts, [], faces)
        else:
            mesh.from_pydata(verts, [], [])
        mesh.update()
        
        obj = bpy.data.objects.new("body", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        basis = obj.shape_key_add(name="Basis", from_mix=False)
        
        log.info(f"Creating {len(all_frames)} shape keys...")
        
        for frame_idx, frame_data in enumerate(all_frames):
            frame_verts = frame_data.get("vertices")
            if not frame_verts:
                continue
            
            frame_world_offset = Vector((0, 0, 0))
            if world_translation_mode == "baked":
                frame_cam_t = frame_data.get("pred_cam_t")
                frame_world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
            
            sk = obj.shape_key_add(name=f"frame_{frame_idx:04d}", from_mix=False)
            
            for j, v in enumerate(frame_verts):
                if j < len(sk.data):
                    v_aligned = [v[0] - mesh_offset[0], v[1] - mesh_offset[1], v[2] - mesh_offset[2]]
                    pos = Vector(transform_func(v_aligned))
                    if world_translation_mode == "baked":
                        pos += frame_world_offset
                    sk.data[j].co = pos
            
            actual_frame = frame_idx + frame_offset
            is_last = (frame_idx == len(all_frames) - 1)
            is_first = (frame_idx == 0)
            
            if not is_first:
                sk.value = 0.0
                sk.keyframe_insert(data_path="value", frame=actual_frame - 1)
            
            sk.value = 1.0
            sk.keyframe_insert(data_path="value", frame=actual_frame)
            
            if not is_last:
                sk.value = 0.0
                sk.keyframe_insert(data_path="value", frame=actual_frame + 1)
            
            log.progress(frame_idx, len(all_frames), "Shape keys", interval=50)
        
        log.info(f"Created mesh with {len(all_frames)} shape keys")
        return obj

    def create_skeleton_with_rotations(all_frames, fps, transform_func, world_translation_mode="none", 
                                        up_axis="Y", root_locator=None, joint_parents=None, 
                                        frame_offset=0, solved_camera_rotations=None):
        """Create animated skeleton using armature with ROTATION keyframes."""
        first_joints = all_frames[0].get("joint_coords")
        first_rotations = all_frames[0].get("joint_rotations")
        
        if not first_joints:
            log.info("No joint_coords in first frame, skipping skeleton")
            return None
        
        if not first_rotations:
            log.info("No joint_rotations available, falling back to position-based skeleton")
            return create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode, 
                                                   up_axis, root_locator, joint_parents, frame_offset)
        
        num_joints = len(first_joints)
        log.info(f"Creating rotation-based armature with {num_joints} bones...")
        
        arm_data = bpy.data.armatures.new("Skeleton")
        armature = bpy.data.objects.new("Skeleton", arm_data)
        bpy.context.collection.objects.link(armature)
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        
        if world_translation_mode == "root" and root_locator:
            armature.parent = root_locator
        
        first_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            first_cam_t = all_frames[0].get("pred_cam_t")
            first_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
        
        bpy.ops.object.mode_set(mode='EDIT')
        
        edit_bones = []
        for i in range(num_joints):
            bone = arm_data.edit_bones.new(f"joint_{i:03d}")
            pos = first_joints[i]
            head_pos = Vector(transform_func(pos))
            
            if world_translation_mode == "baked":
                head_pos += first_offset
            
            bone.head = head_pos
            bone.tail = head_pos + Vector((0, 0.03, 0))
            edit_bones.append(bone)
        
        if joint_parents is not None:
            log.info(f"Setting up bone hierarchy from joint_parents...")
            roots = []
            for i in range(num_joints):
                parent_idx = joint_parents[i]
                if parent_idx >= 0 and parent_idx < num_joints:
                    edit_bones[i].parent = edit_bones[parent_idx]
                else:
                    roots.append(i)
            log.info(f"Found {len(roots)} root bone(s): {roots}")
            
            for i in range(num_joints):
                children = [j for j in range(num_joints) if joint_parents[j] == i]
                if children:
                    child_positions = [edit_bones[c].head for c in children]
                    avg_child_pos = sum(child_positions, Vector((0, 0, 0))) / len(children)
                    direction = avg_child_pos - edit_bones[i].head
                    if direction.length > 0.001:
                        edit_bones[i].tail = edit_bones[i].head + direction.normalized() * 0.05
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        
        for pose_bone in armature.pose.bones:
            pose_bone.rotation_mode = 'QUATERNION'
        
        log.info(f"Animating {num_joints} bones with rotations over {len(all_frames)} frames...")
        
        parent_indices = joint_parents if joint_parents is not None else [-1] * num_joints
        has_camera_rots = solved_camera_rotations is not None and len(solved_camera_rotations) > 0
        
        smoothed_camera_data = None
        if has_camera_rots:
            log.info(f"Pre-smoothing camera rotations for smooth root compensation...")
            smoothed_camera_data = smooth_camera_data(solved_camera_rotations, window=9)
        
        for frame_idx, frame_data in enumerate(all_frames):
            joints = frame_data.get("joint_coords")
            rotations = frame_data.get("joint_rotations")
            
            if not joints or not rotations:
                continue
            
            actual_frame = frame_idx + frame_offset
            bpy.context.scene.frame_set(actual_frame)
            
            world_offset = Vector((0, 0, 0))
            if world_translation_mode == "baked":
                frame_cam_t = frame_data.get("pred_cam_t")
                world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
            
            camera_compensation_matrix = Matrix.Identity(3)
            
            if has_camera_rots and smoothed_camera_data and frame_idx < len(smoothed_camera_data):
                cam_rot = smoothed_camera_data[frame_idx]
                pan = cam_rot.get("pan", 0.0)
                tilt = cam_rot.get("tilt", 0.0)
                roll = cam_rot.get("roll", 0.0)
                
                if up_axis == "Y":
                    euler = Euler((tilt, pan, roll), 'YXZ')
                elif up_axis == "Z":
                    euler = Euler((tilt, roll, pan), 'ZXY')
                else:
                    euler = Euler((tilt, pan, roll), 'YXZ')
                
                camera_rot_matrix = euler.to_matrix().to_3x3()
                camera_compensation_matrix = camera_rot_matrix.inverted()
            
            global_rots_blender = []
            for i in range(num_joints):
                if i < len(rotations) and rotations[i] is not None:
                    rot = rotations[i]
                    if isinstance(rot, (list, np.ndarray)) and len(rot) >= 3:
                        transformed = transform_rotation_matrix(rot, up_axis)
                        
                        if i == 0 and has_camera_rots:
                            transformed = camera_compensation_matrix @ transformed
                        
                        global_rots_blender.append(transformed)
                    else:
                        global_rots_blender.append(Matrix.Identity(3))
                else:
                    global_rots_blender.append(Matrix.Identity(3))
            
            for i in range(num_joints):
                pose_bone = armature.pose.bones[f"joint_{i:03d}"]
                global_rot = global_rots_blender[i]
                
                parent_idx = parent_indices[i]
                if parent_idx >= 0 and parent_idx < num_joints:
                    parent_global = global_rots_blender[parent_idx]
                    local_rot = parent_global.inverted() @ global_rot
                else:
                    local_rot = global_rot
                
                quat = local_rot.to_quaternion()
                pose_bone.rotation_quaternion = quat
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=actual_frame)
                
                if i == 0:
                    base_pos = Vector(transform_func(joints[i]))
                    if world_translation_mode == "baked":
                        base_pos += world_offset
                    pose_bone.location = base_pos - pose_bone.bone.head_local
                    pose_bone.keyframe_insert(data_path="location", frame=actual_frame)
            
            log.progress(frame_idx, len(all_frames), "Skeleton rotations", interval=50)
        
        bpy.ops.object.mode_set(mode='OBJECT')
        log.info(f"Created skeleton with rotation animation")
        return armature

    def create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode="none",
                                        up_axis="Y", root_locator=None, joint_parents=None, frame_offset=0):
        """Create animated skeleton using armature with POSITION keyframes (legacy mode)."""
        first_joints = all_frames[0].get("joint_coords")
        
        if not first_joints:
            log.info("No joint_coords in first frame")
            return None
        
        num_joints = len(first_joints)
        log.info(f"Creating position-based armature with {num_joints} bones...")
        
        arm_data = bpy.data.armatures.new("Skeleton")
        armature = bpy.data.objects.new("Skeleton", arm_data)
        bpy.context.collection.objects.link(armature)
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        
        if world_translation_mode == "root" and root_locator:
            armature.parent = root_locator
        
        first_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            first_cam_t = all_frames[0].get("pred_cam_t")
            first_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
        
        bpy.ops.object.mode_set(mode='EDIT')
        
        edit_bones = []
        for i in range(num_joints):
            bone = arm_data.edit_bones.new(f"joint_{i:03d}")
            pos = first_joints[i]
            head_pos = Vector(transform_func(pos))
            
            if world_translation_mode == "baked":
                head_pos += first_offset
            
            bone.head = head_pos
            bone.tail = head_pos + Vector((0, 0.03, 0))
            edit_bones.append(bone)
        
        if joint_parents is not None:
            for i in range(num_joints):
                parent_idx = joint_parents[i]
                if parent_idx >= 0 and parent_idx < num_joints:
                    edit_bones[i].parent = edit_bones[parent_idx]
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        
        log.info(f"Animating {num_joints} bones with positions over {len(all_frames)} frames...")
        
        for frame_idx, frame_data in enumerate(all_frames):
            joints = frame_data.get("joint_coords")
            if not joints:
                continue
            
            actual_frame = frame_idx + frame_offset
            bpy.context.scene.frame_set(actual_frame)
            
            world_offset = Vector((0, 0, 0))
            if world_translation_mode == "baked":
                frame_cam_t = frame_data.get("pred_cam_t")
                world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
            
            for i, pos in enumerate(joints):
                if i < len(armature.pose.bones):
                    pose_bone = armature.pose.bones[f"joint_{i:03d}"]
                    target = Vector(transform_func(pos))
                    if world_translation_mode == "baked":
                        target += world_offset
                    pose_bone.location = target - pose_bone.bone.head_local
                    pose_bone.keyframe_insert(data_path="location", frame=actual_frame)
            
            log.progress(frame_idx, len(all_frames), "Skeleton positions", interval=50)
        
        bpy.ops.object.mode_set(mode='OBJECT')
        log.info(f"Created skeleton with position animation")
        return armature

    def create_skeleton(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y",
                        root_locator=None, skeleton_mode="rotations", joint_parents=None, 
                        frame_offset=0, solved_camera_rotations=None):
        """Create skeleton using specified mode."""
        if skeleton_mode == "rotations":
            return create_skeleton_with_rotations(all_frames, fps, transform_func, world_translation_mode,
                                                   up_axis, root_locator, joint_parents, frame_offset,
                                                   solved_camera_rotations)
        else:
            return create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode,
                                                   up_axis, root_locator, joint_parents, frame_offset)

    def create_root_locator(all_frames, fps, up_axis, flip_x=False, frame_offset=0):
        """Create root locator with animated position."""
        root = bpy.data.objects.new("root_locator", None)
        root.empty_display_type = 'ARROWS'
        root.empty_display_size = 0.1
        bpy.context.collection.objects.link(root)
        
        for frame_idx, frame_data in enumerate(all_frames):
            frame_cam_t = frame_data.get("pred_cam_t")
            world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
            
            if flip_x:
                world_offset = Vector((-world_offset.x, world_offset.y, world_offset.z))
            
            root.location = world_offset
            root.keyframe_insert(data_path="location", frame=frame_idx + frame_offset)
        
        log.info(f"Root locator animated over {len(all_frames)} frames")
        return root

    def create_root_locator_with_camera_compensation(all_frames, camera_extrinsics, fps, up_axis, 
                                                      flip_x=False, frame_offset=0,
                                                      smoothing_method="kalman", smoothing_strength=0.5):
        """Create root locator with inverse camera extrinsics baked in."""
        log.info("Creating root locator with camera compensation...")
        
        root = bpy.data.objects.new("root_locator", None)
        root.empty_display_type = 'ARROWS'
        root.empty_display_size = 0.1
        bpy.context.collection.objects.link(root)
        
        smoothed_extrinsics = camera_extrinsics
        if smoothing_method != "none" and len(camera_extrinsics) > 3:
            window = int(3 + smoothing_strength * 12)
            smoothed_extrinsics = smooth_camera_data(camera_extrinsics, window)
        
        extrinsics_by_frame = {}
        for ext in smoothed_extrinsics:
            extrinsics_by_frame[ext.get("frame", 0)] = ext
        
        initial_ext = extrinsics_by_frame.get(0, smoothed_extrinsics[0] if smoothed_extrinsics else {})
        initial_pan = initial_ext.get("pan", 0)
        initial_tilt = initial_ext.get("tilt", 0)
        initial_roll = initial_ext.get("roll", 0)
        initial_tx = initial_ext.get("tx", 0)
        initial_ty = initial_ext.get("ty", 0)
        initial_tz = initial_ext.get("tz", 0)
        
        first_frame_cam_t = all_frames[0].get("pred_cam_t") if all_frames else None
        if first_frame_cam_t and len(first_frame_cam_t) >= 3:
            initial_distance = first_frame_cam_t[2]
        else:
            initial_distance = 5.0
        
        for frame_idx, frame_data in enumerate(all_frames):
            frame_cam_t = frame_data.get("pred_cam_t")
            if frame_cam_t and len(frame_cam_t) >= 3:
                distance = frame_cam_t[2]
            else:
                distance = initial_distance
            
            ext = extrinsics_by_frame.get(frame_idx, {})
            
            delta_pan = ext.get("pan", 0) - initial_pan
            delta_tilt = ext.get("tilt", 0) - initial_tilt
            delta_roll = ext.get("roll", 0) - initial_roll
            delta_tx = ext.get("tx", 0) - initial_tx
            delta_ty = ext.get("ty", 0) - initial_ty
            delta_tz = ext.get("tz", 0) - initial_tz
            
            pan_translation = distance * math.tan(delta_pan)
            tilt_translation = distance * math.tan(delta_tilt)
            
            inv_tx = -delta_tx
            inv_ty = -delta_ty
            inv_tz = -delta_tz
            inv_roll = -delta_roll
            
            if up_axis.upper() in ["Y", "-Y"]:
                nodal_trans = Vector((pan_translation, tilt_translation, 0))
                camera_trans = Vector((inv_tx, inv_ty, inv_tz))
                rot_euler = Euler((0, 0, inv_roll), 'XYZ')
            else:
                nodal_trans = Vector((pan_translation, 0, tilt_translation))
                camera_trans = Vector((inv_tx, inv_tz, inv_ty))
                rot_euler = Euler((0, inv_roll, 0), 'XYZ')
            
            final_location = nodal_trans + camera_trans
            
            if flip_x:
                final_location = Vector((-final_location.x, final_location.y, final_location.z))
                rot_euler = Euler((-rot_euler.x, rot_euler.y, -rot_euler.z), 'XYZ')
            
            root.location = final_location
            root.rotation_euler = rot_euler
            root.keyframe_insert(data_path="location", frame=frame_idx + frame_offset)
            root.keyframe_insert(data_path="rotation_euler", frame=frame_idx + frame_offset)
        
        log.info(f"Root locator created with camera compensation ({len(all_frames)} frames)")
        return root

    def create_camera(all_frames, fps, transform_func, up_axis, sensor_width=36.0, 
                      world_translation_mode="none", animate_camera=False, frame_offset=0,
                      camera_follow_root=False, camera_use_rotation=False, camera_static=False,
                      flip_x=False, solved_camera_rotations=None):
        """Create camera with proper intrinsics."""
        first_frame = all_frames[0]
        
        cam_data = bpy.data.cameras.new("Camera")
        cam_data.type = 'PERSP'
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.sensor_width = sensor_width
        
        focal_px = first_frame.get("focal_length", 1500)
        if isinstance(focal_px, (list, tuple, np.ndarray)):
            focal_px = float(focal_px[0]) if hasattr(focal_px, '__len__') and len(focal_px) > 0 else 1500
        
        image_size = first_frame.get("image_size", [1920, 1080])
        image_width = image_size[0] if image_size else 1920
        image_height = image_size[1] if image_size else 1080
        
        focal_mm = sensor_width * focal_px / image_width
        cam_data.lens = focal_mm
        
        camera = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(camera)
        
        pred_cam_t = first_frame.get("pred_cam_t", [0, 0, 5])
        cam_distance = abs(pred_cam_t[2]) if len(pred_cam_t) > 2 else 5.0
        
        if up_axis == "Y":
            base_dir = Vector((0, 0, 1))
        elif up_axis == "Z":
            base_dir = Vector((0, 1, 0))
        elif up_axis == "-Y":
            base_dir = Vector((0, 0, -1))
        elif up_axis == "-Z":
            base_dir = Vector((0, -1, 0))
        else:
            base_dir = Vector((0, 0, 1))
        
        camera.location = base_dir * cam_distance
        
        target = bpy.data.objects.new("cam_target", None)
        target.location = Vector((0, 0, 0))
        bpy.context.collection.objects.link(target)
        
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        
        if up_axis in ["Y", "-Y"]:
            constraint.up_axis = 'UP_Y'
        else:
            constraint.up_axis = 'UP_Z'
        
        bpy.context.view_layer.update()
        
        camera["sensor_width_mm"] = sensor_width
        camera["focal_length_mm"] = focal_mm
        camera["focal_length_px"] = focal_px
        camera["image_width"] = image_width
        camera["image_height"] = image_height
        
        log.info(f"Created camera: focal={focal_mm:.1f}mm ({focal_px:.0f}px), distance={cam_distance:.2f}m")
        return camera

    def apply_per_frame_body_offset(mesh_obj, armature_obj, frames, up_axis, frame_offset=1,
                                     use_depth=True, depth_mode="position", scale_factor=1.0):
        """Apply per-frame body offset based on pred_cam_t."""
        if not frames:
            return
        
        log.info(f"Applying per-frame body offset ({len(frames)} frames)...")
        log.info(f"  Depth: use={use_depth}, mode='{depth_mode}', scale={scale_factor:.3f}")
        
        has_tracked_depth = "tracked_depth" in frames[0] if frames else False
        
        first_frame = frames[0]
        if has_tracked_depth:
            ref_depth = abs(first_frame.get("tracked_depth", 5.0))
        else:
            first_cam_t = first_frame.get("pred_cam_t", [0, 0, 5])
            ref_depth = abs(first_cam_t[2]) if len(first_cam_t) > 2 and abs(first_cam_t[2]) > 0.1 else 5.0
        
        for i, frame_data in enumerate(frames):
            frame_num = frame_offset + i
            pred_cam_t = frame_data.get("pred_cam_t", [0, 0, 5])
            
            tx = pred_cam_t[0] if len(pred_cam_t) > 0 else 0
            ty = pred_cam_t[1] if len(pred_cam_t) > 1 else 0
            
            if has_tracked_depth and "tracked_depth" in frame_data:
                frame_depth = abs(frame_data["tracked_depth"])
            else:
                tz = pred_cam_t[2] if len(pred_cam_t) > 2 else 5
                frame_depth = abs(tz) if abs(tz) > 0.1 else ref_depth
            
            world_x = tx * frame_depth * scale_factor
            world_y = ty * frame_depth * scale_factor
            
            if use_depth and depth_mode in ["position", "both"]:
                depth_delta = frame_depth - ref_depth
                world_z = -depth_delta
            else:
                world_z = 0
            
            if use_depth and depth_mode in ["scale", "both"]:
                mesh_scale = frame_depth / ref_depth if ref_depth > 0 else 1.0
            else:
                mesh_scale = 1.0
            
            world_y_world = -world_y
            
            if up_axis == "Y":
                offset = Vector((world_x, world_y_world, world_z))
            elif up_axis == "Z":
                offset = Vector((world_x, world_z, world_y_world))
            elif up_axis == "-Y":
                offset = Vector((world_x, -world_y_world, -world_z))
            elif up_axis == "-Z":
                offset = Vector((world_x, -world_z, -world_y_world))
            else:
                offset = Vector((world_x, world_y_world, world_z))
            
            if mesh_obj:
                mesh_obj.location = offset
                mesh_obj.keyframe_insert(data_path="location", frame=frame_num)
                if use_depth and depth_mode in ["scale", "both"]:
                    mesh_obj.scale = Vector((mesh_scale, mesh_scale, mesh_scale))
                    mesh_obj.keyframe_insert(data_path="scale", frame=frame_num)
            
            if armature_obj:
                armature_obj.location = offset
                armature_obj.keyframe_insert(data_path="location", frame=frame_num)
                if use_depth and depth_mode in ["scale", "both"]:
                    armature_obj.scale = Vector((mesh_scale, mesh_scale, mesh_scale))
                    armature_obj.keyframe_insert(data_path="scale", frame=frame_num)
        
        log.info(f"Applied per-frame body offset")

    def create_metadata_locator(metadata):
        """Create locator with metadata as custom properties."""
        if not metadata:
            return None
        
        loc = bpy.data.objects.new("SAM3DBody_Metadata", None)
        loc.empty_display_type = 'PLAIN_AXES'
        loc.empty_display_size = 0.01
        bpy.context.collection.objects.link(loc)
        
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                loc[key] = str(value)
            else:
                loc[key] = value
        
        log.info(f"Created metadata locator with {len(metadata)} properties")
        return loc

    def create_world_position_locator(trajectory_compensated, fps=24.0, frame_offset=1):
        """Create locator showing compensated world position."""
        if not trajectory_compensated:
            return None
        
        loc = bpy.data.objects.new("SAM3DBody_WorldPosition", None)
        loc.empty_display_type = 'SPHERE'
        loc.empty_display_size = 0.1
        bpy.context.collection.objects.link(loc)
        
        for i, pos in enumerate(trajectory_compensated):
            frame_num = frame_offset + i
            loc.location = Vector(pos)
            loc.keyframe_insert(data_path="location", frame=frame_num)
        
        log.info(f"Created WorldPosition locator with {len(trajectory_compensated)} keyframes")
        return loc

    def create_screen_position_locator(frames, fps=24.0, frame_offset=1, up_axis="Y"):
        """Create locator showing screen position from pred_cam_t."""
        if not frames:
            return None
        
        loc = bpy.data.objects.new("SAM3DBody_ScreenPosition", None)
        loc.empty_display_type = 'CIRCLE'
        loc.empty_display_size = 0.15
        bpy.context.collection.objects.link(loc)
        
        for i, frame_data in enumerate(frames):
            frame_num = frame_offset + i
            pred_cam_t = frame_data.get("pred_cam_t", [0, 0, 5])
            
            tx = pred_cam_t[0] if len(pred_cam_t) > 0 else 0
            ty = pred_cam_t[1] if len(pred_cam_t) > 1 else 0
            
            if up_axis == "Y":
                loc.location = Vector((tx, -ty, 0))
            else:
                loc.location = Vector((tx, 0, -ty))
            
            loc.keyframe_insert(data_path="location", frame=frame_num)
        
        log.info(f"Created ScreenPosition locator with {len(frames)} keyframes")
        return loc

    def create_trajectory_locator(trajectory, fps=24.0, frame_offset=1):
        """Create locator for body trajectory (legacy)."""
        if not trajectory:
            return None
        
        loc = bpy.data.objects.new("SAM3DBody_Trajectory", None)
        loc.empty_display_type = 'ARROWS'
        loc.empty_display_size = 0.1
        bpy.context.collection.objects.link(loc)
        
        for i, pos in enumerate(trajectory):
            frame_num = frame_offset + i
            loc.location = Vector(pos)
            loc.keyframe_insert(data_path="location", frame=frame_num)
        
        return loc

    def export_fbx_file(output_path, axis_forward, axis_up):
        """Export to FBX."""
        log.info(f"Exporting FBX: {output_path}")
        
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=False,
            global_scale=0.01,
            apply_unit_scale=True,
            apply_scale_options='FBX_SCALE_ALL',
            bake_space_transform=False,
            axis_forward=axis_forward,
            axis_up=axis_up,
            object_types={'MESH', 'ARMATURE', 'EMPTY', 'CAMERA'},
            use_mesh_modifiers=True,
            mesh_smooth_type='FACE',
            use_armature_deform_only=False,
            add_leaf_bones=False,
            use_custom_props=True,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            bake_anim_force_startend_keying=True,
            bake_anim_step=1.0,
            bake_anim_simplify_factor=0.0,
        )
        log.info(f"FBX export complete")

    def export_alembic_file(output_path):
        """Export to Alembic."""
        log.info(f"Exporting Alembic: {output_path}")
        
        bpy.ops.wm.alembic_export(
            filepath=output_path,
            start=bpy.context.scene.frame_start,
            end=bpy.context.scene.frame_end,
            selected=False,
            visible_objects_only=True,
            flatten=False,
            uvs=True,
            normals=True,
            vcolors=False,
            apply_subdiv=False,
            curves_as_mesh=False,
            use_instancing=True,
            global_scale=1.0,
            triangulate=False,
            export_hair=False,
            export_particles=False,
            packuv=True,
        )
        log.info(f"Alembic export complete")

    # =========================================================================
    # MAIN EXPORT LOGIC
    # =========================================================================
    
    # Clear scene
    clear_scene()
    
    # Set up scene
    bpy.context.scene.frame_start = frame_offset
    bpy.context.scene.frame_end = frame_offset + len(frames) - 1
    bpy.context.scene.render.fps = int(fps)
    
    # Check for rotation data
    has_rotations = frames[0].get("joint_rotations") is not None
    log.info(f"Rotation data available: {has_rotations}")
    
    if skeleton_mode == "rotations" and not has_rotations:
        log.warn("Rotation mode requested but no data available. Falling back to positions.")
        skeleton_mode = "positions"
    
    # Handle camera compensation mode
    root_camera_compensation_mode = False
    if world_translation_mode == "root_camera_compensation":
        if camera_extrinsics:
            log.info("MODE: Root Locator + Camera Compensation")
            root_camera_compensation_mode = True
            camera_static = True
            animate_camera = False
            camera_use_rotation = False
            world_translation_mode = "root"
        else:
            log.warn("root_camera_compensation mode requires camera_extrinsics. Falling back to 'root'.")
            world_translation_mode = "root"
    
    # Get transform function
    transform_func = get_transform_for_axis(up_axis, FLIP_X)
    
    # Set image resolution if available
    first_frame = frames[0]
    image_size = first_frame.get("image_size")
    if image_size:
        bpy.context.scene.render.resolution_x = image_size[0]
        bpy.context.scene.render.resolution_y = image_size[1]
        bpy.context.scene.render.resolution_percentage = 100
    
    # Create root locator
    root_locator = None
    body_offset = Vector((0, 0, 0))
    if world_translation_mode == "root":
        if root_camera_compensation_mode and camera_extrinsics:
            root_locator = create_root_locator_with_camera_compensation(
                frames, camera_extrinsics, fps, up_axis, FLIP_X, frame_offset,
                smoothing_method=extrinsics_smoothing_method,
                smoothing_strength=extrinsics_smoothing_strength
            )
        else:
            root_locator = create_root_locator(frames, fps, up_axis, FLIP_X, frame_offset)
        
        first_cam_t = frames[0].get("pred_cam_t")
        body_offset = get_body_offset_from_cam_t(first_cam_t, up_axis)
    
    # Create mesh
    mesh_obj = None
    if include_mesh:
        mesh_obj = create_animated_mesh(frames, faces, fps, transform_func, world_translation_mode, up_axis, frame_offset)
        if world_translation_mode == "root" and root_locator and mesh_obj:
            mesh_obj.parent = root_locator
            mesh_obj.location = body_offset
    
    # Create skeleton
    armature_obj = None
    if include_skeleton:
        skeleton_camera_rots = None
        if not root_camera_compensation_mode:
            skeleton_camera_rots = camera_extrinsics
        armature_obj = create_skeleton(frames, fps, transform_func, world_translation_mode, up_axis,
                                        root_locator, skeleton_mode, joint_parents, frame_offset,
                                        skeleton_camera_rots)
        
        if world_translation_mode == "root" and root_locator and armature_obj:
            armature_obj.location = body_offset
    
    # Apply per-frame body offset
    if world_translation_mode == "root" and root_locator and (mesh_obj or armature_obj):
        apply_per_frame_body_offset(mesh_obj, armature_obj, frames, up_axis, frame_offset,
                                     use_depth=use_depth_positioning, depth_mode=depth_mode,
                                     scale_factor=scale_factor)
    
    # Create camera
    camera_obj = None
    if include_camera:
        effective_sensor_width = sensor_width
        if camera_intrinsics and camera_intrinsics.get("sensor_width_mm"):
            effective_sensor_width = camera_intrinsics["sensor_width_mm"]
        
        camera_obj = create_camera(frames, fps, transform_func, up_axis, effective_sensor_width,
                                    world_translation_mode, animate_camera, frame_offset,
                                    camera_follow_root, camera_use_rotation, camera_static,
                                    FLIP_X, camera_extrinsics)
        
        if camera_follow_root and root_locator and camera_obj:
            camera_obj.parent = root_locator
    
    # Create metadata locator
    if metadata:
        create_metadata_locator(metadata)
    
    # Create locators
    create_screen_position_locator(frames, fps, frame_offset, up_axis)
    
    if body_world_trajectory_compensated:
        create_world_position_locator(body_world_trajectory_compensated, fps, frame_offset)
    elif body_world_trajectory:
        create_world_position_locator(body_world_trajectory, fps, frame_offset)
    
    if body_world_trajectory:
        create_trajectory_locator(body_world_trajectory, fps, frame_offset)
    
    # Set up export orientation
    if up_axis == "Y":
        axis_forward = '-Z'
        axis_up_export = 'Y'
    elif up_axis == "Z":
        axis_forward = '-Y'
        axis_up_export = 'Z'
    elif up_axis == "-Y":
        axis_forward = 'Z'
        axis_up_export = '-Y'
    elif up_axis == "-Z":
        axis_forward = 'Y'
        axis_up_export = '-Z'
    else:
        axis_forward = '-Z'
        axis_up_export = 'Y'
    
    # Export
    if output_format == "abc":
        export_alembic_file(output_path)
        
        fbx_path = output_path.replace(".abc", "_skeleton.fbx")
        if mesh_obj:
            mesh_obj.hide_set(True)
        export_fbx_file(fbx_path, axis_forward, axis_up_export)
        log.info(f"Also exported skeleton/camera to: {fbx_path}")
    else:
        export_fbx_file(output_path, axis_forward, axis_up_export)
    
    # Get file info
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    file_size_mb = file_size / (1024 * 1024)
    
    log.info(f"Export complete! File size: {file_size_mb:.2f} MB")
    
    return {
        "status": "success",
        "path": output_path,
        "frame_count": len(frames),
        "fps": fps,
        "file_size_mb": file_size_mb
    }


# Check if bpy is available
def is_bpy_available():
    """Check if bpy module is available."""
    try:
        import bpy
        return True
    except ImportError:
        return False


__all__ = ['export_animated_fbx', 'is_bpy_available']
