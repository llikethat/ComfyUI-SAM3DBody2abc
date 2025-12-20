"""
Blender Script: Export Animated FBX from Mesh Sequence JSON
Creates animated mesh using vertex keyframes + joint skeleton.

Supports two skeleton animation modes:
- Rotations: Uses true joint rotation matrices from MHR model (recommended for retargeting)
- Positions: Uses joint position offsets (legacy mode)

Usage: blender --background --python blender_animated_fbx.py -- input.json output.fbx [up_axis]

Args:
    input.json: JSON with frames data
    output.fbx: Output FBX path
    up_axis: Y, Z, -Y, or -Z (default: Y)
"""

import bpy
import json
import sys
import os
from mathutils import Vector, Matrix, Euler, Quaternion
import math


def smooth_array(values, window):
    """Moving average smoothing for camera animation to reduce jitter.
    
    Args:
        values: List of float values
        window: Smoothing window size (odd number works best)
    
    Returns:
        Smoothed list of values
    """
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


def clear_scene():
    """Remove all objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clean data blocks
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for cam in bpy.data.cameras:
        bpy.data.cameras.remove(cam)


def get_transform_for_axis(up_axis, flip_x=False):
    """
    Get coordinate transformation based on desired up axis.
    SAM3DBody uses: X-right, Y-up, Z-forward (OpenGL convention)
    
    Args:
        up_axis: Which axis should point up ("Y", "Z", "-Y", "-Z")
        flip_x: If True, mirror the result on X axis (useful if animation appears flipped)
    
    Returns: (flip_func, axis_forward, axis_up)
    """
    # Base X multiplier - flip_x inverts this
    x_mult = 1 if flip_x else -1
    
    if up_axis == "Y":
        return lambda p: (x_mult * p[0], -p[1], -p[2]), '-Z', 'Y'
    elif up_axis == "Z":
        return lambda p: (x_mult * p[0], -p[2], p[1]), 'Y', 'Z'
    elif up_axis == "-Y":
        return lambda p: (x_mult * p[0], p[1], p[2]), 'Z', '-Y'
    elif up_axis == "-Z":
        return lambda p: (x_mult * p[0], p[2], -p[1]), '-Y', '-Z'
    else:
        return lambda p: (x_mult * p[0], -p[1], -p[2]), '-Z', 'Y'


def get_rotation_transform_matrix(up_axis, flip_x=False):
    """
    Get rotation transformation matrix for converting MHR rotations to Blender.
    """
    # Base X multiplier for rotation matrix
    x_mult = 1 if flip_x else -1
    
    if up_axis == "Y":
        # Flip X, Y, Z -> mirror across origin
        return Matrix((
            (x_mult, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ))
    elif up_axis == "Z":
        # X stays, Y<->Z swap with sign changes
        return Matrix((
            (x_mult, 0, 0),
            (0, 0, 1),
            (0, -1, 0)
        ))
    elif up_axis == "-Y":
        return Matrix((
            (x_mult, 0, 0),
            (0, 1, 0),
            (0, 0, 1)
        ))
    elif up_axis == "-Z":
        return Matrix((
            (x_mult, 0, 0),
            (0, 0, -1),
            (0, 1, 0)
        ))
    else:
        return Matrix((
            (x_mult, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ))


def transform_rotation_matrix(rot_3x3, up_axis):
    """
    Transform a 3x3 rotation matrix from MHR space to Blender space.
    
    rot_3x3: List of lists [[r00,r01,r02], [r10,r11,r12], [r20,r21,r22]]
    up_axis: Target up axis
    
    Returns: Blender Matrix (3x3)
    """
    # Convert to Blender Matrix
    m = Matrix((
        (rot_3x3[0][0], rot_3x3[0][1], rot_3x3[0][2]),
        (rot_3x3[1][0], rot_3x3[1][1], rot_3x3[1][2]),
        (rot_3x3[2][0], rot_3x3[2][1], rot_3x3[2][2])
    ))
    
    # Get transformation matrix - flip_x is handled in the global FLIP_X variable set by main()
    T = get_rotation_transform_matrix(up_axis, FLIP_X)
    
    # Transform: T * M * T^-1 (similarity transform)
    # This transforms the rotation from MHR coordinate system to Blender's
    T_inv = T.inverted()
    transformed = T @ m @ T_inv
    
    return transformed


# Global flip_x setting (set by main)
FLIP_X = False


def get_world_offset_from_cam_t(pred_cam_t, up_axis):
    """
    Get world offset for root_locator.
    
    IMPORTANT: root_locator should be at origin (0,0,0) because it parents both
    the body and camera. Moving root_locator moves both together, which doesn't
    affect the body's position relative to camera (i.e., alignment in camera view).
    
    The actual body offset relative to camera is handled by get_body_offset_from_cam_t().
    """
    # Root locator stays at origin - body offset is applied separately
    return Vector((0, 0, 0))


def get_body_offset_from_cam_t(pred_cam_t, up_axis):
    """
    Get offset to apply to body mesh/skeleton for correct camera alignment.
    
    pred_cam_t from SAM3DBody:
    - tx: horizontal offset (positive = body right of center)
    - ty: vertical offset (positive = body above center IN IMAGE SPACE)
    - tz: depth (camera distance)
    
    IMPORTANT: ty must be NEGATED because:
    1. SAM3DBody: ty positive = body above image center
    2. Maya camera is rotated -90° around X
    3. After axis conversion + camera rotation, the sign is flipped
    4. So we negate ty to compensate: -ty in code → +ty in Maya camera view
    
    For Maya with camera rotated -90° around X:
    - Maya X = horizontal in camera view
    - Maya Z = vertical in camera view (after camera rotation)
    - Maya Y = depth
    
    Blender Y-up export mapping:
    - Blender X → Maya X
    - Blender Y → Maya Z  
    - Blender Z → Maya Y
    """
    if not pred_cam_t or len(pred_cam_t) < 3:
        return Vector((0, 0, 0))
    
    tx, ty, tz = pred_cam_t[0], pred_cam_t[1], pred_cam_t[2]
    
    # Apply based on up_axis
    # NOTE: ty is NEGATED to match camera view convention
    if up_axis == "Y":
        # For Maya: tx→horizontal, -ty→vertical in camera view
        # Blender (X, Y, Z) → Maya (X, Z, Y)
        return Vector((tx, -ty, 0))
    elif up_axis == "Z":
        # Blender native Z-up
        return Vector((tx, 0, -ty))
    elif up_axis == "-Y":
        return Vector((tx, ty, 0))  # Double negative = positive
    elif up_axis == "-Z":
        return Vector((tx, 0, ty))  # Double negative = positive
    else:
        return Vector((tx, -ty, 0))


def apply_animated_body_offset(mesh_obj, armature_obj, frames, solved_camera_rotations, up_axis, frame_offset, smoothing=5):
    """
    Apply animated body offset that compensates for camera rotation.
    
    When camera rotates, the body position that gives correct screen alignment changes.
    This function computes per-frame body offset that accounts for:
    1. Target screen position from pred_cam_t
    2. Camera rotation from solver
    
    Math:
    - When camera pans right by θ, objects shift left on screen
    - To keep body at target screen position, body must shift right in world
    - Compensation: body_x = tx + depth × tan(pan_angle)
    - Similarly for tilt: body_y = -ty - depth × tan(tilt_angle)
    """
    if not frames or not solved_camera_rotations:
        print("[Blender] No frames or solved rotations for animated body offset")
        return
    
    num_frames = len(frames)
    has_solved = len(solved_camera_rotations) > 0
    
    if not has_solved:
        print("[Blender] No solved rotations - body offset stays static")
        return
    
    print(f"[Blender] Computing animated body offset with rotation compensation...")
    
    # Collect per-frame values
    all_tx = []
    all_ty = []
    all_tz = []
    all_pan = []
    all_tilt = []
    
    for frame_idx in range(num_frames):
        frame_data = frames[frame_idx]
        pred_cam_t = frame_data.get("pred_cam_t", [0, 0, 5])
        
        tx = pred_cam_t[0] if pred_cam_t and len(pred_cam_t) > 0 else 0
        ty = pred_cam_t[1] if pred_cam_t and len(pred_cam_t) > 1 else 0
        tz = pred_cam_t[2] if pred_cam_t and len(pred_cam_t) > 2 else 5
        
        all_tx.append(tx)
        all_ty.append(ty)
        all_tz.append(abs(tz) if abs(tz) > 0.1 else 5.0)
        
        if frame_idx < len(solved_camera_rotations):
            rot = solved_camera_rotations[frame_idx]
            all_pan.append(rot.get("pan", 0.0))
            all_tilt.append(rot.get("tilt", 0.0))
        else:
            all_pan.append(0.0)
            all_tilt.append(0.0)
    
    # Apply smoothing to ALL values (important for reducing jitter!)
    if smoothing > 1:
        all_tx = smooth_array(all_tx, smoothing)
        all_ty = smooth_array(all_ty, smoothing)
        all_tz = smooth_array(all_tz, smoothing)
        all_pan = smooth_array(all_pan, smoothing)
        all_tilt = smooth_array(all_tilt, smoothing)
        print(f"[Blender] Applied smoothing (window={smoothing}) to body offset and camera rotation")
    
    # Compute compensated body offset for each frame
    body_offsets = []
    for frame_idx in range(num_frames):
        tx = all_tx[frame_idx]
        ty = all_ty[frame_idx]
        depth = all_tz[frame_idx]
        pan = all_pan[frame_idx]
        tilt = all_tilt[frame_idx]
        
        # Compensate for camera rotation:
        # When camera pans right (positive pan), body appears to move left
        # To keep body at correct screen position, shift body right (add pan compensation)
        pan_compensation = depth * math.tan(pan)
        tilt_compensation = depth * math.tan(tilt)
        
        # Compensated body position
        compensated_tx = tx + pan_compensation
        compensated_ty = -ty - tilt_compensation  # ty negated as per convention
        
        if up_axis == "Y":
            body_offset = Vector((compensated_tx, compensated_ty, 0))
        elif up_axis == "Z":
            body_offset = Vector((compensated_tx, 0, compensated_ty))
        elif up_axis == "-Y":
            body_offset = Vector((compensated_tx, -compensated_ty, 0))
        elif up_axis == "-Z":
            body_offset = Vector((compensated_tx, 0, -compensated_ty))
        else:
            body_offset = Vector((compensated_tx, compensated_ty, 0))
        
        body_offsets.append(body_offset)
    
    # Debug: print first and last frame values
    print(f"[Blender] Frame 0: tx={all_tx[0]:.3f}, ty={all_ty[0]:.3f}, pan={math.degrees(all_pan[0]):.2f}°, offset={body_offsets[0]}")
    print(f"[Blender] Frame {num_frames-1}: tx={all_tx[-1]:.3f}, ty={all_ty[-1]:.3f}, pan={math.degrees(all_pan[-1]):.2f}°, offset={body_offsets[-1]}")
    
    # Apply animated offset to mesh
    if mesh_obj:
        for frame_idx, offset in enumerate(body_offsets):
            mesh_obj.location = offset
            mesh_obj.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
        print(f"[Blender] Mesh animated with {num_frames} keyframes (rotation-compensated offset)")
    
    # Apply animated offset to armature
    if armature_obj:
        for frame_idx, offset in enumerate(body_offsets):
            armature_obj.location = offset
            armature_obj.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
        print(f"[Blender] Skeleton animated with {num_frames} keyframes (rotation-compensated offset)")


def create_animated_mesh(all_frames, faces, fps, transform_func, world_translation_mode="none", up_axis="Y", frame_offset=0):
    """
    Create mesh with per-vertex animation using shape keys.
    """
    first_verts = all_frames[0].get("vertices")
    if not first_verts:
        return None
    
    # Get first frame world offset for initial position
    first_world_offset = Vector((0, 0, 0))
    if world_translation_mode == "baked":
        first_cam_t = all_frames[0].get("pred_cam_t")
        first_world_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
    
    # Create mesh with first frame vertices
    mesh = bpy.data.meshes.new("body_mesh")
    verts = []
    for v in first_verts:
        pos = Vector(transform_func(v))
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
    
    # Add basis shape key
    basis = obj.shape_key_add(name="Basis", from_mix=False)
    
    print(f"[Blender] Creating {len(all_frames)} shape keys (translation={world_translation_mode}, offset={frame_offset})...")
    
    # Create shape keys for each frame
    for frame_idx, frame_data in enumerate(all_frames):
        frame_verts = frame_data.get("vertices")
        if not frame_verts:
            continue
        
        # Get world offset for this frame
        frame_world_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            frame_world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        sk = obj.shape_key_add(name=f"frame_{frame_idx:04d}", from_mix=False)
        
        for j, v in enumerate(frame_verts):
            if j < len(sk.data):
                pos = Vector(transform_func(v))
                if world_translation_mode == "baked":
                    pos += frame_world_offset
                sk.data[j].co = pos
        
        # Keyframe shape key value with frame_offset
        actual_frame = frame_idx + frame_offset
        last_frame = frame_offset + len(all_frames) - 1
        is_last = (frame_idx == len(all_frames) - 1)
        is_first = (frame_idx == 0)
        
        # Fade in keyframe (value 0 before this shape activates)
        if not is_first:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=actual_frame - 1)
        
        # Active keyframe (value 1 at this frame)
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=actual_frame)
        
        # Fade out keyframe (value 0 after this shape deactivates)
        # Don't add fadeout for last frame - it should stay at 1
        if not is_last:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=actual_frame + 1)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Shape keys: {frame_idx + 1}/{len(all_frames)}")
    
    print(f"[Blender] Created mesh with {len(all_frames)} shape keys (frames {frame_offset} to {frame_offset + len(all_frames) - 1})")
    return obj


def create_skeleton_with_rotations(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None, joint_parents=None, frame_offset=0):
    """
    Create animated skeleton using armature with ROTATION keyframes.
    
    This uses the true joint rotation matrices from MHR model.
    Produces proper bone rotations for retargeting and animation editing.
    
    joint_parents: Array where joint_parents[i] = parent index of joint i (-1 for root)
    """
    first_joints = all_frames[0].get("joint_coords")
    first_rotations = all_frames[0].get("joint_rotations")
    
    if not first_joints:
        print("[Blender] No joint_coords in first frame, skipping skeleton")
        return None
    
    if not first_rotations:
        print("[Blender] No joint_rotations available, falling back to position-based skeleton")
        return create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator, joint_parents, frame_offset)
    
    num_joints = len(first_joints)
    print(f"[Blender] Creating rotation-based armature with {num_joints} bones...")
    
    # Create armature
    arm_data = bpy.data.armatures.new("Skeleton")
    armature = bpy.data.objects.new("Skeleton", arm_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    
    # Parent to root locator if in "root" mode
    if world_translation_mode == "root" and root_locator:
        armature.parent = root_locator
    
    # Get first frame world offset for initial bone positions
    first_offset = Vector((0, 0, 0))
    if world_translation_mode == "baked":
        first_cam_t = all_frames[0].get("pred_cam_t")
        first_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
    
    # Enter edit mode to create bones WITH HIERARCHY
    bpy.ops.object.mode_set(mode='EDIT')
    
    edit_bones = []
    for i in range(num_joints):
        bone = arm_data.edit_bones.new(f"joint_{i:03d}")
        pos = first_joints[i]
        head_pos = Vector(transform_func(pos))
        
        if world_translation_mode == "baked":
            head_pos += first_offset
        
        bone.head = head_pos
        # Tail will be adjusted after hierarchy is set
        bone.tail = head_pos + Vector((0, 0.03, 0))
        edit_bones.append(bone)
    
    # Set up parent-child hierarchy
    if joint_parents is not None:
        print(f"[Blender] Setting up bone hierarchy from joint_parents...")
        roots = []
        for i in range(num_joints):
            parent_idx = joint_parents[i]
            if parent_idx >= 0 and parent_idx < num_joints:
                edit_bones[i].parent = edit_bones[parent_idx]
                # Optionally connect bones if close enough
                # edit_bones[i].use_connect = False
            else:
                roots.append(i)
        print(f"[Blender] Found {len(roots)} root bone(s): {roots}")
        
        # Adjust bone tails to point toward first child (makes visualization better)
        for i in range(num_joints):
            children = [j for j in range(num_joints) if joint_parents[j] == i]
            if children:
                # Point tail toward average of children positions
                child_positions = [edit_bones[c].head for c in children]
                avg_child_pos = sum(child_positions, Vector((0, 0, 0))) / len(children)
                direction = avg_child_pos - edit_bones[i].head
                if direction.length > 0.001:
                    edit_bones[i].tail = edit_bones[i].head + direction.normalized() * 0.05
    else:
        print("[Blender] Warning: No joint_parents data, creating flat hierarchy")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate bones in pose mode using ROTATION keyframes
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Set rotation mode to quaternion for smoother interpolation
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = 'QUATERNION'
    
    print(f"[Blender] Animating {num_joints} bones with rotations over {len(all_frames)} frames...")
    
    # Pre-compute parent indices for faster lookup
    parent_indices = joint_parents if joint_parents is not None else [-1] * num_joints
    
    for frame_idx, frame_data in enumerate(all_frames):
        joints = frame_data.get("joint_coords")
        rotations = frame_data.get("joint_rotations")
        
        if not joints or not rotations:
            continue
        
        actual_frame = frame_idx + frame_offset
        bpy.context.scene.frame_set(actual_frame)
        
        # Get world offset for this frame (for position updates)
        world_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        # First pass: compute all global rotations in Blender space
        global_rots_blender = []
        for bone_idx in range(min(num_joints, len(rotations))):
            rot_3x3 = rotations[bone_idx]
            if rot_3x3 is None:
                global_rots_blender.append(Matrix.Identity(3))
            else:
                blender_rot = transform_rotation_matrix(rot_3x3, up_axis)
                global_rots_blender.append(blender_rot)
        
        # Second pass: convert global rotations to local rotations and apply
        for bone_idx in range(min(num_joints, len(joints), len(rotations))):
            pose_bone = armature.pose.bones[bone_idx]
            
            global_rot = global_rots_blender[bone_idx]
            
            # Convert global rotation to local rotation
            parent_idx = parent_indices[bone_idx]
            if parent_idx >= 0 and parent_idx < len(global_rots_blender):
                # Local = Parent_global^-1 * Global
                parent_global_rot = global_rots_blender[parent_idx]
                local_rot = parent_global_rot.inverted() @ global_rot
            else:
                # Root bone: local = global
                local_rot = global_rot
            
            # Convert to quaternion and apply
            quat = local_rot.to_quaternion()
            pose_bone.rotation_quaternion = quat
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=actual_frame)
            
            # Also update location for world translation modes (root bone only typically)
            if world_translation_mode == "baked" and parent_idx < 0:
                pos = joints[bone_idx]
                target = Vector(transform_func(pos)) + world_offset
                rest_head = Vector(armature.data.bones[bone_idx].head_local)
                offset = target - rest_head
                pose_bone.location = offset
                pose_bone.keyframe_insert(data_path="location", frame=actual_frame)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Animated {frame_idx + 1}/{len(all_frames)} frames (rotations)")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[Blender] Created hierarchical skeleton with {num_joints} bones (frame_offset={frame_offset})")
    return armature


def create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None, joint_parents=None, frame_offset=0):
    """
    Create animated skeleton using armature with POSITION keyframes.
    
    This is the legacy mode - bones animate via location offset from rest position.
    Shows exact joint positions but limited for retargeting.
    """
    first_joints = all_frames[0].get("joint_coords")
    if not first_joints:
        return None
    
    num_joints = len(first_joints)
    print(f"[Blender] Creating position-based armature with {num_joints} bones...")
    
    # Create armature
    arm_data = bpy.data.armatures.new("Skeleton")
    armature = bpy.data.objects.new("Skeleton", arm_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    
    # Parent to root locator if in "root" mode
    if world_translation_mode == "root" and root_locator:
        armature.parent = root_locator
    
    # Get first frame world offset for initial bone positions
    first_offset = Vector((0, 0, 0))
    if world_translation_mode == "baked":
        first_cam_t = all_frames[0].get("pred_cam_t")
        first_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
    
    # Enter edit mode to create bones WITH HIERARCHY
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
    
    # Set up parent-child hierarchy
    if joint_parents is not None:
        print(f"[Blender] Setting up bone hierarchy from joint_parents...")
        roots = []
        for i in range(num_joints):
            parent_idx = joint_parents[i]
            if parent_idx >= 0 and parent_idx < num_joints:
                edit_bones[i].parent = edit_bones[parent_idx]
            else:
                roots.append(i)
        print(f"[Blender] Found {len(roots)} root bone(s): {roots}")
        
        # Adjust bone tails to point toward first child
        for i in range(num_joints):
            children = [j for j in range(num_joints) if joint_parents[j] == i]
            if children:
                child_positions = [edit_bones[c].head for c in children]
                avg_child_pos = sum(child_positions, Vector((0, 0, 0))) / len(children)
                direction = avg_child_pos - edit_bones[i].head
                if direction.length > 0.001:
                    edit_bones[i].tail = edit_bones[i].head + direction.normalized() * 0.05
    else:
        print("[Blender] Warning: No joint_parents data, creating flat hierarchy")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate bones in pose mode using LOCATION keyframes
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    print(f"[Blender] Animating {num_joints} bones with positions over {len(all_frames)} frames...")
    
    # Store rest positions for offset calculation
    rest_heads = [Vector(armature.pose.bones[i].bone.head_local) for i in range(num_joints)]
    
    for frame_idx, frame_data in enumerate(all_frames):
        joints = frame_data.get("joint_coords")
        if not joints:
            continue
        
        actual_frame = frame_idx + frame_offset
        bpy.context.scene.frame_set(actual_frame)
        
        # Get world offset for this frame
        world_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        for bone_idx in range(min(num_joints, len(joints))):
            pose_bone = armature.pose.bones[bone_idx]
            
            pos = joints[bone_idx]
            target = Vector(transform_func(pos))
            
            if world_translation_mode == "baked":
                target += world_offset
            
            # Calculate offset from rest position
            offset = target - rest_heads[bone_idx]
            pose_bone.location = offset
            pose_bone.keyframe_insert(data_path="location", frame=actual_frame)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Animated {frame_idx + 1}/{len(all_frames)} frames (positions)")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[Blender] Created position-based skeleton with {num_joints} bones (frame_offset={frame_offset})")
    return armature


def create_skeleton(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None, skeleton_mode="rotations", joint_parents=None, frame_offset=0):
    """
    Create animated skeleton - dispatcher function.
    
    skeleton_mode:
    - "rotations": Use true joint rotation matrices from MHR (recommended)
    - "positions": Use joint positions only (legacy)
    
    joint_parents: Array defining bone hierarchy
    """
    if skeleton_mode == "rotations":
        return create_skeleton_with_rotations(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator, joint_parents, frame_offset)
    else:
        return create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator, joint_parents, frame_offset)


def create_root_locator(all_frames, fps, up_axis, flip_x=False, frame_offset=0):
    """
    Create a root locator that carries the world translation.
    """
    print("[Blender] Creating root locator with world translation...")
    
    # DEBUG: Show first frame values
    if all_frames:
        first_cam_t = all_frames[0].get("pred_cam_t")
        if first_cam_t and len(first_cam_t) >= 3:
            tx, ty, tz = first_cam_t[0], first_cam_t[1], first_cam_t[2]
            world_x = tx * abs(tz) * 0.5
            world_y = ty * abs(tz) * 0.5
            print(f"[Blender] DEBUG: Frame 0 pred_cam_t: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
            print(f"[Blender] DEBUG: world_offset calc: x={world_x:.4f}, y={world_y:.4f}")
            print(f"[Blender] DEBUG: Final root_locator (up={up_axis}): x={-world_x:.4f}, y={-world_y:.4f}")
    
    root = bpy.data.objects.new("root_locator", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 0.1
    bpy.context.collection.objects.link(root)
    
    # Animate root position based on pred_cam_t
    for frame_idx, frame_data in enumerate(all_frames):
        frame_cam_t = frame_data.get("pred_cam_t")
        world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        # Apply flip_x to the world offset
        if flip_x:
            world_offset = Vector((-world_offset.x, world_offset.y, world_offset.z))
        
        root.location = world_offset
        root.keyframe_insert(data_path="location", frame=frame_idx + frame_offset)
    
    print(f"[Blender] Root locator animated over {len(all_frames)} frames (offset={frame_offset}, flip_x={flip_x})")
    return root


def create_translation_track(all_frames, fps, up_axis, frame_offset=0):
    """
    Create a separate locator that shows the world path.
    """
    print("[Blender] Creating separate translation track...")
    
    track = bpy.data.objects.new("translation_track", None)
    track.empty_display_type = 'PLAIN_AXES'
    track.empty_display_size = 0.15
    bpy.context.collection.objects.link(track)
    
    for frame_idx, frame_data in enumerate(all_frames):
        frame_cam_t = frame_data.get("pred_cam_t")
        world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        track.location = world_offset
        track.keyframe_insert(data_path="location", frame=frame_idx + frame_offset)
    
    print(f"[Blender] Translation track animated over {len(all_frames)} frames (offset={frame_offset})")
    return track


def create_camera(all_frames, fps, transform_func, up_axis, sensor_width=36.0, world_translation_mode="none", animate_camera=False, frame_offset=0, camera_follow_root=False, camera_use_rotation=False, camera_static=False, camera_smoothing=0, flip_x=False, solved_camera_rotations=None):
    """
    Create camera with focal length from SAM3DBody.
    
    The camera is positioned to match SAM3DBody's projection, accounting for:
    - Focal length (converted from pixels to mm)
    - Distance from pred_cam_t[2]
    - Offset from bbox position relative to image center
    
    Args:
        animate_camera: If True and mode=="camera", animate camera position with world offset.
        camera_follow_root: If True, camera will be parented to root_locator and needs
                           LOCAL animation to show character at correct screen position.
        camera_use_rotation: If True, use pan/tilt rotation instead of translation.
                            Better for tripod/handheld shots where camera rotates to follow subject.
        camera_static: If True, camera stays completely fixed (no rotation or translation animation).
                      Used with body_offset to position body correctly.
        camera_smoothing: Smoothing window for camera values to reduce jitter (0=none).
        flip_x: Whether X axis is flipped (affects camera pan direction).
        frame_offset: Starting frame number for keyframes.
        solved_camera_rotations: Optional list of solved rotations from Camera Rotation Solver.
                                Each entry has {frame, pan, tilt, roll} in radians.
    """
    has_solved = solved_camera_rotations is not None and len(solved_camera_rotations) > 0
    print(f"[Blender] Creating camera (animate={animate_camera}, follow_root={camera_follow_root}, use_rotation={camera_use_rotation}, static={camera_static}, smoothing={camera_smoothing}, solved_rotations={has_solved})...")
    
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)
    
    # Set sensor width
    cam_data.sensor_width = sensor_width
    
    # Get image dimensions
    first_frame = all_frames[0]
    image_size = first_frame.get("image_size")
    
    image_width = 1920  # Default
    image_height = 1080
    if image_size:
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            image_width, image_height = image_size[0], image_size[1]
    
    # Set sensor height to match aspect ratio (important for correct projection!)
    aspect_ratio = image_width / image_height
    cam_data.sensor_height = sensor_width / aspect_ratio
    cam_data.sensor_fit = 'HORIZONTAL'
    
    # Get focal length from first frame
    first_focal = first_frame.get("focal_length")
    focal_px = 1000  # Default
    if first_focal:
        if isinstance(first_focal, (list, tuple)):
            focal_px = first_focal[0]
        else:
            focal_px = first_focal
    
    # Convert focal length: focal_mm = focal_px * sensor_width / image_width
    focal_mm = focal_px * (sensor_width / image_width)
    cam_data.lens = focal_mm
    print(f"[Blender] Focal length: {focal_px:.0f}px -> {focal_mm:.1f}mm")
    print(f"[Blender] Image size: {image_width}x{image_height}, aspect: {aspect_ratio:.3f}")
    print(f"[Blender] Sensor: {sensor_width:.1f}mm x {cam_data.sensor_height:.1f}mm")
    
    # Get pred_cam_t - this is the KEY to matching SAM3DBody's projection
    # pred_cam_t = [tx, ty, tz] where:
    #   tx, ty = normalized screen offset (roughly -1 to 1)
    #   tz = depth (camera distance)
    first_cam_t = first_frame.get("pred_cam_t")
    
    cam_distance = 3.0
    tx, ty = 0.0, 0.0
    if first_cam_t and len(first_cam_t) >= 3:
        tx, ty = first_cam_t[0], first_cam_t[1]
        cam_distance = abs(first_cam_t[2])
    
    # IMPORTANT: When using world_translation_mode="root", body offset is applied
    # directly to the mesh/skeleton via get_body_offset_from_cam_t().
    # In this case, camera should look straight at origin - NO target_offset!
    # 
    # The body_offset positions the body correctly relative to camera.
    # Adding target_offset would DOUBLE-count the offset.
    if world_translation_mode == "root":
        target_offset = Vector((0, 0, 0))
        print(f"[Blender] Root mode: Camera looks at origin, body_offset applied to mesh/skeleton")
    else:
        # Legacy mode: body at origin, camera target offset to frame correctly
        scale_factor = cam_distance * 0.5
        if up_axis == "Y":
            target_offset = Vector((tx * scale_factor, -ty * scale_factor, 0))
        elif up_axis == "Z":
            target_offset = Vector((tx * scale_factor, 0, -ty * scale_factor))
        elif up_axis == "-Y":
            target_offset = Vector((tx * scale_factor, ty * scale_factor, 0))
        elif up_axis == "-Z":
            target_offset = Vector((tx * scale_factor, 0, ty * scale_factor))
        else:
            target_offset = Vector((tx * scale_factor, -ty * scale_factor, 0))
    
    print(f"[Blender] pred_cam_t: tx={tx:.3f}, ty={ty:.3f}, tz={cam_distance:.2f}")
    print(f"[Blender] Camera target offset: {target_offset}")
    
    # Create target for camera orientation
    target = bpy.data.objects.new("cam_target", None)
    target.location = target_offset
    bpy.context.collection.objects.link(target)
    
    # Set camera direction based on up_axis
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
    
    # For "camera" mode with animate_camera=True: animated camera to follow character
    if animate_camera and world_translation_mode == "camera":
        
        if camera_use_rotation:
            # ROTATION MODE: Camera at fixed position, rotates to follow
            # Initial setup: camera looking straight at origin
            camera.location = base_dir * cam_distance  # No target_offset!
            
            # Point camera at origin to get base rotation
            origin_target = bpy.data.objects.new("origin_target", None)
            origin_target.location = Vector((0, 0, 0))
            bpy.context.collection.objects.link(origin_target)
            
            constraint = camera.constraints.new(type='TRACK_TO')
            constraint.target = origin_target
            constraint.track_axis = 'TRACK_NEGATIVE_Z'
            
            if up_axis == "Y" or up_axis == "-Y":
                constraint.up_axis = 'UP_Y'
            elif up_axis == "Z" or up_axis == "-Z":
                constraint.up_axis = 'UP_Z'
            else:
                constraint.up_axis = 'UP_Y'
            
            bpy.context.view_layer.update()
            base_rotation = camera.matrix_world.to_euler()
            camera.rotation_euler = base_rotation.copy()
            camera.constraints.remove(constraint)
            bpy.data.objects.remove(origin_target)
            
            camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
            camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
            camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
            base_rotation = camera.rotation_euler.copy()
            
            # Also remove the offset target we created earlier
            bpy.data.objects.remove(target)
            
            print(f"[Blender] Camera using ROTATION (Pan/Tilt) to follow character...")
            
            # Get pan/tilt axis configuration based on up_axis
            # Pan = rotation around UP axis, Tilt = rotation around horizontal axis
            # 
            # KEY INSIGHT for Y-up (Maya default):
            # - Positive Y rotation = pan LEFT = origin shifts RIGHT in view
            # - Positive X rotation = tilt DOWN = origin shifts UP in view
            # 
            # So for tx > 0 (body on RIGHT), we need positive Y (pan left, origin goes right)
            # For ty > 0 (body BELOW center), we need negative X (tilt up, origin goes down)
            if up_axis == "Y":
                pan_axis = 1   # Y axis for pan (yaw)
                tilt_axis = 0  # X axis for tilt (pitch)
                tilt_sign = -1  # ty > 0 → tilt UP (negative X) → origin appears lower
                pan_sign = 1    # tx > 0 → pan LEFT (positive Y) → origin appears right
            elif up_axis == "Z":
                pan_axis = 2   # Z axis for pan
                tilt_axis = 0  # X axis for tilt
                tilt_sign = -1
                pan_sign = 1
            elif up_axis == "-Y":
                pan_axis = 1
                tilt_axis = 0
                tilt_sign = 1
                pan_sign = -1
            elif up_axis == "-Z":
                pan_axis = 2
                tilt_axis = 0
                tilt_sign = 1
                pan_sign = -1
            else:
                pan_axis = 1
                tilt_axis = 0
                tilt_sign = -1
                pan_sign = 1
            
            for frame_idx, frame_data in enumerate(all_frames):
                frame_cam_t = frame_data.get("pred_cam_t")
                
                frame_distance = cam_distance
                if frame_cam_t and len(frame_cam_t) > 2:
                    frame_distance = abs(frame_cam_t[2])
                
                # Use solved camera rotations if available, otherwise compute from pred_cam_t
                if has_solved and frame_idx < len(solved_camera_rotations):
                    solved_rot = solved_camera_rotations[frame_idx]
                    # Solved rotation = actual camera motion from background tracking
                    # This should be the SAME for all runners in the same video!
                    pan_angle = solved_rot.get("pan", 0.0) * pan_sign
                    tilt_angle = solved_rot.get("tilt", 0.0) * tilt_sign
                    roll_angle = solved_rot.get("roll", 0.0)
                    
                    # Apply solved rotation only - no per-body baseline offset
                    camera.rotation_euler = base_rotation.copy()
                    camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                    camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                    camera.rotation_euler[2] = base_rotation[2] + roll_angle
                    
                elif frame_cam_t and len(frame_cam_t) >= 3:
                    tx, ty, tz = frame_cam_t[0], frame_cam_t[1], frame_cam_t[2]
                    
                    # Fallback: compute from pred_cam_t (body screen position)
                    # This is per-body and should only be used when no solved rotation available
                    depth = abs(tz) if abs(tz) > 0.1 else 0.1
                    pan_angle = math.atan2(tx, depth) * pan_sign
                    tilt_angle = math.atan2(ty, depth) * tilt_sign
                    
                    camera.rotation_euler = base_rotation.copy()
                    camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                    camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                
                # Camera position: just depth along base direction
                camera.location = base_dir * frame_distance
                
                camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset + frame_idx)
                camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
            
            if has_solved:
                print(f"[Blender] Camera uses SOLVED rotation over {len(all_frames)} frames (same for all bodies)")
            else:
                print(f"[Blender] Camera ROTATES (pan/tilt from pred_cam_t) over {len(all_frames)} frames (up_axis={up_axis})")
        
        else:
            # TRANSLATION MODE: Camera moves laterally to follow character
            # Set initial rotation pointing at target
            camera.location = base_dir * cam_distance + target_offset
            
            constraint = camera.constraints.new(type='TRACK_TO')
            constraint.target = target
            constraint.track_axis = 'TRACK_NEGATIVE_Z'
            
            if up_axis == "Y" or up_axis == "-Y":
                constraint.up_axis = 'UP_Y'
            elif up_axis == "Z" or up_axis == "-Z":
                constraint.up_axis = 'UP_Z'
            else:
                constraint.up_axis = 'UP_Y'
            
            bpy.context.view_layer.update()
            camera.rotation_euler = camera.matrix_world.to_euler()
            camera.constraints.remove(constraint)
            
            camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
            camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
            camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
            
            bpy.data.objects.remove(target)
            
            print(f"[Blender] Camera using TRANSLATION to follow character...")
            
            for frame_idx, frame_data in enumerate(all_frames):
                frame_cam_t = frame_data.get("pred_cam_t")
                
                frame_distance = cam_distance
                if frame_cam_t and len(frame_cam_t) > 2:
                    frame_distance = abs(frame_cam_t[2])
                
                world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
                camera.location = base_dir * frame_distance + target_offset - world_offset
                camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
            
            print(f"[Blender] Camera TRANSLATES over {len(all_frames)} frames (up_axis={up_axis})")
    
    elif camera_use_rotation:
        # "None" mode with rotation: Camera rotates to show character at correct screen position
        # Body stays at origin, camera pans/tilts to frame it correctly
        camera.location = base_dir * cam_distance  # No target_offset
        
        # Point camera at origin to get base rotation
        origin_target = bpy.data.objects.new("origin_target", None)
        origin_target.location = Vector((0, 0, 0))
        bpy.context.collection.objects.link(origin_target)
        
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = origin_target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        
        if up_axis == "Y" or up_axis == "-Y":
            constraint.up_axis = 'UP_Y'
        elif up_axis == "Z" or up_axis == "-Z":
            constraint.up_axis = 'UP_Z'
        else:
            constraint.up_axis = 'UP_Y'
        
        bpy.context.view_layer.update()
        base_rotation = camera.matrix_world.to_euler()
        camera.rotation_euler = base_rotation.copy()
        camera.constraints.remove(constraint)
        bpy.data.objects.remove(origin_target)
        bpy.data.objects.remove(target)  # Remove unused target
        
        camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
        camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
        camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
        base_rotation = camera.rotation_euler.copy()
        
        print(f"[Blender] Camera using ROTATION (Pan/Tilt) with body at origin...")
        
        # Get pan/tilt axis configuration
        # Same logic as "Baked into Camera" rotation mode
        if up_axis == "Y":
            pan_axis = 1
            tilt_axis = 0
            tilt_sign = -1  # ty > 0 → tilt UP (negative X) → origin appears lower
            pan_sign = 1    # tx > 0 → pan LEFT (positive Y) → origin appears right
        elif up_axis == "Z":
            pan_axis = 2
            tilt_axis = 0
            tilt_sign = -1
            pan_sign = 1
        elif up_axis == "-Y":
            pan_axis = 1
            tilt_axis = 0
            tilt_sign = 1
            pan_sign = -1
        elif up_axis == "-Z":
            pan_axis = 2
            tilt_axis = 0
            tilt_sign = 1
            pan_sign = -1
        else:
            pan_axis = 1
            tilt_axis = 0
            tilt_sign = -1
            pan_sign = 1
        
        for frame_idx, frame_data in enumerate(all_frames):
            frame_cam_t = frame_data.get("pred_cam_t")
            
            frame_distance = cam_distance
            if frame_cam_t and len(frame_cam_t) > 2:
                frame_distance = abs(frame_cam_t[2])
            
            if frame_cam_t and len(frame_cam_t) >= 3:
                tx, ty, tz = frame_cam_t[0], frame_cam_t[1], frame_cam_t[2]
                
                # angle = atan2(offset, depth) to match 2D projection
                depth = abs(tz) if abs(tz) > 0.1 else 0.1
                pan_angle = math.atan2(tx, depth) * pan_sign
                tilt_angle = math.atan2(ty, depth) * tilt_sign
                
                camera.rotation_euler = base_rotation.copy()
                camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                
                camera.location = base_dir * frame_distance
                
                camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset + frame_idx)
                camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
        
        print(f"[Blender] Camera ROTATES (pan/tilt) over {len(all_frames)} frames, body at origin (up_axis={up_axis})")
    
    else:
        # Static camera - positioned with offset for alignment
        camera.location = base_dir * cam_distance + target_offset
        
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        
        # Set camera up axis to match scene up axis
        if up_axis == "Y" or up_axis == "-Y":
            constraint.up_axis = 'UP_Y'
        elif up_axis == "Z" or up_axis == "-Z":
            constraint.up_axis = 'UP_Z'
        else:
            constraint.up_axis = 'UP_Y'
        
        bpy.context.view_layer.update()
        camera.rotation_euler = camera.matrix_world.to_euler()
        constraint = camera.constraints.get("Track To")
        if constraint:
            camera.constraints.remove(constraint)
        
        bpy.data.objects.remove(target)
        
        camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
        camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
        camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
        
        print(f"[Blender] Camera static at {camera.location}, up_axis={up_axis}")
    
    # For camera_follow_root: animate camera LOCAL position or rotation
    # Camera is parented to root_locator, so we need local animation
    # to show character at correct screen position (not always centered)
    # 
    # EXCEPTION: If camera_static is True, skip ALL camera animation!
    # In static mode, body_offset positions the body correctly and camera stays fixed.
    if camera_follow_root and not camera_static:
        print(f"[Blender] Adding local animation for camera (follows root locator)...")
        
        # Get base camera direction based on up_axis
        if up_axis == "Y":
            base_dir = Vector((0, 0, 1))
            pan_axis = 1   # Y axis for pan
            tilt_axis = 0  # X axis for tilt
        elif up_axis == "Z":
            base_dir = Vector((0, 1, 0))
            pan_axis = 2   # Z axis for pan
            tilt_axis = 0  # X axis for tilt
        elif up_axis == "-Y":
            base_dir = Vector((0, 0, -1))
            pan_axis = 1
            tilt_axis = 0
        elif up_axis == "-Z":
            base_dir = Vector((0, -1, 0))
            pan_axis = 2
            tilt_axis = 0
        else:
            base_dir = Vector((0, 0, 1))
            pan_axis = 1
            tilt_axis = 0
        
        # Store base rotation from static camera setup
        base_rotation = camera.rotation_euler.copy()
        
        if camera_use_rotation:
            # ROTATION MODE: Camera pans/tilts to follow character (like real camera operator)
            print(f"[Blender] Using PAN/TILT rotation to frame character")
            
            # Check if we have solved camera rotations from Camera Solver
            has_solved = solved_camera_rotations is not None and len(solved_camera_rotations) > 0
            
            if has_solved:
                # USE SOLVED ROTATIONS FROM CAMERA SOLVER
                # These come from background tracking and represent actual camera movement
                # body_offset (from frame 0) handles initial positioning
                # solved rotations handle frame-to-frame camera pan/tilt
                print(f"[Blender] Using SOLVED camera rotations ({len(solved_camera_rotations)} frames)")
                
                # Get first frame depth for camera distance
                first_cam_t = all_frames[0].get("pred_cam_t", [0, 0, 5])
                base_depth = abs(first_cam_t[2]) if first_cam_t and len(first_cam_t) > 2 else 5.0
                
                for frame_idx in range(len(all_frames)):
                    if frame_idx < len(solved_camera_rotations):
                        solved_rot = solved_camera_rotations[frame_idx]
                        pan_angle = solved_rot.get("pan", 0.0)
                        tilt_angle = solved_rot.get("tilt", 0.0)
                        roll_angle = solved_rot.get("roll", 0.0)
                    else:
                        pan_angle = 0.0
                        tilt_angle = 0.0
                        roll_angle = 0.0
                    
                    # Get per-frame depth (camera distance can vary)
                    frame_cam_t = all_frames[frame_idx].get("pred_cam_t")
                    if frame_cam_t and len(frame_cam_t) > 2:
                        depth = abs(frame_cam_t[2])
                    else:
                        depth = base_depth
                    
                    # Apply solved rotation
                    # Note: solved rotations are already in the correct sign convention
                    camera.rotation_euler = base_rotation.copy()
                    camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                    camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                    camera.rotation_euler[2] = base_rotation[2] + roll_angle
                    
                    # Camera distance from root
                    camera.location = base_dir * depth
                    
                    camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset + frame_idx)
                    camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
                
                print(f"[Blender] Camera rotation from SOLVED values (background tracking)")
                print(f"[Blender] body_offset is STATIC (frame 0) - camera rotation handles alignment")
            
            else:
                # FALLBACK: Compute pan/tilt from pred_cam_t
                # This is per-body and may cause issues with multi-body alignment
                print(f"[Blender] No solved rotations - computing from pred_cam_t (fallback)")
                
                # Collect all camera values first for smoothing
                all_tx = []
                all_ty = []
                all_tz = []
                for frame_data in all_frames:
                    frame_cam_t = frame_data.get("pred_cam_t", [0, 0, 3])
                    if frame_cam_t and len(frame_cam_t) >= 3:
                        all_tx.append(frame_cam_t[0])
                        all_ty.append(frame_cam_t[1])
                        all_tz.append(frame_cam_t[2])
                    else:
                        all_tx.append(0)
                        all_ty.append(0)
                        all_tz.append(3)
                
                # Apply smoothing if requested
                if camera_smoothing > 1:
                    all_tx = smooth_array(all_tx, camera_smoothing)
                    all_ty = smooth_array(all_ty, camera_smoothing)
                    all_tz = smooth_array(all_tz, camera_smoothing)
                    print(f"[Blender] Applied camera smoothing (window={camera_smoothing})")
                
                # Print first frame values for debugging
                print(f"[Blender] Frame 0 pred_cam_t: tx={all_tx[0]:.3f}, ty={all_ty[0]:.3f}, tz={all_tz[0]:.3f}")
                
                for frame_idx in range(len(all_frames)):
                    tx = all_tx[frame_idx]
                    ty = all_ty[frame_idx]
                    tz = all_tz[frame_idx]
                    depth = abs(tz) if abs(tz) > 0.1 else 0.1
                    
                    # Apply coordinate transform for pan (horizontal)
                    # flip_x affects whether we negate tx
                    if up_axis == "Y" or up_axis == "-Y":
                        tx_cam = tx if flip_x else -tx
                        # ty is NOT negated - positive ty means body lower in frame,
                        # so camera should tilt DOWN (positive X rotation)
                        ty_cam = ty
                    else:  # Z-up
                        tx_cam = tx if flip_x else -tx
                        ty_cam = ty
                    
                    # Compute angles directly using atan2
                    pan_angle = math.atan2(tx_cam, depth)
                    tilt_angle = math.atan2(ty_cam, depth)
                    
                    # Apply rotation
                    camera.rotation_euler = base_rotation.copy()
                    camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                    camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                    
                    # Also animate depth (camera distance from root)
                    camera.location = base_dir * depth
                    
                    camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset + frame_idx)
                    camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
                
                print(f"[Blender] Camera pan/tilt animated over {len(all_frames)} frames")
                print(f"[Blender] Camera ROTATES to follow character (from pred_cam_t fallback)")
        
        else:
            # TRANSLATION MODE: Camera moves laterally to show character at offset position
            print(f"[Blender] Using local TRANSLATION to frame character")
            
            for frame_idx, frame_data in enumerate(all_frames):
                frame_cam_t = frame_data.get("pred_cam_t")
                
                if frame_cam_t and len(frame_cam_t) >= 3:
                    tx, ty, tz = frame_cam_t[0], frame_cam_t[1], frame_cam_t[2]
                    depth = abs(tz) if tz else 3.0
                    
                    # Root locator moves by world_offset = (tx * depth * 0.5, ty * depth * 0.5)
                    # To show character at correct screen position, camera local position
                    # should have the INVERSE lateral offset
                    lateral_x = -tx * depth * 0.5
                    lateral_y = -ty * depth * 0.5
                    
                    # Apply based on up_axis
                    if up_axis == "Y":
                        local_offset = Vector((lateral_x, -lateral_y, 0))
                    elif up_axis == "Z":
                        local_offset = Vector((lateral_x, 0, -lateral_y))
                    elif up_axis == "-Y":
                        local_offset = Vector((lateral_x, lateral_y, 0))
                    elif up_axis == "-Z":
                        local_offset = Vector((lateral_x, 0, lateral_y))
                    else:
                        local_offset = Vector((lateral_x, -lateral_y, 0))
                    
                    # Camera position = forward direction * depth + lateral offset
                    camera.location = base_dir * depth + local_offset
                    camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
            
            print(f"[Blender] Camera local translation animated over {len(all_frames)} frames")
            print(f"[Blender] Camera TRANSLATES to keep character at correct screen position")
    
    elif camera_follow_root and camera_static:
        # Static camera mode - no animation needed
        # body_offset positions the body correctly relative to the fixed camera
        print(f"[Blender] Camera STATIC - body_offset positions character, no camera animation")
    
    # Animate focal length if it varies across frames
    focal_lengths = []
    for frame_data in all_frames:
        fl = frame_data.get("focal_length")
        if fl is not None:
            if isinstance(fl, (list, tuple)):
                fl = fl[0]
            focal_lengths.append(fl)
    
    if len(focal_lengths) > 1:
        # Check if focal length actually varies
        fl_min, fl_max = min(focal_lengths), max(focal_lengths)
        if fl_max - fl_min > 1.0:  # More than 1 pixel difference
            print(f"[Blender] Animating focal length: {fl_min:.0f}px to {fl_max:.0f}px")
            
            for frame_idx, frame_data in enumerate(all_frames):
                fl = frame_data.get("focal_length")
                if fl is not None:
                    if isinstance(fl, (list, tuple)):
                        fl = fl[0]
                    focal_mm = fl * (sensor_width / image_width)
                    cam_data.lens = focal_mm
                    cam_data.keyframe_insert(data_path="lens", frame=frame_offset + frame_idx)
            
            print(f"[Blender] Focal length animated over {len(all_frames)} frames")
        else:
            print(f"[Blender] Focal length constant at ~{fl_min:.0f}px")
    
    return camera


def export_fbx(output_path, axis_forward, axis_up):
    """Export to FBX."""
    print(f"[Blender] Exporting FBX: {output_path}")
    print(f"[Blender] Orientation: forward={axis_forward}, up={axis_up}")
    
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward=axis_forward,
        axis_up=axis_up,
        object_types={'MESH', 'ARMATURE', 'EMPTY', 'CAMERA'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        use_armature_deform_only=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
    )
    print(f"[Blender] FBX export complete")


def export_alembic(output_path):
    """Export to Alembic (.abc)."""
    print(f"[Blender] Exporting Alembic: {output_path}")
    
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
    print(f"[Blender] Alembic export complete")


def main():
    argv = sys.argv
    try:
        idx = argv.index("--")
        args = argv[idx + 1:]
    except ValueError:
        print("[Blender] Error: No arguments")
        sys.exit(1)
    
    if len(args) < 2:
        print("[Blender] Usage: blender --background --python script.py -- input.json output.fbx [up_axis] [include_mesh] [include_camera]")
        sys.exit(1)
    
    input_json = args[0]
    output_path = args[1]
    up_axis = args[2] if len(args) > 2 else "Y"
    include_mesh = args[3] == "1" if len(args) > 3 else True
    include_camera = args[4] == "1" if len(args) > 4 else True
    
    # Detect output format
    output_format = "fbx"
    if output_path.lower().endswith(".abc"):
        output_format = "abc"
    
    print(f"[Blender] Input: {input_json}")
    print(f"[Blender] Output: {output_path}")
    print(f"[Blender] Format: {output_format.upper()}")
    print(f"[Blender] Up axis: {up_axis}")
    print(f"[Blender] Include mesh: {include_mesh}")
    print(f"[Blender] Include camera: {include_camera}")
    
    if not os.path.exists(input_json):
        print(f"[Blender] Error: File not found: {input_json}")
        sys.exit(1)
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    fps = data.get("fps", 24.0)
    frames = data.get("frames", [])
    faces = data.get("faces")
    joint_parents = data.get("joint_parents")  # Get hierarchy data
    sensor_width = data.get("sensor_width", 36.0)
    world_translation_mode = data.get("world_translation_mode", "none")
    skeleton_mode = data.get("skeleton_mode", "rotations")  # New: default to rotations
    flip_x = data.get("flip_x", False)  # Mirror on X axis
    frame_offset = data.get("frame_offset", 0)  # Start frame offset for Maya
    animate_camera = data.get("animate_camera", False)  # Only animate camera if translation baked to it
    camera_follow_root = data.get("camera_follow_root", False)  # Parent camera to root locator
    camera_use_rotation = data.get("camera_use_rotation", False)  # Use rotation instead of translation
    camera_static = data.get("camera_static", False)  # Disable all camera animation
    camera_smoothing = data.get("camera_smoothing", 0)  # Smoothing window for camera animation
    solved_camera_rotations = data.get("solved_camera_rotations", None)  # From Camera Rotation Solver
    
    print(f"[Blender] {len(frames)} frames at {fps} fps")
    print(f"[Blender] Frame offset: {frame_offset} (animation runs from frame {frame_offset} to {frame_offset + len(frames) - 1})")
    print(f"[Blender] Sensor width: {sensor_width}mm")
    print(f"[Blender] World translation mode: {world_translation_mode}")
    print(f"[Blender] Skeleton mode: {skeleton_mode}")
    print(f"[Blender] Flip X: {flip_x}")
    print(f"[Blender] Animate camera: {animate_camera}")
    print(f"[Blender] Camera follow root: {camera_follow_root}")
    print(f"[Blender] Camera use rotation: {camera_use_rotation}")
    print(f"[Blender] Camera static: {camera_static}")
    print(f"[Blender] Camera smoothing: {camera_smoothing}")
    print(f"[Blender] Solved camera rotations: {len(solved_camera_rotations) if solved_camera_rotations else 0} frames")
    print(f"[Blender] Joint parents available: {joint_parents is not None}")
    
    if not frames:
        print("[Blender] Error: No frames")
        sys.exit(1)
    
    # Check if rotation data is available
    has_rotations = frames[0].get("joint_rotations") is not None
    print(f"[Blender] Rotation data available: {has_rotations}")
    
    if skeleton_mode == "rotations" and not has_rotations:
        print("[Blender] Warning: Rotation mode requested but no data available. Falling back to positions.")
        skeleton_mode = "positions"
    
    # Get transformation
    global FLIP_X
    FLIP_X = flip_x
    transform_func, axis_forward, axis_up_export = get_transform_for_axis(up_axis, flip_x)
    
    clear_scene()
    
    # Set scene frame range with offset
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = frame_offset
    bpy.context.scene.frame_end = frame_offset + len(frames) - 1
    
    # Set render resolution to match video (important for camera projection!)
    first_frame = frames[0]
    image_size = first_frame.get("image_size")
    if image_size and len(image_size) >= 2:
        bpy.context.scene.render.resolution_x = int(image_size[0])
        bpy.context.scene.render.resolution_y = int(image_size[1])
        bpy.context.scene.render.resolution_percentage = 100
        print(f"[Blender] Render resolution set to {image_size[0]}x{image_size[1]}")
    
    # Create root locator if needed (for "root" mode)
    root_locator = None
    body_offset = Vector((0, 0, 0))
    if world_translation_mode == "root":
        root_locator = create_root_locator(frames, fps, up_axis, flip_x, frame_offset)
        
        # Get body offset for aligning body relative to camera (STATIC from frame 0)
        first_cam_t = frames[0].get("pred_cam_t")
        body_offset = get_body_offset_from_cam_t(first_cam_t, up_axis)
        print(f"[Blender] Body offset for camera alignment: {body_offset}")
    
    # Create mesh with shape keys
    mesh_obj = None
    if include_mesh:
        mesh_obj = create_animated_mesh(frames, faces, fps, transform_func, world_translation_mode, up_axis, frame_offset)
        # Parent mesh to root locator if in "root" mode
        if world_translation_mode == "root" and root_locator and mesh_obj:
            mesh_obj.parent = root_locator
            # Apply STATIC body offset for correct camera alignment
            mesh_obj.location = body_offset
            print(f"[Blender] Mesh offset applied: {body_offset}")
    
    # Create skeleton (armature with bones and hierarchy)
    armature_obj = create_skeleton(frames, fps, transform_func, world_translation_mode, up_axis, root_locator, skeleton_mode, joint_parents, frame_offset)
    
    # Apply body offset to skeleton as well
    if world_translation_mode == "root" and root_locator and armature_obj:
        armature_obj.location = body_offset
        print(f"[Blender] Skeleton offset applied: {body_offset}")
    
    # Create separate translation track if in "separate" mode
    if world_translation_mode == "separate":
        create_translation_track(frames, fps, up_axis, frame_offset)
    
    # Create camera
    camera_obj = None
    if include_camera:
        camera_obj = create_camera(frames, fps, transform_func, up_axis, sensor_width, world_translation_mode, animate_camera, frame_offset, camera_follow_root, camera_use_rotation, camera_static, camera_smoothing, flip_x, solved_camera_rotations)
        
        # Parent camera to root locator if requested
        # This makes camera follow character movement while preserving screen-space relationship
        if camera_follow_root and root_locator and camera_obj:
            camera_obj.parent = root_locator
            print(f"[Blender] Camera parented to root_locator - follows character movement")
            if camera_static:
                print(f"[Blender] Camera is STATIC - body_offset positions character correctly")
            elif camera_use_rotation:
                print(f"[Blender] Camera uses PAN/TILT rotation to frame character (like real camera operator)")
            else:
                print(f"[Blender] Camera uses local TRANSLATION to frame character")
    
    # Export
    if output_format == "abc":
        if mesh_obj:
            print("[Blender] Baking shape keys to mesh cache for Alembic...")
            bpy.context.view_layer.objects.active = mesh_obj
            mesh_obj.select_set(True)
        export_alembic(output_path)
        
        # Also export FBX for joints/camera
        fbx_path = output_path.replace(".abc", "_skeleton.fbx")
        if mesh_obj:
            mesh_obj.hide_set(True)
        export_fbx(fbx_path, axis_forward, axis_up_export)
        print(f"[Blender] Also exported skeleton/camera to: {fbx_path}")
    else:
        export_fbx(output_path, axis_forward, axis_up_export)
    
    print("[Blender] Done!")


if __name__ == "__main__":
    main()
