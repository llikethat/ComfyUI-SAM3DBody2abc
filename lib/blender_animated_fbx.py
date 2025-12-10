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


def get_transform_for_axis(up_axis):
    """
    Get coordinate transformation based on desired up axis.
    SAM3DBody uses: X-right, Y-up, Z-forward (OpenGL convention)
    
    Returns: (flip_func, axis_forward, axis_up)
    """
    if up_axis == "Y":
        return lambda p: (-p[0], -p[1], -p[2]), '-Z', 'Y'
    elif up_axis == "Z":
        return lambda p: (-p[0], -p[2], p[1]), 'Y', 'Z'
    elif up_axis == "-Y":
        return lambda p: (-p[0], p[1], p[2]), 'Z', '-Y'
    elif up_axis == "-Z":
        return lambda p: (-p[0], p[2], -p[1]), '-Y', '-Z'
    else:
        return lambda p: (-p[0], -p[1], -p[2]), '-Z', 'Y'


def get_rotation_transform_matrix(up_axis):
    """
    Get rotation transformation matrix for converting MHR rotations to Blender.
    """
    if up_axis == "Y":
        # Flip X, Y, Z -> mirror across origin
        return Matrix((
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ))
    elif up_axis == "Z":
        # X stays, Y<->Z swap with sign changes
        return Matrix((
            (-1, 0, 0),
            (0, 0, 1),
            (0, -1, 0)
        ))
    elif up_axis == "-Y":
        return Matrix((
            (-1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)
        ))
    elif up_axis == "-Z":
        return Matrix((
            (-1, 0, 0),
            (0, 0, -1),
            (0, 1, 0)
        ))
    else:
        return Matrix((
            (-1, 0, 0),
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
    
    # Get transformation matrix
    T = get_rotation_transform_matrix(up_axis)
    
    # Transform: T * M * T^-1 (similarity transform)
    # This transforms the rotation from MHR coordinate system to Blender's
    T_inv = T.inverted()
    transformed = T @ m @ T_inv
    
    return transformed


def get_world_offset_from_cam_t(pred_cam_t, up_axis):
    """
    Convert pred_cam_t [tx, ty, tz] to world space offset.
    """
    if not pred_cam_t or len(pred_cam_t) < 3:
        return Vector((0, 0, 0))
    
    tx, ty, tz = pred_cam_t[0], pred_cam_t[1], pred_cam_t[2]
    
    # Scale tx, ty by depth to get world units
    world_x = tx * abs(tz) * 0.5
    world_y = ty * abs(tz) * 0.5
    
    # Apply based on up_axis
    if up_axis == "Y":
        return Vector((-world_x, -world_y, 0))
    elif up_axis == "Z":
        return Vector((-world_x, 0, world_y))
    elif up_axis == "-Y":
        return Vector((-world_x, world_y, 0))
    elif up_axis == "-Z":
        return Vector((-world_x, 0, -world_y))
    else:
        return Vector((-world_x, -world_y, 0))


def create_animated_mesh(all_frames, faces, fps, transform_func, world_translation_mode="none", up_axis="Y"):
    """
    Create mesh with per-vertex animation using shape keys.
    """
    first_verts = all_frames[0].get("vertices")
    if not first_verts:
        return None
    
    # Get first frame world offset for initial position
    first_offset = Vector((0, 0, 0))
    if world_translation_mode == "baked":
        first_cam_t = all_frames[0].get("pred_cam_t")
        first_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
    
    # Create mesh with first frame vertices
    mesh = bpy.data.meshes.new("body_mesh")
    verts = []
    for v in first_verts:
        pos = Vector(transform_func(v))
        if world_translation_mode == "baked":
            pos += first_offset
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
    
    # Set scene settings
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(all_frames) - 1
    
    # Add basis shape key
    basis = obj.shape_key_add(name="Basis", from_mix=False)
    
    print(f"[Blender] Creating {len(all_frames)} shape keys (translation={world_translation_mode})...")
    
    # Create shape keys for each frame
    for frame_idx, frame_data in enumerate(all_frames):
        frame_verts = frame_data.get("vertices")
        if not frame_verts:
            continue
        
        # Get world offset for this frame
        frame_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            frame_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        sk = obj.shape_key_add(name=f"frame_{frame_idx:04d}", from_mix=False)
        
        for j, v in enumerate(frame_verts):
            if j < len(sk.data):
                pos = Vector(transform_func(v))
                if world_translation_mode == "baked":
                    pos += frame_offset
                sk.data[j].co = pos
        
        # Keyframe shape key value
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=max(0, frame_idx - 1))
        
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=frame_idx)
        
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=min(len(all_frames) - 1, frame_idx + 1))
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Shape keys: {frame_idx + 1}/{len(all_frames)}")
    
    print(f"[Blender] Created mesh with {len(all_frames)} shape keys")
    return obj


def create_skeleton_with_rotations(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None):
    """
    Create animated skeleton using armature with ROTATION keyframes.
    
    This uses the true joint rotation matrices from MHR model.
    Produces proper bone rotations for retargeting and animation editing.
    """
    first_joints = all_frames[0].get("joint_coords")
    first_rotations = all_frames[0].get("joint_rotations")
    
    if not first_joints:
        print("[Blender] No joint_coords in first frame, skipping skeleton")
        return None
    
    if not first_rotations:
        print("[Blender] No joint_rotations available, falling back to position-based skeleton")
        return create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator)
    
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
    
    # Enter edit mode to create bones
    bpy.ops.object.mode_set(mode='EDIT')
    
    bones = []
    for i in range(num_joints):
        bone = arm_data.edit_bones.new(f"joint_{i:03d}")
        pos = first_joints[i]
        head_pos = Vector(transform_func(pos))
        
        if world_translation_mode == "baked":
            head_pos += first_offset
        
        bone.head = head_pos
        # Small tail offset pointing up in local space
        bone.tail = head_pos + Vector((0, 0.03, 0))
        bones.append(bone)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate bones in pose mode using ROTATION keyframes
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Set rotation mode to quaternion for smoother interpolation
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = 'QUATERNION'
    
    print(f"[Blender] Animating {num_joints} bones with rotations over {len(all_frames)} frames...")
    
    for frame_idx, frame_data in enumerate(all_frames):
        joints = frame_data.get("joint_coords")
        rotations = frame_data.get("joint_rotations")
        
        if not joints or not rotations:
            continue
        
        bpy.context.scene.frame_set(frame_idx)
        
        # Get world offset for this frame (for position updates)
        frame_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            frame_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        for bone_idx in range(min(num_joints, len(joints), len(rotations))):
            pose_bone = armature.pose.bones[bone_idx]
            
            # Get rotation matrix and transform to Blender space
            rot_3x3 = rotations[bone_idx]
            if rot_3x3 is None:
                continue
            
            # Transform rotation matrix from MHR to Blender coordinate system
            blender_rot = transform_rotation_matrix(rot_3x3, up_axis)
            
            # Convert to quaternion
            quat = blender_rot.to_quaternion()
            
            # Apply rotation
            pose_bone.rotation_quaternion = quat
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
            
            # Also update location for world translation modes
            if world_translation_mode == "baked":
                pos = joints[bone_idx]
                target = Vector(transform_func(pos)) + frame_offset
                rest_head = Vector(armature.data.bones[bone_idx].head_local)
                offset = target - rest_head
                pose_bone.location = offset
                pose_bone.keyframe_insert(data_path="location", frame=frame_idx)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Animated {frame_idx + 1}/{len(all_frames)} frames (rotations)")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[Blender] Created rotation-based skeleton with {num_joints} bones")
    return armature


def create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None):
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
    
    # Enter edit mode to create bones
    bpy.ops.object.mode_set(mode='EDIT')
    
    bones = []
    for i in range(num_joints):
        bone = arm_data.edit_bones.new(f"joint_{i:03d}")
        pos = first_joints[i]
        head_pos = Vector(transform_func(pos))
        
        if world_translation_mode == "baked":
            head_pos += first_offset
        
        bone.head = head_pos
        bone.tail = head_pos + Vector((0, 0.03, 0))
        bones.append(bone)
    
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
        
        bpy.context.scene.frame_set(frame_idx)
        
        # Get world offset for this frame
        frame_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            frame_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        for bone_idx in range(min(num_joints, len(joints))):
            pose_bone = armature.pose.bones[bone_idx]
            
            pos = joints[bone_idx]
            target = Vector(transform_func(pos))
            
            if world_translation_mode == "baked":
                target += frame_offset
            
            # Calculate offset from rest position
            offset = target - rest_heads[bone_idx]
            pose_bone.location = offset
            pose_bone.keyframe_insert(data_path="location", frame=frame_idx)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Animated {frame_idx + 1}/{len(all_frames)} frames (positions)")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[Blender] Created position-based skeleton with {num_joints} bones")
    return armature


def create_skeleton(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None, skeleton_mode="rotations"):
    """
    Create animated skeleton - dispatcher function.
    
    skeleton_mode:
    - "rotations": Use true joint rotation matrices from MHR (recommended)
    - "positions": Use joint positions only (legacy)
    """
    if skeleton_mode == "rotations":
        return create_skeleton_with_rotations(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator)
    else:
        return create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator)


def create_root_locator(all_frames, fps, up_axis):
    """
    Create a root locator that carries the world translation.
    """
    print("[Blender] Creating root locator with world translation...")
    
    root = bpy.data.objects.new("root_locator", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 0.1
    bpy.context.collection.objects.link(root)
    
    # Animate root position based on pred_cam_t
    for frame_idx, frame_data in enumerate(all_frames):
        frame_cam_t = frame_data.get("pred_cam_t")
        world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        root.location = world_offset
        root.keyframe_insert(data_path="location", frame=frame_idx)
    
    print(f"[Blender] Root locator animated over {len(all_frames)} frames")
    return root


def create_translation_track(all_frames, fps, up_axis):
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
        track.keyframe_insert(data_path="location", frame=frame_idx)
    
    print(f"[Blender] Translation track animated over {len(all_frames)} frames")
    return track


def create_camera(all_frames, fps, transform_func, up_axis, sensor_width=36.0, world_translation_mode="none"):
    """
    Create animated camera with focal length from SAM3DBody.
    """
    print("[Blender] Creating camera...")
    
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)
    
    # Set sensor width
    cam_data.sensor_width = sensor_width
    
    # Get focal length from first frame
    first_focal = all_frames[0].get("focal_length")
    if first_focal:
        if isinstance(first_focal, (list, tuple)):
            focal_px = first_focal[0]
        else:
            focal_px = first_focal
        
        image_width = 1920  # Assume HD
        focal_mm = focal_px * (sensor_width / image_width)
        cam_data.lens = focal_mm
        print(f"[Blender] Focal length: {focal_px}px -> {focal_mm:.1f}mm")
    
    # Create target for camera orientation
    target = bpy.data.objects.new("cam_target", None)
    target.location = (0, 0, 0)
    bpy.context.collection.objects.link(target)
    
    # Position camera based on up axis
    first_cam_t = all_frames[0].get("pred_cam_t")
    cam_distance = 3.0
    if first_cam_t and len(first_cam_t) > 2:
        cam_distance = abs(first_cam_t[2])
    
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
    
    # For "camera" mode: fixed rotation, animated position with offset
    if world_translation_mode == "camera":
        # Set fixed rotation pointing at origin
        camera.location = base_dir * cam_distance
        
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
        bpy.context.view_layer.update()
        camera.rotation_euler = camera.matrix_world.to_euler()
        camera.constraints.remove(constraint)
        
        camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
        camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
        camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
        
        # Animate camera position with negative world offset
        for frame_idx, frame_data in enumerate(all_frames):
            frame_cam_t = frame_data.get("pred_cam_t")
            
            frame_distance = cam_distance
            if frame_cam_t and len(frame_cam_t) > 2:
                frame_distance = abs(frame_cam_t[2])
            
            world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
            camera.location = base_dir * frame_distance - world_offset
            camera.keyframe_insert(data_path="location", frame=frame_idx)
        
        bpy.data.objects.remove(target)
        print(f"[Blender] Camera animated over {len(all_frames)} frames (fixed rotation, animated position)")
    else:
        # Static camera with distance animation
        camera.location = base_dir * cam_distance
        
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
        bpy.context.view_layer.update()
        camera.rotation_euler = camera.matrix_world.to_euler()
        camera.constraints.remove(constraint)
        
        bpy.data.objects.remove(target)
        
        camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
        camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
        camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
        
        print(f"[Blender] Camera at {camera.location}, rotation: {[math.degrees(r) for r in camera.rotation_euler]}")
        
        # Animate camera distance only
        for frame_idx, frame_data in enumerate(all_frames):
            frame_cam_t = frame_data.get("pred_cam_t")
            
            if frame_cam_t and len(frame_cam_t) > 2:
                new_distance = abs(frame_cam_t[2])
                camera.location = base_dir * new_distance
                camera.keyframe_insert(data_path="location", frame=frame_idx)
    
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
    sensor_width = data.get("sensor_width", 36.0)
    world_translation_mode = data.get("world_translation_mode", "none")
    skeleton_mode = data.get("skeleton_mode", "rotations")  # New: default to rotations
    
    print(f"[Blender] {len(frames)} frames at {fps} fps")
    print(f"[Blender] Sensor width: {sensor_width}mm")
    print(f"[Blender] World translation mode: {world_translation_mode}")
    print(f"[Blender] Skeleton mode: {skeleton_mode}")
    
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
    transform_func, axis_forward, axis_up_export = get_transform_for_axis(up_axis)
    
    clear_scene()
    
    # Set scene frame range
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames) - 1
    
    # Create root locator if needed (for "root" mode)
    root_locator = None
    if world_translation_mode == "root":
        root_locator = create_root_locator(frames, fps, up_axis)
    
    # Create mesh with shape keys
    mesh_obj = None
    if include_mesh:
        mesh_obj = create_animated_mesh(frames, faces, fps, transform_func, world_translation_mode, up_axis)
        # Parent mesh to root locator if in "root" mode
        if world_translation_mode == "root" and root_locator and mesh_obj:
            mesh_obj.parent = root_locator
    
    # Create skeleton (armature with bones)
    create_skeleton(frames, fps, transform_func, world_translation_mode, up_axis, root_locator, skeleton_mode)
    
    # Create separate translation track if in "separate" mode
    if world_translation_mode == "separate":
        create_translation_track(frames, fps, up_axis)
    
    # Create camera
    if include_camera:
        create_camera(frames, fps, transform_func, up_axis, sensor_width, world_translation_mode)
    
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
