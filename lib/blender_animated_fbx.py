"""
Blender Script: Export Animated FBX from Mesh Sequence JSON
Creates animated mesh using vertex keyframes + joint locators.

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
from mathutils import Vector, Matrix
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


def create_animated_mesh(all_frames, faces, fps, transform_func):
    """
    Create mesh with per-vertex animation using shape keys.
    Only the final animated mesh is exported (no per-frame hidden meshes).
    """
    first_verts = all_frames[0].get("vertices")
    if not first_verts:
        return None
    
    # Create mesh with first frame vertices
    mesh = bpy.data.meshes.new("body_mesh")
    verts = [transform_func(v) for v in first_verts]
    
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
    
    print(f"[Blender] Creating {len(all_frames)} shape keys...")
    
    # Create shape keys for each frame
    for frame_idx, frame_data in enumerate(all_frames):
        frame_verts = frame_data.get("vertices")
        if not frame_verts:
            continue
        
        sk = obj.shape_key_add(name=f"frame_{frame_idx:04d}", from_mix=False)
        
        for j, v in enumerate(frame_verts):
            if j < len(sk.data):
                sk.data[j].co = Vector(transform_func(v))
        
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


def create_joint_locators(all_frames, fps, transform_func):
    """
    Create animated joint locators (empties).
    Each locator animates independently in world space - no parent hierarchy.
    """
    first_joints = all_frames[0].get("joint_coords")
    if not first_joints:
        return None
    
    num_joints = len(first_joints)
    print(f"[Blender] Creating {num_joints} joint locators...")
    
    # Create empties for each joint - NO PARENT to avoid transform issues
    empties = []
    for i in range(num_joints):
        empty = bpy.data.objects.new(f"joint_{i:03d}", None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.02
        bpy.context.collection.objects.link(empty)
        empties.append(empty)
    
    # Animate each joint
    print(f"[Blender] Animating {num_joints} joints over {len(all_frames)} frames...")
    
    for frame_idx, frame_data in enumerate(all_frames):
        joints = frame_data.get("joint_coords")
        if not joints:
            continue
        
        for joint_idx in range(min(num_joints, len(joints))):
            pos = joints[joint_idx]
            world_pos = Vector(transform_func(pos))
            
            # Set location directly in world space
            empties[joint_idx].location = world_pos
            empties[joint_idx].keyframe_insert(data_path="location", frame=frame_idx)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"[Blender] Animated {frame_idx + 1}/{len(all_frames)} frames")
    
    print(f"[Blender] Created {num_joints} animated joint locators")
    return empties


def create_camera(all_frames, fps, transform_func, up_axis, sensor_width):
    """
    Create camera with no rotation (0,0,0) by using track-to constraint.
    """
    first_focal = all_frames[0].get("focal_length")
    first_cam_t = all_frames[0].get("pred_cam_t")
    
    if not first_focal:
        return None
    
    # Create camera
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)
    
    # Set sensor width
    cam_data.sensor_width = sensor_width
    
    # Convert focal length from pixels to mm
    image_width = 1920.0
    focal_mm = (first_focal * sensor_width) / image_width
    cam_data.lens = max(1.0, min(focal_mm, 500.0))
    print(f"[Blender] Camera: {first_focal:.1f}px -> {cam_data.lens:.1f}mm (sensor={sensor_width}mm)")
    
    # Get camera distance
    cam_distance = 3.0
    if first_cam_t and len(first_cam_t) > 2:
        cam_distance = abs(first_cam_t[2])
    
    # Create target at origin for camera to look at
    target = bpy.data.objects.new("CameraTarget", None)
    target.empty_display_type = 'PLAIN_AXES'
    target.empty_display_size = 0.1
    target.location = Vector((0, 0, 0))
    bpy.context.collection.objects.link(target)
    
    # Position camera based on up_axis
    if up_axis == "Y":
        camera.location = Vector((0, 0, cam_distance))
    elif up_axis == "Z":
        camera.location = Vector((0, -cam_distance, 0))
    elif up_axis == "-Y":
        camera.location = Vector((0, 0, -cam_distance))
    elif up_axis == "-Z":
        camera.location = Vector((0, cam_distance, 0))
    
    # Add track-to constraint so camera always looks at target
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    # Apply constraint to get final rotation, then remove constraint
    bpy.context.view_layer.update()
    camera.rotation_euler = camera.matrix_world.to_euler()
    camera.constraints.remove(constraint)
    
    # Delete target
    bpy.data.objects.remove(target)
    
    # Round rotations to clean values
    camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
    camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
    camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
    
    print(f"[Blender] Camera at {camera.location}, rotation: {[math.degrees(r) for r in camera.rotation_euler]}")
    
    # Animate camera distance
    for frame_idx, frame_data in enumerate(all_frames):
        frame_cam_t = frame_data.get("pred_cam_t")
        
        if frame_cam_t and len(frame_cam_t) > 2:
            new_distance = abs(frame_cam_t[2])
            
            if up_axis == "Y":
                camera.location.z = new_distance
            elif up_axis == "Z":
                camera.location.y = -new_distance
            elif up_axis == "-Y":
                camera.location.z = -new_distance
            elif up_axis == "-Z":
                camera.location.y = new_distance
            
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
        object_types={'MESH', 'EMPTY', 'CAMERA'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
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
    """
    Export to Alembic (.abc) - better for vertex animation.
    Alembic stores per-frame vertex positions directly, no blend shapes.
    """
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
    
    print(f"[Blender] {len(frames)} frames at {fps} fps")
    print(f"[Blender] Sensor width: {sensor_width}mm")
    
    if not frames:
        print("[Blender] Error: No frames")
        sys.exit(1)
    
    # Get transformation
    transform_func, axis_forward, axis_up_export = get_transform_for_axis(up_axis)
    
    clear_scene()
    
    # Set scene frame range
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(frames) - 1
    
    # Create mesh with shape keys
    mesh_obj = None
    if include_mesh:
        mesh_obj = create_animated_mesh(frames, faces, fps, transform_func)
    
    # Create joint locators
    create_joint_locators(frames, fps, transform_func)
    
    # Create camera
    if include_camera:
        create_camera(frames, fps, transform_func, up_axis, sensor_width)
    
    # Export
    if output_format == "abc":
        # For Alembic, we need to bake the shape key animation to actual vertex positions
        if mesh_obj:
            print("[Blender] Baking shape keys to mesh cache for Alembic...")
            # Select mesh and bake
            bpy.context.view_layer.objects.active = mesh_obj
            mesh_obj.select_set(True)
        export_alembic(output_path)
        
        # Also export FBX for joints/camera (Alembic doesn't support empties well)
        fbx_path = output_path.replace(".abc", "_skeleton.fbx")
        if mesh_obj:
            mesh_obj.hide_set(True)  # Hide mesh for skeleton-only FBX
        export_fbx(fbx_path, axis_forward, axis_up_export)
        print(f"[Blender] Also exported skeleton/camera to: {fbx_path}")
    else:
        export_fbx(output_path, axis_forward, axis_up_export)
    
    print("[Blender] Done!")


if __name__ == "__main__":
    main()
