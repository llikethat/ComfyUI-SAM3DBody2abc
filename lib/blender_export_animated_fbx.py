"""
Blender script to export SAM3D Body ANIMATED mesh and skeleton to FBX file.
Creates keyframes for each frame in the sequence.

Usage: blender --background --python blender_export_animated_fbx.py -- <animation_json> <output_fbx>

The animation_json contains:
{
    "fps": 30,
    "frames": [
        {
            "frame_index": 0,
            "vertices": [[x,y,z], ...],
            "joint_positions": [[x,y,z], ...],
        },
        ...
    ],
    "faces": [[v1,v2,v3], ...],
    "joint_parents": [parent_idx, ...],
    "skinning_weights": [[[bone_idx, weight], ...], ...],
    "num_joints": 127
}
"""

import bpy
import sys
import os
import json
import numpy as np
from mathutils import Vector, Matrix

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_export_animated_fbx.py -- <animation_json> <output_fbx>")
    sys.exit(1)

animation_json_path = argv[0]
output_fbx = argv[1]

# Load animation data
print(f"[AnimatedFBX] Loading animation data from {animation_json_path}")
with open(animation_json_path, 'r') as f:
    anim_data = json.load(f)

fps = anim_data.get('fps', 30)
frames = anim_data.get('frames', [])
faces = anim_data.get('faces', [])
joint_parents = anim_data.get('joint_parents', [])
skinning_weights = anim_data.get('skinning_weights', [])
num_joints = anim_data.get('num_joints', 127)

if not frames:
    print("[AnimatedFBX] ERROR: No frames in animation data")
    sys.exit(1)

print(f"[AnimatedFBX] Animation: {len(frames)} frames at {fps} fps")
print(f"[AnimatedFBX] Skeleton: {num_joints} joints")

# Clean default scene
def clean_bpy():
    """Remove all default Blender objects"""
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)

clean_bpy()

# Set scene FPS
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = len(frames)

# Create collection
collection = bpy.data.collections.new('SAM3D_Animated')
bpy.context.scene.collection.children.link(collection)

# Get first frame data for initial mesh creation
first_frame = frames[0]
vertices_first = np.array(first_frame['vertices'], dtype=np.float32)
num_vertices = len(vertices_first)

# Apply coordinate transform (flip for Blender)
def transform_coords(coords):
    """Transform coordinates from SAM3D to Blender space"""
    coords = np.array(coords, dtype=np.float32)
    transformed = np.zeros_like(coords)
    transformed[:, 0] = -coords[:, 0]
    transformed[:, 1] = -coords[:, 2]
    transformed[:, 2] = coords[:, 1]
    return transformed

vertices_transformed = transform_coords(vertices_first)

# Create mesh
mesh = bpy.data.meshes.new('SAM3D_Body_Mesh')
mesh_obj = bpy.data.objects.new('SAM3D_Body', mesh)
collection.objects.link(mesh_obj)

# Build mesh from vertices and faces
mesh.from_pydata(vertices_transformed.tolist(), [], faces)
mesh.update()

print(f"[AnimatedFBX] Created mesh with {num_vertices} vertices, {len(faces)} faces")

# Create shape keys for mesh animation
mesh_obj.shape_key_add(name='Basis', from_mix=False)

# Add shape key for each frame
print("[AnimatedFBX] Creating shape keys for mesh animation...")
for frame_idx, frame_data in enumerate(frames):
    if frame_idx == 0:
        continue  # Skip first frame (it's the basis)
    
    frame_verts = transform_coords(frame_data['vertices'])
    
    # Create shape key
    sk = mesh_obj.shape_key_add(name=f'Frame_{frame_idx:04d}', from_mix=False)
    
    # Set shape key vertex positions
    for i, vert in enumerate(frame_verts):
        sk.data[i].co = Vector(vert)

# Animate shape keys
print("[AnimatedFBX] Keyframing shape key animation...")
mesh_obj.data.shape_keys.use_relative = True

for frame_idx in range(len(frames)):
    bpy.context.scene.frame_set(frame_idx + 1)
    
    # Set all shape keys to 0 first
    for sk in mesh_obj.data.shape_keys.key_blocks[1:]:  # Skip Basis
        sk.value = 0.0
        sk.keyframe_insert(data_path='value', frame=frame_idx + 1)
    
    # Set current frame's shape key to 1
    if frame_idx > 0:
        current_sk = mesh_obj.data.shape_keys.key_blocks[frame_idx]  # frame_idx because [0] is Basis
        current_sk.value = 1.0
        current_sk.keyframe_insert(data_path='value', frame=frame_idx + 1)

# Create armature if we have joint data
first_joints = first_frame.get('joint_positions')
if first_joints and len(first_joints) > 0:
    print(f"[AnimatedFBX] Creating armature with {len(first_joints)} joints...")
    
    joints_first = transform_coords(first_joints)
    
    # Calculate skeleton center
    skeleton_center = joints_first.mean(axis=0)
    rel_joints = joints_first - skeleton_center
    
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True, location=skeleton_center)
    armature = bpy.data.armatures.get('Armature')
    armature.name = 'SAM3D_Skeleton'
    armature_obj = bpy.context.active_object
    armature_obj.name = 'SAM3D_Skeleton'
    
    # Move to our collection
    if armature_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(armature_obj)
    collection.objects.link(armature_obj)
    
    edit_bones = armature.edit_bones
    extrude_size = 0.03
    
    # Remove default bone
    default_bone = edit_bones.get('Bone')
    if default_bone:
        edit_bones.remove(default_bone)
    
    # Create bones
    bones_dict = {}
    actual_joints = min(num_joints, len(rel_joints))
    
    for i in range(actual_joints):
        bone_name = f'Joint_{i:03d}'
        bone = edit_bones.new(bone_name)
        bone.head = Vector(rel_joints[i])
        bone.tail = Vector(rel_joints[i]) + Vector((0, 0, extrude_size))
        bones_dict[bone_name] = bone
    
    # Set up hierarchy
    if joint_parents and len(joint_parents) >= actual_joints:
        for i in range(actual_joints):
            parent_idx = joint_parents[i]
            if 0 <= parent_idx < actual_joints and parent_idx != i:
                bone_name = f'Joint_{i:03d}'
                parent_name = f'Joint_{parent_idx:03d}'
                if parent_name in bones_dict:
                    bones_dict[bone_name].parent = bones_dict[parent_name]
                    bones_dict[bone_name].use_connect = False
    else:
        # Flat hierarchy
        for i in range(1, actual_joints):
            bones_dict[f'Joint_{i:03d}'].parent = bones_dict['Joint_000']
            bones_dict[f'Joint_{i:03d}'].use_connect = False
    
    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Apply skinning weights
    if skinning_weights and len(skinning_weights) > 0:
        print("[AnimatedFBX] Applying skinning weights...")
        
        # Create vertex groups
        for i in range(actual_joints):
            mesh_obj.vertex_groups.new(name=f'Joint_{i:03d}')
        
        # Assign weights
        for vert_idx, influences in enumerate(skinning_weights):
            if vert_idx >= num_vertices:
                break
            for bone_idx, weight in influences:
                if 0 <= bone_idx < actual_joints and weight > 0.0001:
                    vg = mesh_obj.vertex_groups.get(f'Joint_{bone_idx:03d}')
                    if vg:
                        vg.add([vert_idx], weight, 'REPLACE')
        
        # Parent mesh to armature with automatic weights
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        mesh_obj.select_set(True)
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.parent_set(type='ARMATURE')
    
    # Animate skeleton
    print("[AnimatedFBX] Keyframing skeleton animation...")
    bpy.ops.object.mode_set(mode='POSE')
    
    for frame_idx, frame_data in enumerate(frames):
        bpy.context.scene.frame_set(frame_idx + 1)
        
        frame_joints = frame_data.get('joint_positions')
        if not frame_joints:
            continue
        
        frame_joints = transform_coords(frame_joints)
        frame_center = frame_joints.mean(axis=0)
        
        # Update armature location
        armature_obj.location = Vector(frame_center)
        armature_obj.keyframe_insert(data_path='location', frame=frame_idx + 1)
        
        # Update bone positions (using pose bones)
        rel_frame_joints = frame_joints - frame_center
        
        for i in range(min(actual_joints, len(frame_joints))):
            bone_name = f'Joint_{i:03d}'
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if pose_bone:
                # Calculate offset from rest position
                rest_pos = Vector(rel_joints[i])
                current_pos = Vector(rel_frame_joints[i])
                offset = current_pos - rest_pos
                
                pose_bone.location = offset
                pose_bone.keyframe_insert(data_path='location', frame=frame_idx + 1)
    
    bpy.ops.object.mode_set(mode='OBJECT')

# Select all for export
for obj in bpy.context.selected_objects:
    obj.select_set(False)
for obj in collection.objects:
    obj.select_set(True)

# Export FBX
print(f"[AnimatedFBX] Exporting to {output_fbx}")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

bpy.ops.export_scene.fbx(
    filepath=output_fbx,
    check_existing=False,
    use_selection=True,
    add_leaf_bones=False,
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=True,
    bake_anim_step=1.0,
    bake_anim_simplify_factor=0.0,
    path_mode='COPY',
    embed_textures=True,
    use_mesh_modifiers=True,
    mesh_smooth_type='OFF',
)

print(f"[AnimatedFBX] SUCCESS: Exported {len(frames)} frames to {output_fbx}")
