# Copyright (c) 2025 - SAM3DBody2abc
# SPDX-License-Identifier: MIT
"""
Overlay renderer for SAM3DBody2abc.
Uses Meta's exact rendering approach.
"""

import os
import sys

# Try osmesa for headless rendering
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any


def render_with_pyrender(
    img_bgr: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
) -> Optional[np.ndarray]:
    """Render using pyrender with Meta's exact approach."""
    try:
        import trimesh
        import pyrender
    except ImportError as e:
        print(f"[SAM3DBody2abc] pyrender/trimesh not available: {e}")
        return None
    
    try:
        h, w = img_bgr.shape[:2]
        
        # Create mesh and apply Meta's 180° X rotation
        mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        
        # Material - Meta's light blue
        LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(LIGHT_BLUE[2], LIGHT_BLUE[1], LIGHT_BLUE[0], 1.0),
        )
        
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        # Scene
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh_pyrender, "mesh")
        
        # Camera (flip X like Meta does)
        camera_translation = cam_t.copy()
        camera_translation[0] *= -1.0
        
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=w / 2.0, cy=h / 2.0,
            zfar=1e12
        )
        scene.add(camera, pose=camera_pose)
        
        # Raymond lights (from Meta)
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)
            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
            scene.add(light, pose=matrix)
        
        # Render
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        
        # Composite
        color = color.astype(np.float32) / 255.0
        image = img_bgr.astype(np.float32) / 255.0
        
        valid_mask = color[:, :, 3:4]
        output = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        output = (output * 255).astype(np.uint8)
        
        return output
        
    except Exception as e:
        print(f"[SAM3DBody2abc] Pyrender error: {e}")
        return None


def render_wireframe_opencv(
    img_bgr: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    color: Tuple[int, int, int] = (166, 189, 219),  # Light blue BGR
    line_thickness: int = 1,
    debug: bool = False,
) -> np.ndarray:
    """
    Wireframe render using OpenCV, matching Meta's camera model.
    
    Key insight: The focal length from SAM3DBody is computed for the
    MODEL's processing size, not the display image size. We need to
    scale it appropriately.
    """
    import trimesh
    
    h, w = img_bgr.shape[:2]
    
    # Create mesh and apply 180° X rotation (same as Meta)
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    verts = np.array(mesh.vertices)
    
    # Camera translation (flip X like Meta)
    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0
    
    # Transform to camera space
    verts_cam = verts - camera_translation
    
    # Depth is -Z (camera looks down -Z in OpenGL)
    depth = -verts_cam[:, 2]
    valid = depth > 0.01
    depth_safe = np.maximum(depth, 0.01)
    
    # Project using OpenGL/pyrender convention
    # x_screen = fx * X / depth + cx
    # y_screen = (h - 1) - (fy * Y / depth + cy)  # Flip Y for image coords
    cx, cy = w / 2.0, h / 2.0
    
    pts_2d = np.zeros((len(verts), 2))
    pts_2d[:, 0] = verts_cam[:, 0] * focal_length / depth_safe + cx
    # Y is flipped in OpenGL viewport transform
    pts_2d[:, 1] = (h - 1) - (verts_cam[:, 1] * focal_length / depth_safe + cy)
    
    if debug:
        print(f"  [Wireframe] Rotated mesh Y range: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}]")
        print(f"  [Wireframe] Camera translation: {camera_translation}")
        print(f"  [Wireframe] Camera-space Y range: [{verts_cam[:, 1].min():.3f}, {verts_cam[:, 1].max():.3f}]")
        print(f"  [Wireframe] Depth range: [{depth[valid].min():.3f}, {depth[valid].max():.3f}]")
        print(f"  [Wireframe] Focal length: {focal_length:.1f}")
        print(f"  [Wireframe] Image size: {w}x{h}")
        print(f"  [Wireframe] 2D X range: [{pts_2d[valid, 0].min():.1f}, {pts_2d[valid, 0].max():.1f}]")
        print(f"  [Wireframe] 2D Y range: [{pts_2d[valid, 1].min():.1f}, {pts_2d[valid, 1].max():.1f}]")
    
    # Draw wireframe with depth sorting
    result = img_bgr.copy()
    pts_2d_int = pts_2d.astype(np.int32)
    
    # Calculate face depths for sorting (draw far faces first)
    face_depths = []
    for face in faces:
        if valid[face[0]] and valid[face[1]] and valid[face[2]]:
            avg_depth = (depth[face[0]] + depth[face[1]] + depth[face[2]]) / 3
            face_depths.append((avg_depth, face))
    
    # Sort by depth (far to near)
    face_depths.sort(key=lambda x: -x[0])
    
    # Draw edges
    edges_drawn = set()
    in_frame_count = 0
    for _, face in face_depths:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            if edge in edges_drawn:
                continue
            edges_drawn.add(edge)
            
            p1 = tuple(pts_2d_int[v1])
            p2 = tuple(pts_2d_int[v2])
            
            # Check if line is at least partially in frame
            if ((0 <= p1[0] < w or 0 <= p2[0] < w) and
                (0 <= p1[1] < h or 0 <= p2[1] < h)):
                cv2.line(result, p1, p2, color, line_thickness, cv2.LINE_AA)
                in_frame_count += 1
    
    if debug:
        print(f"  [Wireframe] Drew {len(edges_drawn)} edges ({in_frame_count} in frame)")
    
    return result


def render_filled_opencv(
    img_bgr: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    color: Tuple[int, int, int] = (166, 189, 219),  # Light blue BGR
    opacity: float = 0.5,
    debug: bool = False,
) -> np.ndarray:
    """
    Filled mesh render using OpenCV (z-buffered triangles).
    Slower than wireframe but looks better when pyrender unavailable.
    """
    import trimesh
    
    h, w = img_bgr.shape[:2]
    
    # Create mesh and apply 180° X rotation
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    verts = np.array(mesh.vertices)
    
    # Camera translation (flip X)
    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0
    
    # Transform to camera space
    verts_cam = verts - camera_translation
    
    # Project
    depth = -verts_cam[:, 2]
    valid = depth > 0.01
    depth_safe = np.maximum(depth, 0.01)
    
    cx, cy = w / 2.0, h / 2.0
    pts_2d = np.zeros((len(verts), 2))
    pts_2d[:, 0] = verts_cam[:, 0] * focal_length / depth_safe + cx
    pts_2d[:, 1] = (h - 1) - (verts_cam[:, 1] * focal_length / depth_safe + cy)
    
    # Create overlay
    overlay = img_bgr.copy()
    pts_2d_int = pts_2d.astype(np.int32)
    
    # Calculate face depths and sort
    face_data = []
    for face in faces:
        if valid[face[0]] and valid[face[1]] and valid[face[2]]:
            avg_depth = (depth[face[0]] + depth[face[1]] + depth[face[2]]) / 3
            pts = pts_2d_int[face]
            face_data.append((avg_depth, pts))
    
    # Sort far to near
    face_data.sort(key=lambda x: -x[0])
    
    # Draw filled triangles
    for _, pts in face_data:
        cv2.fillPoly(overlay, [pts], color)
    
    # Blend with original
    result = cv2.addWeighted(img_bgr, 1 - opacity, overlay, opacity, 0)
    
    if debug:
        print(f"  [Filled] Drew {len(face_data)} triangles")
    
    return result


class SAM3DBody2abcOverlay:
    """Render mesh overlay on single image."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("MESH_DATA", {"tooltip": "Single mesh from SAM3DBody"}),
                "image": ("IMAGE", {"tooltip": "Image to overlay mesh on"}),
            },
            "optional": {
                "render_mode": (["solid", "filled", "wireframe"], {
                    "default": "solid",
                    "tooltip": "solid=pyrender (best), filled=OpenCV triangles, wireframe=OpenCV lines"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_image",)
    FUNCTION = "render_overlay"
    CATEGORY = "SAM3DBody2abc/Visualization"
    
    def render_overlay(
        self,
        mesh_data: Dict,
        image: torch.Tensor,
        render_mode: str = "solid",
    ):
        # Convert image to BGR
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = img_np[..., ::-1].copy()
        
        # Extract mesh data
        vertices = mesh_data.get("vertices") or mesh_data.get("verts") or mesh_data.get("pred_vertices")
        faces = mesh_data.get("faces")
        cam_t = mesh_data.get("camera")
        if cam_t is None:
            cam_t = mesh_data.get("pred_cam_t")
        focal_length = mesh_data.get("focal_length")
        
        if vertices is None or faces is None:
            print("[SAM3DBody2abc] No mesh data")
            return (image,)
        
        # Convert to numpy
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if cam_t is not None and isinstance(cam_t, torch.Tensor):
            cam_t = cam_t.cpu().numpy()
        
        vertices = np.array(vertices).astype(np.float64)
        faces = np.array(faces).astype(np.int32)
        
        if cam_t is None:
            cam_t = np.array([0.0, 0.3, 2.5])
        else:
            cam_t = np.array(cam_t).flatten().astype(np.float64)
        
        h, w = img_bgr.shape[:2]
        if focal_length is None:
            focal_length = max(h, w)
        else:
            if isinstance(focal_length, torch.Tensor):
                focal_length = focal_length.cpu().numpy()
            focal_length = float(np.array(focal_length).flatten()[0])
        
        print(f"[SAM3DBody2abc] Rendering: {vertices.shape[0]} verts, cam_t={cam_t}, focal={focal_length:.1f}")
        
        # Render
        if render_mode == "solid":
            result = render_with_pyrender(img_bgr, vertices, faces, cam_t, focal_length)
            if result is None:
                print("[SAM3DBody2abc] Pyrender failed, using filled")
                result = render_filled_opencv(img_bgr, vertices, faces, cam_t, focal_length, debug=True)
        elif render_mode == "filled":
            result = render_filled_opencv(img_bgr, vertices, faces, cam_t, focal_length, debug=True)
        else:
            result = render_wireframe_opencv(img_bgr, vertices, faces, cam_t, focal_length, debug=True)
        
        # Convert back to RGB tensor
        result_rgb = result[..., ::-1].copy()
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)


class SAM3DBody2abcOverlayBatch:
    """Render mesh overlay on batch of images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {"tooltip": "Sequence of mesh data"}),
                "images": ("IMAGE", {"tooltip": "Batch of images"}),
            },
            "optional": {
                "render_mode": (["solid", "filled", "wireframe"], {
                    "default": "solid",
                    "tooltip": "solid=pyrender (best), filled=OpenCV triangles, wireframe=OpenCV lines"
                }),
                "temporal_smoothing": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_images",)
    FUNCTION = "render_batch"
    CATEGORY = "SAM3DBody2abc/Visualization"
    
    def render_batch(
        self,
        mesh_sequence: List[Dict],
        images: torch.Tensor,
        render_mode: str = "solid",
        temporal_smoothing: float = 0.5,
    ):
        import comfy.utils
        
        print(f"[SAM3DBody2abc] Batch overlay: {len(mesh_sequence)} meshes, {images.shape[0]} images")
        print(f"[SAM3DBody2abc] Render mode: {render_mode}")
        
        # Test pyrender
        pyrender_works = False
        if render_mode == "solid":
            try:
                import pyrender
                test_renderer = pyrender.OffscreenRenderer(64, 64)
                test_renderer.delete()
                pyrender_works = True
                print(f"[SAM3DBody2abc] Pyrender OK")
            except Exception as e:
                print(f"[SAM3DBody2abc] Pyrender failed: {e}")
                print("[SAM3DBody2abc] Using filled fallback")
        
        # Apply temporal smoothing
        if temporal_smoothing > 0 and len(mesh_sequence) > 1:
            from scipy.ndimage import gaussian_filter1d
            
            valid_verts = []
            valid_cams = []
            valid_indices = []
            
            for i, mesh in enumerate(mesh_sequence):
                verts = mesh.get("vertices")
                cam = mesh.get("camera")
                if cam is None:
                    cam = mesh.get("pred_cam_t")
                if verts is not None and mesh.get("valid", True):
                    if isinstance(verts, torch.Tensor):
                        verts = verts.cpu().numpy()
                    if cam is not None and isinstance(cam, torch.Tensor):
                        cam = cam.cpu().numpy()
                    valid_verts.append(np.array(verts))
                    valid_cams.append(np.array(cam).flatten() if cam is not None else np.array([0, 0.3, 2.5]))
                    valid_indices.append(i)
            
            if len(valid_verts) > 2:
                valid_verts = np.array(valid_verts)
                valid_cams = np.array(valid_cams)
                
                sigma = 0.5 + temporal_smoothing * 2.5
                smoothed_verts = gaussian_filter1d(valid_verts, sigma=sigma, axis=0, mode='nearest')
                smoothed_cams = gaussian_filter1d(valid_cams, sigma=sigma, axis=0, mode='nearest')
                
                for idx, orig_idx in enumerate(valid_indices):
                    mesh_sequence[orig_idx]["vertices"] = smoothed_verts[idx]
                    mesh_sequence[orig_idx]["camera"] = smoothed_cams[idx]
                
                print(f"[SAM3DBody2abc] Smoothing applied (sigma={sigma:.2f})")
        
        # Process frames
        result_frames = []
        pbar = comfy.utils.ProgressBar(len(mesh_sequence))
        
        for i, mesh in enumerate(mesh_sequence):
            img_idx = min(i, images.shape[0] - 1)
            img_np = images[img_idx].cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = img_np[..., ::-1].copy()
            
            vertices = mesh.get("vertices")
            faces = mesh.get("faces")
            
            if vertices is None or faces is None or not mesh.get("valid", True):
                result_frames.append(img_np)
                pbar.update(1)
                continue
            
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()
            
            vertices = np.array(vertices).astype(np.float64)
            faces = np.array(faces).astype(np.int32)
            
            cam_t = mesh.get("camera")
            if cam_t is None:
                cam_t = mesh.get("pred_cam_t")
            if cam_t is not None:
                if isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                cam_t = np.array(cam_t).flatten().astype(np.float64)
            else:
                cam_t = np.array([0.0, 0.3, 2.5])
            
            focal_length = mesh.get("focal_length")
            h, w = img_bgr.shape[:2]
            if focal_length is None:
                focal_length = max(h, w)
            else:
                if isinstance(focal_length, torch.Tensor):
                    focal_length = focal_length.cpu().numpy()
                focal_length = float(np.array(focal_length).flatten()[0])
            
            # Debug first frame
            if i == 0:
                print(f"[SAM3DBody2abc] Frame 0: verts={vertices.shape}, cam_t={cam_t}, focal={focal_length:.1f}")
            
            # Render
            if render_mode == "solid" and pyrender_works:
                result = render_with_pyrender(img_bgr, vertices, faces, cam_t, focal_length)
                if result is None:
                    result = render_filled_opencv(img_bgr, vertices, faces, cam_t, focal_length, debug=(i==0))
            elif render_mode == "filled" or (render_mode == "solid" and not pyrender_works):
                result = render_filled_opencv(img_bgr, vertices, faces, cam_t, focal_length, debug=(i==0))
            else:
                result = render_wireframe_opencv(img_bgr, vertices, faces, cam_t, focal_length, debug=(i==0))
            
            result_rgb = result[..., ::-1].copy()
            result_frames.append(result_rgb)
            pbar.update(1)
        
        result_tensor = torch.from_numpy(np.stack(result_frames).astype(np.float32) / 255.0)
        print(f"[SAM3DBody2abc] Rendered {len(result_frames)} frames")
        
        return (result_tensor,)
