# Copyright (c) 2025 - SAM3DBody2abc
# SPDX-License-Identifier: MIT
"""
Overlay renderer for SAM3DBody2abc.
Uses ComfyUI-SAM3DBody's renderer or pyrender for accurate overlay.
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any


def get_sam3dbody_renderer():
    """Try to get the renderer from ComfyUI-SAM3DBody."""
    import sys
    import os
    
    # Search paths for ComfyUI-SAM3DBody
    search_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI-SAM3DBody"),
        os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI-SAM3DBody", "sam_3d_body"),
    ]
    
    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)
    
    # Try to import the renderer
    try:
        from sam_3d_body.visualization.renderer import Renderer
        return Renderer
    except ImportError:
        pass
    
    try:
        from visualization.renderer import Renderer
        return Renderer
    except ImportError:
        pass
    
    return None


def render_with_pyrender(
    img_bgr: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
) -> np.ndarray:
    """Render mesh using pyrender (Meta's approach)."""
    try:
        import pyrender
        import trimesh
    except ImportError:
        return None
    
    h, w = img_bgr.shape[:2]
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh_color = np.array([0.7, 0.7, 0.9, 0.6])  # Light blue with transparency
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=mesh_color
    )
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    # Create scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh_pyrender)
    
    # Camera setup (matching SAM3DBody's approach)
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=w / 2, cy=h / 2,
        znear=0.01, zfar=100.0
    )
    
    # Camera pose - SAM3DBody uses specific transform
    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0  # Flip X
    
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation
    # Rotate 180 degrees around X axis
    camera_pose[1, 1] = -1.0
    camera_pose[2, 2] = -1.0
    
    scene.add(camera, pose=camera_pose)
    
    # Light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    # Render
    try:
        renderer = pyrender.OffscreenRenderer(w, h)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
    except Exception as e:
        print(f"[SAM3DBody2abc] Pyrender failed: {e}")
        return None
    
    # Composite onto image
    color_rgb = color[:, :, :3]
    alpha = color[:, :, 3:4] / 255.0
    
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32)
    result = img_rgb * (1 - alpha) + color_rgb.astype(np.float32) * alpha
    result_bgr = result[:, :, ::-1].astype(np.uint8)
    
    return result_bgr


def render_wireframe_opencv(
    img_bgr: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    color: Tuple[int, int, int] = (100, 255, 100),
    line_thickness: int = 1,
) -> np.ndarray:
    """Simple wireframe render using OpenCV (fallback)."""
    h, w = img_bgr.shape[:2]
    
    # Project vertices to 2D using SAM3DBody's camera model
    verts = vertices.copy()
    
    # Apply 180° rotation around X (flip Y and Z)
    verts[:, 1] = -verts[:, 1]
    verts[:, 2] = -verts[:, 2]
    
    # Camera translation (flip X)
    camera_trans = np.array([-cam_t[0], cam_t[1], cam_t[2]])
    verts_cam = verts - camera_trans
    
    # Project
    z = -verts_cam[:, 2]  # Negate for OpenGL convention
    valid = z > 0.01
    z_safe = np.maximum(z, 0.01)
    
    cx, cy = w / 2.0, h / 2.0
    pts_2d = np.zeros((len(verts), 2))
    pts_2d[:, 0] = verts_cam[:, 0] * focal_length / z_safe + cx
    pts_2d[:, 1] = -verts_cam[:, 1] * focal_length / z_safe + cy
    
    # Draw wireframe
    result = img_bgr.copy()
    pts_2d_int = pts_2d.astype(np.int32)
    
    # Draw edges
    edges_drawn = set()
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            if edge in edges_drawn:
                continue
            edges_drawn.add(edge)
            
            if valid[v1] and valid[v2]:
                p1 = tuple(pts_2d_int[v1])
                p2 = tuple(pts_2d_int[v2])
                if (0 <= p1[0] < w and 0 <= p1[1] < h and
                    0 <= p2[0] < w and 0 <= p2[1] < h):
                    cv2.line(result, p1, p2, color, line_thickness, cv2.LINE_AA)
    
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
                "render_mode": (["solid", "wireframe"], {
                    "default": "solid",
                    "tooltip": "solid uses pyrender, wireframe uses OpenCV"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
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
        opacity: float = 0.5,
    ):
        # Convert image
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = img_np[..., ::-1].copy()
        
        # Extract mesh data
        vertices = mesh_data.get("vertices") or mesh_data.get("verts")
        faces = mesh_data.get("faces")
        cam_t = mesh_data.get("camera") or mesh_data.get("cam_t")
        focal_length = mesh_data.get("focal_length")
        
        if vertices is None or faces is None:
            return (image,)
        
        # Convert to numpy
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if cam_t is not None and isinstance(cam_t, torch.Tensor):
            cam_t = cam_t.cpu().numpy()
        
        vertices = np.array(vertices)
        faces = np.array(faces).astype(np.int32)
        
        if cam_t is None:
            cam_t = np.array([0.0, 0.3, 2.5])
        else:
            cam_t = np.array(cam_t).flatten()
        
        h, w = img_bgr.shape[:2]
        if focal_length is None:
            focal_length = max(h, w)
        else:
            focal_length = float(np.array(focal_length).flatten()[0])
        
        # Render
        if render_mode == "solid":
            result = render_with_pyrender(img_bgr, vertices, faces, cam_t, focal_length)
            if result is None:
                print("[SAM3DBody2abc] Pyrender failed, using wireframe")
                result = render_wireframe_opencv(img_bgr, vertices, faces, cam_t, focal_length)
        else:
            result = render_wireframe_opencv(img_bgr, vertices, faces, cam_t, focal_length)
        
        # Convert back
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
                "render_mode": (["solid", "wireframe"], {
                    "default": "solid",
                    "tooltip": "solid uses pyrender (accurate), wireframe uses OpenCV (fast)"
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
        
        # Check pyrender
        pyrender_available = False
        if render_mode == "solid":
            try:
                import pyrender
                pyrender_available = True
                print(f"[SAM3DBody2abc] pyrender available")
            except ImportError:
                print("[SAM3DBody2abc] pyrender not installed, using wireframe")
                render_mode = "wireframe"
        
        # Apply temporal smoothing
        if temporal_smoothing > 0 and len(mesh_sequence) > 1:
            from scipy.ndimage import gaussian_filter1d
            
            valid_verts = []
            valid_cams = []
            valid_indices = []
            
            for i, mesh in enumerate(mesh_sequence):
                verts = mesh.get("vertices")
                cam = mesh.get("camera")
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
            
            vertices = np.array(vertices)
            faces = np.array(faces).astype(np.int32)
            
            cam_t = mesh.get("camera")
            if cam_t is not None:
                if isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                cam_t = np.array(cam_t).flatten()
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
                print(f"[SAM3DBody2abc] Frame 0: verts={vertices.shape}, cam_t={cam_t}, focal={focal_length}")
            
            # Render
            if render_mode == "solid" and pyrender_available:
                result = render_with_pyrender(img_bgr, vertices, faces, cam_t, focal_length)
                if result is None:
                    result = render_wireframe_opencv(img_bgr, vertices, faces, cam_t, focal_length)
            else:
                result = render_wireframe_opencv(img_bgr, vertices, faces, cam_t, focal_length)
            
            result_rgb = result[..., ::-1].copy()
            result_frames.append(result_rgb)
            pbar.update(1)
        
        result_tensor = torch.from_numpy(np.stack(result_frames).astype(np.float32) / 255.0)
        print(f"[SAM3DBody2abc] Rendered {len(result_frames)} frames")
        
        return (result_tensor,)
