# Copyright (c) 2025 - SAM3DBody2abc
# SPDX-License-Identifier: MIT
"""
Overlay renderer for SAM3DBody2abc.
Renders mesh overlay on images using OpenCV (matching pyrender logic).
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any


def render_mesh_opencv(
    img_bgr: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    cam_t: Optional[np.ndarray],
    focal_length: Optional[float],
    color: Tuple[int, int, int],
    opacity: float,
    line_thickness: int,
    debug: bool = False,
    render_mode: str = "overlay",
) -> np.ndarray:
    """
    Render mesh using OpenCV with projection matching pyrender.
    
    Based on original SAM3DBody renderer.py:
    1. Mesh rotated 180° around X axis
    2. Camera at position cam_t (with X negated)
    3. Mesh is at origin, so mesh position relative to camera = -camera_translation
    4. Y is flipped for image coordinates (OpenGL Y-up to image Y-down)
    """
    
    h, w = img_bgr.shape[:2]
    
    # Default camera params
    if focal_length is None or focal_length <= 0:
        focal_length = max(h, w)
    if cam_t is None:
        cam_t = np.array([0.0, 0.3, 2.5])
    
    cam_t = np.array(cam_t).flatten()
    if len(cam_t) < 3:
        cam_t = np.array([0.0, 0.3, 2.5])
    
    # Step 1: Apply 180° X rotation to vertices (flip Y and Z)
    verts = vertices.copy()
    verts[:, 1] = -verts[:, 1]
    verts[:, 2] = -verts[:, 2]
    
    # Step 2: Transform to camera space
    # In pyrender: camera_translation[0] *= -1.0
    # Camera is at position [-cam_t[0], cam_t[1], cam_t[2]]
    # Mesh is at origin, so in camera space: mesh_pos = -camera_pos
    # verts_cam = verts - camera_translation
    camera_translation = np.array([-cam_t[0], cam_t[1], cam_t[2]])
    verts_cam = verts - camera_translation
    
    if debug:
        print(f"  - Original vertices range: X=[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}], "
              f"Y=[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}], "
              f"Z=[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
        print(f"  - Camera translation: {camera_translation}")
        print(f"  - Camera-space Z range: [{verts_cam[:, 2].min():.3f}, {verts_cam[:, 2].max():.3f}]")
    
    # Step 3: Perspective projection
    # In OpenGL (pyrender), camera looks down -Z, so Z is negative for visible objects
    # We project using -Z to get positive depth values
    z = -verts_cam[:, 2:3]  # Negate Z for OpenGL convention
    
    # Filter vertices that are behind camera (z <= 0 means behind)
    valid_z = z > 0.01
    z_safe = np.where(valid_z, z, 0.01)
    
    cx, cy = w / 2.0, h / 2.0
    
    pts_2d = np.zeros((len(verts), 2))
    # Standard projection: x = fx * X/Z + cx
    # Flip Y for image coordinates (OpenGL Y-up to image Y-down)
    pts_2d[:, 0] = verts_cam[:, 0] * focal_length / z_safe.flatten() + cx
    pts_2d[:, 1] = -verts_cam[:, 1] * focal_length / z_safe.flatten() + cy  # Flip Y
    
    if debug:
        print(f"  - Focal length: {focal_length}, center: ({cx:.1f}, {cy:.1f})")
        print(f"  - Depth (-Z) range: [{z.min():.3f}, {z.max():.3f}]")
        print(f"  - 2D points X range: [{pts_2d[:, 0].min():.1f}, {pts_2d[:, 0].max():.1f}]")
        print(f"  - 2D points Y range: [{pts_2d[:, 1].min():.1f}, {pts_2d[:, 1].max():.1f}]")
        in_frame = ((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) & 
                    (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h) &
                    valid_z.flatten())
        print(f"  - Points in frame: {in_frame.sum()}/{len(pts_2d)}")
    
    # Create mesh overlay
    if render_mode == "mesh_only":
        mesh_img = np.zeros_like(img_bgr)
    else:
        mesh_img = img_bgr.copy()
    
    # Sort faces by depth (far to near for proper occlusion)
    face_depths = []
    for face in faces:
        avg_z = (z_safe[face[0]] + z_safe[face[1]] + z_safe[face[2]]) / 3
        face_depths.append(avg_z[0])
    
    sorted_indices = np.argsort(face_depths)[::-1]  # Far to near
    
    # Draw filled faces with transparency
    overlay = mesh_img.copy()
    faces_drawn = 0
    
    for idx in sorted_indices:
        face = faces[idx]
        
        # Skip if any vertex is behind camera
        if not (valid_z[face[0]] and valid_z[face[1]] and valid_z[face[2]]):
            continue
        
        pts = pts_2d[face].astype(np.int32)
        
        # Check if face is at least partially visible
        if (np.any(pts[:, 0] >= -w) and np.any(pts[:, 0] < 2*w) and 
            np.any(pts[:, 1] >= -h) and np.any(pts[:, 1] < 2*h)):
            cv2.fillPoly(overlay, [pts], color)
            faces_drawn += 1
    
    if debug:
        print(f"  - Faces drawn: {faces_drawn}/{len(faces)}")
    
    # Blend with original
    mesh_img = cv2.addWeighted(overlay, opacity, mesh_img, 1 - opacity, 0)
    
    # Draw edges for definition (only front faces)
    edge_color = tuple(max(0, int(c * 0.6)) for c in color)
    num_edge_faces = max(1, len(sorted_indices) // 3)
    
    for idx in sorted_indices[:num_edge_faces]:
        face = faces[idx]
        
        if not (valid_z[face[0]] and valid_z[face[1]] and valid_z[face[2]]):
            continue
            
        pts = pts_2d[face].astype(np.int32)
        
        for j in range(3):
            p1 = tuple(pts[j])
            p2 = tuple(pts[(j + 1) % 3])
            if ((-w <= p1[0] < 2*w) and (-h <= p1[1] < 2*h) and
                (-w <= p2[0] < 2*w) and (-h <= p2[1] < 2*h)):
                cv2.line(mesh_img, p1, p2, edge_color, line_thickness, cv2.LINE_AA)
    
    if render_mode == "side_by_side":
        result = np.hstack([img_bgr, mesh_img])
    else:
        result = mesh_img
    
    return result


class SAM3DBody2abcOverlay:
    """
    Render 3D mesh overlay on single image.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_MESH", {
                    "tooltip": "Single frame mesh data from SAM3DBody Process node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image to overlay mesh on"
                }),
            },
            "optional": {
                "render_mode": (["overlay", "mesh_only", "side_by_side"], {
                    "default": "overlay"
                }),
                "mesh_color": (["skin", "blue", "green", "red", "white", "cyan"], {
                    "default": "skin"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_image",)
    FUNCTION = "render_overlay"
    CATEGORY = "SAM3DBody2abc/Visualization"
    
    MESH_COLORS = {
        "skin": (200, 180, 255),      # BGR for skin tone
        "blue": (255, 150, 50),       # BGR blue
        "green": (100, 200, 100),     # BGR green
        "red": (100, 100, 255),       # BGR red
        "white": (230, 230, 230),     # BGR white
        "cyan": (255, 255, 100),      # BGR cyan
    }
    
    def render_overlay(
        self,
        mesh_data: Dict,
        image: torch.Tensor,
        render_mode: str = "overlay",
        mesh_color: str = "skin",
        opacity: float = 0.5,
        line_thickness: int = 1,
    ):
        """Render mesh overlay on image."""
        
        print(f"[SAM3DBody2abc] Rendering overlay: mode={render_mode}, color={mesh_color}")
        
        # Convert image to numpy BGR
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = img_np[..., ::-1].copy()
        
        # Extract mesh data
        vertices = mesh_data.get("vertices") or mesh_data.get("verts")
        faces = mesh_data.get("faces")
        cam_t = mesh_data.get("camera") or mesh_data.get("cam_t")
        focal_length = mesh_data.get("focal_length")
        
        if vertices is None or faces is None:
            print("[SAM3DBody2abc] WARNING: No mesh data for overlay")
            return (image,)
        
        # Convert to numpy
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.cpu().numpy()
        if cam_t is not None and isinstance(cam_t, torch.Tensor):
            cam_t = cam_t.cpu().numpy()
        if focal_length is not None:
            if isinstance(focal_length, torch.Tensor):
                focal_length = focal_length.cpu().numpy()
            focal_length = float(np.array(focal_length).flatten()[0])
        
        vertices = np.array(vertices)
        faces = np.array(faces).astype(np.int32)
        
        print(f"[SAM3DBody2abc] Mesh: {len(vertices)} verts, {len(faces)} faces")
        print(f"[SAM3DBody2abc] Camera: {cam_t}, focal: {focal_length}")
        
        # Get mesh color
        color = self.MESH_COLORS.get(mesh_color, self.MESH_COLORS["skin"])
        
        # Render mesh
        result = render_mesh_opencv(
            img_bgr, vertices, faces, cam_t, focal_length, 
            color, opacity, line_thickness, debug=True, render_mode=render_mode
        )
        
        print("[SAM3DBody2abc] [OK] Rendered overlay")
        
        # Convert back to ComfyUI format
        result_rgb = result[..., ::-1].copy()
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)


class SAM3DBody2abcOverlayBatch:
    """
    Render mesh overlay on batch of images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_sequence": ("MESH_SEQUENCE", {
                    "tooltip": "Sequence of mesh data from batch processor"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Batch of input images"
                }),
            },
            "optional": {
                "mesh_color": (["skin", "blue", "green", "red", "white", "cyan"], {
                    "default": "skin"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_images",)
    FUNCTION = "render_batch"
    CATEGORY = "SAM3DBody2abc/Visualization"
    
    MESH_COLORS = {
        "skin": (200, 180, 255),
        "blue": (255, 150, 50),
        "green": (100, 200, 100),
        "red": (100, 100, 255),
        "white": (230, 230, 230),
        "cyan": (255, 255, 100),
    }
    
    def render_batch(
        self,
        mesh_sequence: List[Dict],
        images: torch.Tensor,
        mesh_color: str = "skin",
        opacity: float = 0.5,
        line_thickness: int = 1,
    ):
        """Render mesh overlays on batch of images."""
        import comfy.utils
        
        print(f"[SAM3DBody2abc] Batch overlay: {len(mesh_sequence)} meshes, {images.shape[0]} images")
        
        color = self.MESH_COLORS.get(mesh_color, self.MESH_COLORS["skin"])
        
        # Process each frame
        result_frames = []
        pbar = comfy.utils.ProgressBar(len(mesh_sequence))
        
        for i, mesh in enumerate(mesh_sequence):
            # Get corresponding image
            img_idx = min(i, images.shape[0] - 1)
            img_np = images[img_idx].cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = img_np[..., ::-1].copy()
            
            # Check if valid mesh
            vertices = mesh.get("vertices")
            faces = mesh.get("faces")
            
            if vertices is None or faces is None or not mesh.get("valid", True):
                # No mesh, just use original image
                result_frames.append(img_np)
                pbar.update(1)
                continue
            
            # Convert to numpy
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()
            
            vertices = np.array(vertices)
            faces = np.array(faces).astype(np.int32)
            
            # Get camera params
            cam_t = mesh.get("camera")
            if cam_t is not None:
                if isinstance(cam_t, torch.Tensor):
                    cam_t = cam_t.cpu().numpy()
                cam_t = np.array(cam_t).flatten()
            
            focal_length = mesh.get("focal_length")
            if focal_length is not None:
                if isinstance(focal_length, torch.Tensor):
                    focal_length = focal_length.cpu().numpy()
                focal_length = float(np.array(focal_length).flatten()[0])
            
            # Debug first frame
            if i == 0:
                print(f"[SAM3DBody2abc] Frame 0 debug:")
                print(f"  - Vertices shape: {vertices.shape}")
                print(f"  - Vertices range: X=[{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}], "
                      f"Y=[{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}], "
                      f"Z=[{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
                print(f"  - Camera: {cam_t}")
                print(f"  - Focal length: {focal_length}")
                print(f"  - Image size: {img_bgr.shape}")
            
            # Render
            result = render_mesh_opencv(
                img_bgr, vertices, faces, cam_t, focal_length, 
                color, opacity, line_thickness, debug=(i==0)
            )
            
            # Convert BGR to RGB
            result_rgb = result[..., ::-1].copy()
            result_frames.append(result_rgb)
            pbar.update(1)
        
        # Stack frames
        result_tensor = torch.from_numpy(
            np.stack(result_frames).astype(np.float32) / 255.0
        )
        
        print(f"[SAM3DBody2abc] [OK] Rendered {len(result_frames)} frames")
        return (result_tensor,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_Overlay": SAM3DBody2abcOverlay,
    "SAM3DBody2abc_OverlayBatch": SAM3DBody2abcOverlayBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_Overlay": "🎨 SAM3DBody2abc Overlay",
    "SAM3DBody2abc_OverlayBatch": "🎨 SAM3DBody2abc Overlay Batch",
}
