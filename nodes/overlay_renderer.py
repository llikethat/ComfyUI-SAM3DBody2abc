# Copyright (c) 2025 - SAM3DBody2abc
# SPDX-License-Identifier: MIT
"""
Overlay renderer for SAM3DBody2abc.
Renders mesh overlay on images using OpenCV.
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Any


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
        
        faces = faces.astype(np.int32)
        h, w = img_bgr.shape[:2]
        
        print(f"[SAM3DBody2abc] Mesh: {len(vertices)} verts, {len(faces)} faces")
        print(f"[SAM3DBody2abc] Camera: {cam_t}, focal: {focal_length}")
        
        # Get mesh color
        color = self.MESH_COLORS.get(mesh_color, self.MESH_COLORS["skin"])
        
        # Render mesh
        result = self._render_mesh(
            img_bgr, vertices, faces, cam_t, focal_length, 
            color, render_mode, opacity, line_thickness
        )
        
        print("[SAM3DBody2abc] [OK] Rendered overlay")
        
        # Convert back to ComfyUI format
        result_rgb = result[..., ::-1].copy()
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)
    
    def _render_mesh(
        self,
        img_bgr: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam_t: Optional[np.ndarray],
        focal_length: Optional[float],
        color: Tuple[int, int, int],
        render_mode: str,
        opacity: float,
        line_thickness: int,
    ) -> np.ndarray:
        """Render mesh using OpenCV with proper projection."""
        
        h, w = img_bgr.shape[:2]
        
        # Default camera params
        if focal_length is None or focal_length <= 0:
            focal_length = max(h, w) * 1.2
        if cam_t is None:
            cam_t = np.array([0, 0, 3.0])
        
        cam_t = np.array(cam_t).flatten()
        if len(cam_t) < 3:
            cam_t = np.array([0, 0, 3.0])
        
        # Apply 180 degree X rotation (flip Y and Z) - MHR coordinate fix
        verts = vertices.copy()
        verts[:, 1] = -verts[:, 1]
        verts[:, 2] = -verts[:, 2]
        
        # Perspective projection
        # Add camera translation
        verts_cam = verts.copy()
        verts_cam[:, 0] += cam_t[0]
        verts_cam[:, 1] += cam_t[1]
        verts_cam[:, 2] += cam_t[2]
        
        # Project to 2D with perspective
        z = verts_cam[:, 2:3]
        z = np.maximum(z, 0.01)  # Avoid division by zero
        
        pts_2d = np.zeros((len(verts), 2))
        pts_2d[:, 0] = verts_cam[:, 0] * focal_length / z.flatten() + w / 2
        pts_2d[:, 1] = verts_cam[:, 1] * focal_length / z.flatten() + h / 2
        
        # Create mesh overlay
        if render_mode == "mesh_only":
            mesh_img = np.zeros_like(img_bgr)
        else:
            mesh_img = img_bgr.copy()
        
        # Sort faces by depth for proper rendering
        face_depths = []
        for face in faces:
            avg_z = (z[face[0]] + z[face[1]] + z[face[2]]) / 3
            face_depths.append(avg_z[0])
        
        sorted_indices = np.argsort(face_depths)[::-1]  # Far to near
        
        # Draw filled faces with transparency
        overlay = mesh_img.copy()
        
        for idx in sorted_indices:
            face = faces[idx]
            pts = pts_2d[face].astype(np.int32)
            
            # Check if face is visible (all points in frame)
            if np.all(pts[:, 0] >= -w) and np.all(pts[:, 0] < 2*w) and \
               np.all(pts[:, 1] >= -h) and np.all(pts[:, 1] < 2*h):
                # Draw filled triangle
                cv2.fillPoly(overlay, [pts], color)
        
        # Blend with original
        mesh_img = cv2.addWeighted(overlay, opacity, mesh_img, 1 - opacity, 0)
        
        # Draw edges for definition
        edge_color = tuple(int(c * 0.6) for c in color)  # Darker edges
        for idx in sorted_indices[:len(sorted_indices)//3]:  # Only front-facing edges
            face = faces[idx]
            pts = pts_2d[face].astype(np.int32)
            
            if np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w) and \
               np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h):
                for j in range(3):
                    p1 = tuple(pts[j])
                    p2 = tuple(pts[(j + 1) % 3])
                    cv2.line(mesh_img, p1, p2, edge_color, line_thickness, cv2.LINE_AA)
        
        if render_mode == "side_by_side":
            result = np.hstack([img_bgr, mesh_img])
        else:
            result = mesh_img
        
        return result


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
            
            faces = faces.astype(np.int32)
            
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
            
            # Render
            result = self._render_mesh(
                img_bgr, vertices, faces, cam_t, focal_length, 
                color, opacity, line_thickness
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
    
    def _render_mesh(
        self,
        img_bgr: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam_t: Optional[np.ndarray],
        focal_length: Optional[float],
        color: Tuple[int, int, int],
        opacity: float,
        line_thickness: int,
    ) -> np.ndarray:
        """Render mesh using OpenCV with proper projection."""
        
        h, w = img_bgr.shape[:2]
        
        # Default camera params
        if focal_length is None or focal_length <= 0:
            focal_length = max(h, w) * 1.2
        if cam_t is None:
            cam_t = np.array([0, 0, 3.0])
        
        cam_t = np.array(cam_t).flatten()
        if len(cam_t) < 3:
            cam_t = np.array([0, 0, 3.0])
        
        # Apply 180 degree X rotation (flip Y and Z) - MHR coordinate fix
        verts = vertices.copy()
        verts[:, 1] = -verts[:, 1]
        verts[:, 2] = -verts[:, 2]
        
        # Perspective projection
        verts_cam = verts.copy()
        verts_cam[:, 0] += cam_t[0]
        verts_cam[:, 1] += cam_t[1]
        verts_cam[:, 2] += cam_t[2]
        
        z = verts_cam[:, 2:3]
        z = np.maximum(z, 0.01)
        
        pts_2d = np.zeros((len(verts), 2))
        pts_2d[:, 0] = verts_cam[:, 0] * focal_length / z.flatten() + w / 2
        pts_2d[:, 1] = verts_cam[:, 1] * focal_length / z.flatten() + h / 2
        
        mesh_img = img_bgr.copy()
        
        # Sort faces by depth
        face_depths = []
        for face in faces:
            avg_z = (z[face[0]] + z[face[1]] + z[face[2]]) / 3
            face_depths.append(avg_z[0])
        
        sorted_indices = np.argsort(face_depths)[::-1]
        
        # Draw filled faces
        overlay = mesh_img.copy()
        
        for idx in sorted_indices:
            face = faces[idx]
            pts = pts_2d[face].astype(np.int32)
            
            if np.all(pts[:, 0] >= -w) and np.all(pts[:, 0] < 2*w) and \
               np.all(pts[:, 1] >= -h) and np.all(pts[:, 1] < 2*h):
                cv2.fillPoly(overlay, [pts], color)
        
        mesh_img = cv2.addWeighted(overlay, opacity, mesh_img, 1 - opacity, 0)
        
        # Draw edges
        edge_color = tuple(int(c * 0.6) for c in color)
        for idx in sorted_indices[:len(sorted_indices)//3]:
            face = faces[idx]
            pts = pts_2d[face].astype(np.int32)
            
            if np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w) and \
               np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h):
                for j in range(3):
                    p1 = tuple(pts[j])
                    p2 = tuple(pts[(j + 1) % 3])
                    cv2.line(mesh_img, p1, p2, edge_color, line_thickness, cv2.LINE_AA)
        
        return mesh_img


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_Overlay": SAM3DBody2abcOverlay,
    "SAM3DBody2abc_OverlayBatch": SAM3DBody2abcOverlayBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_Overlay": "🎨 SAM3DBody2abc Overlay",
    "SAM3DBody2abc_OverlayBatch": "🎨 SAM3DBody2abc Overlay Batch",
}
