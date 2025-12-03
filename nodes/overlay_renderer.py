# Copyright (c) 2025 - SAM3DBody2abc
# SPDX-License-Identifier: MIT
"""
Overlay renderer for SAM3DBody2abc.
Uses pyrender for proper 3D mesh rendering with lighting.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

# Set EGL platform for headless rendering
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"


class SAM3DBody2abcOverlay:
    """
    Render 3D mesh overlay on single image using pyrender.
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
                "mesh_color": (["skin", "blue", "green", "red", "white"], {
                    "default": "skin"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_image",)
    FUNCTION = "render_overlay"
    CATEGORY = "SAM3DBody2abc/Visualization"
    
    MESH_COLORS = {
        "skin": (1.0, 0.8, 0.7),
        "blue": (0.0, 0.447, 0.741),
        "green": (0.466, 0.674, 0.188),
        "red": (0.850, 0.325, 0.098),
        "white": (0.9, 0.9, 0.9),
    }
    
    def render_overlay(
        self,
        mesh_data: Dict,
        image: torch.Tensor,
        render_mode: str = "overlay",
        mesh_color: str = "skin",
        opacity: float = 0.8,
    ):
        """Render mesh overlay on image."""
        
        print(f"[SAM3DBody2abc] Rendering overlay: mode={render_mode}")
        
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
        
        # Ensure faces are int
        faces = faces.astype(np.int32)
        
        print(f"[SAM3DBody2abc] Mesh: {len(vertices)} verts, {len(faces)} faces")
        print(f"[SAM3DBody2abc] Camera: {cam_t}, focal: {focal_length}")
        
        # Get mesh color
        color = self.MESH_COLORS.get(mesh_color, self.MESH_COLORS["skin"])
        
        # Try pyrender first
        try:
            result = self._render_with_pyrender(
                img_bgr, vertices, faces, cam_t, focal_length, 
                color, render_mode, opacity
            )
            print("[SAM3DBody2abc] [OK] Rendered with pyrender")
        except Exception as e:
            print(f"[SAM3DBody2abc] pyrender failed: {e}, trying fallback...")
            result = self._render_wireframe_fallback(
                img_bgr, vertices, faces, cam_t, focal_length, color
            )
        
        # Convert back to ComfyUI format
        result_rgb = result[..., ::-1].copy()
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)
    
    def _render_with_pyrender(
        self,
        img_bgr: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam_t: Optional[np.ndarray],
        focal_length: Optional[float],
        color: Tuple[float, float, float],
        render_mode: str,
        opacity: float,
    ) -> np.ndarray:
        """Render mesh using pyrender."""
        import pyrender
        import trimesh
        
        h, w = img_bgr.shape[:2]
        img_float = img_bgr.astype(np.float32) / 255.0
        
        # Default camera params
        if focal_length is None:
            focal_length = max(h, w) * 1.2
        if cam_t is None:
            cam_t = np.array([0, 0, 3.0])
        
        cam_t = np.array(cam_t).flatten()
        if len(cam_t) < 3:
            cam_t = np.array([0, 0, 3.0])
        
        # Camera translation (flip X for OpenGL convention)
        camera_translation = cam_t.copy()
        camera_translation[0] *= -1.0
        
        # Create mesh with 180 degree X rotation (MHR coordinate fix)
        mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        
        # Material
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(color[2], color[1], color[0], 1.0),  # BGR for pyrender
        )
        
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        # Scene
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh_pyrender, "mesh")
        
        # Camera
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=w / 2.0, cy=h / 2.0,
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)
        
        # Lights
        light_nodes = self._create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        
        # Render
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        
        color_float = color_rgba.astype(np.float32) / 255.0
        
        if render_mode == "mesh_only":
            # Just the mesh on black background
            result = (color_float[:, :, :3] * 255).astype(np.uint8)
        elif render_mode == "side_by_side":
            # Original and overlay side by side
            overlay = self._composite(img_float, color_float, opacity)
            result = np.hstack([img_bgr, (overlay * 255).astype(np.uint8)])
        else:  # overlay
            result = (self._composite(img_float, color_float, opacity) * 255).astype(np.uint8)
        
        return result
    
    def _composite(self, background: np.ndarray, foreground_rgba: np.ndarray, opacity: float) -> np.ndarray:
        """Composite RGBA foreground over background."""
        alpha = foreground_rgba[:, :, 3:4] * opacity
        fg_rgb = foreground_rgba[:, :, :3]
        return fg_rgb * alpha + background * (1 - alpha)
    
    def _create_raymond_lights(self):
        """Create raymond lighting setup."""
        import pyrender
        
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
        
        nodes = []
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
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )
        return nodes
    
    def _render_wireframe_fallback(
        self,
        img_bgr: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam_t: Optional[np.ndarray],
        focal_length: Optional[float],
        color: Tuple[float, float, float],
    ) -> np.ndarray:
        """Fallback: render wireframe using OpenCV."""
        try:
            import cv2
            has_cv2 = True
        except:
            has_cv2 = False
            return img_bgr
        
        h, w = img_bgr.shape[:2]
        result = img_bgr.copy()
        
        # Default params
        if focal_length is None:
            focal_length = max(h, w) * 1.2
        if cam_t is None:
            cam_t = np.array([0, 0, 3.0])
        
        cam_t = np.array(cam_t).flatten()
        
        # Apply 180 degree X rotation
        verts = vertices.copy()
        verts[:, 1] = -verts[:, 1]
        verts[:, 2] = -verts[:, 2]
        
        # Project to 2D
        z = verts[:, 2:3] + cam_t[2]
        z = np.maximum(z, 0.1)  # Avoid division by zero
        
        pts_2d = verts[:, :2] * focal_length / z
        pts_2d[:, 0] += w / 2 + cam_t[0] * focal_length / cam_t[2]
        pts_2d[:, 1] += h / 2 + cam_t[1] * focal_length / cam_t[2]
        
        # Draw wireframe (sample every Nth edge)
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        
        for i, face in enumerate(faces[::10]):  # Sample faces
            for j in range(3):
                p1_idx, p2_idx = face[j], face[(j + 1) % 3]
                p1 = tuple(pts_2d[p1_idx].astype(int))
                p2 = tuple(pts_2d[p2_idx].astype(int))
                
                # Bounds check
                if (0 <= p1[0] < w and 0 <= p1[1] < h and
                    0 <= p2[0] < w and 0 <= p2[1] < h):
                    cv2.line(result, p1, p2, color_bgr, 1, cv2.LINE_AA)
        
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
                "mesh_color": (["skin", "blue", "green", "red", "white"], {
                    "default": "skin"
                }),
                "opacity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_images",)
    FUNCTION = "render_batch"
    CATEGORY = "SAM3DBody2abc/Visualization"
    
    MESH_COLORS = {
        "skin": (1.0, 0.8, 0.7),
        "blue": (0.0, 0.447, 0.741),
        "green": (0.466, 0.674, 0.188),
        "red": (0.850, 0.325, 0.098),
        "white": (0.9, 0.9, 0.9),
    }
    
    def render_batch(
        self,
        mesh_sequence: List[Dict],
        images: torch.Tensor,
        mesh_color: str = "skin",
        opacity: float = 0.8,
    ):
        """Render mesh overlays on batch of images."""
        import comfy.utils
        
        print(f"[SAM3DBody2abc] Batch overlay: {len(mesh_sequence)} meshes, {images.shape[0]} images")
        
        # Check for pyrender
        try:
            import pyrender
            import trimesh
            use_pyrender = True
            print("[SAM3DBody2abc] Using pyrender for rendering")
        except ImportError:
            use_pyrender = False
            print("[SAM3DBody2abc] pyrender not available, using wireframe fallback")
        
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
            if use_pyrender:
                try:
                    result = self._render_pyrender(
                        img_bgr, vertices, faces, cam_t, focal_length, color, opacity
                    )
                except Exception as e:
                    result = self._render_wireframe(img_bgr, vertices, faces, cam_t, focal_length, color)
            else:
                result = self._render_wireframe(img_bgr, vertices, faces, cam_t, focal_length, color)
            
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
    
    def _render_pyrender(
        self,
        img_bgr: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam_t: Optional[np.ndarray],
        focal_length: Optional[float],
        color: Tuple[float, float, float],
        opacity: float,
    ) -> np.ndarray:
        """Render using pyrender."""
        import pyrender
        import trimesh
        
        h, w = img_bgr.shape[:2]
        img_float = img_bgr.astype(np.float32) / 255.0
        
        # Defaults
        if focal_length is None:
            focal_length = max(h, w) * 1.2
        if cam_t is None:
            cam_t = np.array([0, 0, 3.0])
        
        if len(cam_t) < 3:
            cam_t = np.array([0, 0, 3.0])
        
        camera_translation = cam_t.copy()
        camera_translation[0] *= -1.0
        
        # Mesh with 180° X rotation
        mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(color[2], color[1], color[0], 1.0),
        )
        
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh_pyrender, "mesh")
        
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=w / 2.0, cy=h / 2.0, zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)
        
        # Raymond lights
        for node in self._create_lights():
            scene.add_node(node)
        
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        
        # Composite
        color_float = color_rgba.astype(np.float32) / 255.0
        alpha = color_float[:, :, 3:4] * opacity
        result = color_float[:, :, :3] * alpha + img_float * (1 - alpha)
        
        return (result * 255).astype(np.uint8)
    
    def _create_lights(self):
        """Create raymond lights."""
        import pyrender
        
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
        
        nodes = []
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
            nodes.append(
                pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )
        return nodes
    
    def _render_wireframe(
        self,
        img_bgr: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        cam_t: Optional[np.ndarray],
        focal_length: Optional[float],
        color: Tuple[float, float, float],
    ) -> np.ndarray:
        """Fallback wireframe rendering."""
        try:
            import cv2
        except:
            return img_bgr
        
        h, w = img_bgr.shape[:2]
        result = img_bgr.copy()
        
        if focal_length is None:
            focal_length = max(h, w) * 1.2
        if cam_t is None:
            cam_t = np.array([0, 0, 3.0])
        if len(cam_t) < 3:
            cam_t = np.array([0, 0, 3.0])
        
        # 180° X rotation
        verts = vertices.copy()
        verts[:, 1] = -verts[:, 1]
        verts[:, 2] = -verts[:, 2]
        
        # Project
        z = verts[:, 2:3] + cam_t[2]
        z = np.maximum(z, 0.1)
        
        pts_2d = verts[:, :2] * focal_length / z
        pts_2d[:, 0] += w / 2 + cam_t[0] * focal_length / cam_t[2]
        pts_2d[:, 1] += h / 2 + cam_t[1] * focal_length / cam_t[2]
        
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        
        # Draw sampled edges
        for face in faces[::10]:
            for j in range(3):
                p1 = tuple(pts_2d[face[j]].astype(int))
                p2 = tuple(pts_2d[face[(j + 1) % 3]].astype(int))
                if (0 <= p1[0] < w and 0 <= p1[1] < h and
                    0 <= p2[0] < w and 0 <= p2[1] < h):
                    cv2.line(result, p1, p2, color_bgr, 1, cv2.LINE_AA)
        
        return result


# Node registration
NODE_CLASS_MAPPINGS = {
    "SAM3DBody2abc_Overlay": SAM3DBody2abcOverlay,
    "SAM3DBody2abc_OverlayBatch": SAM3DBody2abcOverlayBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBody2abc_Overlay": "🎨 SAM3DBody2abc Overlay",
    "SAM3DBody2abc_OverlayBatch": "🎨 SAM3DBody2abc Overlay Batch",
}
