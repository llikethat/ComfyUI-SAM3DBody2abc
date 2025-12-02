"""
Mesh Overlay Renderer
Render 3D mesh overlay on images for visualization.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class RenderMeshOverlay:
    """
    Render 3D mesh overlay on a single image.
    Creates visualization showing mesh wireframe or shaded surface on the original image.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),  # Single image [1, H, W, C]
                "sam3dbody_mesh": ("SAM3D_MESH",),  # From SAM3DBody
            },
            "optional": {
                "render_mode": (["wireframe", "shaded", "both", "joints_only"], {
                    "default": "wireframe"
                }),
                "mesh_color": ("STRING", {
                    "default": "0,255,0",
                    "multiline": False,
                    "tooltip": "R,G,B color for mesh (0-255)"
                }),
                "joint_color": ("STRING", {
                    "default": "255,0,0",
                    "multiline": False,
                    "tooltip": "R,G,B color for joints"
                }),
                "line_thickness": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5
                }),
                "joint_radius": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10
                }),
                "opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "show_joints": ("BOOLEAN", {"default": True}),
                "show_skeleton": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay_image",)
    FUNCTION = "render_overlay"
    CATEGORY = "SAM3DBody/Visualization"
    
    # SMPL skeleton connections
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (0, 3),  # Pelvis to hips and spine
        (1, 4), (2, 5),  # Hips to knees
        (4, 7), (5, 8),  # Knees to ankles
        (7, 10), (8, 11),  # Ankles to feet
        (3, 6), (6, 9),  # Spine
        (9, 12), (12, 15),  # Spine to head
        (9, 13), (9, 14),  # Spine to collars
        (13, 16), (14, 17),  # Collars to shoulders
        (16, 18), (17, 19),  # Shoulders to elbows
        (18, 20), (19, 21),  # Elbows to wrists
        (20, 22), (21, 23),  # Wrists to hands
    ]
    
    def render_overlay(
        self,
        image: torch.Tensor,
        sam3dbody_mesh: Dict,
        render_mode: str = "wireframe",
        mesh_color: str = "0,255,0",
        joint_color: str = "255,0,0",
        line_thickness: int = 1,
        joint_radius: int = 3,
        opacity: float = 0.7,
        show_joints: bool = True,
        show_skeleton: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Render mesh overlay on image."""
        
        # Parse colors
        mesh_rgb = self._parse_color(mesh_color)
        joint_rgb = self._parse_color(joint_color)
        
        # Get image as numpy
        img = image[0].cpu().numpy()  # [H, W, C]
        H, W, C = img.shape
        img = (img * 255).astype(np.uint8)
        overlay = img.copy()
        
        # Get projection data
        joints_2d = self._get_2d_joints(sam3dbody_mesh, W, H)
        verts_2d = self._get_2d_vertices(sam3dbody_mesh, W, H)
        faces = sam3dbody_mesh.get("faces")
        
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False
        
        # Render based on mode
        if render_mode in ["wireframe", "both"] and verts_2d is not None and faces is not None:
            overlay = self._render_wireframe(overlay, verts_2d, faces, mesh_rgb, line_thickness, has_cv2)
        
        if render_mode in ["shaded", "both"] and verts_2d is not None:
            overlay = self._render_shaded(overlay, verts_2d, faces, mesh_rgb, opacity)
        
        # Render joints and skeleton
        if joints_2d is not None:
            if show_skeleton:
                overlay = self._render_skeleton(overlay, joints_2d, joint_rgb, line_thickness, has_cv2)
            if show_joints or render_mode == "joints_only":
                overlay = self._render_joints(overlay, joints_2d, joint_rgb, joint_radius, has_cv2)
        
        # Blend with original
        result = (img.astype(np.float32) * (1 - opacity) + overlay.astype(np.float32) * opacity).astype(np.uint8)
        
        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse R,G,B string to tuple."""
        try:
            parts = [int(x.strip()) for x in color_str.split(",")]
            return tuple(parts[:3])
        except:
            return (0, 255, 0)
    
    def _get_2d_joints(self, mesh: Dict, W: int, H: int) -> Optional[np.ndarray]:
        """Get 2D joint positions."""
        # Try different keys
        for key in ["joints_2d", "J_2d", "keypoints_2d"]:
            if key in mesh and mesh[key] is not None:
                joints = np.array(mesh[key])
                if joints.shape[-1] == 2:
                    return joints
        
        # Project 3D joints if camera available
        joints_3d = mesh.get("joints") or mesh.get("J")
        camera = mesh.get("camera") or mesh.get("cam")
        
        if joints_3d is not None and camera is not None:
            return self._project_to_2d(np.array(joints_3d), camera, W, H)
        
        return None
    
    def _get_2d_vertices(self, mesh: Dict, W: int, H: int) -> Optional[np.ndarray]:
        """Get 2D vertex positions."""
        verts_3d = mesh.get("verts") or mesh.get("vertices")
        if verts_3d is None:
            return None
        
        verts_3d = np.array(verts_3d)
        camera = mesh.get("camera") or mesh.get("cam")
        
        if camera is not None:
            return self._project_to_2d(verts_3d, camera, W, H)
        
        # Simple orthographic projection as fallback
        verts_2d = verts_3d[:, :2]
        verts_2d = (verts_2d - verts_2d.min(axis=0)) / (verts_2d.max(axis=0) - verts_2d.min(axis=0) + 1e-6)
        verts_2d = verts_2d * np.array([W * 0.8, H * 0.8]) + np.array([W * 0.1, H * 0.1])
        return verts_2d
    
    def _project_to_2d(self, points_3d: np.ndarray, camera: Any, W: int, H: int) -> np.ndarray:
        """Project 3D points to 2D using camera parameters."""
        # Handle different camera formats
        if isinstance(camera, dict):
            focal = camera.get("focal_length", 5000)
            cx = camera.get("cx", W / 2)
            cy = camera.get("cy", H / 2)
        elif isinstance(camera, (list, np.ndarray)):
            camera = np.array(camera).flatten()
            if len(camera) >= 3:
                # Assume [s, tx, ty] format
                s, tx, ty = camera[0], camera[1], camera[2]
                points_2d = points_3d[:, :2] * s + np.array([tx, ty])
                points_2d = points_2d * np.array([W, H]) / 2 + np.array([W, H]) / 2
                return points_2d
            focal = 5000
            cx, cy = W / 2, H / 2
        else:
            focal = 5000
            cx, cy = W / 2, H / 2
        
        # Perspective projection
        z = points_3d[:, 2:3] + 1e-6
        points_2d = points_3d[:, :2] * focal / z
        points_2d = points_2d + np.array([cx, cy])
        
        return points_2d
    
    def _render_wireframe(
        self,
        img: np.ndarray,
        verts_2d: np.ndarray,
        faces: np.ndarray,
        color: Tuple,
        thickness: int,
        has_cv2: bool
    ) -> np.ndarray:
        """Render wireframe mesh."""
        if has_cv2:
            import cv2
            faces = np.array(faces)
            for face in faces[:500]:  # Limit for performance
                pts = verts_2d[face].astype(np.int32)
                for i in range(len(pts)):
                    pt1 = tuple(pts[i])
                    pt2 = tuple(pts[(i + 1) % len(pts)])
                    cv2.line(img, pt1, pt2, color, thickness)
        else:
            # Simple numpy rendering
            faces = np.array(faces)
            for face in faces[:200]:
                pts = verts_2d[face].astype(np.int32)
                for i in range(len(pts)):
                    self._draw_line_numpy(img, pts[i], pts[(i + 1) % len(pts)], color)
        return img
    
    def _render_shaded(
        self,
        img: np.ndarray,
        verts_2d: np.ndarray,
        faces: Optional[np.ndarray],
        color: Tuple,
        opacity: float
    ) -> np.ndarray:
        """Render shaded mesh (simplified)."""
        if faces is None:
            return img
        
        try:
            import cv2
            faces = np.array(faces)
            for face in faces[:500]:
                pts = verts_2d[face].astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts], color)
        except ImportError:
            pass
        
        return img
    
    def _render_joints(
        self,
        img: np.ndarray,
        joints_2d: np.ndarray,
        color: Tuple,
        radius: int,
        has_cv2: bool
    ) -> np.ndarray:
        """Render joint points."""
        if has_cv2:
            import cv2
            for joint in joints_2d:
                pt = tuple(joint.astype(np.int32))
                cv2.circle(img, pt, radius, color, -1)
        else:
            for joint in joints_2d:
                x, y = int(joint[0]), int(joint[1])
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius*radius:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0]:
                                img[ny, nx] = color
        return img
    
    def _render_skeleton(
        self,
        img: np.ndarray,
        joints_2d: np.ndarray,
        color: Tuple,
        thickness: int,
        has_cv2: bool
    ) -> np.ndarray:
        """Render skeleton connections."""
        if has_cv2:
            import cv2
            for conn in self.SKELETON_CONNECTIONS:
                if conn[0] < len(joints_2d) and conn[1] < len(joints_2d):
                    pt1 = tuple(joints_2d[conn[0]].astype(np.int32))
                    pt2 = tuple(joints_2d[conn[1]].astype(np.int32))
                    cv2.line(img, pt1, pt2, color, thickness)
        else:
            for conn in self.SKELETON_CONNECTIONS:
                if conn[0] < len(joints_2d) and conn[1] < len(joints_2d):
                    self._draw_line_numpy(img, joints_2d[conn[0]], joints_2d[conn[1]], color)
        return img
    
    def _draw_line_numpy(self, img: np.ndarray, pt1: np.ndarray, pt2: np.ndarray, color: Tuple):
        """Draw line using numpy (fallback)."""
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy)
        
        if steps == 0:
            return
        
        x_inc = (x2 - x1) / steps
        y_inc = (y2 - y1) / steps
        
        x, y = x1, y1
        for _ in range(int(steps)):
            if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]:
                img[int(y), int(x)] = color
            x += x_inc
            y += y_inc


class RenderMeshOverlayBatch:
    """
    Render mesh overlay on a batch of images using mesh sequence.
    Creates visualization for entire video/image sequence.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),  # [N, H, W, C] batch
                "mesh_sequence": ("MESH_SEQUENCE",),
            },
            "optional": {
                "render_mode": (["wireframe", "joints_only", "both"], {"default": "wireframe"}),
                "mesh_color": ("STRING", {"default": "0,255,0"}),
                "joint_color": ("STRING", {"default": "255,0,0"}),
                "opacity": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "show_joints": ("BOOLEAN", {"default": True}),
                "show_skeleton": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay_images",)
    FUNCTION = "render_batch"
    CATEGORY = "SAM3DBody/Visualization"
    
    def render_batch(
        self,
        images: torch.Tensor,
        mesh_sequence: List[Dict],
        render_mode: str = "wireframe",
        mesh_color: str = "0,255,0",
        joint_color: str = "255,0,0",
        opacity: float = 0.7,
        show_joints: bool = True,
        show_skeleton: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Render overlay on all frames."""
        import comfy.utils
        
        renderer = RenderMeshOverlay()
        
        n_frames = min(images.shape[0], len(mesh_sequence))
        results = []
        
        pbar = comfy.utils.ProgressBar(n_frames)
        
        for i in range(n_frames):
            frame = images[i:i+1]
            mesh = mesh_sequence[i]
            
            if mesh.get("valid"):
                # Convert mesh_sequence format to SAM3DBODY_MESH format
                sam3dbody_mesh = {
                    "verts": mesh.get("vertices"),
                    "faces": mesh.get("faces"),
                    "joints": mesh.get("joints"),
                    "joints_2d": mesh.get("joints_2d"),
                    "camera": mesh.get("camera"),
                }
                
                result, = renderer.render_overlay(
                    frame, sam3dbody_mesh,
                    render_mode=render_mode,
                    mesh_color=mesh_color,
                    joint_color=joint_color,
                    opacity=opacity,
                    show_joints=show_joints,
                    show_skeleton=show_skeleton,
                )
                results.append(result)
            else:
                results.append(frame)
            
            pbar.update(1)
        
        return (torch.cat(results, dim=0),)
