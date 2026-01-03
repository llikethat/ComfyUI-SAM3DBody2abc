"""
Intrinsics from SAM3DBody - Extract Camera Intrinsics for SAM3DBody2abc v5.0

This node extracts camera intrinsics from SAM3DBody's mesh_data output.

Usage:
    SAM3DBody Process → mesh_data → IntrinsicsFromSAM3DBody → INTRINSICS
                                              ↓
                                        debug_overlay
                                              ↓
                                      CameraSolverV2

Version: 5.0.0
Author: SAM3DBody2abc Project
"""

import numpy as np
import torch
import cv2
import os
from typing import Dict, Tuple, Any, Optional

# Try to import pyrender for high-quality mesh rendering
PYRENDER_AVAILABLE = False
try:
    # Set environment for headless rendering before importing pyrender
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import pyrender
    import trimesh
    PYRENDER_AVAILABLE = True
    print("[IntrinsicsFromSAM3DBody] pyrender available - using high-quality mesh rendering")
except ImportError:
    print("[IntrinsicsFromSAM3DBody] pyrender not available - using fallback rendering")
    print("[IntrinsicsFromSAM3DBody] For better mesh quality, install: pip install pyrender trimesh PyOpenGL")


def to_numpy(data):
    """Convert tensor to numpy."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data.copy()
    try:
        return np.array(data)
    except:
        return data


class IntrinsicsFromSAM3DBody:
    """
    Extract camera intrinsics from SAM3DBody mesh_data output.
    
    Takes mesh_data (SAM3D_OUTPUT) and extracts:
    - focal_length (in pixels)
    - pred_cam_t (camera translation)
    - Builds standardized INTRINSICS for v5.0 pipeline
    
    Generates a debug overlay showing the mesh projection.
    """
    
    FRAME_SELECTION = ["first", "middle", "last", "specific"]
    RENDER_MODES = ["solid", "wireframe", "points"]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "mesh_data output from SAM3DBody Process node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Video frames for debug overlay"
                }),
            },
            "optional": {
                # === Mask for overlay visualization ===
                "mask": ("MASK", {
                    "tooltip": "Foreground mask from SAM3 - shown on debug overlay"
                }),
                
                # === Frame Selection ===
                "frame_selection": (cls.FRAME_SELECTION, {
                    "default": "specific",
                    "tooltip": "Which frame to use for debug overlay"
                }),
                "specific_frame": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10000,
                    "tooltip": "Frame number when frame_selection='specific'"
                }),
                
                # === Detection Settings ===
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Detection confidence threshold (for reference)"
                }),
                
                # === Sensor Configuration ===
                "sensor_width_mm": ("FLOAT", {
                    "default": 36.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Assumed sensor width for focal length conversion to mm"
                }),
                
                # === Render Settings ===
                "render_mode": (cls.RENDER_MODES, {
                    "default": "solid",
                    "tooltip": "Mesh rendering mode: solid (best), wireframe, or points"
                }),
                "mesh_color": ("STRING", {
                    "default": "skin",
                    "tooltip": "Mesh color: skin, white, green, blue, or hex (#RRGGBB)"
                }),
                "overlay_opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Mesh overlay opacity"
                }),
                "show_skeleton": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show skeleton joints on overlay"
                }),
                "show_mask_outline": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show mask outline on overlay"
                }),
            }
        }
    
    RETURN_TYPES = ("INTRINSICS", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("intrinsics", "debug_overlay", "focal_length_mm", "status")
    FUNCTION = "extract_intrinsics"
    CATEGORY = "SAM3DBody2abc/Camera"
    
    def extract_intrinsics(
        self,
        mesh_data: Dict,
        images: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_selection: str = "specific",
        specific_frame: int = 1,
        bbox_threshold: float = 0.8,
        sensor_width_mm: float = 36.0,
        render_mode: str = "solid",
        mesh_color: str = "skin",
        overlay_opacity: float = 0.7,
        show_skeleton: bool = True,
        show_mask_outline: bool = False,
    ) -> Tuple[Dict, torch.Tensor, float, str]:
        """
        Extract intrinsics from SAM3DBody mesh_data.
        """
        
        try:
            num_frames = images.shape[0]
            H, W = images.shape[1], images.shape[2]
            
            # Select reference frame for overlay
            if frame_selection == "first":
                frame_idx = 0
            elif frame_selection == "middle":
                frame_idx = num_frames // 2
            elif frame_selection == "last":
                frame_idx = num_frames - 1
            else:  # specific
                frame_idx = min(specific_frame, num_frames - 1)
            
            print(f"\n{'='*60}")
            print(f"[IntrinsicsFromSAM3DBody] Extracting intrinsics from mesh_data")
            print(f"[IntrinsicsFromSAM3DBody] Video: {num_frames} frames, {W}x{H}")
            print(f"[IntrinsicsFromSAM3DBody] Debug overlay frame: {frame_idx}")
            print(f"[IntrinsicsFromSAM3DBody] Render mode: {render_mode}, pyrender: {PYRENDER_AVAILABLE}")
            print(f"[IntrinsicsFromSAM3DBody] mesh_data keys: {list(mesh_data.keys())}")
            print(f"{'='*60}")
            
            # Extract focal length
            focal_length_px = mesh_data.get("focal_length")
            if focal_length_px is None:
                print("[IntrinsicsFromSAM3DBody] WARNING: No focal_length in mesh_data, using fallback")
                focal_length_px = float(W)  # Fallback to image width
            else:
                if hasattr(focal_length_px, 'item'):
                    focal_length_px = float(focal_length_px.item())
                elif hasattr(focal_length_px, 'cpu'):
                    focal_length_px = float(focal_length_px.cpu().numpy())
                else:
                    focal_length_px = float(focal_length_px)
            
            print(f"[IntrinsicsFromSAM3DBody] Focal length: {focal_length_px:.1f}px")
            
            # Extract camera translation
            pred_cam_t = mesh_data.get("camera")
            if pred_cam_t is None:
                pred_cam_t = mesh_data.get("pred_cam_t")
            
            cam_t_list = None
            pred_cam_t_np = None
            if pred_cam_t is not None:
                pred_cam_t_np = to_numpy(pred_cam_t)
                if pred_cam_t_np is not None:
                    if pred_cam_t_np.ndim > 1:
                        pred_cam_t_np = pred_cam_t_np.flatten()[:3]
                    cam_t_list = pred_cam_t_np.tolist()
                    print(f"[IntrinsicsFromSAM3DBody] pred_cam_t: [{pred_cam_t_np[0]:.3f}, {pred_cam_t_np[1]:.3f}, {pred_cam_t_np[2]:.3f}]")
            
            # Convert to mm
            focal_length_mm = focal_length_px * sensor_width_mm / W
            
            # Compute FOV
            fov_x_deg = 2 * np.degrees(np.arctan(W / (2 * focal_length_px)))
            fov_y_deg = 2 * np.degrees(np.arctan(H / (2 * focal_length_px)))
            
            # Build INTRINSICS output
            intrinsics = {
                "focal_px": float(focal_length_px),
                "focal_mm": float(focal_length_mm),
                "sensor_width_mm": float(sensor_width_mm),
                "cx": float(W / 2),
                "cy": float(H / 2),
                "width": int(W),
                "height": int(H),
                "fov_x_deg": float(fov_x_deg),
                "fov_y_deg": float(fov_y_deg),
                "aspect_ratio": float(W / H),
                "source": "sam3dbody",
                "confidence": 0.85,
                "k_matrix": [
                    [focal_length_px, 0.0, W / 2],
                    [0.0, focal_length_px, H / 2],
                    [0.0, 0.0, 1.0]
                ],
                "pred_cam_t": cam_t_list,
                "reference_frame": frame_idx,
                "per_frame": None,
                "is_variable": False,
            }
            
            # Generate debug overlay
            debug_overlay = self._generate_overlay(
                images, frame_idx, mesh_data, focal_length_px, pred_cam_t_np,
                W, H, render_mode, mesh_color, overlay_opacity, show_skeleton,
                mask, show_mask_outline
            )
            
            status = f"SAM3DBody: {focal_length_mm:.1f}mm ({fov_x_deg:.1f}° FOV)"
            print(f"[IntrinsicsFromSAM3DBody] {status}")
            
            return (intrinsics, debug_overlay, float(focal_length_mm), status)
            
        except Exception as e:
            print(f"[IntrinsicsFromSAM3DBody] ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback values
            H, W = images.shape[1], images.shape[2]
            focal_length_px = float(W)
            focal_length_mm = focal_length_px * sensor_width_mm / W
            
            intrinsics = {
                "focal_px": focal_length_px,
                "focal_mm": focal_length_mm,
                "sensor_width_mm": float(sensor_width_mm),
                "cx": float(W / 2),
                "cy": float(H / 2),
                "width": int(W),
                "height": int(H),
                "fov_x_deg": 53.0,
                "fov_y_deg": 30.0,
                "aspect_ratio": float(W / H),
                "source": "fallback",
                "confidence": 0.0,
                "k_matrix": [[focal_length_px, 0.0, W/2], [0.0, focal_length_px, H/2], [0.0, 0.0, 1.0]],
                "pred_cam_t": None,
                "reference_frame": 0,
                "per_frame": None,
                "is_variable": False,
            }
            
            # Return first frame as fallback overlay
            debug_overlay = images[0:1].clone()
            
            return (intrinsics, debug_overlay, float(focal_length_mm), f"ERROR: {str(e)}")
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse color string to RGB tuple."""
        color_str = color_str.lower().strip()
        
        color_map = {
            "skin": (210, 180, 160),
            "white": (255, 255, 255),
            "green": (100, 255, 100),
            "blue": (100, 150, 255),
            "red": (255, 100, 100),
            "yellow": (255, 255, 100),
            "cyan": (100, 255, 255),
            "gray": (180, 180, 180),
        }
        
        if color_str in color_map:
            return color_map[color_str]
        
        # Try hex color
        if color_str.startswith("#") and len(color_str) == 7:
            try:
                r = int(color_str[1:3], 16)
                g = int(color_str[3:5], 16)
                b = int(color_str[5:7], 16)
                return (r, g, b)
            except:
                pass
        
        return color_map["skin"]  # Default
    
    def _generate_overlay(
        self,
        images: torch.Tensor,
        frame_idx: int,
        mesh_data: Dict,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
        render_mode: str,
        mesh_color: str,
        opacity: float,
        show_skeleton: bool,
        mask: Optional[torch.Tensor] = None,
        show_mask_outline: bool = False,
    ) -> torch.Tensor:
        """Generate debug overlay with mesh, skeleton, and mask outline."""
        
        print(f"[IntrinsicsFromSAM3DBody] _generate_overlay called")
        print(f"[IntrinsicsFromSAM3DBody]   render_mode: {render_mode}, mesh_color: {mesh_color}")
        print(f"[IntrinsicsFromSAM3DBody]   PYRENDER_AVAILABLE: {PYRENDER_AVAILABLE}")
        
        # Get the selected frame
        frame = images[frame_idx]
        
        # Convert to numpy
        frame_np = to_numpy(frame)
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame_np.astype(np.uint8)
        
        overlay = frame_np.copy()
        print(f"[IntrinsicsFromSAM3DBody]   Frame shape: {frame_np.shape}")
        
        # Get mesh data
        vertices = to_numpy(mesh_data.get("vertices"))
        faces = to_numpy(mesh_data.get("faces"))
        
        if vertices is not None and vertices.ndim == 3:
            vertices = vertices[0]
        if faces is not None and faces.ndim == 3:
            faces = faces[0]
        
        print(f"[IntrinsicsFromSAM3DBody]   Vertices: {vertices.shape if vertices is not None else None}")
        print(f"[IntrinsicsFromSAM3DBody]   Faces: {faces.shape if faces is not None else None}")
        
        # Render mesh
        if vertices is not None and faces is not None:
            color_rgb = self._parse_color(mesh_color)
            print(f"[IntrinsicsFromSAM3DBody]   Color RGB: {color_rgb}")
            
            if render_mode == "solid" and PYRENDER_AVAILABLE:
                print(f"[IntrinsicsFromSAM3DBody]   Using pyrender for solid rendering...")
                # Use pyrender for high-quality solid rendering
                mesh_render = self._render_mesh_pyrender(
                    vertices, faces, focal_length, pred_cam_t, W, H, color_rgb
                )
                if mesh_render is not None:
                    print(f"[IntrinsicsFromSAM3DBody]   pyrender output shape: {mesh_render.shape}")
                    # Blend mesh render with original image
                    mask_render = (mesh_render.sum(axis=2) > 0).astype(np.float32)
                    mask_render = np.stack([mask_render] * 3, axis=2)
                    overlay = (overlay * (1 - mask_render * opacity) + 
                              mesh_render * mask_render * opacity).astype(np.uint8)
                    print(f"[IntrinsicsFromSAM3DBody]   ✅ pyrender blend complete")
                else:
                    print(f"[IntrinsicsFromSAM3DBody]   ❌ pyrender returned None, falling back to OpenCV")
                    overlay = self._render_mesh_solid_cv(
                        overlay, vertices, faces, focal_length, pred_cam_t, W, H, color_rgb, opacity
                    )
            elif render_mode == "solid":
                print(f"[IntrinsicsFromSAM3DBody]   Using OpenCV fallback for solid rendering...")
                # Fallback solid rendering with OpenCV
                overlay = self._render_mesh_solid_cv(
                    overlay, vertices, faces, focal_length, pred_cam_t, W, H, color_rgb, opacity
                )
            elif render_mode == "wireframe":
                print(f"[IntrinsicsFromSAM3DBody]   Using wireframe rendering...")
                overlay = self._render_mesh_wireframe(
                    overlay, vertices, faces, focal_length, pred_cam_t, W, H, color_rgb, opacity
                )
            else:  # points
                print(f"[IntrinsicsFromSAM3DBody]   Using points rendering...")
                overlay = self._render_mesh_points(
                    overlay, vertices, focal_length, pred_cam_t, W, H, color_rgb, opacity
                )
        else:
            print(f"[IntrinsicsFromSAM3DBody]   ❌ No vertices or faces found, skipping mesh render")
        
        # Draw mask outline
        if show_mask_outline and mask is not None:
            overlay = self._draw_mask_outline(overlay, mask, frame_idx, W, H)
        
        # Draw skeleton
        if show_skeleton:
            joints = mesh_data.get("joint_coords")
            if joints is None:
                joints = mesh_data.get("joints")
            joints = to_numpy(joints)
            if joints is not None and joints.ndim == 3:
                joints = joints[0]
            
            if joints is not None:
                overlay = self._draw_skeleton(overlay, joints, focal_length, pred_cam_t, W, H)
        
        # Add info text
        focal_mm = focal_length * 36.0 / W
        fov = 2 * np.degrees(np.arctan(W / (2 * focal_length)))
        
        # Semi-transparent background for text
        text_bg = overlay.copy()
        cv2.rectangle(text_bg, (5, 5), (320, 130), (0, 0, 0), -1)
        overlay = cv2.addWeighted(text_bg, 0.5, overlay, 0.5, 0)
        
        cv2.putText(overlay, f"Focal: {focal_length:.0f}px ({focal_mm:.1f}mm)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"FOV: {fov:.1f} deg", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Source: SAM3DBody | Frame: {frame_idx}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if pred_cam_t is not None:
            cv2.putText(overlay, f"cam_t: [{pred_cam_t[0]:.2f}, {pred_cam_t[1]:.2f}, {pred_cam_t[2]:.2f}]", 
                       (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Convert back to tensor
        overlay_tensor = torch.from_numpy(overlay).float() / 255.0
        overlay_tensor = overlay_tensor.unsqueeze(0)
        
        return overlay_tensor
    
    def _render_mesh_pyrender(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
        color_rgb: Tuple[int, int, int],
    ) -> Optional[np.ndarray]:
        """Render mesh using pyrender for high-quality output."""
        
        print(f"[IntrinsicsFromSAM3DBody] _render_mesh_pyrender called")
        
        if not PYRENDER_AVAILABLE:
            print(f"[IntrinsicsFromSAM3DBody]   pyrender not available")
            return None
        
        try:
            # Apply camera translation to vertices
            verts = vertices.copy()
            
            print(f"[IntrinsicsFromSAM3DBody]   Original vertex bounds:")
            print(f"[IntrinsicsFromSAM3DBody]     X: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}]")
            print(f"[IntrinsicsFromSAM3DBody]     Y: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}]")
            print(f"[IntrinsicsFromSAM3DBody]     Z: [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]")
            
            if pred_cam_t is not None and len(pred_cam_t) >= 3:
                print(f"[IntrinsicsFromSAM3DBody]   Applying cam_t: [{pred_cam_t[0]:.3f}, {pred_cam_t[1]:.3f}, {pred_cam_t[2]:.3f}]")
                verts[:, 0] += pred_cam_t[0]
                verts[:, 1] += pred_cam_t[1]
                verts[:, 2] += pred_cam_t[2]
            else:
                print(f"[IntrinsicsFromSAM3DBody]   No cam_t, using default Z offset")
                verts[:, 2] += 5.0
            
            print(f"[IntrinsicsFromSAM3DBody]   Transformed vertex bounds:")
            print(f"[IntrinsicsFromSAM3DBody]     X: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}]")
            print(f"[IntrinsicsFromSAM3DBody]     Y: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}]")
            print(f"[IntrinsicsFromSAM3DBody]     Z: [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]")
            
            # Create trimesh with vertex colors
            mesh_color = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255]) / 255.0
            tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            tri_mesh.visual.vertex_colors = np.tile(mesh_color, (len(verts), 1))
            
            print(f"[IntrinsicsFromSAM3DBody]   Creating pyrender mesh...")
            
            # Create pyrender mesh with material for better visibility
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                roughnessFactor=0.8,
                baseColorFactor=[color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, 1.0]
            )
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material, smooth=True)
            
            # Create scene with background
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
            scene.add(mesh)
            
            # Add directional light from camera direction
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            light_pose = np.eye(4)
            scene.add(light, pose=light_pose)
            
            # Add point light for fill
            point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)
            point_light_pose = np.eye(4)
            point_light_pose[:3, 3] = [0, 0, 2]  # In front of camera
            scene.add(point_light, pose=point_light_pose)
            
            # Create camera with intrinsics
            # pyrender uses OpenGL convention: +X right, +Y up, -Z into screen
            camera = pyrender.IntrinsicsCamera(
                fx=focal_length, fy=focal_length,
                cx=W/2, cy=H/2,
                znear=0.01, zfar=100.0
            )
            
            # Camera at origin, looking down -Z
            camera_pose = np.eye(4)
            scene.add(camera, pose=camera_pose)
            
            print(f"[IntrinsicsFromSAM3DBody]   Rendering {W}x{H}...")
            
            # Render
            renderer = pyrender.OffscreenRenderer(W, H)
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            renderer.delete()
            
            # Check if anything was rendered
            non_zero_pixels = np.sum(color[:, :, 3] > 0)
            print(f"[IntrinsicsFromSAM3DBody]   Render complete, non-zero alpha pixels: {non_zero_pixels}")
            print(f"[IntrinsicsFromSAM3DBody]   Depth range: [{depth[depth > 0].min() if np.any(depth > 0) else 'none'}, {depth.max():.3f}]")
            
            if non_zero_pixels == 0:
                print(f"[IntrinsicsFromSAM3DBody]   ⚠️ No pixels rendered! Mesh may be outside view.")
                return None
            
            # Convert to BGR for OpenCV
            color_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
            
            return color_bgr
            
        except Exception as e:
            print(f"[IntrinsicsFromSAM3DBody] ❌ pyrender error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _render_mesh_solid_cv(
        self,
        overlay: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
        color_rgb: Tuple[int, int, int],
        opacity: float,
    ) -> np.ndarray:
        """Fallback solid mesh rendering using OpenCV."""
        
        # Project vertices to 2D
        pts_2d = self._project_points(vertices, focal_length, pred_cam_t, W, H)
        if pts_2d is None:
            return overlay
        
        # Calculate depth for sorting
        verts = vertices.copy()
        if pred_cam_t is not None and len(pred_cam_t) >= 3:
            verts[:, 2] += pred_cam_t[2]
        else:
            verts[:, 2] += 5.0
        
        # Sort faces by depth (back to front)
        face_depths = []
        for face in faces:
            if np.all(face < len(verts)):
                avg_depth = np.mean(verts[face, 2])
                face_depths.append((avg_depth, face))
        
        face_depths.sort(key=lambda x: -x[0])  # Far to near
        
        # Draw filled triangles
        mesh_layer = overlay.copy()
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB to BGR
        
        for _, face in face_depths:
            if np.all(face < len(pts_2d)):
                pts = pts_2d[face].astype(np.int32)
                
                # Check if triangle is within image bounds
                if (np.all(pts[:, 0] >= -W) and np.all(pts[:, 0] < 2*W) and
                    np.all(pts[:, 1] >= -H) and np.all(pts[:, 1] < 2*H)):
                    
                    # Simple shading based on face normal
                    v0, v1, v2 = vertices[face]
                    normal = np.cross(v1 - v0, v2 - v0)
                    normal = normal / (np.linalg.norm(normal) + 1e-8)
                    light_dir = np.array([0, 0, -1])
                    shade = max(0.3, abs(np.dot(normal, light_dir)))
                    
                    shaded_color = tuple(int(c * shade) for c in color_bgr)
                    cv2.fillPoly(mesh_layer, [pts.reshape(-1, 1, 2)], shaded_color)
        
        # Blend
        overlay = cv2.addWeighted(mesh_layer, opacity, overlay, 1 - opacity, 0)
        
        return overlay
    
    def _render_mesh_wireframe(
        self,
        overlay: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
        color_rgb: Tuple[int, int, int],
        opacity: float,
    ) -> np.ndarray:
        """Render mesh as wireframe."""
        
        pts_2d = self._project_points(vertices, focal_length, pred_cam_t, W, H)
        if pts_2d is None:
            return overlay
        
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        mesh_layer = overlay.copy()
        
        # Draw edges
        edges_drawn = set()
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))
                if edge not in edges_drawn and v1 < len(pts_2d) and v2 < len(pts_2d):
                    edges_drawn.add(edge)
                    pt1 = tuple(pts_2d[v1].astype(int))
                    pt2 = tuple(pts_2d[v2].astype(int))
                    if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                        0 <= pt2[0] < W and 0 <= pt2[1] < H):
                        cv2.line(mesh_layer, pt1, pt2, color_bgr, 1)
        
        overlay = cv2.addWeighted(mesh_layer, opacity, overlay, 1 - opacity, 0)
        return overlay
    
    def _render_mesh_points(
        self,
        overlay: np.ndarray,
        vertices: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
        color_rgb: Tuple[int, int, int],
        opacity: float,
    ) -> np.ndarray:
        """Render mesh as point cloud."""
        
        pts_2d = self._project_points(vertices, focal_length, pred_cam_t, W, H)
        if pts_2d is None:
            return overlay
        
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        mesh_layer = overlay.copy()
        
        for i in range(0, len(pts_2d), 3):  # Subsample
            x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(mesh_layer, (x, y), 1, color_bgr, -1)
        
        overlay = cv2.addWeighted(mesh_layer, opacity, overlay, 1 - opacity, 0)
        return overlay
    
    def _draw_mask_outline(
        self,
        overlay: np.ndarray,
        mask: torch.Tensor,
        frame_idx: int,
        W: int, H: int,
    ) -> np.ndarray:
        """Draw mask contour on overlay."""
        
        mask_np = to_numpy(mask)
        if mask_np is None:
            return overlay
        
        # Get correct frame
        if mask_np.ndim == 3:
            if frame_idx < mask_np.shape[0]:
                mask_frame = mask_np[frame_idx]
            else:
                mask_frame = mask_np[0]
        elif mask_np.ndim == 4:
            if frame_idx < mask_np.shape[0]:
                mask_frame = mask_np[frame_idx, 0]
            else:
                mask_frame = mask_np[0, 0]
        else:
            mask_frame = mask_np
        
        # Resize if needed
        if mask_frame.shape[0] != H or mask_frame.shape[1] != W:
            mask_frame = cv2.resize(mask_frame.astype(np.float32), (W, H))
        
        # Find and draw contours
        mask_binary = (mask_frame > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)  # Cyan
        
        return overlay
    
    def _draw_skeleton(
        self,
        overlay: np.ndarray,
        joints: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
    ) -> np.ndarray:
        """Draw skeleton joints and connections."""
        
        pts_2d = self._project_points(joints, focal_length, pred_cam_t, W, H)
        if pts_2d is None:
            return overlay
        
        # SMPL joint connections
        connections = [
            (0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8),
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
            (9, 13), (13, 16), (16, 18), (18, 20),
            (9, 14), (14, 17), (17, 19), (19, 21),
        ]
        
        # Draw connections
        for i, j in connections:
            if i < len(pts_2d) and j < len(pts_2d):
                pt1 = (int(pts_2d[i, 0]), int(pts_2d[i, 1]))
                pt2 = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
                if (0 <= pt1[0] < W and 0 <= pt1[1] < H and
                    0 <= pt2[0] < W and 0 <= pt2[1] < H):
                    cv2.line(overlay, pt1, pt2, (255, 255, 0), 2)
        
        # Draw joints
        for i, pt in enumerate(pts_2d):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < W and 0 <= y < H:
                color = (0, 0, 255) if i in [0, 3, 6, 9, 12, 15] else (255, 0, 0)
                cv2.circle(overlay, (x, y), 4, color, -1)
                cv2.circle(overlay, (x, y), 4, (255, 255, 255), 1)
        
        return overlay
    
    def _project_points(
        self,
        points_3d: np.ndarray,
        focal_length: float,
        pred_cam_t: Optional[np.ndarray],
        W: int, H: int,
    ) -> Optional[np.ndarray]:
        """Project 3D points to 2D using SAM3DBody's camera model."""
        
        if points_3d is None or len(points_3d) == 0:
            return None
        
        points = points_3d.copy()
        
        if pred_cam_t is not None and len(pred_cam_t) >= 3:
            tx, ty, tz = pred_cam_t[0], pred_cam_t[1], pred_cam_t[2]
            points[:, 0] = points[:, 0] + tx
            points[:, 1] = points[:, 1] + ty
            points[:, 2] = points[:, 2] + tz
        else:
            points[:, 2] = points[:, 2] + 5.0
        
        z = points[:, 2:3]
        z = np.maximum(z, 0.1)
        
        pts_2d = np.zeros((len(points), 2))
        pts_2d[:, 0] = points[:, 0] * focal_length / z[:, 0] + W / 2
        pts_2d[:, 1] = points[:, 1] * focal_length / z[:, 0] + H / 2
        
        return pts_2d


# Node registration
NODE_CLASS_MAPPINGS = {
    "IntrinsicsFromSAM3DBody": IntrinsicsFromSAM3DBody,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntrinsicsFromSAM3DBody": "📷 Intrinsics from SAM3DBody",
}

