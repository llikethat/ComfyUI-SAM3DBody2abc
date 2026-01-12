"""
Camera class and projection mathematics for multi-camera triangulation.

Handles:
- Camera intrinsic parameters (focal length, principal point)
- Camera extrinsic parameters (position, rotation)
- Pixel to ray conversion
- Coordinate system conversions
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


def rotation_matrix_from_euler(rotation_euler: List[float], order: str = "XYZ") -> np.ndarray:
    """
    Create rotation matrix from Euler angles (degrees).
    
    Args:
        rotation_euler: [rx, ry, rz] in degrees
        order: Rotation order (e.g., "XYZ", "ZXY")
    
    Returns:
        3x3 rotation matrix
    """
    rx, ry, rz = np.radians(rotation_euler)
    
    # Individual rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combine based on order
    matrices = {"X": Rx, "Y": Ry, "Z": Rz}
    
    R = np.eye(3)
    for axis in order:
        R = R @ matrices[axis]
    
    return R


def euler_from_rotation_matrix(R: np.ndarray, order: str = "XYZ") -> List[float]:
    """
    Extract Euler angles (degrees) from rotation matrix.
    
    Args:
        R: 3x3 rotation matrix
        order: Rotation order
    
    Returns:
        [rx, ry, rz] in degrees
    """
    if order == "XYZ":
        # For XYZ order
        if abs(R[0, 2]) < 0.9999:
            ry = np.arcsin(R[0, 2])
            rx = np.arctan2(-R[1, 2], R[2, 2])
            rz = np.arctan2(-R[0, 1], R[0, 0])
        else:
            # Gimbal lock
            ry = np.pi / 2 * np.sign(R[0, 2])
            rx = np.arctan2(R[1, 0], R[1, 1])
            rz = 0
    else:
        # Default fallback - may not be accurate for all orders
        ry = np.arcsin(np.clip(R[0, 2], -1, 1))
        rx = np.arctan2(-R[1, 2], R[2, 2])
        rz = np.arctan2(-R[0, 1], R[0, 0])
    
    return [np.degrees(rx), np.degrees(ry), np.degrees(rz)]


class Camera:
    """
    Camera model for triangulation.
    
    Handles intrinsic and extrinsic parameters, and provides
    methods for projecting 3D points to 2D and vice versa.
    """
    
    def __init__(
        self,
        name: str,
        position: List[float],
        rotation_euler: List[float],
        focal_length_mm: float,
        sensor_width_mm: float,
        sensor_height_mm: float,
        resolution: List[int],
        principal_point: Optional[List[float]] = None,
        rotation_order: str = "XYZ",
        distortion: Optional[Dict[str, float]] = None
    ):
        """
        Initialize camera.
        
        Args:
            name: Camera identifier
            position: [X, Y, Z] world position in meters
            rotation_euler: [rx, ry, rz] rotation in degrees
            focal_length_mm: Focal length in millimeters
            sensor_width_mm: Sensor width in millimeters
            sensor_height_mm: Sensor height in millimeters
            resolution: [width, height] in pixels
            principal_point: [cx, cy] optical center (default: image center)
            rotation_order: Euler rotation order (default: XYZ)
            distortion: Lens distortion coefficients {k1, k2, p1, p2}
        """
        self.name = name
        self.position = np.array(position, dtype=np.float64)
        self.rotation_euler = rotation_euler
        self.rotation_order = rotation_order
        
        # Resolution
        self.width = resolution[0]
        self.height = resolution[1]
        
        # Compute focal length in pixels
        self.focal_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.focal_px = focal_length_mm * self.width / sensor_width_mm
        
        # Principal point (optical center)
        if principal_point is not None:
            self.cx = principal_point[0]
            self.cy = principal_point[1]
        else:
            self.cx = self.width / 2
            self.cy = self.height / 2
        
        # Distortion coefficients
        self.distortion = distortion or {"k1": 0, "k2": 0, "p1": 0, "p2": 0}
        
        # Compute rotation matrix
        self.R = rotation_matrix_from_euler(rotation_euler, rotation_order)
        
        # Intrinsic matrix K
        self.K = np.array([
            [self.focal_px, 0, self.cx],
            [0, self.focal_px, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Inverse intrinsic matrix
        self.K_inv = np.linalg.inv(self.K)
        
        # Extrinsic matrix components
        # Camera looks along its local -Z axis (OpenGL/Maya convention)
        # R transforms from camera space to world space
        # t is the camera position in world space
        self.t = self.position
    
    def pixel_to_ray(self, pixel_x: float, pixel_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D pixel coordinate to 3D ray in world space.
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
        
        Returns:
            ray_origin: Camera position in world space
            ray_direction: Normalized direction vector in world space
        """
        # Undistort if needed (simplified - ignores distortion for now)
        x = pixel_x
        y = pixel_y
        
        # Pixel to normalized camera coordinates
        x_norm = (x - self.cx) / self.focal_px
        y_norm = (y - self.cy) / self.focal_px
        
        # Ray in camera space (looking along -Z, so Z = -1)
        # Y is flipped because image Y increases downward
        ray_cam = np.array([x_norm, -y_norm, -1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Transform to world space using rotation matrix
        # R transforms from camera to world
        ray_world = self.R @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        return self.position.copy(), ray_world
    
    def project_point(self, point_3d: np.ndarray) -> Tuple[float, float, bool]:
        """
        Project 3D world point to 2D pixel coordinates.
        
        Args:
            point_3d: [X, Y, Z] in world space
        
        Returns:
            pixel_x, pixel_y: 2D coordinates
            in_front: True if point is in front of camera
        """
        # Transform to camera space
        point_cam = self.R.T @ (point_3d - self.position)
        
        # Check if in front of camera (negative Z in camera space)
        in_front = point_cam[2] < 0
        
        if abs(point_cam[2]) < 1e-6:
            return 0, 0, False
        
        # Project to image plane
        x_norm = point_cam[0] / (-point_cam[2])
        y_norm = point_cam[1] / (-point_cam[2])
        
        # Convert to pixels (flip Y back)
        pixel_x = x_norm * self.focal_px + self.cx
        pixel_y = -y_norm * self.focal_px + self.cy
        
        return pixel_x, pixel_y, in_front
    
    def is_point_visible(self, point_3d: np.ndarray, margin: int = 0) -> bool:
        """
        Check if a 3D point is visible in this camera.
        
        Args:
            point_3d: [X, Y, Z] in world space
            margin: Pixel margin from image edge
        
        Returns:
            True if point projects within image bounds and is in front
        """
        px, py, in_front = self.project_point(point_3d)
        
        if not in_front:
            return False
        
        if px < margin or px >= self.width - margin:
            return False
        if py < margin or py >= self.height - margin:
            return False
        
        return True
    
    def get_view_direction(self) -> np.ndarray:
        """Get the camera's viewing direction in world space."""
        # Camera looks along -Z in camera space
        return self.R @ np.array([0, 0, -1])
    
    def to_dict(self) -> Dict:
        """Convert camera to dictionary for serialization."""
        return {
            "name": self.name,
            "position": self.position.tolist(),
            "rotation_euler": self.rotation_euler,
            "rotation_order": self.rotation_order,
            "focal_length_mm": self.focal_mm,
            "sensor_width_mm": self.sensor_width_mm,
            "sensor_height_mm": self.sensor_height_mm,
            "resolution": [self.width, self.height],
            "principal_point": [self.cx, self.cy],
            "distortion": self.distortion
        }
    
    @classmethod
    def from_dict(cls, data: Dict, name: str = "camera") -> "Camera":
        """Create camera from dictionary."""
        return cls(
            name=data.get("name", name),
            position=data["position"],
            rotation_euler=data["rotation_euler"],
            focal_length_mm=data["focal_length_mm"],
            sensor_width_mm=data.get("sensor_width_mm", 36.0),
            sensor_height_mm=data.get("sensor_height_mm", 24.0),
            resolution=data["resolution"],
            principal_point=data.get("principal_point"),
            rotation_order=data.get("rotation_order", "XYZ"),
            distortion=data.get("distortion")
        )
    
    def __repr__(self) -> str:
        return (
            f"Camera('{self.name}', pos={self.position.tolist()}, "
            f"rot={self.rotation_euler}, focal={self.focal_mm}mm)"
        )


def convert_coordinate_system(
    position: List[float],
    from_system: str,
    to_system: str
) -> List[float]:
    """
    Convert position between coordinate systems.
    
    Args:
        position: [X, Y, Z] coordinates
        from_system: Source system ("Y-up" or "Z-up")
        to_system: Target system ("Y-up" or "Z-up")
    
    Returns:
        Converted [X, Y, Z] coordinates
    """
    if from_system == to_system:
        return position
    
    x, y, z = position
    
    if from_system == "Z-up" and to_system == "Y-up":
        # Z-up → Y-up: swap Y and Z, negate new Z
        return [x, z, -y]
    
    elif from_system == "Y-up" and to_system == "Z-up":
        # Y-up → Z-up: swap Y and Z, negate new Y
        return [x, -z, y]
    
    return position


def compute_baseline(camera_a: Camera, camera_b: Camera) -> float:
    """
    Compute baseline distance between two cameras.
    
    Args:
        camera_a: First camera
        camera_b: Second camera
    
    Returns:
        Distance in meters
    """
    return float(np.linalg.norm(camera_a.position - camera_b.position))


def compute_angle_between_cameras(camera_a: Camera, camera_b: Camera) -> float:
    """
    Compute angle between camera viewing directions.
    
    Args:
        camera_a: First camera
        camera_b: Second camera
    
    Returns:
        Angle in degrees
    """
    dir_a = camera_a.get_view_direction()
    dir_b = camera_b.get_view_direction()
    
    cos_angle = np.clip(np.dot(dir_a, dir_b), -1, 1)
    return float(np.degrees(np.arccos(cos_angle)))
