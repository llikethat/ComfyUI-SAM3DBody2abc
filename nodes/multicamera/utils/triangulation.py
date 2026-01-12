"""
Triangulation algorithms for multi-camera 3D reconstruction.

Provides methods for:
- Two-ray triangulation (2 cameras)
- Multi-ray triangulation (3+ cameras)
- Error estimation
"""

import numpy as np
from typing import Tuple, List, Optional
from .camera import Camera


def triangulate_two_rays(
    origin_a: np.ndarray,
    dir_a: np.ndarray,
    origin_b: np.ndarray,
    dir_b: np.ndarray
) -> Tuple[Optional[np.ndarray], float]:
    """
    Find the 3D point closest to both rays.
    
    Uses the midpoint of the shortest line segment connecting two skew lines.
    
    Args:
        origin_a: Origin of ray A [X, Y, Z]
        dir_a: Direction of ray A (will be normalized)
        origin_b: Origin of ray B [X, Y, Z]
        dir_b: Direction of ray B (will be normalized)
    
    Returns:
        point_3d: Triangulated 3D position, or None if rays are parallel
        error: Distance between the closest points on each ray (reprojection error)
    """
    # Normalize direction vectors
    d1 = dir_a / np.linalg.norm(dir_a)
    d2 = dir_b / np.linalg.norm(dir_b)
    
    # Vector between origins
    w0 = origin_a - origin_b
    
    # Dot products
    a = np.dot(d1, d1)  # Always 1 if normalized
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)  # Always 1 if normalized
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    
    # Denominator
    denom = a * c - b * b
    
    # Check for parallel rays
    if abs(denom) < 1e-10:
        return None, float('inf')
    
    # Solve for parameters along each ray
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    
    # Closest points on each ray
    point_a = origin_a + s * d1
    point_b = origin_b + t * d2
    
    # Midpoint as triangulated position
    point_3d = (point_a + point_b) / 2
    
    # Error is distance between closest points
    error = float(np.linalg.norm(point_a - point_b))
    
    return point_3d, error


def triangulate_multi_ray(
    rays: List[Tuple[np.ndarray, np.ndarray]],
    weights: Optional[List[float]] = None
) -> Tuple[Optional[np.ndarray], float]:
    """
    Triangulate from multiple rays using weighted least squares.
    
    Finds the point that minimizes the sum of squared distances to all rays.
    
    Args:
        rays: List of (origin, direction) tuples
        weights: Optional confidence weights for each ray
    
    Returns:
        point_3d: Best fit 3D position
        error: RMS distance to all rays
    """
    if len(rays) < 2:
        return None, float('inf')
    
    if len(rays) == 2:
        return triangulate_two_rays(
            rays[0][0], rays[0][1],
            rays[1][0], rays[1][1]
        )
    
    # Default weights
    if weights is None:
        weights = [1.0] * len(rays)
    
    # Build linear system
    # For each ray, we create constraints perpendicular to the ray direction
    # Point P satisfies: (P - origin) · perp = 0 for directions perpendicular to ray
    
    A = []
    b = []
    
    for (origin, direction), weight in zip(rays, weights):
        d = direction / np.linalg.norm(direction)
        w = np.sqrt(weight)
        
        # Create two perpendicular directions to the ray
        if abs(d[0]) < 0.9:
            perp1 = np.cross(d, np.array([1, 0, 0]))
        else:
            perp1 = np.cross(d, np.array([0, 1, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(d, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # Add weighted constraints
        A.append(w * perp1)
        b.append(w * np.dot(perp1, origin))
        A.append(w * perp2)
        b.append(w * np.dot(perp2, origin))
    
    A = np.array(A)
    b = np.array(b)
    
    # Solve least squares: A @ point = b
    try:
        point_3d, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, float('inf')
    
    # Calculate RMS error (average distance to rays)
    total_error = 0.0
    for origin, direction in rays:
        d = direction / np.linalg.norm(direction)
        # Project point onto ray and find perpendicular distance
        v = point_3d - origin
        proj_length = np.dot(v, d)
        closest_on_ray = origin + proj_length * d
        dist = np.linalg.norm(point_3d - closest_on_ray)
        total_error += dist ** 2
    
    rms_error = float(np.sqrt(total_error / len(rays)))
    
    return point_3d, rms_error


def triangulate_point_from_cameras(
    cameras: List[Camera],
    pixel_coords: List[Tuple[float, float]],
    confidences: Optional[List[float]] = None
) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Triangulate a 3D point from multiple camera observations.
    
    Args:
        cameras: List of Camera objects
        pixel_coords: List of (x, y) pixel coordinates, one per camera
                     Use None for cameras that don't see the point
        confidences: Optional confidence values for each observation
    
    Returns:
        point_3d: Triangulated position or None
        error: Reprojection error in meters
        num_cameras: Number of cameras used for triangulation
    """
    rays = []
    weights = []
    
    for i, (camera, coords) in enumerate(zip(cameras, pixel_coords)):
        if coords is None:
            continue
        
        pixel_x, pixel_y = coords
        
        # Skip if outside image bounds
        if pixel_x < 0 or pixel_x >= camera.width:
            continue
        if pixel_y < 0 or pixel_y >= camera.height:
            continue
        
        # Get ray from camera through pixel
        origin, direction = camera.pixel_to_ray(pixel_x, pixel_y)
        rays.append((origin, direction))
        
        # Get weight
        if confidences is not None and i < len(confidences) and confidences[i] is not None:
            weights.append(confidences[i])
        else:
            weights.append(1.0)
    
    num_cameras = len(rays)
    
    if num_cameras < 2:
        return None, float('inf'), num_cameras
    
    # Triangulate
    if num_cameras == 2:
        point_3d, error = triangulate_two_rays(
            rays[0][0], rays[0][1],
            rays[1][0], rays[1][1]
        )
    else:
        point_3d, error = triangulate_multi_ray(rays, weights)
    
    return point_3d, error, num_cameras


def compute_reprojection_error(
    point_3d: np.ndarray,
    cameras: List[Camera],
    pixel_coords: List[Tuple[float, float]]
) -> List[float]:
    """
    Compute reprojection error for each camera.
    
    Args:
        point_3d: Triangulated 3D point
        cameras: List of Camera objects
        pixel_coords: Original 2D observations (None for cameras that didn't see point)
    
    Returns:
        List of reprojection errors in pixels (inf for cameras without observation)
    """
    errors = []
    
    for camera, coords in zip(cameras, pixel_coords):
        if coords is None:
            errors.append(float('inf'))
            continue
        
        # Project 3D point back to camera
        proj_x, proj_y, in_front = camera.project_point(point_3d)
        
        if not in_front:
            errors.append(float('inf'))
            continue
        
        # Compute pixel distance
        obs_x, obs_y = coords
        error = np.sqrt((proj_x - obs_x)**2 + (proj_y - obs_y)**2)
        errors.append(float(error))
    
    return errors


def estimate_triangulation_quality(
    cameras: List[Camera],
    point_3d: np.ndarray
) -> dict:
    """
    Estimate the quality of triangulation at a given 3D point.
    
    Args:
        cameras: List of cameras
        point_3d: 3D point to evaluate
    
    Returns:
        Dictionary with quality metrics
    """
    # Compute viewing angles from each camera
    angles = []
    for camera in cameras:
        to_point = point_3d - camera.position
        to_point = to_point / np.linalg.norm(to_point)
        view_dir = camera.get_view_direction()
        
        cos_angle = np.dot(to_point, view_dir)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        angles.append(angle)
    
    # Compute angle between rays from different cameras
    ray_angles = []
    for i in range(len(cameras)):
        for j in range(i + 1, len(cameras)):
            ray_i = point_3d - cameras[i].position
            ray_j = point_3d - cameras[j].position
            ray_i = ray_i / np.linalg.norm(ray_i)
            ray_j = ray_j / np.linalg.norm(ray_j)
            
            cos_angle = np.dot(ray_i, ray_j)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            ray_angles.append(angle)
    
    # Quality score based on ray angles
    # Best triangulation is when rays are perpendicular (90°)
    # Poor when rays are nearly parallel (0° or 180°)
    if ray_angles:
        best_angle = max(min(a, 180 - a) for a in ray_angles)  # Distance from parallel
        quality_score = best_angle / 90.0  # 1.0 = perpendicular, 0.0 = parallel
    else:
        quality_score = 0.0
    
    return {
        "viewing_angles": angles,
        "ray_angles": ray_angles,
        "quality_score": min(1.0, quality_score),
        "best_ray_angle": max(ray_angles) if ray_angles else 0.0
    }
