"""
Multi-camera triangulation utilities.
"""

from .camera import (
    Camera,
    rotation_matrix_from_euler,
    euler_from_rotation_matrix,
    convert_coordinate_system,
    compute_baseline,
    compute_angle_between_cameras
)

from .triangulation import (
    triangulate_two_rays,
    triangulate_multi_ray,
    triangulate_point_from_cameras,
    compute_reprojection_error,
    estimate_triangulation_quality
)

from .visualization import (
    create_multicamera_debug_view,
    create_topview_with_cameras,
    create_error_graph
)

__all__ = [
    # Camera
    "Camera",
    "rotation_matrix_from_euler",
    "euler_from_rotation_matrix",
    "convert_coordinate_system",
    "compute_baseline",
    "compute_angle_between_cameras",
    
    # Triangulation
    "triangulate_two_rays",
    "triangulate_multi_ray",
    "triangulate_point_from_cameras",
    "compute_reprojection_error",
    "estimate_triangulation_quality",
    
    # Visualization
    "create_multicamera_debug_view",
    "create_topview_with_cameras",
    "create_error_graph",
]
