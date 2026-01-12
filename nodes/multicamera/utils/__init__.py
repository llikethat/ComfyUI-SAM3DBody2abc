"""
Multi-camera triangulation utilities.
"""

import os
import sys

# Add current directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from .camera import (
        Camera,
        rotation_matrix_from_euler,
        euler_from_rotation_matrix,
        convert_coordinate_system,
        compute_baseline,
        compute_angle_between_cameras
    )
except ImportError:
    from camera import (
        Camera,
        rotation_matrix_from_euler,
        euler_from_rotation_matrix,
        convert_coordinate_system,
        compute_baseline,
        compute_angle_between_cameras
    )

try:
    from .triangulation import (
        triangulate_two_rays,
        triangulate_multi_ray,
        triangulate_point_from_cameras,
        compute_reprojection_error,
        estimate_triangulation_quality
    )
except ImportError:
    from triangulation import (
        triangulate_two_rays,
        triangulate_multi_ray,
        triangulate_point_from_cameras,
        compute_reprojection_error,
        estimate_triangulation_quality
    )

try:
    from .visualization import (
        create_multicamera_debug_view,
        create_topview_with_cameras,
        create_error_graph
    )
except ImportError:
    from visualization import (
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
