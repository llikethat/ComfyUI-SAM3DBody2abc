"""
Blender Script: Export Animated FBX from Mesh Sequence JSON
Creates animated mesh using vertex keyframes + joint skeleton.

Supports two skeleton animation modes:
- Rotations: Uses true joint rotation matrices from MHR model (recommended for retargeting)
- Positions: Uses joint position offsets (legacy mode)

Usage: blender --background --python blender_animated_fbx.py -- input.json output.fbx [up_axis]

Args:
    input.json: JSON with frames data
    output.fbx: Output FBX path
    up_axis: Y, Z, -Y, or -Z (default: Y)
"""

import bpy
import json
import sys
import os
from mathutils import Vector, Matrix, Euler, Quaternion
import math


# =============================================================================
# EMBEDDED LOGGER - Verbosity-controlled logging for Blender script
# =============================================================================
from datetime import datetime

class LogLevel:
    SILENT = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    STATUS = 4
    DEBUG = 5

class Log:
    """Simple logger with verbosity control and timestamps."""
    
    def __init__(self, level=LogLevel.INFO):
        self.level = level
        self.prefix = "Blender"
    
    def _ts(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    def error(self, msg): 
        if self.level >= LogLevel.ERROR: print(f"[{self._ts()}] [{self.prefix}] ERROR: {msg}")
    
    def warn(self, msg): 
        if self.level >= LogLevel.WARN: print(f"[{self._ts()}] [{self.prefix}] WARN: {msg}")
    
    def info(self, msg): 
        if self.level >= LogLevel.INFO: print(f"[{self._ts()}] [{self.prefix}] {msg}")
    
    def status(self, msg): 
        if self.level >= LogLevel.STATUS: print(f"[{self._ts()}] [{self.prefix}] {msg}")
    
    def debug(self, msg): 
        if self.level >= LogLevel.DEBUG: print(f"[{self._ts()}] [{self.prefix}] DEBUG: {msg}")
    
    def progress(self, current, total, task="", interval=10):
        """Log progress at intervals."""
        if self.level < LogLevel.STATUS: return
        if current == 0 or current == total - 1 or (current + 1) % interval == 0:
            pct = (current + 1) / total * 100
            msg = f"{task}: {current + 1}/{total} ({pct:.0f}%)" if task else f"Progress: {current + 1}/{total}"
            print(f"[{self._ts()}] [{self.prefix}] {msg}")

# Global logger - set level via environment or default to INFO
_log_level = os.environ.get("SAM3DBODY_LOG_LEVEL", "INFO").upper()
_level_map = {"SILENT": 0, "ERROR": 1, "WARN": 2, "INFO": 3, "STATUS": 4, "DEBUG": 5}
log = Log(level=_level_map.get(_log_level, LogLevel.INFO))
# =============================================================================


def smooth_array(values, window):
    """Moving average smoothing for camera animation to reduce jitter.
    
    Args:
        values: List of float values
        window: Smoothing window size (odd number works best)
    
    Returns:
        Smoothed list of values
    """
    if window <= 1 or len(values) < window:
        return values
    
    result = []
    half = window // 2
    
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        avg = sum(values[start:end]) / (end - start)
        result.append(avg)
    
    return result


def smooth_camera_data(camera_rotations, window=9):
    """
    Pre-smooth camera rotation/translation data to avoid jerky root compensation.
    
    Uses Gaussian-weighted smoothing for better results than simple moving average.
    
    Args:
        camera_rotations: List of dicts with pan, tilt, roll, tx, ty, tz
        window: Smoothing window size (default 9 for strong smoothing)
    
    Returns:
        Smoothed list of camera rotation dicts
    """
    if not camera_rotations or window <= 1:
        return camera_rotations
    
    n = len(camera_rotations)
    if n < window:
        window = max(3, n)
    
    # Extract channels
    pans = [r.get("pan", 0.0) for r in camera_rotations]
    tilts = [r.get("tilt", 0.0) for r in camera_rotations]
    rolls = [r.get("roll", 0.0) for r in camera_rotations]
    txs = [r.get("tx", 0.0) for r in camera_rotations]
    tys = [r.get("ty", 0.0) for r in camera_rotations]
    tzs = [r.get("tz", 0.0) for r in camera_rotations]
    
    # Create Gaussian-like weights for smoother result
    half = window // 2
    weights = []
    for i in range(-half, half + 1):
        # Gaussian weight: exp(-x^2 / (2*sigma^2))
        sigma = half / 2.0
        w = math.exp(-(i * i) / (2 * sigma * sigma))
        weights.append(w)
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    
    def gaussian_smooth(values):
        """Apply Gaussian smoothing to an array."""
        result = []
        for i in range(len(values)):
            total = 0.0
            total_weight = 0.0
            for j, w in enumerate(weights):
                idx = i + j - half
                if 0 <= idx < len(values):
                    total += values[idx] * w
                    total_weight += w
            result.append(total / total_weight if total_weight > 0 else values[i])
        return result
    
    # Smooth all channels
    smooth_pans = gaussian_smooth(pans)
    smooth_tilts = gaussian_smooth(tilts)
    smooth_rolls = gaussian_smooth(rolls)
    smooth_txs = gaussian_smooth(txs)
    smooth_tys = gaussian_smooth(tys)
    smooth_tzs = gaussian_smooth(tzs)
    
    # Rebuild camera data
    result = []
    for i in range(n):
        result.append({
            "frame": camera_rotations[i].get("frame", i),
            "pan": smooth_pans[i],
            "tilt": smooth_tilts[i],
            "roll": smooth_rolls[i],
            "tx": smooth_txs[i],
            "ty": smooth_tys[i],
            "tz": smooth_tzs[i],
        })
    
    return result


def kalman_filter_camera_data(camera_rotations, process_noise=0.01, measurement_noise=0.1):
    """
    Apply Kalman filter to camera rotation/translation data for smooth trajectories.
    
    Kalman filter is optimal for:
    - Real-time or sequential data
    - Known motion model (constant velocity assumption)
    - Balancing between measurement trust and prediction
    
    Args:
        camera_rotations: List of dicts with pan, tilt, roll, tx, ty, tz
        process_noise: How much we expect the state to change (lower = smoother)
        measurement_noise: How noisy we think measurements are (higher = trust prediction more)
    
    Returns:
        Filtered camera rotation list
    """
    if not camera_rotations or len(camera_rotations) < 2:
        return camera_rotations
    
    import numpy as np
    
    n = len(camera_rotations)
    log.info(f"Applying Kalman filter to {n} camera frames (Q={process_noise}, R={measurement_noise})...")
    
    # State: [pan, tilt, roll, tx, ty, tz, d_pan, d_tilt, d_roll, d_tx, d_ty, d_tz]
    # We track both position and velocity for smoother predictions
    state_dim = 12
    meas_dim = 6
    
    # Initialize state
    x = np.zeros(state_dim)
    x[0] = camera_rotations[0].get("pan", 0.0)
    x[1] = camera_rotations[0].get("tilt", 0.0)
    x[2] = camera_rotations[0].get("roll", 0.0)
    x[3] = camera_rotations[0].get("tx", 0.0)
    x[4] = camera_rotations[0].get("ty", 0.0)
    x[5] = camera_rotations[0].get("tz", 0.0)
    
    # State transition matrix (constant velocity model)
    dt = 1.0  # Frame time step
    F = np.eye(state_dim)
    for i in range(6):
        F[i, i + 6] = dt  # Position += velocity * dt
    
    # Measurement matrix (we only observe position, not velocity)
    H = np.zeros((meas_dim, state_dim))
    for i in range(meas_dim):
        H[i, i] = 1.0
    
    # Process noise covariance
    Q = np.eye(state_dim) * process_noise
    # Higher noise for velocity components (less certain about acceleration)
    for i in range(6, 12):
        Q[i, i] = process_noise * 10
    
    # Measurement noise covariance
    R = np.eye(meas_dim) * measurement_noise
    
    # Initial state covariance
    P = np.eye(state_dim) * 1.0
    
    # Store filtered results
    filtered = []
    
    for frame_idx, cam_rot in enumerate(camera_rotations):
        # Measurement
        z = np.array([
            cam_rot.get("pan", 0.0),
            cam_rot.get("tilt", 0.0),
            cam_rot.get("roll", 0.0),
            cam_rot.get("tx", 0.0),
            cam_rot.get("ty", 0.0),
            cam_rot.get("tz", 0.0),
        ])
        
        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        # Update
        y = z - H @ x_pred  # Innovation
        S = H @ P_pred @ H.T + R  # Innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        x = x_pred + K @ y
        P = (np.eye(state_dim) - K @ H) @ P_pred
        
        # Store filtered result
        filtered.append({
            "frame": cam_rot.get("frame", frame_idx),
            "pan": float(x[0]),
            "tilt": float(x[1]),
            "roll": float(x[2]),
            "tx": float(x[3]),
            "ty": float(x[4]),
            "tz": float(x[5]),
        })
    
    log.info(f"Kalman filter complete")
    return filtered


def spline_fit_camera_data(camera_rotations, smoothing_factor=0.5):
    """
    Apply cubic spline fitting to camera rotation/translation data.
    
    Spline fitting is optimal for:
    - Batch processing (all data available)
    - Creating smooth continuous curves
    - Preserving overall motion character while removing high-frequency noise
    
    Args:
        camera_rotations: List of dicts with pan, tilt, roll, tx, ty, tz
        smoothing_factor: 0.0 = interpolation (passes through all points)
                         1.0 = heavy smoothing (may deviate from points)
    
    Returns:
        Spline-fitted camera rotation list
    """
    if not camera_rotations or len(camera_rotations) < 4:
        return camera_rotations
    
    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError:
        log.info("scipy not available for spline fitting, using Gaussian smoothing fallback")
        return smooth_camera_data(camera_rotations, window=9)
    
    import numpy as np
    
    n = len(camera_rotations)
    log.info(f"Applying spline fitting to {n} camera frames (smoothing={smoothing_factor})...")
    
    # Extract channels
    frames = np.array([r.get("frame", i) for i, r in enumerate(camera_rotations)])
    pans = np.array([r.get("pan", 0.0) for r in camera_rotations])
    tilts = np.array([r.get("tilt", 0.0) for r in camera_rotations])
    rolls = np.array([r.get("roll", 0.0) for r in camera_rotations])
    txs = np.array([r.get("tx", 0.0) for r in camera_rotations])
    tys = np.array([r.get("ty", 0.0) for r in camera_rotations])
    tzs = np.array([r.get("tz", 0.0) for r in camera_rotations])
    
    # Compute smoothing parameter based on data variance
    # scipy's s parameter is total squared error allowed
    def fit_channel(y):
        variance = np.var(y)
        s = smoothing_factor * variance * len(y)
        try:
            spline = UnivariateSpline(frames, y, s=s, k=3)
            return spline(frames)
        except Exception as e:
            log.info(f"Spline fit failed: {e}, using original")
            return y
    
    smooth_pans = fit_channel(pans)
    smooth_tilts = fit_channel(tilts)
    smooth_rolls = fit_channel(rolls)
    smooth_txs = fit_channel(txs)
    smooth_tys = fit_channel(tys)
    smooth_tzs = fit_channel(tzs)
    
    # Build result
    filtered = []
    for i in range(n):
        filtered.append({
            "frame": camera_rotations[i].get("frame", i),
            "pan": float(smooth_pans[i]),
            "tilt": float(smooth_tilts[i]),
            "roll": float(smooth_rolls[i]),
            "tx": float(smooth_txs[i]),
            "ty": float(smooth_tys[i]),
            "tz": float(smooth_tzs[i]),
        })
    
    log.info(f"Spline fitting complete")
    return filtered


def smooth_camera_data_combined(camera_rotations, method="kalman", **kwargs):
    """
    Combined camera data smoothing dispatcher.
    
    Args:
        camera_rotations: List of camera rotation dicts
        method: "kalman", "spline", "gaussian", or "none"
        **kwargs: Method-specific parameters
    
    Returns:
        Smoothed camera rotation list
    """
    if not camera_rotations or method == "none":
        return camera_rotations
    
    if method == "kalman":
        process_noise = kwargs.get("process_noise", 0.01)
        measurement_noise = kwargs.get("measurement_noise", 0.1)
        return kalman_filter_camera_data(camera_rotations, process_noise, measurement_noise)
    
    elif method == "spline":
        smoothing_factor = kwargs.get("smoothing_factor", 0.5)
        return spline_fit_camera_data(camera_rotations, smoothing_factor)
    
    elif method == "gaussian":
        window = kwargs.get("window", 9)
        return smooth_camera_data(camera_rotations, window)
    
    else:
        log.info(f"Unknown smoothing method: {method}, using Kalman")
        return kalman_filter_camera_data(camera_rotations)


def bake_camera_to_geometry(frames, solved_camera_rotations, up_axis="Y", 
                            smoothing_method="kalman", smoothing_params=None):
    """
    Bake camera motion into geometry by applying inverse camera transform to all vertices/joints.
    
    This approach:
    1. FIRST applies smoothing (Kalman filter or spline fitting) to camera data
    2. Takes smoothed camera rotation/translation
    3. Computes inverse transform for each frame
    4. Applies inverse transform to vertices and joint positions
    5. Result: geometry moves as if camera was static, camera can be exported as static
    
    Args:
        frames: List of frame data dicts with vertices, joint_coords, joint_rotations
        solved_camera_rotations: List of camera rotation dicts with pan, tilt, roll, tx, ty, tz
        up_axis: Coordinate system up axis
        smoothing_method: "kalman", "spline", "gaussian", or "none"
        smoothing_params: Dict of method-specific parameters
    
    Returns:
        Modified frames list with baked transforms
    """
    import numpy as np
    import copy
    
    if not solved_camera_rotations or len(solved_camera_rotations) == 0:
        log.info("No camera rotations to bake - returning original frames")
        return frames
    
    # Apply smoothing FIRST before baking
    if smoothing_params is None:
        smoothing_params = {}
    
    smoothed_camera = smooth_camera_data_combined(
        solved_camera_rotations, 
        method=smoothing_method,
        **smoothing_params
    )
    
    log.info(f"Baking camera motion into geometry ({len(frames)} frames, smoothing={smoothing_method})...")
    
    baked_frames = []
    
    for frame_idx, frame_data in enumerate(frames):
        # Deep copy frame to avoid modifying original
        baked_frame = copy.deepcopy(frame_data)
        
        # Get SMOOTHED camera rotation for this frame
        if frame_idx < len(smoothed_camera):
            cam_rot = smoothed_camera[frame_idx]
        else:
            # Use last available rotation
            cam_rot = smoothed_camera[-1]
        
        pan = cam_rot.get("pan", 0.0)
        tilt = cam_rot.get("tilt", 0.0)
        roll = cam_rot.get("roll", 0.0)
        tx = cam_rot.get("tx", 0.0)
        ty = cam_rot.get("ty", 0.0)
        tz = cam_rot.get("tz", 0.0)
        
        # Build rotation matrix from Euler angles
        # For Y-up: pan around Y, tilt around X, roll around Z
        # Using rotation order YXZ (common for cameras)
        
        # Individual rotation matrices
        cos_pan, sin_pan = math.cos(pan), math.sin(pan)
        cos_tilt, sin_tilt = math.cos(tilt), math.sin(tilt)
        cos_roll, sin_roll = math.cos(roll), math.sin(roll)
        
        # Rotation around Y (pan)
        Ry = np.array([
            [cos_pan, 0, sin_pan],
            [0, 1, 0],
            [-sin_pan, 0, cos_pan]
        ])
        
        # Rotation around X (tilt)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_tilt, -sin_tilt],
            [0, sin_tilt, cos_tilt]
        ])
        
        # Rotation around Z (roll)
        Rz = np.array([
            [cos_roll, -sin_roll, 0],
            [sin_roll, cos_roll, 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = Ry * Rx * Rz (YXZ order)
        if up_axis == "Y":
            R_cam = Ry @ Rx @ Rz
        elif up_axis == "Z":
            # For Z-up, pan around Z, tilt around X
            Rz_pan = np.array([
                [cos_pan, -sin_pan, 0],
                [sin_pan, cos_pan, 0],
                [0, 0, 1]
            ])
            R_cam = Rz_pan @ Rx @ Rz
        else:
            R_cam = Ry @ Rx @ Rz
        
        # Inverse rotation: R^T
        R_inv = R_cam.T
        
        # Camera translation vector
        t_cam = np.array([tx, ty, tz])
        
        # Inverse translation: -R^T * t
        t_inv = -R_inv @ t_cam
        
        # Apply inverse transform to vertices
        vertices = baked_frame.get("vertices")
        if vertices is not None:
            new_vertices = []
            for v in vertices:
                v_arr = np.array(v)
                v_transformed = R_inv @ v_arr + t_inv
                new_vertices.append(v_transformed.tolist())
            baked_frame["vertices"] = new_vertices
        
        # Apply inverse transform to joint coordinates
        joint_coords = baked_frame.get("joint_coords")
        if joint_coords is not None:
            new_joints = []
            for j in joint_coords:
                j_arr = np.array(j)
                j_transformed = R_inv @ j_arr + t_inv
                new_joints.append(j_transformed.tolist())
            baked_frame["joint_coords"] = new_joints
        
        # Apply inverse rotation to joint rotation matrices
        joint_rotations = baked_frame.get("joint_rotations")
        if joint_rotations is not None:
            new_rotations = []
            for rot_3x3 in joint_rotations:
                if rot_3x3 is not None:
                    rot_arr = np.array(rot_3x3)
                    # New rotation = R_inv * original_rotation
                    rot_transformed = R_inv @ rot_arr
                    new_rotations.append(rot_transformed.tolist())
                else:
                    new_rotations.append(None)
            baked_frame["joint_rotations"] = new_rotations
        
        baked_frames.append(baked_frame)
        
        log.progress(frame_idx, len(frames), "Baking camera to geometry", interval=50)
    
    log.info(f"Camera motion baked into geometry complete")
    return baked_frames


def create_static_camera_with_intrinsics(frames, sensor_width, up_axis, frame_offset=0, focal_length_px=None, principal_point_x=None, principal_point_y=None):
    """
    Create a static camera with correct intrinsics from frame data.
    
    This is used when camera motion is baked into geometry - the camera
    stays fixed at a reasonable position with correct focal length.
    
    Args:
        frames: Frame data (for focal length and image size)
        sensor_width: Camera sensor width in mm
        up_axis: Which axis points up
        frame_offset: Frame offset for keyframes
        focal_length_px: Optional focal length in pixels from MoGe2 (takes precedence)
        principal_point_x: Optional principal point X (cx) in pixels
        principal_point_y: Optional principal point Y (cy) in pixels
    
    Returns:
        Camera object
    """
    log.info("Creating static camera with intrinsics...")
    
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)
    
    # Set rotation mode to ZXY BEFORE any keyframes (critical for Maya compatibility)
    camera.rotation_mode = 'ZXY'
    
    # Set sensor width
    cam_data.sensor_width = sensor_width
    
    # Get image dimensions and focal length from first frame
    first_frame = frames[0]
    image_size = first_frame.get("image_size")
    
    image_width = 1920
    image_height = 1080
    if image_size and len(image_size) >= 2:
        image_width, image_height = image_size[0], image_size[1]
    
    # Set sensor height to match aspect ratio
    aspect_ratio = image_width / image_height
    cam_data.sensor_height = sensor_width / aspect_ratio
    cam_data.sensor_fit = 'HORIZONTAL'
    
    # Get focal length - MoGe2 intrinsics take precedence
    focal_px = 1000  # Default
    focal_source = "default"
    
    if focal_length_px is not None:
        # Use MoGe2 intrinsics
        focal_px = focal_length_px
        focal_source = "MoGe2"
    else:
        # Fall back to frame data
        first_focal = first_frame.get("focal_length")
        if first_focal:
            if isinstance(first_focal, (list, tuple)):
                focal_px = first_focal[0]
            else:
                focal_px = first_focal
            focal_source = "frame_data"
    
    # Convert focal length: focal_mm = focal_px * sensor_width / image_width
    focal_mm = focal_px * (sensor_width / image_width)
    cam_data.lens = focal_mm
    log.info(f"Static camera: {focal_px:.0f}px -> {focal_mm:.1f}mm focal length (source: {focal_source})")
    log.debug(f"Image: {image_width}x{image_height}, sensor: {sensor_width:.1f}mm x {cam_data.sensor_height:.1f}mm")
    
    # Add custom properties for Maya (exported as Extra Attributes)
    camera["sensor_width_mm"] = sensor_width
    camera["sensor_height_mm"] = cam_data.sensor_height
    camera["focal_length_mm"] = focal_mm
    camera["focal_length_px"] = focal_px
    camera["image_width"] = image_width
    camera["image_height"] = image_height
    camera["aspect_ratio"] = aspect_ratio
    
    # ============================================================
    # FILM OFFSET (from principal point cx, cy)
    # ============================================================
    # Principal point is where the optical axis intersects the image plane.
    # If cx/cy != image center, we need to apply film offset.
    #
    # Blender's shift_x/shift_y:
    # - shift_x > 0: shifts rendered image RIGHT (principal point LEFT of center)
    # - shift_y > 0: shifts rendered image UP (principal point BELOW center)
    #
    # Maya's Film Offset:
    # - Film Offset X > 0: shifts film gate RIGHT
    # - Film Offset Y > 0: shifts film gate UP
    #
    # The relationship for cx > image_center_x (principal point RIGHT of center):
    # - We need shift_x < 0 (or equivalently, negative film offset X)
    # ============================================================
    
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    
    if principal_point_x is not None and principal_point_y is not None:
        cx = principal_point_x
        cy = principal_point_y
        
        # Calculate offset from image center (in pixels)
        offset_x_px = cx - center_x
        offset_y_px = cy - center_y
        
        # Convert to Blender shift (normalized by image dimension)
        # Note: Blender shift is opposite sign of principal point offset
        # because shift moves the image, not the principal point
        shift_x = -offset_x_px / image_width
        shift_y = offset_y_px / image_height  # Y is NOT negated (different convention)
        
        # Apply to camera
        cam_data.shift_x = shift_x
        cam_data.shift_y = shift_y
        
        log.debug(f"Principal point: cx={cx:.2f}px, cy={cy:.2f}px")
        log.debug(f"Image center: ({center_x:.1f}, {center_y:.1f})")
        log.debug(f"Principal point offset: dx={offset_x_px:.2f}px, dy={offset_y_px:.2f}px")
        log.debug(f"Film offset (Blender shift): X={shift_x:.4f}, Y={shift_y:.4f}")
        
        # Also calculate Maya-style film offset for reference
        # Maya Film Offset = shift * sensor_size
        maya_film_offset_x = shift_x * sensor_width
        maya_film_offset_y = shift_y * cam_data.sensor_height
        log.debug(f"Film offset (Maya style): X={maya_film_offset_x:.4f}mm, Y={maya_film_offset_y:.4f}mm")
    else:
        # No principal point data - assume centered
        log.debug(f"Principal point: not specified (assuming centered at {center_x:.1f}, {center_y:.1f})")
        cam_data.shift_x = 0
        cam_data.shift_y = 0
    
    # Get camera distance from pred_cam_t
    first_cam_t = first_frame.get("pred_cam_t")
    cam_distance = 3.0
    if first_cam_t and len(first_cam_t) >= 3:
        cam_distance = abs(first_cam_t[2])
    
    # Set rotation mode to XYZ for Maya compatibility
    camera.rotation_mode = 'XYZ'
    
    # Position camera based on up axis
    # Camera looks in -Z direction by default in Blender
    # For Maya Y-up: camera on +Z axis, looking at origin (0,0,0) = looking in -Z direction
    if up_axis == "Y":
        # Y-up: Camera on +Z, looking at origin (no rotation needed, default is -Z)
        camera.location = Vector((0, 0, cam_distance))
        camera.rotation_euler = Euler((0, 0, 0), 'XYZ')
    elif up_axis == "Z":
        # Z-up: Camera on +Y, looking at origin
        camera.location = Vector((0, cam_distance, 0))
        camera.rotation_euler = Euler((math.radians(-90), 0, 0), 'XYZ')
    elif up_axis == "-Y":
        # -Y up: Camera on -Z, looking at origin
        camera.location = Vector((0, 0, -cam_distance))
        camera.rotation_euler = Euler((0, math.radians(180), 0), 'XYZ')
    elif up_axis == "-Z":
        # -Z up: Camera on -Y, looking at origin
        camera.location = Vector((0, -cam_distance, 0))
        camera.rotation_euler = Euler((math.radians(90), 0, 0), 'XYZ')
    else:
        camera.location = Vector((0, 0, cam_distance))
        camera.rotation_euler = Euler((0, 0, 0), 'XYZ')
    
    log.info(f"Static camera at distance {cam_distance:.2f}")
    log.info(f"Camera rotation mode: {camera.rotation_mode}")
    
    # ============================================================
    # ANIMATE FOCAL LENGTH IF IT VARIES
    # ============================================================
    # Check if per-frame focal length varies and animate if so
    focal_lengths = []
    for frame_data in frames:
        fl = frame_data.get("focal_length")
        if fl is not None:
            if isinstance(fl, (list, tuple)):
                fl = fl[0]
            focal_lengths.append(fl)
    
    if len(focal_lengths) > 1:
        fl_min, fl_max = min(focal_lengths), max(focal_lengths)
        if fl_max - fl_min > 1.0:  # More than 1 pixel difference
            log.info(f"Variable focal length detected: {fl_min:.0f}px to {fl_max:.0f}px")
            log.info(f"Animating camera focal length over {len(frames)} frames...")
            
            for frame_idx, frame_data in enumerate(frames):
                fl = frame_data.get("focal_length")
                if fl is not None:
                    if isinstance(fl, (list, tuple)):
                        fl = fl[0]
                    focal_mm = fl * (sensor_width / image_width)
                    cam_data.lens = focal_mm
                    cam_data.keyframe_insert(data_path="lens", frame=frame_offset + frame_idx)
            
            focal_mm_min = fl_min * (sensor_width / image_width)
            focal_mm_max = fl_max * (sensor_width / image_width)
            log.info(f"Focal length animated: {focal_mm_min:.1f}mm to {focal_mm_max:.1f}mm")
        else:
            log.info(f"Focal length constant at ~{fl_min:.0f}px ({fl_min * (sensor_width / image_width):.1f}mm)")
    
    return camera


def clear_scene():
    """Remove all objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clean data blocks
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for cam in bpy.data.cameras:
        bpy.data.cameras.remove(cam)


def get_transform_for_axis(up_axis, flip_x=False):
    """
    Get coordinate transformation based on desired up axis.
    SAM3DBody uses: X-right, Y-up, Z-forward (OpenGL convention)
    
    Args:
        up_axis: Which axis should point up ("Y", "Z", "-Y", "-Z")
        flip_x: If True, mirror the result on X axis (useful if animation appears flipped)
    
    Returns: (flip_func, axis_forward, axis_up)
    """
    # Base X multiplier - flip_x inverts this
    x_mult = 1 if flip_x else -1
    
    if up_axis == "Y":
        return lambda p: (x_mult * p[0], -p[1], -p[2]), '-Z', 'Y'
    elif up_axis == "Z":
        return lambda p: (x_mult * p[0], -p[2], p[1]), 'Y', 'Z'
    elif up_axis == "-Y":
        return lambda p: (x_mult * p[0], p[1], p[2]), 'Z', '-Y'
    elif up_axis == "-Z":
        return lambda p: (x_mult * p[0], p[2], -p[1]), '-Y', '-Z'
    else:
        return lambda p: (x_mult * p[0], -p[1], -p[2]), '-Z', 'Y'


def get_rotation_transform_matrix(up_axis, flip_x=False):
    """
    Get rotation transformation matrix for converting MHR rotations to Blender.
    """
    # Base X multiplier for rotation matrix
    x_mult = 1 if flip_x else -1
    
    if up_axis == "Y":
        # Flip X, Y, Z -> mirror across origin
        return Matrix((
            (x_mult, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ))
    elif up_axis == "Z":
        # X stays, Y<->Z swap with sign changes
        return Matrix((
            (x_mult, 0, 0),
            (0, 0, 1),
            (0, -1, 0)
        ))
    elif up_axis == "-Y":
        return Matrix((
            (x_mult, 0, 0),
            (0, 1, 0),
            (0, 0, 1)
        ))
    elif up_axis == "-Z":
        return Matrix((
            (x_mult, 0, 0),
            (0, 0, -1),
            (0, 1, 0)
        ))
    else:
        return Matrix((
            (x_mult, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ))


def transform_rotation_matrix(rot_3x3, up_axis):
    """
    Transform a 3x3 rotation matrix from MHR space to Blender space.
    
    rot_3x3: List of lists [[r00,r01,r02], [r10,r11,r12], [r20,r21,r22]]
    up_axis: Target up axis
    
    Returns: Blender Matrix (3x3)
    """
    # Convert to Blender Matrix
    m = Matrix((
        (rot_3x3[0][0], rot_3x3[0][1], rot_3x3[0][2]),
        (rot_3x3[1][0], rot_3x3[1][1], rot_3x3[1][2]),
        (rot_3x3[2][0], rot_3x3[2][1], rot_3x3[2][2])
    ))
    
    # Get transformation matrix - flip_x is handled in the global FLIP_X variable set by main()
    T = get_rotation_transform_matrix(up_axis, FLIP_X)
    
    # Transform: T * M * T^-1 (similarity transform)
    # This transforms the rotation from MHR coordinate system to Blender's
    T_inv = T.inverted()
    transformed = T @ m @ T_inv
    
    return transformed


# Global flip_x setting (set by main)
FLIP_X = False
DISABLE_VERTICAL_OFFSET = False  # v5.1.9: Disable Y offset from pred_cam_t
DISABLE_HORIZONTAL_OFFSET = False  # v5.1.10: Disable X offset from pred_cam_t
DISABLE_ALL_OFFSETS = False  # v5.1.10: Disable all offsets from pred_cam_t
FLIP_VERTICAL = False  # v5.1.9: Flip Y offset sign
FLIP_HORIZONTAL = False  # v5.1.10: Flip X offset sign


def get_world_offset_from_cam_t(pred_cam_t, up_axis):
    """
    Get world offset for root_locator.
    
    IMPORTANT: root_locator should be at origin (0,0,0) because it parents both
    the body and camera. Moving root_locator moves both together, which doesn't
    affect the body's position relative to camera (i.e., alignment in camera view).
    
    The actual body offset relative to camera is handled by get_body_offset_from_cam_t().
    """
    # Root locator stays at origin - body offset is applied separately
    return Vector((0, 0, 0))


def get_body_offset_from_cam_t(pred_cam_t, up_axis):
    """
    Get offset to apply to body mesh/skeleton for correct camera alignment.
    
    pred_cam_t from SAM3DBody:
    - tx: horizontal offset (positive = body right of center)
    - ty: vertical offset (positive = body above center IN IMAGE SPACE)
    - tz: depth (camera distance)
    
    Uses global settings:
    - DISABLE_ALL_OFFSETS: Return (0,0,0) - character at origin
    - DISABLE_VERTICAL_OFFSET: Zero out vertical component
    - DISABLE_HORIZONTAL_OFFSET: Zero out horizontal component
    - FLIP_VERTICAL: Flip Y sign
    - FLIP_HORIZONTAL: Flip X sign
    """
    # If all offsets disabled, return origin
    if DISABLE_ALL_OFFSETS:
        return Vector((0, 0, 0))
    
    if not pred_cam_t or len(pred_cam_t) < 3:
        return Vector((0, 0, 0))
    
    tx, ty, tz = pred_cam_t[0], pred_cam_t[1], pred_cam_t[2]
    
    # Apply horizontal offset options
    if DISABLE_HORIZONTAL_OFFSET:
        tx_adjusted = 0
    elif FLIP_HORIZONTAL:
        tx_adjusted = -tx  # Flip sign
    else:
        tx_adjusted = tx  # Default
    
    # Apply vertical offset options
    if DISABLE_VERTICAL_OFFSET:
        ty_adjusted = 0
    elif FLIP_VERTICAL:
        ty_adjusted = ty  # Don't negate (flipped)
    else:
        ty_adjusted = -ty  # Default: negate for camera convention
    
    # Apply based on up_axis
    if up_axis == "Y":
        return Vector((tx_adjusted, ty_adjusted, 0))
    elif up_axis == "Z":
        return Vector((tx_adjusted, 0, ty_adjusted))
    elif up_axis == "-Y":
        return Vector((tx_adjusted, -ty_adjusted, 0))
    elif up_axis == "-Z":
        return Vector((tx_adjusted, 0, -ty_adjusted))
    else:
        return Vector((tx_adjusted, ty_adjusted, 0))



def create_animated_mesh(all_frames, faces, fps, transform_func, world_translation_mode="none", up_axis="Y", frame_offset=0):
    """
    Create mesh with per-vertex animation using shape keys.
    """
    first_verts = all_frames[0].get("vertices")
    if not first_verts:
        return None
    
    # Get first frame world offset for initial position
    first_world_offset = Vector((0, 0, 0))
    if world_translation_mode == "baked":
        first_cam_t = all_frames[0].get("pred_cam_t")
        first_world_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
    
    # Create mesh with first frame vertices
    mesh = bpy.data.meshes.new("body_mesh")
    verts = []
    for v in first_verts:
        pos = Vector(transform_func(v))
        if world_translation_mode == "baked":
            pos += first_world_offset
        verts.append(pos)
    
    if faces:
        mesh.from_pydata(verts, [], faces)
    else:
        mesh.from_pydata(verts, [], [])
    mesh.update()
    
    obj = bpy.data.objects.new("body", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Add basis shape key
    basis = obj.shape_key_add(name="Basis", from_mix=False)
    
    log.info(f"Creating {len(all_frames)} shape keys (translation={world_translation_mode}, offset={frame_offset})...")
    
    # Create shape keys for each frame
    for frame_idx, frame_data in enumerate(all_frames):
        frame_verts = frame_data.get("vertices")
        if not frame_verts:
            continue
        
        # Get world offset for this frame
        frame_world_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            frame_world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        sk = obj.shape_key_add(name=f"frame_{frame_idx:04d}", from_mix=False)
        
        for j, v in enumerate(frame_verts):
            if j < len(sk.data):
                pos = Vector(transform_func(v))
                if world_translation_mode == "baked":
                    pos += frame_world_offset
                sk.data[j].co = pos
        
        # Keyframe shape key value with frame_offset
        actual_frame = frame_idx + frame_offset
        last_frame = frame_offset + len(all_frames) - 1
        is_last = (frame_idx == len(all_frames) - 1)
        is_first = (frame_idx == 0)
        
        # Fade in keyframe (value 0 before this shape activates)
        if not is_first:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=actual_frame - 1)
        
        # Active keyframe (value 1 at this frame)
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=actual_frame)
        
        # Fade out keyframe (value 0 after this shape deactivates)
        # Don't add fadeout for last frame - it should stay at 1
        if not is_last:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=actual_frame + 1)
        
        log.progress(frame_idx, len(all_frames), "Shape keys", interval=50)
    
    log.info(f"Created mesh with {len(all_frames)} shape keys (frames {frame_offset} to {frame_offset + len(all_frames) - 1})")
    return obj


def create_skeleton_with_rotations(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None, joint_parents=None, frame_offset=0, solved_camera_rotations=None):
    """
    Create animated skeleton using armature with ROTATION keyframes.
    
    This uses the true joint rotation matrices from MHR model.
    Produces proper bone rotations for retargeting and animation editing.
    
    joint_parents: Array where joint_parents[i] = parent index of joint i (-1 for root)
    solved_camera_rotations: If provided, apply inverse rotation to root bone to compensate for camera motion
    """
    first_joints = all_frames[0].get("joint_coords")
    first_rotations = all_frames[0].get("joint_rotations")
    
    if not first_joints:
        log.info("No joint_coords in first frame, skipping skeleton")
        return None
    
    if not first_rotations:
        log.info("No joint_rotations available, falling back to position-based skeleton")
        return create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator, joint_parents, frame_offset)
    
    num_joints = len(first_joints)
    log.info(f"Creating rotation-based armature with {num_joints} bones...")
    
    # Create armature
    arm_data = bpy.data.armatures.new("Skeleton")
    armature = bpy.data.objects.new("Skeleton", arm_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    
    # Parent to root locator if in "root" mode
    if world_translation_mode == "root" and root_locator:
        armature.parent = root_locator
    
    # Get first frame world offset for initial bone positions
    first_offset = Vector((0, 0, 0))
    if world_translation_mode == "baked":
        first_cam_t = all_frames[0].get("pred_cam_t")
        first_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
    
    # Enter edit mode to create bones WITH HIERARCHY
    bpy.ops.object.mode_set(mode='EDIT')
    
    edit_bones = []
    for i in range(num_joints):
        bone = arm_data.edit_bones.new(f"joint_{i:03d}")
        pos = first_joints[i]
        head_pos = Vector(transform_func(pos))
        
        if world_translation_mode == "baked":
            head_pos += first_offset
        
        bone.head = head_pos
        # Tail will be adjusted after hierarchy is set
        bone.tail = head_pos + Vector((0, 0.03, 0))
        edit_bones.append(bone)
    
    # Set up parent-child hierarchy
    if joint_parents is not None:
        log.info(f"Setting up bone hierarchy from joint_parents...")
        roots = []
        for i in range(num_joints):
            parent_idx = joint_parents[i]
            if parent_idx >= 0 and parent_idx < num_joints:
                edit_bones[i].parent = edit_bones[parent_idx]
                # Optionally connect bones if close enough
                # edit_bones[i].use_connect = False
            else:
                roots.append(i)
        log.info(f"Found {len(roots)} root bone(s): {roots}")
        
        # Adjust bone tails to point toward first child (makes visualization better)
        for i in range(num_joints):
            children = [j for j in range(num_joints) if joint_parents[j] == i]
            if children:
                # Point tail toward average of children positions
                child_positions = [edit_bones[c].head for c in children]
                avg_child_pos = sum(child_positions, Vector((0, 0, 0))) / len(children)
                direction = avg_child_pos - edit_bones[i].head
                if direction.length > 0.001:
                    edit_bones[i].tail = edit_bones[i].head + direction.normalized() * 0.05
    else:
        log.info("Warning: No joint_parents data, creating flat hierarchy")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate bones in pose mode using ROTATION keyframes
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Set rotation mode to quaternion for smoother interpolation
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = 'QUATERNION'
    
    log.info(f"Animating {num_joints} bones with rotations over {len(all_frames)} frames...")
    
    # Pre-compute parent indices for faster lookup
    parent_indices = joint_parents if joint_parents is not None else [-1] * num_joints
    
    # Check if we have camera rotations to compensate
    has_camera_rots = solved_camera_rotations is not None and len(solved_camera_rotations) > 0
    
    # Pre-smooth camera data to avoid jerky root motion
    smoothed_camera_data = None
    if has_camera_rots:
        log.info(f"Pre-smoothing {len(solved_camera_rotations)} camera rotations for smooth root compensation...")
        smoothed_camera_data = smooth_camera_data(solved_camera_rotations, window=9)
        log.info(f"Camera data smoothed (window=9)")
    
    for frame_idx, frame_data in enumerate(all_frames):
        joints = frame_data.get("joint_coords")
        rotations = frame_data.get("joint_rotations")
        
        if not joints or not rotations:
            continue
        
        actual_frame = frame_idx + frame_offset
        bpy.context.scene.frame_set(actual_frame)
        
        # Get world offset for this frame (for position updates)
        world_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        # Build camera compensation matrix (inverse of camera rotation)
        # Also get camera translation compensation if available
        camera_compensation = Matrix.Identity(3)
        camera_translation_compensation = Vector((0, 0, 0))
        
        if has_camera_rots and smoothed_camera_data and frame_idx < len(smoothed_camera_data):
            cam_rot = smoothed_camera_data[frame_idx]
            pan = cam_rot.get("pan", 0.0)
            tilt = cam_rot.get("tilt", 0.0)
            roll = cam_rot.get("roll", 0.0)
            
            # Build rotation: for Y-up, pan around Y, tilt around X
            if up_axis == "Y":
                euler = Euler((tilt, pan, roll), 'YXZ')
            elif up_axis == "Z":
                euler = Euler((tilt, roll, pan), 'ZXY')
            else:
                euler = Euler((tilt, pan, roll), 'YXZ')
            
            camera_rot_matrix = euler.to_matrix().to_3x3()
            camera_compensation = camera_rot_matrix.inverted()
            
            # Get translation compensation (inverse of camera translation)
            # This keeps the body in world space when camera moves
            tx = cam_rot.get("tx", 0.0)
            ty = cam_rot.get("ty", 0.0)
            tz = cam_rot.get("tz", 0.0)
            if tx != 0.0 or ty != 0.0 or tz != 0.0:
                # Invert translation to compensate
                if up_axis == "Y":
                    camera_translation_compensation = Vector((-tx, -ty, -tz))
                elif up_axis == "Z":
                    camera_translation_compensation = Vector((-tx, -tz, -ty))
        
        # First pass: compute all global rotations in Blender space
        global_rots_blender = []
        for bone_idx in range(min(num_joints, len(rotations))):
            rot_3x3 = rotations[bone_idx]
            if rot_3x3 is None:
                global_rots_blender.append(Matrix.Identity(3))
            else:
                blender_rot = transform_rotation_matrix(rot_3x3, up_axis)
                global_rots_blender.append(blender_rot)
        
        # Second pass: convert global rotations to local rotations and apply
        for bone_idx in range(min(num_joints, len(joints), len(rotations))):
            pose_bone = armature.pose.bones[bone_idx]
            
            global_rot = global_rots_blender[bone_idx]
            
            # Convert global rotation to local rotation
            parent_idx = parent_indices[bone_idx]
            if parent_idx >= 0 and parent_idx < len(global_rots_blender):
                # Local = Parent_global^-1 * Global
                parent_global_rot = global_rots_blender[parent_idx]
                local_rot = parent_global_rot.inverted() @ global_rot
            else:
                # Root bone: apply camera compensation
                local_rot = camera_compensation @ global_rot
            
            # Convert to quaternion and apply
            quat = local_rot.to_quaternion()
            pose_bone.rotation_quaternion = quat
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=actual_frame)
            
            # Also update location for world translation modes or camera compensation (root bone only)
            if parent_idx < 0:  # Root bone
                pos = joints[bone_idx]
                base_pos = Vector(transform_func(pos))
                
                # Apply world offset if baked mode
                if world_translation_mode == "baked":
                    base_pos += world_offset
                
                # Apply camera translation compensation (inverse of camera movement)
                # This keeps the body in world space when camera translates
                base_pos += camera_translation_compensation
                
                rest_head = Vector(armature.data.bones[bone_idx].head_local)
                offset = base_pos - rest_head
                pose_bone.location = offset
                pose_bone.keyframe_insert(data_path="location", frame=actual_frame)
        
        log.progress(frame_idx, len(all_frames), "Skeleton animation", interval=50)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    log.info(f"Created hierarchical skeleton with {num_joints} bones (frame_offset={frame_offset})")
    return armature


def create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None, joint_parents=None, frame_offset=0):
    """
    Create animated skeleton using armature with POSITION keyframes.
    
    This is the legacy mode - bones animate via location offset from rest position.
    Shows exact joint positions but limited for retargeting.
    """
    first_joints = all_frames[0].get("joint_coords")
    if not first_joints:
        return None
    
    num_joints = len(first_joints)
    log.info(f"Creating position-based armature with {num_joints} bones...")
    
    # Create armature
    arm_data = bpy.data.armatures.new("Skeleton")
    armature = bpy.data.objects.new("Skeleton", arm_data)
    bpy.context.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    
    # Parent to root locator if in "root" mode
    if world_translation_mode == "root" and root_locator:
        armature.parent = root_locator
    
    # Get first frame world offset for initial bone positions
    first_offset = Vector((0, 0, 0))
    if world_translation_mode == "baked":
        first_cam_t = all_frames[0].get("pred_cam_t")
        first_offset = get_world_offset_from_cam_t(first_cam_t, up_axis)
    
    # Enter edit mode to create bones WITH HIERARCHY
    bpy.ops.object.mode_set(mode='EDIT')
    
    edit_bones = []
    for i in range(num_joints):
        bone = arm_data.edit_bones.new(f"joint_{i:03d}")
        pos = first_joints[i]
        head_pos = Vector(transform_func(pos))
        
        if world_translation_mode == "baked":
            head_pos += first_offset
        
        bone.head = head_pos
        bone.tail = head_pos + Vector((0, 0.03, 0))
        edit_bones.append(bone)
    
    # Set up parent-child hierarchy
    if joint_parents is not None:
        log.info(f"Setting up bone hierarchy from joint_parents...")
        roots = []
        for i in range(num_joints):
            parent_idx = joint_parents[i]
            if parent_idx >= 0 and parent_idx < num_joints:
                edit_bones[i].parent = edit_bones[parent_idx]
            else:
                roots.append(i)
        log.info(f"Found {len(roots)} root bone(s): {roots}")
        
        # Adjust bone tails to point toward first child
        for i in range(num_joints):
            children = [j for j in range(num_joints) if joint_parents[j] == i]
            if children:
                child_positions = [edit_bones[c].head for c in children]
                avg_child_pos = sum(child_positions, Vector((0, 0, 0))) / len(children)
                direction = avg_child_pos - edit_bones[i].head
                if direction.length > 0.001:
                    edit_bones[i].tail = edit_bones[i].head + direction.normalized() * 0.05
    else:
        log.info("Warning: No joint_parents data, creating flat hierarchy")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Animate bones in pose mode using LOCATION keyframes
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    log.info(f"Animating {num_joints} bones with positions over {len(all_frames)} frames...")
    
    # Store rest positions for offset calculation
    rest_heads = [Vector(armature.pose.bones[i].bone.head_local) for i in range(num_joints)]
    
    for frame_idx, frame_data in enumerate(all_frames):
        joints = frame_data.get("joint_coords")
        if not joints:
            continue
        
        actual_frame = frame_idx + frame_offset
        bpy.context.scene.frame_set(actual_frame)
        
        # Get world offset for this frame
        world_offset = Vector((0, 0, 0))
        if world_translation_mode == "baked":
            frame_cam_t = frame_data.get("pred_cam_t")
            world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        for bone_idx in range(min(num_joints, len(joints))):
            pose_bone = armature.pose.bones[bone_idx]
            
            pos = joints[bone_idx]
            target = Vector(transform_func(pos))
            
            if world_translation_mode == "baked":
                target += world_offset
            
            # Calculate offset from rest position
            offset = target - rest_heads[bone_idx]
            pose_bone.location = offset
            pose_bone.keyframe_insert(data_path="location", frame=actual_frame)
        
        log.progress(frame_idx, len(all_frames), "Skeleton animation", interval=50)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    log.info(f"Created position-based skeleton with {num_joints} bones (frame_offset={frame_offset})")
    return armature


def create_skeleton(all_frames, fps, transform_func, world_translation_mode="none", up_axis="Y", root_locator=None, skeleton_mode="rotations", joint_parents=None, frame_offset=0, solved_camera_rotations=None):
    """
    Create animated skeleton - dispatcher function.
    
    skeleton_mode:
    - "rotations": Use true joint rotation matrices from MHR (recommended)
    - "positions": Use joint positions only (legacy)
    
    joint_parents: Array defining bone hierarchy
    solved_camera_rotations: Camera rotations from COLMAP to compensate body orientation
    """
    if skeleton_mode == "rotations":
        return create_skeleton_with_rotations(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator, joint_parents, frame_offset, solved_camera_rotations)
    else:
        return create_skeleton_with_positions(all_frames, fps, transform_func, world_translation_mode, up_axis, root_locator, joint_parents, frame_offset)


def create_root_locator(all_frames, fps, up_axis, flip_x=False, frame_offset=0):
    """
    Create a root locator that carries the world translation.
    """
    log.info("Creating root locator with world translation...")
    
    if all_frames:
        first_cam_t = all_frames[0].get("pred_cam_t")
        if first_cam_t and len(first_cam_t) >= 3:
            tx, ty, tz = first_cam_t[0], first_cam_t[1], first_cam_t[2]
            world_x = tx * abs(tz) * 0.5
            world_y = ty * abs(tz) * 0.5
    
    root = bpy.data.objects.new("root_locator", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 0.1
    bpy.context.collection.objects.link(root)
    
    # Animate root position based on pred_cam_t
    for frame_idx, frame_data in enumerate(all_frames):
        frame_cam_t = frame_data.get("pred_cam_t")
        world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        # Apply flip_x to the world offset
        if flip_x:
            world_offset = Vector((-world_offset.x, world_offset.y, world_offset.z))
        
        root.location = world_offset
        root.keyframe_insert(data_path="location", frame=frame_idx + frame_offset)
    
    log.info(f"Root locator animated over {len(all_frames)} frames (offset={frame_offset}, flip_x={flip_x})")
    return root


def create_root_locator_with_camera_compensation(all_frames, camera_extrinsics, fps, up_axis, flip_x=False, frame_offset=0, smoothing_method="kalman", smoothing_strength=0.5):
    """
    Create a root locator with inverse camera extrinsics baked in.
    
    IMPORTANT: For nodal cameras (tripod rotation), pan/tilt should be converted
    to TRANSLATION, not rotation. This is because:
    - A nodal camera rotates around its nodal point
    - When camera pans right, subject appears to move left in frame
    - But subject isn't actually rotating - it's appearing to translate
    - The compensation must be translation, not rotation
    
    The math:
    - Pan angle θ at distance d → horizontal translation = d * tan(θ)
    - Tilt angle φ at distance d → vertical translation = d * tan(φ)
    
    This allows:
    - Camera to be exported as static
    - Body animation to include the inverse of camera motion as translation
    - Clean result in Maya/Unity/etc. with stable camera
    
    Args:
        all_frames: Frame data with pred_cam_t (contains distance in tz)
        camera_extrinsics: List of dicts with pan, tilt, roll, tx, ty, tz
        fps: Frame rate
        up_axis: Up axis for coordinate system
        flip_x: Mirror on X axis
        frame_offset: Start frame offset
        smoothing_method: "kalman", "spline", "gaussian", or "none"
        smoothing_strength: Smoothing intensity (0.0-1.0)
    
    Returns:
        Root locator Blender object with animated transform
    """
    log.info("Creating root locator with camera compensation...")
    log.debug(f"  Mode: Nodal rotation → Translation conversion")
    log.debug(f"  Smoothing: {smoothing_method} (strength={smoothing_strength})")
    log.debug(f"  Flip X: {flip_x}")
    
    root = bpy.data.objects.new("root_locator", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 0.1
    bpy.context.collection.objects.link(root)
    
    # Pre-smooth camera extrinsics if requested
    smoothed_extrinsics = camera_extrinsics
    if smoothing_method != "none" and len(camera_extrinsics) > 3:
        if smoothing_method == "gaussian":
            window = int(3 + smoothing_strength * 12)  # 3-15 frames
            smoothed_extrinsics = smooth_camera_data(camera_extrinsics, window)
        elif smoothing_method == "kalman":
            # Use a lighter Kalman smoothing for root locator
            smoothed_extrinsics = kalman_filter_camera_data(
                camera_extrinsics, 
                process_noise=0.01, 
                measurement_noise=0.05 + smoothing_strength * 0.3
            )
        elif smoothing_method == "spline":
            smoothed_extrinsics = spline_fit_camera_data(camera_extrinsics, smoothing_strength)
    
    # Build a frame-indexed dict for easy lookup
    extrinsics_by_frame = {}
    for ext in smoothed_extrinsics:
        extrinsics_by_frame[ext.get("frame", 0)] = ext
    
    # Get initial camera extrinsics as reference (frame 0)
    initial_ext = extrinsics_by_frame.get(0, smoothed_extrinsics[0] if smoothed_extrinsics else {})
    initial_pan = initial_ext.get("pan", 0)
    initial_tilt = initial_ext.get("tilt", 0)
    initial_roll = initial_ext.get("roll", 0)
    initial_tx = initial_ext.get("tx", 0)
    initial_ty = initial_ext.get("ty", 0)
    initial_tz = initial_ext.get("tz", 0)
    
    # Get initial distance from first frame's pred_cam_t
    first_frame_cam_t = all_frames[0].get("pred_cam_t") if all_frames else None
    if first_frame_cam_t and len(first_frame_cam_t) >= 3:
        initial_distance = first_frame_cam_t[2]  # tz = depth/distance
    else:
        initial_distance = 5.0  # Default fallback
    
    log.debug(f"  Initial camera: pan={math.degrees(initial_pan):.2f}°, tilt={math.degrees(initial_tilt):.2f}°")
    log.debug(f"  Initial distance: {initial_distance:.2f}m (from pred_cam_t.z)")
    
    # Track total compensation for logging
    max_pan_trans = 0
    max_tilt_trans = 0
    
    for frame_idx, frame_data in enumerate(all_frames):
        # Get distance from this frame's pred_cam_t
        frame_cam_t = frame_data.get("pred_cam_t")
        if frame_cam_t and len(frame_cam_t) >= 3:
            distance = frame_cam_t[2]  # tz = depth/distance
        else:
            distance = initial_distance
        
        # Get camera extrinsics for this frame
        ext = extrinsics_by_frame.get(frame_idx, {})
        
        # Calculate delta from initial pose (relative camera motion)
        delta_pan = ext.get("pan", 0) - initial_pan
        delta_tilt = ext.get("tilt", 0) - initial_tilt
        delta_roll = ext.get("roll", 0) - initial_roll
        delta_tx = ext.get("tx", 0) - initial_tx
        delta_ty = ext.get("ty", 0) - initial_ty
        delta_tz = ext.get("tz", 0) - initial_tz
        
        # ============================================================
        # NODAL CAMERA ROTATION → TRANSLATION CONVERSION
        # ============================================================
        # When a nodal camera (tripod) pans/tilts, the subject appears
        # to translate in screen space, not rotate.
        # 
        # Formula: translation = distance * tan(angle)
        #
        # - Pan (horizontal rotation) → Horizontal translation
        #   If camera pans RIGHT (positive pan), subject moves LEFT in frame
        #   Compensation: translate character RIGHT (positive X)
        #
        # - Tilt (vertical rotation) → Vertical translation
        #   If camera tilts UP (positive tilt), subject moves DOWN in frame
        #   Compensation: translate character UP (positive Y)
        # ============================================================
        
        # Convert nodal rotation to translation (inverse for compensation)
        # Negative because we want to compensate (inverse transform)
        pan_translation = distance * math.tan(delta_pan)   # Horizontal
        tilt_translation = distance * math.tan(delta_tilt) # Vertical
        
        # Track max values for logging
        max_pan_trans = max(max_pan_trans, abs(pan_translation))
        max_tilt_trans = max(max_tilt_trans, abs(tilt_translation))
        
        # Camera translation compensation (inverse)
        inv_tx = -delta_tx
        inv_ty = -delta_ty
        inv_tz = -delta_tz
        
        # Roll stays as rotation (it doesn't translate the subject)
        inv_roll = -delta_roll
        
        # Apply coordinate system conversion
        if up_axis.upper() in ["Y", "-Y"]:
            # Y-up: X=horizontal, Y=up, Z=depth
            # Pan → X translation, Tilt → Y translation
            nodal_trans = Vector((pan_translation, tilt_translation, 0))
            camera_trans = Vector((inv_tx, inv_ty, inv_tz))
            rot_euler = Euler((0, 0, inv_roll), 'XYZ')  # Only roll stays as rotation
        else:
            # Z-up: X=horizontal, Z=up, Y=depth
            nodal_trans = Vector((pan_translation, 0, tilt_translation))
            camera_trans = Vector((inv_tx, inv_tz, inv_ty))
            rot_euler = Euler((0, inv_roll, 0), 'XYZ')
        
        # Combine all translations:
        # 1. Nodal rotation converted to translation
        # 2. Camera translation (inverse)
        final_location = nodal_trans + camera_trans
        
        # Apply flip_x
        if flip_x:
            final_location = Vector((-final_location.x, final_location.y, final_location.z))
            rot_euler = Euler((-rot_euler.x, rot_euler.y, -rot_euler.z), 'XYZ')
        
        # Set keyframes
        root.location = final_location
        root.rotation_euler = rot_euler
        root.keyframe_insert(data_path="location", frame=frame_idx + frame_offset)
        root.keyframe_insert(data_path="rotation_euler", frame=frame_idx + frame_offset)
    
    # Report final compensation
    if len(smoothed_extrinsics) > 0:
        final_ext = extrinsics_by_frame.get(len(all_frames) - 1, smoothed_extrinsics[-1])
        final_pan = final_ext.get("pan", 0) - initial_pan
        final_tilt = final_ext.get("tilt", 0) - initial_tilt
        final_pan_trans = initial_distance * math.tan(final_pan)
        final_tilt_trans = initial_distance * math.tan(final_tilt)
        
        log.info(f"Root locator with camera compensation (nodal → translation):")
        log.debug(f"  Camera pan range: {math.degrees(final_pan):.2f}° → {final_pan_trans:.3f}m translation")
        log.debug(f"  Camera tilt range: {math.degrees(final_tilt):.2f}° → {final_tilt_trans:.3f}m translation")
        log.debug(f"  Max translation: X={max_pan_trans:.3f}m, Y={max_tilt_trans:.3f}m")
        log.debug(f"  Animated over {len(all_frames)} frames")
    
    return root


def create_translation_track(all_frames, fps, up_axis, frame_offset=0):
    """
    Create a separate locator that shows the world path.
    """
    log.info("Creating separate translation track...")
    
    track = bpy.data.objects.new("translation_track", None)
    track.empty_display_type = 'PLAIN_AXES'
    track.empty_display_size = 0.15
    bpy.context.collection.objects.link(track)
    
    for frame_idx, frame_data in enumerate(all_frames):
        frame_cam_t = frame_data.get("pred_cam_t")
        world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
        
        track.location = world_offset
        track.keyframe_insert(data_path="location", frame=frame_idx + frame_offset)
    
    log.info(f"Translation track animated over {len(all_frames)} frames (offset={frame_offset})")
    return track


def create_camera(all_frames, fps, transform_func, up_axis, sensor_width=36.0, world_translation_mode="none", animate_camera=False, frame_offset=0, camera_follow_root=False, camera_use_rotation=False, camera_static=False, camera_smoothing=0, flip_x=False, solved_camera_rotations=None):
    """
    Create camera with focal length from SAM3DBody.
    
    The camera is positioned to match SAM3DBody's projection, accounting for:
    - Focal length (converted from pixels to mm)
    - Distance from pred_cam_t[2]
    - Offset from bbox position relative to image center
    
    Args:
        animate_camera: If True and mode=="camera", animate camera position with world offset.
        camera_follow_root: If True, camera will be parented to root_locator and needs
                           LOCAL animation to show character at correct screen position.
        camera_use_rotation: If True, use pan/tilt rotation instead of translation.
                            Better for tripod/handheld shots where camera rotates to follow subject.
        camera_static: If True, camera stays completely fixed (no rotation or translation animation).
                      Used with body_offset to position body correctly.
        camera_smoothing: Smoothing window for camera values to reduce jitter (0=none).
        flip_x: Whether X axis is flipped (affects camera pan direction).
        frame_offset: Starting frame number for keyframes.
        solved_camera_rotations: Optional list of solved rotations from Camera Rotation Solver.
                                Each entry has {frame, pan, tilt, roll} in radians.
    """
    has_solved = solved_camera_rotations is not None and len(solved_camera_rotations) > 0
    log.info(f"Creating camera (animate={animate_camera}, follow_root={camera_follow_root}, use_rotation={camera_use_rotation}, static={camera_static}, smoothing={camera_smoothing}, solved_rotations={has_solved})...")
    
    cam_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(camera)
    
    # Set rotation mode to ZXY BEFORE any keyframes (critical for Maya compatibility)
    camera.rotation_mode = 'ZXY'
    
    # Set sensor width
    cam_data.sensor_width = sensor_width
    
    # Get image dimensions
    first_frame = all_frames[0]
    image_size = first_frame.get("image_size")
    
    image_width = 1920  # Default
    image_height = 1080
    if image_size:
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            image_width, image_height = image_size[0], image_size[1]
    
    # Set sensor height to match aspect ratio (important for correct projection!)
    aspect_ratio = image_width / image_height
    cam_data.sensor_height = sensor_width / aspect_ratio
    cam_data.sensor_fit = 'HORIZONTAL'
    
    # Get focal length from first frame
    first_focal = first_frame.get("focal_length")
    focal_px = 1000  # Default
    if first_focal:
        if isinstance(first_focal, (list, tuple)):
            focal_px = first_focal[0]
        else:
            focal_px = first_focal
    
    # Convert focal length: focal_mm = focal_px * sensor_width / image_width
    focal_mm = focal_px * (sensor_width / image_width)
    cam_data.lens = focal_mm
    log.info(f"Focal length: {focal_px:.0f}px -> {focal_mm:.1f}mm")
    log.info(f"Image size: {image_width}x{image_height}, aspect: {aspect_ratio:.3f}")
    log.info(f"Sensor: {sensor_width:.1f}mm x {cam_data.sensor_height:.1f}mm")
    
    # Add custom properties for Maya (exported as Extra Attributes)
    # These help set up the camera correctly after FBX import
    camera["sensor_width_mm"] = sensor_width
    camera["sensor_height_mm"] = cam_data.sensor_height
    camera["focal_length_mm"] = focal_mm
    camera["focal_length_px"] = focal_px
    camera["image_width"] = image_width
    camera["image_height"] = image_height
    camera["aspect_ratio"] = aspect_ratio
    
    # Get pred_cam_t - this is the KEY to matching SAM3DBody's projection
    # pred_cam_t = [tx, ty, tz] where:
    #   tx, ty = normalized screen offset (roughly -1 to 1)
    #   tz = depth (camera distance)
    first_cam_t = first_frame.get("pred_cam_t")
    
    cam_distance = 3.0
    tx, ty = 0.0, 0.0
    if first_cam_t and len(first_cam_t) >= 3:
        tx, ty = first_cam_t[0], first_cam_t[1]
        cam_distance = abs(first_cam_t[2])
    
    # IMPORTANT: When using world_translation_mode="root", body offset is applied
    # directly to the mesh/skeleton via get_body_offset_from_cam_t().
    # In this case, camera should look straight at origin - NO target_offset!
    # 
    # The body_offset positions the body correctly relative to camera.
    # Adding target_offset would DOUBLE-count the offset.
    if world_translation_mode == "root":
        target_offset = Vector((0, 0, 0))
        log.info(f"Root mode: Camera looks at origin, body_offset applied to mesh/skeleton")
    else:
        # Legacy mode: body at origin, camera target offset to frame correctly
        scale_factor = cam_distance * 0.5
        if up_axis == "Y":
            target_offset = Vector((tx * scale_factor, -ty * scale_factor, 0))
        elif up_axis == "Z":
            target_offset = Vector((tx * scale_factor, 0, -ty * scale_factor))
        elif up_axis == "-Y":
            target_offset = Vector((tx * scale_factor, ty * scale_factor, 0))
        elif up_axis == "-Z":
            target_offset = Vector((tx * scale_factor, 0, ty * scale_factor))
        else:
            target_offset = Vector((tx * scale_factor, -ty * scale_factor, 0))
    
    log.info(f"pred_cam_t: tx={tx:.3f}, ty={ty:.3f}, tz={cam_distance:.2f}")
    log.info(f"Camera target offset: {target_offset}")
    
    # Create target for camera orientation
    target = bpy.data.objects.new("cam_target", None)
    target.location = target_offset
    bpy.context.collection.objects.link(target)
    
    # Set camera direction based on up_axis
    if up_axis == "Y":
        base_dir = Vector((0, 0, 1))
    elif up_axis == "Z":
        base_dir = Vector((0, 1, 0))
    elif up_axis == "-Y":
        base_dir = Vector((0, 0, -1))
    elif up_axis == "-Z":
        base_dir = Vector((0, -1, 0))
    else:
        base_dir = Vector((0, 0, 1))
    
    # For "camera" mode with animate_camera=True: animated camera to follow character
    if animate_camera and world_translation_mode == "camera":
        
        if camera_use_rotation:
            # ROTATION MODE: Camera at fixed position, rotates to follow
            # Initial setup: camera looking straight at origin
            camera.location = base_dir * cam_distance  # No target_offset!
            
            # Point camera at origin to get base rotation
            origin_target = bpy.data.objects.new("origin_target", None)
            origin_target.location = Vector((0, 0, 0))
            bpy.context.collection.objects.link(origin_target)
            
            constraint = camera.constraints.new(type='TRACK_TO')
            constraint.target = origin_target
            constraint.track_axis = 'TRACK_NEGATIVE_Z'
            
            if up_axis == "Y" or up_axis == "-Y":
                constraint.up_axis = 'UP_Y'
            elif up_axis == "Z" or up_axis == "-Z":
                constraint.up_axis = 'UP_Z'
            else:
                constraint.up_axis = 'UP_Y'
            
            bpy.context.view_layer.update()
            base_rotation = camera.matrix_world.to_euler()
            camera.rotation_euler = base_rotation.copy()
            camera.constraints.remove(constraint)
            bpy.data.objects.remove(origin_target)
            
            camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
            camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
            camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
            base_rotation = camera.rotation_euler.copy()
            
            # Also remove the offset target we created earlier
            bpy.data.objects.remove(target)
            
            log.info(f"Camera using ROTATION (Pan/Tilt) to follow character...")
            
            # Get pan/tilt axis configuration based on up_axis
            # Pan = rotation around UP axis, Tilt = rotation around horizontal axis
            # 
            # KEY INSIGHT for Y-up (Maya default):
            # - Positive Y rotation = pan LEFT = origin shifts RIGHT in view
            # - Positive X rotation = tilt DOWN = origin shifts UP in view
            # 
            # So for tx > 0 (body on RIGHT), we need positive Y (pan left, origin goes right)
            # For ty > 0 (body BELOW center), we need negative X (tilt up, origin goes down)
            if up_axis == "Y":
                pan_axis = 1   # Y axis for pan (yaw)
                tilt_axis = 0  # X axis for tilt (pitch)
                tilt_sign = -1  # ty > 0 → tilt UP (negative X) → origin appears lower
                pan_sign = 1    # tx > 0 → pan LEFT (positive Y) → origin appears right
            elif up_axis == "Z":
                pan_axis = 2   # Z axis for pan
                tilt_axis = 0  # X axis for tilt
                tilt_sign = -1
                pan_sign = 1
            elif up_axis == "-Y":
                pan_axis = 1
                tilt_axis = 0
                tilt_sign = 1
                pan_sign = -1
            elif up_axis == "-Z":
                pan_axis = 2
                tilt_axis = 0
                tilt_sign = 1
                pan_sign = -1
            else:
                pan_axis = 1
                tilt_axis = 0
                tilt_sign = -1
                pan_sign = 1
            
            for frame_idx, frame_data in enumerate(all_frames):
                frame_cam_t = frame_data.get("pred_cam_t")
                
                frame_distance = cam_distance
                if frame_cam_t and len(frame_cam_t) > 2:
                    frame_distance = abs(frame_cam_t[2])
                
                # Use solved camera rotations if available, otherwise compute from pred_cam_t
                if has_solved and frame_idx < len(solved_camera_rotations):
                    solved_rot = solved_camera_rotations[frame_idx]
                    # Solved rotation = actual camera motion from background tracking
                    # This should be the SAME for all runners in the same video!
                    pan_angle = solved_rot.get("pan", 0.0) * pan_sign
                    tilt_angle = solved_rot.get("tilt", 0.0) * tilt_sign
                    roll_angle = solved_rot.get("roll", 0.0)
                    
                    # Apply solved rotation only - no per-body baseline offset
                    camera.rotation_euler = base_rotation.copy()
                    camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                    camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                    camera.rotation_euler[2] = base_rotation[2] + roll_angle
                    
                elif frame_cam_t and len(frame_cam_t) >= 3:
                    tx, ty, tz = frame_cam_t[0], frame_cam_t[1], frame_cam_t[2]
                    
                    # Fallback: compute from pred_cam_t (body screen position)
                    # This is per-body and should only be used when no solved rotation available
                    depth = abs(tz) if abs(tz) > 0.1 else 0.1
                    pan_angle = math.atan2(tx, depth) * pan_sign
                    tilt_angle = math.atan2(ty, depth) * tilt_sign
                    
                    camera.rotation_euler = base_rotation.copy()
                    camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                    camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                
                # Camera position: just depth along base direction
                camera.location = base_dir * frame_distance
                
                camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset + frame_idx)
                camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
            
            if has_solved:
                log.info(f"Camera uses SOLVED rotation over {len(all_frames)} frames (same for all bodies)")
            else:
                log.info(f"Camera ROTATES (pan/tilt from pred_cam_t) over {len(all_frames)} frames (up_axis={up_axis})")
        
        else:
            # TRANSLATION MODE: Camera moves laterally to follow character
            # Set initial rotation pointing at target
            camera.location = base_dir * cam_distance + target_offset
            
            constraint = camera.constraints.new(type='TRACK_TO')
            constraint.target = target
            constraint.track_axis = 'TRACK_NEGATIVE_Z'
            
            if up_axis == "Y" or up_axis == "-Y":
                constraint.up_axis = 'UP_Y'
            elif up_axis == "Z" or up_axis == "-Z":
                constraint.up_axis = 'UP_Z'
            else:
                constraint.up_axis = 'UP_Y'
            
            bpy.context.view_layer.update()
            camera.rotation_euler = camera.matrix_world.to_euler()
            camera.constraints.remove(constraint)
            
            camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
            camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
            camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
            
            bpy.data.objects.remove(target)
            
            log.info(f"Camera using TRANSLATION to follow character...")
            
            for frame_idx, frame_data in enumerate(all_frames):
                frame_cam_t = frame_data.get("pred_cam_t")
                
                frame_distance = cam_distance
                if frame_cam_t and len(frame_cam_t) > 2:
                    frame_distance = abs(frame_cam_t[2])
                
                world_offset = get_world_offset_from_cam_t(frame_cam_t, up_axis)
                camera.location = base_dir * frame_distance + target_offset - world_offset
                camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
            
            log.info(f"Camera TRANSLATES over {len(all_frames)} frames (up_axis={up_axis})")
    
    elif camera_use_rotation:
        # "None" mode with rotation: Camera rotates to show character at correct screen position
        # Body stays at origin, camera pans/tilts to frame it correctly
        camera.location = base_dir * cam_distance  # No target_offset
        
        # Point camera at origin to get base rotation
        origin_target = bpy.data.objects.new("origin_target", None)
        origin_target.location = Vector((0, 0, 0))
        bpy.context.collection.objects.link(origin_target)
        
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = origin_target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        
        if up_axis == "Y" or up_axis == "-Y":
            constraint.up_axis = 'UP_Y'
        elif up_axis == "Z" or up_axis == "-Z":
            constraint.up_axis = 'UP_Z'
        else:
            constraint.up_axis = 'UP_Y'
        
        bpy.context.view_layer.update()
        base_rotation = camera.matrix_world.to_euler()
        camera.rotation_euler = base_rotation.copy()
        camera.constraints.remove(constraint)
        bpy.data.objects.remove(origin_target)
        bpy.data.objects.remove(target)  # Remove unused target
        
        camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
        camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
        camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
        base_rotation = camera.rotation_euler.copy()
        
        log.info(f"Camera using ROTATION (Pan/Tilt) with body at origin...")
        
        # Get pan/tilt axis configuration
        # Same logic as "Baked into Camera" rotation mode
        if up_axis == "Y":
            pan_axis = 1
            tilt_axis = 0
            tilt_sign = -1  # ty > 0 → tilt UP (negative X) → origin appears lower
            pan_sign = 1    # tx > 0 → pan LEFT (positive Y) → origin appears right
        elif up_axis == "Z":
            pan_axis = 2
            tilt_axis = 0
            tilt_sign = -1
            pan_sign = 1
        elif up_axis == "-Y":
            pan_axis = 1
            tilt_axis = 0
            tilt_sign = 1
            pan_sign = -1
        elif up_axis == "-Z":
            pan_axis = 2
            tilt_axis = 0
            tilt_sign = 1
            pan_sign = -1
        else:
            pan_axis = 1
            tilt_axis = 0
            tilt_sign = -1
            pan_sign = 1
        
        for frame_idx, frame_data in enumerate(all_frames):
            frame_cam_t = frame_data.get("pred_cam_t")
            
            frame_distance = cam_distance
            if frame_cam_t and len(frame_cam_t) > 2:
                frame_distance = abs(frame_cam_t[2])
            
            if frame_cam_t and len(frame_cam_t) >= 3:
                tx, ty, tz = frame_cam_t[0], frame_cam_t[1], frame_cam_t[2]
                
                # angle = atan2(offset, depth) to match 2D projection
                depth = abs(tz) if abs(tz) > 0.1 else 0.1
                pan_angle = math.atan2(tx, depth) * pan_sign
                tilt_angle = math.atan2(ty, depth) * tilt_sign
                
                camera.rotation_euler = base_rotation.copy()
                camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                
                camera.location = base_dir * frame_distance
                
                camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset + frame_idx)
                camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
        
        log.info(f"Camera ROTATES (pan/tilt) over {len(all_frames)} frames, body at origin (up_axis={up_axis})")
    
    else:
        # Static camera - positioned with offset for alignment
        camera.location = base_dir * cam_distance + target_offset
        
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        
        # Set camera up axis to match scene up axis
        if up_axis == "Y" or up_axis == "-Y":
            constraint.up_axis = 'UP_Y'
        elif up_axis == "Z" or up_axis == "-Z":
            constraint.up_axis = 'UP_Z'
        else:
            constraint.up_axis = 'UP_Y'
        
        bpy.context.view_layer.update()
        camera.rotation_euler = camera.matrix_world.to_euler()
        
        constraint = camera.constraints.get("Track To")
        if constraint:
            camera.constraints.remove(constraint)
        
        bpy.data.objects.remove(target)
        
        camera.rotation_euler.x = round(camera.rotation_euler.x, 4)
        camera.rotation_euler.y = round(camera.rotation_euler.y, 4)
        camera.rotation_euler.z = round(camera.rotation_euler.z, 4)
        
        log.info(f"Camera static at {camera.location}, up_axis={up_axis}")
    
    # For camera_follow_root: animate camera LOCAL position or rotation
    # Camera is parented to root_locator, so we need local animation
    # to show character at correct screen position (not always centered)
    # 
    # EXCEPTION: If camera_static is True, skip ALL camera animation!
    # In static mode, body_offset positions the body correctly and camera stays fixed.
    if camera_follow_root and not camera_static:
        log.info(f"Adding local animation for camera (follows root locator)...")
        
        # Get base camera direction based on up_axis
        if up_axis == "Y":
            base_dir = Vector((0, 0, 1))
            pan_axis = 1   # Y axis for pan
            tilt_axis = 0  # X axis for tilt
        elif up_axis == "Z":
            base_dir = Vector((0, 1, 0))
            pan_axis = 2   # Z axis for pan
            tilt_axis = 0  # X axis for tilt
        elif up_axis == "-Y":
            base_dir = Vector((0, 0, -1))
            pan_axis = 1
            tilt_axis = 0
        elif up_axis == "-Z":
            base_dir = Vector((0, -1, 0))
            pan_axis = 2
            tilt_axis = 0
        else:
            base_dir = Vector((0, 0, 1))
            pan_axis = 1
            tilt_axis = 0
        
        # Store base rotation from static camera setup
        base_rotation = camera.rotation_euler.copy()
        
        
        if camera_use_rotation:
            # ROTATION MODE: Camera pans/tilts to follow character (like real camera operator)
            log.info(f"Using PAN/TILT rotation to frame character")
            
            # Check if we have solved camera rotations from Camera Solver
            has_solved = solved_camera_rotations is not None and len(solved_camera_rotations) > 0
            
            if has_solved:
                # USE SOLVED ROTATIONS FROM CAMERA SOLVER
                # These come from background tracking and represent actual camera movement
                # body_offset (from frame 0) handles initial positioning
                # solved rotations handle frame-to-frame camera pan/tilt
                log.info(f"Using SOLVED camera rotations ({len(solved_camera_rotations)} frames)")
                
                for i in range(min(3, len(solved_camera_rotations))):
                    sr = solved_camera_rotations[i]
                
                # Get first frame depth for camera distance
                first_cam_t = all_frames[0].get("pred_cam_t", [0, 0, 5])
                base_depth = abs(first_cam_t[2]) if first_cam_t and len(first_cam_t) > 2 else 5.0
                
                for frame_idx in range(len(all_frames)):
                    if frame_idx < len(solved_camera_rotations):
                        solved_rot = solved_camera_rotations[frame_idx]
                        pan_angle = solved_rot.get("pan", 0.0)
                        tilt_angle = solved_rot.get("tilt", 0.0)
                        roll_angle = solved_rot.get("roll", 0.0)
                    else:
                        pan_angle = 0.0
                        tilt_angle = 0.0
                        roll_angle = 0.0
                    
                    if frame_idx == 0 or frame_idx == 24:
                        final_x = base_rotation[tilt_axis] + tilt_angle
                        final_y = base_rotation[pan_axis] + pan_angle
                        final_z = base_rotation[2] + roll_angle
                    
                    # Get per-frame depth (camera distance can vary)
                    frame_cam_t = all_frames[frame_idx].get("pred_cam_t")
                    if frame_cam_t and len(frame_cam_t) > 2:
                        depth = abs(frame_cam_t[2])
                    else:
                        depth = base_depth
                    
                    # Apply solved rotation
                    # Note: solved rotations are already in the correct sign convention
                    camera.rotation_euler = base_rotation.copy()
                    camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                    camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                    camera.rotation_euler[2] = base_rotation[2] + roll_angle
                    
                    # Camera distance from root
                    camera.location = base_dir * depth
                    
                    camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset + frame_idx)
                    camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
                
                log.info(f"Camera rotation from SOLVED values (background tracking)")
                log.info(f"body_offset is per-frame animated - camera rotation handles pan/tilt")
                
                # Add custom properties for debugging rotation in Maya
                # These show the intended rotation values before FBX axis conversion
                first_rot = solved_camera_rotations[0] if solved_camera_rotations else {}
                last_rot = solved_camera_rotations[-1] if solved_camera_rotations else {}
                camera["intended_rotation_x_deg"] = f"{math.degrees(base_rotation[0] + first_rot.get('tilt', 0)):.2f}"
                camera["intended_rotation_y_deg"] = f"{math.degrees(base_rotation[1] + first_rot.get('pan', 0)):.2f}"
                camera["intended_rotation_z_deg"] = f"{math.degrees(base_rotation[2] + first_rot.get('roll', 0)):.2f}"
                camera["note_rotation"] = "FBX axis conversion adds ~90deg. Use CameraExtrinsics locator for raw values."
            
            else:
                # FALLBACK: Compute pan/tilt from pred_cam_t
                # This is per-body and may cause issues with multi-body alignment
                log.info(f"No solved rotations - computing from pred_cam_t (fallback)")
                
                # Collect all camera values first for smoothing
                all_tx = []
                all_ty = []
                all_tz = []
                for frame_data in all_frames:
                    frame_cam_t = frame_data.get("pred_cam_t", [0, 0, 3])
                    if frame_cam_t and len(frame_cam_t) >= 3:
                        all_tx.append(frame_cam_t[0])
                        all_ty.append(frame_cam_t[1])
                        all_tz.append(frame_cam_t[2])
                    else:
                        all_tx.append(0)
                        all_ty.append(0)
                        all_tz.append(3)
                
                # Apply smoothing if requested
                if camera_smoothing > 1:
                    all_tx = smooth_array(all_tx, camera_smoothing)
                    all_ty = smooth_array(all_ty, camera_smoothing)
                    all_tz = smooth_array(all_tz, camera_smoothing)
                    log.info(f"Applied camera smoothing (window={camera_smoothing})")
                
                # Print first frame values for debugging
                log.info(f"Frame 0 pred_cam_t: tx={all_tx[0]:.3f}, ty={all_ty[0]:.3f}, tz={all_tz[0]:.3f}")
                
                for frame_idx in range(len(all_frames)):
                    tx = all_tx[frame_idx]
                    ty = all_ty[frame_idx]
                    tz = all_tz[frame_idx]
                    depth = abs(tz) if abs(tz) > 0.1 else 0.1
                    
                    # Apply coordinate transform for pan (horizontal)
                    # flip_x affects whether we negate tx
                    if up_axis == "Y" or up_axis == "-Y":
                        tx_cam = tx if flip_x else -tx
                        # ty is NOT negated - positive ty means body lower in frame,
                        # so camera should tilt DOWN (positive X rotation)
                        ty_cam = ty
                    else:  # Z-up
                        tx_cam = tx if flip_x else -tx
                        ty_cam = ty
                    
                    # Compute angles directly using atan2
                    pan_angle = math.atan2(tx_cam, depth)
                    tilt_angle = math.atan2(ty_cam, depth)
                    
                    # Apply rotation
                    camera.rotation_euler = base_rotation.copy()
                    camera.rotation_euler[pan_axis] = base_rotation[pan_axis] + pan_angle
                    camera.rotation_euler[tilt_axis] = base_rotation[tilt_axis] + tilt_angle
                    
                    # Also animate depth (camera distance from root)
                    camera.location = base_dir * depth
                    
                    camera.keyframe_insert(data_path="rotation_euler", frame=frame_offset + frame_idx)
                    camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
                
                log.info(f"Camera pan/tilt animated over {len(all_frames)} frames")
                log.info(f"Camera ROTATES to follow character (from pred_cam_t fallback)")
        
        else:
            # TRANSLATION MODE: Camera moves laterally to show character at offset position
            log.info(f"Using local TRANSLATION to frame character")
            
            for frame_idx, frame_data in enumerate(all_frames):
                frame_cam_t = frame_data.get("pred_cam_t")
                
                if frame_cam_t and len(frame_cam_t) >= 3:
                    tx, ty, tz = frame_cam_t[0], frame_cam_t[1], frame_cam_t[2]
                    depth = abs(tz) if tz else 3.0
                    
                    # Root locator moves by world_offset = (tx * depth * 0.5, ty * depth * 0.5)
                    # To show character at correct screen position, camera local position
                    # should have the INVERSE lateral offset
                    lateral_x = -tx * depth * 0.5
                    lateral_y = -ty * depth * 0.5
                    
                    # Apply based on up_axis
                    if up_axis == "Y":
                        local_offset = Vector((lateral_x, -lateral_y, 0))
                    elif up_axis == "Z":
                        local_offset = Vector((lateral_x, 0, -lateral_y))
                    elif up_axis == "-Y":
                        local_offset = Vector((lateral_x, lateral_y, 0))
                    elif up_axis == "-Z":
                        local_offset = Vector((lateral_x, 0, lateral_y))
                    else:
                        local_offset = Vector((lateral_x, -lateral_y, 0))
                    
                    # Camera position = forward direction * depth + lateral offset
                    camera.location = base_dir * depth + local_offset
                    camera.keyframe_insert(data_path="location", frame=frame_offset + frame_idx)
            
            log.info(f"Camera local translation animated over {len(all_frames)} frames")
            log.info(f"Camera TRANSLATES to keep character at correct screen position")
    
    elif camera_follow_root and camera_static:
        # Static camera mode - no animation needed
        # body_offset positions the body correctly relative to the fixed camera
        log.info(f"Camera STATIC - body_offset positions character, no camera animation")
    
    # Animate focal length if it varies across frames
    focal_lengths = []
    for frame_data in all_frames:
        fl = frame_data.get("focal_length")
        if fl is not None:
            if isinstance(fl, (list, tuple)):
                fl = fl[0]
            focal_lengths.append(fl)
    
    if len(focal_lengths) > 1:
        # Check if focal length actually varies
        fl_min, fl_max = min(focal_lengths), max(focal_lengths)
        if fl_max - fl_min > 1.0:  # More than 1 pixel difference
            log.info(f"Animating focal length: {fl_min:.0f}px to {fl_max:.0f}px")
            
            for frame_idx, frame_data in enumerate(all_frames):
                fl = frame_data.get("focal_length")
                if fl is not None:
                    if isinstance(fl, (list, tuple)):
                        fl = fl[0]
                    focal_mm = fl * (sensor_width / image_width)
                    cam_data.lens = focal_mm
                    cam_data.keyframe_insert(data_path="lens", frame=frame_offset + frame_idx)
            
            log.info(f"Focal length animated over {len(all_frames)} frames")
        else:
            log.info(f"Focal length constant at ~{fl_min:.0f}px")
    
    return camera


def create_metadata_locator(metadata: dict):
    """
    Create a metadata locator with custom properties for FBX export.
    
    These properties become Extra Attributes in Maya, accessible via:
        cmds.getAttr("SAM3DBody_Metadata.world_translation")
    
    All values are stored as strings to appear as text fields (not sliders) in Maya.
    
    Args:
        metadata: Dict of metadata to embed
    
    Returns:
        The metadata empty object
    """
    if not metadata:
        log.info("No metadata to embed")
        return None
    
    # Create empty object as metadata container
    metadata_obj = bpy.data.objects.new("SAM3DBody_Metadata", None)
    metadata_obj.empty_display_type = 'PLAIN_AXES'
    metadata_obj.empty_display_size = 0.1
    bpy.context.collection.objects.link(metadata_obj)
    
    # Add custom properties
    # Note: All values stored as STRINGS to appear as text fields (not sliders) in Maya
    for key, value in metadata.items():
        if value is None:
            continue
        
        # Convert ALL values to strings for Maya text field display
        if isinstance(value, bool):
            metadata_obj[key] = "true" if value else "false"
        elif isinstance(value, float):
            # Format floats nicely (3 decimal places for most, more for small values)
            if abs(value) < 0.001 and value != 0:
                metadata_obj[key] = f"{value:.6f}"
            else:
                metadata_obj[key] = f"{value:.3f}"
        elif isinstance(value, int):
            metadata_obj[key] = str(value)
        elif isinstance(value, str):
            metadata_obj[key] = value
        elif isinstance(value, (list, tuple)):
            metadata_obj[key] = str(value)
        else:
            metadata_obj[key] = str(value)
    
    log.info(f"Metadata locator created with {len(metadata)} properties:")
    for key, value in metadata.items():
        if value is not None:
            log.debug(f"  {key}: {value}")
    
    return metadata_obj


def create_world_position_locator(trajectory_compensated: list, fps: float = 24.0, frame_offset: int = 1):
    """
    Create WorldPosition locator with TRUE character trajectory (camera effects removed).
    
    This locator shows where the character actually moved in world space,
    with camera pan/tilt/zoom effects removed.
    
    Args:
        trajectory_compensated: List of [X, Y, Z] positions with camera effects removed
        fps: Frame rate for animation
        frame_offset: Starting frame number (usually 1)
    
    Returns:
        The WorldPosition empty object
    """
    if not trajectory_compensated or len(trajectory_compensated) < 2:
        log.info("No compensated trajectory data for WorldPosition locator")
        return None
    
    # Create empty object as locator
    loc_obj = bpy.data.objects.new("SAM3DBody_WorldPosition", None)
    loc_obj.empty_display_type = 'PLAIN_AXES'
    loc_obj.empty_display_size = 0.2
    bpy.context.collection.objects.link(loc_obj)
    
    # Ensure animation data exists
    if loc_obj.animation_data is None:
        loc_obj.animation_data_create()
    
    # Create action for keyframes
    action = bpy.data.actions.new(name="WorldPositionAction")
    loc_obj.animation_data.action = action
    
    # Create F-curves for X, Y, Z location
    fcurve_x = action.fcurves.new(data_path="location", index=0)
    fcurve_y = action.fcurves.new(data_path="location", index=1)
    fcurve_z = action.fcurves.new(data_path="location", index=2)
    
    # Add keyframes for each frame
    for i, pos in enumerate(trajectory_compensated):
        frame_num = frame_offset + i
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        fcurve_x.keyframe_points.insert(frame_num, x)
        fcurve_y.keyframe_points.insert(frame_num, y)
        fcurve_z.keyframe_points.insert(frame_num, z)
    
    # Set interpolation to linear
    for fcurve in [fcurve_x, fcurve_y, fcurve_z]:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'LINEAR'
    
    # Add custom properties
    import numpy as np
    traj_arr = np.array(trajectory_compensated)
    displacement = traj_arr[-1] - traj_arr[0]
    velocities = np.diff(traj_arr, axis=0)
    total_dist = np.sum(np.linalg.norm(velocities, axis=1))
    
    loc_obj["description"] = "True world position with camera effects removed"
    loc_obj["total_frames"] = str(len(trajectory_compensated))
    loc_obj["total_distance_m"] = f"{total_dist:.3f}"
    loc_obj["displacement_x"] = f"{displacement[0]:.3f}"
    loc_obj["displacement_y"] = f"{displacement[1]:.3f}"
    loc_obj["displacement_z"] = f"{displacement[2]:.3f}"
    
    log.info(f"WorldPosition locator created with {len(trajectory_compensated)} keyframes")
    log.debug(f"  Start: ({trajectory_compensated[0][0]:.3f}, {trajectory_compensated[0][1]:.3f}, {trajectory_compensated[0][2]:.3f})")
    log.debug(f"  End: ({trajectory_compensated[-1][0]:.3f}, {trajectory_compensated[-1][1]:.3f}, {trajectory_compensated[-1][2]:.3f})")
    
    return loc_obj


def create_camera_extrinsics_locator(camera_extrinsics: list, fps: float = 24.0, frame_offset: int = 1, up_axis: str = "Y"):
    """
    Create CameraExtrinsics locator with animated rotation (pan/tilt/roll).
    
    This locator represents the camera's rotation over time from the CameraSolver.
    The rotation is stored as animated transform so it can be directly applied or
    used as a reference.
    
    Args:
        camera_extrinsics: List of dicts with 'pan', 'tilt', 'roll' keys (in radians)
        fps: Frame rate for animation
        frame_offset: Starting frame number (usually 1)
        up_axis: Up axis for coordinate system
    
    Returns:
        The CameraExtrinsics empty object
    """
    if not camera_extrinsics or len(camera_extrinsics) < 2:
        log.info("No camera extrinsics data for CameraExtrinsics locator")
        return None
    
    # Create empty object as locator
    loc_obj = bpy.data.objects.new("SAM3DBody_CameraExtrinsics", None)
    loc_obj.empty_display_type = 'ARROWS'
    loc_obj.empty_display_size = 0.3
    bpy.context.collection.objects.link(loc_obj)
    
    # Set rotation mode to ZXY - this locator carries camera extrinsic rotation data
    loc_obj.rotation_mode = 'ZXY'
    
    # Ensure animation data exists
    if loc_obj.animation_data is None:
        loc_obj.animation_data_create()
    
    # Create action for keyframes
    action = bpy.data.actions.new(name="CameraExtrinsicsAction")
    loc_obj.animation_data.action = action
    
    # Create F-curves for rotation (Euler XYZ)
    # Map: X = tilt, Y = pan, Z = roll
    fcurve_rx = action.fcurves.new(data_path="rotation_euler", index=0)
    fcurve_ry = action.fcurves.new(data_path="rotation_euler", index=1)
    fcurve_rz = action.fcurves.new(data_path="rotation_euler", index=2)
    
    # Add keyframes for each frame
    for i, ext in enumerate(camera_extrinsics):
        frame_num = frame_offset + i
        pan = ext.get("pan", 0.0)
        tilt = ext.get("tilt", 0.0)
        roll = ext.get("roll", 0.0)
        
        # Store as: X=tilt, Y=pan, Z=roll (standard camera convention)
        fcurve_rx.keyframe_points.insert(frame_num, tilt)
        fcurve_ry.keyframe_points.insert(frame_num, pan)
        fcurve_rz.keyframe_points.insert(frame_num, roll)
    
    # Set interpolation to linear
    for fcurve in [fcurve_rx, fcurve_ry, fcurve_rz]:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'LINEAR'
    
    # Add custom properties with final values in degrees
    first_ext = camera_extrinsics[0]
    last_ext = camera_extrinsics[-1]
    
    loc_obj["description"] = "Camera rotation from CameraSolver (X=tilt, Y=pan, Z=roll)"
    loc_obj["rotation_order"] = "XYZ"
    loc_obj["total_frames"] = str(len(camera_extrinsics))
    loc_obj["frame_0_pan_deg"] = f"{math.degrees(first_ext.get('pan', 0)):.4f}"
    loc_obj["frame_0_tilt_deg"] = f"{math.degrees(first_ext.get('tilt', 0)):.4f}"
    loc_obj["frame_0_roll_deg"] = f"{math.degrees(first_ext.get('roll', 0)):.4f}"
    loc_obj["final_pan_deg"] = f"{math.degrees(last_ext.get('pan', 0)):.4f}"
    loc_obj["final_tilt_deg"] = f"{math.degrees(last_ext.get('tilt', 0)):.4f}"
    loc_obj["final_roll_deg"] = f"{math.degrees(last_ext.get('roll', 0)):.4f}"
    
    log.info(f"CameraExtrinsics locator created with {len(camera_extrinsics)} keyframes")
    log.debug(f"  Frame 0: pan={math.degrees(first_ext.get('pan', 0)):.2f}°, tilt={math.degrees(first_ext.get('tilt', 0)):.2f}°")
    log.debug(f"  Final: pan={math.degrees(last_ext.get('pan', 0)):.2f}°, tilt={math.degrees(last_ext.get('tilt', 0)):.2f}°")
    
    return loc_obj


def create_screen_position_locator(frames: list, fps: float = 24.0, frame_offset: int = 1, up_axis: str = "Y"):
    """
    Create ScreenPosition locator showing character's screen-space position (from pred_cam_t).
    
    This locator shows tx, ty values which represent where the character appears
    on screen relative to center. Useful for compositing reference.
    
    Args:
        frames: List of frame data dicts with 'pred_cam_t'
        fps: Frame rate for animation  
        frame_offset: Starting frame number
        up_axis: Up axis for coordinate system
    
    Returns:
        The ScreenPosition empty object
    """
    if not frames or len(frames) < 2:
        log.info("No frame data for ScreenPosition locator")
        return None
    
    # Create empty object as locator
    loc_obj = bpy.data.objects.new("SAM3DBody_ScreenPosition", None)
    loc_obj.empty_display_type = 'CIRCLE'
    loc_obj.empty_display_size = 0.15
    bpy.context.collection.objects.link(loc_obj)
    
    # Ensure animation data exists
    if loc_obj.animation_data is None:
        loc_obj.animation_data_create()
    
    # Create action for keyframes
    action = bpy.data.actions.new(name="ScreenPositionAction")
    loc_obj.animation_data.action = action
    
    # Create F-curves for X, Y location (screen-space)
    fcurve_x = action.fcurves.new(data_path="location", index=0)
    fcurve_y = action.fcurves.new(data_path="location", index=1)
    fcurve_z = action.fcurves.new(data_path="location", index=2)
    
    # Add keyframes for each frame
    for i, frame_data in enumerate(frames):
        frame_num = frame_offset + i
        pred_cam_t = frame_data.get("pred_cam_t", [0, 0, 5])
        
        tx = pred_cam_t[0] if pred_cam_t and len(pred_cam_t) > 0 else 0
        ty = pred_cam_t[1] if pred_cam_t and len(pred_cam_t) > 1 else 0
        tz = pred_cam_t[2] if pred_cam_t and len(pred_cam_t) > 2 else 5
        
        # Store screen position: X=tx, Y=ty (negated for camera convention), Z=depth
        if up_axis == "Y":
            fcurve_x.keyframe_points.insert(frame_num, tx)
            fcurve_y.keyframe_points.insert(frame_num, -ty)  # Negate for camera view
            fcurve_z.keyframe_points.insert(frame_num, 0)  # Screen space, no Z
        else:
            fcurve_x.keyframe_points.insert(frame_num, tx)
            fcurve_y.keyframe_points.insert(frame_num, 0)
            fcurve_z.keyframe_points.insert(frame_num, -ty)
    
    # Set interpolation to linear
    for fcurve in [fcurve_x, fcurve_y, fcurve_z]:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'LINEAR'
    
    # Add custom properties
    first_cam_t = frames[0].get("pred_cam_t", [0, 0, 5])
    last_cam_t = frames[-1].get("pred_cam_t", [0, 0, 5])
    
    loc_obj["description"] = "Screen-space position from pred_cam_t (tx, ty)"
    loc_obj["total_frames"] = str(len(frames))
    loc_obj["frame_0_tx"] = f"{first_cam_t[0]:.4f}"
    loc_obj["frame_0_ty"] = f"{first_cam_t[1]:.4f}"
    loc_obj["final_tx"] = f"{last_cam_t[0]:.4f}"
    loc_obj["final_ty"] = f"{last_cam_t[1]:.4f}"
    
    log.info(f"ScreenPosition locator created with {len(frames)} keyframes")
    log.debug(f"  Frame 0: tx={first_cam_t[0]:.3f}, ty={first_cam_t[1]:.3f}")
    log.debug(f"  Final: tx={last_cam_t[0]:.3f}, ty={last_cam_t[1]:.3f}")
    
    return loc_obj


def apply_per_frame_body_offset(mesh_obj, armature_obj, frames: list, up_axis: str, frame_offset: int = 1, 
                                 use_depth: bool = True, depth_mode: str = "position", scale_factor: float = 1.0):
    """
    Apply per-frame body offset based on pred_cam_t for each frame.
    
    FIXED in v4.8.8: Now correctly converts screen coordinates to world coordinates
    by multiplying tx/ty by depth and scale_factor, matching Motion Analyzer calculation.
    
    When character moves toward/away from camera:
    - tx, ty are screen-space offsets
    - tz is depth
    - World position = screen_pos * depth * scale_factor
    
    If tracked_depth is available (from CharacterTrajectoryTracker), it will be
    used instead of pred_cam_t.tz for more accurate depth.
    
    Args:
        mesh_obj: The mesh object to animate
        armature_obj: The armature object to animate
        frames: List of frame data with pred_cam_t
        up_axis: Up axis for coordinate system
        frame_offset: Starting frame number
        use_depth: If True, apply depth-based positioning/scaling.
                   If False, no depth handling (legacy behavior).
        depth_mode: How to handle depth changes:
                   - "position": Character moves in Z axis (RECOMMENDED)
                   - "scale": Mesh scales with depth (2D compositing only)
                   - "both": Both Z movement AND scaling
        scale_factor: Scale factor from Motion Analyzer for consistent world units
    """
    if not frames:
        log.info("No frames for per-frame body offset")
        return
    
    log.info(f"Applying per-frame body offset ({len(frames)} frames)...")
    log.info(f"  Depth settings: use_depth={use_depth}, depth_mode='{depth_mode}', scale_factor={scale_factor:.3f}")
    
    # Check if tracked depth is available (from CharacterTrajectoryTracker)
    has_tracked_depth = "tracked_depth" in frames[0] if frames else False
    if has_tracked_depth:
        log.info(f"  Depth source: tracked_depth (from Character Trajectory Tracker + DepthAnything V2)")
    else:
        log.info(f"  Depth source: pred_cam_t[2] (from SAM3DBody)")
    
    # Get reference depth from first frame
    first_frame = frames[0]
    if has_tracked_depth:
        ref_depth = abs(first_frame.get("tracked_depth", 5.0))
    else:
        first_cam_t = first_frame.get("pred_cam_t", [0, 0, 5])
        ref_depth = abs(first_cam_t[2]) if len(first_cam_t) > 2 and abs(first_cam_t[2]) > 0.1 else 5.0
    
    log.info(f"  Reference depth (frame 0): {ref_depth:.3f}m")
    
    # Track depth range for debug
    min_depth = ref_depth
    max_depth = ref_depth
    
    for i, frame_data in enumerate(frames):
        frame_num = frame_offset + i
        pred_cam_t = frame_data.get("pred_cam_t", [0, 0, 5])
        
        tx = pred_cam_t[0] if pred_cam_t and len(pred_cam_t) > 0 else 0
        ty = pred_cam_t[1] if pred_cam_t and len(pred_cam_t) > 1 else 0
        
        # Get depth - prefer tracked_depth if available
        if has_tracked_depth and "tracked_depth" in frame_data:
            frame_depth = abs(frame_data["tracked_depth"])
        else:
            tz = pred_cam_t[2] if pred_cam_t and len(pred_cam_t) > 2 else 5
            frame_depth = abs(tz) if abs(tz) > 0.1 else ref_depth
        
        min_depth = min(min_depth, frame_depth)
        max_depth = max(max_depth, frame_depth)
        
        # Convert screen position (tx, ty) to world coordinates
        # In weak perspective: world_pos = screen_pos * depth * scale_factor
        # This matches how body_world_3d is calculated in Motion Analyzer:
        #   body_world_3d = [tx * tz * scale_factor, ty * tz * scale_factor, tz * scale_factor]
        world_x = tx * frame_depth * scale_factor
        world_y = ty * frame_depth * scale_factor
        
        # Z position (only if position mode)
        if use_depth and depth_mode in ["position", "both"]:
            depth_delta = frame_depth - ref_depth
            world_z = -depth_delta  # Negative = toward camera
        else:
            world_z = 0
        
        # Mesh scale factor for scale mode: closer = smaller mesh
        # (This is separate from scale_factor parameter which is for world coordinate conversion)
        if use_depth and depth_mode in ["scale", "both"]:
            mesh_scale = frame_depth / ref_depth if ref_depth > 0 else 1.0
        else:
            mesh_scale = 1.0
        
        # Debug logging for first frame
        if i == 0:
            log.info(f"  Frame 0 world coords: X={world_x:.3f}m, Y={world_y:.3f}m, Z={world_z:.3f}m")
            log.debug(f"    Calculation: tx={tx:.4f} * depth={frame_depth:.2f}m * scale={scale_factor:.3f}")
        
        # Compute body offset for this frame
        # Apply offset options from globals
        if DISABLE_ALL_OFFSETS:
            world_x_adjusted = 0
            world_y_adjusted = 0
        else:
            # Horizontal
            if DISABLE_HORIZONTAL_OFFSET:
                world_x_adjusted = 0
            elif FLIP_HORIZONTAL:
                world_x_adjusted = -world_x
            else:
                world_x_adjusted = world_x
            
            # Vertical - also negate by default for camera convention
            if DISABLE_VERTICAL_OFFSET:
                world_y_adjusted = 0
            elif FLIP_VERTICAL:
                world_y_adjusted = world_y  # Don't negate
            else:
                world_y_adjusted = -world_y  # Default: negate for camera convention
        
        if up_axis == "Y":
            offset = Vector((world_x_adjusted, world_y_adjusted, world_z))
        elif up_axis == "Z":
            offset = Vector((world_x_adjusted, world_z, world_y_adjusted))
        elif up_axis == "-Y":
            offset = Vector((world_x_adjusted, -world_y_adjusted, -world_z))
        elif up_axis == "-Z":
            offset = Vector((world_x_adjusted, -world_z, -world_y_adjusted))
        else:
            offset = Vector((world_x_adjusted, world_y_adjusted, world_z))
        
        # Apply to mesh
        if mesh_obj:
            mesh_obj.location = offset
            mesh_obj.keyframe_insert(data_path="location", frame=frame_num)
            if use_depth and depth_mode in ["scale", "both"]:
                mesh_obj.scale = Vector((mesh_scale, mesh_scale, mesh_scale))
                mesh_obj.keyframe_insert(data_path="scale", frame=frame_num)
        
        # Apply to armature
        if armature_obj:
            armature_obj.location = offset
            armature_obj.keyframe_insert(data_path="location", frame=frame_num)
            if use_depth and depth_mode in ["scale", "both"]:
                armature_obj.scale = Vector((mesh_scale, mesh_scale, mesh_scale))
                armature_obj.keyframe_insert(data_path="scale", frame=frame_num)
    
    # Debug output
    last_frame = frames[-1]
    if has_tracked_depth and "tracked_depth" in last_frame:
        last_depth = abs(last_frame["tracked_depth"])
    else:
        last_cam_t = last_frame.get("pred_cam_t", [0, 0, 5])
        last_depth = abs(last_cam_t[2]) if len(last_cam_t) > 2 else ref_depth
    
    depth_change = last_depth - ref_depth
    depth_pct = (depth_change / ref_depth * 100) if ref_depth > 0 else 0
    
    first_cam_t = frames[0].get("pred_cam_t", [0, 0, 5])
    last_cam_t = frames[-1].get("pred_cam_t", [0, 0, 5])
    
    log.info(f"  Depth range: {min_depth:.2f}m to {max_depth:.2f}m (change: {depth_change:+.2f}m, {depth_pct:+.1f}%)")
    log.debug(f"  Frame 0: tx={first_cam_t[0]:.3f}, ty={first_cam_t[1]:.3f}, depth={ref_depth:.3f}")
    log.debug(f"  Frame {len(frames)-1}: tx={last_cam_t[0]:.3f}, ty={last_cam_t[1]:.3f}, depth={last_depth:.3f}")
    
    if use_depth and depth_mode in ["scale", "both"]:
        min_scale = min_depth / ref_depth if ref_depth > 0 else 1.0
        max_scale = max_depth / ref_depth if ref_depth > 0 else 1.0
        log.info(f"  Scale range: {min_scale:.3f} to {max_scale:.3f}")
    
    if mesh_obj:
        log.info(f"Mesh animated with {len(frames)} keyframes")
    if armature_obj:
        log.info(f"Skeleton animated with {len(frames)} keyframes")


def create_trajectory_locator(trajectory: list, fps: float = 24.0, frame_offset: int = 1):
    """
    Create an animated locator that follows the body world trajectory.
    
    The trajectory comes from pred_cam_t and represents the body's position
    in world/camera space over time. This is useful for:
    - Visualizing the motion path in Maya/Blender
    - Constraining other objects to follow the path
    - Analyzing movement patterns
    
    Args:
        trajectory: List of [X, Y, Z] positions for each frame (Y-up coordinate system)
        fps: Frame rate for animation
        frame_offset: Starting frame number (usually 1)
    
    Returns:
        The trajectory empty object
    """
    if not trajectory or len(trajectory) < 2:
        log.info("No trajectory data to create locator")
        return None
    
    # Create empty object as trajectory locator
    traj_obj = bpy.data.objects.new("SAM3DBody_Trajectory", None)
    traj_obj.empty_display_type = 'SPHERE'
    traj_obj.empty_display_size = 0.1
    bpy.context.collection.objects.link(traj_obj)
    
    # Ensure animation data exists
    if traj_obj.animation_data is None:
        traj_obj.animation_data_create()
    
    # Create action for keyframes
    action = bpy.data.actions.new(name="TrajectoryAction")
    traj_obj.animation_data.action = action
    
    # Create F-curves for X, Y, Z location
    fcurve_x = action.fcurves.new(data_path="location", index=0)
    fcurve_y = action.fcurves.new(data_path="location", index=1)
    fcurve_z = action.fcurves.new(data_path="location", index=2)
    
    # Add keyframes for each frame
    for i, pos in enumerate(trajectory):
        frame_num = frame_offset + i
        
        # Position is already in Y-up coordinate system from motion_analyzer
        # [X, Y, Z] where Y is up
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        
        # Insert keyframes
        fcurve_x.keyframe_points.insert(frame_num, x)
        fcurve_y.keyframe_points.insert(frame_num, y)
        fcurve_z.keyframe_points.insert(frame_num, z)
    
    # Set interpolation to linear for accurate path
    for fcurve in [fcurve_x, fcurve_y, fcurve_z]:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'LINEAR'
    
    log.info(f"Trajectory locator created with {len(trajectory)} keyframes")
    log.debug(f"  Start pos: ({trajectory[0][0]:.3f}, {trajectory[0][1]:.3f}, {trajectory[0][2]:.3f})")
    log.debug(f"  End pos: ({trajectory[-1][0]:.3f}, {trajectory[-1][1]:.3f}, {trajectory[-1][2]:.3f})")
    
    # Add trajectory info as custom properties
    import numpy as np
    traj_arr = np.array(trajectory)
    displacement = traj_arr[-1] - traj_arr[0]
    velocities = np.diff(traj_arr, axis=0)
    total_dist = np.sum(np.linalg.norm(velocities, axis=1))
    
    traj_obj["total_frames"] = str(len(trajectory))
    traj_obj["total_distance_m"] = f"{total_dist:.3f}"
    traj_obj["displacement_x"] = f"{displacement[0]:.3f}"
    traj_obj["displacement_y"] = f"{displacement[1]:.3f}"
    traj_obj["displacement_z"] = f"{displacement[2]:.3f}"
    
    return traj_obj


def export_fbx(output_path, axis_forward, axis_up):
    """Export to FBX."""
    log.info(f"Exporting FBX: {output_path}")
    log.info(f"Orientation: forward={axis_forward}, up={axis_up}")
    log.info(f"Scale: 0.01 (meter to centimeter), Apply Scalings: All Local")
    
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        global_scale=0.01,  # Meter to Centimeter - Maya will show scale=1
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',  # "All Local" - correct axis remap
        bake_space_transform=False,  # Apply Transform: OFF
        axis_forward=axis_forward,
        axis_up=axis_up,
        object_types={'MESH', 'ARMATURE', 'EMPTY', 'CAMERA'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        use_armature_deform_only=False,
        add_leaf_bones=False,
        use_custom_props=True,  # Export custom properties for Maya Extra Attributes
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
    )
    log.info(f"FBX export complete")


def export_alembic(output_path):
    """Export to Alembic (.abc)."""
    log.info(f"Exporting Alembic: {output_path}")
    
    bpy.ops.wm.alembic_export(
        filepath=output_path,
        start=bpy.context.scene.frame_start,
        end=bpy.context.scene.frame_end,
        selected=False,
        visible_objects_only=True,
        flatten=False,
        uvs=True,
        normals=True,
        vcolors=False,
        apply_subdiv=False,
        curves_as_mesh=False,
        use_instancing=True,
        global_scale=1.0,
        triangulate=False,
        export_hair=False,
        export_particles=False,
        packuv=True,
    )
    log.info(f"Alembic export complete")


def main():
    argv = sys.argv
    try:
        idx = argv.index("--")
        args = argv[idx + 1:]
    except ValueError:
        log.info("Error: No arguments")
        sys.exit(1)
    
    if len(args) < 2:
        log.info("Usage: blender --background --python script.py -- input.json output.fbx [up_axis] [include_mesh] [include_camera]")
        sys.exit(1)
    
    input_json = args[0]
    output_path = args[1]
    up_axis = args[2] if len(args) > 2 else "Y"
    include_mesh = args[3] == "1" if len(args) > 3 else True
    include_camera = args[4] == "1" if len(args) > 4 else True
    
    # Detect output format
    output_format = "fbx"
    if output_path.lower().endswith(".abc"):
        output_format = "abc"
    
    log.info(f"Input: {input_json}")
    log.info(f"Output: {output_path}")
    log.info(f"Format: {output_format.upper()}")
    log.info(f"Up axis: {up_axis}")
    log.info(f"Include mesh: {include_mesh}")
    log.info(f"Include camera: {include_camera}")
    
    if not os.path.exists(input_json):
        log.info(f"Error: File not found: {input_json}")
        sys.exit(1)
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    fps = data.get("fps", 24.0)
    frames = data.get("frames", [])
    faces = data.get("faces")
    joint_parents = data.get("joint_parents")  # Get hierarchy data
    sensor_width = data.get("sensor_width", 36.0)
    world_translation_mode = data.get("world_translation_mode", "none")
    skeleton_mode = data.get("skeleton_mode", "rotations")  # New: default to rotations
    flip_x = data.get("flip_x", False)  # Mirror on X axis
    disable_vertical_offset = data.get("disable_vertical_offset", False)  # v5.1.9: Disable Y offset
    disable_horizontal_offset = data.get("disable_horizontal_offset", False)  # v5.1.10: Disable X offset
    disable_all_offsets = data.get("disable_all_offsets", False)  # v5.1.10: Disable all offsets
    flip_vertical = data.get("flip_vertical", False)  # v5.1.9: Flip Y offset sign
    flip_horizontal = data.get("flip_horizontal", False)  # v5.1.10: Flip X offset sign
    # Backwards compatibility with old flip_ty option
    if data.get("flip_ty", False):
        flip_vertical = True
    frame_offset = data.get("frame_offset", 0)  # Start frame offset for Maya
    include_skeleton = data.get("include_skeleton", True)  # v4.6.10: Option to exclude skeleton
    animate_camera = data.get("animate_camera", False)  # Only animate camera if translation baked to it
    camera_follow_root = data.get("camera_follow_root", False)  # Parent camera to root locator
    camera_use_rotation = data.get("camera_use_rotation", False)  # Use rotation instead of translation
    camera_static = data.get("camera_static", False)  # Disable all camera animation
    camera_compensation = data.get("camera_compensation", False)  # Bake inverse extrinsics to root
    
    # Depth handling (v4.6.9 fix)
    use_depth_positioning = data.get("use_depth_positioning", True)  # Use pred_cam_t.tz for positioning
    depth_mode = data.get("depth_mode", "position")  # How to show depth: position, scale, or both
    scale_factor = data.get("scale_factor", 1.0)  # For consistent world coordinates (v4.8.8)
    
    # Camera data - support both old and new field names for backwards compatibility
    camera_extrinsics = data.get("camera_extrinsics") or data.get("solved_camera_rotations")
    camera_intrinsics = data.get("camera_intrinsics")  # From MoGe2
    
    # Smoothing settings
    extrinsics_smoothing_method = data.get("extrinsics_smoothing_method") or data.get("bake_smoothing_method", "kalman")
    extrinsics_smoothing_strength = data.get("extrinsics_smoothing_strength") or data.get("bake_smoothing_strength", 0.5)
    
    log.info(f"{len(frames)} frames at {fps} fps")
    log.info(f"Frame offset: {frame_offset} (animation runs from frame {frame_offset} to {frame_offset + len(frames) - 1})")
    log.info(f"Sensor width: {sensor_width}mm")
    log.info(f"Depth positioning: {use_depth_positioning}, mode: {depth_mode}, scale_factor: {scale_factor:.3f}")
    log.info(f"World translation mode: {world_translation_mode}")
    log.info(f"Skeleton mode: {skeleton_mode}")
    log.info(f"Include skeleton: {include_skeleton}")
    log.info(f"Flip X: {flip_x}")
    log.info(f"Animate camera: {animate_camera}")
    log.info(f"Camera follow root: {camera_follow_root}")
    log.info(f"Camera use rotation: {camera_use_rotation}")
    log.info(f"Camera static: {camera_static}")
    log.info(f"Camera compensation: {camera_compensation}")
    log.info(f"Camera extrinsics: {len(camera_extrinsics) if camera_extrinsics else 0} frames")
    log.info(f"Camera intrinsics: {'Yes (MoGe2)' if camera_intrinsics else 'No (using manual sensor_width)'}")
    log.info(f"Joint parents available: {joint_parents is not None}")
    
    if not frames:
        log.info("Error: No frames")
        sys.exit(1)
    
    # Check if rotation data is available
    has_rotations = frames[0].get("joint_rotations") is not None
    log.info(f"Rotation data available: {has_rotations}")
    
    if skeleton_mode == "rotations" and not has_rotations:
        log.info("Warning: Rotation mode requested but no data available. Falling back to positions.")
        skeleton_mode = "positions"
    
    # Handle root_camera_compensation mode: bake inverse camera extrinsics to root locator
    # This keeps the camera static while the root absorbs the camera motion
    root_camera_compensation_mode = False
    if world_translation_mode == "root_camera_compensation":
        if camera_extrinsics:
            log.info("MODE: Root Locator + Camera Compensation")
            log.info("  Inverse camera extrinsics will be baked into root locator")
            log.info("  Camera will be exported as static")
            root_camera_compensation_mode = True
            # Force static camera
            camera_static = True
            animate_camera = False
            camera_use_rotation = False
            # Set translation mode to "root" so mesh/skeleton are parented to root
            world_translation_mode = "root"
        else:
            log.info("Warning: root_camera_compensation mode requires camera_extrinsics. Falling back to 'root'.")
            world_translation_mode = "root"
    
    # Handle bake_to_geometry mode: apply inverse camera transforms to geometry
    bake_to_geometry_mode = False
    if world_translation_mode == "bake_to_geometry":
        if camera_extrinsics:
            log.info("MODE: Baking camera motion into geometry (static camera export)")
            log.info(f"Smoothing: {extrinsics_smoothing_method} (strength={extrinsics_smoothing_strength})")
            
            # Build smoothing params based on method
            smoothing_params = {}
            if extrinsics_smoothing_method == "kalman":
                # For Kalman: strength affects measurement noise (higher = trust measurements less)
                smoothing_params["process_noise"] = 0.01
                smoothing_params["measurement_noise"] = 0.05 + extrinsics_smoothing_strength * 0.45  # 0.05 to 0.5
            elif extrinsics_smoothing_method == "spline":
                smoothing_params["smoothing_factor"] = extrinsics_smoothing_strength
            elif extrinsics_smoothing_method == "gaussian":
                # Map strength to window size: 0=3, 0.5=9, 1.0=15
                smoothing_params["window"] = int(3 + extrinsics_smoothing_strength * 12)
            
            frames = bake_camera_to_geometry(
                frames, 
                camera_extrinsics, 
                up_axis,
                smoothing_method=extrinsics_smoothing_method,
                smoothing_params=smoothing_params
            )
            # Force static camera when geometry is baked
            camera_static = True
            animate_camera = False
            camera_use_rotation = False
            bake_to_geometry_mode = True
            # Set mode to "none" so mesh/skeleton don't apply additional transforms
            world_translation_mode = "none"
        else:
            log.info("Warning: bake_to_geometry mode requires camera_extrinsics. Falling back to 'none'.")
            world_translation_mode = "none"
    
    # Get transformation
    global FLIP_X, DISABLE_VERTICAL_OFFSET, DISABLE_HORIZONTAL_OFFSET, DISABLE_ALL_OFFSETS, FLIP_VERTICAL, FLIP_HORIZONTAL
    FLIP_X = flip_x
    DISABLE_VERTICAL_OFFSET = disable_vertical_offset
    DISABLE_HORIZONTAL_OFFSET = disable_horizontal_offset
    DISABLE_ALL_OFFSETS = disable_all_offsets
    FLIP_VERTICAL = flip_vertical
    FLIP_HORIZONTAL = flip_horizontal
    transform_func, axis_forward, axis_up_export = get_transform_for_axis(up_axis, flip_x)
    
    log.info(f"Position offsets: disable_all={DISABLE_ALL_OFFSETS}, disable_vert={DISABLE_VERTICAL_OFFSET}, disable_horiz={DISABLE_HORIZONTAL_OFFSET}")
    log.info(f"Position flips: flip_vert={FLIP_VERTICAL}, flip_horiz={FLIP_HORIZONTAL}")
    
    clear_scene()
    
    # Set scene frame range with offset
    bpy.context.scene.render.fps = int(fps)
    bpy.context.scene.frame_start = frame_offset
    bpy.context.scene.frame_end = frame_offset + len(frames) - 1
    
    # Set render resolution to match video (important for camera projection!)
    first_frame = frames[0]
    image_size = first_frame.get("image_size")
    if image_size and len(image_size) >= 2:
        bpy.context.scene.render.resolution_x = int(image_size[0])
        bpy.context.scene.render.resolution_y = int(image_size[1])
        bpy.context.scene.render.resolution_percentage = 100
        log.info(f"Render resolution set to {image_size[0]}x{image_size[1]}")
    
    # Create root locator if needed (for "root" mode)
    root_locator = None
    body_offset = Vector((0, 0, 0))
    if world_translation_mode == "root":
        if root_camera_compensation_mode and camera_extrinsics:
            # Create root locator with inverse camera extrinsics baked in
            root_locator = create_root_locator_with_camera_compensation(
                frames, camera_extrinsics, fps, up_axis, flip_x, frame_offset,
                smoothing_method=extrinsics_smoothing_method,
                smoothing_strength=extrinsics_smoothing_strength
            )
            log.info("Root locator created with inverse camera extrinsics")
        else:
            # Standard root locator (world translation from pred_cam_t)
            root_locator = create_root_locator(frames, fps, up_axis, flip_x, frame_offset)
        
        # Get body offset from frame 0 for initial setup (will be overwritten by per-frame)
        first_cam_t = frames[0].get("pred_cam_t")
        body_offset = get_body_offset_from_cam_t(first_cam_t, up_axis)
        log.info(f"Initial body offset (frame 0): {body_offset} (disable_vert={DISABLE_VERTICAL_OFFSET}, flip_vert={FLIP_VERTICAL})")
    
    # Create mesh with shape keys
    mesh_obj = None
    if include_mesh:
        mesh_obj = create_animated_mesh(frames, faces, fps, transform_func, world_translation_mode, up_axis, frame_offset)
        # Parent mesh to root locator if in "root" mode
        if world_translation_mode == "root" and root_locator and mesh_obj:
            mesh_obj.parent = root_locator
            # Initial position (will be overwritten by per-frame animation)
            mesh_obj.location = body_offset
    
    # Create skeleton (armature with bones and hierarchy)
    # Pass camera_extrinsics to compensate body orientation for camera motion
    # BUT NOT when in bake_to_geometry mode or camera_compensation mode (transforms already applied)
    armature_obj = None
    if include_skeleton:
        skeleton_camera_rots = None
        if not bake_to_geometry_mode and not root_camera_compensation_mode:
            skeleton_camera_rots = camera_extrinsics
        armature_obj = create_skeleton(frames, fps, transform_func, world_translation_mode, up_axis, root_locator, skeleton_mode, joint_parents, frame_offset, skeleton_camera_rots)
        
        # Apply body offset to skeleton as well (initial position)
        if world_translation_mode == "root" and root_locator and armature_obj:
            armature_obj.location = body_offset
    else:
        log.info("Skeleton excluded (camera-only export)")
    
    # Apply PER-FRAME body offset (fixes drift issue)
    # This overwrites the static offset with animated keyframes
    # v4.8.8: Now properly converts screen coords to world using depth * scale_factor
    if world_translation_mode == "root" and root_locator and (mesh_obj or armature_obj):
        apply_per_frame_body_offset(mesh_obj, armature_obj, frames, up_axis, frame_offset,
                                    use_depth=use_depth_positioning, depth_mode=depth_mode,
                                    scale_factor=scale_factor)
    
    # Create separate translation track if in "separate" mode
    if world_translation_mode == "separate":
        create_translation_track(frames, fps, up_axis, frame_offset)
    
    # Create camera
    camera_obj = None
    if include_camera:
        # Determine effective sensor width (MoGe2 intrinsics take precedence)
        effective_sensor_width = sensor_width
        effective_focal_px = None
        effective_cx = None
        effective_cy = None
        if camera_intrinsics:
            if camera_intrinsics.get("focal_length_px"):
                effective_focal_px = camera_intrinsics["focal_length_px"]
            if camera_intrinsics.get("sensor_width_mm"):
                effective_sensor_width = camera_intrinsics["sensor_width_mm"]
            if camera_intrinsics.get("principal_point_x") is not None:
                effective_cx = camera_intrinsics["principal_point_x"]
            if camera_intrinsics.get("principal_point_y") is not None:
                effective_cy = camera_intrinsics["principal_point_y"]
            log.info(f"Using MoGe2 intrinsics: focal={effective_focal_px}px, sensor={effective_sensor_width}mm")
            if effective_cx is not None and effective_cy is not None:
                log.debug(f"Principal point from intrinsics: cx={effective_cx:.2f}, cy={effective_cy:.2f}")
        
        if bake_to_geometry_mode or root_camera_compensation_mode:
            # Use dedicated static camera with intrinsics
            camera_obj = create_static_camera_with_intrinsics(
                frames, effective_sensor_width, up_axis, frame_offset,
                focal_length_px=effective_focal_px,
                principal_point_x=effective_cx,
                principal_point_y=effective_cy
            )
            log.info("Static camera created with intrinsics")
        else:
            camera_obj = create_camera(
                frames, fps, transform_func, up_axis, effective_sensor_width, 
                world_translation_mode, animate_camera, frame_offset, 
                camera_follow_root, camera_use_rotation, camera_static, 
                0,  # camera_smoothing - now handled in extrinsics smoothing
                flip_x, camera_extrinsics
            )
            
            # Parent camera to root locator if requested
            # This makes camera follow character movement while preserving screen-space relationship
            if camera_follow_root and root_locator and camera_obj:
                camera_obj.parent = root_locator
                log.info(f"Camera parented to root_locator - follows character movement")
                if camera_static:
                    log.info(f"Camera is STATIC - per-frame body_offset positions character")
                elif camera_use_rotation:
                    log.info(f"Camera uses PAN/TILT rotation to frame character")
                    # Note: Per-frame body offset is already applied above for all root modes
                else:
                    log.info(f"Camera uses local TRANSLATION to frame character")
    
    # Create metadata locator (for Maya Extra Attributes)
    metadata = data.get("metadata", {})
    if metadata:
        create_metadata_locator(metadata)
    
    # ========== NEW LOCATOR SYSTEM (v4.6.7) ==========
    # 1. ScreenPosition - where character appears on screen (from pred_cam_t)
    create_screen_position_locator(frames, fps, frame_offset, up_axis)
    
    # 2. WorldPosition - true trajectory with camera effects removed (compensated)
    trajectory_compensated = data.get("body_world_trajectory_compensated", [])
    if trajectory_compensated:
        create_world_position_locator(trajectory_compensated, fps, frame_offset)
    else:
        # Fallback: use raw trajectory if compensated not available
        trajectory_raw = data.get("body_world_trajectory", [])
        if trajectory_raw:
            log.info("WARNING: Using raw trajectory (compensated not available)")
            create_world_position_locator(trajectory_raw, fps, frame_offset)
    
    # 3. CameraExtrinsics - camera rotation from CameraSolver
    if camera_extrinsics and len(camera_extrinsics) > 0:
        create_camera_extrinsics_locator(camera_extrinsics, fps, frame_offset, up_axis)
    
    # Legacy: Create old trajectory locator for backward compatibility
    trajectory = data.get("body_world_trajectory", [])
    if trajectory:
        create_trajectory_locator(trajectory, fps, frame_offset)
    # ========== END NEW LOCATOR SYSTEM ==========
    
    # Export
    if output_format == "abc":
        if mesh_obj:
            log.info("Baking shape keys to mesh cache for Alembic...")
            bpy.context.view_layer.objects.active = mesh_obj
            mesh_obj.select_set(True)
        export_alembic(output_path)
        
        # Also export FBX for joints/camera
        fbx_path = output_path.replace(".abc", "_skeleton.fbx")
        if mesh_obj:
            mesh_obj.hide_set(True)
        export_fbx(fbx_path, axis_forward, axis_up_export)
        log.info(f"Also exported skeleton/camera to: {fbx_path}")
    else:
        export_fbx(output_path, axis_forward, axis_up_export)
    
    log.info("Done!")


if __name__ == "__main__":
    main()
