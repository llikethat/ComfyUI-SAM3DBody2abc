"""SAM3DBody2abc library modules."""

from .logger import (
    log, 
    set_verbosity, 
    set_verbosity_from_string,
    set_module, 
    LogLevel,
    LOG_LEVEL_CHOICES,
    LOG_LEVEL_MAP
)

# Try to import bpy exporter (may not be available if bpy not installed)
try:
    from .bpy_exporter import export_animated_fbx, is_bpy_available
    _BPY_EXPORTER_AVAILABLE = True
except ImportError:
    _BPY_EXPORTER_AVAILABLE = False
    export_animated_fbx = None
    is_bpy_available = lambda: False

__all__ = [
    'log', 
    'set_verbosity', 
    'set_verbosity_from_string',
    'set_module', 
    'LogLevel',
    'LOG_LEVEL_CHOICES',
    'LOG_LEVEL_MAP',
    'export_animated_fbx',
    'is_bpy_available',
]
