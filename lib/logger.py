"""
SAM3DBody2abc Centralized Logger

Provides consistent logging with verbosity levels across all modules.

Usage:
    from lib.logger import log, set_verbosity, LogLevel
    
    set_verbosity(LogLevel.INFO)  # Default
    
    log.info("Processing started")
    log.debug("Detailed info here")
    log.status("Frame 10/100")
    log.warn("Something unusual")
    log.error("Something failed")

Verbosity Levels:
    SILENT  (0) - No output
    ERROR   (1) - Errors only
    WARN    (2) - Errors + warnings
    INFO    (3) - Normal operation messages (DEFAULT)
    STATUS  (4) - Info + progress updates
    DEBUG   (5) - Everything including diagnostics
"""

from enum import IntEnum
from typing import Optional
from datetime import datetime
import os


class LogLevel(IntEnum):
    """Verbosity levels for logging."""
    SILENT = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    STATUS = 4
    DEBUG = 5


# String to LogLevel mapping for node parameter
LOG_LEVEL_MAP = {
    "Silent": LogLevel.SILENT,
    "Errors Only": LogLevel.ERROR,
    "Warnings": LogLevel.WARN,
    "Normal (Info)": LogLevel.INFO,
    "Verbose (Status)": LogLevel.STATUS,
    "Debug (All)": LogLevel.DEBUG,
}

LOG_LEVEL_CHOICES = list(LOG_LEVEL_MAP.keys())


class Logger:
    """Centralized logger with verbosity control and timestamps."""
    
    def __init__(self, prefix: str = "SAM3DBody2abc"):
        self.prefix = prefix
        self.level = LogLevel.INFO
        self._module = None
        self.show_timestamp = True
    
    def set_level(self, level: LogLevel):
        """Set global verbosity level."""
        self.level = level
    
    def set_level_from_string(self, level_str: str):
        """Set level from node parameter string."""
        self.level = LOG_LEVEL_MAP.get(level_str, LogLevel.INFO)
    
    def set_module(self, module: str):
        """Set current module name for log prefix."""
        self._module = module
    
    def _timestamp(self) -> str:
        """Get current timestamp."""
        if self.show_timestamp:
            return datetime.now().strftime("%H:%M:%S.%f")[:-3]
        return ""
    
    def _format(self, msg: str, tag: Optional[str] = None) -> str:
        """Format message with timestamp and prefix."""
        ts = self._timestamp()
        
        if self._module:
            prefix = f"[{self._module}]"
        else:
            prefix = f"[{self.prefix}]"
        
        if ts:
            if tag:
                return f"[{ts}] {prefix} [{tag}] {msg}"
            return f"[{ts}] {prefix} {msg}"
        else:
            if tag:
                return f"{prefix} [{tag}] {msg}"
            return f"{prefix} {msg}"
    
    def error(self, msg: str):
        """Log error message (always shown unless SILENT)."""
        if self.level >= LogLevel.ERROR:
            print(self._format(msg, "ERROR"))
    
    def warn(self, msg: str):
        """Log warning message."""
        if self.level >= LogLevel.WARN:
            print(self._format(msg, "WARN"))
    
    def info(self, msg: str):
        """Log info message - normal operation status."""
        if self.level >= LogLevel.INFO:
            print(self._format(msg))
    
    def status(self, msg: str):
        """Log status/progress message."""
        if self.level >= LogLevel.STATUS:
            print(self._format(msg))
    
    def debug(self, msg: str):
        """Log debug message - detailed diagnostics."""
        if self.level >= LogLevel.DEBUG:
            print(self._format(msg, "DEBUG"))
    
    def progress(self, current: int, total: int, task: str = "", interval: int = 10):
        """
        Log progress at intervals to avoid spam.
        
        Args:
            current: Current item (0-indexed)
            total: Total items
            task: Task description
            interval: Print every N items (default 10)
        """
        if self.level < LogLevel.STATUS:
            return
        
        # Always print first, last, and at intervals
        if current == 0 or current == total - 1 or (current + 1) % interval == 0:
            pct = (current + 1) / total * 100
            if task:
                print(self._format(f"{task}: {current + 1}/{total} ({pct:.0f}%)"))
            else:
                print(self._format(f"Progress: {current + 1}/{total} ({pct:.0f}%)"))
    
    def section(self, title: str):
        """Log section header for major operations."""
        if self.level >= LogLevel.INFO:
            print(self._format(f"===== {title} ====="))
    
    def data(self, name: str, value, show_at: LogLevel = LogLevel.DEBUG):
        """
        Log data value at specified level.
        
        Args:
            name: Variable/data name
            value: Value to display
            show_at: Minimum level to show (default DEBUG)
        """
        if self.level >= show_at:
            if hasattr(value, 'shape'):
                print(self._format(f"{name}: shape={value.shape}"))
            elif isinstance(value, (list, tuple)):
                print(self._format(f"{name}: len={len(value)}"))
            elif isinstance(value, dict):
                print(self._format(f"{name}: keys={list(value.keys())}"))
            else:
                print(self._format(f"{name}: {value}"))
    
    def frame_info(self, frame_idx: int, **kwargs):
        """
        Log frame-specific debug info.
        
        Usage:
            log.frame_info(0, tx=0.5, ty=-0.2, depth=3.5)
        """
        if self.level >= LogLevel.DEBUG:
            parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                     for k, v in kwargs.items()]
            print(self._format(f"Frame {frame_idx}: {', '.join(parts)}"))


# Global logger instance
log = Logger()


def set_verbosity(level: LogLevel):
    """Set global verbosity level."""
    log.set_level(level)


def set_verbosity_from_string(level_str: str):
    """Set global verbosity from node parameter string."""
    log.set_level_from_string(level_str)


def set_module(module: str):
    """Set current module for log prefix."""
    log.set_module(module)


# Environment variable override (fallback if not set via node)
_env_level = os.environ.get("SAM3DBODY_LOG_LEVEL", "").upper()
if _env_level:
    _env_map = {"SILENT": 0, "ERROR": 1, "WARN": 2, "INFO": 3, "STATUS": 4, "DEBUG": 5}
    if _env_level in _env_map:
        log.set_level(LogLevel(_env_map[_env_level]))


# Blender-specific logger (standalone script context)
class BlenderLogger(Logger):
    """Logger for Blender script context."""
    
    def __init__(self):
        super().__init__(prefix="Blender")


# Create Blender logger instance
blender_log = BlenderLogger()
