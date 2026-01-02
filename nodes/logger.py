"""
Centralized logging for SAM3DBody2abc
Provides consistent timestamp (IST) and version info across all nodes.
"""

from datetime import datetime, timezone, timedelta

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Import version from package
try:
    from .. import __version__
except ImportError:
    __version__ = "unknown"


def get_timestamp():
    """Get current timestamp in IST format."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def log(node_name: str, message: str, level: str = "INFO"):
    """
    Log a message with timestamp, version, and node name.
    
    Args:
        node_name: Name of the node (e.g., "Export", "Motion Analyzer", "CameraSolver")
        message: The message to log
        level: Log level (INFO, DEBUG, WARNING, ERROR)
    
    Example output:
        [2026-01-02 10:30:08 IST] [v4.6.4] [Export] Processing 50 frames
    """
    timestamp = get_timestamp()
    
    if level == "DEBUG":
        prefix = f"[{timestamp}] [v{__version__}] [{node_name} DEBUG]"
    elif level == "WARNING":
        prefix = f"[{timestamp}] [v{__version__}] [{node_name} WARNING]"
    elif level == "ERROR":
        prefix = f"[{timestamp}] [v{__version__}] [{node_name} ERROR]"
    else:
        prefix = f"[{timestamp}] [v{__version__}] [{node_name}]"
    
    print(f"{prefix} {message}")


def log_info(node_name: str, message: str):
    """Log info message."""
    log(node_name, message, "INFO")


def log_debug(node_name: str, message: str):
    """Log debug message."""
    log(node_name, message, "DEBUG")


def log_warning(node_name: str, message: str):
    """Log warning message."""
    log(node_name, message, "WARNING")


def log_error(node_name: str, message: str):
    """Log error message."""
    log(node_name, message, "ERROR")


def log_section(node_name: str, title: str):
    """Log a section header."""
    timestamp = get_timestamp()
    print(f"\n[{timestamp}] [v{__version__}] [{node_name}] ========== {title} ==========")


def log_separator(node_name: str):
    """Log a separator line."""
    timestamp = get_timestamp()
    print(f"[{timestamp}] [v{__version__}] [{node_name}] " + "=" * 50)


# Convenience function for quick logging without specifying node
class NodeLogger:
    """
    Logger instance for a specific node.
    
    Usage:
        logger = NodeLogger("Export")
        logger.info("Processing started")
        logger.debug("Frame 0 data: ...")
        logger.warning("Missing data")
    """
    
    def __init__(self, node_name: str):
        self.node_name = node_name
    
    def info(self, message: str):
        log_info(self.node_name, message)
    
    def debug(self, message: str):
        log_debug(self.node_name, message)
    
    def warning(self, message: str):
        log_warning(self.node_name, message)
    
    def error(self, message: str):
        log_error(self.node_name, message)
    
    def section(self, title: str):
        log_section(self.node_name, title)
    
    def separator(self):
        log_separator(self.node_name)


# Export version for external use
def get_version():
    """Get current package version."""
    return __version__
