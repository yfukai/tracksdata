"""A common data structure and basic tools for multi-object tracking."""

try:
    from tracksdata.__about__ import __version__
except ImportError:
    # Fallback for development installs without proper build
    __version__ = "unknown"
