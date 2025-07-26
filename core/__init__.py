"""
Core business logic for image categorization.

This package contains the core functionality for image processing,
HTML generation, and configuration management.
"""

from .image_processor import ImageProcessor
from .html_generator import HTMLGenerator

__all__ = ['ImageProcessor', 'HTMLGenerator']