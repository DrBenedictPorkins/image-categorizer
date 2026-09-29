"""
Image processing utilities for the categorization system.

This module handles image file discovery, validation, and basic processing
operations that are common across all providers.
"""

import os
from typing import List, Tuple
from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Lets Pillow open HEIC/HEIF, the default photo format on iPhone and iPad
register_heif_opener()

# Define common image file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.heic', '.heif')
# Formats browsers cannot display; the report shows JPEG previews of these instead
BROWSER_UNSUPPORTED_EXTENSIONS = ('.heic', '.heif')
# Hidden folder, next to the photos, that holds those previews
PREVIEW_DIR = ".image-categorizer-previews"
PREVIEW_MAX_SIZE = 1600


class ImageProcessor:
    """Handles image file discovery and basic processing operations."""
    
    @staticmethod
    def is_image_file(filename: str) -> bool:
        """
        Check if a filename represents an image file based on its extension.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if the file has an image extension, False otherwise
        """
        return filename.lower().endswith(IMAGE_EXTENSIONS)
    
    @staticmethod
    def discover_images(directory: str) -> List[str]:
        """
        Discover all image files in a directory.
        
        Args:
            directory: Path to the directory to search
            
        Returns:
            List of absolute paths to image files, sorted alphabetically
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")
        
        image_paths = []
        directory_path = Path(directory)
        
        for filename in sorted(os.listdir(directory)):
            if ImageProcessor.is_image_file(filename):
                full_path = str(directory_path / filename)
                image_paths.append(full_path)
        
        return image_paths
    
    @staticmethod
    def validate_image(filepath: str) -> Tuple[bool, str]:
        """
        Validate that an image file can be opened and processed.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        try:
            path_obj = Path(filepath)
            
            # Check if file exists
            if not path_obj.exists():
                return False, f"File does not exist: {filepath}"
            
            # Check if it's a file
            if not path_obj.is_file():
                return False, f"Path is not a file: {filepath}"
            
            # Try to open with PIL
            with Image.open(filepath) as img:
                # Verify the image by loading it
                img.verify()
            
            # Re-open to get basic info (verify() closes the image)
            with Image.open(filepath) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    return False, f"Invalid image dimensions: {width}x{height}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    @staticmethod
    def get_image_info(filepath: str) -> dict:
        """
        Get basic information about an image file.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        try:
            path_obj = Path(filepath)
            
            with Image.open(filepath) as img:
                return {
                    'filename': path_obj.name,
                    'filepath': str(path_obj.absolute()),
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': path_obj.stat().st_size
                }
        except Exception as e:
            return {
                'filename': Path(filepath).name,
                'filepath': filepath,
                'error': str(e)
            }
    
    @staticmethod
    def batch_validate_images(image_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a batch of image files.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (valid_paths, invalid_paths_with_errors)
        """
        valid_paths = []
        invalid_info = []
        
        for path in image_paths:
            is_valid, error = ImageProcessor.validate_image(path)
            if is_valid:
                valid_paths.append(path)
            else:
                invalid_info.append(f"{path}: {error}")
        
        return valid_paths, invalid_info
    
    @staticmethod
    def prepare_image_for_processing(filepath: str) -> Image.Image:
        """
        Prepare an image for processing by ensuring it's in RGB mode.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            PIL Image object in RGB mode
        """
        img = Image.open(filepath)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    @staticmethod
    def browser_preview(directory: str, filename: str) -> str:
        """
        Return the path, relative to directory, that a browser can display for an image.

        Formats browsers cannot show (HEIC/HEIF) get a JPEG preview in PREVIEW_DIR,
        created once and refreshed when the original is newer. The original file
        is never changed.

        Args:
            directory: Directory containing the image
            filename: Image filename within that directory

        Returns:
            The filename itself, or the relative path of its JPEG preview
        """
        if not filename.lower().endswith(BROWSER_UNSUPPORTED_EXTENSIONS):
            return filename
        source = Path(directory) / filename
        preview = Path(directory) / PREVIEW_DIR / f"{filename}.jpg"
        if not preview.exists() or preview.stat().st_mtime < source.stat().st_mtime:
            preview.parent.mkdir(exist_ok=True)
            with Image.open(source) as img:
                # Browsers honour EXIF orientation for JPEGs, so match that here
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE))
                img.save(preview, "JPEG", quality=85)
        return f"{PREVIEW_DIR}/{filename}.jpg"

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        Get list of supported image file extensions.
        
        Returns:
            List of supported extensions (without dots)
        """
        return [ext.lstrip('.') for ext in IMAGE_EXTENSIONS]