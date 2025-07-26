"""
Data models for image categorization system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class ImageData:
    """Represents a single image with its metadata and categorization results."""
    
    filename: str
    filepath: str
    description: str
    suggested_categories: List[str] = field(default_factory=list)  # ALL categories suggested by initial LLM
    primary_category: str = "Uncategorized"  # The single picked/final category
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def secondary_categories(self) -> List[str]:
        """Get secondary categories (all suggested categories except the primary one)."""
        return [cat for cat in self.suggested_categories if cat != self.primary_category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            'filename': self.filename,
            'filepath': self.filepath,
            'description': self.description,
            'suggested_categories': self.suggested_categories,  # All categories suggested by initial LLM
            'primary_category': self.primary_category,  # Single picked category
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageData':
        """Create ImageData from dictionary."""
        # Handle backward compatibility with old format
        suggested_categories = data.get('suggested_categories', data.get('categories', []))
        primary_category = data.get('primary_category', data.get('category', 'Uncategorized'))
        
        # If no suggested categories but we have a primary_category, add it to suggested_categories
        if not suggested_categories and primary_category != 'Uncategorized':
            suggested_categories = [primary_category]
        
        return cls(
            filename=data['filename'],
            filepath=data.get('filepath', ''),
            description=data.get('description', ''),
            suggested_categories=suggested_categories,
            primary_category=primary_category,
            metadata=data.get('metadata', {})
        )


@dataclass
class CategorizationResult:
    """Complete result of image categorization process."""
    
    images: List[ImageData]
    category_groups: Dict[str, List[str]] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Automatically generate category_groups if not provided."""
        if not self.category_groups and self.images:
            self._generate_category_groups()
    
    def _generate_category_groups(self):
        """Generate category_groups from images."""
        groups = {}
        for image in self.images:
            category = image.primary_category
            if category not in groups:
                groups[category] = []
            groups[category].append(image.filename)
        self.category_groups = groups
    
    def get_category_stats(self) -> Dict[str, int]:
        """Get statistics about category distribution."""
        return {category: len(files) for category, files in self.category_groups.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            'images': [image.to_dict() for image in self.images],
            'category_groups': self.category_groups,
            'processing_stats': self.processing_stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CategorizationResult':
        """Create CategorizationResult from dictionary."""
        images = [ImageData.from_dict(img_data) for img_data in data.get('images', [])]
        return cls(
            images=images,
            category_groups=data.get('category_groups', {}),
            processing_stats=data.get('processing_stats', {})
        )
    
    def get_legacy_results_format(self) -> List[tuple]:
        """Convert to legacy format for backward compatibility: [(filename, description, category), ...]"""
        return [(img.filename, img.description, img.primary_category) for img in self.images]


@dataclass 
class ProviderConfig:
    """Configuration for LLM providers."""
    
    provider_name: str
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.settings[key] = value
    
    def validate_required_keys(self, required_keys: List[str]) -> List[str]:
        """Validate that required configuration keys are present."""
        errors = []
        for key in required_keys:
            if key not in self.settings:
                errors.append(f"Missing required configuration key: {key}")
        return errors