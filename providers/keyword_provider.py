"""
Keyword-based categorization provider for image categorization.

This provider implements simple keyword matching for categorization only.
It does not support image description (use BLIP or vision models for that).
Useful for fast local categorization, testing, and fallback scenarios.
"""

from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

from .base import BaseLLMProvider, ProviderError, ProviderProcessingError
from models.image_data import ImageData, CategorizationResult, ProviderConfig


class KeywordCategorizationProvider(BaseLLMProvider):
    """Provider for keyword-based categorization without LLM or vision capabilities."""

    # Category keywords mapping
    CATEGORY_KEYWORDS = {
        "People": ["person", "people", "man", "woman", "child", "human", "face", "portrait"],
        "Nature": ["tree", "forest", "mountain", "sky", "flower", "landscape", "outdoor", "nature"],
        "Food": ["food", "meal", "dish", "cooking", "restaurant", "kitchen"],
        "Animals": ["animal", "dog", "cat", "pet", "wildlife", "bird", "fish"],
        "Architecture": ["building", "house", "structure", "bridge", "architecture"],
        "Vehicles": ["car", "vehicle", "truck", "bike", "motorcycle", "transportation"],
        "Technology": ["computer", "phone", "device", "screen", "electronic"],
        "Art": ["art", "painting", "drawing", "sculpture", "artwork"]
    }

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

    @property
    def provider_name(self) -> str:
        return "Keyword Categorization"

    @property
    def required_config_keys(self) -> List[str]:
        return []  # No configuration required

    @property
    def supports_vision(self) -> bool:
        """This provider does not support vision/image input."""
        return False

    @property
    def supports_description(self) -> bool:
        """This provider does not support the description phase."""
        return False

    @property
    def supports_categorization(self) -> bool:
        """This provider only supports categorization of pre-described images."""
        return True

    def initialize(self) -> bool:
        """Initialize the keyword provider (no setup needed)."""
        self._initialized = True
        return True

    def validate_config(self) -> None:
        """Validate configuration (no requirements for keyword provider)."""
        # No validation needed - keyword provider requires no configuration
        pass

    def test_connection(self) -> bool:
        """Test connection (always succeeds for local keyword matching)."""
        return True

    def process_images(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        initial_categories: Optional[List[str]] = None
    ) -> CategorizationResult:
        """
        Process images - NOT SUPPORTED.

        This provider cannot describe images. Use categorize_described_images() instead.

        Raises:
            NotImplementedError: Always raises since this provider doesn't support description
        """
        raise NotImplementedError(
            f"{self.provider_name} provider does not support image description. "
            "Use a vision-capable provider (e.g., Ollama) for image description, "
            "then use categorize_described_images() for keyword-based categorization."
        )

    def categorize_described_images(
        self,
        images: List[ImageData],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> CategorizationResult:
        """
        Categorize images that already have descriptions using keyword matching.

        This method analyzes the description text and assigns categories based on
        keyword frequency matching. It's fast, deterministic, and works offline.

        Args:
            images: List of ImageData with descriptions already populated
            progress_callback: Optional callback for progress updates (message, progress_0_to_1)

        Returns:
            Complete CategorizationResult with categories assigned based on keywords
        """
        if not self._initialized:
            raise ProviderError("Provider not initialized")

        if not images:
            raise ProviderProcessingError("No images provided for categorization")

        self._report_progress(progress_callback, "Starting keyword-based categorization...", 0.0)

        # Validate that all images have descriptions
        for img in images:
            if not img.description or img.description.strip() == "":
                raise ProviderProcessingError(
                    f"Image {img.filename} has no description. "
                    "All images must be described before using keyword categorization."
                )

        # Categorize each image
        total_images = len(images)
        for i, image_data in enumerate(images):
            self._report_progress(
                progress_callback,
                f"Categorizing {i+1}/{total_images}: {image_data.filename}",
                (i / total_images) * 0.9
            )

            # Apply keyword-based categorization
            category = self._categorize_by_keywords(image_data)
            image_data.primary_category = category

            # If suggested_categories is empty, populate it with the matched category
            if not image_data.suggested_categories:
                image_data.suggested_categories = [category]

            # Add provider metadata
            if not image_data.metadata:
                image_data.metadata = {}
            image_data.metadata['categorization_provider'] = 'keyword'
            image_data.metadata['categorization_method'] = 'keyword_matching'

        self._report_progress(progress_callback, "Finalizing results...", 0.95)

        # Create the result
        result = CategorizationResult(
            images=images,
            processing_stats={
                'provider': 'keyword',
                'categorization_method': 'keyword_matching',
                'total_images': len(images),
                'categories_generated': len(set(img.primary_category for img in images)),
                'uncategorized': len([img for img in images if img.primary_category == 'Uncategorized'])
            }
        )

        self._report_progress(progress_callback, "Categorization complete!", 1.0)
        return result

    def _categorize_by_keywords(self, image_data: ImageData) -> str:
        """
        Categorize a single image based on keyword matching in its description.

        Strategy:
        1. First check suggested_categories if available (from description phase)
        2. If no suggested categories or they're generic, use keyword matching
        3. Count keyword matches in description for each category
        4. Return category with most matches, or "Uncategorized" if no matches

        Args:
            image_data: ImageData object with description

        Returns:
            Category name as string
        """
        # Strategy 1: Use existing suggested_categories if meaningful
        if (image_data.suggested_categories
            and image_data.suggested_categories != ['Error']
            and image_data.suggested_categories != ['Uncategorized']):
            # Check if first suggested category is in our keyword categories or is specific
            first_suggestion = image_data.suggested_categories[0]
            if first_suggestion in self.CATEGORY_KEYWORDS or len(image_data.suggested_categories) == 1:
                return first_suggestion

        # Strategy 2: Use keyword matching on description
        description = image_data.description.lower()
        best_category = "Uncategorized"
        max_matches = 0

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            # Count how many keywords from this category appear in the description
            matches = sum(1 for keyword in keywords if keyword in description)
            if matches > max_matches:
                max_matches = matches
                best_category = category

        return best_category

    def get_capabilities(self) -> Dict[str, Any]:
        """Get keyword provider capabilities."""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            'supports_vision': False,
            'supports_description': False,
            'supports_categorization': True,
            'requires_internet': False,
            'requires_api_key': False,
            'processing_speed': 'very_fast',
            'method': 'keyword_matching',
            'deterministic': True,
            'categories': list(self.CATEGORY_KEYWORDS.keys()) + ['Uncategorized']
        })
        return base_capabilities

    def cleanup(self):
        """Clean up resources (none needed for keyword provider)."""
        pass
