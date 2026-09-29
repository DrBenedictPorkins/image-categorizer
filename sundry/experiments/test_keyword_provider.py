"""
Test script for KeywordCategorizationProvider.

This demonstrates how to use the keyword provider for fast, offline categorization
of images that already have descriptions.
"""

from providers.keyword_provider import KeywordCategorizationProvider
from models.image_data import ImageData, ProviderConfig


def test_keyword_provider():
    """Test the keyword categorization provider."""

    # Create provider with minimal config (no settings needed for keyword provider)
    config = ProviderConfig(provider_name="keyword")
    provider = KeywordCategorizationProvider(config)

    # Initialize provider
    if not provider.initialize():
        print("Failed to initialize keyword provider")
        return

    print(f"Provider: {provider.provider_name}")
    print(f"Supports vision: {provider.supports_vision}")
    print(f"Supports description: {provider.supports_description}")
    print(f"Supports categorization: {provider.supports_categorization}")
    print()

    # Test connection (always succeeds for keyword provider)
    print(f"Connection test: {provider.test_connection()}")
    print()

    # Show capabilities
    capabilities = provider.get_capabilities()
    print("Capabilities:")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    print()

    # Create sample images with descriptions
    test_images = [
        ImageData(
            filename="family_portrait.jpg",
            filepath="/path/to/family_portrait.jpg",
            description="A happy family of four people standing together outdoors. Parents with two children, smiling at the camera.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="mountain_view.jpg",
            filepath="/path/to/mountain_view.jpg",
            description="Scenic mountain landscape with snow-capped peaks. Clear blue sky, pine forest in foreground, dramatic natural scenery.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="pasta_dish.jpg",
            filepath="/path/to/pasta_dish.jpg",
            description="Delicious Italian pasta dish with tomato sauce and herbs. Restaurant-style plating on white plate, food photography.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="golden_retriever.jpg",
            filepath="/path/to/golden_retriever.jpg",
            description="Friendly golden retriever dog playing in a park. Happy pet running through grass, outdoor animal photography.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="skyscraper.jpg",
            filepath="/path/to/skyscraper.jpg",
            description="Modern glass and steel skyscraper building reaching into the sky. Urban architecture, downtown cityscape.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="sports_car.jpg",
            filepath="/path/to/sports_car.jpg",
            description="Sleek red sports car parked in front of modern building. Luxury vehicle, automotive photography.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="laptop_setup.jpg",
            filepath="/path/to/laptop_setup.jpg",
            description="Modern workspace with laptop computer, external monitor, and phone on desk. Technology setup, home office.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="abstract_painting.jpg",
            filepath="/path/to/abstract_painting.jpg",
            description="Colorful abstract painting with geometric shapes and bold colors. Modern art displayed in gallery.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="unknown_object.jpg",
            filepath="/path/to/unknown_object.jpg",
            description="Blurry image of an undefined object on a plain background.",
            suggested_categories=[],
            primary_category="",
            metadata={}
        )
    ]

    # Categorize the images
    print("Categorizing images with keyword matching...")
    print("-" * 60)

    result = provider.categorize_described_images(test_images)

    # Display results
    print(f"\nProcessing Stats:")
    for key, value in result.processing_stats.items():
        print(f"  {key}: {value}")
    print()

    print("Image Categorization Results:")
    print("-" * 60)
    for img in result.images:
        print(f"\nFilename: {img.filename}")
        print(f"Description: {img.description[:80]}...")
        print(f"Primary Category: {img.primary_category}")
        print(f"Suggested Categories: {img.suggested_categories}")
        if img.metadata:
            print(f"Metadata: {img.metadata}")

    # Show category distribution
    print("\n" + "=" * 60)
    print("Category Distribution:")
    print("-" * 60)
    category_counts = {}
    for img in result.images:
        category = img.primary_category
        category_counts[category] = category_counts.get(category, 0) + 1

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} image(s)")

    # Cleanup
    provider.cleanup()
    print("\n" + "=" * 60)
    print("Test completed successfully!")


def test_with_suggested_categories():
    """Test keyword provider with images that already have suggested categories."""

    print("\n" + "=" * 60)
    print("Testing with pre-existing suggested categories")
    print("=" * 60)

    config = ProviderConfig(provider_name="keyword")
    provider = KeywordCategorizationProvider(config)
    provider.initialize()

    # Create images with suggested categories (e.g., from BLIP or vision model)
    test_images = [
        ImageData(
            filename="wedding_photo.jpg",
            filepath="/path/to/wedding_photo.jpg",
            description="Bride and groom at wedding ceremony with guests in background.",
            suggested_categories=["Wedding", "People", "Celebration"],
            primary_category="",
            metadata={}
        ),
        ImageData(
            filename="beach_sunset.jpg",
            filepath="/path/to/beach_sunset.jpg",
            description="Beautiful ocean sunset with waves and palm trees silhouetted against orange sky.",
            suggested_categories=["Sunset", "Beach", "Nature"],
            primary_category="",
            metadata={}
        )
    ]

    result = provider.categorize_described_images(test_images)

    print("\nResults (using suggested categories when available):")
    print("-" * 60)
    for img in result.images:
        print(f"\nFilename: {img.filename}")
        print(f"Suggested Categories: {img.suggested_categories}")
        print(f"Final Primary Category: {img.primary_category}")


if __name__ == "__main__":
    test_keyword_provider()
    test_with_suggested_categories()
