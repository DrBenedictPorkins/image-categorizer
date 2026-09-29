#!/usr/bin/env python3
"""
Test script to verify phase-specific methods in OllamaProvider.
"""

import os
from pathlib import Path
from providers.ollama_provider import OllamaProvider
from models.image_data import ProviderConfig

def test_phase_separation():
    """Test describe_images() and categorize_described_images() separately."""

    # Setup configuration
    config = ProviderConfig(
        host=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
        model=os.getenv('OLLAMA_MODEL', 'llava:latest'),
        timeout=int(os.getenv('OLLAMA_TIMEOUT', '300')),
        max_retries=int(os.getenv('OLLAMA_MAX_RETRIES', '2')),
        retry_delay=float(os.getenv('OLLAMA_RETRY_DELAY', '1.0'))
    )

    # Initialize provider
    provider = OllamaProvider(config)
    if not provider.initialize():
        print("Failed to initialize provider")
        return False

    print("✓ Provider initialized successfully")

    # Test with a simple test image directory
    test_dir = Path(__file__).parent / "test_images"
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        print("Please create test_images directory with some sample images")
        return False

    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {test_dir}")
        return False

    image_paths = [str(f.absolute()) for f in image_files[:3]]  # Test with max 3 images
    print(f"\n✓ Found {len(image_paths)} test images")

    # Test Phase 1: Description
    print("\n=== Testing Phase 1: describe_images() ===")
    try:
        described_images = provider.describe_images(
            image_paths,
            progress_callback=lambda msg, prog: print(f"  [{prog*100:.0f}%] {msg}")
        )
        print(f"✓ Successfully described {len(described_images)} images")

        # Show sample data
        if described_images:
            sample = described_images[0]
            print(f"\nSample description:")
            print(f"  File: {sample.filename}")
            print(f"  Description: {sample.description[:100]}...")
            print(f"  Suggested categories: {sample.suggested_categories}")
            print(f"  Primary category: {sample.primary_category}")

    except Exception as e:
        print(f"✗ Phase 1 failed: {e}")
        return False

    # Test Phase 2: Categorization
    print("\n=== Testing Phase 2: categorize_described_images() ===")
    try:
        result = provider.categorize_described_images(
            described_images,
            progress_callback=lambda msg, prog: print(f"  [{prog*100:.0f}%] {msg}")
        )
        print(f"✓ Successfully categorized {len(result.images)} images")

        # Show final categories
        print(f"\nFinal categories:")
        categories = {}
        for img in result.images:
            cat = img.primary_category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(img.filename)

        for cat, files in categories.items():
            print(f"  {cat}: {len(files)} images")
            for f in files:
                print(f"    - {f}")

        print(f"\nProcessing stats:")
        for key, value in result.processing_stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"✗ Phase 2 failed: {e}")
        return False

    print("\n✓ All phase separation tests passed!")
    return True

if __name__ == "__main__":
    success = test_phase_separation()
    exit(0 if success else 1)
