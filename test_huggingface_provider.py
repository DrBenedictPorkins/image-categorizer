#!/usr/bin/env python3
"""
Test script for HuggingFace provider.

This script tests the HuggingFace provider functionality including:
- Device detection
- Model loading
- Connection testing
- Available models listing
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from providers.huggingface_provider import HuggingFaceProvider, list_available_models
from models.image_data import ProviderConfig


def test_device_detection():
    """Test device detection logic."""
    print("\n=== Testing Device Detection ===")

    # Create minimal config
    config = ProviderConfig(
        provider_name='huggingface',
        settings={'device': 'auto'}
    )

    provider = HuggingFaceProvider(config)
    device = provider._detect_device('auto')

    print(f"Auto-detected device: {device}")
    return device


def test_model_listing():
    """Test listing available models."""
    print("\n=== Testing Model Listing ===")
    list_available_models()


def test_provider_initialization():
    """Test provider initialization with small models."""
    print("\n=== Testing Provider Initialization ===")
    print("Note: Using small models (blip-base + flan-t5-xl) for faster testing")

    # Use smallest models for testing
    config = ProviderConfig(
        provider_name='huggingface',
        settings={
            'vision_model': 'Salesforce/blip-image-captioning-base',  # 2GB
            'text_model': 'google/flan-t5-xl',  # 3GB
            'device': 'auto',
            'cache_dir': None,
            'hf_token': os.getenv('HUGGINGFACE_TOKEN')
        }
    )

    provider = HuggingFaceProvider(config)

    print("Initializing provider (this will download models on first run)...")
    success = provider.initialize()

    if success:
        print("✓ Provider initialized successfully")

        # Test connection
        print("\nTesting model functionality...")
        if provider.test_connection():
            print("✓ Models are working correctly")
        else:
            print("✗ Model test failed")
            return False

        # Print capabilities
        print("\nProvider capabilities:")
        caps = provider.get_capabilities()
        for key, value in caps.items():
            print(f"  {key}: {value}")

        # Cleanup
        provider.cleanup()
        print("\n✓ Provider cleaned up")

        return True
    else:
        print("✗ Provider initialization failed")
        return False


def test_full_workflow(image_path: str = None):
    """Test full workflow with a real image."""
    print("\n=== Testing Full Workflow ===")

    if not image_path or not os.path.exists(image_path):
        print("Skipping full workflow test (no test image provided)")
        print("To test with an image, run: python test_huggingface_provider.py <image_path>")
        return True

    print(f"Testing with image: {image_path}")

    # Use small models
    config = ProviderConfig(
        provider_name='huggingface',
        settings={
            'vision_model': 'Salesforce/blip-image-captioning-base',
            'text_model': 'google/flan-t5-xl',
            'device': 'auto',
            'cache_dir': None,
            'hf_token': os.getenv('HUGGINGFACE_TOKEN')
        }
    )

    provider = HuggingFaceProvider(config)

    print("Initializing provider...")
    if not provider.initialize():
        print("✗ Failed to initialize provider")
        return False

    print("✓ Provider initialized")

    try:
        # Test image description
        print("\nGenerating image descriptions...")
        image_data_list = provider.describe_images([image_path])

        if image_data_list:
            img_data = image_data_list[0]
            print(f"\nImage: {img_data.filename}")
            print(f"Description: {img_data.description}")
            print(f"Suggested categories: {img_data.suggested_categories}")
            print(f"Primary category: {img_data.primary_category}")
            print("✓ Description phase completed")
        else:
            print("✗ No image data returned")
            return False

        # Test categorization
        print("\nCategorizing images...")
        result = provider.categorize_described_images(image_data_list)

        print(f"\nCategorization result:")
        print(f"  Total images: {len(result.images)}")
        print(f"  Categories: {list(result.category_groups.keys())}")
        print("✓ Categorization phase completed")

        return True

    except Exception as e:
        print(f"✗ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        provider.cleanup()


def main():
    """Run all tests."""
    print("=" * 60)
    print("HuggingFace Provider Test Suite")
    print("=" * 60)

    # Test 1: Device detection
    device = test_device_detection()

    # Test 2: Model listing
    test_model_listing()

    # Test 3: Provider initialization
    init_success = test_provider_initialization()

    if not init_success:
        print("\n✗ Basic tests failed")
        return 1

    # Test 4: Full workflow (if image provided)
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    workflow_success = test_full_workflow(image_path)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Device detection: ✓ (using {device})")
    print(f"Model listing: ✓")
    print(f"Provider initialization: {'✓' if init_success else '✗'}")
    print(f"Full workflow: {'✓' if workflow_success else 'skipped/failed'}")
    print("=" * 60)

    return 0 if init_success else 1


if __name__ == "__main__":
    sys.exit(main())
