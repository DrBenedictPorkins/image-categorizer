#!/usr/bin/env python3
"""
Test the fixed Ollama provider that uses REST API instead of Python client.
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the provider
from providers.ollama_provider import OllamaProvider
from models.image_data import ProviderConfig

def create_test_images():
    """Create some test images"""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)

    # Create a few test images
    images = []

    # Blue square
    img = Image.new('RGB', (200, 200), color='blue')
    img_path = test_dir / "blue_square.jpg"
    img.save(img_path)
    images.append(str(img_path))

    # Red circle (simulated with red square)
    img = Image.new('RGB', (200, 200), color='red')
    img_path = test_dir / "red_square.jpg"
    img.save(img_path)
    images.append(str(img_path))

    # Green rectangle
    img = Image.new('RGB', (300, 150), color='green')
    img_path = test_dir / "green_rect.jpg"
    img.save(img_path)
    images.append(str(img_path))

    return images

def test_provider():
    """Test the fixed Ollama provider"""

    print("Testing Fixed Ollama Provider (REST API)")
    print("=" * 60)

    # Create test images
    print("1. Creating test images...")
    image_paths = create_test_images()
    print(f"   Created {len(image_paths)} test images")

    # Configure provider
    config = ProviderConfig(
        provider_name='ollama',
        settings={
            'host': os.environ.get('OLLAMA_HOST', 'http://localhost:11434'),
            'model': os.environ.get('OLLAMA_MODEL', 'llama3.2-vision:latest'),
            'timeout': 30,  # 30 second timeout for testing
            'max_retries': 1,  # Just 1 retry for testing
            'retry_delay': 1.0
        }
    )

    print("\n2. Initializing Ollama provider...")
    print(f"   Host: {config.get('host')}")
    print(f"   Model: {config.get('model')}")

    provider = OllamaProvider(config)

    # Validate config
    provider.validate_config()

    # Initialize
    if not provider.initialize():
        print("✗ Failed to initialize provider")
        return False

    print("   ✓ Provider initialized")

    # Process images
    print("\n3. Processing images with Ollama vision model...")

    def progress_callback(message, progress):
        print(f"   [{int(progress * 100):3d}%] {message}")

    try:
        result = provider.process_images(
            image_paths=image_paths,
            progress_callback=progress_callback,
            initial_categories=["Shapes", "Colors", "Geometry"]
        )

        print("\n4. Results:")
        print("-" * 40)

        for img_data in result.images:
            print(f"\nImage: {img_data.filename}")
            print(f"  Description: {img_data.description[:100]}...")
            print(f"  Categories: {img_data.suggested_categories}")
            print(f"  Primary: {img_data.primary_category}")

        print("\n" + "=" * 60)
        print("✓ SUCCESS! The fixed provider works!")
        print("\nKey changes that fixed the issue:")
        print("  - Uses REST API directly instead of ollama Python client")
        print("  - Avoids the hanging bug in ollama.Client().chat()")
        print("  - Maintains all the same functionality")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up test images
        print("\n5. Cleaning up test images...")
        test_dir = Path("test_images")
        if test_dir.exists():
            for img_path in test_dir.glob("*.jpg"):
                img_path.unlink()
            test_dir.rmdir()

if __name__ == "__main__":
    success = test_provider()
    sys.exit(0 if success else 1)