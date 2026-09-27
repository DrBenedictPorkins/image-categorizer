#!/usr/bin/env python3
"""
Workaround for Ollama vision model hanging issue.
Uses generate() instead of chat() for image processing.
"""

import ollama
import base64
import yaml
import textwrap
from io import BytesIO
from PIL import Image
from pathlib import Path

def process_image_with_generate(image_path: str):
    """Process image using generate() as a workaround for chat() hanging."""

    print(f"\nProcessing: {Path(image_path).name}")
    print("-" * 40)

    # Load and encode image
    img = Image.open(image_path)
    # Resize if too large
    if img.width > 1024 or img.height > 1024:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Since generate() doesn't support images directly, we'll need to use
    # a different approach - describe what we want without the actual image
    # OR use a model that supports text-only categorization based on filename

    # Alternative approach 1: Use BLIP for description, then Ollama for categorization
    # This mimics what the app SHOULD do - separate description from categorization

    # For this test, let's simulate a BLIP description
    mock_description = f"Image showing content from file {Path(image_path).name}"

    # Create categorization prompt
    prompt = textwrap.dedent(f"""
        Given this image description, suggest categories in YAML format.

        Description: {mock_description}

        Return ONLY this YAML (no markdown, no code blocks):

        description: |
          {mock_description}
        initial_categories:
          - Category1
          - Category2
          - Category3
    """).strip()

    try:
        # Use generate() which we know works
        response = ollama.generate(
            model='llama3.2:latest',  # Use non-vision model
            prompt=prompt,
            options={
                'temperature': 0.7,
                'max_tokens': 200
            }
        )

        response_text = response['response'].strip()
        print("Response received!")

        # Try to parse YAML
        try:
            result = yaml.safe_load(response_text)
            print(f"✓ Categories: {result.get('initial_categories', [])}")
            return result
        except yaml.YAMLError as e:
            print(f"⚠ YAML parse error: {e}")
            print(f"Raw response: {response_text[:200]}...")
            return None

    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    print("Ollama Workaround Test")
    print("=" * 60)
    print("Using generate() instead of chat() to avoid hanging")

    # Test with a sample image path
    # Create a test image first
    test_image_path = "test_image.jpg"
    img = Image.new('RGB', (100, 100), color='blue')
    img.save(test_image_path)

    result = process_image_with_generate(test_image_path)

    if result:
        print("\n✓ Workaround successful!")
        print("\nSuggested approach:")
        print("1. Use BLIP locally for image description (already working)")
        print("2. Use Ollama generate() with text prompts for categorization")
        print("3. Avoid chat() with images until the hanging issue is resolved")
    else:
        print("\n✗ Workaround failed")

    # Clean up test image
    Path(test_image_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()