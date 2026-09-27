#!/usr/bin/env python3
"""
Option 3: Separate BLIP description from Ollama categorization
- Step 1: BLIP generates text description of image (local, no Ollama)
- Step 2: Ollama categorizes based on text only (no images sent)

This is how the app SHOULD work to avoid the hanging issue.
"""

import ollama
import yaml
import json
import time
from PIL import Image
from pathlib import Path

# Mock BLIP since we're just testing the concept
def mock_blip_describe_image(image_path):
    """
    Simulates what BLIP would do - generate a text description of an image.
    In the real app, this uses transformers library with Salesforce/blip model.
    """
    print(f"Step 1: BLIP describing image (local, no network)...")

    # In reality, BLIP would analyze the image and return something like:
    # "a blue square on a white background"
    # "a sunset over the ocean with palm trees in the foreground"
    # "a person wearing a red jacket standing in front of a mountain"

    # For this test, we'll simulate BLIP's output
    mock_descriptions = {
        "test_blue.jpg": "a solid blue square image with uniform color throughout",
        "test_sunset.jpg": "a beautiful sunset over the ocean with orange and pink hues, palm trees silhouetted in the foreground",
        "test_person.jpg": "a person wearing a red jacket standing in front of snow-capped mountains"
    }

    filename = Path(image_path).name
    description = mock_descriptions.get(filename, f"an image file named {filename}")

    print(f"   BLIP output: '{description}'")
    return description

def categorize_with_ollama_text_only(description, filename):
    """
    Use Ollama to categorize based on text description only.
    NO IMAGES are sent to Ollama - just text!
    """
    print(f"\nStep 2: Ollama categorizing from text description...")
    print(f"   (No images sent - just text!)")

    # Build a text-only prompt
    prompt = f"""You are categorizing images based on their descriptions.

Image: {filename}
Description: {description}

Based on this description, suggest 3-5 relevant categories and a primary category.

Return ONLY this YAML format (no markdown, no explanations):

description: |
  {description}
suggested_categories:
  - Category1
  - Category2
  - Category3
primary_category: MainCategory
"""

    try:
        # Use ollama.generate (which we know works) instead of chat
        start_time = time.time()

        response = ollama.generate(
            model='llama3.2:latest',  # Regular model, not vision
            prompt=prompt,
            options={
                'temperature': 0.7,
                'num_predict': 200
            }
        )

        elapsed = time.time() - start_time
        print(f"   ✓ Ollama responded in {elapsed:.2f} seconds")

        # Parse the response
        response_text = response['response'].strip()

        # Clean up response if it has markdown
        if '```yaml' in response_text:
            response_text = response_text.split('```yaml')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        try:
            result = yaml.safe_load(response_text)
            return result
        except yaml.YAMLError:
            # Fallback if YAML parsing fails
            return {
                'description': description,
                'suggested_categories': ['Uncategorized'],
                'primary_category': 'Uncategorized'
            }

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return None

def test_full_pipeline():
    """Test the complete separated pipeline"""

    print("Option 3: Separated BLIP + Ollama Pipeline")
    print("=" * 60)
    print("How it works:")
    print("1. BLIP describes image locally (no network, no Ollama)")
    print("2. Ollama categorizes from text only (no images sent)")
    print("\nThis avoids the hanging issue completely!\n")

    # Create test images
    test_images = []

    # Create a blue square
    img = Image.new('RGB', (100, 100), color='blue')
    img.save('test_blue.jpg')
    test_images.append('test_blue.jpg')

    # Process each image
    results = []
    for image_path in test_images:
        print(f"\nProcessing: {image_path}")
        print("-" * 40)

        # Step 1: BLIP describes the image (local)
        description = mock_blip_describe_image(image_path)

        # Step 2: Ollama categorizes from text (no image)
        categories = categorize_with_ollama_text_only(description, image_path)

        if categories:
            print(f"\n   Result:")
            print(f"   - Description: {categories.get('description', 'N/A')[:50]}...")
            print(f"   - Categories: {categories.get('suggested_categories', [])}")
            print(f"   - Primary: {categories.get('primary_category', 'N/A')}")
            results.append(categories)
        else:
            print("   Failed to categorize")

    # Clean up
    for img_path in test_images:
        Path(img_path).unlink(missing_ok=True)

    return len(results) == len(test_images)

def test_batch_categorization():
    """Test categorizing multiple images at once"""

    print("\n" + "=" * 60)
    print("Batch Categorization Test")
    print("-" * 60)

    # Simulate multiple BLIP descriptions
    image_descriptions = [
        {"filename": "sunset1.jpg", "description": "sunset over ocean with orange sky"},
        {"filename": "cat1.jpg", "description": "gray tabby cat sitting on a windowsill"},
        {"filename": "food1.jpg", "description": "plate of pasta with tomato sauce"},
    ]

    # Build a single prompt for all images
    descriptions_text = "\n\n".join([
        f"Filename: {item['filename']}\nDescription: {item['description']}"
        for item in image_descriptions
    ])

    prompt = f"""Categorize these images based on their descriptions.

{descriptions_text}

Create 3-5 folder categories and assign each image.

Return ONLY this YAML format:

categories:
  - name: Category1
    images: [filename1.jpg, filename2.jpg]
  - name: Category2
    images: [filename3.jpg]
"""

    print("Sending batch categorization request...")

    try:
        response = ollama.generate(
            model='llama3.2:latest',
            prompt=prompt,
            options={'temperature': 0.7, 'num_predict': 300}
        )

        print("✓ Batch categorization successful!")
        print(f"Response preview: {response['response'][:200]}...")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Option 3: Separated Pipeline")
    print("This approach COMPLETELY AVOIDS sending images to Ollama")
    print()

    # Test the separated pipeline
    success1 = test_full_pipeline()

    # Test batch categorization
    success2 = test_batch_categorization()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Single image pipeline: {'✓ Working' if success1 else '✗ Failed'}")
    print(f"  Batch categorization: {'✓ Working' if success2 else '✗ Failed'}")

    if success1 and success2:
        print("\n✓ Option 3 works perfectly!")
        print("\nRecommendation:")
        print("1. Keep using BLIP locally for image descriptions")
        print("2. Send only text descriptions to Ollama (no images)")
        print("3. This avoids the vision model hanging issue entirely")