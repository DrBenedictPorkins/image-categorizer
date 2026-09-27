#!/usr/bin/env python3
"""
Test different Ollama vision models to find which ones work.
"""

import requests
import base64
import json
import time
from io import BytesIO
from PIL import Image

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_model(model_name, image_b64):
    """Test a specific model with an image"""

    print(f"\nTesting: {model_name}")
    print("-" * 40)

    url = "http://localhost:11434/api/chat"

    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": "What color is this image? Answer in one word.",
            "images": [image_b64]
        }],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 10
        }
    }

    try:
        start = time.time()
        response = requests.post(url, json=payload, timeout=10)
        elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            content = result.get('message', {}).get('content', 'No response')
            print(f"✓ SUCCESS in {elapsed:.2f}s")
            print(f"  Response: {content[:100]}")
            return True, elapsed
        else:
            print(f"✗ HTTP {response.status_code}: {response.text[:100]}")
            return False, elapsed

    except requests.Timeout:
        print(f"✗ TIMEOUT after 10 seconds")
        return False, 10.0
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False, 0

def test_all_vision_models():
    """Test all potential vision models"""

    print("Vision Model Compatibility Test")
    print("=" * 60)

    # Create test image
    print("Creating test image...")
    image_b64 = create_test_image()
    print(f"Test image created: {len(image_b64)} bytes")

    # List of vision models to test
    vision_models = [
        "llama3.2-vision:latest",    # The one you're trying to use
        "llama3.2-vision:90b",        # Larger version
        "llava:latest",               # LLaVA model
        "minicpm-v:latest",           # MiniCPM Vision
        "moondream:latest",           # Moondream vision model
    ]

    # Also test some non-vision models to compare
    text_models = [
        "llama3.2:latest",            # Text-only model
        "gpt-oss:20b",                # GPT-like model
    ]

    results = {}

    print("\n" + "=" * 60)
    print("Testing Vision Models:")
    print("=" * 60)

    for model in vision_models:
        success, time_taken = test_model(model, image_b64)
        results[model] = (success, time_taken)

    print("\n" + "=" * 60)
    print("Testing Text Models (for comparison):")
    print("=" * 60)

    for model in text_models:
        success, time_taken = test_model(model, image_b64)
        results[model] = (success, time_taken)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    working_models = []
    broken_models = []

    for model, (success, time_taken) in results.items():
        if success:
            working_models.append((model, time_taken))
            print(f"✓ {model:30} - Works ({time_taken:.2f}s)")
        else:
            broken_models.append(model)
            print(f"✗ {model:30} - Failed/Timeout")

    if working_models:
        print(f"\n✓ Found {len(working_models)} working model(s)!")
        fastest = min(working_models, key=lambda x: x[1])
        print(f"  Fastest: {fastest[0]} ({fastest[1]:.2f}s)")
        print(f"\nRecommendation: Use '{fastest[0]}' in your OLLAMA_MODEL environment variable")
    else:
        print("\n✗ No working vision models found!")
        print("  Try pulling a different model: ollama pull llava:latest")

if __name__ == "__main__":
    test_all_vision_models()