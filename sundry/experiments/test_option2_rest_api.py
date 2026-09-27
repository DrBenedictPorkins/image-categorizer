#!/usr/bin/env python3
"""
Option 2: Use Ollama REST API directly instead of Python client
This bypasses the ollama-python library that has the hanging issue.
"""

import requests
import base64
import json
import time
from io import BytesIO
from PIL import Image
from pathlib import Path

def test_rest_api_with_image():
    """Test Ollama REST API directly with vision model and image"""

    print("Option 2: Direct REST API Test")
    print("=" * 60)

    # Configuration
    OLLAMA_HOST = "http://localhost:11434"
    MODEL = "llama3.2-vision:latest"

    # Create a simple test image
    print("1. Creating test image...")
    img = Image.new('RGB', (200, 200), color='blue')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print(f"   Created {len(image_b64)} byte base64 image")

    # Prepare the request
    prompt = """Describe this image and suggest 3 categories for it.

    Return your response in this format:
    Description: [your description]
    Categories: [category1, category2, category3]"""

    # Build the API request
    api_endpoint = f"{OLLAMA_HOST}/api/chat"

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        }],
        "stream": False,  # Don't stream for simplicity
        "options": {
            "temperature": 0.7,
            "num_predict": 200
        }
    }

    print(f"\n2. Sending POST request to {api_endpoint}")
    print(f"   Model: {MODEL}")
    print(f"   Payload size: {len(json.dumps(payload))} bytes")

    start_time = time.time()

    try:
        # Make the REST API call with a timeout
        response = requests.post(
            api_endpoint,
            json=payload,
            timeout=30  # 30 second timeout
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"\n✓ SUCCESS! Got response in {elapsed:.2f} seconds")

            result = response.json()
            if 'message' in result:
                content = result['message'].get('content', 'No content')
                print(f"\nResponse:\n{content}")
            else:
                print(f"\nFull response: {json.dumps(result, indent=2)}")

            return True

        else:
            print(f"\n✗ HTTP Error {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.Timeout:
        elapsed = time.time() - start_time
        print(f"\n✗ Request timed out after {elapsed:.2f} seconds")
        print("   The REST API is also hanging with vision models!")
        return False

    except requests.ConnectionError:
        print(f"\n✗ Cannot connect to Ollama at {OLLAMA_HOST}")
        print("   Make sure Ollama is running: ollama serve")
        return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Error after {elapsed:.2f} seconds: {e}")
        return False

def test_rest_api_generate():
    """Test using /api/generate endpoint instead of /api/chat"""

    print("\n" + "=" * 60)
    print("Alternative: Using /api/generate endpoint")
    print("-" * 60)

    OLLAMA_HOST = "http://localhost:11434"
    MODEL = "llama3.2:latest"  # Non-vision model

    # Simple text-only prompt
    prompt = "List 3 categories for a blue square image: "

    api_endpoint = f"{OLLAMA_HOST}/api/generate"

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 50
        }
    }

    print(f"Sending request to {api_endpoint}")

    try:
        response = requests.post(
            api_endpoint,
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"✗ HTTP Error {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Ollama REST API directly (bypassing Python client)")
    print("This tests if the hanging issue is in the Python client or server")
    print()

    # Test the chat endpoint with vision model
    success1 = test_rest_api_with_image()

    # Test the generate endpoint as alternative
    success2 = test_rest_api_generate()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  /api/chat with vision: {'✓ Working' if success1 else '✗ Hanging'}")
    print(f"  /api/generate (text): {'✓ Working' if success2 else '✗ Failed'}")

    if not success1:
        print("\nThe hanging issue is in the Ollama server itself, not just the Python client!")
        print("Use Option 3: Separate BLIP description from Ollama text categorization")