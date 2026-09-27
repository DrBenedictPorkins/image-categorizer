#!/usr/bin/env python3
"""
Test Ollama chat() method with images - the actual method the app uses.
This mimics the exact pattern from providers/ollama_provider.py
"""

import ollama
import base64
import time
import textwrap
from io import BytesIO
from PIL import Image

def create_test_image():
    """Create a simple test image"""
    # Create a simple colored square image
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_ollama_chat_with_image():
    """Test the exact pattern used in the application"""

    print("Testing Ollama chat() with image (exact app pattern)")
    print("=" * 60)

    # Configuration (matching the app)
    model = 'llama3.2-vision:latest'  # The vision model
    host = 'http://localhost:11434'
    timeout = 10  # Short timeout for testing

    print(f"Config: model={model}, host={host}, timeout={timeout}")

    # Create client
    client = ollama.Client(host=host)

    # Create a test image
    print("\n1. Creating test image...")
    image_b64 = create_test_image()
    print(f"   Image created: {len(image_b64)} bytes (base64)")

    # Test the exact prompt pattern from the app
    description_prompt = textwrap.dedent("""
        Create a factual description for image categorization and suggest 2-5 initial categories.

        CRITICAL: Return ONLY the YAML format below. NO markdown, NO code blocks, NO ```yaml, NO explanations.
        Just the raw YAML:

        description: |
          Your detailed factual description of what you see
        initial_categories:
          - Category 1
          - Category 2
          - Category 3

        Include: content type, visible elements, environment/setting, color information, readable text.
        Base categories only on what's visible. Use the literal block (|) for description to avoid quote issues.
    """).strip()

    print(f"\n2. Prompt length: {len(description_prompt)} characters")

    print("\n3. Calling client.chat() with image...")
    print("   (This is where the app hangs)")

    start_time = time.time()

    try:
        # This is the EXACT call that the app makes (line 261 in ollama_provider.py)
        response = client.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': description_prompt,
                'images': [image_b64]
            }],
            options={'timeout': timeout}
        )

        elapsed = time.time() - start_time
        print(f"\n✓ Response received in {elapsed:.2f} seconds!")

        if response and 'message' in response:
            content = response['message'].get('content', 'No content')
            print(f"\nResponse preview: {content[:200]}...")
        else:
            print(f"\nUnexpected response structure: {response}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Error after {elapsed:.2f} seconds: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Check if it's a timeout or connection issue
        if "timeout" in str(e).lower():
            print("\n   This is a timeout issue!")
            print("   Possible causes:")
            print("   - Model not downloaded (run: ollama pull llama3.2-vision:latest)")
            print("   - Model is loading (first run can be slow)")
            print("   - Server is overloaded")
        elif "connection" in str(e).lower():
            print("\n   This is a connection issue!")
            print("   - Check if Ollama is running: ollama serve")
            print("   - Check the host setting")

def test_simple_chat_without_image():
    """Test chat without image to isolate the issue"""

    print("\n" + "=" * 60)
    print("Testing chat() WITHOUT image (for comparison)")
    print("-" * 60)

    client = ollama.Client()

    try:
        start = time.time()
        response = client.chat(
            model='llama3.2:latest',  # Non-vision model
            messages=[{
                'role': 'user',
                'content': 'Say hello'
            }]
        )
        elapsed = time.time() - start
        print(f"✓ Chat without image worked in {elapsed:.2f}s")
        print(f"  Response: {response['message']['content'][:50]}...")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    # First test without image
    test_simple_chat_without_image()

    # Then test with image (the problematic one)
    test_ollama_chat_with_image()