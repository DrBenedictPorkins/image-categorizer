#!/usr/bin/env python3
"""
Minimal test specifically for Ollama vision model.
"""

import ollama
import base64
import signal
import sys
from io import BytesIO
from PIL import Image

def timeout_handler(signum, frame):
    print("\n✗ Test timed out after 5 seconds!")
    print("The chat() call with vision model is hanging.")
    sys.exit(1)

# Set up timeout handler
signal.signal(signal.SIGALRM, timeout_handler)

print("Testing Ollama Vision Model")
print("=" * 60)

# Create a simple red square image
print("1. Creating test image...")
img = Image.new('RGB', (100, 100), color='red')
buffer = BytesIO()
img.save(buffer, format='JPEG')
image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
print(f"   Created {len(image_b64)} byte base64 image")

# Create client
print("\n2. Creating Ollama client...")
client = ollama.Client(host='http://localhost:11434')

# Test with vision model
print("\n3. Testing chat() with llama3.2-vision:latest...")
print("   Setting 5 second timeout...")

# Set alarm for 5 seconds
signal.alarm(5)

try:
    response = client.chat(
        model='llama3.2-vision:latest',
        messages=[{
            'role': 'user',
            'content': 'What color is this image?',
            'images': [image_b64]
        }]
    )

    # Cancel alarm if we got here
    signal.alarm(0)

    print("✓ SUCCESS! Got response:")
    print(f"  {response['message']['content']}")

except Exception as e:
    signal.alarm(0)
    print(f"✗ Error: {e}")
    print(f"  Type: {type(e).__name__}")