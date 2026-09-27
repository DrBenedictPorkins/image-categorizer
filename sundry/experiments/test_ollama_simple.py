#!/usr/bin/env python3
"""
Simple test to diagnose why Ollama hangs in the main application.
This mimics exactly what the app does when processing images.
"""

import ollama
import time
import os

def test_with_real_image_description():
    """Test with the exact pattern used in the application"""

    print("Testing Ollama with image description pattern...")
    print("-" * 60)

    # Check if Ollama is running
    try:
        # Try to ping Ollama
        client = ollama.Client()
        models = client.list()
        print(f"✓ Connected to Ollama")
        print(f"  Models available: {len(models.get('models', []))}")
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return

    # This mimics what happens in providers/ollama_provider.py
    model = os.environ.get('OLLAMA_MODEL', 'llama3.2:latest')
    timeout = int(os.environ.get('OLLAMA_TIMEOUT', '300'))

    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Timeout: {timeout} seconds")

    # Mock BLIP description (what would come from the image processor)
    blip_description = """A person wearing a blue jacket standing in front of a mountain landscape.
    The scene shows snow-capped peaks in the background with green valleys below.
    The lighting suggests late afternoon with golden hour colors."""

    # Exact prompt pattern from the app
    prompt = f"""You are an AI assistant specialized in categorizing images based on their descriptions.

Given the following image description, suggest relevant categories and provide a primary category.

Image Description:
{blip_description}

Provide your response in the following JSON format:
{{
    "suggested_categories": ["category1", "category2", "category3"],
    "primary_category": "main_category"
}}

Categories should be:
- Descriptive and specific
- Suitable for organizing files into folders
- Between 1-3 words each
- Semantically meaningful

Return ONLY the JSON object, no additional text."""

    print(f"\nPrompt length: {len(prompt)} characters")
    print("\nSending request to Ollama...")
    print("(Press Ctrl+C to abort if it hangs)")

    start_time = time.time()

    try:
        # Method 1: Direct generation (what the app does)
        print("\n1. Testing direct generation...")
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': 0.7,
                'max_tokens': 150,
                'format': 'json'
            }
        )

        elapsed = time.time() - start_time
        print(f"✓ Response received in {elapsed:.2f} seconds")
        print(f"Response: {response['response'][:200]}...")

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user (was hanging)")
        elapsed = time.time() - start_time
        print(f"Hung for {elapsed:.2f} seconds before interruption")
        return
    except Exception as e:
        print(f"✗ Error: {e}")
        return

    # Method 2: Try with streaming to see if that helps
    print("\n2. Testing with streaming (to see progress)...")
    start_time = time.time()

    try:
        stream = ollama.generate(
            model=model,
            prompt=prompt[:100] + "... (truncated for test)",  # Shorter prompt
            stream=True,
            options={
                'temperature': 0.7,
                'max_tokens': 50
            }
        )

        print("Streaming chunks: ", end="", flush=True)
        response_text = ""
        chunk_count = 0
        for chunk in stream:
            if 'response' in chunk:
                response_text += chunk['response']
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(".", end="", flush=True)

        elapsed = time.time() - start_time
        print(f"\n✓ Streaming completed in {elapsed:.2f} seconds")
        print(f"  Received {chunk_count} chunks")

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user (was hanging)")
        elapsed = time.time() - start_time
        print(f"Hung for {elapsed:.2f} seconds before interruption")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("\nIf the test hangs, possible issues:")
    print("1. Model not downloaded (run: ollama pull llama3.2:latest)")
    print("2. Ollama server overloaded or not responding")
    print("3. Network/firewall issues if using remote Ollama")
    print("4. Model is too large for available memory")

if __name__ == "__main__":
    test_with_real_image_description()