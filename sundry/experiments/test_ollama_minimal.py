#!/usr/bin/env python3
"""
Minimal test script to diagnose Ollama hanging issue.
Tests basic image description functionality with hardcoded prompt.
"""

import ollama
import time
import sys
from typing import Dict, Any

def test_ollama_basic():
    """Test basic Ollama connectivity and response"""
    print("=" * 60)
    print("TEST 1: Basic Ollama connectivity test")
    print("-" * 60)

    try:
        # List available models
        models = ollama.list()
        print(f"Available models: {[m['name'] for m in models['models']]}")

        # Test with a simple text prompt
        print("\nTesting simple text generation...")
        response = ollama.generate(
            model='llama3.2:latest',  # or whatever model you have
            prompt='Say "Hello World" and nothing else.',
            options={
                'temperature': 0.1,
                'max_tokens': 10
            }
        )
        print(f"Response: {response['response']}")
        print("✓ Basic connectivity working")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_ollama_with_timeout():
    """Test Ollama with timeout settings"""
    print("\n" + "=" * 60)
    print("TEST 2: Ollama with timeout")
    print("-" * 60)

    try:
        client = ollama.Client(timeout=5.0)  # 5 second timeout

        print("Testing with 5 second timeout...")
        response = client.generate(
            model='llama3.2:latest',
            prompt='Count to 3',
            options={
                'temperature': 0.1,
                'max_tokens': 20
            }
        )
        print(f"Response: {response['response']}")
        print("✓ Timeout test passed")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_image_description_mock():
    """Test the image description pattern without actual image"""
    print("\n" + "=" * 60)
    print("TEST 3: Image description pattern (mock)")
    print("-" * 60)

    # This is the pattern the app uses
    prompt = """You are an AI assistant that describes images in detail.

Describe this image: [Imagine a photo of a sunset over the ocean with palm trees]

Provide a detailed description including:
- Main subjects and objects
- Colors and lighting
- Setting and environment
- Notable details
- Mood or atmosphere

Keep your response under 200 words."""

    try:
        print("Sending image description prompt...")
        print(f"Prompt length: {len(prompt)} characters")

        start_time = time.time()

        # Try with streaming first to see progress
        stream = ollama.generate(
            model='llama3.2:latest',
            prompt=prompt,
            stream=True,
            options={
                'temperature': 0.7,
                'max_tokens': 200,
                'stop': ['</description>', '\n\n\n']
            }
        )

        print("Streaming response:")
        response_text = ""
        for chunk in stream:
            if 'response' in chunk:
                response_text += chunk['response']
                print(chunk['response'], end='', flush=True)

        elapsed = time.time() - start_time
        print(f"\n\nElapsed time: {elapsed:.2f} seconds")
        print("✓ Image description pattern working")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_categorization_pattern():
    """Test the categorization pattern the app uses"""
    print("\n" + "=" * 60)
    print("TEST 4: Categorization pattern")
    print("-" * 60)

    # Mock description as if from BLIP
    mock_description = "A beautiful sunset over the ocean with golden and orange hues painting the sky. Palm trees silhouette against the vibrant backdrop."

    prompt = f"""Given this image description, suggest 3 relevant categories:

Description: {mock_description}

Return ONLY a JSON object with this structure:
{{
  "suggested_categories": ["category1", "category2", "category3"],
  "primary_category": "main_category"
}}"""

    try:
        print("Testing categorization prompt...")
        print(f"Prompt length: {len(prompt)} characters")

        start_time = time.time()

        response = ollama.generate(
            model='llama3.2:latest',
            prompt=prompt,
            options={
                'temperature': 0.3,
                'max_tokens': 100,
                'format': 'json'  # Force JSON output
            }
        )

        elapsed = time.time() - start_time
        print(f"Response: {response['response']}")
        print(f"Elapsed time: {elapsed:.2f} seconds")

        # Try to parse as JSON
        import json
        try:
            result = json.loads(response['response'])
            print(f"Parsed categories: {result}")
            print("✓ Categorization pattern working")
            return True
        except json.JSONDecodeError as e:
            print(f"⚠ Response is not valid JSON: {e}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("OLLAMA MINIMAL TEST SUITE")
    print("=" * 60)
    print("Testing Ollama connectivity and patterns...")
    print(f"Using ollama library version: {ollama.__version__ if hasattr(ollama, '__version__') else 'unknown'}")

    # Check if Ollama is running
    try:
        models = ollama.list()
        if not models['models']:
            print("⚠ WARNING: No models found. Please pull a model first:")
            print("  ollama pull llama3.2:latest")
            return
    except Exception as e:
        print(f"✗ Cannot connect to Ollama. Is it running?")
        print(f"  Error: {e}")
        print("\nStart Ollama with: ollama serve")
        return

    # Run tests
    results = []
    results.append(("Basic connectivity", test_ollama_basic()))
    results.append(("Timeout handling", test_ollama_with_timeout()))
    results.append(("Image description", test_image_description_mock()))
    results.append(("Categorization", test_categorization_pattern()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("-" * 60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    main()