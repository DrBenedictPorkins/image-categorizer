#!/usr/bin/env python3
"""
Debug script to identify exactly where Ollama hangs.
"""

import ollama
import sys
import time

print("Ollama Debug Test")
print("=" * 60)

# Step 1: Import test
print("1. Import test: ✓ (ollama module imported)")

# Step 2: Client creation
print("2. Creating client...", end=" ", flush=True)
try:
    client = ollama.Client()
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Step 3: List models (where it seems to hang)
print("3. Listing models...", end=" ", flush=True)
try:
    start = time.time()
    models = client.list()
    elapsed = time.time() - start
    print(f"✓ ({elapsed:.2f}s)")
    print(f"   Found {len(models.get('models', []))} models")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Step 4: Simple generate test
print("4. Testing simple generation...", end=" ", flush=True)
try:
    start = time.time()
    response = ollama.generate(
        model='llama3.2:latest',
        prompt='Say hi',
        options={'max_tokens': 5}
    )
    elapsed = time.time() - start
    print(f"✓ ({elapsed:.2f}s)")
    print(f"   Response: {response.get('response', 'No response')}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nAll tests completed!")