#!/usr/bin/env python3
"""
Comprehensive Test Script for Phase Separation Workflows

This script tests all phase separation features of the image categorization system,
including hybrid workflows, description-only phases, categorization-only phases,
and backward compatibility.

USAGE INSTRUCTIONS:
===================

1. Setup Environment:
   - Ensure .env file is configured with provider settings
   - For Ollama: OLLAMA_HOST, OLLAMA_MODEL
   - For HuggingFace: HF_VISION_MODEL, HF_TEXT_MODEL (optional, has defaults)

2. Run All Tests:
   python test_phase_separation_workflows.py

3. Run Specific Test:
   python test_phase_separation_workflows.py --test <test_name>

   Available tests:
   - test_1_fully_local         : HuggingFace description → HuggingFace categorization
   - test_2_hybrid_hf_ollama    : HuggingFace description → Ollama categorization
   - test_3_cloud_baseline      : Ollama description → Ollama categorization
   - test_4_description_only    : Save intermediate description results
   - test_5_categorization_only : Load descriptions and categorize
   - test_6_keyword_categorization : HuggingFace description → Keyword categorization
   - test_7_backward_compat     : Single provider for both phases

4. Examples:
   # Test fully local workflow with HuggingFace
   python test_phase_separation_workflows.py --test test_1_fully_local

   # Test hybrid workflow (HF description + Ollama categorization)
   python test_phase_separation_workflows.py --test test_2_hybrid_hf_ollama

   # Test description-only phase (save intermediate results)
   python test_phase_separation_workflows.py --test test_4_description_only

   # Test categorization-only phase (load and categorize)
   python test_phase_separation_workflows.py --test test_5_categorization_only

TEST IMAGES:
============
The script uses test images from test_images/ in the repository root
Make sure this directory contains some test images (JPG, PNG, etc.)

OUTPUT:
=======
- Results are saved to test_output/<test_name>/
- Each test creates categorization_results.json
- Test 4 (description-only) creates descriptions.json
- Test 5 (categorization-only) loads descriptions.json

"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from dotenv import load_dotenv
from core.config import get_provider_config
from providers import BaseLLMProvider, OllamaProvider, HuggingFaceProvider, KeywordCategorizationProvider
from models.image_data import ImageData, CategorizationResult
from core.image_processor import ImageProcessor


# Test configuration
TEST_IMAGE_DIR = script_dir / "test_images"
TEST_OUTPUT_DIR = script_dir / "test_output"
MAX_TEST_IMAGES = 5  # Limit to 5 images for faster testing


class TestRunner:
    """Manages test execution and result reporting."""

    def __init__(self):
        self.results = {}
        self.current_test = None

    def run_test(self, test_name: str, test_func):
        """Run a single test and capture results."""
        print("\n" + "=" * 80)
        print(f"TEST: {test_name}")
        print("=" * 80)

        self.current_test = test_name

        try:
            result = test_func()
            self.results[test_name] = {
                'status': 'PASSED' if result else 'FAILED',
                'message': 'Test completed successfully' if result else 'Test failed'
            }
            print(f"\n✓ {test_name}: PASSED")
            return True

        except Exception as e:
            self.results[test_name] = {
                'status': 'ERROR',
                'message': str(e)
            }
            print(f"\n✗ {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        errors = sum(1 for r in self.results.values() if r['status'] == 'ERROR')

        for test_name, result in self.results.items():
            status_icon = '✓' if result['status'] == 'PASSED' else '✗'
            print(f"{status_icon} {test_name}: {result['status']}")
            if result['status'] != 'PASSED':
                print(f"  └─ {result['message']}")

        print(f"\nTotal: {len(self.results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")


def get_test_images(limit: int = MAX_TEST_IMAGES) -> List[str]:
    """Get list of test image paths."""
    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Test image directory not found: {TEST_IMAGE_DIR}")

    image_paths = ImageProcessor.discover_images(str(TEST_IMAGE_DIR))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {TEST_IMAGE_DIR}")

    # Limit number of images for faster testing
    image_paths = image_paths[:limit]
    print(f"Using {len(image_paths)} test images from {TEST_IMAGE_DIR}")

    return image_paths


def save_test_output(test_name: str, data: Dict[str, Any], filename: str = "categorization_results.json"):
    """Save test output to JSON file."""
    output_dir = TEST_OUTPUT_DIR / test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / filename
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Results saved to: {output_file}")
    return output_file


def load_test_input(test_name: str, filename: str = "descriptions.json") -> Dict[str, Any]:
    """Load test input from JSON file."""
    input_file = TEST_OUTPUT_DIR / test_name / filename

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded input from: {input_file}")
    return data


def initialize_provider(provider_name: str) -> BaseLLMProvider:
    """Initialize a provider."""
    print(f"\nInitializing {provider_name} provider...")

    provider_classes = {
        'ollama': OllamaProvider,
        'huggingface': HuggingFaceProvider,
        'keyword': KeywordCategorizationProvider,
    }

    if provider_name not in provider_classes:
        raise ValueError(f"Unknown provider: {provider_name}")

    # Get configuration
    config = get_provider_config(provider_name)
    if not config:
        raise ValueError(f"No configuration found for {provider_name}")

    # Create provider instance
    provider_class = provider_classes[provider_name]
    provider = provider_class(config)

    # Validate and initialize
    provider.validate_config()

    if not provider.initialize():
        raise RuntimeError(f"Failed to initialize {provider_name} provider")

    # Test connection
    if not provider.test_connection():
        raise RuntimeError(f"Failed to connect to {provider_name} service")

    print(f"✓ {provider_name} provider ready")
    return provider


def verify_result_structure(result: CategorizationResult, expect_categories: bool = True):
    """Verify that result has expected structure."""
    print("\nVerifying result structure...")

    assert result is not None, "Result is None"
    assert len(result.images) > 0, "No images in result"

    # Check each image has required fields
    for img in result.images:
        assert img.filename, f"Missing filename"
        assert img.filepath, f"Missing filepath for {img.filename}"
        assert img.description, f"Missing description for {img.filename}"

        if expect_categories:
            assert img.primary_category, f"Missing primary_category for {img.filename}"
            assert img.primary_category != "", f"Empty primary_category for {img.filename}"

    # Check category groups
    if expect_categories:
        assert len(result.category_groups) > 0, "No category groups"

    print(f"✓ Result structure valid")
    print(f"  - {len(result.images)} images")
    print(f"  - {len(result.category_groups)} categories")
    print(f"  - Categories: {', '.join(result.category_groups.keys())}")


def print_sample_results(result: CategorizationResult, limit: int = 3):
    """Print sample results for inspection."""
    print(f"\nSample results (showing {min(limit, len(result.images))} of {len(result.images)} images):")

    for img in result.images[:limit]:
        print(f"\n  {img.filename}:")
        print(f"    Description: {img.description[:100]}...")
        print(f"    Primary Category: {img.primary_category}")
        if img.suggested_categories:
            print(f"    Suggested Categories: {', '.join(img.suggested_categories)}")


# ============================================================================
# TEST 1: Fully Local Workflow (HuggingFace → HuggingFace)
# ============================================================================

def test_1_fully_local():
    """
    Test fully local workflow using HuggingFace for both phases.

    Phase 1: HuggingFace vision model describes images
    Phase 2: HuggingFace text model categorizes based on descriptions

    Benefits:
    - Complete privacy (no cloud API calls)
    - No internet required after model download
    - Free to use

    Drawbacks:
    - Requires significant local compute (GPU recommended)
    - Slower than cloud APIs
    - Large model downloads required
    """
    print("\nWorkflow: HuggingFace Vision → HuggingFace Text Model")
    print("Privacy: HIGH (fully local)")
    print("Cost: FREE (local compute)")

    # Initialize HuggingFace provider
    provider = initialize_provider('huggingface')

    try:
        # Get test images
        image_paths = get_test_images()

        # Process images (both phases)
        result = provider.process_images(image_paths)

        # Verify results
        verify_result_structure(result, expect_categories=True)
        print_sample_results(result)

        # Save results
        save_test_output('test_1_fully_local', result.to_dict())

        return True

    finally:
        provider.cleanup()


# ============================================================================
# TEST 2: Hybrid Workflow (HuggingFace → Ollama)
# ============================================================================

def test_2_hybrid_hf_ollama():
    """
    Test hybrid workflow: HuggingFace description → Ollama categorization.

    Phase 1: HuggingFace vision model describes images locally
    Phase 2: Ollama text model categorizes (local or remote)

    Benefits:
    - Privacy-conscious: images never leave local machine
    - Flexible: Ollama can run locally or on remote server
    - High quality categorization from Ollama

    Use case:
    - Sensitive images that can't be sent to cloud
    - Want better categorization than HuggingFace text models
    """
    print("\nWorkflow: HuggingFace Vision (local) → Ollama (local/remote)")
    print("Privacy: HIGH (images stay local)")
    print("Cost: FREE (local/self-hosted)")

    # Initialize providers
    hf_provider = initialize_provider('huggingface')
    ollama_provider = initialize_provider('ollama')

    try:
        # Get test images
        image_paths = get_test_images()

        # Phase 1: Describe images with HuggingFace
        print("\nPhase 1: Describing images with HuggingFace...")
        described_images = hf_provider.describe_images(image_paths)

        print(f"✓ Described {len(described_images)} images")

        # Phase 2: Categorize with Ollama
        print("\nPhase 2: Categorizing with Ollama...")
        result = ollama_provider.categorize_described_images(described_images)

        # Verify results
        verify_result_structure(result, expect_categories=True)
        print_sample_results(result)

        # Save results
        save_test_output('test_2_hybrid_hf_ollama', result.to_dict())

        return True

    finally:
        hf_provider.cleanup()
        ollama_provider.cleanup()


# ============================================================================
# TEST 3: Cloud Workflow (Ollama → Ollama) - Baseline
# ============================================================================

def test_3_cloud_baseline():
    """
    Test baseline workflow: Ollama for both phases.

    Phase 1: Ollama vision model describes images
    Phase 2: Ollama model categorizes based on descriptions

    This is the traditional single-provider approach for comparison.

    Benefits:
    - Simple setup (one provider)
    - Consistent quality across phases
    - Can run locally or remotely
    """
    print("\nWorkflow: Ollama Vision → Ollama Categorization")
    print("Privacy: MEDIUM (depends on Ollama deployment)")
    print("Cost: FREE (local/self-hosted)")

    # Initialize Ollama provider
    provider = initialize_provider('ollama')

    try:
        # Get test images
        image_paths = get_test_images()

        # Process images (both phases)
        result = provider.process_images(image_paths)

        # Verify results
        verify_result_structure(result, expect_categories=True)
        print_sample_results(result)

        # Save results
        save_test_output('test_3_cloud_baseline', result.to_dict())

        return True

    finally:
        provider.cleanup()


# ============================================================================
# TEST 4: Description-Only Workflow (Save Intermediate Results)
# ============================================================================

def test_4_description_only():
    """
    Test description-only workflow: Generate and save descriptions.

    Phase 1 ONLY: HuggingFace vision model describes images
    Saves intermediate results to descriptions.json

    Use case:
    - Generate descriptions for later categorization
    - Experiment with different categorization strategies
    - Batch processing: describe locally, categorize later with cloud
    """
    print("\nWorkflow: HuggingFace Vision → Save Descriptions")
    print("Purpose: Generate intermediate results for later categorization")

    # Initialize HuggingFace provider
    provider = initialize_provider('huggingface')

    try:
        # Get test images
        image_paths = get_test_images()

        # Phase 1: Describe images only
        print("\nPhase 1: Describing images...")
        described_images = provider.describe_images(image_paths)

        print(f"✓ Described {len(described_images)} images")

        # Verify descriptions exist
        for img in described_images:
            assert img.description, f"Missing description for {img.filename}"
            assert img.suggested_categories, f"Missing suggested_categories for {img.filename}"

        # Save intermediate results
        descriptions_data = {
            'images': [img.to_dict() for img in described_images],
            'metadata': {
                'provider': 'huggingface',
                'phase': 'description_only',
                'total_images': len(described_images)
            }
        }

        save_test_output('test_4_description_only', descriptions_data, 'descriptions.json')

        # Also save as regular result for inspection
        result = CategorizationResult(
            images=described_images,
            processing_stats={'phase': 'description_only'}
        )
        save_test_output('test_4_description_only', result.to_dict())

        print_sample_results(result)

        return True

    finally:
        provider.cleanup()


# ============================================================================
# TEST 5: Categorization-Only Workflow (Load and Categorize)
# ============================================================================

def test_5_categorization_only():
    """
    Test categorization-only workflow: Load descriptions and categorize.

    Phase 2 ONLY: Ollama categorizes pre-described images
    Loads descriptions from Test 4's output

    Prerequisites:
    - Test 4 must have been run first to generate descriptions.json

    Use case:
    - Re-categorize images with different model/parameters
    - Categorize after batch description generation
    - Use cloud categorization on locally-described images
    """
    print("\nWorkflow: Load Descriptions → Ollama Categorization")
    print("Purpose: Categorize pre-described images")
    print("Prerequisite: Test 4 must have been run first")

    # Load descriptions from Test 4
    try:
        descriptions_data = load_test_input('test_4_description_only', 'descriptions.json')
    except FileNotFoundError:
        print("\n✗ ERROR: Test 4 must be run first to generate descriptions.json")
        print("  Run: python test_phase_separation_workflows.py --test test_4_description_only")
        return False

    # Reconstruct ImageData objects
    described_images = [ImageData.from_dict(img_data) for img_data in descriptions_data['images']]
    print(f"✓ Loaded {len(described_images)} pre-described images")

    # Initialize Ollama provider for categorization
    provider = initialize_provider('ollama')

    try:
        # Phase 2: Categorize pre-described images
        print("\nPhase 2: Categorizing with Ollama...")
        result = provider.categorize_described_images(described_images)

        # Verify results
        verify_result_structure(result, expect_categories=True)
        print_sample_results(result)

        # Save results
        save_test_output('test_5_categorization_only', result.to_dict())

        return True

    finally:
        provider.cleanup()


# ============================================================================
# TEST 6: Keyword Categorization Workflow
# ============================================================================

def test_6_keyword_categorization():
    """
    Test keyword-based categorization: HuggingFace description → Keyword matching.

    Phase 1: HuggingFace vision model describes images
    Phase 2: Simple keyword matching for categorization (no LLM)

    Benefits:
    - Fast categorization
    - Deterministic results
    - No additional compute needed
    - Good for simple/common categories

    Drawbacks:
    - Limited to predefined categories
    - Less flexible than LLM categorization
    """
    print("\nWorkflow: HuggingFace Vision → Keyword Matching")
    print("Privacy: HIGH (fully local)")
    print("Speed: VERY FAST (no LLM for categorization)")

    # Initialize providers
    hf_provider = initialize_provider('huggingface')
    keyword_provider = initialize_provider('keyword')

    try:
        # Get test images
        image_paths = get_test_images()

        # Phase 1: Describe images with HuggingFace
        print("\nPhase 1: Describing images with HuggingFace...")
        described_images = hf_provider.describe_images(image_paths)

        print(f"✓ Described {len(described_images)} images")

        # Phase 2: Categorize with keyword matching
        print("\nPhase 2: Categorizing with keyword matching...")
        result = keyword_provider.categorize_described_images(described_images)

        # Verify results
        verify_result_structure(result, expect_categories=True)
        print_sample_results(result)

        # Print available keyword categories
        print("\nAvailable keyword categories:")
        for category in keyword_provider.CATEGORY_KEYWORDS.keys():
            print(f"  - {category}")

        # Save results
        save_test_output('test_6_keyword_categorization', result.to_dict())

        return True

    finally:
        hf_provider.cleanup()
        keyword_provider.cleanup()


# ============================================================================
# TEST 7: Backward Compatibility (Single Provider)
# ============================================================================

def test_7_backward_compat():
    """
    Test backward compatibility: Single provider for both phases.

    Uses Ollama's process_images() method which internally handles both phases.
    This validates that the new phase separation doesn't break existing code.

    Benefits:
    - Simple API
    - Backward compatible with old code
    - Single provider configuration
    """
    print("\nWorkflow: Ollama process_images() - Single Provider")
    print("Purpose: Validate backward compatibility")
    print("API: Traditional process_images() method")

    # Initialize Ollama provider
    provider = initialize_provider('ollama')

    try:
        # Get test images
        image_paths = get_test_images()

        # Process with single method (both phases internal)
        print("\nProcessing images with single provider...")
        result = provider.process_images(image_paths)

        # Verify results
        verify_result_structure(result, expect_categories=True)
        print_sample_results(result)

        # Save results
        save_test_output('test_7_backward_compat', result.to_dict())

        return True

    finally:
        provider.cleanup()


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Test phase separation workflows for image categorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--test', '-t',
        help='Run specific test (default: run all tests)',
        choices=[
            'test_1_fully_local',
            'test_2_hybrid_hf_ollama',
            'test_3_cloud_baseline',
            'test_4_description_only',
            'test_5_categorization_only',
            'test_6_keyword_categorization',
            'test_7_backward_compat'
        ]
    )

    parser.add_argument(
        '--images', '-i',
        type=int,
        default=MAX_TEST_IMAGES,
        help=f'Number of test images to use (default: {MAX_TEST_IMAGES})'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(override=True)

    # Create test output directory
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize test runner
    runner = TestRunner()

    # Define all tests
    all_tests = {
        'test_1_fully_local': test_1_fully_local,
        'test_2_hybrid_hf_ollama': test_2_hybrid_hf_ollama,
        'test_3_cloud_baseline': test_3_cloud_baseline,
        'test_4_description_only': test_4_description_only,
        'test_5_categorization_only': test_5_categorization_only,
        'test_6_keyword_categorization': test_6_keyword_categorization,
        'test_7_backward_compat': test_7_backward_compat
    }

    # Use custom image limit if specified
    max_images = args.images

    print("=" * 80)
    print("IMAGE CATEGORIZATION - PHASE SEPARATION WORKFLOW TESTS")
    print("=" * 80)
    print(f"\nTest image directory: {TEST_IMAGE_DIR}")
    print(f"Test output directory: {TEST_OUTPUT_DIR}")
    print(f"Max images per test: {max_images}")

    # Monkey-patch get_test_images with custom limit if needed
    if max_images != MAX_TEST_IMAGES:
        original_get_test_images = get_test_images
        def get_test_images_custom(limit: int = max_images) -> List[str]:
            return original_get_test_images(limit)
        globals()['get_test_images'] = get_test_images_custom

    # Run tests
    if args.test:
        # Run specific test
        test_func = all_tests[args.test]
        runner.run_test(args.test, test_func)
    else:
        # Run all tests
        print(f"\nRunning all {len(all_tests)} tests...")
        for test_name, test_func in all_tests.items():
            runner.run_test(test_name, test_func)

    # Print summary
    runner.print_summary()

    # Return exit code
    failed = sum(1 for r in runner.results.values() if r['status'] != 'PASSED')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
