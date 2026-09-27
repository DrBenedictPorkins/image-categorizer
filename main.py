"""
Image Categorizer - AI-powered image organization tool.

This tool uses LLM providers to automatically describe and categorize images,
generating interactive HTML reports for easy organization.

Supports three workflow modes:
1. Full pipeline: Description + Categorization (default)
2. Description only: Generate descriptions and save to JSON
3. Categorization only: Load descriptions from JSON and categorize
"""

import os
import sys
import json
import argparse
import webbrowser
import textwrap
from typing import Dict, Any, Optional, List
from pathlib import Path

# Phase 1 output file, also read back to resume an interrupted run
DESCRIPTIONS_FILE = "descriptions_only.json"
# Phase 1 saves progress after this many newly described images
DESCRIBE_SAVE_EVERY = 10

# Import our new modular components
from core.config import (
    get_provider_config,
    get_description_provider_config,
    get_categorization_provider_config
)
from dotenv import load_dotenv
from core.image_processor import ImageProcessor
from models.image_data import CategorizationResult, ImageData
from providers import (
    BaseLLMProvider,
    OllamaProvider,
    HuggingFaceProvider,
    KeywordCategorizationProvider
)


def load_initial_categories(categories_input: str) -> list[str]:
    """
    Load initial categories from comma-separated string or file path.
    
    Args:
        categories_input: Either a comma-separated list of categories or a file path
                         containing one category per line
    
    Returns:
        List of cleaned category strings
    
    Raises:
        FileNotFoundError: If file path provided but file doesn't exist
        ValueError: If no valid categories found
    """
    # Check if input is a file path
    if os.path.isfile(categories_input):
        # Read file, strip whitespace, filter empty lines
        with open(categories_input, 'r', encoding='utf-8') as f:
            categories = [line.strip() for line in f if line.strip()]
    elif ',' in categories_input or not ('.' in categories_input or '/' in categories_input):
        # Parse comma-separated values, strip whitespace
        categories = [cat.strip() for cat in categories_input.split(',') if cat.strip()]
    else:
        # Treat as potential file path that doesn't exist
        raise FileNotFoundError(f"File not found: {categories_input}")
    
    if not categories:
        raise ValueError("No valid categories found in input")
    
    return categories


def get_provider_class(provider_name: str) -> type:
    """Get the provider class for the given provider name."""
    provider_classes = {
        'ollama': OllamaProvider,
        'huggingface': HuggingFaceProvider,
        'keyword': KeywordCategorizationProvider,
        # Future providers will be added here
        # 'anthropic': AnthropicProvider,
        # 'openai': OpenAIProvider,
        # 'bedrock': BedrockProvider,
    }

    if provider_name not in provider_classes:
        available = ', '.join(provider_classes.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")

    return provider_classes[provider_name]


def initialize_provider(provider_name: str) -> BaseLLMProvider:
    """
    Initialize the selected LLM provider using general configuration.

    This function is maintained for backward compatibility with existing workflows.
    For two-phase workflows, use initialize_description_provider() or
    initialize_categorization_provider() instead.

    Args:
        provider_name: Name of the provider to initialize

    Returns:
        Initialized provider instance
    """
    # Get provider configuration
    provider_config = get_provider_config(provider_name)
    if not provider_config:
        print(f"Error: Unsupported provider '{provider_name}'")
        sys.exit(1)

    # Get provider class and create instance
    try:
        provider_class = get_provider_class(provider_name)
        provider = provider_class(provider_config)

        # Let the provider validate its own configuration
        provider.validate_config()

        print(f"Initializing {provider.provider_name} provider...")

        if not provider.initialize():
            print(f"Error: Failed to initialize {provider.provider_name} provider")
            sys.exit(1)

        # Test connection
        print(f"Testing connection to {provider.provider_name}...")
        if not provider.test_connection():
            print(f"Error: Cannot connect to {provider.provider_name} service")
            sys.exit(1)

        print(f"✓ {provider.provider_name} provider ready")
        return provider

    except Exception as e:
        print(f"Error initializing provider: {e}")
        sys.exit(1)


def initialize_description_provider(provider_name: str) -> BaseLLMProvider:
    """
    Initialize a provider for Phase 1 (description) using description-specific configuration.

    Args:
        provider_name: Name of the provider to initialize

    Returns:
        Initialized provider instance
    """
    # Get description-specific configuration
    provider_config = get_description_provider_config(provider_name)
    if not provider_config:
        print(f"Error: Unsupported description provider '{provider_name}'")
        sys.exit(1)

    # Get provider class and create instance
    try:
        provider_class = get_provider_class(provider_name)
        provider = provider_class(provider_config)

        # Validate that provider supports description
        if not provider.supports_description:
            print(f"Error: Provider '{provider_name}' does not support image description")
            print(f"Please use a vision-capable provider like 'ollama' or 'huggingface'")
            sys.exit(1)

        # Let the provider validate its own configuration
        provider.validate_config()

        print(f"Initializing {provider.provider_name} provider for description...")

        if not provider.initialize():
            print(f"Error: Failed to initialize {provider.provider_name} provider")
            sys.exit(1)

        # Test connection
        print(f"Testing connection to {provider.provider_name}...")
        if not provider.test_connection():
            print(f"Error: Cannot connect to {provider.provider_name} service")
            sys.exit(1)

        print(f"✓ {provider.provider_name} provider ready for description")
        return provider

    except Exception as e:
        print(f"Error initializing description provider: {e}")
        sys.exit(1)


def initialize_categorization_provider(provider_name: str) -> BaseLLMProvider:
    """
    Initialize a provider for Phase 2 (categorization) using categorization-specific configuration.

    Args:
        provider_name: Name of the provider to initialize

    Returns:
        Initialized provider instance
    """
    # Get categorization-specific configuration
    provider_config = get_categorization_provider_config(provider_name)
    if not provider_config:
        print(f"Error: Unsupported categorization provider '{provider_name}'")
        sys.exit(1)

    # Get provider class and create instance
    try:
        provider_class = get_provider_class(provider_name)
        provider = provider_class(provider_config)

        # Validate that provider supports categorization
        if not provider.supports_categorization:
            print(f"Error: Provider '{provider_name}' does not support categorization")
            sys.exit(1)

        # Let the provider validate its own configuration
        provider.validate_config()

        print(f"Initializing {provider.provider_name} provider for categorization...")

        if not provider.initialize():
            print(f"Error: Failed to initialize {provider.provider_name} provider")
            sys.exit(1)

        # Test connection
        print(f"Testing connection to {provider.provider_name}...")
        if not provider.test_connection():
            print(f"Error: Cannot connect to {provider.provider_name} service")
            sys.exit(1)

        print(f"✓ {provider.provider_name} provider ready for categorization")
        return provider

    except Exception as e:
        print(f"Error initializing categorization provider: {e}")
        sys.exit(1)


def process_images_with_provider(directory: str, provider: BaseLLMProvider, initial_categories: Optional[list[str]] = None) -> CategorizationResult:
    """
    Process images using the initialized provider (full pipeline).

    This function maintains backward compatibility with existing workflows.

    Args:
        directory: Directory containing images
        provider: Initialized provider instance
        initial_categories: Optional initial category suggestions

    Returns:
        Complete CategorizationResult
    """
    # Discover images
    print(f"Discovering images in {directory}...")
    image_paths = ImageProcessor.discover_images(directory)

    if not image_paths:
        print(f"No image files found in {directory}")
        return None

    print(f"Found {len(image_paths)} images")

    # Validate images
    print("Validating images...")
    valid_paths, invalid_info = ImageProcessor.batch_validate_images(image_paths)

    if invalid_info:
        print(f"Warning: {len(invalid_info)} images could not be validated:")
        for info in invalid_info[:5]:  # Show first 5 errors
            print(f"  {info}")
        if len(invalid_info) > 5:
            print(f"  ... and {len(invalid_info) - 5} more")

    if not valid_paths:
        print("Error: No valid images found")
        return None

    print(f"Processing {len(valid_paths)} valid images...")

    # Progress callback
    def progress_callback(message: str, progress: float):
        progress_percent = int(progress * 100)
        print(f"[{progress_percent:3d}%] {message}")

    # Process images with provider
    try:
        result = provider.process_images(valid_paths, progress_callback, initial_categories)
        print(f"\n✓ Successfully processed {len(result.images)} images")
        print(f"✓ Generated {len(result.category_groups)} categories")
        return result

    except Exception as e:
        print(f"Error during image processing: {e}")
        return None


def describe_images_only(directory: str, provider: BaseLLMProvider, initial_categories: Optional[list[str]] = None) -> List[ImageData]:
    """
    Phase 1 only: Generate descriptions for images and save to JSON.

    Args:
        directory: Directory containing images
        provider: Initialized description provider
        initial_categories: Optional initial category suggestions

    Returns:
        List of ImageData with descriptions populated
    """
    # Discover images
    print(f"\n=== PHASE 1: DESCRIPTION ===")
    print(f"Discovering images in {directory}...")
    image_paths = ImageProcessor.discover_images(directory)

    if not image_paths:
        print(f"No image files found in {directory}")
        return None

    print(f"Found {len(image_paths)} images")

    # Validate images
    print("Validating images...")
    valid_paths, invalid_info = ImageProcessor.batch_validate_images(image_paths)

    if invalid_info:
        print(f"Warning: {len(invalid_info)} images could not be validated:")
        for info in invalid_info[:5]:
            print(f"  {info}")
        if len(invalid_info) > 5:
            print(f"  ... and {len(invalid_info) - 5} more")

    if not valid_paths:
        print("Error: No valid images found")
        return None

    # Resume: reuse successful descriptions already saved in descriptions_only.json
    output_file = os.path.join(directory, DESCRIPTIONS_FILE)
    described = load_saved_descriptions(output_file)
    remaining = [p for p in valid_paths if Path(p).name not in described]
    if described:
        print(f"Resuming: {len(described)} already described, {len(remaining)} remaining")

    print(f"Describing {len(remaining)} images...")

    # Describe one image per provider call and save every DESCRIBE_SAVE_EVERY
    # images, so a crash loses at most that many descriptions
    total = len(valid_paths)
    try:
        for n, path in enumerate(remaining, 1):
            done = len(described) + 1
            print(f"[{int(done / total * 100):3d}%] Describing image {done}/{total}: {Path(path).name}")
            for image in provider.describe_images([path], None, initial_categories):
                described[image.filename] = image
            if n % DESCRIBE_SAVE_EVERY == 0 or n == len(remaining):
                write_descriptions(output_file, _in_path_order(valid_paths, described))

    except KeyboardInterrupt:
        if described:
            write_descriptions(output_file, _in_path_order(valid_paths, described))
            print(f"\nSaved {len(described)} descriptions to {output_file}; rerun to resume")
        raise
    except Exception as e:
        print(f"Error during image description: {e}")
        if described:
            write_descriptions(output_file, _in_path_order(valid_paths, described))
            print(f"Saved {len(described)} descriptions to {output_file}; rerun to resume")
        return None

    images = _in_path_order(valid_paths, described)
    print(f"\n✓ Successfully described {len(images)} images")
    return images


def _in_path_order(paths: List[str], described: Dict[str, ImageData]) -> List[ImageData]:
    """Return described images in the order of paths, skipping undescribed ones."""
    return [described[Path(p).name] for p in paths if Path(p).name in described]


def _is_failed_description(image: ImageData) -> bool:
    """Whether a saved description is a failure marker that should be retried."""
    return image.description == "failed" or image.description.startswith("Error processing image")


def load_saved_descriptions(json_path: str) -> Dict[str, ImageData]:
    """
    Load successful descriptions from an existing descriptions file, by filename.

    Failed descriptions are left out so they are retried.
    """
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not read {json_path} ({e}); describing all images")
        return {}
    images = [ImageData.from_dict(d) for d in data]
    return {img.filename: img for img in images if not _is_failed_description(img)}


def write_descriptions(output_file: str, images: List[ImageData]):
    """Write descriptions to JSON atomically (temp file, then rename)."""
    tmp_file = f"{output_file}.tmp"
    with open(tmp_file, "w") as f:
        json.dump([img.to_dict() for img in images], f, indent=2)
    os.replace(tmp_file, output_file)


def categorize_images_from_descriptions(images: List[ImageData], provider: BaseLLMProvider) -> CategorizationResult:
    """
    Phase 2 only: Categorize images that already have descriptions.

    Args:
        images: List of ImageData with descriptions populated
        provider: Initialized categorization provider

    Returns:
        Complete CategorizationResult with categories assigned
    """
    print(f"\n=== PHASE 2: CATEGORIZATION ===")
    print(f"Categorizing {len(images)} described images...")

    # Progress callback
    def progress_callback(message: str, progress: float):
        progress_percent = int(progress * 100)
        print(f"[{progress_percent:3d}%] {message}")

    # Categorize with provider
    try:
        result = provider.categorize_described_images(images, progress_callback)
        print(f"\n✓ Successfully categorized {len(result.images)} images")
        print(f"✓ Generated {len(result.category_groups)} categories")
        return result

    except Exception as e:
        print(f"Error during categorization: {e}")
        return None


def load_descriptions_from_json(json_path: str) -> List[ImageData]:
    """
    Load image descriptions from a JSON file.

    Args:
        json_path: Path to JSON file with image descriptions

    Returns:
        List of ImageData objects
    """
    print(f"Loading descriptions from {json_path}...")

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Support multiple JSON formats
        if isinstance(data, list):
            # List of ImageData dictionaries
            images = [ImageData.from_dict(img_data) for img_data in data]
        elif 'images' in data:
            # CategorizationResult format
            images = [ImageData.from_dict(img_data) for img_data in data['images']]
        else:
            print(f"Error: Unrecognized JSON format in {json_path}")
            return None

        print(f"✓ Loaded {len(images)} image descriptions")
        return images

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading descriptions: {e}")
        return None


def save_descriptions_only(directory: str, images: List[ImageData]):
    """
    Save image descriptions to JSON file (Phase 1 output).

    Args:
        directory: Directory to save JSON file
        images: List of ImageData with descriptions
    """
    output_file = os.path.join(directory, DESCRIPTIONS_FILE)
    write_descriptions(output_file, images)

    print(f"✓ Descriptions saved to {output_file}")

    # Display summary
    described_count = len([img for img in images if img.description])
    print(f"\nDescription summary:")
    print(f"  Total images: {len(images)}")
    print(f"  Successfully described: {described_count}")

    if any(img.suggested_categories for img in images):
        print(f"  Images with suggested categories: {len([img for img in images if img.suggested_categories])}")


def save_results(directory: str, result: CategorizationResult):
    """Save final categorization results to a single JSON file."""
    
    # Save complete result to single file
    output_file = os.path.join(directory, "categorization_results.json")
    
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"✓ Final results saved to {output_file}")
    
    # Display category distribution
    print("\nCategory distribution:")
    for category, count in sorted(result.get_category_stats().items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} images")






def main():
    """Main function with support for three workflow modes."""

    # Load environment variables once at startup - explicitly load .env file
    # Get the directory where main.py is located
    script_dir = Path(__file__).parent
    env_file = script_dir / '.env'
    load_dotenv(dotenv_path=env_file, override=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI-powered image categorization tool with support for two-phase workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Workflow Modes:

            1. Full Pipeline (default):
              python main.py /path/to/images --provider ollama

            2. Description Only (Phase 1):
              python main.py /path/to/images --description-provider huggingface --describe-only

            3. Categorization Only (Phase 2):
              python main.py /path/to/images --categorization-provider keyword --categorize-from descriptions_only.json

            4. Two-Phase with Different Providers:
              python main.py /path/to/images --description-provider huggingface --categorization-provider ollama

            Available Providers:
              - ollama: Local Ollama server (supports vision + categorization)
              - huggingface: HuggingFace models (supports vision + categorization)
              - keyword: Keyword-based categorization (categorization only, no vision)

            For HTML generation from existing results:
              python core/html_generator.py /path/to/categorization_results.json
        """)
    )

    parser.add_argument("directory", nargs="?", help="Directory containing images to process")

    # Legacy single provider argument (backward compatibility)
    parser.add_argument("--provider", "-p",
                        help="LLM provider for full pipeline (ollama, huggingface). Shortcut for using same provider for both phases.")

    # Two-phase provider arguments
    parser.add_argument("--description-provider",
                        help="Provider for Phase 1 (description): ollama, huggingface")

    parser.add_argument("--categorization-provider",
                        help="Provider for Phase 2 (categorization): ollama, huggingface, keyword")

    # Phase control arguments
    parser.add_argument("--describe-only", action="store_true",
                        help="Run Phase 1 only: describe images and save to descriptions_only.json")

    parser.add_argument("--categorize-from",
                        help="Run Phase 2 only: load descriptions from JSON file and categorize")

    parser.add_argument("--model", "-m",
                        help="Model to use (e.g., llava:latest for Ollama). Overrides environment variable.")

    parser.add_argument("--no-html", action="store_true",
                        help="Skip opening HTML report in browser")

    parser.add_argument("--init-categories",
                        help="Initial category suggestions (comma-separated list or file path with one category per line)")

    args = parser.parse_args()

    # Validate arguments and determine workflow mode
    workflow_mode = None
    desc_provider_name = None
    cat_provider_name = None

    if args.categorize_from:
        # Mode 3: Categorization only
        workflow_mode = "categorization_only"
        if not args.categorization_provider:
            print("Error: --categorization-provider is required when using --categorize-from")
            parser.print_help()
            sys.exit(1)
        cat_provider_name = args.categorization_provider
        # Directory is optional for categorization-only mode
        if not args.directory:
            args.directory = os.path.dirname(os.path.abspath(args.categorize_from))

    elif args.describe_only:
        # Mode 2: Description only
        workflow_mode = "description_only"
        if not args.description_provider:
            print("Error: --description-provider is required when using --describe-only")
            parser.print_help()
            sys.exit(1)
        desc_provider_name = args.description_provider
        if not args.directory or not os.path.isdir(args.directory):
            print("Error: directory argument is required for description mode")
            parser.print_help()
            sys.exit(1)

    elif args.description_provider or args.categorization_provider:
        # Mode 4: Two-phase with different providers
        workflow_mode = "two_phase"
        if not args.description_provider or not args.categorization_provider:
            print("Error: Both --description-provider and --categorization-provider are required for two-phase mode")
            parser.print_help()
            sys.exit(1)
        desc_provider_name = args.description_provider
        cat_provider_name = args.categorization_provider
        if not args.directory or not os.path.isdir(args.directory):
            print("Error: directory argument is required")
            parser.print_help()
            sys.exit(1)

    elif args.provider:
        # Mode 1: Full pipeline (backward compatibility)
        workflow_mode = "full_pipeline"
        desc_provider_name = args.provider
        cat_provider_name = args.provider
        if not args.directory or not os.path.isdir(args.directory):
            print("Error: directory argument is required")
            parser.print_help()
            sys.exit(1)

    else:
        print("Error: Must specify either --provider OR (--description-provider and/or --categorization-provider)")
        parser.print_help()
        sys.exit(1)

    # Main processing workflow
    print("AI Image Categorizer")
    print("===================")
    print(f"Workflow mode: {workflow_mode.replace('_', ' ').title()}")
    print()

    # Load initial categories if provided
    initial_categories = None
    if args.init_categories:
        try:
            initial_categories = load_initial_categories(args.init_categories)
            print(f"Loaded {len(initial_categories)} initial categories: {', '.join(initial_categories)}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading initial categories: {e}")
            sys.exit(1)

    # Override model if specified via command line
    if args.model:
        if desc_provider_name == 'ollama' or cat_provider_name == 'ollama':
            os.environ['OLLAMA_MODEL'] = args.model
        elif desc_provider_name == 'anthropic' or cat_provider_name == 'anthropic':
            os.environ['ANTHROPIC_MODEL'] = args.model
        elif desc_provider_name == 'openai' or cat_provider_name == 'openai':
            os.environ['OPENAI_MODEL'] = args.model
        print(f"Using model: {args.model}")

    # Execute workflow based on mode
    try:
        if workflow_mode == "description_only":
            # Phase 1 only: Describe images
            provider = initialize_description_provider(desc_provider_name)
            try:
                images = describe_images_only(args.directory, provider, initial_categories)
                if not images:
                    sys.exit(1)

                # Save descriptions
                save_descriptions_only(args.directory, images)
                print("\n✓ Description phase complete!")
                print(f"\nNext step: Run categorization phase with:")
                print(f"  python main.py {args.directory} --categorization-provider <provider> --categorize-from descriptions_only.json")

            finally:
                provider.cleanup()

        elif workflow_mode == "categorization_only":
            # Phase 2 only: Categorize from JSON
            images = load_descriptions_from_json(args.categorize_from)
            if not images:
                sys.exit(1)

            provider = initialize_categorization_provider(cat_provider_name)
            try:
                result = categorize_images_from_descriptions(images, provider)
                if not result:
                    sys.exit(1)

                # Save results
                save_results(args.directory, result)

                # Generate HTML report
                from core.html_generator import HTMLGenerator
                generator = HTMLGenerator()
                html_file = generator.generate_report(args.directory, result)

                # Open HTML report unless --no-html flag is set
                if html_file and not args.no_html:
                    print("Opening HTML report in default browser...")
                    webbrowser.open(f"file://{os.path.abspath(html_file)}")

                print("\n✓ Categorization phase complete!")

            finally:
                provider.cleanup()

        elif workflow_mode == "two_phase":
            # Two-phase workflow with different providers
            desc_provider = initialize_description_provider(desc_provider_name)
            cat_provider = initialize_categorization_provider(cat_provider_name)

            try:
                # Phase 1: Describe
                images = describe_images_only(args.directory, desc_provider, initial_categories)
                if not images:
                    sys.exit(1)

                # Save intermediate results
                save_descriptions_only(args.directory, images)

                # Phase 2: Categorize
                result = categorize_images_from_descriptions(images, cat_provider)
                if not result:
                    sys.exit(1)

                # Save final results
                save_results(args.directory, result)

                # Generate HTML report
                from core.html_generator import HTMLGenerator
                generator = HTMLGenerator()
                html_file = generator.generate_report(args.directory, result)

                # Open HTML report unless --no-html flag is set
                if html_file and not args.no_html:
                    print("Opening HTML report in default browser...")
                    webbrowser.open(f"file://{os.path.abspath(html_file)}")

                print("\n✓ Two-phase processing complete!")

            finally:
                desc_provider.cleanup()
                cat_provider.cleanup()

        elif workflow_mode == "full_pipeline":
            # Full pipeline (backward compatibility)
            provider = initialize_provider(args.provider)
            try:
                result = process_images_with_provider(args.directory, provider, initial_categories)
                if not result:
                    sys.exit(1)

                # Save results
                save_results(args.directory, result)

                # Generate HTML report
                from core.html_generator import HTMLGenerator
                generator = HTMLGenerator()
                html_file = generator.generate_report(args.directory, result)

                # Open HTML report unless --no-html flag is set
                if html_file and not args.no_html:
                    print("Opening HTML report in default browser...")
                    webbrowser.open(f"file://{os.path.abspath(html_file)}")

                print("\n✓ Processing complete!")

            finally:
                provider.cleanup()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()