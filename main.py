"""
Image Categorizer - AI-powered image organization tool.

This tool uses LLM providers to automatically describe and categorize images,
generating interactive HTML reports for easy organization.
"""

import os
import sys
import json
import argparse
import webbrowser
import textwrap
from typing import Dict, Any, Optional
from pathlib import Path

# Import our new modular components
from core.config import get_provider_config
from dotenv import load_dotenv
from core.image_processor import ImageProcessor
from models.image_data import CategorizationResult, ImageData
from providers import BaseLLMProvider, OllamaProvider


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
    """Initialize the selected LLM provider."""
    
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


def process_images_with_provider(directory: str, provider: BaseLLMProvider, initial_categories: Optional[list[str]] = None) -> CategorizationResult:
    """Process images using the initialized provider."""
    
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
    """Main function."""

    # Load environment variables once at startup
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI-powered image categorization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python main.py /path/to/images
              python main.py /path/to/images --provider ollama
              
            For HTML generation from existing results:
              python core/html_generator.py /path/to/categorization_results.json
        """)
    )

    parser.add_argument("directory", nargs="?", help="Directory containing images to process")

    parser.add_argument("--provider", "-p",
                        help="LLM provider to use (ollama, anthropic, openai, bedrock). Required.")


    parser.add_argument("--no-html", action="store_true",
                        help="Skip opening HTML report in browser")

    parser.add_argument("--init-categories",
                        help="Initial category suggestions (comma-separated list or file path with one category per line)")

    args = parser.parse_args()

    # Validate directory
    if not args.directory or not os.path.isdir(args.directory):
        print("Error: directory argument is not specified or is not a directory.")
        parser.print_help()
        sys.exit(1)

    if not args.provider:
        print("Error: --provider argument is required.")
        parser.print_help()
        sys.exit(1)

    # Main processing workflow
    print("AI Image Categorizer")
    print("===================")

    # Load initial categories if provided
    initial_categories = None
    if args.init_categories:
        try:
            initial_categories = load_initial_categories(args.init_categories)
            print(f"Loaded {len(initial_categories)} initial categories: {', '.join(initial_categories)}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading initial categories: {e}")
            sys.exit(1)

    # Initialize provider
    provider = initialize_provider(args.provider)

    try:
        # Process images
        result = process_images_with_provider(args.directory, provider, initial_categories)
        if not result:
            sys.exit(1)

        # Save results to JSON file
        save_results(args.directory, result)

        # Generate HTML report using HTMLGenerator
        from core.html_generator import HTMLGenerator
        generator = HTMLGenerator()
        html_file = generator.generate_report(args.directory, result)

        # Open HTML report unless --no-html flag is set
        if html_file and not args.no_html:
            print("Opening HTML report in default browser...")
            webbrowser.open(f"file://{os.path.abspath(html_file)}")

        print("\n✓ Processing complete!")

    finally:
        # Cleanup provider resources
        provider.cleanup()


if __name__ == "__main__":
    main()