"""
HuggingFace LLM provider for image categorization.

This provider uses local HuggingFace models for both image description (vision models)
and categorization (text models). All processing is done locally without external API calls.

Supported Vision Models:
- BLIP-2 (Salesforce/blip2-flan-t5-xl-coco) - 15GB, high quality
- LLaVA-1.5 (llava-hf/llava-1.5-7b-hf) - 13GB, conversational
- BLIP-base (Salesforce/blip-image-captioning-base) - 2GB, faster
- MiniCPM-V (openbmb/MiniCPM-V-2) - 8GB, efficient

Supported Text Models:
- Flan-T5-XL (google/flan-t5-xl) - 3GB, good quality
- Phi-2 (microsoft/phi-2) - 5GB, powerful
"""

import os
import gc
import textwrap
import yaml
import json
import time
from io import BytesIO
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration,
)

from .base import BaseLLMProvider, ProviderError, ProviderConnectionError, ProviderProcessingError
from models.image_data import ImageData, CategorizationResult, ProviderConfig
from core.image_processor import ImageProcessor


# Model configurations
VISION_MODELS = {
    'blip2': {
        'model_id': 'Salesforce/blip2-flan-t5-xl-coco',
        'processor_class': Blip2Processor,
        'model_class': Blip2ForConditionalGeneration,
        'size_gb': 15,
        'description': 'BLIP-2 with Flan-T5-XL - High quality descriptions'
    },
    'llava': {
        'model_id': 'llava-hf/llava-1.5-7b-hf',
        'processor_class': AutoProcessor,
        'model_class': AutoModelForImageTextToText,
        'size_gb': 13,
        'description': 'LLaVA 1.5 - Conversational vision model'
    },
    'blip-base': {
        'model_id': 'Salesforce/blip-image-captioning-base',
        'processor_class': BlipProcessor,
        'model_class': BlipForConditionalGeneration,
        'size_gb': 2,
        'description': 'BLIP Base - Fast and lightweight'
    },
    'minicpm': {
        'model_id': 'openbmb/MiniCPM-V-2',
        'processor_class': AutoProcessor,
        'model_class': AutoModelForImageTextToText,
        'size_gb': 8,
        'description': 'MiniCPM-V - Efficient and accurate'
    },
}

TEXT_MODELS = {
    'flan-t5-xl': {
        'model_id': 'google/flan-t5-xl',
        'model_class': AutoModelForSeq2SeqLM,
        'tokenizer_class': AutoTokenizer,
        'size_gb': 3,
        'description': 'Flan-T5-XL - Good quality text generation'
    },
    'phi-2': {
        'model_id': 'microsoft/phi-2',
        'model_class': AutoModelForCausalLM,
        'tokenizer_class': AutoTokenizer,
        'size_gb': 5,
        'description': 'Phi-2 - Powerful small language model'
    },
}


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace provider for local image description and categorization."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.vision_model_name = None
        self.text_model_name = None
        self.device = None
        self.vision_processor = None
        self.vision_model = None
        self.text_tokenizer = None
        self.text_model = None
        self.cache_dir = None
        self.hf_token = None

    @property
    def provider_name(self) -> str:
        return "HuggingFace"

    @property
    def required_config_keys(self) -> List[str]:
        # No required keys - everything has defaults
        return []

    @property
    def supports_vision(self) -> bool:
        return True

    def initialize(self) -> bool:
        """Initialize the HuggingFace provider with device detection and model loading."""
        try:
            # Get configuration
            self.vision_model_name = self.config.get('vision_model', 'Salesforce/blip2-flan-t5-xl-coco')
            self.text_model_name = self.config.get('text_model', 'google/flan-t5-xl')
            self.cache_dir = self.config.get('cache_dir', None)
            self.hf_token = self.config.get('hf_token', None)

            # Detect device: MPS (M3 MAX) > CUDA > CPU
            self.device = self._detect_device(self.config.get('device', 'auto'))
            print(f"Using device: {self.device}")

            # Login to HuggingFace if token provided
            if self.hf_token:
                print("Authenticating with HuggingFace...")
                from huggingface_hub import login
                login(token=self.hf_token)

            # Load models
            print(f"Loading vision model: {self.vision_model_name}")
            self._load_vision_model()

            print(f"Loading text model: {self.text_model_name}")
            self._load_text_model()

            self._initialized = True
            print("HuggingFace provider initialized successfully")
            return True

        except Exception as e:
            print(f"Failed to initialize HuggingFace provider: {e}")
            return False

    def _detect_device(self, device_preference: str) -> str:
        """Detect the best available device for inference."""
        if device_preference != 'auto':
            # User specified a device
            if device_preference == 'mps' and not torch.backends.mps.is_available():
                print(f"Warning: MPS requested but not available, falling back to CPU")
                return 'cpu'
            if device_preference == 'cuda' and not torch.cuda.is_available():
                print(f"Warning: CUDA requested but not available, falling back to CPU")
                return 'cpu'
            return device_preference

        # Auto-detect: MPS > CUDA > CPU
        try:
            if torch.backends.mps.is_available():
                return 'mps'
        except:
            pass

        if torch.cuda.is_available():
            return 'cuda'

        return 'cpu'

    def _load_vision_model(self):
        """Load the vision model for image description."""
        # Find model config
        model_config = None
        for config in VISION_MODELS.values():
            if config['model_id'] == self.vision_model_name:
                model_config = config
                break

        if not model_config:
            # Try to load as a custom model ID
            print(f"Warning: '{self.vision_model_name}' not in predefined models, attempting to load as custom model")
            self.vision_processor = AutoProcessor.from_pretrained(
                self.vision_model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            self.vision_model = AutoModelForImageTextToText.from_pretrained(
                self.vision_model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                dtype=torch.float32
            ).to(self.device)
        else:
            # Load predefined model
            print(f"Loading {model_config['description']} (~{model_config['size_gb']}GB)")
            processor_class = model_config['processor_class']
            model_class = model_config['model_class']

            self.vision_processor = processor_class.from_pretrained(
                self.vision_model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            self.vision_model = model_class.from_pretrained(
                self.vision_model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                dtype=torch.float32
            ).to(self.device)

        # Set to eval mode
        self.vision_model.eval()

    def _load_text_model(self):
        """Load the text model for categorization."""
        # Find model config
        model_config = None
        for config in TEXT_MODELS.values():
            if config['model_id'] == self.text_model_name:
                model_config = config
                break

        if not model_config:
            # Try to load as a custom model ID
            print(f"Warning: '{self.text_model_name}' not in predefined models, attempting to load as custom model")
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            # Try seq2seq first, then causal LM
            try:
                self.text_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.text_model_name,
                    cache_dir=self.cache_dir,
                    token=self.hf_token,
                    dtype=torch.float32
                ).to(self.device)
            except:
                self.text_model = AutoModelForCausalLM.from_pretrained(
                    self.text_model_name,
                    cache_dir=self.cache_dir,
                    token=self.hf_token,
                    dtype=torch.float32
                ).to(self.device)
        else:
            # Load predefined model
            print(f"Loading {model_config['description']} (~{model_config['size_gb']}GB)")
            tokenizer_class = model_config['tokenizer_class']
            model_class = model_config['model_class']

            self.text_tokenizer = tokenizer_class.from_pretrained(
                self.text_model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            self.text_model = model_class.from_pretrained(
                self.text_model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                dtype=torch.float32
            ).to(self.device)

        # Set to eval mode
        self.text_model.eval()

    def validate_config(self) -> None:
        """Validate HuggingFace configuration."""
        # All configuration is optional with defaults, so just validate types
        vision_model = self.config.get('vision_model', 'Salesforce/blip2-flan-t5-xl-coco')
        text_model = self.config.get('text_model', 'google/flan-t5-xl')

        if not isinstance(vision_model, str):
            print(f"Error: vision_model must be a string, got {type(vision_model)}")
            import sys
            sys.exit(1)

        if not isinstance(text_model, str):
            print(f"Error: text_model must be a string, got {type(text_model)}")
            import sys
            sys.exit(1)

    def test_connection(self) -> bool:
        """Test if models are loaded and functional."""
        if not self._initialized:
            return False

        try:
            # Test vision model with a dummy image
            dummy_image = Image.new('RGB', (224, 224), color='red')
            with torch.no_grad():
                inputs = self.vision_processor(images=dummy_image, return_tensors="pt").to(self.device)
                _ = self.vision_model.generate(**inputs, max_new_tokens=10)

            # Test text model with a dummy prompt
            with torch.no_grad():
                inputs = self.text_tokenizer("Test prompt", return_tensors="pt").to(self.device)
                _ = self.text_model.generate(**inputs, max_new_tokens=10)

            return True

        except Exception as e:
            print(f"Model test failed: {e}")
            return False

    def process_images(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        initial_categories: Optional[List[str]] = None
    ) -> CategorizationResult:
        """Process images with HuggingFace models for description and categorization."""

        if not self._initialized:
            raise ProviderError("Provider not initialized")

        # Validate image paths
        valid_paths = self._validate_image_paths(image_paths)
        if not valid_paths:
            raise ProviderProcessingError("No valid images found")

        self._report_progress(progress_callback, "Starting image processing...", 0.0)

        # Step 1: Generate descriptions for all images
        image_data_list = self.describe_images(valid_paths, progress_callback, initial_categories)

        # Step 2: Categorize all images collectively
        self._report_progress(progress_callback, "Analyzing images for categorization...", 0.7)

        result = self.categorize_described_images(image_data_list, progress_callback)

        self._report_progress(progress_callback, "Processing complete!", 1.0)
        return result

    def describe_images(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        initial_categories: Optional[List[str]] = None
    ) -> List[ImageData]:
        """Phase 1: Generate descriptions for images using vision model."""

        if not self._initialized:
            raise ProviderError("Provider not initialized")

        image_data_list = []
        total_images = len(image_paths)

        for i, image_path in enumerate(image_paths):
            self._report_progress(
                progress_callback,
                f"Describing image {i+1}/{total_images}: {Path(image_path).name}",
                (i / total_images) * 0.6  # Description takes 60% of progress
            )

            try:
                description_data = self._describe_image(image_path, initial_categories)
                image_data = ImageData(
                    filename=Path(image_path).name,
                    filepath=image_path,
                    description=description_data['description'],
                    suggested_categories=description_data['initial_categories'],
                    primary_category=description_data['initial_categories'][0] if description_data['initial_categories'] else 'Uncategorized',
                    metadata={
                        'provider': 'huggingface',
                        'vision_model': self.vision_model_name,
                        'device': self.device
                    }
                )
                image_data_list.append(image_data)

            except Exception as e:
                print(f"Error describing image {image_path}: {e}")
                # Create a fallback image data entry
                image_data = ImageData(
                    filename=Path(image_path).name,
                    filepath=image_path,
                    description=f"Error processing image: {str(e)}",
                    suggested_categories=['Error'],
                    primary_category='Error',
                    metadata={
                        'provider': 'huggingface',
                        'vision_model': self.vision_model_name,
                        'device': self.device,
                        'error': str(e)
                    }
                )
                image_data_list.append(image_data)

            # Force garbage collection after each image to manage memory
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return image_data_list

    def _describe_image(self, image_path: str, initial_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a structured description for a single image using vision model."""

        # Load and prepare image
        try:
            img = ImageProcessor.prepare_image_for_processing(image_path)
        except Exception as e:
            raise ProviderProcessingError(f"Failed to process image {image_path}: {e}")

        # Build description prompt with optional initial categories guidance
        base_prompt = "Describe this image in detail for categorization purposes. Include visible objects, people, setting, colors, and text."

        # Add initial categories guidance if provided
        if initial_categories:
            categories_list = ", ".join(initial_categories)
            base_prompt += f" Consider these categories: {categories_list}"

        # Generate description using vision model
        try:
            with torch.no_grad():
                # Check if model supports text prompts (like BLIP-2, LLaVA)
                if hasattr(self.vision_processor, 'tokenizer') or 'blip2' in self.vision_model_name.lower() or 'llava' in self.vision_model_name.lower():
                    # Model supports text prompts
                    inputs = self.vision_processor(images=img, text=base_prompt, return_tensors="pt").to(self.device)
                else:
                    # Model doesn't support text prompts (like BLIP-base)
                    inputs = self.vision_processor(images=img, return_tensors="pt").to(self.device)

                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.8,
                    num_beams=5,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    length_penalty=1.5,
                    no_repeat_ngram_size=3
                )

                # Decode the generated text
                generated_text = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Clean up the output - remove the prompt if it appears at the beginning
                if base_prompt and generated_text.startswith(base_prompt):
                    description = generated_text[len(base_prompt):].strip()
                else:
                    description = generated_text.strip()

                # Validate description quality
                if len(description.split()) < 3:
                    raise ValueError("Description too short")

        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            description = "Unable to generate detailed description"

        # Use text model to suggest initial categories based on description
        initial_categories_list = self._suggest_categories_from_description(description, initial_categories)

        return {
            "description": description,
            "initial_categories": initial_categories_list
        }

    def _suggest_categories_from_description(
        self,
        description: str,
        initial_categories: Optional[List[str]] = None
    ) -> List[str]:
        """Use text model to suggest 2-5 initial categories based on description."""

        categories_guidance = ""
        if initial_categories:
            categories_list = ", ".join(initial_categories)
            categories_guidance = f" Consider these suggested categories: {categories_list}."

        prompt = textwrap.dedent(f"""
            Based on this image description, suggest 2-5 appropriate categories for organization.

            Description: {description}
            {categories_guidance}

            Return ONLY a YAML list of categories:
            - Category 1
            - Category 2
            - Category 3

            Use simple, folder-friendly category names (1-3 words).
        """).strip()

        try:
            with torch.no_grad():
                inputs = self.text_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

                # Generate response
                outputs = self.text_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )

                response = self.text_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                # Try to parse YAML
                try:
                    categories = yaml.safe_load(response)
                    if isinstance(categories, list) and len(categories) > 0:
                        # Clean and validate categories
                        categories = [str(cat).strip() for cat in categories if cat][:5]
                        if categories:
                            return categories
                except:
                    pass

                # Fallback: extract categories from text
                lines = response.strip().split('\n')
                categories = []
                for line in lines:
                    # Remove list markers (-, *, numbers)
                    clean_line = line.strip().lstrip('-*•').strip()
                    if clean_line and len(clean_line.split()) <= 5:
                        categories.append(clean_line)
                        if len(categories) >= 5:
                            break

                if categories:
                    return categories[:5]

        except Exception as e:
            print(f"Error suggesting categories from description: {e}")

        # Ultimate fallback: keyword-based categorization
        return self._keyword_based_categories(description)

    def _keyword_based_categories(self, description: str) -> List[str]:
        """Simple keyword-based category suggestion as fallback."""
        description_lower = description.lower()

        category_keywords = {
            "People": ["person", "people", "man", "woman", "child", "human", "face", "portrait"],
            "Nature": ["tree", "forest", "mountain", "sky", "flower", "landscape", "outdoor", "nature"],
            "Food": ["food", "meal", "dish", "cooking", "restaurant", "kitchen"],
            "Animals": ["animal", "dog", "cat", "pet", "wildlife", "bird", "fish"],
            "Architecture": ["building", "house", "structure", "bridge", "architecture"],
            "Vehicles": ["car", "vehicle", "truck", "bike", "motorcycle", "transportation"],
            "Technology": ["computer", "phone", "device", "screen", "electronic"],
            "Art": ["art", "painting", "drawing", "sculpture", "artwork"]
        }

        matches = []
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                matches.append((category, score))

        # Sort by score and take top 3
        matches.sort(key=lambda x: x[1], reverse=True)
        categories = [cat for cat, _ in matches[:3]]

        if not categories:
            categories = ["General"]

        return categories

    def categorize_described_images(
        self,
        images: List[ImageData],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> CategorizationResult:
        """Phase 2: Categorize images that already have descriptions using text model."""

        # Filter out failed images
        valid_images = [img for img in images if img.description and not img.description.startswith("Error")]

        if not valid_images:
            print("No valid images to categorize")
            # Return all images with fallback categorization
            self._apply_fallback_categorization(images)
            return CategorizationResult(
                images=images,
                processing_stats={
                    'provider': 'huggingface',
                    'vision_model': self.vision_model_name,
                    'text_model': self.text_model_name,
                    'device': self.device,
                    'total_images': len(images),
                    'successful_descriptions': 0,
                    'categories_generated': len(set(img.primary_category for img in images))
                }
            )

        print(f"Categorizing {len(valid_images)} valid images (skipping {len(images) - len(valid_images)} failed)")

        # Prepare data for categorization
        try:
            self._categorize_images_with_text_model(images)
        except Exception as e:
            print(f"Error during categorization: {e}")
            # Apply fallback categorization
            self._apply_fallback_categorization(images)

        # Create the result
        result = CategorizationResult(
            images=images,
            processing_stats={
                'provider': 'huggingface',
                'vision_model': self.vision_model_name,
                'text_model': self.text_model_name,
                'device': self.device,
                'total_images': len(images),
                'successful_descriptions': len([img for img in images if not img.description.startswith('Error')]),
                'categories_generated': len(set(img.primary_category for img in images))
            }
        )

        return result

    def _categorize_images_with_text_model(self, image_data_list: List[ImageData]):
        """Categorize images using the text model."""

        # Filter valid images
        valid_images = [img for img in image_data_list if img.description and not img.description.startswith("Error")]

        if not valid_images:
            return

        # Prepare prompt with all images
        filename_blocks = []
        for img in valid_images:
            initial_categories_str = ", ".join(img.suggested_categories) if img.suggested_categories else "None"
            block = f"Filename: {img.filename}\nDescription: {img.description}\nSuggested Categories: {initial_categories_str}"
            filename_blocks.append(block)

        filename_blocks_text = "\n\n".join(filename_blocks)

        prompt = textwrap.dedent(f"""
            Analyze these images and their suggested categories to create final categories:

            {filename_blocks_text}

            INSTRUCTIONS:
            1. Look at ALL "Suggested Categories" to identify common themes
            2. Create 5-10 consolidated final categories
            3. Assign each image to ONE final category

            Return ONLY this YAML format:
            - filename: exact_filename.jpg
              final_category: Category Name
            - filename: another_file.jpg
              final_category: Another Category

            Include ALL filenames.
        """).strip()

        try:
            with torch.no_grad():
                # Truncate prompt if too long
                inputs = self.text_tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=2048,
                    truncation=True
                ).to(self.device)

                outputs = self.text_model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )

                response = self.text_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                # Parse YAML response
                try:
                    categories_list = yaml.safe_load(response)

                    if isinstance(categories_list, list) and categories_list:
                        # Validate and apply categories
                        self._apply_categories_by_filename(image_data_list, categories_list)
                        return
                except:
                    pass

                # Fallback: try to extract from text
                print("Failed to parse YAML, applying fallback categorization")
                self._apply_fallback_categorization(valid_images)

        except Exception as e:
            print(f"Error in text model categorization: {e}")
            self._apply_fallback_categorization(valid_images)

    def _apply_categories_by_filename(self, image_data_list: List[ImageData], categories_list: List[Dict[str, str]]):
        """Apply final categories using filename matching."""
        category_lookup = {}
        for item in categories_list:
            if isinstance(item, dict) and 'filename' in item and 'final_category' in item:
                category_lookup[item['filename']] = item['final_category']

        # Update existing ImageData objects
        for image_data in image_data_list:
            if image_data.filename in category_lookup:
                final_category = category_lookup[image_data.filename].strip()
                image_data.primary_category = final_category
            else:
                # Keep initial category if no match found
                if not image_data.primary_category or image_data.primary_category == 'Error':
                    image_data.primary_category = image_data.suggested_categories[0] if image_data.suggested_categories else 'Uncategorized'

    def _apply_fallback_categorization(self, image_data_list: List[ImageData]):
        """Apply fallback categorization based on suggested categories."""
        for image_data in image_data_list:
            # Use first suggested category if available
            if image_data.suggested_categories and image_data.suggested_categories != ['Error']:
                image_data.primary_category = image_data.suggested_categories[0]
            elif not image_data.primary_category or image_data.primary_category == 'Error':
                image_data.primary_category = 'Uncategorized'

    def get_capabilities(self) -> Dict[str, Any]:
        """Get HuggingFace provider capabilities."""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            'max_image_size': None,  # No specific limit
            'concurrent_requests': 1,  # Process sequentially to manage memory
            'supports_batch_processing': True,
            'requires_internet': False,  # Local processing only
            'supported_vision_models': list(VISION_MODELS.keys()),
            'supported_text_models': list(TEXT_MODELS.keys()),
            'device': self.device if self._initialized else 'unknown'
        })
        return base_capabilities

    def cleanup(self):
        """Clean up resources used by the provider."""
        if self.vision_model:
            del self.vision_model
            self.vision_model = None

        if self.vision_processor:
            del self.vision_processor
            self.vision_processor = None

        if self.text_model:
            del self.text_model
            self.text_model = None

        if self.text_tokenizer:
            del self.text_tokenizer
            self.text_tokenizer = None

        # Clear GPU cache if using CUDA
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        print("HuggingFace provider cleaned up")


def list_available_models():
    """Print available vision and text models."""
    print("\n=== Available Vision Models ===")
    for key, config in VISION_MODELS.items():
        print(f"\n{key}:")
        print(f"  Model ID: {config['model_id']}")
        print(f"  Size: ~{config['size_gb']}GB")
        print(f"  Description: {config['description']}")

    print("\n=== Available Text Models ===")
    for key, config in TEXT_MODELS.items():
        print(f"\n{key}:")
        print(f"  Model ID: {config['model_id']}")
        print(f"  Size: ~{config['size_gb']}GB")
        print(f"  Description: {config['description']}")

    print("\n")


if __name__ == "__main__":
    list_available_models()
