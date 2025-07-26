"""
Ollama LLM provider for image categorization.

This provider uses a remote Ollama server to perform both image description
and categorization in a single workflow.
"""

import json
import base64
import ollama
import textwrap
import yaml
from io import BytesIO
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

from .base import BaseLLMProvider, ProviderError, ProviderConnectionError, ProviderProcessingError
from models.image_data import ImageData, CategorizationResult, ProviderConfig
from core.image_processor import ImageProcessor


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for image description and categorization."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None
        self.model = None
        self.timeout = 300
    
    @property
    def provider_name(self) -> str:
        return "Ollama"
    
    @property
    def required_config_keys(self) -> List[str]:
        return ['host', 'model']
    
    @property
    def supports_vision(self) -> bool:
        return True
    
    def initialize(self) -> bool:
        """Initialize the Ollama provider."""
        try:
            host = self.config.get('host', 'http://localhost:11434')
            self.model = self.config.get('model', 'llama3.2-vision:latest')
            self.timeout = self.config.get('timeout', 300)
            
            # Create Ollama client
            self.client = ollama.Client(host=host)
            
            # Test connection
            if not self.test_connection():
                return False
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize Ollama provider: {e}")
            return False
    
    def validate_config(self) -> None:
        """Validate Ollama configuration."""
        import sys
        
        # Check required keys
        for key in self.required_config_keys:
            if not self.config.get(key):
                print(f"Error: Missing required configuration for Ollama: {key}")
                if key == 'host':
                    print("Set OLLAMA_HOST environment variable (e.g., http://localhost:11434)")
                elif key == 'model':
                    print("Set OLLAMA_MODEL environment variable (e.g., llama3.2-vision:latest)")
                sys.exit(1)
        
        # Validate host format
        host = self.config.get('host', '')
        if host and not (host.startswith('http://') or host.startswith('https://')):
            print(f"Error: Ollama host must start with http:// or https://, got: {host}")
            sys.exit(1)
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            # Try to list models to test connection
            models = self.client.list()
            return True
        except Exception:
            return False
    
    def process_images(
        self, 
        image_paths: List[str], 
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> CategorizationResult:
        """Process images with Ollama for description and categorization."""
        
        if not self._initialized:
            raise ProviderError("Provider not initialized")
        
        # Validate image paths
        valid_paths = self._validate_image_paths(image_paths)
        if not valid_paths:
            raise ProviderProcessingError("No valid images found")
        
        self._report_progress(progress_callback, "Starting image processing...", 0.0)
        
        # Step 1: Generate descriptions for all images
        image_data_list = []
        total_images = len(valid_paths)
        
        for i, image_path in enumerate(valid_paths):
            self._report_progress(
                progress_callback, 
                f"Describing image {i+1}/{total_images}: {Path(image_path).name}", 
                (i / total_images) * 0.6  # Description takes 60% of progress
            )
            
            try:
                description_data = self._describe_image(image_path)
                image_data = ImageData(
                    filename=Path(image_path).name,
                    filepath=image_path,
                    description=description_data['description'],
                    suggested_categories=description_data['initial_categories'],  # All suggested categories from initial LLM
                    primary_category=description_data['initial_categories'][0] if description_data['initial_categories'] else 'Uncategorized',  # First suggested as primary
                    metadata={'provider': 'ollama', 'model': self.model}
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
                    metadata={'provider': 'ollama', 'model': self.model, 'error': str(e)}
                )
                image_data_list.append(image_data)
        
        # Step 2: Categorize all images collectively
        self._report_progress(progress_callback, "Analyzing images for categorization...", 0.7)
        
        try:
            self._categorize_images(image_data_list)
        except Exception as e:
            print(f"Error during categorization: {e}")
            # Apply fallback categorization
            self._apply_fallback_categorization(image_data_list)
        
        self._report_progress(progress_callback, "Finalizing results...", 0.9)
        
        # Create the result
        result = CategorizationResult(
            images=image_data_list,
            processing_stats={
                'provider': 'ollama',
                'model': self.model,
                'total_images': len(image_data_list),
                'successful_descriptions': len([img for img in image_data_list if not img.description.startswith('Error')]),
                'categories_generated': len(set(img.primary_category for img in image_data_list))
            }
        )
        
        self._report_progress(progress_callback, "Processing complete!", 1.0)
        return result
    
    def _describe_image(self, image_path: str) -> Dict[str, Any]:
        """Generate a structured description for a single image using Ollama with retry on format failure."""
        
        # Load and encode image once
        try:
            img = ImageProcessor.prepare_image_for_processing(image_path)
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            raise ProviderProcessingError(f"Failed to process image {image_path}: {e}")
        
        # Clean structured YAML prompt
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
        
        # Try up to 2 times (original + 1 retry)
        for attempt in range(2):
            try:
                # Make request to Ollama
                response = self.client.chat(
                    model=self.model,
                    messages=[{
                        'role': 'user',
                        'content': description_prompt,
                        'images': [image_b64]
                    }]
                )
                
                response_text = response['message']['content'].strip()
                
                if not response_text:
                    if attempt == 0:
                        print(f"Empty response for {image_path}, retrying...")
                        continue
                    else:
                        break
                
                
                # Validate YAML format strictly
                try:
                    result = yaml.safe_load(response_text)
                    
                    # Validate required fields and types
                    if not isinstance(result, dict):
                        raise ValueError("Response is not a YAML object")
                    if 'description' not in result or not isinstance(result['description'], str):
                        raise ValueError("Missing or invalid 'description' field")
                    if 'initial_categories' not in result or not isinstance(result['initial_categories'], list):
                        raise ValueError("Missing or invalid 'initial_categories' field")
                    if not result['initial_categories']:
                        raise ValueError("initial_categories cannot be empty")
                    
                    return result
                    
                except (yaml.YAMLError, ValueError) as e:
                    if attempt == 0:
                        print(f"Invalid YAML format for {image_path}, retrying... (Error: {e})")
                        continue
                    else:
                        print(f"Failed to get valid YAML after retry for {image_path}: {e}")
                        print(f"DEBUG - Final failed response: {response_text}")
                        break
                        
            except Exception as e:
                if attempt == 0:
                    print(f"Request failed for {image_path}, retrying... (Error: {e})")
                    continue
                else:
                    print(f"Request failed after retry for {image_path}: {e}")
                    break
        
        # Both attempts failed - return failure marker
        return {
            "description": "failed",
            "initial_categories": ["Uncategorized"]
        }
    
    def _categorize_images(self, image_data_list: List[ImageData]):
        """Re-categorize non-failed images using filename-based matching with retry on format failure."""
        
        # Filter out failed images (those with description: "failed")
        valid_images = [img for img in image_data_list if img.description != "failed"]
        
        if not valid_images:
            print("No valid images to re-categorize")
            return
        
        print(f"Re-categorizing {len(valid_images)} valid images (skipping {len(image_data_list) - len(valid_images)} failed)")
        
        # Prepare filename-based data blocks for the prompt
        filename_blocks = []
        for img in valid_images:
            initial_categories_str = ", ".join(img.suggested_categories) if img.suggested_categories else "None"
            block = f"Filename: {img.filename}\nDescription: {img.description}\nInitial Categories: {initial_categories_str}"
            filename_blocks.append(block)
        
        filename_blocks_text = "\n\n".join(filename_blocks)
        
        categorization_prompt = textwrap.dedent(f"""
            Re-analyze these images using their descriptions AND suggested categories to create final categories:

            {filename_blocks_text}

            INSTRUCTIONS:
            1. Look at ALL the "Initial Categories" suggested across ALL images to identify common themes
            2. Re-examine each image's description in context of these category suggestions  
            3. Find common denominators and logical groupings among the suggested categories
            4. Create 5-10 consolidated final categories that capture the essence of the suggestions
            5. Each final category should ideally group 2+ related images when possible

            Rules:
            - Use the suggested categories as your primary guide, not just the descriptions
            - Look for patterns across ALL suggested categories (e.g., if you see "Software Documentation", "API Documentation", "Technical Documentation" → group as "Documentation")
            - Don't over-generalize or ignore the nuanced suggestions
            - Use folder-friendly names (1-3 words)
            - Re-read descriptions to ensure the final category fits the actual image content

            CRITICAL: Return ONLY the YAML list below. NO markdown, NO code blocks, NO ```yaml, NO explanations.
            Just the raw YAML list:
            
            - filename: exact_filename.jpg
              final_category: Category Name
            - filename: another_file.jpg
              final_category: Another Category
            
            Include ALL filenames in response.
        """).strip()
        
        # Try up to 2 times (original + 1 retry)
        for attempt in range(2):
            try:
                # Make request to Ollama for re-categorization
                response = self.client.chat(
                    model=self.model,
                    messages=[{
                        'role': 'user',
                        'content': categorization_prompt
                    }]
                )
                
                response_text = response['message']['content'].strip()
                
                if not response_text:
                    if attempt == 0:
                        print("Empty re-categorization response, retrying...")
                        continue
                    else:
                        break
                
                
                # Extract YAML from code blocks if needed
                yaml_content = response_text
                if '```yaml' in response_text or '```' in response_text:
                    lines = response_text.split('\n')
                    yaml_lines = []
                    inside_yaml_block = False
                    
                    for line in lines:
                        if line.strip().startswith('```yaml') or (line.strip() == '```' and not inside_yaml_block):
                            inside_yaml_block = True
                            continue
                        elif line.strip() == '```' and inside_yaml_block:
                            break
                        elif inside_yaml_block:
                            yaml_lines.append(line)
                    
                    if yaml_lines:
                        yaml_content = '\n'.join(yaml_lines)
                
                # Validate YAML format strictly
                try:
                    categories_list = yaml.safe_load(yaml_content)
                    
                    # Validate structure
                    if not isinstance(categories_list, list):
                        raise ValueError("Response is not a YAML list")
                    
                    if not categories_list:
                        raise ValueError("Response list is empty")
                    
                    # Validate each item
                    for i, item in enumerate(categories_list):
                        if not isinstance(item, dict):
                            raise ValueError(f"Item {i} is not a YAML object")
                        if 'filename' not in item or not isinstance(item['filename'], str):
                            raise ValueError(f"Item {i} missing or invalid 'filename' field")
                        if 'final_category' not in item or not isinstance(item['final_category'], str):
                            raise ValueError(f"Item {i} missing or invalid 'final_category' field")
                    
                    # Apply categories to images using filename matching
                    self._apply_categories_by_filename(image_data_list, categories_list)
                    return
                    
                except (yaml.YAMLError, ValueError) as e:
                    if attempt == 0:
                        print(f"Invalid YAML format in re-categorization, retrying... (Error: {e})")
                        continue
                    else:
                        print(f"Failed to get valid YAML after retry in re-categorization: {e}")
                        print(f"DEBUG - Final failed re-categorization response: {response_text}")
                        break
                        
            except Exception as e:
                if attempt == 0:
                    print(f"Re-categorization request failed, retrying... (Error: {e})")
                    continue
                else:
                    print(f"Re-categorization failed after retry: {e}")
                    break
        
        # Both attempts failed - apply fallback categorization to valid images only
        print("Re-categorization failed, applying fallback categorization to valid images")
        self._apply_fallback_categorization(valid_images)
    
    def _apply_categories_by_filename(self, image_data_list: List[ImageData], categories_list: List[Dict[str, str]]):
        """Apply final categories using filename matching."""
        # Create filename lookup from re-categorization response
        category_lookup = {item['filename']: item['final_category'] for item in categories_list if 'filename' in item and 'final_category' in item}
        
        # Update existing ImageData objects
        for image_data in image_data_list:
            if image_data.filename in category_lookup:
                final_category = category_lookup[image_data.filename].strip()
                image_data.primary_category = final_category
                # Keep all initial categories in the categories list
            else:
                # Fallback if filename not found in response
                print(f"Warning: No final category found for {image_data.filename}, keeping initial primary category")
                # Keep existing primary category if no final category was provided
    
    def _apply_fallback_categorization(self, image_data_list: List[ImageData]):
        """Apply fallback categorization based on description keywords and initial categories."""
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
        
        for image_data in image_data_list:
            # Only apply fallback if no primary category exists or it's Error
            if image_data.primary_category and image_data.primary_category != 'Error':
                continue
            
            # First, try to use initial suggested categories if available
            if image_data.suggested_categories and image_data.suggested_categories != ['Error']:
                image_data.primary_category = image_data.suggested_categories[0]
                continue
            
            # If no initial categories, use keyword-based fallback
            description = image_data.description.lower()
            best_category = "Uncategorized"
            max_matches = 0
            
            for category, keywords in category_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in description)
                if matches > max_matches:
                    max_matches = matches
                    best_category = category
            
            image_data.primary_category = best_category
            image_data.suggested_categories = [best_category]
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Ollama provider capabilities."""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            'max_image_size': 10 * 1024 * 1024,  # 10MB limit
            'concurrent_requests': 1,  # Ollama typically handles one request at a time
            'supports_batch_processing': True,
            'requires_internet': True
        })
        return base_capabilities