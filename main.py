import os
import sys
import time
import gc
import json
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Any

from PIL import Image
import torch
from transformers import pipeline
from psutil import virtual_memory
from openai import OpenAI
from huggingface_hub import login as hf_login

# Define common image file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)

def categorize_images_with_llm(image_descriptions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Categorize all images with a single LLM call.
    
    Args:
        image_descriptions: List of dicts with 'filename' and 'description' keys
        
    Returns:
        The same list with 'category' field added to each item
    """
    print(f"Categorizing {len(image_descriptions)} images with LLM...")
    
    # Format descriptions for the prompt
    descriptions_text = "\n".join([f"Image {i+1}: {item['filename']} - {item['description']}" 
                                 for i, item in enumerate(image_descriptions)])
    
    # System prompt requesting simple categorization
    system_prompt = """You are an AI expert in image categorization.
    Your task is to analyze image descriptions and assign a concise category to each image.
    
    Rules for categories:
    - Each category should be a single word or very short phrase (1-3 words max)
    - Focus on the primary subject of each image, based on the description
    - Avoid overly specific or overly general categories
    - Be consistent with similar images
    - Use commonly understood terms
    
    IMPORTANT: You MUST return ONLY a valid JSON object with no additional text. The JSON format must be:
    {
      "filename1.jpg": "category1",
      "filename2.jpg": "category2",
      ...
    }
    
    Where each key is the image filename and each value is the category label.
    """
    
    user_prompt = f"""Here are descriptions of {len(image_descriptions)} images:

{descriptions_text}

Analyze these descriptions and assign a category to each image.

IMPORTANT: YOUR ENTIRE RESPONSE MUST BE ONLY VALID JSON MAPPING FILENAMES TO CATEGORIES.
Do not include anything else besides the JSON."""
    
    categories_dict = {}
    
    # Check if OpenAI API key is available
    if os.environ.get("OPENAI_API_KEY"):
        try:
            print("Using OpenAI API for image categorization...")
            client = OpenAI()
            
            # Use OpenAI for categorization
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.1  # Very low temperature for consistent, structured JSON output
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            print(f"Raw response from OpenAI API: {content}")
            
            try:
                # Direct JSON parsing with strict validation
                categories_dict = json.loads(content)
                print(f"Successfully parsed JSON response from OpenAI")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from OpenAI: {e}")
                
                # Attempt to extract JSON from text if there's non-JSON content
                try:
                    # Look for JSON block within the text
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        categories_dict = json.loads(json_content)
                        print(f"Successfully extracted and parsed JSON from OpenAI")
                    else:
                        print("No valid JSON found in OpenAI response")
                except Exception as e:
                    print(f"Failed to extract JSON from OpenAI: {e}")
        except Exception as e:
            print(f"Error getting response from OpenAI: {e}")
            # Fall through to Mistral
    
    # Use Mistral model if OpenAI not available or failed
    if not categories_dict:
        try:
            print("Using Mistral-7B for image categorization...")
            
            # Check for Hugging Face token and login if available
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token:
                print("Authenticating with Hugging Face...")
                hf_login(token=hf_token)
            else:
                print("No Hugging Face token found. If the model requires authentication, this will fail.")
                print("Set the HF_TOKEN or HUGGINGFACE_TOKEN environment variable to access gated models.")
                
            # Determine device for Mistral model
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            
            # Create pipeline for text generation
            mistral_pipe = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-v0.1",
                device=device
            )
            
            # Combine prompts for Mistral
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = mistral_pipe(
                combined_prompt,
                max_new_tokens=1000,
                temperature=0.1,  # Very low temperature for consistent output
                do_sample=True
            )
            
            # Extract the generated text
            content = response[0]["generated_text"]
            # Extract just the response after the prompt
            content = content[len(combined_prompt):].strip()
            
            print(f"Raw response from Mistral: {content}")
            
            try:
                # Direct JSON parsing with strict validation
                categories_dict = json.loads(content)
                print(f"Successfully parsed JSON response from Mistral")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from Mistral: {e}")
                
                # Attempt to extract JSON from text if there's non-JSON content
                try:
                    # Look for JSON block within the text
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        categories_dict = json.loads(json_content)
                        print(f"Successfully extracted and parsed JSON from Mistral")
                    else:
                        print("No valid JSON found in Mistral response")
                except Exception as e:
                    print(f"Failed to extract JSON from Mistral: {e}")
        except Exception as e:
            print(f"Error getting response from Mistral: {e}")
    
    # Apply categories to the images from the categories_dict
    if categories_dict:
        for item in image_descriptions:
            filename = item['filename']
            if filename in categories_dict:
                category = categories_dict[filename]
                # Clean up the category
                if isinstance(category, str):
                    category = category.strip('\'".,;:!?()[]{}')
                    item['category'] = category.capitalize()
    
    # Apply fallback categories for any images without categories
    for item in image_descriptions:
        if 'category' not in item or not item['category']:
            # Create a better default category
            description = item['description']
            filename = item['filename']
            
            if "screenshot" in filename.lower() or "screen capture" in description.lower():
                item['category'] = "Screenshot"
            elif "unclear image content" in description.lower() or "unable to determine content" in description.lower():
                item['category'] = "Uncategorized"
            elif description.lower().startswith("an ") and "image with dimensions" in description.lower():
                item['category'] = "Uncategorized"
            else:
                # Extract first meaningful noun as fallback
                words = description.split()
                filtered_words = [
                    w for w in words 
                    if w.lower() not in ['a', 'an', 'the', 'is', 'are', 'that', 'this', 'of', 'in', 'on', 'with', 'unclear']
                ]
                
                if filtered_words:
                    category = filtered_words[0]
                else:
                    category = words[0] if words else "Uncategorized"
                
                # Clean up the category (remove punctuation, etc.)
                category = category.strip('.,:;!?()[]{}""\'')
                
                # Make sure it's capitalized
                item['category'] = category.capitalize()
    
    return image_descriptions

# This function has been replaced by the combined process_images_with_llm function
def _deprecated_get_group_categories(image_descriptions: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """DEPRECATED: This function has been replaced by process_images_with_llm.
    
    Args:
        image_descriptions: List of dicts with 'filename' and 'description' keys
        
    Returns:
        Dictionary mapping category names to lists of filenames
    """
    # Format descriptions for the prompt
    descriptions_text = "\n".join([f"Image {i+1} ({item['filename']}): {item['description']}" 
                                   for i, item in enumerate(image_descriptions)])
    
    # System prompt for categorization with strict JSON enforcement
    system_prompt = """You are an expert image organizer and taxonomist. 
    Your task is to analyze a set of image descriptions and create a logical organization system.
    
    Create a concise set of clear categories that would make sense as directory names 
    where these images could be organized. Then assign each image to exactly one category.
    
    Rules:
    - Try to use between 2-5 categories, even for small sets of images
    - Categories should be 1-3 words, using only letters, numbers, and underscores (no spaces)
    - Use commonly understood terms that would make sense as directory names
    - Group similar images together into logical categories
    - Avoid overly specific or overly general categories
    - Every image must be assigned to exactly one category
    
    IMPORTANT: You MUST return ONLY a valid JSON object with no additional text. The JSON format must be:
    {
      "category_name1": ["image1.jpg", "image2.jpg"],
      "category_name2": ["image3.jpg", "image4.jpg"],
      ...
    }
    
    Where each key is a category name and each value is an array of image filenames.
    Do not include explanations, comments, or any text outside the JSON structure.
    """
    
    user_prompt = f"""Here are descriptions of {len(image_descriptions)} images:

{descriptions_text}

Create categories and assign each image to exactly one category.

IMPORTANT: YOUR ENTIRE RESPONSE MUST BE ONLY VALID JSON IN THIS FORMAT:
{{
  "category_name1": ["image1.jpg", "image2.jpg"],
  "category_name2": ["image3.jpg", "image4.jpg"],
  ...
}}

Do not include anything else besides the JSON."""

    content = None
    categories_dict = None
    
    # Check if OpenAI API key is available
    if os.environ.get("OPENAI_API_KEY"):
        try:
            client = OpenAI()
            
            # Use OpenAI for categorization
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using GPT-3.5 which is faster and cheaper
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.1  # Very low temperature for consistent, structured JSON output
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            print(f"Raw response from OpenAI API: {content}")
            
            try:
                # Try to parse the response directly as JSON
                categories_dict = json.loads(content)
                print(f"Successfully parsed JSON with {len(categories_dict)} categories")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from OpenAI: {e}")
                content = None
                categories_dict = None
        except Exception as e:
            print(f"Error getting group categories from OpenAI: {e}")
            content = None
            categories_dict = None
    else:
        # Use Mistral model if OpenAI not available
        try:
            print("Using Mistral-7B for group categorization...")
            
            # Determine device for Mistral model
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            
            # Create pipeline for text generation
            mistral_pipe = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-v0.1",
                device=device
            )
            
            # Combine prompts for Mistral
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = mistral_pipe(
                combined_prompt,
                max_new_tokens=1000,
                temperature=0.3,
                do_sample=True
            )
            
            # Extract the generated text
            content = response[0]["generated_text"]
            # Extract just the response after the prompt
            content = content[len(combined_prompt):].strip()
            
            print(f"Raw response from Mistral: {content}")
            
            try:
                # Try to parse the response directly as JSON
                categories_dict = json.loads(content)
                print(f"Successfully parsed JSON with {len(categories_dict)} categories")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from Mistral: {e}")
                
                # Attempt to extract JSON from text if there's non-JSON content
                try:
                    # Look for JSON block within the text
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        categories_dict = json.loads(json_content)
                        print(f"Successfully extracted and parsed JSON with {len(categories_dict)} categories")
                    else:
                        print("No valid JSON found in Mistral response")
                        content = None
                        categories_dict = None
                except Exception as e:
                    print(f"Failed to extract JSON from Mistral: {e}")
                    content = None
                    categories_dict = None
        except Exception as e:
            print(f"Error getting group categories from Mistral: {e}")
            content = None
            categories_dict = None
    
    # If we didn't get valid JSON from OpenAI or Mistral API responses directly,
    # try one more extraction from content as a fallback
    if content and not categories_dict:
        try:
            # Look for JSON block within the text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                categories_dict = json.loads(json_content)
                print(f"Successfully extracted JSON in fallback: {categories_dict}")
        except Exception as e:
            print(f"Error in JSON fallback extraction: {e}")
            categories_dict = None
    
    # If we successfully got and parsed categories
    if categories_dict:
        # Handle the case where the model might have returned filenames directly
        # Check if this is already a mapping of categories to filenames
        first_value = list(categories_dict.values())[0] if categories_dict else []
        
        if first_value and isinstance(first_value, list) and isinstance(first_value[0], str) and not first_value[0].isdigit():
            # The model already returned filenames, no need to convert
            print("Model returned filenames directly, using as is")
            return categories_dict
        else:
            # Convert image numbers to filenames
            print("Converting image numbers to filenames")
            result = {}
            for category, image_ids in categories_dict.items():
                # Handle both string numbers and integer numbers
                filenames = []
                for num in image_ids:
                    if isinstance(num, str) and num.isdigit():
                        idx = int(num) - 1
                    elif isinstance(num, int):
                        idx = num - 1
                    else:
                        # Try to extract digits if in format "Image X"
                        if isinstance(num, str) and "image" in num.lower():
                            try:
                                idx = int(''.join(filter(str.isdigit, num))) - 1
                            except:
                                continue
                        else:
                            # If it's a string but not a number, check if it's directly a filename
                            matching_files = [item['filename'] for item in image_descriptions if item['filename'] == num]
                            if matching_files:
                                filenames.extend(matching_files)
                                continue
                            else:
                                continue
                    
                    if 0 <= idx < len(image_descriptions):
                        filenames.append(image_descriptions[idx]['filename'])
                
                if filenames:
                    result[category] = filenames
            
            return result
    
    # Fallback: create categories from individual image categories
    print("Using fallback categorization from individual categories")
    categories_map = {}
    for item in image_descriptions:
        category = item['category']
        if category not in categories_map:
            categories_map[category] = []
        categories_map[category].append(item['filename'])
    
    return categories_map

def generate_html_report(directory: str, results: List[Tuple[str, str, str]], 
                   thumbnail_size: Tuple[int, int] = (200, 200), 
                   group_categories: Dict[str, List[str]] = None) -> str:
    """Generate an interactive HTML report with drag-and-drop categorization using local file paths."""
    
    # Load the external template
    template_path = os.path.join(os.path.dirname(__file__), "template.html")
    
    if not os.path.exists(template_path):
        print(f"Warning: Template file not found at {template_path}. Using default template.")
        return ""
    
    # Get the current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Read the template
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    
    # Prepare image data for JavaScript
    card_data = []
    for filename, description, category in results:
        try:
            # Use direct file path instead of base64 encoding
            filepath = os.path.join(directory, filename)
            rel_filepath = filename  # Since HTML will be in the same directory
            
            # Find the category in group_categories
            assigned_category = category
            if group_categories:
                for cat, files in group_categories.items():
                    if filename in files:
                        assigned_category = cat
                        break
            
            # For individual thumbnails, don't need to include category in card data
            # as they are already grouped by category
            card_data.append({
                "filename": filename,
                "description": description,
                # Only include category for sorting/filtering purposes
                "category": assigned_category,
                "thumbnail": rel_filepath,  # Use file path instead of base64
                "preview": rel_filepath     # Same image for preview
            })
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            card_data.append({
                "filename": filename,
                "description": description,
                "category": "Error",  # Use a consistent error category
                "thumbnail": "",  # No thumbnail for error case
                "preview": "",    # No preview for error case
                "error": str(e)
            })
    
    # Prepare category data
    categories_data = []
    if group_categories:
        for category, files in group_categories.items():
            categories_data.append({
                "name": category,
                "count": len(files)
            })
    else:
        # Extract categories from results
        categories = {}
        for _, _, category in results:
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        for category, count in categories.items():
            categories_data.append({
                "name": category,
                "count": count
            })
    
    # Build the summary section with integrated category stats
    summary_html = f"""<div class="summary">
        <div class="summary-content">
            <div class="summary-main">
                <h2>Image Categories Summary</h2>
                <p>Directory: {directory}</p>
                <p>Total images processed: {len(results)}</p>
                <p>Generated on: {now}</p>
            </div>"""
    
    # Add compact directory structure info alongside summary
    if group_categories:
        category_stats = []
        for category, files in group_categories.items():
            category_stats.append((category, len(files)))
        
        # Sort by category name alphabetically
        category_stats.sort(key=lambda x: x[0].lower())
        
        summary_html += """
            <div class="directory-structure">
                <h3>Suggested Categories</h3>
                <div class="category-stats">"""
                
        for category, count in category_stats:
            summary_html += f"""
                    <div class="category-stat-item">
                        <span class="category-stat-name">{category}</span>
                        <span class="category-stat-count">{count}</span>
                    </div>"""
                
        summary_html += """
                </div>
            </div>"""
            
    summary_html += """
        </div>
    </div>"""
    
    # Replace template placeholders with actual content
    html_content = template_content.replace("{{timestamp}}", now)
    html_content = html_content.replace("{{summary}}", summary_html)
    html_content = html_content.replace("{{card_data}}", json.dumps(card_data))
    html_content = html_content.replace("{{categories_data}}", json.dumps(categories_data))
    
    # Write the HTML file
    output_file = os.path.join(directory, "image_categories.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return output_file

def process_images_with_llm(image_descriptions: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
    """Process all images with a single LLM call to get categories and directory structure.
    
    Args:
        image_descriptions: List of dicts with 'filename' and 'description' keys
        
    Returns:
        Tuple containing:
        - The same list with 'category' field added to each item
        - A dictionary mapping category names to lists of filenames (directory structure)
    """
    # First, categorize all images with the LLM
    categorized_data = categorize_images_with_llm(image_descriptions)
    
    # Extract categories for directory structure organization
    category_map = {}
    for item in categorized_data:
        category = item.get('category', 'Uncategorized')
        if category not in category_map:
            category_map[category] = []
        category_map[category].append(item['filename'])
    
    return categorized_data, category_map

def main(directory: str) -> None:
    start_time = time.time()
    # Initialize with maximum stability settings
    # Force garbage collection at the start
    gc.collect()
    
    # Try using MPS again since BLIP may be more stable
    try:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except:
        device = "cpu"
    
    # Set maximum threads for optimal CPU performance
    torch.set_num_threads(os.cpu_count())
        
    mem = virtual_memory()
    print(f"Using device: {device}")
    print(f"System memory: {mem.total / (1024**3):.1f}GB total, {mem.available / (1024**3):.1f}GB available")
    print(f"Processing capacity: {torch.get_num_threads()} threads, {os.cpu_count()} CPU cores")
    
    # Check if OpenAI API key is set
    if os.environ.get("OPENAI_API_KEY"):
        print("Using OpenAI API for categorization (OPENAI_API_KEY found)")
    else:
        print("OPENAI_API_KEY not found. Will use Mistral-7B model for categorization")
        
    # Check for Hugging Face token and login if available (needed for Mistral model)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Authenticating with Hugging Face...")
        hf_login(token=hf_token)
    else:
        print("No Hugging Face token found. If using Mistral model requires authentication, this may fail.")
        print("Set the HF_TOKEN or HUGGINGFACE_TOKEN environment variable if needed.")
    
    # Load the BLIP image captioning model
    print("Loading Salesforce BLIP image captioning model...")
    # Use a larger model for better descriptions
    pipe = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-large",  # Use larger model for more detail
        device=device,
        use_fast=True
    )

    results: List[Tuple[str, str, str]] = []  # Now stores filename, description, category

    # Iterate over files in the specified directory to get descriptions
    image_data = []  # Store image descriptions without categories initially
    
    for filename in sorted(os.listdir(directory)):
        if is_image_file(filename):
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath).convert("RGB")
            except Exception as e:
                print(f"Error opening {filename}: {e}")
                continue

            # Clear memory before processing each image
            gc.collect()
            
            try:
                # Process the image with BLIP
                with torch.no_grad():
                    # Generate a detailed description with BLIP
                    # Using only supported parameters
                    output = pipe(
                        img,
                        max_new_tokens=300,  # Allow for longer descriptions
                        # Other parameters removed as they're not supported
                    )
                    
                    # Get the description
                    raw_description = output[0]['generated_text'].strip()
                    
                    # Check if BLIP returned a good description - more lenient now
                    problem_phrases = ["describe", "unable to provide", "not enough information", "can't describe", 
                                       "cannot describe", "don't know what", "no description"]
                    
                    has_problem_phrase = any(phrase in raw_description.lower() for phrase in problem_phrases)
                    is_too_short = len(raw_description.split()) < 3  # Much more lenient - only reject extremely short descriptions
                    
                    if raw_description.lower().startswith("describe") or has_problem_phrase or is_too_short:
                        # Try a different approach if the first one fails
                        print(f"Initial BLIP description failed for {filename}, trying with adjusted parameters...")
                        fallback_output = pipe(
                            img,
                            max_new_tokens=300,
                            # Other parameters removed as they're not supported
                        )
                        description = fallback_output[0]['generated_text'].strip()
                        
                        # If still getting a non-descriptive response, examine file content directly
                        has_problem_phrase_fallback = any(phrase in description.lower() for phrase in problem_phrases)
                        if description.lower().startswith("what is") or has_problem_phrase_fallback:
                            # Extract information from filename as a last resort
                            if "screenshot" in filename.lower():
                                # It's a screenshot, try to extract date or context from filename
                                date_parts = [part for part in filename.split() if part.replace('-', '').replace(':', '').isdigit()]
                                if date_parts:
                                    date_str = ' '.join(date_parts)
                                    description = f"A screenshot taken on {date_str} showing computer interface or application window."
                                else:
                                    description = "A screenshot showing computer interface or application window."
                            else:
                                # Try to describe based on image dimensions and characteristics or use fallback category
                                # If we have any description at all, even if it's problematic, use it
                                # This allows the LLM to potentially still categorize based on partial info
                                if len(description.strip()) > 0 and not description.lower().startswith("what"):
                                    description = f"Unclear image content: {description}"
                                else:
                                    # No usable description at all, use image properties
                                    width, height = img.size
                                    aspect = "landscape" if width > height else "portrait" if height > width else "square"
                                    description = f"An {aspect} image with dimensions {width}x{height}. Unable to determine content."
                    else:
                        description = raw_description
                    
                    print(f"Description: {description}")
                    
                    # Only store filename and description at this point
                    image_data.append({"filename": filename, "description": description})
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                image_data.append({"filename": filename, "description": "Error"})

    # Print a summary of descriptions
    if image_data:
        print(f"\nGenerated descriptions for {len(image_data)} images in {time.time() - start_time:.2f} seconds")
        
        # Now process all images together to get both categories and directory structure with a single LLM call
        print("\nAnalyzing all images with LLM to assign categories and create directory structure...")
        categorized_data, directory_structure = process_images_with_llm(image_data)
        
        # Create results list in the format expected by downstream code
        results = [(item["filename"], item["description"], item["category"]) for item in categorized_data]
        
        print("\nFinal categorization results:")
        for fname, desc, cat in results:
            print(f"{fname}: {cat}")
        
        # Save individual results to a JSON file
        output_file = os.path.join(directory, "image_categories.json")
        results_data = [{"filename": r[0], "description": r[1], "category": r[2]} for r in results]
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {output_file}")
        
        # We got the directory structure directly from the LLM
        group_cats = directory_structure
        
        # Output the directory structure
        print("\nSuggested directory structure:")
        for category, files in group_cats.items():
            print(f"\n{category}/")
            for f in files:
                print(f"  - {f}")
        
        # Save group categories to a JSON file
        group_file = os.path.join(directory, "group_categories.json")
        with open(group_file, "w") as f:
            json.dump(group_cats, f, indent=2)
        print(f"\nGroup categories saved to {group_file}")
        
        # Always generate HTML report with group categories
        html_file = generate_html_report(directory, results, group_categories=group_cats)
        print(f"\nHTML report generated: {html_file}")
    else:
        print(f"\nNo image files found in {directory}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Image categorization tool using BLIP model")
    parser.add_argument("directory", help="Directory containing images to process")
    # Both HTML generation and group categories are now always on
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        sys.exit(1)
        
    main(args.directory)