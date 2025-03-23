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

# Define common image file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)

def get_category_from_description(description: str) -> str:
    """Use OpenAI API to get a concise category from a description."""
    try:
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful image categorizer. When given an image description, respond with the single most appropriate category label as a single word or very short phrase (1-3 words max). Focus on the primary subject."},
                {"role": "user", "content": f"Based on this image description, provide a single category label (1-3 words): '{description}'"}
            ],
            max_tokens=10,
            temperature=0.2  # Low temperature for more consistent responses
        )
        
        category = response.choices[0].message.content.strip()
        
        # Clean up quotes and periods that might be in the response
        category = category.strip('\'".,;:!?()[]{}')
        
        return category
    except Exception as e:
        print(f"Error getting category from LLM: {e}")
        # Extract first meaningful noun from description as fallback
        words = description.split()
        for word in words:
            if word.lower() not in ['a', 'an', 'the', 'is', 'are', 'that', 'this', 'of', 'in', 'on', 'with']:
                return word.capitalize()
        return "Unknown"

def get_group_categories(image_descriptions: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Analyze all image descriptions as a group and generate consistent categories.
    
    Args:
        image_descriptions: List of dicts with 'filename' and 'description' keys
        
    Returns:
        Dictionary mapping category names to lists of filenames
    """
    try:
        client = OpenAI()
        
        # Format descriptions for the prompt
        descriptions_text = "\n".join([f"Image {i+1} ({item['filename']}): {item['description']}" 
                                       for i, item in enumerate(image_descriptions)])
        
        # Always use OpenAI to generate categories, regardless of the number of images
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5 which is faster and cheaper
            messages=[
                {"role": "system", "content": """You are an expert image organizer and taxonomist. 
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
                - Return your answer in valid JSON format with category names as keys and lists of image filenames as values
                """},
                {"role": "user", "content": f"Here are descriptions of {len(image_descriptions)} images:\n\n{descriptions_text}\n\nCreate categories and assign each image to exactly one category. Return ONLY valid JSON with your categorization, no explanation text."}
            ],
            max_tokens=1000,
            temperature=0.3  # Low temperature for consistent, logical results
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content.strip()
        
        print(f"Raw response from API: {content}")
        
        # Find JSON block in the response
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_content = content[json_start:json_end]
            # Parse the JSON
            try:
                categories_dict = json.loads(json_content)
                print(f"Successfully parsed JSON: {categories_dict}")
                
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
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from response: {e}")
                print(f"JSON content: {json_content}")
                
                # Fallback: create categories from individual image categories
                print("Falling back to individual categories")
                categories_map = {}
                for item in image_descriptions:
                    category = item['category']
                    if category not in categories_map:
                        categories_map[category] = []
                    categories_map[category].append(item['filename'])
                
                return categories_map
        
        print("Could not extract valid JSON from the response, using fallback")
        # Fallback to individual categories
        categories_map = {}
        for item in image_descriptions:
            category = item['category']
            if category not in categories_map:
                categories_map[category] = []
            categories_map[category].append(item['filename'])
        
        return categories_map
        
    except Exception as e:
        print(f"Error getting group categories from LLM: {e}")
        return {}

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
        
        # Sort by category name
        category_stats.sort(key=lambda x: x[0])
        
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
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required for categorization.")
        print("Please set this environment variable and try again.")
        sys.exit(1)
    
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

    # Iterate over files in the specified directory
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
                    
                    # Check if BLIP returned a good description
                    if raw_description.lower().startswith("describe") or "unable to provide" in raw_description.lower() or "not enough information" in raw_description.lower() or len(raw_description.split()) < 10:
                        # Try a different approach if the first one fails
                        print(f"Initial BLIP description failed for {filename}, trying with adjusted parameters...")
                        fallback_output = pipe(
                            img,
                            max_new_tokens=300,
                            # Other parameters removed as they're not supported
                        )
                        description = fallback_output[0]['generated_text'].strip()
                        
                        # If still getting a non-descriptive response, examine file content directly
                        if description.lower().startswith("what is") or "unable to provide" in description.lower() or "not enough information" in description.lower():
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
                                # Try to describe based on image dimensions and characteristics
                                width, height = img.size
                                aspect = "landscape" if width > height else "portrait" if height > width else "square"
                                description = f"An {aspect} image with dimensions {width}x{height}."
                    else:
                        description = raw_description
                    
                    print(f"Description: {description}")
                    
                    # Get a category from the description using LLM
                    if os.environ.get("OPENAI_API_KEY"):
                        category = get_category_from_description(description)
                    else:
                        # Create a better default category
                        if "screenshot" in filename.lower() or "screen capture" in description.lower():
                            category = "Screenshot"
                        else:
                            # Extract first meaningful noun as fallback
                            words = description.split()
                            filtered_words = [
                                w for w in words 
                                if w.lower() not in ['a', 'an', 'the', 'is', 'are', 'that', 'this', 'of', 'in', 'on', 'with']
                            ]
                            
                            if filtered_words:
                                category = filtered_words[0]
                            else:
                                category = words[0] if words else "Unknown"
                            
                            # Clean up the category (remove punctuation, etc.)
                            category = category.strip('.,:;!?()[]{}""\'')
                        
                        # Make sure it's capitalized
                        category = category.capitalize()
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                description = "Error"
                category = "Error"

            results.append((filename, description, category))
            print(f"{filename}: {category}")

    # Print a summary of results with performance metrics.
    if results:
        print("\nFinal categorization results:")
        for fname, desc, cat in results:
            print(f"{fname}: {cat} (Description: {desc})")
        print(f"\nProcessed {len(results)} images in {time.time() - start_time:.2f} seconds")
        
        # Save individual results to a JSON file
        output_file = os.path.join(directory, "image_categories.json")
        results_data = [{"filename": r[0], "description": r[1], "category": r[2]} for r in results]
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {output_file}")
        
        # Always generate group categories since we require API key
        print("\nGenerating group categories for all images...")
        group_cats = get_group_categories(results_data)
        
        # Ensure we always have group categories - if none were generated, use individual categories
        if not group_cats:
            print("No group categories generated, using individual categories as fallback...")
            categories_map = {}
            for item in results_data:
                category = item['category']
                if category not in categories_map:
                    categories_map[category] = []
                categories_map[category].append(item['filename'])
            group_cats = categories_map
        
        # Output the group categories
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