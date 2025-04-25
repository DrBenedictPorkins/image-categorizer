import os
import sys
import time
import gc
import json
import argparse
import webbrowser
from datetime import datetime
from typing import List, Tuple, Dict, Any

from PIL import Image
import torch
from transformers import pipeline
from psutil import virtual_memory
from openai import OpenAI
from huggingface_hub import login as hf_login
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Define common image file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def categorize_images_with_llm(image_descriptions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Categorize all images with a single LLM call with improved categorization logic.

    Args:
        image_descriptions: List of dicts with 'filename' and 'description' keys

    Returns:
        The same list with 'category' field added to each item
    """
    print(f"Categorizing {len(image_descriptions)} images with LLM...")

    # Format descriptions for the prompt - use a numeric ID instead of filename to prevent bias
    descriptions_text = "\n".join([f"Image {i + 1}: {item['description']}"
                                   for i, item in enumerate(image_descriptions)])

    # Create a mapping from image number to filename for later use
    image_number_to_filename = {f"Image {i + 1}": item['filename'] for i, item in enumerate(image_descriptions)}

    # Log what's being sent to the LLM for debugging
    print("\n=== DESCRIPTIONS SENT TO LLM FOR CATEGORIZATION ===")
    print(descriptions_text[:1000] + "..." if len(descriptions_text) > 1000 else descriptions_text)
    print("=== END OF DESCRIPTIONS SAMPLE ===\n")

    # Improved system prompt with better categorization guidance
    system_prompt = """You are an expert image librarian and taxonomy specialist.
    Your task is to analyze image descriptions and create a coherent organization system with meaningful categories.

    CATEGORIZATION PROCESS:
    1. First, carefully analyze ALL image descriptions to understand the collection's scope
    2. Identify recurring themes, subjects, settings, and visual elements
    3. Create a balanced categorization system using these guidelines:
       - Create 5-10 distinct categories that form a coherent organization system
       - Use 1-3 word categories that are specific but not overly narrow
       - Prioritize CONTENT/SUBJECT categories over STYLE/FORMAT categories
       - Consider the intended use case (organizing a personal photo library)
       - Aim for categories that would each contain at least 3-5 images
    4. Assign each image to exactly ONE category that best represents its content

    CATEGORY GUIDELINES:
    - Use intuitive, folder-friendly category names (e.g., "Landscapes," "Food," "Architecture")
    - Be consistent in naming style and specificity level across categories
    - Avoid overlapping categories that would make classification ambiguous
    - Avoid using "Screenshots" as a category unless the content is clearly a computer interface
    - Use categories that describe the SUBJECT MATTER, not technical aspects
    - If an image could fit multiple categories, choose the most specific appropriate one

    OUTPUT FORMAT:
    You MUST return ONLY a valid JSON object with image numbers as keys and category names as values:
    {
      "Image 1": "Category",
      "Image 2": "Category",
      ...
    }

    DO NOT include explanations, markdown formatting, or any text outside the JSON structure.
    """

    # Log the system prompt for debugging
    print("\n=== SYSTEM PROMPT FOR CATEGORIZATION ===")
    print(system_prompt)
    print("=== END OF SYSTEM PROMPT ===\n")

    # More detailed user prompt with examples
    user_prompt = f"""Here are descriptions of {len(image_descriptions)} images to categorize:

{descriptions_text}

Create a meaningful organization system for these images based on their content.

IMPORTANT REQUIREMENTS:
1. Base your categorization SOLELY on the image descriptions
2. Focus on the CONTENT/SUBJECT, not technical aspects or quality
3. Create categories that would make sense as folder names in a photo library
4. Use a balanced approach - neither too general nor too specific
5. Use simple, clear category names (1-3 words)
6. Every image must be assigned to exactly one category

YOUR RESPONSE MUST BE ONLY A VALID JSON OBJECT:
{{
  "Image 1": "Category",
  "Image 2": "Category",
  ...
}}

NO EXPLANATIONS OR ADDITIONAL TEXT - JUST THE JSON OBJECT!"""

    # Log the user prompt for debugging (limiting the display of the long descriptions part)
    print("\n=== USER PROMPT FOR CATEGORIZATION (abbreviated) ===")
    abbreviated_user_prompt = user_prompt.replace(descriptions_text, "[...descriptions omitted for brevity...]")
    print(abbreviated_user_prompt)
    print("=== END OF USER PROMPT ===\n")

    categories_dict = {}

    # Check for API keys - either OpenAI or Anthropic must be available
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not has_openai and not has_anthropic:
        raise ValueError("Either OPENAI_API_KEY or ANTHROPIC_API_KEY must be provided. No API keys found.")

    # Determine which API to use (prefer Anthropic if available)
    use_anthropic = has_anthropic

    if use_anthropic:
        print("Using Anthropic API for image categorization...")
        try:
            # Import Anthropic library
            try:
                import anthropic
            except ImportError:
                print("Anthropic Python SDK not found. Installing...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
                import anthropic

            # Initialize Anthropic client
            client = anthropic.Anthropic()

            # Use Anthropic for categorization with Claude 3.7 Sonnet for best results
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                temperature=0.2,  # Slightly higher temperature for more thought diversity
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract JSON from response
            content = response.content[0].text

            print(f"Raw response from Anthropic API: {content}")

            try:
                # Direct JSON parsing with strict validation
                categories_dict = json.loads(content)
                print(f"Successfully parsed JSON response from Anthropic")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from Anthropic: {e}")

                # Attempt to extract JSON from text if there's non-JSON content
                try:
                    # Look for JSON block within the text
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1

                    if json_start >= 0 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        try:
                            # Try parsing the extracted JSON
                            categories_dict = json.loads(json_content)
                            print(f"Successfully extracted and parsed JSON from Anthropic")
                        except json.JSONDecodeError as extract_error:
                            print(f"Error extracting JSON from Anthropic: {extract_error}")

                            # Create explicit retry prompt
                            retry_prompt = """Your previous response was not in the required JSON format. 

I need a valid JSON object that maps each image number to a category name.

IMPORTANT:
1. Include ALL images from the original list
2. Make sure the JSON is properly formatted with quotes around keys and values
3. No explanations or comments - JUST THE JSON

Example of correct format:
{
  "Image 1": "Nature",
  "Image 2": "Architecture",
  ...
}

Please provide ONLY the JSON object:"""

                            # Request retry from Anthropic
                            retry_response = client.messages.create(
                                model="claude-3-opus-20240229",
                                max_tokens=2000,
                                temperature=0.1,
                                system=system_prompt,
                                messages=[
                                    {"role": "user", "content": user_prompt},
                                    {"role": "assistant", "content": content},
                                    {"role": "user", "content": retry_prompt}
                                ]
                            )

                            retry_content = retry_response.content[0].text
                            print(f"Raw response from Anthropic API retry: {retry_content}")

                            try:
                                # Parse retry response
                                categories_dict = json.loads(retry_content)
                                print(f"Successfully parsed JSON from Anthropic retry response")
                            except json.JSONDecodeError as retry_error:
                                # Try to extract JSON one more time
                                json_start = retry_content.find('{')
                                json_end = retry_content.rfind('}') + 1

                                if json_start >= 0 and json_end > json_start:
                                    json_content = retry_content[json_start:json_end]
                                    try:
                                        categories_dict = json.loads(json_content)
                                        print(f"Successfully extracted and parsed JSON from Anthropic retry response")
                                    except:
                                        print("Failed to extract JSON from Anthropic retry response")
                                        # Fall back to OpenAI if available
                                        if has_openai:
                                            print("Falling back to OpenAI API")
                                            use_anthropic = False
                                        else:
                                            raise ValueError(
                                                "Failed to categorize images with Anthropic API and no OpenAI fallback available")
                                else:
                                    print("No valid JSON found in Anthropic retry response")
                                    # Fall back to OpenAI if available
                                    if has_openai:
                                        print("Falling back to OpenAI API")
                                        use_anthropic = False
                                    else:
                                        raise ValueError(
                                            "Failed to categorize images with Anthropic API and no OpenAI fallback available")
                    else:
                        print("No valid JSON found in Anthropic response")
                        # Fall back to OpenAI if available
                        if has_openai:
                            print("Falling back to OpenAI API")
                            use_anthropic = False
                        else:
                            raise ValueError(
                                "Failed to categorize images with Anthropic API and no OpenAI fallback available")
                except Exception as e:
                    print(f"Failed to extract JSON from Anthropic: {e}")
                    # Fall back to OpenAI if available
                    if has_openai:
                        print("Falling back to OpenAI API")
                        use_anthropic = False
                    else:
                        raise ValueError(
                            "Failed to categorize images with Anthropic API and no OpenAI fallback available")
        except Exception as e:
            print(f"Error getting response from Anthropic: {e}")
            # Fall back to OpenAI if available
            if has_openai:
                print("Falling back to OpenAI API")
                use_anthropic = False
            else:
                raise ValueError(
                    f"Failed to categorize images with Anthropic API and no OpenAI fallback available. Error: {e}")

    # Try OpenAI if Anthropic wasn't used or failed
    if not use_anthropic or not categories_dict:
        if not has_openai:
            raise ValueError("Cannot use OpenAI API as fallback because OPENAI_API_KEY is not set")

        try:
            print("Using OpenAI API for image categorization...")
            client = OpenAI()

            # Use GPT-4o-mini or latest model for higher quality categorization without fallback
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use latest GPT model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.2
            )
            model_used = "gpt-4o-mini"

            # Extract JSON from response
            content = response.choices[0].message.content.strip()

            print(f"Raw response from OpenAI API ({model_used}): {content}")

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
                        try:
                            # Try parsing the extracted JSON
                            categories_dict = json.loads(json_content)
                            print(f"Successfully extracted and parsed JSON from OpenAI")
                        except json.JSONDecodeError as extract_error:
                            # If there's still an error, check if it's due to a truncated response
                            if "Unterminated string" in str(extract_error) or "Expecting" in str(extract_error):
                                print("Detected truncated JSON response, attempting to fix...")
                                # Try to auto-complete truncated JSON
                                if content.count('{') > content.count('}'):
                                    # Missing closing braces
                                    fixed_content = content + "}" * (content.count('{') - content.count('}'))
                                    try:
                                        categories_dict = json.loads(fixed_content)
                                        print("Successfully fixed and parsed truncated JSON")
                                    except:
                                        print("Could not fix truncated JSON automatically")
                                elif '"' in content and content.count('"') % 2 != 0:
                                    # Has unclosed quote
                                    if content.rstrip().endswith('"'):
                                        # Quote at the end, just add closing brace
                                        fixed_content = content + "}"
                                        try:
                                            categories_dict = json.loads(fixed_content)
                                            print("Successfully fixed and parsed truncated JSON with unclosed quotes")
                                        except:
                                            print("Could not fix truncated JSON with unclosed quotes")

                            # If auto-fixing failed, try to request a completion from the model
                            if not categories_dict:
                                print("Trying to get completion for truncated JSON...")

                                # Create a truncated JSON completion prompt
                                truncated_json = content
                                completion_prompt = f"""
The following JSON is truncated or malformed:

{truncated_json}

Please provide a properly formatted and complete version of this JSON. The response should be ONLY valid JSON.
"""

                                # Request completion
                                try:
                                    completion_response = client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "user", "content": completion_prompt}
                                        ],
                                        max_tokens=2000,
                                        temperature=0.1
                                    )

                                    completion_content = completion_response.choices[0].message.content.strip()

                                    # Try to extract valid JSON from the completion
                                    json_start = completion_content.find('{')
                                    json_end = completion_content.rfind('}') + 1

                                    if json_start >= 0 and json_end > json_start:
                                        json_content = completion_content[json_start:json_end]
                                        try:
                                            categories_dict = json.loads(json_content)
                                            print("Successfully completed and parsed truncated JSON")
                                        except:
                                            print("Could not parse JSON completion")
                                except Exception as completion_error:
                                    print(f"Error getting JSON completion: {completion_error}")
                    else:
                        print("No valid JSON found in OpenAI response, retrying with a clearer prompt")

                        # Create a more explicit prompt
                        retry_prompt = """Your previous response was not in the required JSON format.

I need you to create a valid JSON object that maps each image number to a category name.

REQUIREMENTS:
1. Return ONLY a properly formatted JSON object
2. Each key should be an image number (e.g., "Image 1")
3. Each value should be a category name (e.g., "Nature")
4. Include ALL images from the original list
5. NO markdown formatting, explanations, or additional text
6. The response must start with { and end with }

Example of required format:
{
  "Image 1": "Nature",
  "Image 2": "Architecture",
  ...
}"""

                        retry_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": content},
                                {"role": "user", "content": retry_prompt}
                            ],
                            max_tokens=2000,
                            temperature=0.1
                        )

                        retry_content = retry_response.choices[0].message.content.strip()
                        print(f"Raw response from OpenAI API retry: {retry_content}")

                        try:
                            # Parse retry response
                            categories_dict = json.loads(retry_content)
                            print(f"Successfully parsed JSON from retry response")
                        except json.JSONDecodeError as retry_error:
                            # Try to extract JSON one more time
                            json_start = retry_content.find('{')
                            json_end = retry_content.rfind('}') + 1

                            if json_start >= 0 and json_end > json_start:
                                json_content = retry_content[json_start:json_end]
                                try:
                                    categories_dict = json.loads(json_content)
                                    print(f"Successfully extracted and parsed JSON from retry response")
                                except:
                                    print("Failed to extract JSON from retry response")
                            else:
                                print("No valid JSON found in retry response")
                except Exception as e:
                    print(f"Failed to extract JSON from OpenAI: {e}")
        except Exception as e:
            print(f"Error getting response from OpenAI: {e}")
            raise ValueError(f"Failed to categorize images with OpenAI API. Error: {e}")

    # Apply categories to the images from the categories_dict
    if categories_dict:
        # For response format using image numbers, convert to filename
        if all(key.startswith("Image ") for key in list(categories_dict.keys())[:5]):
            # Convert from image numbers back to filenames
            filename_categories = {}
            for image_number, category in categories_dict.items():
                if image_number in image_number_to_filename:
                    filename = image_number_to_filename[image_number]
                    filename_categories[filename] = category
            categories_dict = filename_categories

        # Now apply categories to images
        for item in image_descriptions:
            filename = item['filename']
            if filename in categories_dict:
                category = categories_dict[filename]
                # Clean up the category
                if isinstance(category, str):
                    category = category.strip('\'".,;:!?()[]{}')
                    item['category'] = category.capitalize()

    # Apply improved fallback categories for any images without categories
    for item in image_descriptions:
        if 'category' not in item or not item['category']:
            # Create a better default category based on description content analysis
            description = item['description']
            filename = item['filename']

            # Define category keywords for common image types
            category_keywords = {
                "Screenshot": ["screenshot", "screen capture", "computer screen", "interface", "application window",
                               "browser"],
                "Food": ["food", "meal", "dish", "restaurant", "cooking", "baking", "dinner", "lunch", "breakfast"],
                "People": ["person", "people", "man", "woman", "child", "group", "family", "portrait", "face", "human"],
                "Animals": ["animal", "dog", "cat", "pet", "wildlife", "bird", "fish", "horse", "zoo"],
                "Nature": ["nature", "landscape", "mountain", "tree", "forest", "sky", "sunset", "beach", "outdoor",
                           "flower"],
                "Urban": ["building", "city", "street", "architecture", "urban", "downtown", "skyline"],
                "Art": ["art", "drawing", "painting", "illustration", "sketch", "design", "artwork"],
                "Document": ["document", "text", "paper", "letter", "form", "certificate", "page"],
                "Vehicle": ["car", "vehicle", "truck", "motorcycle", "bike", "transportation", "automobile"],
                "Product": ["product", "item", "object", "gadget", "device", "electronics"]
            }

            # Try to match keywords in the description
            best_category = None
            max_matches = 0

            for category, keywords in category_keywords.items():
                description_lower = description.lower()
                matches = sum(1 for keyword in keywords if keyword in description_lower)

                if matches > max_matches:
                    max_matches = matches
                    best_category = category

            # Fallback to custom logic if no keywords matched
            if not best_category or max_matches == 0:
                if "screenshot" in filename.lower():
                    best_category = "Screenshot"
                elif "unclear image content" in description.lower() or "unable to determine content" in description.lower():
                    best_category = "Uncategorized"
                elif description.lower().startswith("an ") and "image with dimensions" in description.lower():
                    best_category = "Uncategorized"
                else:
                    # Extract first meaningful noun as fallback
                    words = description.split()
                    filtered_words = [
                        w for w in words
                        if w.lower() not in ['a', 'an', 'the', 'is', 'are', 'that', 'this', 'of', 'in', 'on', 'with',
                                             'unclear']
                    ]

                    if filtered_words:
                        best_category = filtered_words[0].capitalize()
                    else:
                        best_category = "Uncategorized"

            # Apply the category
            item['category'] = best_category

    # Log the final category distribution
    category_counts = {}
    for item in image_descriptions:
        category = item.get('category', 'Uncategorized')
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1

    print("\nFinal category distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} images")

    return image_descriptions


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
    - Strictly use ONLY 3-5 categories total, even for large sets of images
    - Keep category names short (1-3 words max) and use conventional folder-style names
    - Use simple, intuitive terms like "Screenshots", "Diagrams", "Social Media", etc.
    - Group many similar images together - aim for at least 4+ images per category
    - Avoid creating specialized categories for just 1-2 images
    - Prioritize usability over specificity - folders should be easy to navigate
    - Every image must be assigned to exactly one category

    IMPORTANT: You MUST return ONLY a valid JSON object with no additional text. The JSON format must be:
    {
      "Category1": ["image1.jpg", "image2.jpg"],
      "Category2": ["image3.jpg", "image4.jpg"],
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

The response must start with {{ and end with }}.
DO NOT include ANY explanations, markdown formatting, or anything else.
JUST RETURN THE JSON OBJECT, NOTHING ELSE."""

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

                # Try to extract JSON from the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    try:
                        categories_dict = json.loads(json_content)
                        print(f"Successfully extracted and parsed JSON with {len(categories_dict)} categories")
                    except:
                        print("Failed to extract JSON")
                else:
                    print("No valid JSON found in OpenAI response, retrying with a clearer prompt")

                    # Create a more explicit prompt
                    retry_prompt = """You previously returned a response that was not valid JSON. 

Your task is simply to organize images into logical categories.

YOUR ENTIRE RESPONSE MUST BE A VALID JSON OBJECT THAT STARTS WITH { AND ENDS WITH }.
DO NOT USE MARKDOWN FORMATTING. DO NOT ADD ANY EXPLANATIONS.
RETURN ONLY THE JSON, NOTHING ELSE."""

                    retry_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": retry_prompt}
                        ],
                        max_tokens=1000,
                        temperature=0.1  # Very low temperature for consistent, structured JSON output
                    )

                    retry_content = retry_response.choices[0].message.content.strip()
                    print(f"Raw response from OpenAI API retry: {retry_content}")

                    try:
                        # Parse retry response
                        categories_dict = json.loads(retry_content)
                        print(f"Successfully parsed JSON from retry response with {len(categories_dict)} categories")
                    except json.JSONDecodeError as retry_error:
                        # Try to extract JSON one more time
                        json_start = retry_content.find('{')
                        json_end = retry_content.rfind('}') + 1

                        if json_start >= 0 and json_end > json_start:
                            json_content = retry_content[json_start:json_end]
                            try:
                                categories_dict = json.loads(json_content)
                                print(f"Successfully extracted and parsed JSON from retry response with {len(categories_dict)} categories")
                            except:
                                print("Failed to extract JSON from retry response")
                                content = None
                                categories_dict = None
                        else:
                            print("No valid JSON found in retry response")
                            content = None
                            categories_dict = None
        except Exception as e:
            print(f"Error getting group categories from OpenAI: {e}")
            content = None
            categories_dict = None

    # If we didn't get valid JSON from OpenAI or Llama-2 API responses directly,
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

    # Get the current date and time in a more human-readable format
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")

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
                <h2>AI Image Analysis Results</h2>
                <p><strong>Directory:</strong> {directory}</p>
                <p><strong>Images processed:</strong> {len(results)}</p>
                <p><strong>Generated on:</strong> {now}</p>
                <p class="summary-description">AI-powered image categorization using BLIP LLMs for image descriptions and Claude 3.7 Sonnet or Chat-4o mini for intelligent categorization. Images are analyzed and sorted into logical categories based on their content.</p>
            </div>"""

    # Add compact directory structure info alongside summary
    if group_categories:
        category_stats = []
        for category, files in group_categories.items():
            # Truncate category name if it's too long (for UI purposes)
            display_category = category
            if len(category) > 30:
                display_category = category[:27] + "..."

            category_stats.append((display_category, len(files)))

        # Ensure Trash category exists in the list
        if not any(category == 'Trash' for category, _ in category_stats):
            category_stats.append(('Trash', 0))

        # Sort by category name alphabetically, but keep Trash at the end
        category_stats.sort(key=lambda x: ('zzz' if x[0] == 'Trash' else x[0].lower()))

        summary_html += """
            <div class="directory-structure">
                <h3>AI-Generated Categories</h3>
                <div class="category-stats">"""

        for category, count in category_stats:
            # Add special class for Trash category
            extra_class = " trash-category-stat" if category == "Trash" else ""

            summary_html += f"""
                    <div class="category-stat-item{extra_class}">
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

def process_existing_json(directory: str, json_file_path: str) -> None:
    """Generate HTML report from existing JSON file without processing images.

    Args:
        directory: Path to the directory containing images
        json_file_path: Path to the JSON file with existing categories
    """
    print(f"Loading existing categories from {json_file_path}")

    try:
        with open(json_file_path, "r") as f:
            existing_data = json.load(f)

        if not existing_data:
            print("Error: JSON file is empty or invalid")
            return

        # Check format to determine which JSON file type it is
        if isinstance(existing_data, list) and len(existing_data) > 0 and isinstance(existing_data[0], dict) and "filename" in existing_data[0] and "category" in existing_data[0]:
            # This is an image_categories.json file
            print("Detected image_categories.json format")
            results = [(item["filename"], item.get("description", ""), item["category"]) for item in existing_data]

            # Create directory structure from individual results
            group_cats = {}
            for item in existing_data:
                category = item["category"]
                if category not in group_cats:
                    group_cats[category] = []
                group_cats[category].append(item["filename"])

        elif isinstance(existing_data, dict):
            # This is a group_categories.json file
            print("Detected group_categories.json format")
            group_cats = existing_data

            # Create results list from group categories
            results = []
            for category, filenames in group_cats.items():
                for filename in filenames:
                    # No description available from group_categories.json
                    results.append((filename, "", category))
        else:
            print("Error: Unrecognized JSON format")
            return

        print(f"Loaded {len(results)} images across {len(group_cats)} categories")

        # Generate HTML report
        html_file = generate_html_report(directory, results, group_categories=group_cats)
        print(f"\nHTML report generated: {html_file}")

        # Open the HTML file in the default browser
        print("Opening HTML report in default browser...")
        webbrowser.open(f"file://{os.path.abspath(html_file)}")

    except Exception as e:
        print(f"Error processing JSON file: {e}")

def main(directory: str, json_file: str = None, blip_model: str = "Salesforce/blip2-flan-t5-xl-coco") -> None:
    """Main function to process images or use existing JSON data.

    Args:
        directory: Path to the directory containing images
        json_file: Optional path to a JSON file with existing categorization data
        blip_model: HuggingFace model ID for BLIP model (default: Salesforce/blip2-flan-t5-xl-coco)
    """
    # If json_file is provided, just generate HTML report from it
    if json_file:
        if not os.path.exists(json_file):
            print(f"Error: JSON file '{json_file}' not found")
            return
        process_existing_json(directory, json_file)
        return

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
        print("OPENAI_API_KEY not found. Will use Llama-2-7B model for categorization")

    # Check for Hugging Face token and login if available (needed for Llama-2 model)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Authenticating with Hugging Face...")
        hf_login(token=hf_token)
    else:
        print("No Hugging Face token found. Using Hugging Face LLMs requires authentication, this may fail.")
        print("Set the HF_TOKEN or HUGGINGFACE_TOKEN environment variable if needed.")

    # Load the prompt from file - useful for understanding what aspects to focus on
    prompt_path = os.path.join(os.path.dirname(__file__), "blip_prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            blip_prompt = f.read().strip()
        print(f"Using custom BLIP prompt from {prompt_path}")
    else:
        blip_prompt = "Provide a detailed description of this image."
        print(f"Warning: BLIP prompt file not found at {prompt_path}. Using default prompt.")

    # Load the BLIP-2 image captioning model with prompt support
    print(f"Loading BLIP-2 model '{blip_model}' with prompt support...")
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        print("Successfully imported BLIP-2 classes")

        print(f"Loading BLIP-2 model and processor with prompt support...")
        processor = Blip2Processor.from_pretrained(blip_model)
        model = Blip2ForConditionalGeneration.from_pretrained(blip_model).to(device)

        # This implementation supports prompting
        print(f"Using prompt: '{blip_prompt}'")
    except Exception as e:
        print(f"Error loading BLIP model with prompt support: {e}")
        print("Falling back to standard pipeline...")
        # Use the standard pipeline as fallback
        pipe = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-large",
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
                    # Check if we're using the processor+model approach or the pipeline
                    if 'processor' in locals() and 'model' in locals():
                        # Define problem phrases here, before conditional branches
                        problem_phrases = ["describe", "unable to provide", "not enough information", "can't describe",
                                           "cannot describe", "don't know what", "no description"]

                        # Use the BLIP-2 model with custom prompt
                        print(f"Processing {filename} with custom prompt...")
                        # Fixed code
                        inputs = processor(images=img, text=blip_prompt, return_tensors="pt").to(device)
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=200,  # Increase from 100 to 200
                            do_sample=True,
                            temperature=0.8,  # Slightly higher for more detailed descriptions
                            num_beams=5,  # Keep beam search for better quality
                            top_p=0.95,  # Slightly higher top_p allows more diversity
                            repetition_penalty=1.2,  # Discourage repetition
                            length_penalty=1.5,  # Encourage longer outputs (>1.0 favors longer sequences)
                            no_repeat_ngram_size=3  # Prevent repeating the same phrases
                        )

                        # Extract only the generated text (not including the prompt)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                        # Clean up the output - remove the prompt if it appears at the beginning
                        if generated_text.startswith(blip_prompt):
                            raw_description = generated_text[len(blip_prompt):].strip()
                        else:
                            raw_description = generated_text.strip()

                        print(f"DEBUG - Raw BLIP output: '{raw_description}'")

                        # Simple length-based validation
                        is_too_short = len(raw_description.split()) < 3
                        has_problem_phrase = any(phrase in raw_description.lower() for phrase in problem_phrases)
                        should_fallback = is_too_short or has_problem_phrase or raw_description.lower() == blip_prompt.lower()
                    else:
                        # Fallback to pipeline without prompt
                        print(f"Processing {filename} with standard pipeline...")
                        output = pipe(
                            img,
                            max_new_tokens=300,  # Allow for longer descriptions
                            # Other parameters removed as they're not supported
                        )
                        # Get the description
                        raw_description = output[0]['generated_text'].strip()

                    # Use different checks based on whether we're using BLIP-2 or pipeline
                    if 'processor' in locals() and 'model' in locals():
                        # For BLIP-2, be more forgiving because it might include parts of the prompt
                        # Only check if the description is too short
                        is_too_short = len(raw_description.split()) < 3
                        if is_too_short:
                            print(f"DEBUG - Description too short: {len(raw_description.split())} words")
                            # Only fall back for too-short descriptions
                            should_fallback = is_too_short
                        else:
                            # If we have a reasonably long description from BLIP-2, accept it
                            should_fallback = False
                    else:
                        # For standard pipeline, use the original problem detection
                        problem_phrases = ["describe", "unable to provide", "not enough information", "can't describe", 
                                        "cannot describe", "don't know what", "no description"]

                        has_problem_phrase = any(phrase in raw_description.lower() for phrase in problem_phrases)
                        if has_problem_phrase:
                            problem_phrase_found = next((phrase for phrase in problem_phrases if phrase in raw_description.lower()), None)
                            print(f"DEBUG - Problem phrase found: '{problem_phrase_found}'")

                        is_too_short = len(raw_description.split()) < 3
                        if is_too_short:
                            print(f"DEBUG - Description too short: {len(raw_description.split())} words")

                        if raw_description.lower().startswith("describe"):
                            print(f"DEBUG - Description starts with 'describe'")

                        should_fallback = raw_description.lower().startswith("describe") or has_problem_phrase or is_too_short

                    # Use the should_fallback flag to determine if we need to try again
                    if should_fallback:
                        # Try a different approach if the first one fails
                        print(f"Initial BLIP description failed for {filename}, trying with a different prompt...")

                        # Use a more direct prompt for the fallback
                        fallback_prompt = "Describe this image in detail, focusing on the main subjects and background."

                        if 'processor' in locals() and 'model' in locals():
                            # Use the processor with a different prompt
                            fallback_inputs = processor(img, fallback_prompt, return_tensors="pt").to(device)
                            # Use max_new_tokens instead of max_length to avoid prompt length issues
                            fallback_output = model.generate(**fallback_inputs, max_new_tokens=100)
                            description = processor.decode(fallback_output[0], skip_special_tokens=True).strip()
                            print(f"DEBUG - Fallback description: '{description}'")
                        else:
                            # Fallback to standard pipeline
                            fallback_output = pipe(
                                img,
                                max_new_tokens=300,
                                # Other parameters removed as they're not supported
                            )
                            description = fallback_output[0]['generated_text'].strip()

                        # If still getting a non-descriptive response, examine file content directly
                        has_problem_phrase_fallback = any(phrase in description.lower() for phrase in problem_phrases)
                        if has_problem_phrase_fallback:
                            problem_phrase_found = next((phrase for phrase in problem_phrases if phrase in description.lower()), None)
                            print(f"DEBUG - Fallback problem phrase found: '{problem_phrase_found}'")

                        if description.lower().startswith("what is"):
                            print(f"DEBUG - Fallback description starts with 'what is'")

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

        # Open the HTML file in the default browser
        print("Opening HTML report in default browser...")
        webbrowser.open(f"file://{os.path.abspath(html_file)}")
    else:
        print(f"\nNo image files found in {directory}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Image categorization tool using BLIP model")
    parser.add_argument("directory", help="Directory containing images to process")
    parser.add_argument("--json", "-j", nargs="?", const=True, help="Path to existing JSON file to generate HTML report without processing images. If no path provided, defaults to group_categories.json in the image directory")
    parser.add_argument("--save-json", "-s", action="store_true", help="Save categorization data to JSON files (always enabled)")
    parser.add_argument("--blip-model", default="Salesforce/blip2-flan-t5-xl-coco", help="HuggingFace model ID for BLIP model for image captioning (default: Salesforce/blip2-flan-t5-xl-coco)")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        sys.exit(1)

    # Handle the --json parameter
    json_file = None
    if args.json:
        # If --json is provided without a value, use group_categories.json in the image directory
        if args.json is True:
            default_json = os.path.join(args.directory, "group_categories.json")
            if os.path.exists(default_json):
                json_file = default_json
                print(f"Using default group_categories.json in the specified directory")
            else:
                print(f"Default JSON file {default_json} not found")
                sys.exit(1)
        # If a specific JSON file path is provided, verify it exists
        else:
            if not os.path.exists(args.json):
                print(f"Error: JSON file {args.json} not found")
                sys.exit(1)
            json_file = args.json

    main(args.directory, json_file, args.blip_model)
