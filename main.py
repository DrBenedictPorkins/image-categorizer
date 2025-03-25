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
    
    # Format descriptions for the prompt - use a numeric ID instead of filename to prevent bias
    descriptions_text = "\n".join([f"Image {i+1}: {item['description']}" 
                                 for i, item in enumerate(image_descriptions)])
    
    # Create a mapping from image number to filename for later use
    image_number_to_filename = {f"Image {i+1}": item['filename'] for i, item in enumerate(image_descriptions)}
    
    # Log what's being sent to the LLM for debugging
    print("\n=== DESCRIPTIONS SENT TO LLM FOR CATEGORIZATION ===")
    print(descriptions_text[:1000] + "..." if len(descriptions_text) > 1000 else descriptions_text)
    print("=== END OF DESCRIPTIONS SAMPLE ===\n")
    
    # System prompt requesting simple categorization based ONLY on descriptions
    system_prompt = """You are an AI expert in image categorization.
    Your task is to analyze image descriptions and assign a concise category to each image.
    
    Rules for categories:
    - Use 1-3 word categories that are specific but not overly detailed
    - CONSOLIDATE similar images into the same categories - be conservative with category creation
    - Aim for 5-8 total categories, not dozens of unique categories
    - Avoid using "Screenshots" as a category unless the content is clearly a computer interface
    - Focus on the SUBJECT MATTER in the image, not how it was taken (avoid categories like "Closeup")
    - Make categories broad enough to group multiple similar images
    - Prefer general categories that could each contain 4+ images
    - Use plain language folder-style names without fancy descriptions or punctuation
    
    CRITICALLY IMPORTANT:
    - Base your categorization SOLELY on the image descriptions
    - DO NOT use image numbers or IDs as a shortcut
    - DO NOT consider the image number in your decision making
    - DO NOT use generic terms like "Screenshot" just because they appear in descriptions
    - FOCUS on the CONTENT of the images, not the type/format of the image
    
    IMPORTANT: You MUST return ONLY a valid JSON object with no additional text. The JSON format must be:
    {
      "Image 1": "Category",
      "Image 2": "Category",
      ...
    }
    
    Where each key is the image number (e.g., "Image 1") and each value is the category label.
    """
    
    # Log the system prompt for debugging
    print("\n=== SYSTEM PROMPT FOR CATEGORIZATION ===")
    print(system_prompt)
    print("=== END OF SYSTEM PROMPT ===\n")
    
    user_prompt = f"""Here are descriptions of {len(image_descriptions)} images:

{descriptions_text}

Analyze ONLY these descriptions (ignore the image numbers) and assign a category to each image.

IMPORTANT: 
1. Base your decisions SOLELY on the image descriptions, not the image numbers
2. FOCUS on the CONTENT/SUBJECT of the images, not how they were captured
3. If the description mentions "screenshot" but describes other content, categorize based on that content
4. YOUR ENTIRE RESPONSE MUST BE ONLY VALID JSON with image numbers as keys and categories as values
5. The response must start with {{ and end with }}
6. Do not include any explanations, markdown formatting, or anything else
7. JUST RETURN THE JSON OBJECT, NOTHING ELSE."""

    # Log the user prompt for debugging (limiting the display of the long descriptions part)
    print("\n=== USER PROMPT FOR CATEGORIZATION (abbreviated) ===")
    abbreviated_user_prompt = user_prompt.replace(descriptions_text, "[...descriptions omitted for brevity...]")
    print(abbreviated_user_prompt)
    print("=== END OF USER PROMPT ===\n")
    
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
                max_tokens=2000,  # Increased token limit to handle larger responses
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
The following JSON is truncated:

{truncated_json}

Please provide a properly formatted and complete version of this JSON. The response should be valid JSON only.
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
                        retry_prompt = """You previously returned a response that was not valid JSON. 
                        
Your task is to create a valid JSON object that contains all image filenames as keys and their corresponding category labels as values.

IMPORTANT: Make sure you include ALL images from the original list, not just a subset!

YOUR ENTIRE RESPONSE MUST BE A VALID JSON OBJECT THAT STARTS WITH { AND ENDS WITH }.
DO NOT USE MARKDOWN FORMATTING. DO NOT ADD ANY EXPLANATIONS.
RETURN ONLY THE COMPLETE JSON OBJECT, NOTHING ELSE."""
                        
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
            # Fall through to Llama-2
    
    # Use Llama-2 model if OpenAI not available or failed
    if not categories_dict:
        try:
            print("Using Llama-2-7b for image categorization...")
            
            # Check for Hugging Face token and login if available
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if hf_token:
                print("Authenticating with Hugging Face...")
                hf_login(token=hf_token)
            else:
                print("No Hugging Face token found. If the model requires authentication, this will fail.")
                print("Set the HF_TOKEN or HUGGINGFACE_TOKEN environment variable to access gated models.")
                
            # Determine device for Llama model
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            
            # Create pipeline for text generation
            llama_pipe = pipeline(
                "text-generation",
                model="meta-llama/Llama-2-7b-chat-hf",
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32  # Use float16 for GPU to save memory
            )
            
            # Format prompt for Llama-2 chat model (requires specific formatting)
            llama_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""
            
            # Generate response with Llama
            response = llama_pipe(
                llama_prompt,
                max_new_tokens=1000,
                temperature=0.1,  # Very low temperature for consistent output
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2  # Add slight repetition penalty to avoid getting stuck
            )
            
            # Extract the generated text
            content = response[0]["generated_text"]
            # Extract just the response after the prompt for Llama-2
            # Look for content after the instruction closing tag [/INST]
            inst_end = content.find("[/INST]")
            if inst_end > 0:
                content = content[inst_end + 7:].strip()
            else:
                # Fallback in case the format changes
                content = content[len(llama_prompt):].strip()
            
            print(f"Raw response from Llama-2: {content}")
            
            try:
                # Direct JSON parsing with strict validation
                categories_dict = json.loads(content)
                print(f"Successfully parsed JSON response from Llama-2")
                # Log a few of the categories for debugging
                sample_categories = dict(list(categories_dict.items())[:5])
                print(f"Sample categories: {sample_categories}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from Llama-2: {e}")
                
                # Attempt to extract JSON from text if there's non-JSON content
                try:
                    # Look for JSON block within the text
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_content = content[json_start:json_end]
                        categories_dict = json.loads(json_content)
                        print(f"Successfully extracted and parsed JSON from Llama-2")
                    else:
                        print("No valid JSON found in Llama-2 response, retrying with a clearer prompt")
                        
                        # Create a more explicit prompt for retry
                        retry_prompt = """You previously returned a response that was not valid JSON. 
                        
Your task is simply to convert the image descriptions to categories.

YOUR ENTIRE RESPONSE MUST BE A VALID JSON OBJECT THAT STARTS WITH { AND ENDS WITH }.
DO NOT USE MARKDOWN FORMATTING. DO NOT ADD ANY EXPLANATIONS.
RETURN ONLY THE JSON, NOTHING ELSE."""
                        
                        retry_llama_prompt = f"{llama_prompt}\n\n{content}\n\n{retry_prompt}"
                        
                        retry_response = llama_pipe(
                            retry_llama_prompt,
                            max_new_tokens=1000,
                            temperature=0.05,  # Even lower temperature for the retry
                            do_sample=True,
                            top_p=0.9,
                            repetition_penalty=1.2
                        )
                        
                        # Extract the retry response
                        retry_content = retry_response[0]["generated_text"]
                        # Extract just the response after the prompt for Llama-2
                        # Look for content after the instruction closing tag [/INST]
                        inst_end = retry_content.find("[/INST]")
                        if inst_end > 0:
                            retry_content = retry_content[inst_end + 7:].strip()
                        else:
                            # Fallback in case the format changes
                            retry_content = retry_content[len(retry_llama_prompt):].strip()
                        
                        print(f"Raw response from Llama-2 retry: {retry_content}")
                        
                        try:
                            # Parse retry response
                            categories_dict = json.loads(retry_content)
                            print(f"Successfully parsed JSON from Llama-2 retry response")
                        except json.JSONDecodeError as retry_error:
                            # Try to extract JSON one more time
                            json_start = retry_content.find('{')
                            json_end = retry_content.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_content = retry_content[json_start:json_end]
                                try:
                                    categories_dict = json.loads(json_content)
                                    print(f"Successfully extracted and parsed JSON from Llama-2 retry response")
                                except:
                                    print("Failed to extract JSON from Llama-2 retry response")
                            else:
                                print("No valid JSON found in Llama-2 retry response")
                except Exception as e:
                    print(f"Failed to extract JSON from Llama-2: {e}")
        except Exception as e:
            print(f"Error getting response from Llama-2: {e}")
    
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
    else:
        # Use Llama-2 model if OpenAI not available
        try:
            print("Using Llama-2-7B for group categorization...")
            
            # Determine device for Llama-2 model
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            
            # Create pipeline for text generation
            llama_pipe = pipeline(
                "text-generation",
                model="mistralai/Llama-2-7B-v0.1",
                device=device
            )
            
            # Combine prompts for Llama-2
            llama_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = llama_pipe(
                llama_prompt,
                max_new_tokens=1000,
                temperature=0.3,
                do_sample=True
            )
            
            # Extract the generated text
            content = response[0]["generated_text"]
            # Extract just the response after the prompt for Llama-2
            # Look for content after the instruction closing tag [/INST]
            inst_end = content.find("[/INST]")
            if inst_end > 0:
                content = content[inst_end + 7:].strip()
            else:
                # Fallback in case the format changes
                content = content[len(llama_prompt):].strip()
            
            print(f"Raw response from Llama-2: {content}")
            
            try:
                # Try to parse the response directly as JSON
                categories_dict = json.loads(content)
                print(f"Successfully parsed JSON with {len(categories_dict)} categories")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from Llama-2: {e}")
                
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
                        print("No valid JSON found in Llama-2 response, retrying with a clearer prompt")
                        
                        # Create a more explicit prompt for retry
                        retry_prompt = """You previously returned a response that was not valid JSON. 
                        
Your task is simply to organize images into logical categories.

YOUR ENTIRE RESPONSE MUST BE A VALID JSON OBJECT THAT STARTS WITH { AND ENDS WITH }.
DO NOT USE MARKDOWN FORMATTING. DO NOT ADD ANY EXPLANATIONS.
RETURN ONLY THE JSON, NOTHING ELSE."""
                        
                        retry_llama_prompt = f"{llama_prompt}\n\n{content}\n\n{retry_prompt}"
                        
                        retry_response = llama_pipe(
                            retry_llama_prompt,
                            max_new_tokens=1000,
                            temperature=0.05,  # Even lower temperature for the retry
                            do_sample=True,
                            top_p=0.9,
                            repetition_penalty=1.2
                        )
                        
                        # Extract the retry response
                        retry_content = retry_response[0]["generated_text"]
                        # Extract just the response after the prompt for Llama-2
                        # Look for content after the instruction closing tag [/INST]
                        inst_end = retry_content.find("[/INST]")
                        if inst_end > 0:
                            retry_content = retry_content[inst_end + 7:].strip()
                        else:
                            # Fallback in case the format changes
                            retry_content = retry_content[len(retry_llama_prompt):].strip()
                        
                        print(f"Raw response from Llama-2 retry: {retry_content}")
                        
                        try:
                            # Parse retry response
                            categories_dict = json.loads(retry_content)
                            print(f"Successfully parsed JSON from Llama-2 retry response with {len(categories_dict)} categories")
                        except json.JSONDecodeError as retry_error:
                            # Try to extract JSON one more time
                            json_start = retry_content.find('{')
                            json_end = retry_content.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_content = retry_content[json_start:json_end]
                                try:
                                    categories_dict = json.loads(json_content)
                                    print(f"Successfully extracted and parsed JSON from Llama-2 retry response with {len(categories_dict)} categories")
                                except:
                                    print("Failed to extract JSON from Llama-2 retry response")
                                    content = None
                                    categories_dict = None
                            else:
                                print("No valid JSON found in Llama-2 retry response")
                                content = None
                                categories_dict = None
                except Exception as e:
                    print(f"Failed to extract JSON from Llama-2: {e}")
                    content = None
                    categories_dict = None
        except Exception as e:
            print(f"Error getting group categories from Llama-2: {e}")
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
                <h3>Suggested Categories</h3>
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
        print("OPENAI_API_KEY not found. Will use Llama-2-7B model for categorization")
        
    # Check for Hugging Face token and login if available (needed for Llama-2 model)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Authenticating with Hugging Face...")
        hf_login(token=hf_token)
    else:
        print("No Hugging Face token found. If using Llama-2 model requires authentication, this may fail.")
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