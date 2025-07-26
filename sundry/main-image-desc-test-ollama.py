import ollama
import base64
import webbrowser
import os
import glob
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants with environment variable fallbacks
VISUAL_MODEL = os.getenv('VISUAL_MODEL', 'qwen2.5vl:7b')
IMAGE_PATH = os.getenv('IMAGE_PATH', '/tmp/screenshots')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')


def get_image_description(client, image_path):
    """Get description for a single image from ollama"""
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        response = client.chat(
            model=VISUAL_MODEL,
            messages=[{
                'role': 'user',
                'content': 'Create a factual description for image categorization. Include: content type, visible elements, environment/setting, color information, readable text. Conclude with listing 2-5 factual category suggestions based only on what\'s visible. List categories as: Category 1, Category 2, Category 3 (no explanations or numbering). IMPORTANT!!!: Avoid assumptions about source, purpose, or context.',
                # 'content': 'Describe this image focusing on categorization elements. Include: visual content type, primary subjects, setting/environment, color scheme, and any text. End with "Potential categories:" and list 2-3 factual category suggestions based only on what\'s visible. Be factual, no assumptions about source or purpose.',
                'images': [image_data]
            }]
        )
        return image_data, response['message']['content']
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, f"Error processing image: {e}"

# Get directory path from command line argument or use default
if len(sys.argv) > 1:
    image_directory = sys.argv[1]
else:
    image_directory = IMAGE_PATH

print(f"Scanning directory: {image_directory}")

# Find all .jpg files in the directory
jpg_files = glob.glob(os.path.join(image_directory, "*.jpg"))
jpg_files.extend(glob.glob(os.path.join(image_directory, "*.JPG")))
jpg_files.extend(glob.glob(os.path.join(image_directory, "*.jpeg")))
jpg_files.extend(glob.glob(os.path.join(image_directory, "*.JPEG")))

if not jpg_files:
    print(f"No .jpg files found in {image_directory}")
    sys.exit(1)

jpg_files.sort()  # Sort files alphabetically
print(f"Found {len(jpg_files)} image files")

# Process each image
client = ollama.Client(host=OLLAMA_HOST)
image_data_list = []

for i, image_path in enumerate(jpg_files, 1):
    filename = os.path.basename(image_path)
    print(f"Processing {i}/{len(jpg_files)}: {filename}")
    
    image_data, description = get_image_description(client, image_path)
    
    if image_data:
        image_data_list.append({
            'filename': filename,
            'image_data': image_data,
            'description': description
        })
        print(f"✓ Completed: {filename}")
    else:
        print(f"✗ Failed: {filename}")

print(f"\nProcessed {len(image_data_list)} images successfully")

# Create image data JSON for JavaScript
images_json = json.dumps(image_data_list)

# Create HTML content with navigation
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Gallery with Descriptions</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }}
        .container {{
            max-width: 900px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 30px;
            margin: 20px;
            position: relative;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .counter {{
            color: #666;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .filename {{
            color: #2c3e50;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
        }}
        .image-container {{
            text-align: center;
            margin-bottom: 30px;
            position: relative;
        }}
        .image-container img {{
            max-width: 60%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        }}
        .description {{
            line-height: 1.6;
            color: #333;
            font-size: 16px;
            white-space: pre-wrap;
            text-align: left;
        }}
        .nav-button {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            border: none;
            padding: 15px 20px;
            font-size: 24px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s;
        }}
        .nav-button:hover {{
            background-color: rgba(0, 0, 0, 0.7);
        }}
        .nav-button:disabled {{
            background-color: rgba(0, 0, 0, 0.2);
            cursor: not-allowed;
        }}
        .prev {{
            left: -60px;
        }}
        .next {{
            right: -60px;
        }}
        .keyboard-hint {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="counter" id="counter">1 of {len(image_data_list)}</div>
            <div class="filename" id="filename">{image_data_list[0]['filename'] if image_data_list else ''}</div>
        </div>
        
        <div class="image-container">
            <button class="nav-button prev" id="prevBtn" onclick="showPrevious()">‹</button>
            <img id="currentImage" src="data:image/jpeg;base64,{image_data_list[0]['image_data'] if image_data_list else ''}" alt="Gallery Image">
            <button class="nav-button next" id="nextBtn" onclick="showNext()">›</button>
        </div>
        
        <div class="description" id="description">
            {image_data_list[0]['description'] if image_data_list else ''}
        </div>
        
        <div class="keyboard-hint">
            Use ← → arrow keys or click buttons to navigate
        </div>
    </div>

    <script>
        const images = {images_json};
        let currentIndex = 0;

        function updateDisplay() {{
            if (images.length === 0) return;
            
            const current = images[currentIndex];
            document.getElementById('counter').textContent = `${{currentIndex + 1}} of ${{images.length}}`;
            document.getElementById('filename').textContent = current.filename;
            document.getElementById('currentImage').src = `data:image/jpeg;base64,${{current.image_data}}`;
            document.getElementById('description').textContent = current.description;
            
            // Update button states
            document.getElementById('prevBtn').disabled = currentIndex === 0;
            document.getElementById('nextBtn').disabled = currentIndex === images.length - 1;
        }}

        function showNext() {{
            if (currentIndex < images.length - 1) {{
                currentIndex++;
                updateDisplay();
            }}
        }}

        function showPrevious() {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateDisplay();
            }}
        }}

        // Keyboard navigation
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'ArrowRight') {{
                showNext();
            }} else if (event.key === 'ArrowLeft') {{
                showPrevious();
            }}
        }});

        // Initialize display
        updateDisplay();
    </script>
</body>
</html>"""

# Write HTML to file
html_file = 'image_gallery.html'
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nHTML gallery created: {html_file}")
print("Opening in browser...")

# Open the HTML file in the default browser
webbrowser.open('file://' + os.path.abspath(html_file))