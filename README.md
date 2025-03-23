# Gemma3 Categorizer

An image categorization tool using AI for automatic image categorization and organization.

## Overview

This tool processes a directory of images and:
1. Generates detailed descriptions for each image using the BLIP model
2. Assigns concise category labels to each image
3. Groups images into logical categories for organization
4. Creates an interactive HTML report for viewing and organizing images

## Requirements

- Python 3.13+
- PyTorch 2.6+
- transformers 4.50+
- OpenAI API key for categorization
- An Apple Silicon Mac (M1/M2/M3) for optimal performance with MPS

## Setup

1. Clone this repository

2. Create and activate a virtual environment:
   ```bash
   uv venv create
   uv venv activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

4. Set up OpenAI API key (required):
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the script with a directory containing images:

```bash
python main.py /path/to/your/images
```

The script will:
1. Process each image using BLIP to generate detailed descriptions
2. Use OpenAI to create concise category labels
3. Group images into logical categories
4. Generate an interactive HTML report
5. Show a summary of results with performance metrics

## Features

### Interactive HTML Report

An HTML report is automatically generated with:
- Interactive interface for viewing and organizing images
- Images grouped by their assigned categories
- Dropdown selectors to move images between categories
- Fullscreen image previews when clicking on thumbnails
- Complete descriptions for each image
- Summary statistics and category counts
- Option to generate a bash script for physically moving files into category directories

### Group Categories

Images are automatically analyzed as a collection to create logical directory categories:
- Uses GPT-3.5-turbo to analyze all image descriptions holistically
- Creates logical categories suitable for directory names (2-5 categories)
- Assigns each image to the most appropriate category
- Outputs a suggested directory structure

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)

## Model Configuration

The script uses the Salesforce BLIP image captioning model (large version) and OpenAI's GPT models:

1. **BLIP (image-to-text)**: Generates detailed descriptions of images
   - Uses MPS (Metal Performance Shaders) acceleration on Apple Silicon
   - Falls back to CPU if needed for stability

2. **OpenAI GPT-3.5-turbo**: Converts descriptions to categories and groups images
   - Used for single-word category generation
   - Used for analyzing images as a group and creating logical categories

## Output Files

The script generates several output files in the image directory:

1. `image_categories.json`: JSON file with descriptions and categories for each image
2. `group_categories.json`: JSON file with grouped categories and assigned images
3. `image_categories.html`: Interactive HTML report for viewing and reorganizing images

## Customization

The HTML template (`template.html`) can be customized to change the appearance and functionality of the report.