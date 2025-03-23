# Image Categorizer

An image categorization tool using AI for automatic image categorization and organization.

## Overview

This tool processes a directory of images and:
1. Generates detailed descriptions for each image using the BLIP model
2. Assigns concise category labels to each image using either OpenAI API or Mistral-7B
3. Groups images into logical categories for organization
4. Creates an interactive HTML report for viewing and organizing images

## Requirements

- Python 3.13+
- PyTorch 2.6+
- transformers 4.50+
- For OpenAI: OpenAI API key (optional, will use Mistral-7B if not provided)
- For Mistral: Hugging Face login credentials (if using Mistral model)
- GPU or Apple Silicon Mac (M1/M2/M3) for optimal performance

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

4. Authentication options:

   a. OpenAI API (optional):
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

   b. Hugging Face login (for Mistral model access):
   
   Option 1: Use environment variable (recommended):
   ```bash
   export HF_TOKEN=your_token_here
   # or
   export HUGGINGFACE_TOKEN=your_token_here
   ```
   
   Option 2: Login via CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
   You'll be prompted for your Hugging Face token, which you can get from your [Hugging Face account settings](https://huggingface.co/settings/tokens)

## Usage

Run the script with a directory containing images:

```bash
python main.py /path/to/your/images
```

The script will:
1. Process each image using BLIP to generate detailed descriptions
2. Create concise category labels using either:
   - OpenAI GPT-3.5 (if OPENAI_API_KEY is set)
   - Mistral-7B (if no OpenAI key is available)
3. Group images into logical categories
4. Generate an interactive HTML report
5. Show a summary of results with performance metrics

Note: The first run using Mistral-7B will download the model (around 4GB), which may take some time depending on your internet connection.

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
- Uses either GPT-3.5-turbo or Mistral-7B to analyze all image descriptions holistically
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

The script uses the Salesforce BLIP image captioning model (large version) and either OpenAI GPT-3.5 or Hugging Face's Mistral-7B models:

1. **BLIP (image-to-text)**: Generates detailed descriptions of images
   - Uses MPS (Metal Performance Shaders) acceleration on Apple Silicon
   - Uses CUDA on NVIDIA GPUs if available
   - Falls back to CPU if needed

2. **Text Generation Models** (depending on available credentials):
   - **OpenAI GPT-3.5-turbo**: Used if OPENAI_API_KEY is provided
     - Fast and efficient for category generation
     - Requires API key and internet connection
   
   - **Mistral-7B**: Used when OpenAI API key is not available
     - Runs locally after initial download
     - Requires Hugging Face login for model access
     - More resource-intensive than OpenAI API
     - Provides complete privacy as processing is done locally

## Output Files

The script generates several output files in the image directory:

1. `image_categories.json`: JSON file with descriptions and categories for each image
2. `group_categories.json`: JSON file with grouped categories and assigned images
3. `image_categories.html`: Interactive HTML report for viewing and reorganizing images

## Customization

The HTML template (`template.html`) can be customized to change the appearance and functionality of the report.
