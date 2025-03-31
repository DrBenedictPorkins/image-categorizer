# AI Image Categorizer

An intelligent image categorization tool using AI for automatic image organization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Claude 3.7](https://img.shields.io/badge/Claude-3.7%20Sonnet-green)](https://www.anthropic.com/)
[![GPT-4o mini](https://img.shields.io/badge/GPT--4o-mini-orange)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This tool processes a directory of images and:
1. Generates detailed descriptions for each image using Salesforce BLIP model
2. Assigns concise category labels to each image using advanced LLMs (Anthropic Claude or OpenAI GPT)
3. Groups images into logical categories for organization
4. Creates an interactive HTML report for viewing and organizing images with drag-and-drop functionality

## System Requirements

- **Python**: 3.9.19+ or 3.13+ (recommended)
- **Hardware**: One of the following:
  - Apple Silicon Mac (M1/M2/M3) - optimal performance
  - NVIDIA GPU with CUDA support
  - Any modern CPU (will be slower)
- **API Keys**: At least one of:
  - Anthropic API key (recommended for Claude 3.7 Sonnet)
  - OpenAI API key (alternative for GPT-4o mini)
- **Internet connection** for API access

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/image-categorizer.git
   cd image-categorizer
   ```

2. **Install Python 3.9.19+ or 3.13+**:
   ```bash
   # Option A: Using pyenv (recommended)
   pyenv install 3.9.19  # or 3.13.0
   pyenv local 3.9.19    # or 3.13.0
   
   # Option B: Use your system Python if it's version 3.9.19+ or 3.13+
   python --version  # Verify it's 3.9.19+ or 3.13+
   ```

3. **Set up the environment**:
   ```bash
   # Install uv package manager
   pip install uv
   
   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate  # On Linux/macOS
   # OR
   .venv\Scripts\activate     # On Windows
   
   # Install dependencies
   uv pip install -e .
   ```

4. **Set up API key** (required for categorization):
   ```bash
   # Option A: Anthropic API (recommended for best results)
   export ANTHROPIC_API_KEY=your_api_key_here  # Linux/macOS
   # OR
   set ANTHROPIC_API_KEY=your_api_key_here     # Windows
   
   # Option B: OpenAI API (alternative)
   export OPENAI_API_KEY=your_api_key_here     # Linux/macOS
   # OR
   set OPENAI_API_KEY=your_api_key_here        # Windows
   ```
   
   > **Note**: At least one of these API keys must be provided for image categorization to work.

5. **Run the program**:
   ```bash
   python main.py /path/to/your/images
   ```
   
   After processing completes, an interactive HTML report will open in your default browser.

## Usage Options

### Basic Usage

Run the script with a directory containing images:

```bash
python main.py /path/to/your/images
```

The program will:
1. Process each image using BLIP to generate detailed descriptions
2. Create concise category labels using either:
   - Claude 3.7 Sonnet (if ANTHROPIC_API_KEY is set - preferred)
   - GPT-4o mini (if OPENAI_API_KEY is set)
3. Group images into logical categories
4. Generate an interactive HTML report that opens automatically
5. Show a summary of results with performance metrics

### Additional Options

```bash
# Process a specific directory of images
python main.py /path/to/your/images

# Reuse existing JSON data to regenerate HTML report (much faster)
python main.py /path/to/images --json /path/to/image_categories.json

# Use a specific existing group categorization
python main.py /path/to/images -j /path/to/group_categories.json

# Specify output directory for generated files
python main.py /path/to/images --output /path/to/output/dir

# Help and options
python main.py --help
```

### Output Files

After running, the program creates these files in your images directory (or specified output directory):
1. `image_categories.json`: Contains descriptions and categories for each image
2. `group_categories.json`: Contains the organized category structure
3. `image_categories.html`: The interactive web interface for organizing images

## Features

### Interactive HTML Report

An HTML report is automatically generated with:
- Fully interactive interface for viewing and organizing images
- Intuitive drag-and-drop functionality for moving images between categories
- Color-coded categories with consistent visual styling
- Images grouped by their assigned categories
- Dropdown selectors to quickly change image categories
- Enhanced fullscreen image previews with navigation and trash/restore controls
- Special "Deleted Items" category with restore functionality
- Complete descriptions and metadata for each image
- Summary statistics with visual category counts
- Option to generate a bash script for physically moving files into category directories
- Responsive design that works on various screen sizes

### Group Categories

Images are automatically analyzed as a collection to create logical directory categories:
- Uses advanced LLMs (primarily Anthropic Claude 3.7 Sonnet) to analyze all image descriptions holistically
- Creates logical, semantically meaningful categories suitable for directory organization (5-10 categories)
- Assigns each image to the most appropriate category
- Handles edge cases gracefully with special categorization rules
- Outputs a suggested directory structure
- Maintains category consistency across similar images
- Stores original category information to enable restore operations from "Deleted Items"

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)

## Model Configuration

The script uses the Salesforce BLIP image captioning model (large version) for image analysis and utilizes advanced language models for categorization:

1. **BLIP (image-to-text)**: Generates detailed descriptions of images
   - Uses MPS (Metal Performance Shaders) acceleration on Apple Silicon
   - Uses CUDA on NVIDIA GPUs if available
   - Falls back to CPU if needed

2. **Text Generation Models** (depending on available credentials):
   - **Anthropic Claude 3.7 Sonnet** (Primary option): 
     - Used automatically if ANTHROPIC_API_KEY is provided
     - Superior categorization with nuanced understanding of image contexts
     - Excellent at identifying meaningful, intuitive categories
     - Creates more consistent and logical groupings
     - Handles diverse image collections particularly well
     - Requires Anthropic API key
   
   - **OpenAI GPT-4o mini** (Fallback option):
     - Used when Anthropic API is unavailable or fails
     - Fast and efficient for category generation
     - Good quality categorization capabilities
     - Requires OpenAI API key

## Output Files

The script generates several output files in the image directory:

1. `image_categories.json`: JSON file with descriptions and categories for each image
2. `group_categories.json`: JSON file with grouped categories and assigned images
3. `image_categories.html`: Interactive HTML report for viewing and reorganizing images

## Customization

The HTML template (`template.html`) can be customized to change the appearance and functionality of the report.
