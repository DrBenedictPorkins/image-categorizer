# AI Image Categorizer

An intelligent image categorization tool using AI for automatic image organization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Claude 3.7](https://img.shields.io/badge/Claude-3.7%20Sonnet-green)](https://www.anthropic.com/)
[![GPT-4o mini](https://img.shields.io/badge/GPT--4o-mini-orange)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## TL;DR - Quick Start

This tool uses:
- Local Salesforce BLIP model (downloaded automatically) to analyze and describe images
- Remote LLM (ANTHROPIC or OPENAI) for categorization - you must provide one API key

First, install uv package manager:
* Get uv from https://github.com/astral-sh/uv/releases or via pip: `pip install uv`

```bash
# Clone repo
git clone https://github.com/DrBenedictPorkins/image-categorizer.git
```

```bash
# Navigate to directory
cd image-categorizer
```

```bash
# Create virtual environment
uv venv
```

For macOS/Linux:
```bash
# Activate virtual environment (macOS/Linux)
source .venv/bin/activate
```

For Windows:
```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate
```

```bash
# Install dependencies
uv pip install -e .
```

To run with Anthropic API (recommended):
```bash
# Set API key and run
export ANTHROPIC_API_KEY=your_key_here
python main.py /path/to/your/images
```

OR one-line version:
```bash
ANTHROPIC_API_KEY=your_key_here python main.py /path/to/your/images
```

Alternatively, to run with OpenAI API:
```bash
# Set API key and run
export OPENAI_API_KEY=your_key_here
python main.py /path/to/your/images
```

OR one-line version:
```bash
OPENAI_API_KEY=your_key_here python main.py /path/to/your/images
```

That's it! An HTML report (`image_categories.html`) will be generated in your image directory.

## Overview

This tool processes a directory of images and:
1. Generates detailed descriptions for each image using Salesforce BLIP model (Local LLM)
2. Assigns concise category labels to each image using advanced AI (ANTHROPIC Claude 3.7 Sonnet preferred, or OPENAI as an alternative option)
3. Groups images into logical categories for organization
4. Creates an interactive HTML report for viewing and organizing images with drag-and-drop functionality

## System Requirements

- **Python**: 3.9.19+ or 3.13+ (recommended)
- **Hardware**: One of the following:
  - Apple Silicon Mac (M1/M2/M3) - optimal performance
  - NVIDIA GPU with CUDA support
  - Any modern CPU (will be slower)
- **LLM Requirements**:
  - Local: Salesforce BLIP for image descriptions (included in package)
  - Remote: **REQUIRED** - You must choose one of:
    - ANTHROPIC API key (recommended for Claude 3.7 Sonnet - preferred option)
    - OPENAI API key (alternative option for GPT-4o mini)
- **Internet connection** for API access

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/makram/image-categorizer.git
   cd image-categorizer
   ```

2. **Verify Python requirements**:
   ```bash
   # Ensure you have Python 3.9.19+ or 3.13+
   python --version
   ```

3. **Set up with uv**:
   ```bash
   # Get uv from https://github.com/astral-sh/uv/releases
   # Or install via pip
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
   # Process images in your Photos folder
   python main.py ~/Pictures/MyPhotos
   
   # Or a specific folder of images you want to categorize
   python main.py /path/to/your/images
   ```
   
   After processing completes, an interactive HTML report (`image_categories.html`) will open in your default browser.

## Usage Options

### Basic Usage

Run the script with a directory containing images:

```bash
python main.py /path/to/your/images
```

The program will:
1. Process each image using BLIP (Local LLM from Salesforce) to generate detailed descriptions
2. Create concise category labels using your chosen remote LLM:
   - ANTHROPIC Claude 3.7 Sonnet (if ANTHROPIC_API_KEY is set - preferred)
   - OPENAI GPT-4o mini (if OPENAI_API_KEY is set - alternative option)
3. Group images into logical categories
4. Generate an interactive HTML report that opens automatically
5. Show a summary of results with performance metrics

### Additional Options

```bash
# Process a specific directory of images
python main.py ~/Pictures/Vacation2023

# Reuse existing JSON data to regenerate HTML report (much faster)
python main.py ~/Pictures/Vacation2023 --json ~/Pictures/Vacation2023/image_categories.json

# Use a specific existing group categorization
python main.py ~/Pictures/Vacation2023 -j ~/Pictures/Vacation2023/group_categories.json

# Just create a symlink to view results elsewhere if needed
ln -s ~/Pictures/Vacation2023/image_categories.html ~/Documents/vacation_report.html

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
- Uses remote LLM (ANTHROPIC Claude 3.7 Sonnet preferred) to analyze all image descriptions holistically
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

The tool uses two types of AI models to process and categorize your images:

1. **Local LLM: BLIP (image-to-text)**: 
   - Developed by Salesforce Research
   - Runs locally on your machine (no API key required)
   - Generates detailed descriptions of image content
   - Uses Metal Performance Shaders (MPS) acceleration on Apple Silicon
   - Uses CUDA on NVIDIA GPUs if available
   - Falls back to CPU if needed

2. **Remote LLM (REQUIRED for categorization)**: 
   
   **Option 1 (Preferred): ANTHROPIC Claude 3.7 Sonnet**
   - State-of-the-art multimodal AI model
   - Performs exceptional image categorization
   - Creates intuitive, semantically meaningful categories
   - Handles diverse image collections with deep understanding
   - Understands nuanced image contexts and relationships
   - Requires Anthropic API key
   
   **Option 2: OPENAI GPT-4o mini**
   - Alternative choice for image categorization
   - Provides good quality categorization capabilities
   - Requires OpenAI API key
   
   > **Note:** You MUST configure ONE of these remote LLMs via API key for the categorization functionality to work properly.


## Customization

The HTML template (`template.html`) can be customized to change the appearance and functionality of the report.
