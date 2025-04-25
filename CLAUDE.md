# Image Categorizer - Guidelines for Claude

## Project Overview
- Image categorization tool using Salesforce BLIP (local) for image descriptions and either Claude 3.7 or GPT-4o mini (remote) for categorization
- Takes directory of images, creates detailed descriptions, and organizes them into semantic categories
- Generates an interactive HTML report with drag-and-drop functionality

## Environment & Dependencies
- Python 3.9.19+
- Key packages: transformers, torch, pillow, httpx, openai, anthropic
- Install: `uv pip install -e .`
- Requires either ANTHROPIC_API_KEY or OPENAI_API_KEY as environment variable

## Commands
- Run: `python main.py <directory_path>`
- Use existing JSON: `python main.py <directory_path> --json <json_file>`
- Specify BLIP model: `python main.py <directory_path> --blip-model <model_name>`
- Default BLIP model: 'Salesforce/blip2-flan-t5-xl-coco'

## Code Style Guidelines
- Follow PEP 8 conventions
- Imports: standard library first, then third-party, then local
- Type hints: Use for function parameters and return values
- Variable naming: lowercase_with_underscores for variables/functions
- Error handling: Use try/except blocks with specific exceptions
- String formatting: Use f-strings for string interpolation
- Comments: Docstrings for modules and functions, inline comments for complex logic
- Max line length: 88 characters (Black formatter default)
- Use constants for configuration values

## Linting & Formatting
- Install dev tools: `pip install black ruff mypy`
- Format: `black .`
- Lint: `ruff check .`
- Type check: `mypy .`

## Project Files
- `main.py`: Core script that processes images and generates reports
- `template.html`: HTML template for the interactive report
- `blip_prompt.txt`: Custom prompt for the BLIP model (optional)
- `fix_scripts/`: Contains utility scripts for fixing various issues
- `test_scripts/`: Contains test scripts for different models and functionality


## Git & Version Control
- Do not perform git-related commands (add, commit, push, etc.) unless explicitly requested
- Do not run linting or type checking
- Always ask for confirmation before modifying version control