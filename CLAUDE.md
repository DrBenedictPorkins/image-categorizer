# Image Categorizer - Guidelines for Claude

## Project Overview
- Image categorization tool using Google's Gemma 3 model
- Takes directory of images and assigns single-word labels to each

## Environment & Dependencies
- Python 3.13+
- Key packages: transformers, torch, pillow, httpx
- Install: `uv pip install -e .`

## Commands
- Run: `python main.py <directory_path>`

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


## Git & Version Control
- Do not perform git-related commands (add, commit, push, etc.) unless explicitly requested
- Do not run linting or type checking
- Always ask for confirmation before modifying version control