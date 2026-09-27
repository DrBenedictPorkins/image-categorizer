# Image Categorizer - Guidelines for Claude

## Project Overview
- CLI tool that describes a folder of images with a vision model, groups them
  into categories, and writes an interactive drag-and-drop HTML report.
- **Phase 1 (Description)**: vision model produces a description plus 2-5
  suggested categories per image.
- **Phase 2 (Categorization)**: text model reads all descriptions together and
  assigns one final category per image.
- **Implemented providers**: `ollama`, `huggingface`, `keyword`.
- **Not implemented**: `anthropic`, `openai`, `bedrock` (config stubs only in
  `core/config.py`; no provider class, not selectable from the CLI).

## Project State (audited 2026-09-26)
- Last code activity: Nov 2025. Last commit: `b1d7fdd`. Large amount of
  uncommitted and untracked work sits on `main`.
- Old one-off experiment scripts and the earlier Ollama provider copies live in
  `sundry/experiments/`; several import the removed `ollama` package and no
  longer run. The remaining root `test_*.py` files are ad-hoc scripts, not a
  test suite; there is no pytest setup.
- `sundry/` holds the pre-refactor `main_original.py` and old template. It is the
  only code that imports `openai`, `anthropic`, or `psutil`.
- `.venv` is native arm64 CPython 3.12 (rebuilt 2026-09-26; the previous
  venv was x86_64 Python 3.9 under Rosetta, which capped torch at 2.2).

## Architecture

### Entry points
- `main.py` - CLI, workflow selection, provider initialization, JSON output.
- `core/html_generator.py` - builds `image_categories.html` from results; also
  runnable standalone on a `categorization_results.json`.
- `core/image_processor.py` - image discovery, validation, RGB conversion.
- `core/config.py` - reads env vars into a `ProviderConfig` per provider.
- `models/image_data.py` - `ImageData`, `CategorizationResult`, `ProviderConfig`.
- `providers/base.py` - `BaseLLMProvider` abstract interface.

### Providers
| Provider | Phase 1 | Phase 2 | Transport |
|---|---|---|---|
| `ollama` | Yes | Yes | Ollama REST API via `requests` (not the `ollama` package) |
| `huggingface` | Yes | Yes | Local `transformers` + `torch` |
| `keyword` | No | Yes | Keyword matching on descriptions, no model |

### Workflow modes (`main.py`)
1. Full pipeline: `--provider <name>`
2. Describe only: `--description-provider <name> --describe-only`
   (writes `descriptions_only.json`)
3. Categorize only: `--categorization-provider <name> --categorize-from <json>`
4. Two-phase, mixed providers: `--description-provider A --categorization-provider B`

### Output files (written into the image directory)
- `descriptions_only.json` - Phase 1 output (modes 2 and 4)
- `categorization_results.json` - final result (modes 1, 3, 4)
- `image_categories.html` - interactive report (modes 1, 3, 4)

## Environment & Dependencies
- Python 3.12 (`.python-version`, `requires-python >=3.12`), managed with `uv`.
- Install: `uv sync`
- Run: `uv run python main.py ...`
- `.env` in the repo root is loaded by `main.py` with `override=True`.

## Provider Configuration

### Ollama
- `OLLAMA_HOST` - default `http://localhost:11434`
- `OLLAMA_MODEL` - vision model; default in `core/config.py` is
  `llama3.2-vision:latest` (note: a comment in `ollama_provider.py` says
  llama3.2-vision hangs and recommends `llava:latest`; the config default wins)
- `OLLAMA_TEXT_MODEL` - Phase 2 model, default `llama3.2:latest`
- `OLLAMA_TIMEOUT` (300), `OLLAMA_MAX_RETRIES` (2), `OLLAMA_RETRY_DELAY` (1.0)
- Retries use exponential backoff: `delay * 2**attempt`.
- `--model` on the CLI only overrides `OLLAMA_MODEL`.

### HuggingFace
- `HF_VISION_MODEL` - default `Salesforce/blip2-flan-t5-xl-coco`
- `HF_TEXT_MODEL` - default `google/flan-t5-xl`
- `HF_DEVICE` - `auto` / `mps` / `cuda` / `cpu` (auto order: MPS > CUDA > CPU)
- `HF_CACHE_DIR` - optional
- `HUGGINGFACE_TOKEN` - optional, for gated models
- Predefined vision models: BLIP-2, LLaVA-1.5, BLIP-base, MiniCPM-V-2.
  Predefined text models: Flan-T5-XL, Phi-2. Any other model ID is loaded
  through the Auto classes.
- MiniCPM-V-2 is listed but loaded without `trust_remote_code=True`; it will
  not load as written.

### Anthropic / OpenAI / Bedrock
- Env vars are read in `core/config.py` but no provider class exists.

## Phase 2 and Saved Categories (Ollama provider)
- `core/categories.py` loads/saves the user's category list (name, rule, status
  kept|new), default `~/.config/image-categorizer/categories.yaml`, env
  `CATEGORIES_FILE`. Unreadable file: one-off list, file never overwritten.
- `_categorize_in_batches`: no saved list -> `_build_taxonomy` from suggestion
  tally, all marked new. Saved list -> `_assign_in_batches(allow_none=True)`;
  photos answered "None of these" are pooled; if >= max(3, n/100) the model
  proposes few broad categories for that pool (`_propose_categories_for`), kept
  only if each gets >= that many photos. Leftovers go to `Unsorted` (never saved).
- Batches of 40 (`CATEGORIZE_BATCH_SIZE`); off-list answers are re-asked for
  those photos only (`CATEGORIZE_REASK_ROUNDS`). Requests send `think: false` and
  `num_ctx` 16384.
- `CategorizationResult.category_definitions` carries the list to the report.
- Report (`template.html`) data: `{{card_data}}`, `{{categories_data}}` (name,
  count, rule, status), `{{report_meta}}`. Old template: `sundry/template_legacy.html`.
- Phase 1 (`main.py describe_images_only`) saves every 10 images and resumes.

## Dependency Maintenance
Upgrade done 2026-09-26; details in `docs/DEPENDENCY_AUDIT.md`.
- Python 3.9 x86_64 -> 3.12 arm64; torch 2.2.2 -> 2.14.0; transformers
  4.56.1 -> 5.17.0; numpy 1.26 -> 2.5; all other packages at latest.
- `httpx` and `ollama` removed from dependencies. The scripts in
  `sundry/experiments/` that import `ollama` no longer run.
- transformers 5 changes applied in `providers/huggingface_provider.py`:
  `AutoModelForVision2Seq` -> `AutoModelForImageTextToText`,
  `torch_dtype=` -> `dtype=`.
- transformers 5.17 requires `huggingface_hub<2.0`, so it is held at 1.x.
- Known pre-existing HF defects (not caused by the upgrade): BLIP output starts
  with the lowercased prompt, which the case-sensitive strip at
  `huggingface_provider.py:454` does not remove; MiniCPM-V-2 lacks
  `trust_remote_code=True`.

## Commands
- Ollama: `uv run python main.py <dir> --provider ollama`
- HuggingFace: `uv run python main.py <dir> --provider huggingface`
- Mixed: `uv run python main.py <dir> --description-provider huggingface --categorization-provider ollama`
- Keyword Phase 2 from saved descriptions:
  `uv run python main.py <dir> --categorization-provider keyword --categorize-from <dir>/descriptions_only.json`
- Initial categories: `--init-categories "Nature,People,Food"` or a file path
  (one category per line, e.g. `CATEGORIES.txt`)
- Skip opening the browser: `--no-html`
- Rebuild HTML only: `uv run python core/html_generator.py <dir>/categorization_results.json`

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
- Dev tools are not declared in `pyproject.toml`.
- Format: `black .` / Lint: `ruff check .` / Types: `mypy .`

## Project Files
- `main.py` - CLI entry point
- `template.html` - HTML report template (loaded by relative path; run from repo root)
- `prompts/` - prompt text files
- `blip_prompt.txt` - legacy BLIP prompt
- `CATEGORIES.txt` - sample initial-categories file
- `docs/` - HuggingFace provider docs and the dependency audit
- `test_images/` - sample images
- `sundry/` - pre-refactor code, legacy report template and experiment scripts, not used
- `LICENSE` - MIT
- Root `*_GUIDE.md` / `*_USAGE.md` / `*_TESTING.md` - design notes from the
  phase-separation work

## Git & Version Control
- Do not perform git-related commands (add, commit, push, etc.) unless explicitly requested
- Do not run linting or type checking
- Always ask for confirmation before modifying version control
