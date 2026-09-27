# AI Image Categorizer

Describes a folder of images with a vision model, groups them into categories,
and produces an interactive HTML report for reorganizing them.

<div align="center">
  <a href="https://www.youtube.com/watch?v=8wniawe13Xc">
    <img src="https://img.youtube.com/vi/8wniawe13Xc/0.jpg" alt="Introduction Video" width="400">
  </a>
  <p>Introduction video</p>
</div>

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Vision%20Models-purple)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Maintenance status:** dependencies were upgraded on 2026-09-26 (Python 3.12,
> torch 2.14, transformers 5.17). See [docs/DEPENDENCY_AUDIT.md](docs/DEPENDENCY_AUDIT.md).

## Quick Start

Requires [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/DrBenedictPorkins/image-categorizer.git
cd image-categorizer
uv sync
```

Run fully locally with Ollama. Pull a vision model for descriptions and a text
model for categorization, then point the tool at them in `.env` (values in `.env`
take precedence over exported variables):

```bash
ollama pull qwen3.6:35b            # vision model, Phase 1
ollama pull mistral-small3.2:24b   # text model, Phase 2
cp .env.example .env               # then set OLLAMA_MODEL and OLLAMA_TEXT_MODEL
uv run python main.py /path/to/images --provider ollama
```

Or fully locally with HuggingFace models (downloaded on first run):

```bash
uv run python main.py /path/to/images --provider huggingface
```

No API key is required. The report `image_categories.html` is written into the
image directory and opened in the default browser.

### Recommended workflow for large libraries

Describe once, then categorize as often as you like:

```bash
# Phase 1: describe every photo (saves progress; rerun to resume after a crash)
uv run python main.py "/path/to/images" --description-provider ollama --describe-only \
    --init-categories "Portraits,Family Photos,Accidental Shots"

# Phase 2: categorize from the saved descriptions (minutes, no image processing)
uv run python main.py "/path/to/images" --categorization-provider ollama \
    --categorize-from "/path/to/images/descriptions_only.json"
```

Then open the report, move or trash photos, click **Save categories** to keep
your category list for next time, and **Export move script** to sort the files
into folders.

For reference: 1,195 iPad photos took about 80 minutes for Phase 1 and 8 minutes
for Phase 2 with `qwen3.6:35b` and `mistral-small3.2:24b` on an RTX 4090.

## How It Works

Processing is split into two phases that can use different providers.

**Phase 1 - Description.** A vision model describes each image, judges whether
it is an accidental or failed shot, and suggests 2-5 categories for it. Progress
is saved to `descriptions_only.json` every 10 images; rerunning the same command
resumes and retries failed images.

**Phase 2 - Categorization (Ollama).** A text model assigns each image to your
saved category list, in batches of 40. Each category has a rule describing what
belongs in it. On the first run, with no saved list, the list is built from the
images' suggestions. On later runs, images that fit no saved rule are grouped
into new categories, which are kept only if enough images land in them, marked
new, and saved for you to review. Images that still fit nothing go to Unsorted.

### Saved categories

The category list lives in `~/.config/image-categorizer/categories.yaml`
(override with `CATEGORIES_FILE`) and is reused for every photo library:

```yaml
categories:
  - name: Portraits
    rule: Photos of people looking at the camera, including selfies.
    status: kept
```

`status: new` marks categories the model added and you have not reviewed. Edit
the file by hand, or use **Save categories** in the report: it downloads the list
with your renames, merges, rule edits and new categories, all marked kept.

Splitting the phases allows:
- Re-running categorization without re-describing images.
- Describing locally and categorizing with a different provider.
- Using a cheap keyword-based Phase 2 for quick iteration.

## Providers

| Provider | Phase 1 | Phase 2 | Runs where | Status |
|---|---|---|---|---|
| `ollama` | Yes | Yes | Ollama server (local or remote) | Implemented |
| `huggingface` | Yes | Yes | In-process, local models | Implemented |
| `keyword` | No | Yes | In-process, keyword matching | Implemented |
| `anthropic`, `openai`, `bedrock` | - | - | - | Config stubs only, not implemented |

## Usage

### Workflow modes

```bash
# 1. Full pipeline, one provider for both phases
uv run python main.py /path/to/images --provider ollama

# 2. Phase 1 only - writes descriptions_only.json
uv run python main.py /path/to/images --description-provider huggingface --describe-only

# 3. Phase 2 only - from a saved descriptions file
uv run python main.py /path/to/images --categorization-provider keyword \
    --categorize-from /path/to/images/descriptions_only.json

# 4. Two phases, different providers
uv run python main.py /path/to/images \
    --description-provider huggingface --categorization-provider ollama
```

### Options

| Option | Meaning |
|---|---|
| `--init-categories "A,B,C"` or `--init-categories file.txt` | Category hints for Phase 1 (file: one per line) |
| `--model <name>` | Overrides `OLLAMA_MODEL` |
| `--no-html` | Do not open the report in a browser |

### Rebuild the HTML report from existing results

```bash
uv run python core/html_generator.py /path/to/images/categorization_results.json
```

### Output files

Written into the image directory:

| File | Produced by |
|---|---|
| `descriptions_only.json` | Modes 2 and 4 |
| `categorization_results.json` | Modes 1, 3 and 4 |
| `image_categories.html` | Modes 1, 3 and 4 |

## Configuration

Settings are read from environment variables. A `.env` file in the repository
root is loaded automatically; `.env.example` lists the variables.

### Ollama

| Variable | Default |
|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.2-vision:latest` |
| `OLLAMA_TEXT_MODEL` | `llama3.2:latest` |
| `OLLAMA_TIMEOUT` | `300` seconds |
| `OLLAMA_MAX_RETRIES` | `2` |
| `OLLAMA_RETRY_DELAY` | `1.0` seconds, doubled on each retry |

### HuggingFace

| Variable | Default |
|---|---|
| `HF_VISION_MODEL` | `Salesforce/blip2-flan-t5-xl-coco` |
| `HF_TEXT_MODEL` | `google/flan-t5-xl` |
| `HF_DEVICE` | `auto` (MPS, then CUDA, then CPU) |
| `HF_CACHE_DIR` | HuggingFace default cache |
| `HUGGINGFACE_TOKEN` | unset; needed only for gated models |

Predefined models:

- Vision: `Salesforce/blip2-flan-t5-xl-coco` (~15GB),
  `llava-hf/llava-1.5-7b-hf` (~13GB),
  `Salesforce/blip-image-captioning-base` (~2GB),
  `openbmb/MiniCPM-V-2` (~8GB; does not currently load, see audit)
- Text: `google/flan-t5-xl` (~3GB), `microsoft/phi-2` (~5GB)

Other model IDs are attempted through the transformers Auto classes.

Example, smaller and faster:

```bash
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
export HF_TEXT_MODEL="google/flan-t5-xl"
uv run python main.py /path/to/images --provider huggingface
```

See [docs/HUGGINGFACE_PROVIDER.md](docs/HUGGINGFACE_PROVIDER.md) and
[docs/QUICK_START_HUGGINGFACE.md](docs/QUICK_START_HUGGINGFACE.md).

## HTML Report

- Photos grouped by category, each category with its rule (click to edit) and a
  New badge for categories added on this run
- Category rail with counts; drag photos onto a category there or in the grid
- Per-photo category menu, including suggested categories as new categories
- Rename a category (renaming to an existing name merges them), create, remove
  empty categories, move a whole category to Trash
- Trash with restore to the original category
- Full-size preview with arrow-key browsing, trash and restore
- **Save categories**: downloads your category list for future runs
- **Export move script**: a bash script that moves each photo into a folder
  named after its category (trashed photos go to a Trash folder)

The template is `template.html` and is loaded by relative path, so run commands
from the repository root.

### Supported image formats

`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`

## Troubleshooting

**Ollama: connection refused** - start `ollama serve`, or set `OLLAMA_HOST` to
the remote server.

**Ollama: model not found** - `ollama pull <model>` for both `OLLAMA_MODEL` and
`OLLAMA_TEXT_MODEL`.

**HuggingFace: out of memory** - use `Salesforce/blip-image-captioning-base`,
or set `HF_DEVICE=cpu`.

**HuggingFace: slow** - check the startup line `Using device: ...` reports `mps`
or `cuda`.

## Requirements

- Python 3.12+ (see `.python-version`)
- For `ollama`: a running Ollama server with a vision model and a text model
- For `huggingface`: disk and RAM for the chosen models; Apple Silicon (MPS) or
  NVIDIA (CUDA) recommended

## Project Layout

| Path | Contents |
|---|---|
| `main.py` | Command line, workflow modes, Phase 1 progress saving |
| `core/config.py` | Provider settings from environment variables |
| `core/categories.py` | Saved category list (load, save) |
| `core/image_processor.py` | Image discovery and validation |
| `core/html_generator.py` | Builds the HTML report from results |
| `template.html` | Report template (layout, drag and drop, move script) |
| `models/image_data.py` | `ImageData`, `CategorizationResult` |
| `providers/` | Ollama, HuggingFace and keyword providers |
| `prompts/` | Prompt text files |
| `docs/` | Provider guides and the dependency audit |
| `sundry/` | Pre-refactor code and experiment scripts, not used by the app |

## License

MIT. See [LICENSE](LICENSE).
