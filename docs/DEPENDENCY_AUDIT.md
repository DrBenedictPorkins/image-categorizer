# Dependency Audit

Date: 2026-09-26. Source: PyPI JSON API and the installed `.venv`.

## Environment

- `.python-version`: 3.9.19. Python 3.9 reached end-of-life in October 2025.
- `.venv/bin/python3.9` is an x86_64 binary (Rosetta on Apple Silicon).
- `torch` 2.2.x is the last release with macOS x86_64 wheels, which is why
  `pyproject.toml` pins `torch>=2.2.0,<2.3.0`. torch 2.2 is built against
  numpy 1.x, which is why `numpy<2.0.0` is pinned.

## Package table

| Package | Declared | Installed | Latest | Latest needs Python | Used by live code |
|---|---|---|---|---|---|
| torch | >=2.2.0,<2.3.0 | 2.2.2 | 2.14.0 | >=3.10 | `providers/huggingface_provider.py` |
| transformers | >=4.47.0 | 4.56.1 | 5.17.0 | >=3.10 | `providers/huggingface_provider.py` |
| huggingface_hub | >=0.27.0 | 0.34.4 | 2.0.0 | >=3.10 | `huggingface_provider.py` (`login`) |
| accelerate | >=1.2.0 | 1.10.1 | 1.15.0 | >=3.10 | Not imported (optional transformers backend) |
| numpy | >=1.26.0,<2.0.0 | 1.26.4 | 2.5.3 | >=3.12 | Not imported directly (torch/transformers) |
| pillow | >=11.1.0 | 11.1.0 | 12.3.0 | >=3.10 | `core/image_processor.py`, providers |
| requests | >=2.32.0 | 2.32.3 | 2.34.2 | >=3.10 | `providers/ollama_provider.py` |
| python-dotenv | >=1.0.0 | 1.1.1 | 1.2.3 | >=3.10 | `main.py` |
| PyYAML | >=6.0.2 | 6.0.2 | 6.0.3 | >=3.8 | providers |
| ollama | >=0.4.5 | 0.5.1 | 0.6.2 | >=3.8 | Root `test_ollama_*.py` scripts and `ollama_provider_original.py` only |
| openai | >=1.59.0 | 1.68.2 | 3.19.2 | >=3.10 | `sundry/main_original.py` only |
| psutil | >=6.1.0 | 7.0.0 | 7.2.2 | >=3.6 | `sundry/main_original.py` only |
| httpx | >=0.28.1 | 0.28.1 | 0.28.1 | >=3.8 | Not imported anywhere |

## Breaking changes that affect this code

### transformers 4.x -> 5.x
Source: `MIGRATION_GUIDE_V5.md` in the transformers repository.

- `AutoModelForVision2Seq` is removed in favor of `AutoModelForImageTextToText`.
  Used at `providers/huggingface_provider.py:32`, `:59`, `:73`, `:200`.
  Import fails at module load, so every run breaks, including Ollama runs,
  because `providers/__init__.py` imports `HuggingFaceProvider` unconditionally.
- The default load dtype becomes `auto`. This code passes
  `torch_dtype=torch.float32` explicitly, so the numeric default does not change.
  `torch_dtype` itself is deprecated in favor of `dtype`.
- Generation parameters are no longer read from the model config. This code
  passes all generation parameters to `generate()` directly; no change expected.
- Remote-code models: internal tokenizer module paths moved. Relevant only to
  MiniCPM-V-2, which already fails to load (no `trust_remote_code=True`).

### Python 3.9 -> 3.10+
- Required by the latest release of 9 of 13 declared packages.
- numpy 2.5 requires Python >=3.12.
- The code already uses `list[str]` and `tuple[...]` generics, which are valid
  on 3.9+, so no syntax change is required for a newer interpreter.

### torch 2.2 -> 2.14
- Requires a native arm64 (or Linux) interpreter on this Mac.
- Unblocks numpy 2.x.

### Minor / low risk
- pillow 12: code uses only `Image.open`, `convert`, `save`, `verify`, `new`.
- requests, python-dotenv, PyYAML: patch/minor releases.
- huggingface_hub 2.0: only `login(token=...)` is used; verify on upgrade.

## Open decisions (resolved, see below)

1. Target Python version (3.11, 3.12, or newer).
2. Recreate `.venv` on a native arm64 interpreter.
3. Whether to remove `httpx`, `psutil`, `openai` (unused by live code), and
   `ollama` (used only by experiment scripts).

## Upgrade applied (2026-09-26)

Decisions: Python 3.12, native arm64 venv, remove `httpx` and `ollama`.

| Package | Before | After |
|---|---|---|
| Python | 3.9.19 x86_64 | 3.12.9 arm64 |
| torch | 2.2.2 | 2.14.0 |
| transformers | 4.56.1 | 5.17.0 |
| huggingface_hub | 0.34.4 | 1.33.0 (transformers 5.17 requires <2.0) |
| numpy | 1.26.4 | 2.5.3 |
| accelerate | 1.10.1 | 1.15.0 |
| pillow | 11.1.0 | 12.3.0 |
| requests | 2.32.3 | 2.34.2 |
| python-dotenv | 1.1.1 | 1.2.3 |
| PyYAML | 6.0.2 | 6.0.3 |
| openai | 1.68.2 | 3.19.2 |
| psutil | 7.0.0 | 7.2.2 |
| httpx | 0.28.1 | removed |
| ollama | 0.5.1 | removed |

Code changes, `providers/huggingface_provider.py` only:
- `AutoModelForVision2Seq` -> `AutoModelForImageTextToText`
- `torch_dtype=torch.float32` -> `dtype=torch.float32`

Verification:
- `import main` and `main.py --help` succeed.
- Ollama provider, full pipeline, local server, `llava:latest` +
  `llama3.2:latest`, 5 images in `test_images/`: all described and categorized.
- HuggingFace provider, full pipeline, `blip-image-captioning-base` +
  `google/flan-t5-small`, same 5 images: runs to completion and writes
  `categorization_results.json`. Output quality is poor (see below).
- The old stack (transformers 4.56 + torch 2.2) could not load
  `blip-image-captioning-base` or `flan-t5-small` at all: transformers refuses
  non-safetensors weights on torch <2.6 (CVE-2025-32434).

Pre-existing defects found, not fixed:
- BLIP-base returns the prompt, lowercased, followed by the caption. The strip at
  `providers/huggingface_provider.py:454` is case-sensitive, so the prompt stays
  in the description.
- MiniCPM-V-2 is loaded without `trust_remote_code=True`.
- The experiment scripts now in `sundry/experiments/` that import `ollama` fail.
