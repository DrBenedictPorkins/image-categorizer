# Phase Separation Testing Guide

This document provides a quick reference for testing the phase separation features of the image categorization system.

## Quick Start

```bash
# Run all tests
python test_phase_separation_workflows.py

# Run specific test
python test_phase_separation_workflows.py --test test_1_fully_local

# Limit number of test images
python test_phase_separation_workflows.py --images 3
```

## Test Overview

### Test 1: Fully Local Workflow
**Command:** `python test_phase_separation_workflows.py --test test_1_fully_local`

**Description:** HuggingFace vision model → HuggingFace text model

**Benefits:**
- Complete privacy (no cloud API calls)
- No internet required after model download
- Free to use

**Use Case:** Maximum privacy for sensitive images

---

### Test 2: Hybrid Workflow (HF → Ollama)
**Command:** `python test_phase_separation_workflows.py --test test_2_hybrid_hf_ollama`

**Description:** HuggingFace vision (local) → Ollama categorization (local/remote)

**Benefits:**
- Images never leave local machine
- Better categorization quality than HF text models
- Flexible deployment (Ollama can be local or remote)

**Use Case:** Privacy-conscious workflows with high-quality categorization

---

### Test 3: Cloud Baseline (Ollama → Ollama)
**Command:** `python test_phase_separation_workflows.py --test test_3_cloud_baseline`

**Description:** Traditional single-provider approach

**Benefits:**
- Simple setup (one provider)
- Consistent quality
- Can run locally or remotely

**Use Case:** Baseline comparison for other workflows

---

### Test 4: Description-Only Workflow
**Command:** `python test_phase_separation_workflows.py --test test_4_description_only`

**Description:** Generate and save image descriptions for later categorization

**Output:** Creates `descriptions.json` for use in Test 5

**Benefits:**
- Separate description and categorization phases
- Experiment with different categorization strategies
- Batch processing workflows

**Use Case:** Generate descriptions locally, categorize later with different provider

---

### Test 5: Categorization-Only Workflow
**Command:** `python test_phase_separation_workflows.py --test test_5_categorization_only`

**Description:** Load pre-described images and categorize them

**Prerequisites:** Test 4 must be run first

**Benefits:**
- Re-categorize with different models/parameters
- Use cloud categorization on locally-described images
- Faster iteration on categorization strategies

**Use Case:** Testing different categorization approaches without re-describing images

---

### Test 6: Keyword Categorization
**Command:** `python test_phase_separation_workflows.py --test test_6_keyword_categorization`

**Description:** HuggingFace vision → Simple keyword matching

**Benefits:**
- Very fast categorization
- Deterministic results
- No additional LLM compute
- Good for common categories

**Drawbacks:**
- Limited to predefined categories (People, Nature, Food, Animals, Architecture, Vehicles, Technology, Art)

**Use Case:** Fast local categorization with predefined categories

---

### Test 7: Backward Compatibility
**Command:** `python test_phase_separation_workflows.py --test test_7_backward_compat`

**Description:** Single provider using traditional `process_images()` method

**Purpose:** Validate that new phase separation doesn't break existing code

**Use Case:** Ensure backward compatibility

---

## Test Workflow Examples

### Example 1: Privacy-First Workflow
```bash
# Step 1: Describe images locally with HuggingFace
python test_phase_separation_workflows.py --test test_4_description_only

# Step 2: Categorize with fast keyword matching (no cloud)
python test_phase_separation_workflows.py --test test_6_keyword_categorization
```

### Example 2: Quality-First Workflow
```bash
# Step 1: Describe images locally (privacy)
python test_phase_separation_workflows.py --test test_4_description_only

# Step 2: Categorize with Ollama (quality)
python test_phase_separation_workflows.py --test test_5_categorization_only
```

### Example 3: Compare Providers
```bash
# Test all workflows to compare quality/speed
python test_phase_separation_workflows.py

# Review results in test_output/ directory
ls -la test_output/*/categorization_results.json
```

## Test Output Structure

Each test creates output in `test_output/<test_name>/`:

```
test_output/
├── test_1_fully_local/
│   └── categorization_results.json
├── test_2_hybrid_hf_ollama/
│   └── categorization_results.json
├── test_3_cloud_baseline/
│   └── categorization_results.json
├── test_4_description_only/
│   ├── descriptions.json          ← Intermediate results
│   └── categorization_results.json
├── test_5_categorization_only/
│   └── categorization_results.json
├── test_6_keyword_categorization/
│   └── categorization_results.json
└── test_7_backward_compat/
    └── categorization_results.json
```

## Environment Configuration

Ensure your `.env` file has the necessary provider settings:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision:latest
OLLAMA_TIMEOUT=300
OLLAMA_MAX_RETRIES=2
OLLAMA_RETRY_DELAY=1.0

# HuggingFace Configuration (optional, uses defaults)
HF_VISION_MODEL=Salesforce/blip2-flan-t5-xl-coco
HF_TEXT_MODEL=google/flan-t5-xl
HF_DEVICE=auto  # auto, cpu, cuda, or mps
# HF_CACHE_DIR=/path/to/cache  # Optional
# HUGGINGFACE_TOKEN=your_token  # Optional, for gated models
```

## Troubleshooting

### HuggingFace Out of Memory
```bash
# Use smaller models
export HF_VISION_MODEL=Salesforce/blip-image-captioning-base
export HF_TEXT_MODEL=google/flan-t5-base
```

### Ollama Connection Issues
```bash
# Check Ollama is running
ollama list

# Test connection
curl http://localhost:11434/api/tags

# Check model is available
ollama pull llama3.2-vision:latest
```

### Test 5 Prerequisites
```bash
# If Test 5 fails, run Test 4 first
python test_phase_separation_workflows.py --test test_4_description_only
python test_phase_separation_workflows.py --test test_5_categorization_only
```

## Performance Tips

1. **Limit Test Images:** Use `--images 3` for faster testing
2. **Use Smaller Models:** Configure smaller HF models in `.env`
3. **Run Tests Individually:** Test specific workflows instead of all at once
4. **GPU Acceleration:** Enable CUDA/MPS for faster HuggingFace processing

## Understanding Results

Each test outputs:
- **Passed/Failed Status:** Whether the test completed successfully
- **Image Count:** Number of images processed
- **Category Count:** Number of unique categories generated
- **Sample Results:** First 3 images with descriptions and categories
- **JSON Output:** Complete results saved to file

Compare results across tests to evaluate:
- Quality: Which provider gives better categories?
- Speed: Which workflow is fastest?
- Privacy: Which workflow keeps data local?
- Cost: Which workflow is most cost-effective?
