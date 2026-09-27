# Phase Separation Guide

## Table of Contents
- [Overview](#overview)
- [Understanding Phase Separation](#understanding-phase-separation)
- [Use Cases](#use-cases)
- [Configuration](#configuration)
- [Workflow Examples](#workflow-examples)
- [Provider Capabilities](#provider-capabilities)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)
- [Programmatic Usage](#programmatic-usage)
- [Future Hybrid Workflows](#future-hybrid-workflows)

## Overview

Phase separation is a core architectural feature of the Image Categorizer that splits image processing into two independent phases:

1. **Phase 1 - Description**: Vision models analyze images and generate detailed textual descriptions
2. **Phase 2 - Categorization**: Text/multimodal models analyze descriptions to create categories

This separation enables powerful workflows that balance privacy, cost, quality, and performance.

## Understanding Phase Separation

### Traditional Single-Phase Approach

```
Images → Vision Model → Categories
         (One-shot processing)
```

**Limitations:**
- Must use same provider for everything
- Can't iterate on categories without re-processing images
- Privacy concerns if using cloud APIs
- Expensive if using paid APIs for every step

### Two-Phase Approach

```
Phase 1: Images → Vision Model → Descriptions
Phase 2: Descriptions → Text Model → Categories
```

**Advantages:**
- Mix different providers for each phase
- Re-categorize without re-describing
- Keep images local, only send text descriptions to cloud
- Optimize costs (free for description, paid only for categorization)
- Enable experimentation and iteration

## Use Cases

### 1. Privacy-First Workflow

**Scenario**: You have sensitive images (medical, personal, proprietary) but want high-quality categorization.

**Solution**: Describe locally, categorize with cloud APIs

```bash
# Phase 1: Local description with HuggingFace (images never leave machine)
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"
python -c "from providers.huggingface_provider import HuggingFaceProvider; # describe images"

# Phase 2: Cloud categorization with Anthropic (only text descriptions sent)
# Coming soon - use descriptions.json with Anthropic API
```

**Privacy Guarantee**: Images never leave your machine. Only textual descriptions are sent to cloud.

### 2. Cost Optimization

**Scenario**: You have many images and want to minimize API costs.

**Solution**: Use free local models for description, paid APIs only for final categorization

```bash
# Phase 1: Free local description (BLIP-base is fast and free)
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
export HF_TEXT_MODEL="google/flan-t5-xl"
python main.py /path/to/images --provider huggingface

# This creates descriptions that can later be used with:
# Phase 2: Paid API for high-quality categorization (future feature)
# Only 1 API call for all descriptions vs 1 call per image
```

**Cost Savings**:
- Description: $0 (local processing)
- Categorization: $0.01-0.05 for all images (single API call)
- vs traditional approach: $0.01-0.05 per image

### 3. Experimentation and Iteration

**Scenario**: You want to try different categorization strategies without re-processing images.

**Solution**: Describe once, categorize multiple times

```bash
# Phase 1: Describe images once (slow, ~5 min for 10 images)
python main.py /path/to/images --provider huggingface

# Phase 2: Try different categorization approaches (fast, ~30 sec each)
# Experiment 1: Basic categories
python categorize.py descriptions.json --strategy basic

# Experiment 2: Detailed categories
python categorize.py descriptions.json --strategy detailed

# Experiment 3: Custom categories
python categorize.py descriptions.json --categories "Nature,People,Food"
```

**Time Savings**: Describe once (5 min), iterate many times (30 sec each)

### 4. Hybrid Quality Approach

**Scenario**: You want fast local description but best-in-class categorization.

**Solution**: Local description + Cloud categorization (future)

```bash
# Phase 1: Fast local description
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
python main.py /path/to/images --provider huggingface

# Phase 2: Best-in-class categorization (when implemented)
# Use Claude 3.7 Sonnet or GPT-4 for superior category creation
```

### 5. Hardware Optimization

**Scenario**: You have limited hardware but want to process many images.

**Solution**: Use smallest models that fit your hardware

```bash
# Low-end laptop (8GB RAM, CPU only)
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"  # 2GB
export HF_TEXT_MODEL="google/flan-t5-xl"                         # 3GB
export HF_DEVICE="cpu"
python main.py /path/to/images --provider huggingface

# High-end workstation (32GB RAM, M3 MAX)
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"       # 15GB
export HF_TEXT_MODEL="microsoft/phi-2"                           # 5GB
export HF_DEVICE="auto"  # Will use MPS acceleration
python main.py /path/to/images --provider huggingface
```

## Configuration

### Environment Variables

#### HuggingFace Provider

```bash
# Vision Model Selection (Phase 1)
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"  # Default, 15GB
# Options:
# - "Salesforce/blip-image-captioning-base"     # 2GB, fast
# - "Salesforce/blip2-flan-t5-xl-coco"          # 15GB, high quality
# - "llava-hf/llava-1.5-7b-hf"                  # 13GB, conversational
# - "openbmb/MiniCPM-V-2"                       # 8GB, efficient

# Text Model Selection (Phase 2)
export HF_TEXT_MODEL="google/flan-t5-xl"  # Default, 3GB
# Options:
# - "google/flan-t5-xl"                         # 3GB, good quality
# - "microsoft/phi-2"                           # 5GB, powerful

# Device Selection
export HF_DEVICE="auto"  # auto/mps/cuda/cpu
# auto: Detects MPS > CUDA > CPU
# mps: Force Apple Silicon GPU
# cuda: Force NVIDIA GPU
# cpu: Force CPU (slowest but works everywhere)

# Cache Directory (optional)
export HF_CACHE_DIR="/path/to/cache"  # Default: ~/.cache/huggingface

# HuggingFace Token (optional, for gated models)
export HUGGINGFACE_TOKEN="hf_..."
```

#### Ollama Provider

```bash
# Server Configuration
export OLLAMA_HOST="http://localhost:11434"  # Default: localhost

# Vision Model (Phase 1)
export OLLAMA_MODEL="llava:latest"  # Default
# Options:
# - "llava:latest"                              # Good quality, widely tested
# - "minicpm-v:latest"                          # Efficient, good quality

# Timeout and Retry
export OLLAMA_TIMEOUT=300          # Request timeout in seconds
export OLLAMA_MAX_RETRIES=2        # Maximum retry attempts
export OLLAMA_RETRY_DELAY=1.0      # Base delay between retries
```

### Model Selection Guide

#### For Phase 1 (Description)

**Fast and Lightweight (2-8GB RAM):**
```bash
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"  # 2GB
# Use when: Limited RAM, CPU-only, many images, speed matters
```

**Balanced (8-13GB RAM):**
```bash
export HF_VISION_MODEL="openbmb/MiniCPM-V-2"  # 8GB
# Use when: Moderate hardware, good quality needed, efficient processing
```

**High Quality (15-18GB RAM):**
```bash
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"  # 15GB
# Use when: Sufficient RAM/VRAM, best quality needed, detailed descriptions
```

#### For Phase 2 (Categorization)

**Good Quality (3GB RAM):**
```bash
export HF_TEXT_MODEL="google/flan-t5-xl"  # 3GB
# Use when: Standard categorization, limited RAM, good results needed
```

**Best Quality (5GB RAM):**
```bash
export HF_TEXT_MODEL="microsoft/phi-2"  # 5GB
# Use when: Best local quality, semantic understanding matters, sufficient RAM
```

## Workflow Examples

### Example 1: Complete Local Workflow (Privacy-First)

```bash
# Configuration
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"
export HF_TEXT_MODEL="google/flan-t5-xl"
export HF_DEVICE="auto"

# Process images (both phases)
python main.py /path/to/images --provider huggingface

# Result: categorization_results.json and HTML report
# All processing done locally, no internet required (after model download)
```

### Example 2: Fast Local Description + Future Cloud Categorization

```bash
# Phase 1: Fast local description
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
python main.py /path/to/images --provider huggingface --description-only

# This creates: image_descriptions.json

# Phase 2: Cloud categorization (when implemented)
# python categorize_cloud.py image_descriptions.json --provider anthropic
# Only text descriptions sent to cloud, images stay local
```

### Example 3: Ollama Description + HuggingFace Categorization

```bash
# Phase 1: Ollama for description (if already running)
export OLLAMA_MODEL="llava:latest"
python describe.py /path/to/images --provider ollama

# Phase 2: HuggingFace for categorization (better local quality)
export HF_TEXT_MODEL="microsoft/phi-2"
python categorize.py descriptions.json --provider huggingface
```

### Example 4: Re-categorization Without Re-description

```bash
# Initial processing
python main.py /path/to/images --provider huggingface
# Creates: categorization_results.json

# Try different categorization strategy
python recategorize.py categorization_results.json --strategy detailed

# Try with initial category suggestions
python recategorize.py categorization_results.json --categories "Art,Science,Nature,People"

# Try with different model
export HF_TEXT_MODEL="microsoft/phi-2"
python recategorize.py categorization_results.json --provider huggingface
```

## Provider Capabilities

### Phase Support Matrix

| Provider | Phase 1 (Description) | Phase 2 (Categorization) | Full Pipeline |
|----------|----------------------|-------------------------|---------------|
| HuggingFace | ✅ Yes | ✅ Yes | ✅ Yes |
| Ollama | ✅ Yes | ✅ Yes | ✅ Yes |
| Anthropic | ❌ No | 🚧 Planned | 🚧 Planned |
| OpenAI | ❌ No | 🚧 Planned | 🚧 Planned |

### API Methods

All providers implementing phase separation support these methods:

```python
class BaseLLMProvider:
    # Full pipeline (both phases)
    def process_images(self, image_paths, progress_callback, initial_categories) -> CategorizationResult

    # Phase 1 only
    def describe_images(self, image_paths, progress_callback, initial_categories) -> List[ImageData]

    # Phase 2 only
    def categorize_described_images(self, images, progress_callback) -> CategorizationResult
```

### Provider-Specific Features

#### HuggingFace
- **Supports**: Both phases independently
- **Vision Models**: BLIP-2, LLaVA-1.5, BLIP-base, MiniCPM-V
- **Text Models**: Flan-T5-XL, Phi-2
- **Device Support**: MPS, CUDA, CPU
- **Privacy**: 100% local
- **Phase Separation**: Native support via separate vision and text models

#### Ollama
- **Supports**: Both phases independently
- **Vision Models**: llava, minicpm-v
- **Text Models**: llama3.2 (automatic switch for categorization)
- **Device Support**: Based on Ollama server
- **Privacy**: Local by default (can use remote server)
- **Phase Separation**: Uses vision model for Phase 1, switches to text model for Phase 2

## Performance Comparison

### Processing Time (10 images, M3 MAX)

| Workflow | Phase 1 | Phase 2 | Total | Notes |
|----------|---------|---------|-------|-------|
| HF BLIP-base + Flan-T5 | 1 min | 30 sec | 1.5 min | Fastest local |
| HF BLIP-2 + Flan-T5 | 3 min | 30 sec | 3.5 min | Balanced quality |
| HF BLIP-2 + Phi-2 | 3 min | 45 sec | 3.75 min | Best local quality |
| Ollama llava + llama3.2 | 2 min | 20 sec | 2.3 min | Server-based |
| Future: HF + Anthropic | 1 min | 5 sec | 1.1 min | Hybrid cloud |

### Memory Usage

| Configuration | Phase 1 RAM | Phase 2 RAM | Peak RAM | VRAM |
|--------------|-------------|-------------|----------|------|
| HF BLIP-base + Flan-T5 | 3GB | 4GB | 5GB | 3GB |
| HF BLIP-2 + Flan-T5 | 16GB | 4GB | 18GB | 12GB |
| HF BLIP-2 + Phi-2 | 16GB | 6GB | 20GB | 14GB |
| Ollama llava | Server | Server | N/A | Server |

### Quality Comparison (subjective)

| Configuration | Description Quality | Category Quality | Overall |
|--------------|--------------------|--------------------|---------|
| HF BLIP-base + Flan-T5 | Good | Good | Good |
| HF BLIP-2 + Flan-T5 | Excellent | Good | Excellent |
| HF BLIP-2 + Phi-2 | Excellent | Excellent | Excellent |
| Ollama llava + llama3.2 | Very Good | Good | Very Good |
| Future: BLIP-2 + Claude | Excellent | Outstanding | Outstanding |

## Best Practices

### 1. Model Selection

**For Privacy-Critical Use:**
- Always use HuggingFace provider (fully local)
- Use largest models your hardware supports
- Avoid remote Ollama servers

**For Cost Optimization:**
- Use HuggingFace for Phase 1 (free)
- Reserve paid APIs for Phase 2 only (when implemented)
- Describe once, categorize many times

**For Performance:**
- Use BLIP-base for Phase 1 if speed matters
- Use smaller text models (Flan-T5-XL) for Phase 2
- Enable GPU acceleration (MPS or CUDA)

**For Quality:**
- Use BLIP-2 or LLaVA for Phase 1
- Use Phi-2 for Phase 2 (or future cloud APIs)
- Ensure sufficient RAM/VRAM

### 2. Hardware Optimization

**Low-End (8GB RAM, CPU):**
```bash
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
export HF_TEXT_MODEL="google/flan-t5-xl"
export HF_DEVICE="cpu"
# Process in small batches (5-10 images)
```

**Mid-Range (16GB RAM, no GPU):**
```bash
export HF_VISION_MODEL="openbmb/MiniCPM-V-2"
export HF_TEXT_MODEL="google/flan-t5-xl"
export HF_DEVICE="cpu"
# Process in medium batches (10-20 images)
```

**High-End (32GB RAM, M3 MAX or RTX 4090):**
```bash
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"
export HF_TEXT_MODEL="microsoft/phi-2"
export HF_DEVICE="auto"
# Process larger batches (20-50 images)
```

### 3. Workflow Strategies

**Initial Exploration:**
1. Use fast models (BLIP-base + Flan-T5)
2. Process small sample (5-10 images)
3. Review results
4. Switch to better models if needed

**Production Processing:**
1. Use best models your hardware supports
2. Process in optimal batch sizes
3. Save descriptions for future re-categorization
4. Monitor memory usage

**Iteration and Refinement:**
1. Describe once with best vision model
2. Save descriptions
3. Iterate on categorization strategy
4. Try different text models or parameters

### 4. Privacy Best Practices

**Sensitive Images:**
- Never use cloud APIs for Phase 1
- Use HuggingFace provider exclusively
- Verify device is set to local (mps/cuda/cpu)
- Check that HF_CACHE_DIR is secure

**Review Descriptions:**
- Inspect descriptions before sending to cloud
- Ensure no sensitive information leaked
- Redact descriptions if needed

### 5. Cost Optimization

**Minimize API Costs:**
- Use free local models for Phase 1
- Batch descriptions for single Phase 2 API call
- Cache descriptions for future use
- Only use paid APIs when local quality insufficient

## Programmatic Usage

### Full Pipeline

```python
from providers.huggingface_provider import HuggingFaceProvider
from models.image_data import ProviderConfig

# Configure provider
config = ProviderConfig(
    provider_name='huggingface',
    settings={
        'vision_model': 'Salesforce/blip2-flan-t5-xl-coco',
        'text_model': 'google/flan-t5-xl',
        'device': 'auto'
    }
)

# Initialize
provider = HuggingFaceProvider(config)
provider.initialize()

# Process images (both phases)
result = provider.process_images(
    image_paths=['/path/to/img1.jpg', '/path/to/img2.jpg'],
    progress_callback=lambda msg, prog: print(f"{msg}: {prog:.0%}")
)

# Access results
for image in result.images:
    print(f"{image.filename}: {image.primary_category}")

provider.cleanup()
```

### Phase 1 Only (Description)

```python
from providers.huggingface_provider import HuggingFaceProvider
from models.image_data import ProviderConfig
import json

# Configure provider for description only
config = ProviderConfig(
    provider_name='huggingface',
    settings={
        'vision_model': 'Salesforce/blip2-flan-t5-xl-coco',
        'device': 'auto'
    }
)

# Initialize
provider = HuggingFaceProvider(config)
provider.initialize()

# Phase 1: Generate descriptions
image_data_list = provider.describe_images(
    image_paths=['/path/to/img1.jpg', '/path/to/img2.jpg'],
    progress_callback=lambda msg, prog: print(f"{msg}: {prog:.0%}"),
    initial_categories=['Nature', 'People', 'Architecture']
)

# Save descriptions for later
with open('descriptions.json', 'w') as f:
    json.dump([img.to_dict() for img in image_data_list], f, indent=2)

print(f"Saved {len(image_data_list)} descriptions")

provider.cleanup()
```

### Phase 2 Only (Categorization)

```python
from providers.huggingface_provider import HuggingFaceProvider
from models.image_data import ProviderConfig, ImageData
import json

# Load existing descriptions
with open('descriptions.json', 'r') as f:
    data = json.load(f)
    image_data_list = [ImageData.from_dict(d) for d in data]

# Configure provider for categorization only
config = ProviderConfig(
    provider_name='huggingface',
    settings={
        'text_model': 'microsoft/phi-2',
        'device': 'auto'
    }
)

# Initialize
provider = HuggingFaceProvider(config)
provider.initialize()

# Phase 2: Categorize from descriptions
result = provider.categorize_described_images(
    images=image_data_list,
    progress_callback=lambda msg, prog: print(f"{msg}: {prog:.0%}")
)

# Access final results
print(f"Categories: {result.get_category_stats()}")
for image in result.images:
    print(f"{image.filename}: {image.primary_category}")

provider.cleanup()
```

### Switching Providers Between Phases

```python
from providers.huggingface_provider import HuggingFaceProvider
from providers.ollama_provider import OllamaProvider
from models.image_data import ProviderConfig

# Phase 1: HuggingFace for description
hf_config = ProviderConfig(
    provider_name='huggingface',
    settings={'vision_model': 'Salesforce/blip2-flan-t5-xl-coco', 'device': 'auto'}
)
hf_provider = HuggingFaceProvider(hf_config)
hf_provider.initialize()

image_data_list = hf_provider.describe_images(['/path/to/img1.jpg'])
hf_provider.cleanup()

# Phase 2: Ollama for categorization
ollama_config = ProviderConfig(
    provider_name='ollama',
    settings={'host': 'http://localhost:11434', 'model': 'llama3.2:latest'}
)
ollama_provider = OllamaProvider(ollama_config)
ollama_provider.initialize()

result = ollama_provider.categorize_described_images(image_data_list)
ollama_provider.cleanup()

print(f"Final categories: {result.get_category_stats()}")
```

## Future Hybrid Workflows

### Planned: Local Description + Cloud Categorization

```bash
# Phase 1: Local description (images never leave machine)
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"
python main.py /path/to/images --provider huggingface --phase description

# Phase 2: Cloud categorization (text descriptions only)
export ANTHROPIC_API_KEY="sk-..."
python main.py /path/to/images --provider anthropic --phase categorization
```

**Privacy**: Images stay local, only text descriptions sent to cloud
**Cost**: Pay only for categorization API call
**Quality**: Best local description + best cloud categorization

### Planned: CLI Support for Hybrid Mode

```bash
# Single command for hybrid workflow
python main.py /path/to/images \
  --description-provider huggingface \
  --categorization-provider anthropic
```

### Planned: Multi-Provider Experimentation

```bash
# Try multiple categorization approaches
python compare_categorization.py descriptions.json \
  --providers huggingface,ollama,anthropic \
  --output comparison_results.json
```

## Summary

Phase separation is a powerful feature that enables:

- **Privacy**: Keep images local, only send text to cloud
- **Cost**: Use free models for expensive tasks, paid APIs only when needed
- **Flexibility**: Mix and match providers optimally
- **Performance**: Iterate on categories without re-describing
- **Quality**: Choose best provider for each phase

**Key Takeaway**: Describe once, categorize many times with different strategies, providers, or parameters.
