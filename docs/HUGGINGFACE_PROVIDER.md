# HuggingFace Provider Documentation

## Overview

The HuggingFace provider enables fully local image categorization using open-source models from HuggingFace. All processing happens on your machine without any external API calls.

## Features

- **Fully Local**: No internet required after models are downloaded
- **No API Keys**: No costs or rate limits
- **Multiple Models**: Support for various vision and text models
- **Device Flexibility**: Automatic detection of MPS (Apple Silicon), CUDA, or CPU
- **Full Precision**: Uses float32 for maximum quality
- **Two-Phase Processing**: Separate description and categorization phases
- **Auto-Download**: Models are automatically downloaded on first use

## Supported Models

### Vision Models (for image description)

| Model | Model ID | Size | Description |
|-------|----------|------|-------------|
| **BLIP-2** | `Salesforce/blip2-flan-t5-xl-coco` | 15GB | Default - High quality descriptions |
| **LLaVA-1.5** | `llava-hf/llava-1.5-7b-hf` | 13GB | Conversational vision model |
| **BLIP-base** | `Salesforce/blip-image-captioning-base` | 2GB | Fast and lightweight |
| **MiniCPM-V** | `openbmb/MiniCPM-V-2` | 8GB | Efficient and accurate |

### Text Models (for categorization)

| Model | Model ID | Size | Description |
|-------|----------|------|-------------|
| **Flan-T5-XL** | `google/flan-t5-xl` | 3GB | Default - Good quality text generation |
| **Phi-2** | `microsoft/phi-2` | 5GB | Powerful small language model |

## Installation

### 1. Install Required Packages

```bash
uv add transformers torch pillow
```

### 2. Install PyTorch with Device Support

**For Apple Silicon (M1/M2/M3 MAX):**
```bash
# PyTorch with MPS support (already included in standard torch)
uv add torch
```

**For NVIDIA GPU (CUDA):**
```bash
# Install PyTorch with CUDA support
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
# Standard PyTorch (CPU)
uv add torch
```

## Configuration

### Environment Variables

Configure the provider using environment variables:

```bash
# Model Selection
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"  # Default
export HF_TEXT_MODEL="google/flan-t5-xl"                   # Default

# Device Selection (auto/mps/cuda/cpu)
export HF_DEVICE="auto"  # Default - auto-detects best device

# Cache Directory (optional)
export HF_CACHE_DIR="/path/to/cache"  # Default: ~/.cache/huggingface

# HuggingFace Token (optional, for gated models)
export HUGGINGFACE_TOKEN="hf_..."
```

### Model Selection Examples

**Fast Processing (5GB total):**
```bash
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
export HF_TEXT_MODEL="google/flan-t5-xl"
```

**Best Quality (18GB total):**
```bash
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"
export HF_TEXT_MODEL="microsoft/phi-2"
```

**Balanced (21GB total):**
```bash
export HF_VISION_MODEL="llava-hf/llava-1.5-7b-hf"
export HF_TEXT_MODEL="microsoft/phi-2"
```

**Memory Efficient (10GB total):**
```bash
export HF_VISION_MODEL="openbmb/MiniCPM-V-2"
export HF_TEXT_MODEL="google/flan-t5-xl"
```

## Usage

### Basic Usage

```bash
python main.py /path/to/images --provider huggingface
```

### With Custom Models

```bash
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
export HF_TEXT_MODEL="google/flan-t5-xl"
python main.py /path/to/images --provider huggingface
```

### With Initial Categories

```bash
python main.py /path/to/images --provider huggingface --categories "Nature,People,Food,Technology"
```

### Testing the Provider

```bash
# Basic test (models and device detection)
python test_huggingface_provider.py

# Full test with an image
python test_huggingface_provider.py /path/to/test/image.jpg
```

### Listing Available Models

```bash
python -c "from providers.huggingface_provider import list_available_models; list_available_models()"
```

## Device Detection

The provider automatically detects the best available device:

1. **MPS** (Metal Performance Shaders) - Apple Silicon (M1/M2/M3)
2. **CUDA** - NVIDIA GPUs
3. **CPU** - Fallback for all systems

### Manual Device Selection

```bash
# Force CPU (useful for debugging)
export HF_DEVICE="cpu"

# Force MPS (Apple Silicon)
export HF_DEVICE="mps"

# Force CUDA (NVIDIA GPU)
export HF_DEVICE="cuda"
```

## Performance Considerations

### Hardware Requirements

**Minimum:**
- RAM: 8GB
- Storage: 5GB free (for smallest models)
- CPU: Multi-core recommended

**Recommended:**
- RAM: 16GB+ (for larger models)
- Storage: 20GB+ free (for all models)
- GPU: Apple M1/M2/M3 or NVIDIA with 8GB+ VRAM

**Optimal:**
- RAM: 32GB+
- Storage: 50GB+ free
- GPU: Apple M3 MAX or NVIDIA RTX 4090

### Speed Comparison

Processing 10 images (approximate):

| Configuration | Device | Time |
|--------------|--------|------|
| BLIP-base + Flan-T5 | M3 MAX (MPS) | ~2 min |
| BLIP-2 + Flan-T5 | M3 MAX (MPS) | ~5 min |
| BLIP-base + Flan-T5 | CPU (8 cores) | ~10 min |
| BLIP-2 + Flan-T5 | CPU (8 cores) | ~20 min |

### Memory Usage

| Model Configuration | RAM Usage | VRAM Usage |
|--------------------|-----------|------------|
| BLIP-base + Flan-T5 | ~5GB | ~3GB |
| BLIP-2 + Flan-T5 | ~18GB | ~12GB |
| LLaVA + Phi-2 | ~18GB | ~13GB |
| MiniCPM + Flan-T5 | ~11GB | ~7GB |

## Troubleshooting

### Models Not Downloading

```bash
# Ensure you have HuggingFace CLI installed
uv add huggingface-hub

# Login if using gated models
huggingface-cli login
```

### Out of Memory Errors

1. Use smaller models:
   ```bash
   export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
   export HF_TEXT_MODEL="google/flan-t5-xl"
   ```

2. Force CPU mode:
   ```bash
   export HF_DEVICE="cpu"
   ```

3. Clear cache between runs:
   ```bash
   rm -rf ~/.cache/huggingface/hub
   ```

### MPS Not Working on Mac

```bash
# Check if MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, use CPU
export HF_DEVICE="cpu"
```

### Slow Performance

1. Ensure you're using GPU acceleration (MPS or CUDA)
2. Use smaller models for faster processing
3. Close other applications to free up memory
4. Consider processing images in smaller batches

## Advanced Usage

### Custom Model Integration

You can use any HuggingFace model by providing the model ID:

```bash
# Use a custom vision model
export HF_VISION_MODEL="custom/vision-model-id"

# Use a custom text model
export HF_TEXT_MODEL="custom/text-model-id"
```

### Programmatic Usage

```python
from providers.huggingface_provider import HuggingFaceProvider
from models.image_data import ProviderConfig

# Create configuration
config = ProviderConfig(
    provider_name='huggingface',
    settings={
        'vision_model': 'Salesforce/blip-image-captioning-base',
        'text_model': 'google/flan-t5-xl',
        'device': 'auto',
        'cache_dir': None,
        'hf_token': None
    }
)

# Initialize provider
provider = HuggingFaceProvider(config)
provider.initialize()

# Process images
result = provider.process_images(['/path/to/image1.jpg', '/path/to/image2.jpg'])

# Access results
for image in result.images:
    print(f"{image.filename}: {image.primary_category}")
    print(f"  Description: {image.description}")
    print(f"  Suggested: {image.suggested_categories}")

# Cleanup
provider.cleanup()
```

### Phase Separation

Process description and categorization separately:

```python
# Phase 1: Description only
image_data_list = provider.describe_images(image_paths)

# Save descriptions to file or process further...

# Phase 2: Categorization only
result = provider.categorize_described_images(image_data_list)
```

## Comparison with Other Providers

| Feature | HuggingFace | Ollama | Anthropic | OpenAI |
|---------|------------|---------|-----------|---------|
| **Cost** | Free | Free | Paid | Paid |
| **Privacy** | Fully local | Local | Cloud | Cloud |
| **Internet** | Download only | Local | Required | Required |
| **Quality** | Good | Good | Excellent | Excellent |
| **Speed** | Medium | Fast | Fast | Fast |
| **GPU Support** | Yes | Yes | N/A | N/A |
| **Model Choice** | Many options | Limited | Fixed | Fixed |

## Best Practices

1. **First Run**: Allow extra time for model downloads
2. **Model Selection**: Start with small models, upgrade if needed
3. **Memory Management**: Close other applications when processing
4. **Batch Size**: Process 10-50 images at a time
5. **Device Selection**: Use auto-detection unless troubleshooting
6. **Cache Directory**: Use default location unless low on disk space
7. **Testing**: Run test script before processing large batches

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [Flan-T5 Paper](https://arxiv.org/abs/2210.11416)
- [Phi-2 Blog Post](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
