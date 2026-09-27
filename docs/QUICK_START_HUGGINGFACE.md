# HuggingFace Provider Quick Start

## 1-Minute Setup

```bash
# 1. Install dependencies
uv sync

# 2. Set environment variables (optional - defaults work fine)
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"  # Fastest (2GB)
export HF_TEXT_MODEL="google/flan-t5-xl"                        # Default (3GB)

# 3. Run image categorization
python main.py /path/to/images --provider huggingface
```

## Quick Examples

### Fast Processing (5GB total, ~2 min for 10 images on M3 MAX)
```bash
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
export HF_TEXT_MODEL="google/flan-t5-xl"
python main.py ~/Pictures --provider huggingface
```

### Best Quality (18GB total, ~5 min for 10 images on M3 MAX)
```bash
export HF_VISION_MODEL="Salesforce/blip2-flan-t5-xl-coco"
export HF_TEXT_MODEL="microsoft/phi-2"
python main.py ~/Pictures --provider huggingface
```

### With Initial Categories
```bash
python main.py ~/Pictures --provider huggingface \
  --categories "Vacation,Family,Work,Screenshots,Food"
```

## Available Models

### Vision Models (Image Description)
- `Salesforce/blip-image-captioning-base` - 2GB, **fastest**
- `openbmb/MiniCPM-V-2` - 8GB, efficient
- `llava-hf/llava-1.5-7b-hf` - 13GB, conversational
- `Salesforce/blip2-flan-t5-xl-coco` - 15GB, **default**, high quality

### Text Models (Categorization)
- `google/flan-t5-xl` - 3GB, **default**, good quality
- `microsoft/phi-2` - 5GB, powerful

## Common Issues

### Out of Memory?
```bash
# Use smallest models
export HF_VISION_MODEL="Salesforce/blip-image-captioning-base"
export HF_DEVICE="cpu"
python main.py /path/to/images --provider huggingface
```

### First Run Taking Forever?
The models are being downloaded (5-20GB). Be patient! Subsequent runs will be fast.

### MPS Not Available on Mac?
```bash
# Fall back to CPU
export HF_DEVICE="cpu"
```

## Test Before Use

```bash
# Basic functionality test
python test_huggingface_provider.py

# Test with a real image
python test_huggingface_provider.py /path/to/test/image.jpg
```

## What Gets Installed?

Models are downloaded to `~/.cache/huggingface/hub/` and reused across runs.

| Model | Size | Description |
|-------|------|-------------|
| BLIP-base | 2GB | Vision model (fast) |
| BLIP-2 | 15GB | Vision model (quality) |
| Flan-T5-XL | 3GB | Text model (default) |
| Phi-2 | 5GB | Text model (powerful) |

## Performance Tips

1. **Use GPU**: Auto-detected (MPS on Mac, CUDA on Linux/Windows)
2. **Start Small**: Test with blip-base first
3. **Upgrade Later**: Switch to BLIP-2 if you need better descriptions
4. **Batch Processing**: Process 10-50 images at a time
5. **Free Memory**: Close other apps before running

## Full Documentation

See [HUGGINGFACE_PROVIDER.md](./HUGGINGFACE_PROVIDER.md) for complete documentation.
