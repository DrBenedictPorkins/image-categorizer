# Keyword Categorization Provider - Usage Guide

## Overview

The `KeywordCategorizationProvider` is a fast, offline, deterministic categorization provider that uses keyword matching to assign categories to images. It does **not** support image description - you must use another provider (like BLIP or Ollama) to generate descriptions first.

## Key Features

- **No LLM Required**: Pure keyword matching, no API calls
- **Offline**: Works without internet connection
- **Fast**: Processes images instantly
- **Deterministic**: Same description always produces same category
- **No Configuration**: No API keys or servers needed

## Capabilities

### Supported Phases
- ✅ **Categorization**: Assigns categories based on description keywords
- ❌ **Description**: Cannot generate image descriptions
- ❌ **Vision**: Cannot process raw images

### Built-in Categories

The provider recognizes these categories based on keywords:

1. **People**: person, people, man, woman, child, human, face, portrait
2. **Nature**: tree, forest, mountain, sky, flower, landscape, outdoor, nature
3. **Food**: food, meal, dish, cooking, restaurant, kitchen
4. **Animals**: animal, dog, cat, pet, wildlife, bird, fish
5. **Architecture**: building, house, structure, bridge, architecture
6. **Vehicles**: car, vehicle, truck, bike, motorcycle, transportation
7. **Technology**: computer, phone, device, screen, electronic
8. **Art**: art, painting, drawing, sculpture, artwork
9. **Uncategorized**: Fallback when no keywords match

## Usage Examples

### Basic Usage (Categorize Pre-Described Images)

```python
from providers.keyword_provider import KeywordCategorizationProvider
from models.image_data import ImageData, ProviderConfig

# Create provider
config = ProviderConfig(provider_name="keyword")
provider = KeywordCategorizationProvider(config)
provider.initialize()

# Create images with descriptions (from another source)
images = [
    ImageData(
        filename="portrait.jpg",
        filepath="/path/to/portrait.jpg",
        description="A person smiling at the camera, outdoor portrait",
        suggested_categories=[],
        primary_category="",
        metadata={}
    )
]

# Categorize
result = provider.categorize_described_images(images)

# Access results
for img in result.images:
    print(f"{img.filename}: {img.primary_category}")
```

### Hybrid Workflow (Vision Model + Keyword Categorization)

```python
from providers.ollama_provider import OllamaProvider
from providers.keyword_provider import KeywordCategorizationProvider
from models.image_data import ProviderConfig

# Step 1: Use Ollama to describe images
ollama_config = ProviderConfig(
    provider_name="ollama",
    settings={"host": "http://localhost:11434", "model": "llava:latest"}
)
ollama = OllamaProvider(ollama_config)
ollama.initialize()

# Get descriptions only
images_with_descriptions = ollama.describe_images(image_paths)

# Step 2: Use keyword provider for fast categorization
keyword_config = ProviderConfig(provider_name="keyword")
keyword = KeywordCategorizationProvider(keyword_config)
keyword.initialize()

# Categorize
result = keyword.categorize_described_images(images_with_descriptions)
```

### Error Handling

```python
try:
    # This will fail - keyword provider can't process raw images
    result = provider.process_images(image_paths)
except NotImplementedError as e:
    print(f"Expected error: {e}")

try:
    # This will fail - images need descriptions
    images_without_descriptions = [
        ImageData(
            filename="test.jpg",
            filepath="/path/to/test.jpg",
            description="",  # Empty!
            suggested_categories=[],
            primary_category="",
            metadata={}
        )
    ]
    result = provider.categorize_described_images(images_without_descriptions)
except ProviderProcessingError as e:
    print(f"Error: {e}")
```

## How It Works

### Categorization Strategy

1. **Check Suggested Categories**: If the image already has suggested categories from the description phase, use the first one (if it matches a known category)
2. **Keyword Matching**: Count keyword occurrences in the description for each category
3. **Best Match**: Select the category with the most keyword matches
4. **Fallback**: If no keywords match, assign "Uncategorized"

### Example Matching

**Description**: "A golden retriever dog playing in a park with people watching"

**Keyword Counts**:
- Animals: 2 matches (dog, pet implied)
- People: 1 match (people)
- Nature: 1 match (park)

**Result**: Animals (highest match count)

## Use Cases

### 1. Fast Local Processing
When you need instant categorization without waiting for LLM inference:
```bash
# Generate descriptions with vision model
python main.py /path/to/images --provider ollama --save-json descriptions.json

# Later: Fast re-categorization with keywords
python main.py /path/to/images --json descriptions.json --provider keyword
```

### 2. Testing and Development
Test your categorization pipeline without waiting for slow LLM responses:
```python
# Quick test of image processing pipeline
test_images = create_test_images_with_descriptions()
result = keyword_provider.categorize_described_images(test_images)
assert len(result.images) == len(test_images)
```

### 3. Fallback Mechanism
Use as a fallback when primary provider fails:
```python
try:
    result = primary_provider.process_images(images)
except ProviderError:
    # Fall back to keyword categorization
    descriptions = get_cached_descriptions(images)
    result = keyword_provider.categorize_described_images(descriptions)
```

### 4. Privacy-Conscious Workflows
Generate descriptions locally, categorize with keywords (no cloud API calls):
```python
# Use local BLIP for descriptions
blip_descriptions = generate_blip_descriptions(images)

# Use keyword provider for categorization (fully offline)
result = keyword_provider.categorize_described_images(blip_descriptions)
```

## Limitations

1. **No Vision**: Cannot process raw images
2. **Fixed Categories**: Limited to predefined category keywords
3. **Simple Logic**: No context understanding, just keyword matching
4. **English Only**: Keywords are in English
5. **No Learning**: Cannot adapt to new categories without code changes

## Extending the Provider

To add custom categories, modify the `CATEGORY_KEYWORDS` dictionary:

```python
from providers.keyword_provider import KeywordCategorizationProvider

class CustomKeywordProvider(KeywordCategorizationProvider):
    CATEGORY_KEYWORDS = {
        **KeywordCategorizationProvider.CATEGORY_KEYWORDS,
        "Sports": ["sport", "game", "player", "ball", "field", "court"],
        "Music": ["music", "instrument", "concert", "band", "guitar", "piano"]
    }
```

## Performance

- **Speed**: ~1000 images/second (description-only processing)
- **Memory**: Minimal (no model loading)
- **CPU**: Very low usage
- **Network**: None required

## Comparison with Other Providers

| Feature | Keyword | Ollama | Anthropic |
|---------|---------|--------|-----------|
| Description | ❌ | ✅ | ❌ (not implemented) |
| Categorization | ✅ | ✅ | ❌ (not implemented) |
| Speed | Very Fast | Slow | N/A |
| Offline | ✅ | ✅ | ❌ |
| API Key Required | ❌ | ❌ | ✅ |
| Cost | Free | Free | Paid |
| Accuracy | Basic | High | N/A |

## See Also

- `test_keyword_provider.py` - Complete test examples
- `providers/base.py` - Base provider interface
- `providers/ollama_provider.py` - Full vision+categorization provider
