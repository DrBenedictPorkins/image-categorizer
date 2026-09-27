# Phase Separation Serialization Guide

## Overview

The `models/image_data.py` module supports intermediate state serialization, enabling phase separation between description generation (BLIP) and categorization (LLM).

## Key Features

### 1. Round-Trip Serialization
- Full serialization/deserialization of complete and partial states
- Preserves all metadata and processing statistics
- Backward compatible with legacy formats

### 2. Phase Detection
- Automatic detection of processing phase: `empty`, `descriptions_only`, `partial`, or `complete`
- Validation methods for descriptions and categorization
- Helper methods to check individual image states

### 3. Flexible Data Model
- `ImageData`: Represents a single image with optional description and categorization
- `CategorizationResult`: Container for all images with processing stats
- Default values enable partial state representation

## API Reference

### ImageData Methods

```python
# Validation helpers
image.has_description() -> bool          # Check if description exists
image.has_categories() -> bool           # Check if categorized
image.is_fully_processed() -> bool       # Check if both exist

# Serialization
image.to_dict() -> dict                  # Serialize to dictionary
ImageData.from_dict(data) -> ImageData   # Deserialize from dictionary
```

### CategorizationResult Methods

```python
# Phase detection
result.get_processing_phase() -> str
# Returns: 'empty', 'descriptions_only', 'partial', or 'complete'

# Validation
result.validate_descriptions() -> tuple[bool, List[str]]
# Returns: (is_valid, list of filenames missing descriptions)

result.validate_categorization() -> tuple[bool, List[str]]
# Returns: (is_valid, list of filenames missing categorization)

# Serialization
result.to_dict() -> dict                              # Serialize to dictionary
CategorizationResult.from_dict(data) -> CategorizationResult  # Deserialize
```

## Usage Examples

### Basic Round-Trip

```python
import json
from models.image_data import CategorizationResult

# Save
with open("output.json", "w") as f:
    json.dump(result.to_dict(), f)

# Load
with open("output.json") as f:
    data = json.load(f)
result = CategorizationResult.from_dict(data)
```

### Phase 1: Save Descriptions Only

```python
from models.image_data import ImageData, CategorizationResult

# Create images with descriptions but no categories
images = [
    ImageData(
        filename="photo.jpg",
        filepath="/path/to/photo.jpg",
        description="A beautiful landscape"
        # No categories set - will default to empty list and "Uncategorized"
    )
]

result = CategorizationResult(
    images=images,
    processing_stats={"phase": "descriptions_only"}
)

# Validate before saving
is_valid, missing = result.validate_descriptions()
if is_valid:
    with open("descriptions_only.json", "w") as f:
        json.dump(result.to_dict(), f)
```

### Phase 2: Load and Add Categorization

```python
import json
from models.image_data import CategorizationResult

# Load intermediate state
with open("descriptions_only.json") as f:
    data = json.load(f)
result = CategorizationResult.from_dict(data)

# Verify phase
phase = result.get_processing_phase()
assert phase == "descriptions_only"

# Add categorization
for image in result.images:
    # Call LLM to categorize based on description
    image.suggested_categories = ["Category1", "Category2", "Category3"]
    image.primary_category = "Category1"

# Regenerate category groups
result._generate_category_groups()

# Validate and save
is_valid, missing = result.validate_categorization()
if is_valid:
    with open("categorization_complete.json", "w") as f:
        json.dump(result.to_dict(), f)
```

### Validation and Error Handling

```python
# Check processing phase
phase = result.get_processing_phase()
print(f"Current phase: {phase}")

# Validate descriptions
is_valid, missing_files = result.validate_descriptions()
if not is_valid:
    print(f"Missing descriptions for: {', '.join(missing_files)}")

# Validate categorization
is_valid, missing_files = result.validate_categorization()
if not is_valid:
    print(f"Missing categorization for: {', '.join(missing_files)}")

# Check individual images
for image in result.images:
    if not image.is_fully_processed():
        print(f"{image.filename} is incomplete")
        print(f"  Has description: {image.has_description()}")
        print(f"  Has categories: {image.has_categories()}")
```

## JSON Format Examples

### Descriptions Only State

```json
{
  "images": [
    {
      "filename": "photo.jpg",
      "filepath": "/path/to/photo.jpg",
      "description": "A beautiful landscape",
      "suggested_categories": [],
      "primary_category": "Uncategorized",
      "metadata": {}
    }
  ],
  "category_groups": {
    "Uncategorized": ["photo.jpg"]
  },
  "processing_stats": {
    "phase": "descriptions_only",
    "timestamp": "2025-11-04T10:00:00"
  }
}
```

### Complete State

```json
{
  "images": [
    {
      "filename": "photo.jpg",
      "filepath": "/path/to/photo.jpg",
      "description": "A beautiful landscape",
      "suggested_categories": ["Nature", "Landscape", "Photography"],
      "primary_category": "Landscape",
      "metadata": {}
    }
  ],
  "category_groups": {
    "Landscape": ["photo.jpg"]
  },
  "processing_stats": {
    "phase": "complete",
    "description_timestamp": "2025-11-04T10:00:00",
    "categorization_timestamp": "2025-11-04T10:30:00"
  }
}
```

## Benefits of Phase Separation

1. **Resource Optimization**: Run BLIP locally, LLM remotely
2. **Resumable Processing**: Save progress and resume later
3. **Independent Scaling**: Scale description and categorization separately
4. **Debugging**: Test each phase independently
5. **Cost Control**: Generate descriptions once, experiment with categorization
6. **Flexibility**: Use different LLM providers for categorization without regenerating descriptions

## Testing

Run the test suite to verify serialization:

```bash
# Unit tests
uv run python test_serialization.py

# Phase separation example
uv run python example_phase_separation.py
```

## Backward Compatibility

The data model maintains backward compatibility with older formats:
- Legacy `category` field maps to `primary_category`
- Legacy `categories` field maps to `suggested_categories`
- Missing fields default to safe values (empty strings, empty lists, "Uncategorized")
