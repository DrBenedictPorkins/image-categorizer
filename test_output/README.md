# Test Output Directory

This directory contains the results of phase separation workflow tests.

## Directory Structure

Each test creates its own subdirectory with test results:

```
test_output/
├── test_1_fully_local/             # HuggingFace → HuggingFace (fully local)
│   └── categorization_results.json
├── test_2_hybrid_hf_ollama/        # HuggingFace → Ollama (hybrid)
│   └── categorization_results.json
├── test_3_cloud_baseline/          # Ollama → Ollama (baseline)
│   └── categorization_results.json
├── test_4_description_only/        # Description phase only
│   ├── descriptions.json           # Intermediate results
│   └── categorization_results.json
├── test_5_categorization_only/     # Categorization phase only
│   └── categorization_results.json
├── test_6_keyword_categorization/  # HuggingFace → Keyword matching
│   └── categorization_results.json
└── test_7_backward_compat/         # Traditional single provider
    └── categorization_results.json
```

## Result File Format

### categorization_results.json

Complete categorization results including images, categories, and stats:

```json
{
  "images": [
    {
      "filename": "example.jpg",
      "filepath": "/path/to/example.jpg",
      "description": "Detailed image description",
      "suggested_categories": ["Category1", "Category2", "Category3"],
      "primary_category": "Category1",
      "metadata": {
        "provider": "huggingface",
        "vision_model": "Salesforce/blip2-flan-t5-xl-coco"
      }
    }
  ],
  "category_groups": {
    "Category1": ["example1.jpg", "example2.jpg"],
    "Category2": ["example3.jpg"]
  },
  "processing_stats": {
    "provider": "huggingface",
    "total_images": 5,
    "categories_generated": 3
  }
}
```

### descriptions.json (Test 4 only)

Intermediate results from description phase:

```json
{
  "images": [
    {
      "filename": "example.jpg",
      "filepath": "/path/to/example.jpg",
      "description": "Detailed image description",
      "suggested_categories": ["Category1", "Category2", "Category3"],
      "primary_category": "Uncategorized",
      "metadata": {}
    }
  ],
  "metadata": {
    "provider": "huggingface",
    "phase": "description_only",
    "total_images": 5
  }
}
```

## Comparing Results

To compare the quality of different workflows:

1. **Check Category Distribution:**
   ```bash
   jq '.category_groups' test_*/categorization_results.json
   ```

2. **Compare Descriptions:**
   ```bash
   jq '.images[0].description' test_*/categorization_results.json
   ```

3. **Check Processing Stats:**
   ```bash
   jq '.processing_stats' test_*/categorization_results.json
   ```

4. **Count Categories:**
   ```bash
   jq '.category_groups | keys | length' test_*/categorization_results.json
   ```

## Cleaning Up

To remove all test results and start fresh:

```bash
rm -rf test_output/*/
```

Or remove specific test results:

```bash
rm -rf test_output/test_1_fully_local/
```
