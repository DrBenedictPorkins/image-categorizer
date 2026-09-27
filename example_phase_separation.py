#!/usr/bin/env python3
"""
Example demonstrating phase separation workflow:
1. Generate descriptions and save intermediate state
2. Load intermediate state and add categorization
"""

import json
from pathlib import Path
from models.image_data import ImageData, CategorizationResult


def phase_1_generate_descriptions():
    """Phase 1: Generate descriptions only."""
    print("=" * 60)
    print("PHASE 1: Generate Descriptions")
    print("=" * 60)

    # Simulate generating descriptions for images
    images = [
        ImageData(
            filename="sunset.jpg",
            filepath="/photos/sunset.jpg",
            description="A vibrant sunset over mountain peaks with orange and purple hues"
        ),
        ImageData(
            filename="cat.jpg",
            filepath="/photos/cat.jpg",
            description="A gray tabby cat lounging on a sunny windowsill"
        ),
        ImageData(
            filename="coffee.jpg",
            filepath="/photos/coffee.jpg",
            description="A steaming cup of coffee on a wooden table with morning light"
        ),
    ]

    result = CategorizationResult(
        images=images,
        processing_stats={
            "total_images": len(images),
            "phase": "descriptions_only",
            "timestamp": "2025-11-04T10:00:00"
        }
    )

    # Validate phase
    phase = result.get_processing_phase()
    is_valid, missing = result.validate_descriptions()

    print(f"\nProcessing phase: {phase}")
    print(f"Descriptions valid: {is_valid}")
    print(f"Images processed: {len(images)}")

    # Save intermediate state
    output_file = "descriptions_only.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\n✓ Saved intermediate state to: {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size} bytes")

    return output_file


def phase_2_add_categorization(input_file: str):
    """Phase 2: Load descriptions and add categorization."""
    print("\n" + "=" * 60)
    print("PHASE 2: Add Categorization")
    print("=" * 60)

    # Load intermediate state
    print(f"\nLoading state from: {input_file}")
    with open(input_file) as f:
        data = json.load(f)

    result = CategorizationResult.from_dict(data)

    # Verify we have descriptions
    phase = result.get_processing_phase()
    is_valid, missing = result.validate_descriptions()

    print(f"Processing phase: {phase}")
    print(f"Descriptions loaded: {is_valid}")
    print(f"Images to categorize: {len(result.images)}")

    # Simulate categorization (normally would call LLM here)
    print("\nCategorizing images...")

    # Example categorization logic
    category_mapping = {
        "sunset.jpg": (["Nature", "Landscape", "Photography"], "Landscape"),
        "cat.jpg": (["Animals", "Pets", "Indoor"], "Pets"),
        "coffee.jpg": (["Food & Drink", "Lifestyle", "Still Life"], "Food & Drink"),
    }

    for image in result.images:
        if image.filename in category_mapping:
            suggested, primary = category_mapping[image.filename]
            image.suggested_categories = suggested
            image.primary_category = primary
            print(f"  {image.filename}: {primary} (+ {len(suggested)-1} alternatives)")

    # Regenerate category groups
    result._generate_category_groups()

    # Update processing stats
    result.processing_stats.update({
        "phase": "complete",
        "categorization_timestamp": "2025-11-04T10:30:00"
    })

    # Validate complete state
    phase = result.get_processing_phase()
    is_valid, missing = result.validate_categorization()

    print(f"\nProcessing phase: {phase}")
    print(f"Categorization valid: {is_valid}")

    # Save complete state
    output_file = "categorization_complete.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\n✓ Saved complete state to: {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size} bytes")

    # Display summary
    print("\nCategory Summary:")
    for category, count in result.get_category_stats().items():
        print(f"  {category}: {count} image(s)")

    return output_file


def verify_final_state(complete_file: str):
    """Verify the final complete state."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Final State")
    print("=" * 60)

    with open(complete_file) as f:
        data = json.load(f)

    result = CategorizationResult.from_dict(data)

    print(f"\nTotal images: {len(result.images)}")
    print(f"Processing phase: {result.get_processing_phase()}")

    # Check each image
    print("\nImage details:")
    for img in result.images:
        print(f"\n  {img.filename}:")
        print(f"    Description: {img.description[:50]}...")
        print(f"    Primary: {img.primary_category}")
        print(f"    Suggested: {', '.join(img.suggested_categories)}")
        print(f"    Fully processed: {img.is_fully_processed()}")

    # Final validation
    desc_valid, _ = result.validate_descriptions()
    cat_valid, _ = result.validate_categorization()

    print(f"\n✓ Descriptions valid: {desc_valid}")
    print(f"✓ Categorization valid: {cat_valid}")
    print(f"✓ All images fully processed: {all(img.is_fully_processed() for img in result.images)}")


if __name__ == "__main__":
    print("\nDemonstrating Phase Separation Workflow")
    print("This example shows how to split processing into two phases:\n")

    # Phase 1: Generate descriptions
    intermediate_file = phase_1_generate_descriptions()

    # Phase 2: Add categorization
    complete_file = phase_2_add_categorization(intermediate_file)

    # Verify final state
    verify_final_state(complete_file)

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {intermediate_file} - Descriptions only (intermediate)")
    print(f"  2. {complete_file} - Complete categorization (final)")
    print("\nThis workflow enables:")
    print("  - Separating expensive operations (BLIP vs LLM)")
    print("  - Resumable processing from intermediate states")
    print("  - Independent scaling of description/categorization phases")
    print("  - Easier debugging and testing of each phase")
