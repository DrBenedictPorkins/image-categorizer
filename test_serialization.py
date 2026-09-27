#!/usr/bin/env python3
"""
Test script to verify intermediate state serialization for phase separation.
"""

import json
from models.image_data import ImageData, CategorizationResult


def test_round_trip_serialization():
    """Test full round-trip serialization."""
    print("Testing full round-trip serialization...")

    # Create sample data
    images = [
        ImageData(
            filename="img1.jpg",
            filepath="/path/to/img1.jpg",
            description="A beautiful sunset over mountains",
            suggested_categories=["Nature", "Landscape", "Sunset"],
            primary_category="Landscape"
        ),
        ImageData(
            filename="img2.jpg",
            filepath="/path/to/img2.jpg",
            description="A cat sitting on a windowsill",
            suggested_categories=["Animals", "Pets", "Indoor"],
            primary_category="Pets"
        )
    ]

    result = CategorizationResult(
        images=images,
        processing_stats={"total": 2, "phase": "complete"}
    )

    # Serialize to dict
    data = result.to_dict()
    print(f"Serialized data keys: {data.keys()}")

    # Serialize to JSON string
    json_str = json.dumps(data, indent=2)
    print(f"JSON length: {len(json_str)} bytes\n")

    # Deserialize back
    loaded_data = json.loads(json_str)
    restored_result = CategorizationResult.from_dict(loaded_data)

    # Verify
    assert len(restored_result.images) == 2
    assert restored_result.images[0].filename == "img1.jpg"
    assert restored_result.images[0].description == "A beautiful sunset over mountains"
    assert restored_result.images[0].primary_category == "Landscape"
    assert len(restored_result.images[0].suggested_categories) == 3

    print("✓ Full round-trip serialization successful\n")


def test_descriptions_only_phase():
    """Test serialization with descriptions but no categorization."""
    print("Testing descriptions_only phase...")

    # Create sample data with only descriptions
    images = [
        ImageData(
            filename="img1.jpg",
            filepath="/path/to/img1.jpg",
            description="A beautiful sunset over mountains"
            # No categories set
        ),
        ImageData(
            filename="img2.jpg",
            filepath="/path/to/img2.jpg",
            description="A cat sitting on a windowsill"
            # No categories set
        )
    ]

    result = CategorizationResult(
        images=images,
        processing_stats={"total": 2, "phase": "descriptions_only"}
    )

    # Check phase detection
    phase = result.get_processing_phase()
    print(f"Detected phase: {phase}")
    assert phase == "descriptions_only"

    # Validate descriptions
    is_valid, missing = result.validate_descriptions()
    print(f"Descriptions valid: {is_valid}, Missing: {missing}")
    assert is_valid

    # Validate categorization (should fail)
    is_valid, missing = result.validate_categorization()
    print(f"Categorization valid: {is_valid}, Missing: {missing}")
    assert not is_valid
    assert len(missing) == 2

    # Save to JSON
    json_str = json.dumps(result.to_dict(), indent=2)
    print(f"Saved descriptions_only JSON ({len(json_str)} bytes)")

    # Load back
    loaded_data = json.loads(json_str)
    restored_result = CategorizationResult.from_dict(loaded_data)

    # Verify
    assert restored_result.images[0].has_description()
    assert not restored_result.images[0].has_categories()
    assert restored_result.images[0].primary_category == "Uncategorized"
    assert len(restored_result.images[0].suggested_categories) == 0

    print("✓ Descriptions-only phase serialization successful\n")


def test_categorization_phase_continuation():
    """Test loading descriptions-only state and adding categorization."""
    print("Testing categorization phase continuation...")

    # Simulate loading descriptions-only JSON
    descriptions_only_data = {
        "images": [
            {
                "filename": "img1.jpg",
                "filepath": "/path/to/img1.jpg",
                "description": "A beautiful sunset over mountains",
                "suggested_categories": [],
                "primary_category": "Uncategorized",
                "metadata": {}
            },
            {
                "filename": "img2.jpg",
                "filepath": "/path/to/img2.jpg",
                "description": "A cat sitting on a windowsill",
                "suggested_categories": [],
                "primary_category": "Uncategorized",
                "metadata": {}
            }
        ],
        "category_groups": {},
        "processing_stats": {"total": 2, "phase": "descriptions_only"}
    }

    # Load the state
    result = CategorizationResult.from_dict(descriptions_only_data)
    print(f"Loaded {len(result.images)} images")
    print(f"Phase: {result.get_processing_phase()}")

    # Verify descriptions exist
    is_valid, missing = result.validate_descriptions()
    assert is_valid
    print(f"✓ All descriptions present")

    # Now add categorization (simulating categorization phase)
    result.images[0].suggested_categories = ["Nature", "Landscape", "Sunset"]
    result.images[0].primary_category = "Landscape"

    result.images[1].suggested_categories = ["Animals", "Pets", "Indoor"]
    result.images[1].primary_category = "Pets"

    # Update category groups
    result._generate_category_groups()

    # Verify complete state
    assert result.get_processing_phase() == "complete"
    is_valid, missing = result.validate_categorization()
    assert is_valid
    print(f"✓ Categorization added successfully")

    # Save complete state
    complete_json = json.dumps(result.to_dict(), indent=2)
    print(f"Saved complete JSON ({len(complete_json)} bytes)")

    print("✓ Categorization phase continuation successful\n")


def test_validation_helpers():
    """Test validation helper methods."""
    print("Testing validation helpers...")

    # Test has_description
    img_with_desc = ImageData(
        filename="test.jpg",
        filepath="/path/test.jpg",
        description="Some description"
    )
    assert img_with_desc.has_description()
    assert not img_with_desc.has_categories()
    assert not img_with_desc.is_fully_processed()

    # Test empty description
    img_no_desc = ImageData(
        filename="test2.jpg",
        filepath="/path/test2.jpg",
        description=""
    )
    assert not img_no_desc.has_description()

    # Test whitespace-only description
    img_whitespace = ImageData(
        filename="test3.jpg",
        filepath="/path/test3.jpg",
        description="   \n  "
    )
    assert not img_whitespace.has_description()

    # Test has_categories
    img_with_cats = ImageData(
        filename="test4.jpg",
        filepath="/path/test4.jpg",
        description="Description",
        suggested_categories=["Cat1", "Cat2"]
    )
    assert img_with_cats.has_categories()

    # Test fully processed
    img_full = ImageData(
        filename="test5.jpg",
        filepath="/path/test5.jpg",
        description="Description",
        suggested_categories=["Cat1"],
        primary_category="Cat1"
    )
    assert img_full.is_fully_processed()

    print("✓ All validation helpers working correctly\n")


if __name__ == "__main__":
    try:
        test_round_trip_serialization()
        test_descriptions_only_phase()
        test_categorization_phase_continuation()
        test_validation_helpers()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe serialization system supports:")
        print("  1. Full round-trip serialization (complete state)")
        print("  2. Partial state serialization (descriptions only)")
        print("  3. Phase detection and validation")
        print("  4. Continuation from intermediate states")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
