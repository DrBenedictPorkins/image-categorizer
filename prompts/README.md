# Prompts Directory

This directory contains customizable prompt templates used by the image categorizer. You can modify these files to adjust how the AI analyzes and categorizes your images.

## Available Prompts

### `image_description.txt`
Controls how images are analyzed and described. This prompt is used for the initial image analysis phase.

**What you can customize:**
- **Categories list**: Modify the PREFERRED CATEGORIES section to add, remove, or rename categories
- **Instructions**: Adjust the INSTRUCTIONS section to emphasize different aspects (e.g., focus more on technical details, UI elements, etc.)
- **Description requirements**: Change what details should be included in descriptions
- **Tone and style**: Modify the prompt language to better suit your domain

**Important sections:**
- `PREFERRED CATEGORIES`: The list of categories that images will be sorted into
- `INSTRUCTIONS`: Detailed instructions for the AI on how to analyze images
- `<expected_response>`: The YAML format specification (keep this intact)

## How to Customize

1. **Back up the original**: Make a copy of the original prompt file before modifying
2. **Edit the prompt**: Modify the categories list and instructions to suit your needs
3. **Keep format intact**: Don't modify the `<expected_response>` YAML format section
4. **Test your changes**: Run the categorizer on a small set of test images to verify your changes work as expected

## Example Customizations

### For Software Screenshots
Update the PREFERRED CATEGORIES section:
```
- Bug Reports
- UI Mockups
- Error Dialogs
- Feature Demos
- Code Reviews
- API Documentation
```

And modify instructions to emphasize UI analysis:
```
- Identify specific UI components (buttons, menus, dialogs, etc.)
- Note any error messages or status indicators
- Describe the application or website context
```

### For Personal Photos
Replace categories with personal organization:
```
- Family Events
- Vacations
- Holidays
- Daily Life
- Pets
- Hobbies
- Friends
- Work
```

### For Research/Academic
Focus on academic content:
```
- Research Papers
- Data Visualizations
- Experimental Results
- Conference Presentations
- Literature Reviews
- Field Notes
```

## Troubleshooting

- **File not found errors**: Ensure the prompt file exists and has the correct name
- **Format errors**: Check that you haven't accidentally removed the `<expected_response>` YAML format section
- **Parsing errors**: Make sure the YAML format specification remains intact

## Restoring Defaults

To restore the original prompt, you can check the git history or reinstall the project.