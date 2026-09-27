"""
HTML report generation for the image categorization system.

This module handles generating interactive HTML reports from categorization results.
"""

import os
import json
import textwrap
import argparse
import sys
import webbrowser
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

# Add parent directory to path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image_data import CategorizationResult, ImageData
from core.categories import categories_file_path


class HTMLGenerator:
    """Generates interactive HTML reports from categorization results."""
    
    def __init__(self, template_path: str = "template.html"):
        """
        Initialize HTML generator.
        
        Args:
            template_path: Path to HTML template file
        """
        self.template_path = template_path
    
    def generate_report(
        self, 
        directory: str, 
        result: CategorizationResult,
        output_filename: str = "image_categories.html"
    ) -> str:
        """
        Generate interactive HTML report from categorization results.
        
        Args:
            directory: Directory containing the images
            result: Categorization results
            output_filename: Name of output HTML file
            
        Returns:
            Path to generated HTML file
        """
        
        # Load the template
        if not os.path.exists(self.template_path):
            raise ValueError(f"Template file not found: {self.template_path}")
        
        with open(self.template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
        
        # Get current timestamp
        now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        # Prepare data for the template
        card_data = self._prepare_card_data(directory, result.images)
        categories_data = self._prepare_categories_data(
            result.category_groups, result.category_definitions)
        summary_html = self._generate_summary_html(directory, result, now)
        
        report_meta = {
            "directory": os.path.abspath(directory),
            "provider": result.processing_stats.get('provider', 'Unknown'),
            "model": result.processing_stats.get('model', 'Unknown'),
            "generated": datetime.now().isoformat(timespec='seconds'),
            "total_images": len(result.images),
            "categories_file": str(categories_file_path(os.getenv('CATEGORIES_FILE'))),
        }

        # Replace template placeholders
        html_content = template_content.replace("{{timestamp}}", now)
        html_content = html_content.replace("{{summary}}", summary_html)
        html_content = html_content.replace("{{card_data}}", self._json_for_script(card_data))
        html_content = html_content.replace("{{categories_data}}", self._json_for_script(categories_data))
        html_content = html_content.replace("{{report_meta}}", self._json_for_script(report_meta))
        
        # Write the HTML file
        output_path = os.path.join(directory, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return output_path
    
    @staticmethod
    def _json_for_script(value: Any) -> str:
        """Serialize to JSON that cannot close the surrounding <script> element."""
        return json.dumps(value).replace("</", "<\\/")

    def _prepare_card_data(self, directory: str, images: List[ImageData]) -> List[Dict[str, Any]]:
        """Prepare image card data for JavaScript."""
        card_data = []
        
        for image in images:
            try:
                # Use relative file path for HTML
                rel_filepath = image.filename
                
                card_data.append({
                    "filename": image.filename,
                    "description": image.description,
                    "primary_category": image.primary_category,
                    "suggested_categories": image.suggested_categories,  # All categories suggested by initial LLM
                    "thumbnail": rel_filepath,
                    "preview": rel_filepath,
                    "metadata": image.metadata
                })
                
            except Exception as e:
                print(f"Error processing image data for {image.filename}: {e}")
                card_data.append({
                    "filename": image.filename,
                    "description": image.description,
                    "primary_category": "Error",
                    "suggested_categories": ["Error"],
                    "thumbnail": "",
                    "preview": "",
                    "error": str(e)
                })
        
        return card_data
    
    def _prepare_categories_data(
        self,
        category_groups: Dict[str, List[str]],
        definitions: List[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Prepare category data for the template.

        Saved categories with no photos in this result are included, so saving the
        category list from the report does not drop them.
        """
        categories_data = []
        by_name = {d['name']: d for d in (definitions or [])}

        for category, files in category_groups.items():
            definition = by_name.pop(category, {})
            categories_data.append({
                "name": category,
                "count": len(files),
                "rule": definition.get("rule", ""),
                "status": definition.get("status", "")
            })
        for definition in by_name.values():
            categories_data.append({
                "name": definition["name"],
                "count": 0,
                "rule": definition.get("rule", ""),
                "status": definition.get("status", "")
            })
        
        # Ensure Trash category exists (for drag-and-drop functionality)
        if not any(cat["name"] == "Trash" for cat in categories_data):
            categories_data.append({"name": "Trash", "count": 0})
        
        # Sort categories alphabetically, but keep Trash at the end
        categories_data.sort(key=lambda x: ('zzz' if x["name"] == 'Trash' else x["name"].lower()))
        
        return categories_data
    
    def _generate_summary_html(self, directory: str, result: CategorizationResult, timestamp: str) -> str:
        """Generate the summary section HTML."""
        
        # Get statistics
        total_images = len(result.images)
        successful_descriptions = len([img for img in result.images if not img.description.startswith('Error')])
        categories_count = len(result.category_groups)
        
        # Get provider info from processing stats
        provider_info = result.processing_stats.get('provider', 'Unknown')
        model_info = result.processing_stats.get('model', 'Unknown')
        
        summary_html = textwrap.dedent(f"""
            <div class="summary">
                <div class="summary-content">
                    <div class="summary-main">
                        <h2>AI Image Analysis Results</h2>
                        <p><strong>Directory:</strong> {directory}</p>
                        <p><strong>Images processed:</strong> {total_images}</p>
                        <p><strong>Successful descriptions:</strong> {successful_descriptions}</p>
                        <p><strong>Provider:</strong> {provider_info} ({model_info})</p>
                        <p><strong>Generated on:</strong> {timestamp}</p>
                        <p class="summary-description">AI-powered image categorization using {provider_info} for both image descriptions and intelligent categorization. Images are analyzed and sorted into logical categories based on their content.</p>
                    </div>
        """).strip()
        
        # Add category statistics
        if result.category_groups:
            category_stats = []
            for category, files in result.category_groups.items():
                display_category = category
                if len(category) > 30:
                    display_category = category[:27] + "..."
                category_stats.append((display_category, len(files)))
            
            # Ensure Trash category is included
            if not any(category == 'Trash' for category, _ in category_stats):
                category_stats.append(('Trash', 0))
            
            # Sort categories
            category_stats.sort(key=lambda x: ('zzz' if x[0] == 'Trash' else x[0].lower()))
            
            summary_html += textwrap.dedent("""
                <div class="directory-structure">
                    <h3>AI-Generated Categories</h3>
                    <div class="category-stats">
            """).strip()
            
            for category, count in category_stats:
                extra_class = " trash-category-stat" if category == "Trash" else ""
                summary_html += textwrap.dedent(f"""
                    <div class="category-stat-item{extra_class}">
                        <span class="category-stat-name">{category}</span>
                        <span class="category-stat-count">{count}</span>
                    </div>
                """).strip()
            
            summary_html += textwrap.dedent("""
                    </div>
                </div>
            """).strip()
        
        summary_html += textwrap.dedent("""
                </div>
            </div>
        """).strip()
        
        return summary_html
    
    def generate_legacy_report(
        self,
        directory: str,
        results: List[Tuple[str, str, str]],
        group_categories: Dict[str, List[str]] = None,
        output_filename: str = "image_categories.html"
    ) -> str:
        """
        Generate report from legacy format for backward compatibility.
        
        Args:
            directory: Directory containing images
            results: List of (filename, description, category) tuples
            group_categories: Optional category groupings
            output_filename: Output HTML filename
            
        Returns:
            Path to generated HTML file
        """
        
        # Convert legacy format to new format
        images = []
        for filename, description, category in results:
            image_data = ImageData(
                filename=filename,
                filepath=os.path.join(directory, filename),
                description=description,
                suggested_categories=[category],  # Single category becomes the suggested category
                primary_category=category,  # Same category as primary
                metadata={}
            )
            images.append(image_data)
        
        # Create category groups if not provided
        if group_categories is None:
            group_categories = {}
            for image in images:
                category = image.primary_category
                if category not in group_categories:
                    group_categories[category] = []
                group_categories[category].append(image.filename)
        
        # Create CategorizationResult
        result = CategorizationResult(
            images=images,
            category_groups=group_categories,
            processing_stats={
                'provider': 'legacy',
                'total_images': len(images),
                'categories_generated': len(group_categories)
            }
        )
        
        return self.generate_report(directory, result, output_filename)


def main():
    """Standalone HTML generator CLI."""
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML report from categorization JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python core/html_generator.py /path/to/group_categories.json
              python core/html_generator.py ./images/results.json
        """)
    )
    
    parser.add_argument("json_file", 
                       help="Path to JSON file containing categorization results")
    
    args = parser.parse_args()
    
    # Validate JSON file exists
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    if not json_path.suffix.lower() == '.json':
        print(f"Error: File must be a JSON file: {json_path}")
        sys.exit(1)
    
    # Load and validate JSON data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)
    
    # Create CategorizationResult from JSON data
    try:
        result = CategorizationResult.from_dict(data)
    except Exception as e:
        print(f"Error parsing categorization data: {e}")
        sys.exit(1)
    
    # Generate HTML report in the same directory as JSON file
    output_dir = json_path.parent
    generator = HTMLGenerator()
    
    try:
        html_file = generator.generate_report(str(output_dir), result)
        if html_file:
            print(f"✓ HTML report generated: {html_file}")
            print("Opening HTML report in default browser...")
            webbrowser.open(f"file://{os.path.abspath(html_file)}")
        else:
            print("Error: Failed to generate HTML report")
            sys.exit(1)
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()