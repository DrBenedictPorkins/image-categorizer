"""
Ollama LLM provider for image categorization - FIXED VERSION.

This provider uses the REST API directly instead of the Python client
to avoid the hanging issue with vision models.
"""

import json
import base64
import requests
import textwrap
import yaml
import time
from collections import Counter
from io import BytesIO
from typing import List, Dict, Any, Callable, Optional
from pathlib import Path

from .base import BaseLLMProvider, ProviderError, ProviderConnectionError, ProviderProcessingError
from models.image_data import ImageData, CategorizationResult, ProviderConfig
from core.image_processor import ImageProcessor
from core.categories import STATUS_NEW, categories_file_path, load_categories, save_categories

# Category for blurry, obstructed or otherwise failed shots
ACCIDENTAL_CATEGORY = "Accidental Shots"
ACCIDENTAL_RULE = "Images that are accidental, failed, blurry or unusable, whatever they show."
# Most categories the model may add to the saved list in one run
MAX_NEW_CATEGORIES_PER_RUN = 5
# Answer offered in Phase 2 when a photo fits none of the saved categories
NONE_OPTION = "None of these"
# Where photos go that fit no category; never saved to the category file
UNSORTED_CATEGORY = "Unsorted"
# Most unmatched photos shown to the model when proposing new categories
PROPOSE_SAMPLE_SIZE = 60

# Phase 2 sends at most this many images per request; larger sets first get a
# fixed category list, then are assigned to it batch by batch
CATEGORIZE_BATCH_SIZE = 40
# Context window for Phase 2 requests (Ollama's default is too small for a batch)
CATEGORIZE_NUM_CTX = 16384
# Most frequent suggested categories shown to the model when building the list
TAXONOMY_TOP_SUGGESTIONS = 300
# Extra Phase 2 requests per batch for images left unassigned or off-list
CATEGORIZE_REASK_ROUNDS = 2


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for image description and categorization using REST API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.host = None
        self.model = None
        self.text_model = None
        self.timeout = 300
        self.max_retries = 2
        self.retry_delay = 1.0
        self.categories_file = None
        # Category list used by the last categorization, passed to the report
        self.category_definitions: List[Dict[str, str]] = []

    @property
    def provider_name(self) -> str:
        return "Ollama"

    @property
    def required_config_keys(self) -> List[str]:
        return ['host', 'model']

    @property
    def supports_vision(self) -> bool:
        return True

    def initialize(self) -> bool:
        """Initialize the Ollama provider."""
        try:
            self.host = self.config.get('host', 'http://localhost:11434')
            # Use llava:latest as default - llama3.2-vision models are currently broken/hanging
            self.model = self.config.get('model', 'llava:latest')
            self.text_model = self.config.get('text_model', 'llama3.2:latest')
            self.timeout = self.config.get('timeout', 300)
            self.max_retries = self.config.get('max_retries', 2)
            self.retry_delay = self.config.get('retry_delay', 1.0)
            self.categories_file = self.config.get('categories_file')

            # Ensure host has proper format
            if not self.host.startswith('http'):
                self.host = f'http://{self.host}'

            # Test connection
            if not self.test_connection():
                return False

            self._initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize Ollama provider: {e}")
            return False

    def validate_config(self) -> None:
        """Validate Ollama configuration."""
        import sys

        # Check required keys
        for key in self.required_config_keys:
            if not self.config.get(key):
                print(f"Error: Missing required configuration for Ollama: {key}")
                if key == 'host':
                    print("Set OLLAMA_HOST environment variable (e.g., http://localhost:11434)")
                elif key == 'model':
                    print("Set OLLAMA_MODEL environment variable (e.g., llava:latest or minicpm-v:latest)")
                sys.exit(1)

        # Validate host format
        host = self.config.get('host', '')
        if host and not (host.startswith('http://') or host.startswith('https://')):
            print(f"Error: Ollama host must start with http:// or https://, got: {host}")
            sys.exit(1)

    def test_connection(self) -> bool:
        """Test connection to Ollama server using REST API."""
        try:
            print(f"  Host: {self.host}")
            print(f"  Vision model: {self.model}")
            print(f"  Text model: {self.text_model}")
            # Try to list models via REST API
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])

                # Check if our configured model is available
                model_names = [m.get('name', '') for m in models]

                if self.model and model_names:
                    # Check for exact match or partial match
                    found = False
                    for name in model_names:
                        if self.model in name or name in self.model:
                            found = True
                            break

                    if not found:
                        available = ', '.join(model_names[:3])
                        if len(model_names) > 3:
                            available += f' ... and {len(model_names) - 3} more'
                        print(f"Warning: Model '{self.model}' not found. Available models: {available}")
                        print(f"Continuing anyway - the model might still work if it's being pulled.")

                return True
            else:
                print(f"Cannot connect to Ollama at {self.host}")
                print(f"Status code: {response.status_code}")
                return False

        except requests.ConnectionError:
            print(f"Cannot connect to Ollama at {self.host}")
            print(f"Please ensure Ollama is running: ollama serve")
            return False
        except Exception as e:
            print(f"Unexpected error testing connection: {e}")
            return False

    def _make_api_request(self, endpoint: str, payload: dict, timeout: int = None) -> dict:
        """Make a REST API request to Ollama."""
        if timeout is None:
            timeout = self.timeout

        url = f"{self.host}/api/{endpoint}"
        print(f"  -> POST {url} (model={payload.get('model')}, timeout={timeout}s)")

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise ProviderError(f"API request failed with status {response.status_code}: {response.text}")

        except requests.Timeout:
            raise ProviderError(f"Request timed out after {timeout} seconds")
        except requests.ConnectionError:
            raise ProviderConnectionError(f"Cannot connect to Ollama at {self.host}")
        except Exception as e:
            raise ProviderError(f"API request failed: {e}")

    def describe_images(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        initial_categories: Optional[List[str]] = None
    ) -> List[ImageData]:
        """
        Phase 1: Generate descriptions for images only.

        Args:
            image_paths: List of absolute paths to image files
            progress_callback: Optional callback for progress updates (message, progress_0_to_1)
            initial_categories: Optional list of initial category suggestions for the LLM

        Returns:
            List of ImageData with descriptions and suggested_categories populated
        """
        if not self._initialized:
            raise ProviderError("Provider not initialized")

        # Validate image paths
        valid_paths = self._validate_image_paths(image_paths)
        if not valid_paths:
            raise ProviderProcessingError("No valid images found")

        self._report_progress(progress_callback, "Starting image description...", 0.0)

        # Generate descriptions for all images
        image_data_list = []
        total_images = len(valid_paths)

        for i, image_path in enumerate(valid_paths):
            self._report_progress(
                progress_callback,
                f"Describing image {i+1}/{total_images}: {Path(image_path).name}",
                (i / total_images)
            )

            try:
                description_data = self._describe_image(image_path, initial_categories)
                image_data = ImageData(
                    filename=Path(image_path).name,
                    filepath=image_path,
                    description=description_data['description'],
                    suggested_categories=description_data['initial_categories'],
                    primary_category=description_data['initial_categories'][0] if description_data['initial_categories'] else 'Uncategorized',
                    metadata={'provider': 'ollama', 'model': self.model}
                )
                image_data_list.append(image_data)

            except Exception as e:
                print(f"Error describing image {image_path}: {e}")
                # Create a fallback image data entry
                image_data = ImageData(
                    filename=Path(image_path).name,
                    filepath=image_path,
                    description=f"Error processing image: {str(e)}",
                    suggested_categories=['Error'],
                    primary_category='Error',
                    metadata={'provider': 'ollama', 'model': self.model, 'error': str(e)}
                )
                image_data_list.append(image_data)

        self._report_progress(progress_callback, "Description phase complete!", 1.0)
        return image_data_list

    def categorize_described_images(
        self,
        images: List[ImageData],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> CategorizationResult:
        """
        Phase 2: Categorize images that already have descriptions.

        Args:
            images: List of ImageData with descriptions already populated
            progress_callback: Optional callback for progress updates (message, progress_0_to_1)

        Returns:
            Complete CategorizationResult with final categories assigned
        """
        if not self._initialized:
            raise ProviderError("Provider not initialized")

        if not images:
            raise ProviderProcessingError("No images provided for categorization")

        self._report_progress(progress_callback, "Starting categorization...", 0.0)

        # Make a copy to avoid modifying the input
        image_data_list = [img for img in images]

        self._report_progress(progress_callback, "Analyzing images for categorization...", 0.3)

        try:
            self._categorize_images(image_data_list)
        except Exception as e:
            print(f"Error during categorization: {e}")
            # Apply fallback categorization
            self._apply_fallback_categorization(image_data_list)

        self._report_progress(progress_callback, "Finalizing results...", 0.8)

        # Create the result
        result = CategorizationResult(
            images=image_data_list,
            processing_stats={
                'provider': 'ollama',
                'model': self.model,
                'total_images': len(image_data_list),
                'successful_descriptions': len([img for img in image_data_list if not img.description.startswith('Error')]),
                'categories_generated': len(set(img.primary_category for img in image_data_list))
            },
            category_definitions=self.category_definitions
        )

        self._report_progress(progress_callback, "Categorization complete!", 1.0)
        return result

    def process_images(
        self,
        image_paths: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        initial_categories: Optional[List[str]] = None
    ) -> CategorizationResult:
        """Process images with Ollama for description and categorization."""

        if not self._initialized:
            raise ProviderError("Provider not initialized")

        # Validate image paths
        valid_paths = self._validate_image_paths(image_paths)
        if not valid_paths:
            raise ProviderProcessingError("No valid images found")

        self._report_progress(progress_callback, "Starting image processing...", 0.0)

        # Step 1: Generate descriptions for all images
        image_data_list = []
        total_images = len(valid_paths)

        for i, image_path in enumerate(valid_paths):
            self._report_progress(
                progress_callback,
                f"Describing image {i+1}/{total_images}: {Path(image_path).name}",
                (i / total_images) * 0.6  # Description takes 60% of progress
            )

            try:
                description_data = self._describe_image(image_path, initial_categories)
                image_data = ImageData(
                    filename=Path(image_path).name,
                    filepath=image_path,
                    description=description_data['description'],
                    suggested_categories=description_data['initial_categories'],
                    primary_category=description_data['initial_categories'][0] if description_data['initial_categories'] else 'Uncategorized',
                    metadata={'provider': 'ollama', 'model': self.model}
                )
                image_data_list.append(image_data)

            except Exception as e:
                print(f"Error describing image {image_path}: {e}")
                # Create a fallback image data entry
                image_data = ImageData(
                    filename=Path(image_path).name,
                    filepath=image_path,
                    description=f"Error processing image: {str(e)}",
                    suggested_categories=['Error'],
                    primary_category='Error',
                    metadata={'provider': 'ollama', 'model': self.model, 'error': str(e)}
                )
                image_data_list.append(image_data)

        # Step 2: Categorize all images collectively
        self._report_progress(progress_callback, "Analyzing images for categorization...", 0.7)

        try:
            self._categorize_images(image_data_list)
        except Exception as e:
            print(f"Error during categorization: {e}")
            # Apply fallback categorization
            self._apply_fallback_categorization(image_data_list)

        self._report_progress(progress_callback, "Finalizing results...", 0.9)

        # Create the result
        result = CategorizationResult(
            images=image_data_list,
            processing_stats={
                'provider': 'ollama',
                'model': self.model,
                'total_images': len(image_data_list),
                'successful_descriptions': len([img for img in image_data_list if not img.description.startswith('Error')]),
                'categories_generated': len(set(img.primary_category for img in image_data_list))
            },
            category_definitions=self.category_definitions
        )

        self._report_progress(progress_callback, "Processing complete!", 1.0)
        return result

    def _strip_code_blocks(self, text: str) -> str:
        """Strip markdown code blocks from model response."""
        text = text.strip()
        # Remove ```json or ```yaml or ``` fences
        if text.startswith('```'):
            lines = text.split('\n')
            # Drop first line (the fence) and last ``` line
            inner = lines[1:]
            if inner and inner[-1].strip() == '```':
                inner = inner[:-1]
            text = '\n'.join(inner).strip()
        return text

    def _describe_image(self, image_path: str, initial_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a structured description for a single image using REST API."""
        import json as _json

        # Load and encode image once
        try:
            img = ImageProcessor.prepare_image_for_processing(image_path)
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            raise ProviderProcessingError(f"Failed to process image {image_path}: {e}")

        # Build category guidance if provided
        categories_guidance = ""
        if initial_categories:
            categories_list = ", ".join(initial_categories)
            categories_guidance = (
                f" Use these as category inspiration: {categories_list}."
            )

        description_prompt = (
            f"Describe this image for categorization purposes.{categories_guidance} "
            "First judge photo quality: is this an accidental or failed shot "
            "(heavy motion blur, out of focus, pointed at the floor or ceiling, "
            "lens covered, no clear subject)? "
            "Return a JSON object with exactly two keys: "
            "\"description\" (a factual 2-4 sentence description covering subject, setting, colors, and any text visible; "
            "if the shot is accidental or failed, the first sentence must say so and why) "
            "and \"initial_categories\" (a JSON array of 2-5 short category strings based only on what is visible; "
            f"if the shot is accidental or failed, the first category must be \"{ACCIDENTAL_CATEGORY}\"). "
            "Return only the JSON object, no explanation, no markdown."
        )

        # Try up to max_retries times
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that returns structured JSON only."
                        },
                        {
                            "role": "user",
                            "content": description_prompt,
                            "images": [image_b64]
                        }
                    ],
                    "stream": False,
                    # Thinking models otherwise spend num_predict on reasoning
                    # and truncate the JSON; ignored by non-thinking models.
                    "think": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 512
                    }
                }

                response = self._make_api_request("chat", payload, timeout=self.timeout)
                response_text = response.get('message', {}).get('content', '').strip()

                if not response_text:
                    if attempt < self.max_retries - 1:
                        print(f"Empty response for {image_path}, retrying... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        break

                # Strip code blocks defensively before parsing
                clean_text = self._strip_code_blocks(response_text)

                try:
                    result = _json.loads(clean_text)

                    if not isinstance(result, dict):
                        raise ValueError("Response is not a JSON object")
                    if 'description' not in result or not isinstance(result['description'], str):
                        raise ValueError("Missing or invalid 'description' field")
                    if 'initial_categories' not in result or not isinstance(result['initial_categories'], list):
                        raise ValueError("Missing or invalid 'initial_categories' field")
                    if not result['initial_categories']:
                        raise ValueError("initial_categories cannot be empty")

                    return result

                except (_json.JSONDecodeError, ValueError) as e:
                    if attempt < self.max_retries - 1:
                        print(f"Invalid JSON for {image_path}, retrying... (attempt {attempt + 1}/{self.max_retries}, Error: {e})")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        print(f"Failed to get valid JSON after {self.max_retries} attempts for {image_path}: {e}")
                        print(f"DEBUG - Final failed response: {response_text}")
                        break

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Request failed for {image_path}, retrying... (attempt {attempt + 1}/{self.max_retries}, Error: {e})")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    print(f"Request failed after {self.max_retries} attempts for {image_path}: {e}")
                    break

        # All attempts failed - return failure marker
        return {
            "description": "failed",
            "initial_categories": ["Uncategorized"]
        }

    def _categorize_images(self, image_data_list: List[ImageData]):
        """Assign non-failed images to the user's category list, in batches."""

        # Filter out failed images
        valid_images = [img for img in image_data_list if img.description != "failed"]

        if not valid_images:
            print("No valid images to re-categorize")
            return

        print(f"Re-categorizing {len(valid_images)} valid images (skipping {len(image_data_list) - len(valid_images)} failed)")
        self._categorize_in_batches(valid_images)

    def _categorize_in_batches(self, valid_images: List[ImageData]):
        """Assign images to the user's saved category list, adding categories if needed.

        With no saved list, one is built from these photos. With a saved list, every
        photo is assigned to it with a "none of these" option; new categories are
        proposed only from the photos that fit none, and kept only if photos land in
        them. New categories are marked "new" for review and saved to the file.
        """
        path = categories_file_path(self.categories_file)
        can_save = True
        try:
            saved = load_categories(path)
        except (OSError, ValueError, yaml.YAMLError) as e:
            # Never overwrite a file we could not read; use a one-off list instead
            print(f"Could not read saved categories at {path} ({e}); "
                  f"building a list for this run only, not saved")
            saved, can_save = [], False

        if not saved:
            if can_save:
                print(f"No saved categories at {path}; building a category list from these photos")
            taxonomy = self._with_accidental(
                [dict(c, status=STATUS_NEW) for c in self._build_taxonomy(valid_images)])
            if len(taxonomy) <= 1:
                print("Could not build a category list, applying fallback categorization")
                self._apply_fallback_categorization(valid_images)
                return
            assigned = self._assign_in_batches(valid_images, taxonomy, allow_none=False)
        else:
            print(f"Loaded {len(saved)} saved categories from {path}")
            taxonomy = self._with_accidental(saved)
            assigned = self._assign_in_batches(valid_images, taxonomy, allow_none=True)
            unmatched = [img for img in valid_images if assigned.get(img.filename) == NONE_OPTION]
            minimum = max(3, len(valid_images) // 100)
            if len(unmatched) >= minimum:
                print(f"{len(unmatched)} photos fit no saved category; proposing new categories for them")
                proposed = self._propose_categories_for(unmatched, taxonomy, minimum)
                if proposed:
                    second = self._assign_in_batches(unmatched, proposed, allow_none=True)
                    sizes = Counter(second.values())
                    kept = [c for c in proposed if sizes[c['name']] >= minimum]
                    kept_names = {c['name'] for c in kept}
                    small = [f"{c['name']} ({sizes[c['name']]})" for c in proposed if c['name'] not in kept_names]
                    if small:
                        print(f"Dropped proposed categories with fewer than {minimum} photos: {', '.join(small)}")
                    # Photos in dropped categories stay NONE_OPTION and go to Unsorted
                    assigned.update({f: c for f, c in second.items() if c in kept_names})
                    taxonomy = taxonomy + kept
            elif unmatched:
                print(f"{len(unmatched)} photos fit no saved category, fewer than {minimum}; "
                      f"they go to {UNSORTED_CATEGORY}")

        # Photos that still fit nothing go to Unsorted, which is not saved
        for filename, category in assigned.items():
            if category == NONE_OPTION:
                assigned[filename] = UNSORTED_CATEGORY

        self.category_definitions = taxonomy
        print(f"Category list ({len(taxonomy)}):")
        for cat in taxonomy:
            marker = " [new]" if cat.get('status') == STATUS_NEW else ""
            print(f"  {cat['name']}{marker}: {cat['rule']}")

        self._apply_categories_by_filename(
            valid_images, [{'filename': f, 'final_category': c} for f, c in assigned.items()])
        missing = [img for img in valid_images if img.filename not in assigned]
        if missing:
            print(f"{len(missing)} images still unassigned, applying fallback categorization")
            self._apply_fallback_categorization(missing)

        if can_save and taxonomy != saved:
            save_categories(path, taxonomy)
            added = [c['name'] for c in taxonomy if c.get('status') == STATUS_NEW]
            print(f"Saved {len(taxonomy)} categories to {path}"
                  + (f"; new, for review: {', '.join(added)}" if added else ""))

    @staticmethod
    def _with_accidental(taxonomy: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Return the list with the Accidental Shots category present."""
        if any(c['name'].lower() == ACCIDENTAL_CATEGORY.lower() for c in taxonomy):
            return list(taxonomy)
        return list(taxonomy) + [{'name': ACCIDENTAL_CATEGORY, 'rule': ACCIDENTAL_RULE, 'status': STATUS_NEW}]

    def _assign_in_batches(
        self,
        images: List[ImageData],
        taxonomy: List[Dict[str, str]],
        allow_none: bool
    ) -> Dict[str, str]:
        """Assign images to the category list in batches; returns filename -> category.

        With allow_none, a photo may be answered NONE_OPTION when it fits no rule.
        """
        canonical = {cat['name'].lower(): cat['name'] for cat in taxonomy}
        lines = [f"- {cat['name']}: {cat['rule']}" for cat in taxonomy]
        if allow_none:
            canonical[NONE_OPTION.lower()] = NONE_OPTION
            lines.append(f"- {NONE_OPTION}: the photo fits none of the rules above")
        categories_text = "\n".join(lines)
        batches = [images[i:i + CATEGORIZE_BATCH_SIZE]
                   for i in range(0, len(images), CATEGORIZE_BATCH_SIZE)]

        assigned: Dict[str, str] = {}
        for n, batch in enumerate(batches, 1):
            print(f"Assigning batch {n}/{len(batches)} ({len(batch)} images)")
            pending = batch
            # First pass covers the whole batch; follow-up passes re-ask only for
            # images that came back missing or with a category not on the list
            for attempt in range(1 + CATEGORIZE_REASK_ROUNDS):
                label = f"batch {n}" if attempt == 0 else f"batch {n} re-ask {attempt}"
                assigned.update(self._assign_to_taxonomy(
                    pending, categories_text, canonical, label, allow_none=allow_none))
                pending = [img for img in batch if img.filename not in assigned]
                if not pending:
                    break
                print(f"Batch {n}: {len(pending)} images unassigned, re-asking for them only")
            if pending:
                print(f"Batch {n}: {len(pending)} images still unassigned")
        return assigned

    def _propose_categories_for(
        self,
        unmatched: List[ImageData],
        taxonomy: List[Dict[str, str]],
        minimum: int
    ) -> List[Dict[str, str]]:
        """Ask the text model for new categories for photos that fit no saved rule."""
        max_new = max(1, min(MAX_NEW_CATEGORIES_PER_RUN, len(unmatched) // minimum))
        # Keep the prompt bounded: an even sample of at most PROPOSE_SAMPLE_SIZE photos
        step = max(1, len(unmatched) // PROPOSE_SAMPLE_SIZE)
        sample = unmatched[::step][:PROPOSE_SAMPLE_SIZE]
        existing_text = "\n".join(f"- {c['name']}: {c['rule']}" for c in taxonomy)
        prompt = textwrap.dedent(f"""
            A user sorts photos into these saved categories. Each category is followed by
            the rule for what belongs in it:

            {{existing}}

            These photos fit none of the saved rules:

            {{blocks}}

            Create as few new categories as possible, at most {max_new}, each broad enough
            to hold at least {minimum} of these photos. For each write one rule sentence
            stating which photos belong in it and, where it could overlap a saved category,
            which do not.

            Rules:
            - Prefer one broad category over several narrow ones: for example one
              Portraits category for all photos of people, including selfies, rather than
              separate categories by age, setting or pose
            - Never create a synonym, near-duplicate or overlap of a saved category
            - Use folder-friendly names (1-3 words)
            - If these photos share nothing worth a category, return an empty list: []

            CRITICAL: Return ONLY a raw YAML list. NO markdown, NO code blocks, NO explanations.

            - name: Category Name
              rule: Photos that ... Not photos that ...
        """).strip().replace("{existing}", existing_text).replace(
            "{blocks}", self._filename_blocks(sample))

        items = self._request_yaml_list(
            prompt, self._validate_category, num_predict=800,
            label="new categories", allow_empty=True)
        if not items:
            print("No new categories proposed")
            return []
        known = {c['name'].lower() for c in taxonomy} | {NONE_OPTION.lower(), UNSORTED_CATEGORY.lower()}
        proposed = []
        for item in items[:max_new]:
            name = item['name'].strip()
            if name.lower() in known:
                continue
            known.add(name.lower())
            proposed.append({'name': name, 'rule': item['rule'].strip(), 'status': STATUS_NEW})
        print(f"Proposed: {', '.join(c['name'] for c in proposed) or 'none'}")
        return proposed

    def _assign_to_taxonomy(
        self,
        images: List[ImageData],
        categories_text: str,
        canonical: Dict[str, str],
        label: str,
        allow_none: bool = False
    ) -> Dict[str, str]:
        """Ask the text model to assign images to the fixed category list.

        Returns filename -> canonical category for the images it answered with a
        category from the list; off-list answers and unknown filenames are dropped.
        """
        prompt = textwrap.dedent(f"""
            Assign each image below to exactly one category from this fixed list.
            Each category is followed by the rule for what belongs in it:

            {{categories}}

            Images:

            {{blocks}}

            Rules:
            - Use only category names from the list, spelled exactly as written
            - Photo quality comes before subject: every image whose description or
              initial categories mark it as accidental, failed, blurry or unusable goes
              into "{ACCIDENTAL_CATEGORY}", whatever it shows
            - Otherwise choose the category whose rule the description matches best
            {{none_rule}}

            CRITICAL: Return ONLY a raw YAML list. NO markdown, NO code blocks, NO explanations.

            - filename: exact_filename.jpg
              final_category: Category Name

            Include ALL {len(images)} filenames in response.
        """).strip()
        none_rule = (
            f'- A photo belongs in a category only if its rule describes the photo\'s main '
            f'subject. If no rule does, answer "{NONE_OPTION}". Never pick a category just '
            f'because it is the closest one'
        ) if allow_none else ""
        prompt = prompt.replace("{none_rule}", none_rule)
        # Substituted after dedent so multi-line blocks do not break it
        prompt = prompt.replace("{categories}", categories_text).replace(
            "{blocks}", self._filename_blocks(images))

        assignments = self._request_yaml_list(
            prompt, self._validate_assignment, num_predict=2500, label=label)
        if assignments is None:
            return {}

        filenames = {img.filename for img in images}
        result = {}
        for item in assignments:
            filename = item['filename'].strip()
            category = canonical.get(item['final_category'].strip().lower())
            if filename not in filenames:
                continue
            if category is None:
                print(f"  {label}: {filename} got off-list category '{item['final_category']}'")
                continue
            result[filename] = category
        return result

    def _build_taxonomy(self, valid_images: List[ImageData]) -> List[Dict[str, str]]:
        """Ask the text model for one category list, each with an inclusion rule."""
        suggestions_text = self._suggestion_tally(valid_images)

        prompt = textwrap.dedent(f"""
            A photo collection of {len(valid_images)} images was described one image at a
            time. These are the categories suggested per image, with how many images
            received each suggestion:

            {{suggestions}}

            Create 8-15 final categories for sorting the whole collection into folders.
            For each category write one rule sentence that states which images belong
            in it and, where it could overlap another category, which do not.

            Rules:
            - Cover the frequent suggestions; merge synonyms and near-duplicates into one
              category (for example portraits and self-portraits are one category)
            - Categories must not overlap: any image fits exactly one rule
            - Use folder-friendly names (1-3 words)
            - Include a category named exactly "{ACCIDENTAL_CATEGORY}" for accidental,
              failed, blurry or unusable shots

            CRITICAL: Return ONLY a raw YAML list. NO markdown, NO code blocks, NO explanations.

            - name: Category Name
              rule: Images that ... Not images that ...
            - name: Another Category
              rule: Images that ...
        """).strip().replace("{suggestions}", suggestions_text)

        items = self._request_yaml_list(
            prompt, self._validate_category, num_predict=1500, label="category list")
        if items is None:
            return []
        taxonomy = []
        for item in items:
            name = item['name'].strip()
            if name.lower() not in (t['name'].lower() for t in taxonomy):
                taxonomy.append({'name': name, 'rule': item['rule'].strip()})
        if ACCIDENTAL_CATEGORY.lower() not in (t['name'].lower() for t in taxonomy):
            taxonomy.append({
                'name': ACCIDENTAL_CATEGORY,
                'rule': ACCIDENTAL_RULE})
        return taxonomy

    @staticmethod
    def _suggestion_tally(valid_images: List[ImageData]) -> str:
        """Most frequent Phase 1 suggested categories, one '- Name (count)' per line."""
        counts = Counter()
        display = {}
        for img in valid_images:
            for cat in img.suggested_categories or []:
                key = cat.strip().lower()
                if key:
                    counts[key] += 1
                    display.setdefault(key, cat.strip())
        return "\n".join(
            f"- {display[key]} ({count})"
            for key, count in counts.most_common(TAXONOMY_TOP_SUGGESTIONS))

    @staticmethod
    def _validate_category(item: Any, i: int):
        """Check one name/rule YAML item."""
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a YAML object")
        for key in ('name', 'rule'):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Item {i} missing or invalid '{key}' field")

    @staticmethod
    def _filename_blocks(images: List[ImageData]) -> str:
        """Format images as Filename/Description/Initial Categories blocks."""
        blocks = []
        for img in images:
            initial_categories_str = ", ".join(img.suggested_categories) if img.suggested_categories else "None"
            blocks.append(f"Filename: {img.filename}\nDescription: {img.description}\nInitial Categories: {initial_categories_str}")
        return "\n\n".join(blocks)

    @staticmethod
    def _validate_assignment(item: Any, i: int):
        """Check one filename/final_category YAML item."""
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a YAML object")
        if 'filename' not in item or not isinstance(item['filename'], str):
            raise ValueError(f"Item {i} missing or invalid 'filename' field")
        if 'final_category' not in item or not isinstance(item['final_category'], str):
            raise ValueError(f"Item {i} missing or invalid 'final_category' field")

    def _request_yaml_list(
        self,
        prompt: str,
        validate_item: Callable[[Any, int], None],
        num_predict: int,
        label: str,
        allow_empty: bool = False
    ) -> Optional[List[Any]]:
        """Send a text-model prompt and return its YAML list, retrying on bad output.

        Returns None if every attempt fails.
        """
        for attempt in range(self.max_retries):
            try:
                # Build REST API payload - use non-vision model for text-only categorization
                payload = {
                    "model": self.text_model,  # Use text model for categorization
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": num_predict,
                        "num_ctx": CATEGORIZE_NUM_CTX
                    }
                }

                # Make REST API request
                response = self._make_api_request("chat", payload, timeout=self.timeout)

                response_text = response.get('message', {}).get('content', '').strip()

                if not response_text:
                    if attempt < self.max_retries - 1:
                        print(f"Empty {label} response, retrying... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        break

                # Extract YAML from code blocks if needed
                yaml_content = response_text
                if '```yaml' in response_text or '```' in response_text:
                    lines = response_text.split('\n')
                    yaml_lines = []
                    inside_yaml_block = False

                    for line in lines:
                        if line.strip().startswith('```yaml') or (line.strip() == '```' and not inside_yaml_block):
                            inside_yaml_block = True
                            continue
                        elif line.strip() == '```' and inside_yaml_block:
                            break
                        elif inside_yaml_block:
                            yaml_lines.append(line)

                    if yaml_lines:
                        yaml_content = '\n'.join(yaml_lines)

                # Validate YAML format
                try:
                    parsed = yaml.safe_load(yaml_content)

                    # Validate structure
                    if parsed is None and allow_empty:
                        return []
                    if not isinstance(parsed, list):
                        raise ValueError("Response is not a YAML list")
                    if not parsed and not allow_empty:
                        raise ValueError("Response list is empty")

                    # Validate each item
                    for i, item in enumerate(parsed):
                        validate_item(item, i)

                    return parsed

                except (yaml.YAMLError, ValueError) as e:
                    if attempt < self.max_retries - 1:
                        print(f"Invalid YAML format in {label}, retrying... (attempt {attempt + 1}/{self.max_retries}, Error: {e})")
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        print(f"Failed to get valid YAML after {self.max_retries} attempts in {label}: {e}")
                        print(f"DEBUG - Final failed {label} response: {response_text}")
                        break

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"{label} request failed, retrying... (attempt {attempt + 1}/{self.max_retries}, Error: {e})")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    print(f"{label} failed after {self.max_retries} attempts: {e}")
                    break

        return None

    def _apply_categories_by_filename(self, image_data_list: List[ImageData], categories_list: List[Dict[str, str]]):
        """Apply final categories using filename matching."""
        # Create filename lookup from re-categorization response
        category_lookup = {item['filename']: item['final_category'] for item in categories_list if 'filename' in item and 'final_category' in item}

        # Update existing ImageData objects
        for image_data in image_data_list:
            if image_data.filename in category_lookup:
                final_category = category_lookup[image_data.filename].strip()
                image_data.primary_category = final_category
            else:
                # Fallback if filename not found in response
                print(f"Warning: No final category found for {image_data.filename}, keeping initial primary category")

    def _apply_fallback_categorization(self, image_data_list: List[ImageData]):
        """Apply fallback categorization based on description keywords and initial categories."""
        category_keywords = {
            "People": ["person", "people", "man", "woman", "child", "human", "face", "portrait"],
            "Nature": ["tree", "forest", "mountain", "sky", "flower", "landscape", "outdoor", "nature"],
            "Food": ["food", "meal", "dish", "cooking", "restaurant", "kitchen"],
            "Animals": ["animal", "dog", "cat", "pet", "wildlife", "bird", "fish"],
            "Architecture": ["building", "house", "structure", "bridge", "architecture"],
            "Vehicles": ["car", "vehicle", "truck", "bike", "motorcycle", "transportation"],
            "Technology": ["computer", "phone", "device", "screen", "electronic"],
            "Art": ["art", "painting", "drawing", "sculpture", "artwork"]
        }

        for image_data in image_data_list:
            # Only apply fallback if no primary category exists or it's Error
            if image_data.primary_category and image_data.primary_category != 'Error':
                continue

            # First, try to use initial suggested categories if available
            if image_data.suggested_categories and image_data.suggested_categories != ['Error']:
                image_data.primary_category = image_data.suggested_categories[0]
                continue

            # If no initial categories, use keyword-based fallback
            description = image_data.description.lower()
            best_category = "Uncategorized"
            max_matches = 0

            for category, keywords in category_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in description)
                if matches > max_matches:
                    max_matches = matches
                    best_category = category

            image_data.primary_category = best_category
            image_data.suggested_categories = [best_category]

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Ollama provider capabilities."""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            'max_image_size': 10 * 1024 * 1024,  # 10MB limit
            'concurrent_requests': 1,
            'supports_batch_processing': True,
            'requires_internet': True
        })
        return base_capabilities