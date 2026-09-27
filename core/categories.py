"""
The user's saved category list.

Categories persist across runs and photo libraries in one YAML file. Each
category has a name, a rule describing which photos belong in it, and a status:
"kept" once the user has reviewed it, "new" when the model added it and the user
has not reviewed it yet.

File format:

    categories:
      - name: Portraits
        rule: Photos of one or two people posed for the camera, including selfies.
        status: kept
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

DEFAULT_CATEGORIES_FILE = Path.home() / ".config" / "image-categorizer" / "categories.yaml"
STATUS_KEPT = "kept"
STATUS_NEW = "new"


def categories_file_path(configured: Optional[str] = None) -> Path:
    """Return the categories file path: the configured one, or the default."""
    return Path(configured).expanduser() if configured else DEFAULT_CATEGORIES_FILE


def load_categories(path: Path) -> List[Dict[str, str]]:
    """
    Load saved categories. A missing file means no saved categories yet.

    Raises:
        ValueError: If the file exists but is not a valid categories file.
    """
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("categories") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a top-level 'categories' list")

    categories = []
    seen = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: category {i + 1} is not a mapping")
        name = str(item.get("name") or "").strip()
        rule = str(item.get("rule") or "").strip()
        if not name:
            raise ValueError(f"{path}: category {i + 1} has no name")
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        status = item.get("status", STATUS_KEPT)
        categories.append({
            "name": name,
            "rule": rule,
            "status": status if status in (STATUS_KEPT, STATUS_NEW) else STATUS_KEPT,
        })
    return categories


def save_categories(path: Path, categories: List[Dict[str, str]]):
    """Write categories atomically, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"categories": [
        {"name": c["name"], "rule": c.get("rule", ""), "status": c.get("status", STATUS_KEPT)}
        for c in categories
    ]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)
    os.replace(tmp, path)
