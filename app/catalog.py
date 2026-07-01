from curses import raw
import json
import os
from functools import lru_cache
from typing import Dict, List, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REAL_CATALOG = os.path.join(DATA_DIR, "catalog.json")
SAMPLE_CATALOG = os.path.join(DATA_DIR, "catalog.sample.json")


@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, Any]:
    """Loads data/catalog.json (real scrape output). Falls back to the
    bundled sample catalog with a loud stderr warning so local dev / CI
    never silently ships placeholder data to prod."""
    path = REAL_CATALOG if os.path.exists(REAL_CATALOG) else SAMPLE_CATALOG
    if path == SAMPLE_CATALOG:
        import sys
        print(
            "WARNING: data/catalog.json not found -- using bundled SAMPLE "
            "catalog. Run scripts/scrape_catalog.py and place the output at "
            "data/catalog.json before deploying.",
            file=sys.stderr,
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Support both the sample catalog format and your real SHL catalog
    if isinstance(raw, list):
        items: List[Dict[str, Any]] = raw
        legend = {}
    else:
        items: List[Dict[str, Any]] = raw.get("items", [])
        legend = raw.get("legend", {})

    # Normalize fields so the rest of the code works
    for it in items:
        it.setdefault("description", "")

        # Real SHL catalog uses "link" instead of "url"
        if "link" in it and "url" not in it:
            it["url"] = it["link"]

        # Real SHL catalog uses "keys" instead of "test_type"
        if "keys" in it and "test_type" not in it:
            it["test_type"] = it["keys"]

        it.setdefault("test_type", [])

        if isinstance(it["test_type"], str):
            it["test_type"] = [it["test_type"]]

    return {
        "legend": legend,
        "items": items,
        "source": path
    }


def get_items() -> List[Dict[str, Any]]:
    return load_catalog()["items"]


def get_legend() -> Dict[str, str]:
    return load_catalog()["legend"]


def find_by_name(name: str):
    name_l = name.strip().lower()
    for it in get_items():
        if it["name"].strip().lower() == name_l or name_l in it["name"].strip().lower():
            return it
    return None
