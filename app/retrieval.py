from functools import lru_cache
from typing import List, Dict, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .catalog import get_items

TYPE_LETTERS = {"A", "B", "C", "D", "E", "K", "P", "S"}


def _doc_text(item: Dict[str, Any]) -> str:
    types = " ".join(item.get("test_type") or [])
    return f"{item['name']} {item.get('description', '')} {types}"


@lru_cache(maxsize=1)
def _index():
    items = get_items()
    docs = [_doc_text(it) for it in items]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    return items, vectorizer, matrix


def reset_index_cache():
    _index.cache_clear()


def search(query: str, top_k: int = 10, type_filter: Optional[List[str]] = None,
           exclude_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Returns up to top_k catalog items ranked by relevance to `query`.

    type_filter: optional list of letter codes (e.g. ["K", "P"]) -- if given,
    only items having at least one matching test_type are considered, so a
    user saying "add personality tests" reliably narrows results.
    """
    items, vectorizer, matrix = _index()
    if not query.strip():
        candidates_idx = list(range(len(items)))
        scores = [0.0] * len(items)
    else:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix).flatten()
        candidates_idx = list(range(len(items)))
        scores = sims.tolist()

    ranked = sorted(zip(candidates_idx, scores), key=lambda t: t[1], reverse=True)

    exclude = {n.strip().lower() for n in (exclude_names or [])}
    type_filter_norm = {t.strip().upper() for t in (type_filter or [])} & TYPE_LETTERS

    results = []
    for idx, score in ranked:
        item = items[idx]
        if item["name"].strip().lower() in exclude:
            continue
        if type_filter_norm:
            item_types = {t.upper() for t in (item.get("test_type") or [])}
            if not (item_types & type_filter_norm):
                continue
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def lookup(names: List[str]) -> List[Dict[str, Any]]:
    """Exact/fuzzy lookup by name, used for comparison requests."""
    items, _, _ = _index()
    out = []
    for n in names:
        n_l = n.strip().lower()
        match = next((it for it in items if n_l == it["name"].strip().lower()), None)
        if not match:
            match = next((it for it in items if n_l in it["name"].strip().lower()), None)
        if match:
            out.append(match)
    return out
