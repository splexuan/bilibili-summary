"""
Persistent knowledge-base indexes for local video search.

Hybrid retrieval: BM25 (keyword) + TF-IDF cosine (fuzzy) — merged & deduped.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from downloader import OUTPUT_DIR

try:
    from joblib import dump as _dump, load as _load
except Exception:  # pragma: no cover - fallback for minimal installs
    import pickle

    def _dump(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def _load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


INDEX_VERSION = 2  # bumped: added BM25 helper
INDEX_DIR = OUTPUT_DIR / "_index"
SUMMARY_INDEX_FILE = INDEX_DIR / "summaries.joblib"
CHECKSUM_FILE = INDEX_DIR / "summaries.sha256"


def _scan_summary_snapshot() -> dict:
    """Return cheap file metadata for all available summary files (from DB) — videos + articles."""
    from db import _conn, load_all_article_summaries
    snapshot = {}
    try:
        db = _conn()
        rows = db.execute("SELECT vid, summary FROM videos WHERE summary IS NOT NULL AND summary != ''").fetchall()
        for vid, summary in rows:
            snapshot[vid] = {"mtime_ns": 0, "size": len(summary) if summary else 0}
        db.close()
    except Exception:
        pass
    # 添加文章总结
    try:
        for aid, summary in load_all_article_summaries().items():
            snapshot[aid] = {"mtime_ns": 0, "size": len(summary) if summary else 0}
    except Exception:
        pass
    return snapshot


def _compute_checksum(path: Path) -> str:
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return ""


def _load_cached_index(snapshot: dict) -> dict | None:
    if not SUMMARY_INDEX_FILE.exists():
        return None

    if CHECKSUM_FILE.exists():
        try:
            stored_checksum = CHECKSUM_FILE.read_text(encoding="utf-8").strip()
            actual_checksum = _compute_checksum(SUMMARY_INDEX_FILE)
            if stored_checksum and actual_checksum and stored_checksum != actual_checksum:
                return None
        except Exception:
            pass

    try:
        payload = _load(SUMMARY_INDEX_FILE)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != INDEX_VERSION:
        return None
    if payload.get("snapshot") != snapshot:
        return None
    if not payload.get("vids") or payload.get("matrix") is None:
        return None

    return payload


def _build_index(snapshot: dict) -> dict | None:
    from db import _conn, load_all_article_summaries
    vids = []
    texts = []

    # 视频总结
    for vid in sorted(snapshot):
        if vid.startswith("A_"):
            continue  # 文章单独处理
        try:
            db = _conn()
            row = db.execute("SELECT summary FROM videos WHERE vid=?", (vid,)).fetchone()
            db.close()
            text = row[0].strip() if row and row[0] else ""
        except Exception:
            continue
        if text:
            vids.append(vid)
            texts.append(text)

    # 文章总结
    try:
        for aid, summary in load_all_article_summaries().items():
            if summary.strip():
                vids.append(aid)
                texts.append(summary.strip())
    except Exception:
        pass

    if not vids:
        return None

    # TF-IDF for cosine similarity (fuzzy semantic)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(texts)

    # Tokenized texts for BM25
    tokenized = [t.split() for t in texts]

    payload = {
        "version": INDEX_VERSION,
        "snapshot": snapshot,
        "vids": vids,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "tokenized": tokenized,
    }

    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        _dump(payload, SUMMARY_INDEX_FILE)
        checksum = _compute_checksum(SUMMARY_INDEX_FILE)
        if checksum:
            CHECKSUM_FILE.write_text(checksum, encoding="utf-8")
    except Exception:
        pass

    return payload


def get_summary_index() -> tuple[dict | None, bool]:
    """
    Return (index, has_summaries).

    The index is rebuilt only when summary files are added, removed, or changed.
    """
    snapshot = _scan_summary_snapshot()
    if not snapshot:
        return None, False

    cached = _load_cached_index(snapshot)
    if cached:
        return cached, True

    return _build_index(snapshot), True


def _bm25_scores(query: str, index: dict) -> dict[str, float]:
    """BM25 keyword scores (lazy import rank_bm25)."""
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(index["tokenized"])
    tokenized_q = query.split()
    raw = bm25.get_scores(tokenized_q)
    # Normalize to [0, 1]
    mx = raw.max()
    if mx > 0:
        raw = raw / mx
    return {index["vids"][i]: float(raw[i]) for i in range(len(raw))}


def _tfidf_scores(query: str, index: dict) -> dict[str, float]:
    """TF-IDF cosine similarity scores."""
    q_vec = index["vectorizer"].transform([query])
    scores = cosine_similarity(q_vec, index["matrix"])[0]
    return {index["vids"][i]: float(scores[i]) for i in range(len(scores))}


def _merge_scores(bm25: dict[str, float], tfidf: dict[str, float],
                  alpha: float = 0.5) -> list[tuple[str, float]]:
    """Weighted merge: alpha for BM25, (1-alpha) for TF-IDF."""
    all_vids = set(bm25) | set(tfidf)
    merged = []
    for vid in all_vids:
        s = alpha * bm25.get(vid, 0) + (1 - alpha) * tfidf.get(vid, 0)
        merged.append((vid, s))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


def rank_summary_videos(question: str, target_vid: str = ""
                        ) -> tuple[list[tuple[str, float]], bool]:
    """
    Hybrid rank: BM25 + TF-IDF cosine, merged and sorted.

    Returns (scores, has_summaries). scores = [(vid, score), ...] descending.
    """
    index, has_summaries = get_summary_index()
    if not has_summaries or not index:
        return [], has_summaries

    vids = index["vids"]

    if target_vid:
        return ([(target_vid, 1.0)] if target_vid in vids else []), True

    try:
        bm = _bm25_scores(question, index)
        tf = _tfidf_scores(question, index)
        ranked = _merge_scores(bm, tf)
    except Exception:
        ranked = [(vid, 0.0) for vid in vids]

    return ranked, True
