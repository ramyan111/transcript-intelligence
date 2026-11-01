"""Exploratory topic discovery: embeddings -> KMeans, with representative examples per cluster.

Embeddings: Sentence Transformers ``all-MiniLM-L6-v2`` when installed (downloads the model on first use);
otherwise a TF-IDF + SVD fallback, and the method used is recorded. Silhouette is reported for each k but is only
a guide: choose k after reading the representative transcripts. Clusters inform the taxonomy; they are not the taxonomy.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)
ST_MODEL = "all-MiniLM-L6-v2"


def embed_texts(texts: list[str], prefer_sentence_transformers: bool = True) -> tuple[np.ndarray, str]:
    if prefer_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer
            vecs = SentenceTransformer(ST_MODEL).encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(vecs), f"sentence-transformers:{ST_MODEL}"
        except Exception as exc:  # not installed, offline, model download failed...
            logger.warning("Sentence Transformers unavailable (%s); falling back to TF-IDF+SVD", exc)
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=3000).fit_transform(texts)
    dims = max(2, min(50, X.shape[0] - 1, X.shape[1] - 1))
    return normalize(TruncatedSVD(dims, random_state=42).fit_transform(X)), "tfidf-svd-fallback"


def discover_clusters(df: pd.DataFrame, k: int | None = None, k_range=range(4, 11), seed: int = 42,
                      text_col: str = "redacted_text", prefer_sentence_transformers: bool = True) -> dict:
    """Cluster transcripts; return JSON-serializable summary. ``k=None`` picks the best silhouette in ``k_range``."""
    texts = df[text_col].tolist()
    if len(texts) < 6:
        return {"error": "too few transcripts for clustering", "n": len(texts)}
    emb, method = embed_texts(texts, prefer_sentence_transformers)
    scores = {}
    for kk in k_range:
        if 2 <= kk < len(texts):
            labels = KMeans(kk, n_init=10, random_state=seed).fit_predict(emb)
            if len(set(labels)) > 1:
                scores[kk] = float(silhouette_score(emb, labels))
    chosen = k or max(scores, key=scores.get)
    km = KMeans(chosen, n_init=10, random_state=seed).fit(emb)
    terms_vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=2000)
    T = terms_vec.fit_transform(texts)
    vocab = np.array(terms_vec.get_feature_names_out())
    clusters = []
    for c in range(chosen):
        idx = np.where(km.labels_ == c)[0]
        dist = np.linalg.norm(emb[idx] - km.cluster_centers_[c], axis=1)
        reps = idx[np.argsort(dist)[:3]]
        top_terms = vocab[np.asarray(T[idx].mean(axis=0)).ravel().argsort()[::-1][:8]].tolist()
        sub = df.iloc[idx]
        clusters.append({
            "cluster": c, "size": int(len(idx)), "top_terms": top_terms,
            "call_type_mix": {k_: int(v) for k_, v in sub["call_type"].value_counts().items()},
            "representatives": [{"transcript_id": df.iloc[i]["transcript_id"], "excerpt": texts[i][:300]} for i in reps]})
    return {"method": method, "k": int(chosen), "k_was_chosen_by_silhouette": k is None,
            "silhouette_by_k": {str(a): round(b, 4) for a, b in scores.items()}, "clusters": clusters,
            "note": "Inspect representatives before trusting k; silhouette alone is a weak guide on small samples."}
