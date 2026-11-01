"""Human-review workflow and metrics. Metrics are computed ONLY when reviewer labels exist."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

REVIEW_COLUMNS = ["transcript_id", "call_type", "model_primary_topic", "model_sentiment_label", "model_renewal_flag",
                  "reviewer_topic", "reviewer_sentiment", "reviewer_renewal_risk", "unsupported_claims", "notes"]
GROUND_TRUTH_COLUMNS = ["transcript_id", "call_type", "topic_label", "sentiment_label", "data_origin", "text_hash"]


def make_review_sample(df: pd.DataFrame, n: int = 24, seed: int = 42) -> pd.DataFrame:
    """Sample ~n transcripts stratified by call type (round-robin over types, spread across topics)."""
    parts, types = [], sorted(df["call_type"].unique())
    per = max(1, n // max(1, len(types)))
    for t in types:
        sub = df[df["call_type"] == t].sample(frac=1, random_state=seed).sort_values("primary_topic", kind="stable")
        step = max(1, len(sub) // per)
        parts.append(sub.iloc[::step].head(per))
    s = pd.concat(parts).head(n)
    out = pd.DataFrame({"transcript_id": s["transcript_id"], "call_type": s["call_type"],
                        "model_primary_topic": s["primary_topic"], "model_sentiment_label": s["sentiment_label"],
                        "model_renewal_flag": s["renewal_risk_signals"].map(lambda x: "y" if len(x) else "n")})
    for c in REVIEW_COLUMNS[5:]:
        out[c] = ""
    return out[REVIEW_COLUMNS].reset_index(drop=True)


def compute_metrics(review: pd.DataFrame) -> dict | None:
    """Accuracy, macro-F1, confusion matrices, renewal-flag precision/recall; None if nothing has been reviewed."""
    rv = review.fillna("").astype(str)
    out: dict = {}
    for name, truth, pred in [("topic", "reviewer_topic", "model_primary_topic"), ("sentiment", "reviewer_sentiment", "model_sentiment_label")]:
        sub = rv[rv[truth].str.strip() != ""]
        if len(sub):
            labels = sorted(set(sub[truth]) | set(sub[pred]))
            out[name] = {"n_reviewed": int(len(sub)), "accuracy": round(float(accuracy_score(sub[truth], sub[pred])), 3),
                         "macro_f1": round(float(f1_score(sub[truth], sub[pred], average="macro", labels=labels, zero_division=0)), 3),
                         "labels": labels, "confusion_matrix": confusion_matrix(sub[truth], sub[pred], labels=labels).tolist()}
    sub = rv[rv["reviewer_renewal_risk"].str.strip().str.lower().isin(["y", "n"])]
    if len(sub):
        t, p = sub["reviewer_renewal_risk"].str.lower().eq("y"), sub["model_renewal_flag"].str.lower().eq("y")
        tp = int((t & p).sum())
        out["renewal_flag"] = {"n_reviewed": int(len(sub)), "flagged": int(p.sum()), "true_positive": tp,
                               "precision": round(float(tp / p.sum()), 3) if p.sum() else None,
                               "recall": round(float(tp / t.sum()), 3) if t.sum() else None}
    uns = rv[rv["unsupported_claims"].str.strip().str.lower().isin(["y", "n"])]
    if len(uns):
        out["unsupported_claims"] = {"n_reviewed": int(len(uns)), "rate": round(float(uns["unsupported_claims"].str.lower().eq("y").mean()), 3)}
    return out or None


def load_ground_truth(path) -> pd.DataFrame | None:
    """Load a separate synthetic ground-truth file, validating its required columns."""
    p = Path(path)
    if not p.exists():
        return None
    labels = pd.read_csv(p, dtype=str, keep_default_na=False)
    missing = set(GROUND_TRUTH_COLUMNS) - set(labels.columns)
    if missing:
        raise ValueError(f"Ground-truth file is missing required columns: {', '.join(sorted(missing))}")
    return labels[GROUND_TRUTH_COLUMNS]


def compute_ground_truth_metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict | None:
    """Compare synthetic predictions with scenario-designed labels; never treat real data as synthetic truth."""
    prediction_columns = {"transcript_id", "call_type", "primary_topic", "sentiment_label", "data_origin", "text_hash"}
    missing_predictions = prediction_columns - set(predictions.columns)
    missing_labels = set(GROUND_TRUTH_COLUMNS) - set(labels.columns)
    if missing_predictions:
        raise ValueError(f"Predictions are missing required columns: {', '.join(sorted(missing_predictions))}")
    if missing_labels:
        raise ValueError(f"Ground truth is missing required columns: {', '.join(sorted(missing_labels))}")
    if set(predictions["data_origin"].dropna()) != {"synthetic"}:
        raise ValueError("Ground-truth metrics are only valid for synthetic predictions.")
    if set(labels["data_origin"].dropna()) != {"synthetic"}:
        raise ValueError("Ground-truth labels must be marked synthetic.")
    if predictions["transcript_id"].duplicated().any() or labels["transcript_id"].duplicated().any():
        raise ValueError("Transcript IDs must be unique in predictions and ground truth.")

    merged = labels.merge(predictions, on="transcript_id", how="inner", suffixes=("_truth", "_prediction"),
                          validate="one_to_one")
    if merged.empty:
        return None
    if not merged["call_type_truth"].eq(merged["call_type_prediction"]).all():
        raise ValueError("Call types differ between predictions and ground truth.")
    if not merged["text_hash_truth"].eq(merged["text_hash_prediction"]).all():
        raise ValueError("Transcript text differs between predictions and ground truth.")

    def metrics(truth: str, predicted: str) -> dict:
        categories = sorted(set(merged[truth]) | set(merged[predicted]))
        return {
            "n": int(len(merged)),
            "accuracy": round(float(accuracy_score(merged[truth], merged[predicted])), 3),
            "macro_f1": round(float(f1_score(merged[truth], merged[predicted], average="macro",
                                             labels=categories, zero_division=0)), 3),
            "labels": categories,
            "confusion_matrix": confusion_matrix(merged[truth], merged[predicted], labels=categories).tolist(),
        }

    return {
        "n_matched": int(len(merged)),
        "n_predictions": int(len(predictions)),
        "n_ground_truth": int(len(labels)),
        "topic": metrics("topic_label", "primary_topic"),
        "sentiment": metrics("sentiment_label_truth", "sentiment_label_prediction"),
    }


def load_review(path) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else None
