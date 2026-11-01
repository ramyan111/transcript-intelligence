import pandas as pd
import pytest

from src.evaluation import compute_ground_truth_metrics, load_ground_truth


def _predictions():
    return pd.DataFrame([
        {"transcript_id": "SYN-001", "call_type": "support", "primary_topic": "Billing & Payments",
         "sentiment_label": "Negative", "data_origin": "synthetic", "text_hash": "hash-1"},
        {"transcript_id": "SYN-002", "call_type": "internal", "primary_topic": "Engineering Blockers",
         "sentiment_label": "Neutral", "data_origin": "synthetic", "text_hash": "hash-2"},
    ])


def _labels():
    return pd.DataFrame([
        {"transcript_id": "SYN-001", "call_type": "support", "topic_label": "Billing & Payments",
         "sentiment_label": "Negative", "data_origin": "synthetic", "text_hash": "hash-1"},
        {"transcript_id": "SYN-002", "call_type": "internal", "topic_label": "Engineering Blockers",
         "sentiment_label": "Mixed", "data_origin": "synthetic", "text_hash": "hash-2"},
    ])


def test_synthetic_ground_truth_metrics_match_by_id_and_report_counts():
    result = compute_ground_truth_metrics(_predictions(), _labels())

    assert result["n_matched"] == 2
    assert result["topic"]["accuracy"] == 1.0
    assert result["sentiment"]["accuracy"] == 0.5
    assert result["sentiment"]["n"] == 2


def test_ground_truth_metrics_reject_real_data():
    predictions = _predictions()
    predictions.loc[0, "data_origin"] = "original"
    with pytest.raises(ValueError, match="only valid for synthetic"):
        compute_ground_truth_metrics(predictions, _labels())


def test_ground_truth_metrics_reject_call_type_mismatches():
    labels = _labels()
    labels.loc[0, "call_type"] = "external"
    with pytest.raises(ValueError, match="Call types differ"):
        compute_ground_truth_metrics(_predictions(), labels)


def test_ground_truth_metrics_reject_stale_predictions():
    predictions = _predictions()
    predictions.loc[0, "text_hash"] = "previous-transcript-version"
    with pytest.raises(ValueError, match="Transcript text differs"):
        compute_ground_truth_metrics(predictions, _labels())


def test_load_ground_truth_checks_required_columns(tmp_path):
    path = tmp_path / "ground_truth.csv"
    path.write_text("transcript_id,topic_label\nSYN-001,Other\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_ground_truth(path)
