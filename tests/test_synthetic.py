import csv
import json
import re
from collections import Counter

from src.synthetic import write_synthetic


def test_default_dataset_matches_assignment_contract(tmp_path):
    transcript_path = write_synthetic(tmp_path)
    transcript_json = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcripts = transcript_json["transcripts"]
    labels = json.loads((tmp_path / "ground_truth.json").read_text(encoding="utf-8"))["labels"]

    assert len(transcripts) == 100
    assert Counter(row["call_type"] for row in transcripts) == {
        "support": 40, "external": 35, "internal": 25}
    assert all(row["data_origin"] == "synthetic" for row in transcripts)
    assert all(row["transcript_id"].startswith("SYN-") for row in transcripts)
    assert all(row["call_date"].endswith("+00:00") for row in transcripts)
    assert all("topic_label" not in row and "sentiment_label" not in row for row in transcripts)
    assert all(10 <= len(row["transcript_text"].splitlines()) <= 20 for row in transcripts)
    assert all(re.match(r"^\[\d{2}:\d{2}\] [^:]+: .+", line)
               for row in transcripts for line in row["transcript_text"].splitlines())

    assert len(labels) == 100
    assert {label["transcript_id"] for label in labels} == {row["transcript_id"] for row in transcripts}
    assert all(label["data_origin"] == "synthetic" for label in labels)
    assert {label["sentiment_label"] for label in labels} == {"Positive", "Neutral", "Negative", "Mixed"}

    with (tmp_path / "transcripts.csv").open(encoding="utf-8", newline="") as file:
        csv_transcripts = list(csv.DictReader(file))
    with (tmp_path / "ground_truth.csv").open(encoding="utf-8", newline="") as file:
        csv_labels = list(csv.DictReader(file))
    assert len(csv_transcripts) == len(csv_labels) == 100
    assert {row["transcript_id"] for row in csv_transcripts} == {row["transcript_id"] for row in labels}
    json_by_id = {row["transcript_id"]: row for row in transcripts}
    csv_by_id = {row["transcript_id"]: row for row in csv_transcripts}
    assert all(csv_by_id[tid]["transcript_text"] == row["transcript_text"] and
               csv_by_id[tid]["call_date"] == row["call_date"] for tid, row in json_by_id.items())
    assert {row["transcript_id"]: row["text_hash"] for row in csv_labels} == {
        row["transcript_id"]: row["text_hash"] for row in labels}
