from dataclasses import replace
from pathlib import Path

from config.settings import Settings, get_settings
from src import storage
from src.demo_analyzer import analyze_record
from src.llm_client import SetupError
from src.pipeline import run_pipeline
from src.synthetic import write_synthetic
from src.topic_classifier import load_taxonomy

TAX = load_taxonomy(get_settings().taxonomy_path)


def _setup(tmp_path):
    src_path = write_synthetic(tmp_path / "synthetic", n_total=24)
    return src_path, tmp_path / "out"


def test_demo_pipeline_end_to_end_and_provenance(tmp_path):
    src_path, out = _setup(tmp_path)
    meta = run_pipeline(src_path, "demo", out, skip_clusters=True)
    assert meta["records_loaded"] == 24 and meta["records_analyzable"] == 24
    assert meta["processed_this_run"] == 24 and meta["failed"] == 0
    assert "SYNTHETIC" in meta["disclaimer"] and "HEURISTIC" in meta["disclaimer"]
    data = storage.load_processed(out)
    df = data["df"]
    assert len(df) == 24 and set(df["data_origin"]) == {"synthetic"} and set(df["analysis_mode"]) == {"heuristic_demo"}
    assert df["sentiment_score"].between(-1, 1).all()
    assert data["quality"]["empty_text"]["count"] == 0 and data["quality"]["duplicate_text"]["count"] == 0


def test_resume_skips_done_work_and_detects_changed_text(tmp_path):
    src_path, out = _setup(tmp_path)
    run_pipeline(src_path, "demo", out, skip_clusters=True)
    again = run_pipeline(src_path, "demo", out, skip_clusters=True)
    assert again["processed_this_run"] == 0 and again["skipped_resumed"] == 24
    fresh = run_pipeline(src_path, "demo", out, skip_clusters=True, resume=False)
    assert fresh["processed_this_run"] == 24


def test_one_bad_record_does_not_stop_the_run_and_failures_are_retried(tmp_path):
    src_path, out = _setup(tmp_path)

    def flaky(rec):
        if rec["transcript_id"] == "SYN-005":
            raise RuntimeError("boom")
        return analyze_record(rec, TAX)

    meta = run_pipeline(src_path, "demo", out, analyzer=flaky, skip_clusters=True)
    assert meta["failed"] == 1 and meta["processed_this_run"] == 23
    assert storage.read_jsonl(out / "failures.jsonl")[0]["transcript_id"] == "SYN-005"
    assert "SYN-005" not in set(storage.load_processed(out)["df"]["transcript_id"])
    meta2 = run_pipeline(src_path, "demo", out, skip_clusters=True)
    assert meta2["processed_this_run"] == 1 and meta2["failed"] == 0 and meta2["analyses_total"] == 24


def test_corrupt_checkpoint_line_is_skipped(tmp_path):
    src_path, out = _setup(tmp_path)
    run_pipeline(src_path, "demo", out, skip_clusters=True)
    with open(out / "analyses.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"transcript_id": "SYN-0')  # simulated crash mid-write
    assert len(storage.read_jsonl(out / "analyses.jsonl")) == 24


def test_llm_mode_without_key_raises_setup_error_and_writes_nothing(tmp_path):
    src_path, out = _setup(tmp_path)
    try:
        run_pipeline(src_path, "llm", out, settings=replace(get_settings(), openai_api_key=None))
        assert False, "expected SetupError"
    except SetupError:
        assert not (out / "analyses.jsonl").exists()


def test_load_processed_returns_none_when_empty(tmp_path):
    assert storage.load_processed(tmp_path) is None


def test_clusters_written_when_enabled(tmp_path):
    src_path, out = _setup(tmp_path)
    run_pipeline(src_path, "demo", out)
    cl = storage.read_json(out / "clusters.json")
    assert cl["clusters"] and sum(c["size"] for c in cl["clusters"]) == 24 and cl["method"]
