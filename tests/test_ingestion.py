import json

import pandas as pd

from src.ingestion import infer_call_type_from_path, load_transcripts, normalize_call_type
from pathlib import Path


def test_csv_with_aliases_and_missing_values(tmp_path):
    p = tmp_path / "calls.csv"
    p.write_text("id,type,date,transcript\nA1,Customer Support,2026-01-05,Agent: hi\nA2,,,Customer: hello\n", encoding="utf-8")
    df, bad = load_transcripts(p)
    assert bad == [] and list(df["transcript_id"]) == ["A1", "A2"]
    assert df.loc[0, "call_type"] == "support" and not df.loc[0, "call_type_inferred"]
    assert df.loc[1, "call_type"] == "unknown"  # missing and not inferable: never invented
    assert pd.isna(df.loc[1, "call_date"])


def test_json_list_wrapped_dict_and_turn_lists(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"transcripts": [{"transcript_id": "T1", "call_type": "internal",
                 "transcript": [{"speaker": "Eng", "text": "we are blocked"}]}]}), encoding="utf-8")
    df, _ = load_transcripts(p)
    assert df.loc[0, "transcript_text"] == "Eng: we are blocked"


def test_jsonl_and_malformed_records(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"id": "1", "text": "Agent: a", "call_type": "external"}\nnot json\n', encoding="utf-8")
    df, bad = load_transcripts(p)
    assert len(df) == 1 and len(bad) == 1


def test_invalid_json_file_is_reported_not_fatal(tmp_path):
    (tmp_path / "bad.json").write_text("{oops", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("Agent: hi", encoding="utf-8")
    df, bad = load_transcripts(tmp_path)
    assert len(df) == 1 and len(bad) == 1 and "bad.json" in bad[0]["source_file"]


def test_txt_folder_infers_call_type_from_folder_and_flags_it(tmp_path):
    (tmp_path / "support").mkdir()
    (tmp_path / "support" / "call1.txt").write_text("Agent: hi\nCustomer: help", encoding="utf-8")
    df, _ = load_transcripts(tmp_path)
    assert df.loc[0, "call_type"] == "support" and df.loc[0, "call_type_inferred"]
    assert df.loc[0, "transcript_id"] == "call1"


def test_generated_ids_are_flagged(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps([{"text": "Agent: hi"}]), encoding="utf-8")
    df, _ = load_transcripts(p)
    assert df.loc[0, "id_generated"]


def test_origin_detection_and_override(tmp_path):
    d = tmp_path / "synthetic"
    d.mkdir()
    (d / "a.txt").write_text("Agent: hi", encoding="utf-8")
    assert load_transcripts(d)[0].loc[0, "data_origin"] == "synthetic"
    assert load_transcripts(d, default_origin="original")[0].loc[0, "data_origin"] == "original"


def test_missing_path_raises(tmp_path):
    try:
        load_transcripts(tmp_path / "nope")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_call_type_helpers():
    assert normalize_call_type("Customer Support") == "support"
    assert normalize_call_type("weird") == "unknown" and normalize_call_type("") is None
    assert infer_call_type_from_path(Path("internal/a.txt")) == "internal"
