import pandas as pd

from src.preprocessing import (build_quality_report, normalize_whitespace, parse_turns, preprocess, quote_in_text,
                               redact_pii)


def _df(rows):
    return pd.DataFrame([{"transcript_id": i, "call_type": "support", "call_date": pd.NaT, "transcript_text": t,
                          "customer_id": None, "source_file": "f", "data_origin": "original",
                          "call_type_inferred": False, "id_generated": False} for i, t in rows])


def test_whitespace_preserves_negation_and_case():
    assert normalize_whitespace("Agent:   It is  NOT working \n\n\n Customer: no  way") == "Agent: It is NOT working\nCustomer: no way"


def test_parse_turns_handles_timestamps_and_continuations():
    turns = parse_turns("[00:01] Agent: Hello\nCustomer: first\nsecond part")
    assert turns == [("Agent", "Hello"), ("Customer", "first second part")]


def test_redaction():
    out = redact_pii("mail a@b.com call 555-123-4567 card 4111 1111 1111 1111 ssn 123-45-6789")
    assert "a@b.com" not in out and "555-123-4567" not in out and "4111" not in out and "123-45-6789" not in out


def test_empty_and_duplicate_records_are_excluded_with_reasons():
    full = preprocess(_df([("1", "Agent: hello there friend"), ("2", "Agent: hello there friend"), ("1", "Agent: other"),
                           ("3", "   "), ("4", None)]))
    reasons = dict(zip(full["transcript_id"] + "/" + full.index.astype(str), full["exclude_reason"]))
    assert reasons["1/0"] is None and reasons["2/1"] == "duplicate_text" and reasons["1/2"] == "duplicate_id"
    assert reasons["3/3"] == "empty_text" and reasons["4/4"] == "empty_text"
    q = build_quality_report(full, [{"error": "x"}])
    assert q["empty_text"]["count"] == 2 and q["duplicate_text"]["count"] == 1 and q["duplicate_ids"]["count"] == 1
    assert q["analyzable_records"] == 1 and q["malformed_records"]["count"] == 1 and q["missing_dates"]["count"] == 5


def test_length_flags():
    full = preprocess(_df([("1", "Agent: hi"), ("2", " ".join(["word"] * 40)), ("3", " ".join(["more"] * 41))]))
    assert full.loc[0, "length_flag"] == "very_short"


def test_quote_in_text():
    t = "Customer: We are \u201cblocked\u201d   on the Platform team."
    assert quote_in_text('we are "blocked" on the platform team', t)
    assert not quote_in_text("totally invented sentence", t) and not quote_in_text("", t)
