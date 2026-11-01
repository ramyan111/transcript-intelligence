import pandas as pd

from src import analytics
from src.insight_extractor import extract_signals, group_feature_requests


def make_df(rows):
    out = []
    for i, r in enumerate(rows):
        base = {"transcript_id": f"T{i}", "call_type": "support", "primary_topic": "Other", "sentiment_label": "Neutral",
                "sentiment_score": 0.0, "customer_sentiment": None, "customer_sentiment_score": None,
                "escalation_required": False, "feature_requests": [], "renewal_risk_signals": [], "engineering_blockers": [],
                "pain_points": [], "action_items": [], "summary": "s", "secondary_topics": [], "call_date": pd.NaT,
                "redacted_text": "x"}
        base.update(r)
        out.append(base)
    return pd.DataFrame(out)


def test_denominators_are_group_sizes_not_global():
    df = make_df([{"call_type": "support", "sentiment_label": "Negative"}] * 2 + [{"call_type": "support"}] * 2
                 + [{"call_type": "external", "sentiment_label": "Negative"}] + [{"call_type": "external"}] * 5)
    s = analytics.sentiment_by_group(df, "call_type").set_index("call_type")
    assert s.loc["support", "negative_pct"] == 50.0 and s.loc["external", "negative_pct"] == round(100 / 6, 1)
    c = analytics.negative_concentration(df, "call_type").set_index("call_type")
    assert c.loc["support", "share_of_all_negative_pct"] == round(100 * 2 / 3, 1)
    assert c.loc["support", "negative_rate_pct"] == 50.0
    assert bool(s.loc["support", "small_sample"])


def test_null_customer_labels_are_excluded_from_denominator():
    df = make_df([{"customer_sentiment": "Negative", "customer_sentiment_score": -0.5}, {"customer_sentiment": None}])
    s = analytics.sentiment_by_group(df, "call_type", "customer_sentiment", "customer_sentiment_score")
    assert s.loc[0, "n"] == 1 and s.loc[0, "negative_pct"] == 100.0


def test_trend_only_with_reliable_dates():
    assert analytics.sentiment_trend(make_df([{}, {}])) is None
    dates = pd.to_datetime(["2026-01-05", "2026-02-05", "2026-03-05", "2026-03-06"])
    df = make_df([{"call_date": d} for d in dates])
    t = analytics.sentiment_trend(df)
    assert t is not None and len(t) == 3 and t["n"].sum() == 4
    assert analytics.sentiment_trend(make_df([{"call_date": dates[0]}, {}, {}, {}])) is None  # 25% coverage


def test_overview_metrics_count_from_data():
    df = make_df([{"escalation_required": True, "feature_requests": ["a", "b"], "renewal_risk_signals": ["x"]}, {}])
    m = analytics.overview_metrics(df)
    assert (m["total"], m["escalations"], m["feature_requests"], m["renewal_flagged"]) == (2, 1, 2, 1)


def test_insights_have_required_fields_and_evidence_links():
    df = make_df([{"call_type": "external", "renewal_risk_signals": ["We may switch vendors."], "feature_requests": ["Is there a way to get CSV export?"]},
                  {"call_type": "support", "escalation_required": True, "pain_points": ["It is broken."]},
                  {"call_type": "internal", "engineering_blockers": ["We are blocked."]}])
    ins = analytics.build_insights(df)
    assert ins
    for i in ins:
        for k in ("title", "metric", "stakeholder", "transcript_ids", "evidence", "interpretation", "recommended_action", "caveat"):
            assert i[k] not in (None, "")
        assert set(e["transcript_id"] for e in i["evidence"]) <= set(df["transcript_id"])
    assert "1 of 2 customer-facing" in next(i for i in ins if i["id"] == "renewal_risk")["metric"]


def test_insights_on_empty_and_no_internal_calls():
    assert analytics.build_insights(make_df([])) == []
    assert not any(i["id"] == "eng_blockers" for i in analytics.build_insights(make_df([{}])))


def test_feature_grouping_dedupes_per_transcript_and_ranks():
    g = group_feature_requests([("A", "Is there a way to get CSV export?"), ("B", "Is there a way to get CSV export?"),
                                ("B", "Can you add CSV export please?"), ("C", "Do you support SAML group mapping?")])
    assert g.iloc[0]["n_transcripts"] == 2 and set(g.iloc[0]["transcript_ids"]) == {"A", "B"}
    assert group_feature_requests([]).empty


def test_extraction_only_uses_customer_turns_and_excludes_happy_renewals():
    text = "Agent: We could cancel the old plan for you.\nClient: We are happy to renew and love it."
    assert extract_signals(text, "external")["renewal_risk_signals"] == []
    risky = extract_signals("Client: We are evaluating a competitor.", "external")
    assert risky["renewal_risk_signals"] == ["We are evaluating a competitor."]
