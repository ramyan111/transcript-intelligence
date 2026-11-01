import pydantic

from src.schemas import TranscriptAnalysis
from src.sentiment_analyzer import analyze_sentiment, score_text


def test_scores_are_bounded():
    for t in ["", "great excellent perfect wonderful " * 20, "terrible unacceptable angry nightmare " * 20]:
        assert -1.0 <= score_text(t)["score"] <= 1.0


def test_negation_flips_polarity():
    assert score_text("this is good")["score"] > 0 > score_text("this is not good")["score"]


def test_customer_sentiment_is_separate_from_overall_tone():
    text = "Agent: Thanks so much, happy to help, great, wonderful, glad.\nCustomer: This is unacceptable and I am angry and frustrated."
    s = analyze_sentiment(text, "support")
    assert s["customer_label"] == "Negative"
    assert s["customer_score"] < s["score"]  # polite agent does not make the customer happy


def test_internal_calls_have_no_customer_sentiment_and_blockers_are_not_automatically_negative():
    s = analyze_sentiment("Eng: The release is blocked.\nPM: Okay, let's plan the rollback.", "internal")
    assert s["customer_label"] is None and s["label"] != "Negative"


def test_positive_customer():
    assert analyze_sentiment("Customer: Great, thanks! It works and I appreciate the help.", "support")["customer_label"] == "Positive"


def test_schema_rejects_out_of_range_score():
    base = dict(transcript_id="1", call_type="support", primary_topic="Other", sentiment_label="Neutral", summary="s",
                explanation="e", analysis_mode="llm")
    TranscriptAnalysis(sentiment_score=0.5, **base)
    for bad in (1.5, -1.01):
        try:
            TranscriptAnalysis(sentiment_score=bad, **base)
            assert False
        except pydantic.ValidationError:
            pass
    try:
        TranscriptAnalysis(sentiment_score=0, **{**base, "sentiment_label": "Ecstatic"})
        assert False
    except pydantic.ValidationError:
        pass
