from types import SimpleNamespace

from config.settings import Settings, get_settings
from src import llm_client
from src.llm_client import LLMClient, LLMRefusal, SetupError, to_analysis
from src.schemas import LLMAnalysis, LLMEvidence
from src.topic_classifier import load_taxonomy, rule_classify, topic_names, validate_topic, validate_topics

TAX = load_taxonomy(get_settings().taxonomy_path)
ALLOWED = topic_names(TAX)
REC = {"transcript_id": "T1", "call_type": "support", "redacted_text": "Customer: SSO login is broken.\nAgent: sorry"}


def _out(**kw):
    base = dict(primary_topic="Authentication & Access", secondary_topics=[], sentiment_label="Negative", sentiment_score=-0.5,
                customer_sentiment="Negative", summary="s", pain_points=[], feature_requests=[], renewal_risk_signals=[],
                engineering_blockers=[], action_items=[], escalation_required=False, supporting_evidence=[], explanation="e")
    base.update(kw)
    return LLMAnalysis(**base)


def test_validate_topic_remaps_unknown_to_other():
    assert validate_topic("authentication & access", ALLOWED) == ("Authentication & Access", False)
    assert validate_topic("Made Up Topic", ALLOWED) == ("Other", True)
    assert validate_topic(None, ALLOWED) == ("Other", True)


def test_validate_topics_filters_secondaries():
    p, s, r = validate_topics("Billing & Payments", ["Billing & Payments", "bogus", "Performance & Latency", "Performance & Latency"], ALLOWED)
    assert (p, s, r) == ("Billing & Payments", ["Performance & Latency"], False)


def test_rule_classify_and_other_fallback():
    assert rule_classify("We got charged twice and need a refund for the invoice", TAX)[0] == "Billing & Payments"
    assert rule_classify("hello there", TAX)[0] == "Other"


def test_to_analysis_clamps_score_remaps_topic_and_verifies_evidence():
    out = _out(primary_topic="Nonsense", sentiment_score=3.7,
               supporting_evidence=[LLMEvidence(claim="c1", quote="SSO login is broken"), LLMEvidence(claim="c2", quote="invented quote")])
    a = to_analysis(out, REC, TAX, "m")
    assert a.sentiment_score == 1.0 and a.primary_topic == "Other" and a.topic_was_remapped
    assert [e.verified for e in a.supporting_evidence] == [True, False]
    assert a.analysis_mode == "llm"


def test_internal_calls_have_no_customer_sentiment():
    a = to_analysis(_out(), {**REC, "call_type": "internal"}, TAX, "m")
    assert a.customer_sentiment is None


def test_missing_api_key_is_a_setup_error_not_fake_results():
    try:
        LLMClient(Settings(openai_api_key=None))
        assert False, "expected SetupError"
    except SetupError as exc:
        assert "OPENAI_API_KEY" in str(exc)


def _fake_client(responses):
    calls = {"n": 0}

    def parse(**kw):
        calls["n"] += 1
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=r)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    return client, calls


def test_retries_then_succeeds_without_real_sleep():
    good = SimpleNamespace(parsed=_out(), refusal=None)
    client, calls = _fake_client([TimeoutError("t"), good])
    real_time, llm_client.time = llm_client.time, SimpleNamespace(sleep=lambda s: None)
    try:
        res = LLMClient(Settings(openai_api_key="k", max_retries=2), client).analyze("support", "text", TAX)
    finally:
        llm_client.time = real_time
    assert calls["n"] == 2 and res.primary_topic == "Authentication & Access"


def test_gives_up_after_max_retries_and_refusals_are_not_retried():
    real_time, llm_client.time = llm_client.time, SimpleNamespace(sleep=lambda s: None)
    try:
        client, calls = _fake_client([ValueError("x")] * 3)
        try:
            LLMClient(Settings(openai_api_key="k", max_retries=2), client).analyze("support", "t", TAX)
            assert False
        except ValueError:
            assert calls["n"] == 3
        client, calls = _fake_client([SimpleNamespace(parsed=None, refusal="no")])
        try:
            LLMClient(Settings(openai_api_key="k", max_retries=2), client).analyze("support", "t", TAX)
            assert False
        except LLMRefusal:
            assert calls["n"] == 1
    finally:
        llm_client.time = real_time
