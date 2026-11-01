"""Heuristic DEMO analyzer: keyword topics + lexicon sentiment + regex extraction.

This is NOT an LLM and its output is labeled ``analysis_mode="heuristic_demo"`` everywhere. It exists so the
pipeline, dashboard and tests can run without an API key. Do not present its output as model results.
"""
from src.insight_extractor import extract_signals, turn_sentences
from src.preprocessing import quote_in_text
from src.schemas import Evidence, TranscriptAnalysis
from src.sentiment_analyzer import analyze_sentiment
from src.topic_classifier import Topic, rule_classify, topic_names, validate_topics


def _sentence_with(sents, needle: str) -> str | None:
    for _, s in sents:
        if needle.lower() in s.lower():
            return s
    return None


def analyze_record(record: dict, taxonomy: list[Topic]) -> TranscriptAnalysis:
    text = record["redacted_text"]
    call_type = record["call_type"]
    primary, secondary, matched = rule_classify(text, taxonomy)
    primary, secondary, remapped = validate_topics(primary, secondary, topic_names(taxonomy))
    sent = analyze_sentiment(text, call_type)
    sig = extract_signals(text, call_type)

    sents = turn_sentences(text)
    summary_pool = turn_sentences(text, customer_only=call_type != "internal") or sents
    summary = next((s for _, s in summary_pool if len(s.split()) >= 8), summary_pool[0][1] if summary_pool else "")[:240]

    ev: list[Evidence] = []
    if primary in matched and matched[primary]:
        term = matched[primary][0]
        q = _sentence_with(sents, term)
        if q:
            ev.append(Evidence(claim=f"Topic '{primary}' (keyword match: '{term}')", quote=q[:260]))
    for key, label in [("renewal_risk_signals", "Renewal-risk signal"), ("feature_requests", "Feature request"),
                       ("engineering_blockers", "Engineering blocker"), ("escalations", "Escalation language")]:
        for q in sig[key][:2]:
            ev.append(Evidence(claim=label, quote=q))
    for e in ev:
        e.verified = quote_in_text(e.quote, text)

    explanation = (f"HEURISTIC DEMO (not an LLM). Topic by keyword counts ({', '.join(matched.get(primary, [])) or 'no keyword hits'}); "
                   f"sentiment by weighted lexicon (score is not a calibrated probability); "
                   f"signals by regex over transcript sentences.")
    return TranscriptAnalysis(
        transcript_id=record["transcript_id"], call_type=call_type, primary_topic=primary, secondary_topics=secondary,
        sentiment_label=sent["label"], sentiment_score=round(sent["score"], 3),
        customer_sentiment=sent["customer_label"],
        customer_sentiment_score=None if sent["customer_score"] is None else round(sent["customer_score"], 3),
        summary=summary, pain_points=sig["pain_points"], feature_requests=sig["feature_requests"],
        renewal_risk_signals=sig["renewal_risk_signals"], engineering_blockers=sig["engineering_blockers"],
        action_items=sig["action_items"], escalation_required=bool(sig["escalations"]),
        supporting_evidence=ev, explanation=explanation, analysis_mode="heuristic_demo",
        llm_model=None, topic_was_remapped=remapped)
