"""Lexicon-based sentiment used by the heuristic demo mode.

Scores are in [-1, 1] and are NOT calibrated probabilities: they are (pos - neg) / (pos + neg + 2) over
weighted lexicon hits, with a simple 2-token negation flip. Known blind spots: sarcasm, domain jargon,
long-range negation, multi-word expressions. Customer sentiment is computed from customer/client turns only,
so a polite agent does not make an unhappy customer look happy, and a technical topic is not assumed negative.
"""
import re

from src.preprocessing import parse_turns

NEG = {"frustrating": 1.5, "frustrated": 1.5, "unacceptable": 2, "terrible": 2, "angry": 2, "disappointed": 1.5,
       "disappointing": 1.5, "worried": 1, "concerned": 1, "blocked": 1, "blocking": 1, "broken": 1.5, "slow": 1,
       "outage": 1.5, "failing": 1, "failed": 1, "fails": 1, "locked": 1, "wrong": 1, "ridiculous": 2, "cancel": 1.5,
       "expensive": 1, "unhappy": 1.5, "delayed": 1, "delay": 1, "stuck": 1, "losing": 1.5, "painful": 1.5,
       "hesitant": 1, "unfortunately": 1, "mess": 1.5, "crashed": 1.5, "nightmare": 2, "annoyed": 1.5,
       "costing": 1, "hurting": 1.5, "unresolved": 1, "gap": 0.5, "optimistic": 0}
POS = {"thanks": 0.5, "thank": 0.5, "great": 1.5, "appreciate": 1.5, "perfect": 1.5, "happy": 1.5, "love": 1.5,
       "loves": 1.5, "helpful": 1, "resolved": 1, "works": 1, "fixed": 1, "excellent": 2, "glad": 1, "impressed": 1.5,
       "awesome": 2, "good": 1, "nice": 1, "pleased": 1.5, "excited": 1.5, "wonderful": 2, "solid": 1,
       "improved": 1, "absolutely": 0.5, "wonderful ": 0}
NEGATIONS = {"not", "no", "never", "isn't", "wasn't", "don't", "doesn't", "didn't", "can't", "cannot", "won't", "hardly", "haven't"}
CUSTOMER_ROLES = ("customer", "client")


# Words that describe routine engineering/operational problems. In internal calls they are down-weighted so that
# talking about a blocker or delay is not automatically scored as negative sentiment.
OPERATIONAL = {"blocked", "blocking", "failing", "failed", "fails", "delayed", "delay", "outage", "crashed", "stuck",
               "unresolved", "gap", "slow", "broken", "wrong", "mess"}


def score_text(text: str, operational_weight: float = 1.0) -> dict:
    """Return {'score', 'pos', 'neg'} for a piece of text. ``operational_weight`` scales OPERATIONAL words."""
    tokens = re.findall(r"[a-z']+", text.lower().replace("\u2019", "'"))
    pos = neg = 0.0
    for i, tok in enumerate(tokens):
        w_pos, w_neg = POS.get(tok, 0), NEG.get(tok, 0) * (operational_weight if tok in OPERATIONAL else 1.0)
        if not (w_pos or w_neg):
            continue
        if any(t in NEGATIONS for t in tokens[max(0, i - 2):i]):
            w_pos, w_neg = w_neg * 0.7, w_pos * 0.7  # "not good" -> mildly negative; "not blocked" -> mildly positive
        pos += w_pos
        neg += w_neg
    return {"score": max(-1.0, min(1.0, (pos - neg) / (pos + neg + 2.0))), "pos": pos, "neg": neg}


def label_from(score: float, pos: float, neg: float) -> str:
    if pos >= 2 and neg >= 2 and abs(score) < 0.4:
        return "Mixed"
    if score >= 0.25:
        return "Positive"
    if score <= -0.25:
        return "Negative"
    return "Neutral"


def analyze_sentiment(text: str, call_type: str) -> dict:
    """Overall call tone for all turns, plus customer sentiment (customer/client turns) when roles exist."""
    turns = parse_turns(text)
    overall = score_text(" ".join(u for _, u in turns), operational_weight=0.25 if call_type == "internal" else 1.0)
    out = {"score": overall["score"], "label": label_from(**overall),
           "customer_score": None, "customer_label": None}
    cust = [u for s, u in turns if s.lower().startswith(CUSTOMER_ROLES)]
    if call_type != "internal" and cust:
        c = score_text(" ".join(cust))
        out["customer_score"], out["customer_label"] = c["score"], label_from(**c)
    return out
