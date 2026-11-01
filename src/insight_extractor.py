"""Deterministic extraction helpers (regex) + feature-request grouping + cross-team candidate matching.

These power the heuristic demo mode and the dashboard's stakeholder insights. They only return sentences that
literally appear in the transcript, so every item is traceable.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import parse_turns

_I = re.IGNORECASE
FEATURE_PATTERNS = [re.compile(p, _I) for p in [
    r"\bwould be (?:great|nice|helpful|amazing) if\b", r"\bwe need (?:a way to|the ability|to be able to|support for)\b",
    r"\bcan you (?:add|support|build)\b", r"\bis there a way to\b", r"\bdo you (?:support|have|plan)\b",
    r"\b(?:doesn't|does not|don't|do not) (?:support|have|offer)\b", r"\bno way to\b", r"\bmissing\b",
    r"\bwish (?:you|it|we)\b", r"\bfeature request\b", r"\bwe'd love\b", r"\bany plans (?:to|for)\b"]]
RENEWAL_PATTERNS = [re.compile(p, _I) for p in [
    r"\bcancel\w*", r"\bnot renew\w*", r"\bwon't renew\b", r"\bswitch\w*\b", r"\bevaluating\b", r"\bcompar(?:e|ing)\b",
    r"\balternatives?\b", r"\bcompetitors?\b", r"\banother vendor\b", r"\bother vendors\b", r"\bon the fence\b",
    r"\bprice increase\b", r"\btoo expensive\b", r"\bbudget (?:is tight|cut|freeze)\b", r"\bcut our budget\b",
    r"\bneed a discount\b", r"\bnot optimistic\b", r"\bhaven't decided\b", r"\blook at (?:alternatives|\w+)\b"]]
RENEWAL_EXCLUDE = re.compile(r"\b(?:happy|glad|excited|plan(?:ning)?|ready) to renew\b|\bwill renew\b|\bwe're renewing\b", _I)
BLOCKER_PATTERNS = [re.compile(p, _I) for p in [
    r"\bblocked\b", r"\bblocker\b", r"\bwaiting (?:on|for)\b", r"\bdelayed\b", r"\bslip\w*\b", r"\bflaky\b",
    r"\bno capacity\b", r"\bcan'?t ship\b", r"\brolled? back\b", r"\brollback\b", r"\bdependency\b",
    r"\bunresolved\b", r"\bfailed twice\b", r"\bstill failing\b", r"\bbottleneck\b"]]
ACTION_PATTERNS = [re.compile(p, _I) for p in [
    r"\bI'll\b", r"\bI will\b", r"\bwe'll\b", r"\bwe will\b", r"\blet's\b", r"\baction item\b", r"\bfollow up\b",
    r"\bwill send\b", r"\bsomeone should\b", r"\bcan you take\b"]]
ESCALATION_PATTERNS = [re.compile(p, _I) for p in [
    r"\bescalat\w*", r"\bunacceptable\b", r"\bexecutive\b", r"\bsev ?[12]\b", r"\blegal\b", r"\bcancel\w*"]]
PAIN_PATTERN = re.compile(r"\b(?:can't|cannot|unable|error|errors|fail\w*|locked out|slow|stuck|wrong|twice|down|timeouts?|"
                          r"blocked|not working|broken|losing|worried|disappointed|frustrating|unacceptable|angry)\b", _I)
CUSTOMER_ROLES = ("customer", "client")
# Conversational filler that carries no topical meaning
STOP = list(ENGLISH_STOP_WORDS | {"ll", "ve", "let", "okay", "thanks", "thank", "follow", "hour", "today", "hi", "hello", "sorry", "fine", "yes", "great"})


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def turn_sentences(text: str, customer_only: bool = False) -> list[tuple[str, str]]:
    """(speaker, sentence) pairs, optionally restricted to customer/client turns."""
    out = []
    for speaker, utt in parse_turns(text):
        if customer_only and not speaker.lower().startswith(CUSTOMER_ROLES):
            continue
        out.extend((speaker, s) for s in split_sentences(utt))
    return out


def _matches(sents, patterns, exclude=None, limit=6) -> list[str]:
    out = []
    for _, s in sents:
        if exclude is not None and exclude.search(s):
            continue
        if any(p.search(s) for p in patterns) and s not in out:
            out.append(s[:260])
    return out[:limit]


def extract_signals(text: str, call_type: str) -> dict:
    """Heuristic extraction. Customer-facing calls use customer/client turns for customer-side signals."""
    internal = call_type == "internal"
    cust = turn_sentences(text, customer_only=not internal)
    allsents = turn_sentences(text)
    pain = [s[:260] for _, s in cust if PAIN_PATTERN.search(s)][:4]
    return {
        "feature_requests": [] if internal else _matches(cust, FEATURE_PATTERNS),
        "renewal_risk_signals": _matches(cust, RENEWAL_PATTERNS, RENEWAL_EXCLUDE),
        "engineering_blockers": _matches(allsents, BLOCKER_PATTERNS) if internal else [],
        "action_items": _matches(allsents, ACTION_PATTERNS),
        "escalations": _matches(allsents, ESCALATION_PATTERNS),
        "pain_points": pain,
    }


def group_feature_requests(requests: list[tuple[str, str]], distance: float = 0.75) -> pd.DataFrame:
    """Group near-duplicate feature requests (TF-IDF cosine, agglomerative). Input: [(transcript_id, text)].

    Returns one row per group: representative text, number of distinct transcripts, transcript_ids. Ranked by frequency.
    This is lexical grouping: it will miss paraphrases with no shared words (use embeddings for that).
    """
    cols = ["group", "representative", "n_transcripts", "transcript_ids", "examples"]
    if not requests:
        return pd.DataFrame(columns=cols)
    from sklearn.cluster import AgglomerativeClustering
    texts = [t for _, t in requests]
    if len(texts) == 1:
        labels = np.array([0])
    else:
        try:
            X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
            labels = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, metric="cosine",
                                             linkage="average").fit_predict(X.toarray())
        except ValueError:  # no usable vocabulary (e.g. one-word requests): group only exact duplicates
            uniq = {t.lower(): i for i, t in enumerate(dict.fromkeys(texts))}
            labels = np.array([uniq[t.lower()] for t in texts])
    rows = []
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        ids = sorted({requests[i][0] for i in idx})
        rep = min((texts[i] for i in idx), key=len)  # shortest = clearest statement
        rows.append({"group": int(lab), "representative": rep, "n_transcripts": len(ids),
                     "transcript_ids": ids, "examples": [texts[i] for i in idx[:3]]})
    return pd.DataFrame(rows, columns=cols).sort_values(["n_transcripts", "representative"], ascending=[False, True]).reset_index(drop=True)


def cross_team_candidates(df: pd.DataFrame, threshold: float = 0.12, per_internal: int = 3) -> pd.DataFrame:
    """CANDIDATE links between engineering (internal) calls with blockers/incidents and customer-facing calls.

    Similarity = TF-IDF cosine between internal (full text) and customer-facing
    (summary + pain points) text. These are candidate matches only: no explicit identifier verifies them.
    """
    cols = ["internal_id", "customer_facing_id", "similarity", "shared_terms"]
    internal = df[(df["call_type"] == "internal") & (df["engineering_blockers"].map(len) > 0)]
    external = df[df["call_type"].isin(["support", "external"]) & (df["pain_points"].map(len) > 0)]
    if internal.empty or external.empty:
        return pd.DataFrame(columns=cols)

    text_col = "redacted_text" if "redacted_text" in df else "summary"
    a = internal[text_col].tolist()
    b = external[text_col].tolist()
    try:
        vec = TfidfVectorizer(stop_words=STOP, min_df=1).fit(a + b)
    except ValueError:  # texts contain no usable words
        return pd.DataFrame(columns=cols)
    A, B = vec.transform(a), vec.transform(b)
    sims = cosine_similarity(A, B)
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for i, (_, ri) in enumerate(internal.iterrows()):
        for j in np.argsort(-sims[i])[:per_internal]:
            if sims[i, j] >= threshold:
                shared = vocab[(A[i].toarray()[0] > 0) & (B[j].toarray()[0] > 0)]
                rows.append({"internal_id": ri["transcript_id"], "customer_facing_id": external.iloc[j]["transcript_id"],
                             "similarity": round(float(sims[i, j]), 3), "shared_terms": ", ".join(shared[:6])})
    return pd.DataFrame(rows, columns=cols).sort_values("similarity", ascending=False).reset_index(drop=True)
