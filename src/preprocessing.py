"""Preprocessing: whitespace normalization, speaker-turn parsing, PII redaction, flags, data-quality report.

Normalization never lowercases or strips negations: meaning must survive for sentiment analysis.
"""
import hashlib
import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TS_PREFIX = re.compile(r"^\s*\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*")
_TURN_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 ._\-/()]{0,40}?)\s*:\s*(\S.*)$")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,2}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\w)")

SHORT_WORDS = 20


def normalize_whitespace(text: str | None) -> str:
    """Collapse runs of spaces/tabs, drop blank lines, keep one turn per line."""
    if not text:
        return ""
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(ln for ln in lines if ln)


def parse_turns(text: str) -> list[tuple[str, str]]:
    """Split ``Speaker: utterance`` lines into (speaker, utterance); unlabeled lines continue the previous turn."""
    turns: list[tuple[str, str]] = []
    for line in normalize_whitespace(text).split("\n"):
        line = _TS_PREFIX.sub("", line)
        m = _TURN_RE.match(line)
        if m:
            turns.append((m.group(1).strip(), m.group(2).strip()))
        elif turns:
            turns[-1] = (turns[-1][0], turns[-1][1] + " " + line)
        else:
            turns.append(("", line))
    return turns


def redact_pii(text: str) -> str:
    """Mask emails, SSN-like, card-like and phone-like numbers. Names are NOT redacted (documented limitation)."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _SSN.sub("[SSN]", text)
    text = _CARD.sub("[CARD]", text)
    return _PHONE.sub("[PHONE]", text)


def _norm_for_match(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip(" \"'.,;:!?-")


def quote_in_text(quote: str, text: str) -> bool:
    """True if ``quote`` appears verbatim (modulo case/whitespace/edge punctuation) in ``text``."""
    q = _norm_for_match(quote or "")
    return bool(q) and q in _norm_for_match(text or "")


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add clean/redacted text, counts, hashes, length flags and an ``exclude_reason`` column.

    Rows are never dropped here; callers filter on ``exclude_reason.isna()`` so the quality report can
    describe everything that was excluded and why.
    """
    out = df.copy()
    raw = out["transcript_text"].fillna("").astype(str)
    out["clean_text"] = raw.map(normalize_whitespace)
    out["redacted_text"] = out["clean_text"].map(redact_pii)
    out["n_words"] = out["clean_text"].str.split().str.len().astype(int)
    out["n_turns"] = out["clean_text"].map(lambda t: len(parse_turns(t)) if t else 0)
    out["text_hash"] = out["clean_text"].map(_hash)

    empty = out["clean_text"].eq("")
    dup_id = out.duplicated("transcript_id", keep="first")
    dup_text = (~empty) & out.duplicated("clean_text", keep="first")
    reason = pd.Series([None] * len(out), index=out.index, dtype="object")
    reason[dup_text] = "duplicate_text"
    reason[dup_id] = "duplicate_id"
    reason[empty] = "empty_text"
    out["exclude_reason"] = reason

    nonempty = out.loc[~empty, "n_words"]
    long_cut = max(float(nonempty.median()) * 4, 50) if len(nonempty) else np.inf
    flag = pd.Series("ok", index=out.index, dtype="object")
    flag[(out["n_words"] < SHORT_WORDS) & ~empty] = "very_short"
    flag[out["n_words"] > long_cut] = "very_long"
    flag[empty] = "empty"
    out["length_flag"] = flag
    return out


def build_quality_report(full: pd.DataFrame, malformed: list[dict]) -> dict:
    """Counts + examples for every data-quality check. ``full`` is the output of :func:`preprocess`."""
    def ids(mask, k=5):
        return full.loc[mask, "transcript_id"].astype(str).head(k).tolist()

    reason = full["exclude_reason"]
    n = len(full)
    return {
        "total_records": int(n),
        "malformed_records": {"count": len(malformed), "examples": malformed[:5]},
        "empty_text": {"count": int((reason == "empty_text").sum()), "examples": ids(reason == "empty_text")},
        "duplicate_ids": {"count": int((reason == "duplicate_id").sum()), "examples": ids(reason == "duplicate_id")},
        "duplicate_text": {"count": int((reason == "duplicate_text").sum()), "examples": ids(reason == "duplicate_text")},
        "missing_dates": {"count": int(full["call_date"].isna().sum()), "examples": ids(full["call_date"].isna())},
        "unknown_call_type": {"count": int((full["call_type"] == "unknown").sum()), "examples": ids(full["call_type"] == "unknown")},
        "call_type_inferred": {"count": int(full["call_type_inferred"].sum()), "examples": ids(full["call_type_inferred"])},
        "ids_generated": {"count": int(full["id_generated"].sum()), "examples": ids(full["id_generated"])},
        "very_short": {"count": int((full["length_flag"] == "very_short").sum()), "examples": ids(full["length_flag"] == "very_short")},
        "very_long": {"count": int((full["length_flag"] == "very_long").sum()), "examples": ids(full["length_flag"] == "very_long")},
        "words": {"min": int(full["n_words"].min()) if n else 0, "median": float(full["n_words"].median()) if n else 0,
                  "max": int(full["n_words"].max()) if n else 0},
        "by_call_type": {k: int(v) for k, v in full["call_type"].value_counts().items()},
        "by_origin": {k: int(v) for k, v in full["data_origin"].value_counts().items()},
        "analyzable_records": int(reason.isna().sum()),
    }
