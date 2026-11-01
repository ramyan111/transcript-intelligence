"""Taxonomy loading, topic-name validation, and a transparent keyword baseline.

Trade-offs: keyword rules are transparent and free but brittle (synonyms, negation, overlapping topics);
clustering finds themes you did not anticipate but gives unlabeled groups; the LLM handles nuance and
overlap but costs money, can drift, and must be constrained to the taxonomy (see :func:`validate_topics`).
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

OTHER = "Other"


@dataclass(frozen=True)
class Topic:
    name: str
    description: str
    keywords: tuple[str, ...]


def load_taxonomy(path) -> list[Topic]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Topic(t["name"], t.get("description", ""), tuple(t.get("keywords", []))) for t in data["topics"]]


def topic_names(taxonomy: list[Topic]) -> list[str]:
    return [t.name for t in taxonomy]


def validate_topic(name: str | None, allowed: list[str]) -> tuple[str, bool]:
    """Return (valid_topic, was_remapped). Unknown / empty names become 'Other'."""
    lookup = {a.lower(): a for a in allowed}
    if name and name.strip().lower() in lookup:
        return lookup[name.strip().lower()], False
    return (OTHER if OTHER in allowed else allowed[-1]), True


def validate_topics(primary: str | None, secondary: list[str] | None, allowed: list[str]) -> tuple[str, list[str], bool]:
    """Validate primary (remap to Other if invalid) and secondaries (drop invalid / duplicate / equal-to-primary)."""
    valid_primary, remapped = validate_topic(primary, allowed)
    lookup = {a.lower(): a for a in allowed}
    seen, out = {valid_primary}, []
    for s in secondary or []:
        canon = lookup.get((s or "").strip().lower())
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return valid_primary, out, remapped


def _compile(taxonomy: list[Topic]) -> dict[str, list[tuple[str, re.Pattern]]]:
    return {t.name: [(k, re.compile(r"\b" + re.escape(k) + r"\b", re.I)) for k in t.keywords] for t in taxonomy}


def rule_scores(text: str, taxonomy: list[Topic]) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Keyword hit counts per topic and the matched terms."""
    scores, matched = {}, {}
    for name, pats in _compile(taxonomy).items():
        hits = [(k, len(p.findall(text))) for k, p in pats]
        hits = [(k, c) for k, c in hits if c]
        scores[name] = sum(c for _, c in hits)
        matched[name] = [k for k, _ in hits]
    return scores, matched


def rule_classify(text: str, taxonomy: list[Topic], max_secondary: int = 2) -> tuple[str, list[str], dict[str, list[str]]]:
    """Primary topic = highest keyword score (ties -> taxonomy order). Secondary = other topics scoring >= half of top."""
    scores, matched = rule_scores(text, taxonomy)
    order = [t.name for t in taxonomy]
    ranked = sorted((n for n in order if scores[n] > 0), key=lambda n: (-scores[n], order.index(n)))
    if not ranked:
        return OTHER, [], {}
    top = scores[ranked[0]]
    secondary = [n for n in ranked[1:] if scores[n] >= max(2, top / 2)][:max_secondary]
    return ranked[0], secondary, {n: matched[n] for n in [ranked[0], *secondary]}
