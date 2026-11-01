"""Deterministic aggregates and stakeholder insights. Every number is computed here in pandas, never by an LLM.

Denominators: rates for a group (call type or topic) always divide by the number of analyzed transcripts IN that group.
Concentration shares divide by the total across all groups. Both are labeled in the column names.
"""
import pandas as pd

from src.insight_extractor import cross_team_candidates, group_feature_requests

LABELS = ["Positive", "Neutral", "Mixed", "Negative"]
SMALL_N = 10


def _pct(a, b) -> float:
    return round(100 * float(a) / float(b), 1) if b else 0.0


def overview_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "total": n,
        "by_call_type": {k: int(v) for k, v in df["call_type"].value_counts().items()},
        "n_topics": int(df["primary_topic"].nunique()),
        "negative_n": int((df["sentiment_label"] == "Negative").sum()),
        "negative_pct": _pct((df["sentiment_label"] == "Negative").sum(), n),
        "escalations": int(df["escalation_required"].sum()),
        "feature_requests": int(df["feature_requests"].map(len).sum()),
        "renewal_flagged": int((df["renewal_risk_signals"].map(len) > 0).sum()),
    }


def topic_counts(df: pd.DataFrame) -> pd.DataFrame:
    t = df["primary_topic"].value_counts().rename_axis("topic").reset_index(name="count")
    t["pct_of_all"] = (100 * t["count"] / max(1, len(df))).round(1)
    return t


def topic_by_call_type(df: pd.DataFrame) -> pd.DataFrame:
    """Counts; each call type's share of its OWN calls is available via :func:`topic_by_call_type_pct`."""
    return pd.crosstab(df["primary_topic"], df["call_type"])


def topic_by_call_type_pct(df: pd.DataFrame) -> pd.DataFrame:
    return (pd.crosstab(df["primary_topic"], df["call_type"], normalize="columns") * 100).round(1)


def sentiment_by_group(df: pd.DataFrame, group_col: str, label_col: str = "sentiment_label",
                       score_col: str = "sentiment_score") -> pd.DataFrame:
    """Per group: n, % of the GROUP in each label, mean score. Rows with a null label (e.g. no customer) are excluded."""
    d = df[df[label_col].notna()]
    rows = []
    for g, sub in d.groupby(group_col):
        n = len(sub)
        row = {group_col: g, "n": n}
        for lab in LABELS:
            row[f"{lab.lower()}_pct"] = _pct((sub[label_col] == lab).sum(), n)
        row["avg_score"] = round(float(sub[score_col].mean()), 3) if sub[score_col].notna().any() else None
        row["small_sample"] = n < SMALL_N
        rows.append(row)
    return pd.DataFrame(rows).sort_values("negative_pct", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def negative_concentration(df: pd.DataFrame, group_col: str, label_col: str = "sentiment_label") -> pd.DataFrame:
    """Where do the negatives sit? share_of_all_negative = group negatives / ALL negatives; negative_rate = group negatives / group n."""
    d = df[df[label_col].notna()]
    total_neg = int((d[label_col] == "Negative").sum())
    g = d.groupby(group_col).agg(n=(label_col, "size"), negative_n=(label_col, lambda s: int((s == "Negative").sum()))).reset_index()
    g["negative_rate_pct"] = [_pct(a, b) for a, b in zip(g["negative_n"], g["n"])]
    g["share_of_all_negative_pct"] = [_pct(a, total_neg) for a in g["negative_n"]]
    return g.sort_values("negative_n", ascending=False).reset_index(drop=True)


def reliable_dates(df: pd.DataFrame, min_coverage: float = 0.8, min_months: int = 3) -> bool:
    if df.empty or "call_date" not in df:
        return False
    d = df["call_date"].dropna()
    return len(d) / len(df) >= min_coverage and d.dt.to_period("M").nunique() >= min_months


def sentiment_trend(df: pd.DataFrame) -> pd.DataFrame | None:
    """Monthly n / negative % / mean score, only when dates are reliable; otherwise None."""
    if not reliable_dates(df):
        return None
    d = df.dropna(subset=["call_date"]).copy()
    d["month"] = d["call_date"].dt.to_period("M").dt.to_timestamp()
    return (d.groupby("month").agg(n=("sentiment_label", "size"),
                                   negative_n=("sentiment_label", lambda s: int((s == "Negative").sum())),
                                   avg_score=("sentiment_score", "mean")).reset_index()
            .assign(negative_pct=lambda t: [_pct(a, b) for a, b in zip(t["negative_n"], t["n"])]))


def feature_group_table(df: pd.DataFrame) -> pd.DataFrame:
    reqs = [(r["transcript_id"], q) for _, r in df.iterrows() for q in r["feature_requests"]]
    return group_feature_requests(reqs)


def _ev(df: pd.DataFrame, ids: list[str], col: str, per: int = 1, limit: int = 6) -> list[dict]:
    out = []
    for tid in ids[:limit]:
        row = df[df["transcript_id"] == tid].iloc[0]
        for q in row[col][:per]:
            out.append({"transcript_id": tid, "quote": q})
    return out


def build_insights(df: pd.DataFrame) -> list[dict]:
    """Stakeholder insights with metric, transcript ids, evidence, interpretation, action and caveat."""
    insights: list[dict] = []
    if df.empty:
        return insights
    cf = df[df["call_type"].isin(["support", "external"])]

    # A. Renewal / churn-risk SIGNALS (not a validated churn prediction)
    flagged = cf[cf["renewal_risk_signals"].map(len) > 0]
    ids = flagged["transcript_id"].tolist()
    by_ct = {ct: f"{int((flagged['call_type'] == ct).sum())}/{int((cf['call_type'] == ct).sum())}" for ct in sorted(cf["call_type"].unique())}
    insights.append({
        "id": "renewal_risk", "title": "Renewal-risk signals in customer-facing calls", "stakeholder": "Customer Success",
        "metric": f"{len(ids)} of {len(cf)} customer-facing calls ({_pct(len(ids), len(cf))}%) contain at least one signal; by type (flagged/total): {by_ct}",
        "transcript_ids": ids, "evidence": _ev(df, ids, "renewal_risk_signals"),
        "interpretation": "Calls where the customer voiced cancellation, competitor evaluation, renewal hesitation or pricing/budget pressure. Review flagged accounts first, starting with those that also needed escalation.",
        "recommended_action": "Customer Success reviews flagged calls, confirms account status, and prepares a save plan for confirmed at-risk accounts.",
        "caveat": "Signals, not a validated churn model. Rules/LLM can false-flag (e.g. sarcasm, hypothetical price talk). Check precision on a human-reviewed sample before acting at scale."})

    # B. Feature demand
    fg = feature_group_table(df)
    if not fg.empty:
        top = fg.head(3)
        for i, r in top.iterrows():
            tids = list(r["transcript_ids"])
            insights.append({
                "id": f"feature_{i + 1}", "title": f"Feature demand: {r['representative'][:90]}", "stakeholder": "Product",
                "metric": f"Requested in {r['n_transcripts']} transcript(s) ({_pct(r['n_transcripts'], len(df))}% of analyzed calls)",
                "transcript_ids": tids, "evidence": [{"transcript_id": t, "quote": q} for t, q in zip(tids, r["examples"])][:3],
                "interpretation": "Group of near-duplicate requests (lexical similarity). Frequency is a demand signal, not revenue impact.",
                "recommended_action": "Product triages this request group against roadmap, weighting by affected account value.",
                "caveat": "Lexical grouping can split paraphrases or merge different asks; counts are per transcript, not per customer."})

    # C. Engineering blockers + candidate cross-team links
    eng = df[(df["call_type"] == "internal") & (df["engineering_blockers"].map(len) > 0)]
    internal_n = int((df["call_type"] == "internal").sum())
    if internal_n:
        ids = eng["transcript_id"].tolist()
        topics = eng["primary_topic"].value_counts().head(3).to_dict()
        insights.append({
            "id": "eng_blockers", "title": "Engineering blockers and unresolved problems", "stakeholder": "Engineering",
            "metric": f"{len(ids)} of {internal_n} internal calls ({_pct(len(ids), internal_n)}%) mention a blocker/delay; top topics: {topics}",
            "transcript_ids": ids, "evidence": _ev(df, ids, "engineering_blockers"),
            "interpretation": "Recurring blockers (dependencies, flaky tests, capacity) that delay releases or fixes.",
            "recommended_action": "Engineering leads assign owners to repeated blockers and report unresolved ones at the next planning review.",
            "caveat": "Phrase-based extraction: 'delayed' may describe a past, already-resolved delay."})
        cand = cross_team_candidates(df)
        if not cand.empty:
            c_ids = sorted(set(cand["internal_id"]) | set(cand["customer_facing_id"]))
            insights.append({
                "id": "cross_team", "title": "Candidate links between customer reports and engineering discussions",
                "stakeholder": "Support + Engineering",
                "metric": f"{len(cand)} candidate pair(s) above similarity threshold; top similarity {cand['similarity'].max()}",
                "transcript_ids": c_ids,
                "evidence": [{"transcript_id": r["customer_facing_id"], "quote": f"~ internal {r['internal_id']} (shared terms: {r['shared_terms']})"} for _, r in cand.head(4).iterrows()],
                "interpretation": "Customer-facing and internal calls that discuss similar problems, suggesting customers may be feeling an issue engineering already knows about.",
                "recommended_action": "Support and Engineering jointly verify the top pairs and, where confirmed, share ETAs with affected customers.",
                "caveat": "CANDIDATE matches by text similarity only. No ticket or incident id verifies any link."})

    # Escalations (support)
    sup = df[df["call_type"] == "support"]
    if len(sup):
        esc = sup[sup["escalation_required"]]
        topics = esc["primary_topic"].value_counts().head(3).to_dict()
        insights.append({
            "id": "support_escalations", "title": "Escalation pressure in support", "stakeholder": "Support",
            "metric": f"{len(esc)} of {len(sup)} support calls ({_pct(len(esc), len(sup))}%) show escalation language; top topics: {topics}",
            "transcript_ids": esc["transcript_id"].tolist(), "evidence": _ev(df, esc["transcript_id"].tolist(), "pain_points"),
            "interpretation": "Topics concentrating escalations indicate where tier-1 cannot resolve issues.",
            "recommended_action": "Support leadership reviews the top escalating topics for playbook or tooling gaps.",
            "caveat": "Escalation language includes the agent saying they escalated; it does not prove the escalation was required."})
    return insights


def executive_summary(df: pd.DataFrame) -> list[str]:
    """Plain-language bullets assembled from computed numbers (no hardcoded conclusions)."""
    if df.empty:
        return []
    m = overview_metrics(df)
    tc = topic_counts(df).iloc[0]
    sct = sentiment_by_group(df, "call_type")
    worst = sct.iloc[0]
    out = [f"{m['total']} transcripts analyzed across {len(m['by_call_type'])} call types; the most common primary topic is "
           f"'{tc['topic']}' ({tc['count']} calls, {tc['pct_of_all']}%).",
           f"Overall {m['negative_pct']}% of calls ({m['negative_n']}/{m['total']}) are labeled Negative. Highest negative share by call type: "
           f"{worst['call_type']} at {worst['negative_pct']}% of its {worst['n']} calls."]
    if m["renewal_flagged"]:
        out.append(f"{m['renewal_flagged']} call(s) contain renewal-risk signals (signals, not predictions).")
    if m["escalations"]:
        out.append(f"{m['escalations']} call(s) show escalation language.")
    if m["feature_requests"]:
        out.append(f"{m['feature_requests']} feature-request statement(s) were extracted for product review.")
    if (sct["n"] < SMALL_N).any():
        out.append("Some call-type groups have fewer than 10 calls; treat their percentages as directional only.")
    return out


def describe_group_sentiment(sdf: pd.DataFrame, group_col: str) -> str:
    """'What it means' text computed from a :func:`sentiment_by_group` table."""
    if sdf.empty:
        return "No data for this view."
    hi, lo = sdf.iloc[0], sdf.sort_values("avg_score").iloc[0]
    txt = (f"{hi[group_col]} has the highest negative share ({hi['negative_pct']}% of {hi['n']} calls). "
           f"The lowest average score is {lo[group_col]} ({lo['avg_score']}). ")
    small = sdf[sdf["small_sample"]][group_col].tolist()
    if small:
        txt += f"Small groups (n<{SMALL_N}): {', '.join(map(str, small))}; interpret cautiously. "
    return txt + "Scores come from an uncalibrated lexicon/LLM rating, and call types are unevenly sized."
