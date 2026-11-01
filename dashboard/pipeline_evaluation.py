"""Page 6 - Pipeline & Evaluation."""
import pandas as pd
import streamlit as st

from config.settings import get_settings
from src.evaluation import (compute_ground_truth_metrics, compute_metrics, load_ground_truth,
                            load_review)


def render(ctx) -> None:
    st.title("Pipeline & Evaluation")
    meta, q = ctx.meta, ctx.quality
    st.warning(meta.get("disclaimer", "Run metadata not found."))
    c = st.columns(5)
    c[0].metric("Records loaded", meta.get("records_loaded", "?"))
    c[1].metric("Analyzable", meta.get("records_analyzable", "?"))
    c[2].metric("Analyses stored", meta.get("analyses_total", "?"))
    c[3].metric("Failed (last run)", meta.get("failed", "?"))
    c[4].metric("Runtime (s)", meta.get("runtime_seconds", "?"))

    st.subheader("Data quality")
    if q:
        rows = [{"check": k, "count": v["count"], "examples": ", ".join(map(str, v.get("examples", [])))}
                for k, v in q.items() if isinstance(v, dict) and "count" in v]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.write({"call types": q.get("by_call_type"), "data origin": q.get("by_origin"), "words": q.get("words")})
    st.subheader("Failures / error analysis")
    st.dataframe(pd.DataFrame(ctx.failures)) if ctx.failures else st.write("No failed records in the last run.")

    synthetic = ctx.full_df[ctx.full_df["data_origin"] == "synthetic"]
    if not synthetic.empty:
        st.subheader("Synthetic ground-truth evaluation")
        labels = load_ground_truth(get_settings().synthetic_dir / "ground_truth.csv")
        if labels is None:
            st.info("No separate synthetic ground-truth labels were found.")
        else:
            truth_metrics = compute_ground_truth_metrics(synthetic, labels)
            if truth_metrics is None:
                st.info("No synthetic ground-truth labels matched the processed transcript IDs.")
            else:
                st.caption("Scenario-designed labels for this synthetic dataset; these are not human-review or real-world accuracy.")
                st.write({k: v for k, v in truth_metrics.items() if k not in ("topic", "sentiment")})
                for name in ("topic", "sentiment"):
                    result = truth_metrics[name]
                    st.markdown(f"**{name.title()}**")
                    st.json({k: v for k, v in result.items() if k not in ("confusion_matrix", "labels")})
                    st.dataframe(pd.DataFrame(result["confusion_matrix"], index=[f"true: {x}" for x in result["labels"]],
                                              columns=[f"pred: {x}" for x in result["labels"]]))

    st.subheader("Evaluation against human review")
    review = load_review(f"{ctx.run_dir}/review_sheet.csv")
    metrics = compute_metrics(review) if review is not None else None
    if metrics is None:
        st.info("No reviewed labels yet, so **no accuracy numbers are shown**. Run `python main.py review-sample`, fill "
                "reviewer_topic / reviewer_sentiment / reviewer_renewal_risk / unsupported_claims in review_sheet.csv, then reload.")
    else:
        for name, m in metrics.items():
            st.markdown(f"**{name}**")
            st.json({k: v for k, v in m.items() if k not in ("confusion_matrix", "labels")})
            if "confusion_matrix" in m:
                st.dataframe(pd.DataFrame(m["confusion_matrix"], index=[f"true: {x}" for x in m["labels"]], columns=[f"pred: {x}" for x in m["labels"]]))
    st.subheader("Limitations")
    st.markdown("- Small sample and uneven call types: percentages move a lot with a few calls.\n"
                "- Sentiment scores are uncalibrated; sarcasm and mixed tone are error-prone.\n"
                "- Renewal items are signals, not validated churn predictions; cross-team links are candidates.\n"
                "- Redaction masks emails, phones, card/SSN-like numbers but not names.\n"
                "- Heuristic demo mode is rules + lexicon, not an LLM.")
