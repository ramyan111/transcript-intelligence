"""Page 3 - Sentiment Intelligence."""
import plotly.express as px
import streamlit as st

from src import analytics


def render(ctx) -> None:
    st.title("Sentiment Intelligence")
    df = ctx.df
    if df.empty:
        st.info("No transcripts match the current filters.")
        return
    basis = st.radio("Sentiment basis", ["Overall call tone", "Customer sentiment (customer-facing calls)"], horizontal=True)
    if basis.startswith("Overall"):
        lab, sc = "sentiment_label", "sentiment_score"
    else:
        lab, sc = "customer_sentiment", "customer_sentiment_score"
        if df[lab].notna().sum() == 0:
            st.warning("No customer sentiment available in the filtered data.")
            return
    d = df[df[lab].notna()]
    dist = d[lab].value_counts().rename_axis("sentiment").reset_index(name="count")
    st.plotly_chart(px.bar(dist, x="sentiment", y="count", title=f"Distribution ({len(d)} calls)"))
    with st.expander("How scores work"):
        st.write("Labels: Positive / Neutral / Negative / Mixed. Scores are in [-1, 1] and are **not calibrated probabilities**. "
                 "In demo mode they come from a weighted lexicon; in LLM mode from the model's rating. "
                 "Customer sentiment is separate from overall call tone and is empty for internal calls.")

    sct = analytics.sentiment_by_group(d, "call_type", lab, sc)
    pct_cols = ["positive_pct", "neutral_pct", "mixed_pct", "negative_pct"]
    st.plotly_chart(px.bar(sct.melt(id_vars=["call_type", "n"], value_vars=pct_cols, var_name="label", value_name="% of call type"),
                           x="call_type", y="% of call type", color="label", title="Sentiment by call type (% of that call type's calls)"))
    st.dataframe(sct)
    st.info("What it means: " + analytics.describe_group_sentiment(sct, "call_type"))

    stp = analytics.sentiment_by_group(d, "primary_topic", lab, sc)
    st.plotly_chart(px.bar(stp.sort_values("avg_score"), x="avg_score", y="primary_topic", orientation="h", title="Average score by topic"))
    st.dataframe(stp)
    st.info("What it means: " + analytics.describe_group_sentiment(stp, "primary_topic"))

    st.subheader("Where the negatives are concentrated")
    st.dataframe(analytics.negative_concentration(d, "primary_topic", lab))
    st.caption("negative_rate_pct = group negatives / group size; share_of_all_negative_pct = group negatives / all negatives.")

    trend = analytics.sentiment_trend(d.assign(sentiment_label=d[lab], sentiment_score=d[sc]))
    st.subheader("Trend over time")
    if trend is None:
        st.write("Not shown: call dates are missing for too many calls or span fewer than 3 months.")
    else:
        st.plotly_chart(px.line(trend, x="month", y="negative_pct", markers=True, title="Negative share by month (%)"))
        st.info("What it means: month-level percentages rest on the per-month counts in `n`; small months move a lot.")
        st.dataframe(trend)
