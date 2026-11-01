"""Page 1 - Executive Overview."""
import plotly.express as px
import streamlit as st

from src import analytics


def render(ctx) -> None:
    st.title("TranscriptIQ — Conversation Intelligence")
    st.caption("Executive overview")
    df = ctx.df
    if df.empty:
        st.info("No transcripts match the current filters.")
        return
    m = analytics.overview_metrics(df)
    cols = st.columns(6)
    cols[0].metric("Transcripts", m["total"])
    cols[1].metric("Topics", m["n_topics"])
    cols[2].metric("Negative share", f"{m['negative_pct']}%", help=f"{m['negative_n']} of {m['total']} analyzed calls")
    cols[3].metric("Escalation signals", m["escalations"])
    cols[4].metric("Feature requests", m["feature_requests"])
    cols[5].metric("Renewal-risk signals", m["renewal_flagged"], help="Calls with at least one signal; not a churn prediction")

    left, right = st.columns(2)
    ct = df["call_type"].value_counts().rename_axis("call_type").reset_index(name="count")
    left.plotly_chart(px.bar(ct, x="call_type", y="count", title="Calls by type", text="count"))
    tc = analytics.topic_counts(df).head(10)
    right.plotly_chart(px.bar(tc.iloc[::-1], x="count", y="topic", orientation="h", title="Top topics (primary)"))
    sb = df["sentiment_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
    st.plotly_chart(px.pie(sb, names="sentiment", values="count", hole=0.5, title="Sentiment breakdown (overall call tone)"))

    st.subheader("Executive summary (computed from the filtered data)")
    for bullet in analytics.executive_summary(df):
        st.markdown(f"- {bullet}")
