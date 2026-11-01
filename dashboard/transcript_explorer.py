"""Page 5 - Transcript Explorer."""
import streamlit as st


def render(ctx) -> None:
    st.title("Transcript Explorer")
    df = ctx.df
    if df.empty:
        st.info("No transcripts match the current filters.")
        return
    q = st.text_input("Search transcript text, summary or id")
    view = df
    if q:
        mask = (df["clean_text"].str.contains(q, case=False, regex=False) | df["summary"].str.contains(q, case=False, regex=False)
                | df["transcript_id"].str.contains(q, case=False, regex=False))
        view = df[mask]
    cols = ["transcript_id", "call_type", "call_date", "primary_topic", "sentiment_label", "sentiment_score", "escalation_required", "data_origin"]
    st.dataframe(view[cols], hide_index=True)
    if view.empty:
        st.write("No matches.")
        return
    tid = st.selectbox("Open transcript", view["transcript_id"].tolist())
    r = view[view["transcript_id"] == tid].iloc[0]
    left, right = st.columns(2)
    with left:
        st.subheader("Original transcript")
        st.text(r["transcript_text"])
    with right:
        st.subheader("Analysis")
        st.markdown(f"**Mode:** `{r['analysis_mode']}`" + (f" (`{r['llm_model']}`)" if r.get("llm_model") else ""))
        st.markdown(f"**Topic:** {r['primary_topic']}" + (f" (+ {', '.join(r['secondary_topics'])})" if r["secondary_topics"] else ""))
        st.markdown(f"**Sentiment:** {r['sentiment_label']} ({r['sentiment_score']}); customer: {r['customer_sentiment'] or 'n/a'}")
        st.markdown(f"**Summary:** {r['summary']}")
        for title, key in [("Pain points", "pain_points"), ("Feature requests", "feature_requests"),
                           ("Renewal-risk signals", "renewal_risk_signals"), ("Engineering blockers", "engineering_blockers"),
                           ("Action items", "action_items")]:
            if r[key]:
                st.markdown(f"**{title}**")
                for x in r[key]:
                    st.markdown(f"- {x}")
        st.markdown(f"**Escalation signal:** {'yes' if r['escalation_required'] else 'no'}")
        st.markdown("**Evidence**")
        for e in r["supporting_evidence"]:
            badge = "✅ found in transcript" if e.get("verified") else "⚠️ quote NOT found verbatim"
            st.markdown(f"- {e['claim']}: “{e['quote']}” — {badge}")
        st.caption(r["explanation"])
