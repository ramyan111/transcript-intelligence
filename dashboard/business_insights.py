"""Page 4 - Business Insights (customer success, product, engineering)."""
import streamlit as st

from src import analytics
from src.insight_extractor import cross_team_candidates

SECTIONS = [("Customer Success — renewal risk", ["Customer Success"]),
            ("Product — feature demand", ["Product"]),
            ("Engineering & Support — blockers and escalations", ["Engineering", "Support + Engineering", "Support"])]


def _card(ins: dict) -> None:
    with st.expander(ins["title"], expanded=True):
        st.markdown(f"**Metric:** {ins['metric']}")
        st.markdown(f"**Interpretation:** {ins['interpretation']}")
        st.markdown(f"**Recommended action:** {ins['recommended_action']}")
        st.caption(f"Caveat: {ins['caveat']}")
        st.markdown("**Evidence**")
        for e in ins["evidence"]:
            st.markdown(f"- `{e['transcript_id']}` — {e['quote']}")
        if not ins["evidence"]:
            st.write("No evidence excerpts available.")
        st.caption(f"Transcript IDs ({len(ins['transcript_ids'])}): " + ", ".join(ins["transcript_ids"][:30])
                   + (" …" if len(ins["transcript_ids"]) > 30 else ""))


def render(ctx) -> None:
    st.title("Business Insights")
    st.caption("Numbers are computed deterministically from the filtered data. Renewal items are signals, not predictions; cross-team links are candidates.")
    df = ctx.df
    if df.empty:
        st.info("No transcripts match the current filters.")
        return
    insights = analytics.build_insights(df)
    tabs = st.tabs([s[0] for s in SECTIONS])
    for tab, (_, stakeholders) in zip(tabs, SECTIONS):
        with tab:
            items = [i for i in insights if i["stakeholder"] in stakeholders]
            if not items:
                st.write("No insight available for the filtered data.")
            for ins in items:
                _card(ins)
    st.subheader("Feature request groups")
    st.dataframe(analytics.feature_group_table(df)[["representative", "n_transcripts", "transcript_ids"]])
    st.subheader("Candidate cross-team matches")
    cand = cross_team_candidates(df)
    st.dataframe(cand)
    st.caption("Text-similarity candidates only; verify with ticket or incident ids before acting.")
