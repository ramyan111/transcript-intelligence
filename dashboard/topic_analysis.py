"""Page 2 - Topic Intelligence."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import analytics


def render(ctx) -> None:
    st.title("Topic Intelligence")
    df = ctx.df
    if df.empty:
        st.info("No transcripts match the current filters.")
        return
    tc = analytics.topic_counts(df)
    st.plotly_chart(px.bar(tc.iloc[::-1], x="count", y="topic", orientation="h", title="Topic frequency (primary topic)"))
    ct = analytics.topic_by_call_type(df)
    st.plotly_chart(px.imshow(ct, text_auto=True, aspect="auto", title="Topics by call type (counts)", color_continuous_scale="Blues"))
    st.caption("What it means: each column shows where that call type's conversations concentrate. Compare columns using the % table below, since call types differ in size.")
    st.dataframe(analytics.topic_by_call_type_pct(df).rename_axis("% of calls within each call type"))

    st.subheader("Secondary topics")
    sec = df.explode("secondary_topics").dropna(subset=["secondary_topics"])
    if sec.empty:
        st.write("No secondary topics assigned in the filtered data.")
    else:
        st.dataframe(sec["secondary_topics"].value_counts().rename_axis("secondary topic").reset_index(name="count"))

    st.subheader("Taxonomy rationale")
    st.markdown("The taxonomy is a **candidate** list (`config/taxonomy.json`), meant to be revised after reading the cluster "
                "representatives below. Rules are transparent but brittle; clusters find unanticipated themes but are unlabeled; "
                "the LLM handles nuance but must be constrained to the allowed list and validated.")
    if ctx.taxonomy:
        st.dataframe(pd.DataFrame([{"topic": t.name, "description": t.description} for t in ctx.taxonomy]))

    st.subheader("Representative excerpts")
    topic = st.selectbox("Topic", tc["topic"].tolist())
    for _, r in df[df["primary_topic"] == topic].head(3).iterrows():
        ev = r["supporting_evidence"][0]["quote"] if r["supporting_evidence"] else r["summary"]
        st.markdown(f"**{r['transcript_id']}** ({r['call_type']}, {r['sentiment_label']}): “{ev}”")

    st.subheader("Exploratory clusters")
    cl = ctx.clusters
    if not cl or "clusters" not in cl:
        st.write("No cluster summary available for this run.")
    else:
        st.write(f"Embedding method: `{cl['method']}`; k={cl['k']} "
                 f"({'best silhouette' if cl['k_was_chosen_by_silhouette'] else 'user-set'}). {cl['note']}")
        st.dataframe(pd.DataFrame([{"k": k, "silhouette": v} for k, v in cl["silhouette_by_k"].items()]))
        for c in cl["clusters"]:
            with st.expander(f"Cluster {c['cluster']} — {c['size']} calls — {', '.join(c['top_terms'][:5])}"):
                st.write(f"Call-type mix: {c['call_type_mix']}")
                for rep in c["representatives"]:
                    st.markdown(f"**{rep['transcript_id']}**: {rep['excerpt']}…")
