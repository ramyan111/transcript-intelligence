"""TranscriptIQ — Conversation Intelligence (Streamlit). Run: streamlit run app.py"""
import streamlit as st

from config.settings import get_settings
from dashboard import business_insights, overview, pipeline_evaluation, sentiment_analysis, topic_analysis, transcript_explorer
from dashboard.common import Context
from src import storage
from src.topic_classifier import load_taxonomy

st.set_page_config(page_title="TranscriptIQ — Conversation Intelligence", layout="wide")
settings = get_settings()

PAGES = {"Executive Overview": overview, "Topic Intelligence": topic_analysis, "Sentiment Intelligence": sentiment_analysis,
         "Business Insights": business_insights, "Transcript Explorer": transcript_explorer, "Pipeline & Evaluation": pipeline_evaluation}


@st.cache_data(show_spinner=False)
def load(run_dir: str, mtime: float):
    return storage.load_processed(run_dir)


runs = [d for d in sorted(settings.processed_dir.glob("*")) if (d / "analyses.jsonl").exists()]
st.sidebar.title("TranscriptIQ")
if not runs:
    st.title("TranscriptIQ — Conversation Intelligence")
    st.info("No processed data yet.\n\n1. Put transcripts (CSV/JSON/TXT) in `data/raw/`, or use the labeled synthetic demo set.\n"
            "2. Run `python main.py run --mode demo` (no API key) or `python main.py run --mode llm` (needs OPENAI_API_KEY).\n"
            "3. Reload this page.")
    st.stop()

run_dir = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
data = load(str(run_dir), (run_dir / "analyses.jsonl").stat().st_mtime)
if data is None:
    st.warning("This run has no analyses yet.")
    st.stop()
full = data["df"]

origins = sorted(full["data_origin"].unique())
origin = st.sidebar.selectbox("Data origin (never mixed)", origins)
full = full[full["data_origin"] == origin]
if origin == "synthetic":
    st.warning("SYNTHETIC demo data: invented transcripts. Nothing here is a company finding.")
if data["meta"].get("mode") == "demo":
    st.caption("Analysis mode: heuristic demo (rules + lexicon), not an LLM.")

page = st.sidebar.radio("Page", list(PAGES))
st.sidebar.markdown("**Filters**")
f_ct = st.sidebar.multiselect("Call type", sorted(full["call_type"].unique()), default=sorted(full["call_type"].unique()))
f_topic = st.sidebar.multiselect("Topic", sorted(full["primary_topic"].unique()), default=sorted(full["primary_topic"].unique()))
f_sent = st.sidebar.multiselect("Sentiment", sorted(full["sentiment_label"].unique()), default=sorted(full["sentiment_label"].unique()))
view = full[full["call_type"].isin(f_ct) & full["primary_topic"].isin(f_topic) & full["sentiment_label"].isin(f_sent)]

ctx = Context(df=view, full_df=full, meta=data["meta"], quality=data["quality"], clusters=data["clusters"],
              failures=data["failures"], taxonomy=load_taxonomy(settings.taxonomy_path), run_dir=str(run_dir))
PAGES[page].render(ctx)
