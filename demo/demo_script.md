# TranscriptIQ — Demo script (target 8 minutes)

Prep: venv active, `.env` set (or use demo mode), terminal and browser side by side. State up front whether the data is real or synthetic.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:45 | README top + `data/raw/` listing | The question: what recurs in calls, how sentiment differs, what each team should do. State the data origin and analysis mode. |
| 0:45–1:45 | Open an input file (CSV/JSON/TXT); then `python main.py run --mode llm --limit 3` | Accepted formats; missing metadata is flagged, never invented; PII is masked before any API call; transcript text is treated as untrusted data. |
| 1:45–2:45 | Terminal output + `data/processed/<mode>/analyses.jsonl` (one record) | Structured, schema-validated result: topic from the allowed list, sentiment + score, evidence quotes checked verbatim. Checkpoints make runs resumable; failures go to `failures.jsonl`. |
| 2:45–3:15 | `python main.py run --mode llm` (or show finished run) then `streamlit run app.py` | Full run, then the dashboard. If it is demo mode, say it is a heuristic baseline. |
| 3:15–4:00 | **Executive Overview** | KPIs are computed from the processed records; the executive summary is generated from computed numbers. |
| 4:00–5:00 | **Topic Intelligence**: topic chart, heatmap, cluster expander; apply the call-type filter | How the taxonomy came from clusters + review; show representative excerpts. |
| 5:00–6:00 | **Sentiment Intelligence**: switch basis to customer sentiment | Denominators are group sizes; customer sentiment ≠ call tone; read the "what it means" text and the small-n warnings. |
| 6:00–7:15 | **Business Insights** tabs, then **Transcript Explorer** for one insight's transcript | Each insight: metric, evidence, action, caveat. Open the transcript and show the quote highlighted as verified. Signals ≠ predictions; cross-team matches are candidates. |
| 7:15–8:00 | **Pipeline & Evaluation** | Data quality, failures, and evaluation. Distinguish scenario-designed synthetic-label agreement from human-review metrics; if no human review has been done, real-world accuracy is unknown. Close with limitations and the roadmap. |

Backup if the API fails live: switch the dashboard "Run" selector to the saved run, or run `--mode demo` and say so.
