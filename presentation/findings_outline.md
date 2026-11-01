# TranscriptIQ — Leadership Findings (outline)

> Fill `[ ]` placeholders from a run on **real** transcripts (`python main.py run --mode llm`). Values from the synthetic demo set must not be presented as company findings.
> Lead with business findings; keep architecture to one slide.

1. **Business problem** — Calls hold unstructured signals about churn, product gaps and delivery risk. Today they are read one at a time. Question: what recurs, how does sentiment differ by call type, and what should each team do?
2. **Dataset** — [N] transcripts: support [ ], external [ ], internal [ ]; date range [ ] (or "no reliable dates"); excluded [ ] (empty/duplicate); data origin [original].
3. **Executive findings (3–5 bullets)** — each: finding · number · transcripts that prove it.
   - [ ] Top topic and share · [ ] highest-negativity call type/topic · [ ] renewal-risk signal rate · [ ] top feature request · [ ] top engineering blocker
4. **Topic results** — frequency chart; topics by call type; 2 example transcripts per top topic; how the taxonomy was derived (clusters → taxonomy → LLM assignment).
5. **Sentiment results** — distribution; by call type; by topic; where negatives concentrate. Always show n and the caveats (small groups, uncalibrated scores).
6. **Extra insight A — Renewal-risk signals** — [k of n] customer-facing calls; evidence quotes; recommended action for Customer Success; "signals, not predictions".
7. **Extra insight B — Feature demand** — ranked request groups with transcript links; what Product should triage.
8. **Extra insight C — Engineering blockers and cross-team candidates** — blockers by theme; candidate customer↔engineering links to verify; recommended joint action.
9. **Recommendations by team** — Support / Customer Success / Product / Engineering: 1 action each with owner and how success is measured.
10. **Architecture (one slide)** — ingestion → redaction → hybrid topics → structured LLM → deterministic analytics → dashboard; where humans review.
11. **Evaluation** — human-reviewed sample [20–30]: topic accuracy [ ], macro-F1 [ ], sentiment accuracy [ ], renewal-flag precision [ ], unsupported-claim rate [ ]. Failure modes found: [ ]. If not yet reviewed, say so.
12. **Limitations** — sample size, imbalance, uncalibrated sentiment, sarcasm, signals vs predictions, candidate links, PII (names not redacted).
13. **Roadmap** — verified ticket/incident links, semantic grouping, larger sample, scheduled runs, per-account views, review loop to improve prompts.

Q&A prep: why hybrid topics? how do you prevent hallucination? how are denominators defined? what would change your mind about a finding? what does it cost per 100 transcripts? (measure from `run_meta.json` + API usage.)
