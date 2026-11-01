"""End-to-end pipeline: ingest -> preprocess -> quality report -> analyze (checkpointed) -> clusters -> insights.

Resumable: every analysis is appended to ``analyses.jsonl`` as soon as it succeeds. A re-run skips ids whose
``text_hash`` is unchanged and retries failures. One bad record never stops the run; failures go to ``failures.jsonl``.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from config.settings import Settings, get_settings
from src import analytics, storage
from src.demo_analyzer import analyze_record
from src.ingestion import SUPPORTED_EXTENSIONS, load_transcripts
from src.llm_client import build_llm_analyzer
from src.preprocessing import build_quality_report, preprocess
from src.schemas import TranscriptAnalysis
from src.synthetic import write_synthetic
from src.topic_classifier import load_taxonomy
from src.topic_discovery import discover_clusters

logger = logging.getLogger(__name__)
Analyzer = Callable[[dict], TranscriptAnalysis]


def resolve_default_input(settings: Settings) -> Path:
    """data/raw if it holds files; otherwise the labeled synthetic demo set (generated if missing)."""
    if any(p.suffix.lower() in SUPPORTED_EXTENSIONS for p in settings.raw_dir.rglob("*") if p.is_file()):
        return settings.raw_dir
    path = settings.synthetic_dir / "transcripts.json"
    if not path.exists():
        write_synthetic(settings.synthetic_dir)
    logger.warning("No transcripts in %s - using the SYNTHETIC demo dataset. Results are not company findings.", settings.raw_dir)
    return path


def build_analyzer(mode: str, taxonomy, settings: Settings) -> Analyzer:
    if mode == "demo":
        return lambda rec: analyze_record(rec, taxonomy)
    if mode == "llm":
        return build_llm_analyzer(taxonomy, settings)  # raises SetupError without an API key
    raise ValueError(f"Unknown mode: {mode!r} (use 'demo' or 'llm')")


def run_pipeline(input_path=None, mode: str = "demo", out_dir=None, limit: int | None = None, resume: bool = True,
                 analyzer: Analyzer | None = None, settings: Settings | None = None, n_clusters: int | None = None,
                 skip_clusters: bool = False, origin: str | None = None) -> dict:
    settings = settings or get_settings()
    t0, started = time.time(), datetime.now(timezone.utc).isoformat()
    taxonomy = load_taxonomy(settings.taxonomy_path)
    analyzer = analyzer or build_analyzer(mode, taxonomy, settings)

    input_path = Path(input_path) if input_path else resolve_default_input(settings)
    out = Path(out_dir) if out_dir else settings.processed_dir / mode
    out.mkdir(parents=True, exist_ok=True)

    raw, malformed = load_transcripts(input_path, default_origin=origin)
    full = preprocess(raw)
    quality = build_quality_report(full, malformed)
    storage.write_json(out / "quality_report.json", quality)
    todo = full[full["exclude_reason"].isna()]
    if limit:
        todo = todo.head(limit)
    todo.drop(columns=["exclude_reason"]).to_csv(out / "transcripts.csv", index=False)

    ckpt = out / "analyses.jsonl"
    done = {tid: r.get("text_hash") for tid, r in storage.latest_by_id(storage.read_jsonl(ckpt)).items()} if resume else {}
    if not resume and ckpt.exists():
        ckpt.rename(ckpt.with_suffix(".jsonl.bak"))
    processed = skipped = 0
    failures: list[dict] = []
    for rec in todo.to_dict("records"):
        if done.get(rec["transcript_id"]) == rec["text_hash"]:
            skipped += 1
            continue
        try:
            analysis = analyzer(rec)
            storage.append_jsonl(ckpt, {**analysis.model_dump(), "text_hash": rec["text_hash"]})
            processed += 1
        except Exception as exc:  # per-record isolation
            logger.error("Failed %s: %s: %s", rec["transcript_id"], type(exc).__name__, exc)
            failures.append({"transcript_id": rec["transcript_id"], "error": f"{type(exc).__name__}: {exc}",
                             "at": datetime.now(timezone.utc).isoformat()})
    fail_path = out / "failures.jsonl"
    fail_path.unlink(missing_ok=True)
    for f in failures:
        storage.append_jsonl(fail_path, f)

    merged = storage.merge_frame(todo, storage.read_jsonl(ckpt))
    cluster_info = None
    if not merged.empty:
        if not skip_clusters:
            try:
                cluster_info = discover_clusters(merged, k=n_clusters)
            except Exception as exc:
                logger.warning("Cluster discovery failed: %s", exc)
                cluster_info = {"error": str(exc)}
            storage.write_json(out / "clusters.json", cluster_info)
        storage.write_json(out / "insights.json", analytics.build_insights(merged))

    models = sorted({str(m) for m in merged["llm_model"].dropna()}) if not merged.empty else []
    meta = {"mode": mode, "llm_models": models, "input_path": str(input_path), "started_utc": started,
            "runtime_seconds": round(time.time() - t0, 2), "records_loaded": quality["total_records"],
            "records_analyzable": int(len(todo)), "processed_this_run": processed, "skipped_resumed": skipped,
            "failed": len(failures), "analyses_total": int(len(merged)), "data_origins": quality["by_origin"],
            "disclaimer": ("HEURISTIC DEMO analysis (rules + lexicon), not an LLM." if mode == "demo" else "LLM analysis.")
            + (" Data origin includes SYNTHETIC records: findings are not company findings." if "synthetic" in quality["by_origin"] else "")}
    storage.write_json(out / "run_meta.json", meta)
    logger.info("Done: %s", meta)
    return meta
