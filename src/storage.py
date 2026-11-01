"""CSV / JSON / JSONL persistence. JSONL gives append-only checkpoints so runs are resumable."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return None if pd.isna(o) else o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def append_jsonl(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=_default) + "\n")


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt checkpoint line %d in %s", i, path)  # e.g. a crash mid-write
    return out


def latest_by_id(records: list[dict]) -> dict[str, dict]:
    """Last record wins per transcript_id."""
    return {r["transcript_id"]: r for r in records}


def merge_frame(transcripts: pd.DataFrame, analyses: list[dict]) -> pd.DataFrame:
    """Join transcripts with analyses. Only analyses whose text_hash matches the current transcript are used."""
    latest = latest_by_id(analyses)
    rows = []
    for rec in transcripts.to_dict("records"):
        a = latest.get(rec["transcript_id"])
        if a is None or a.get("text_hash") != rec["text_hash"]:
            continue
        merged = {**rec, **{k: v for k, v in a.items() if k not in ("transcript_id", "call_type", "text_hash")}}
        rows.append(merged)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
    return df


def load_processed(run_dir) -> dict | None:
    """Load everything a run produced, or None if the run has no analyses yet."""
    d = Path(run_dir)
    if not (d / "transcripts.csv").exists() or not (d / "analyses.jsonl").exists():
        return None
    tr = pd.read_csv(d / "transcripts.csv", keep_default_na=False, na_values=[""], dtype={"transcript_id": str})
    tr["call_date"] = pd.to_datetime(tr["call_date"], errors="coerce")
    for col in ("call_type_inferred", "id_generated"):
        tr[col] = tr[col].astype(str).str.lower().eq("true")
    df = merge_frame(tr, read_jsonl(d / "analyses.jsonl"))
    if df.empty:
        return None
    return {"df": df, "meta": read_json(d / "run_meta.json", {}), "quality": read_json(d / "quality_report.json", {}),
            "clusters": read_json(d / "clusters.json"), "failures": read_jsonl(d / "failures.jsonl"), "dir": d}
