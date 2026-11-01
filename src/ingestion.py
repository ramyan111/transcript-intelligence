"""Ingestion: CSV / JSON / JSONL / TXT files (or folders of them) -> one normalized DataFrame.

Original text and speaker turns are preserved. Missing metadata is never invented: an inferred call type
is flagged (``call_type_inferred``) and a generated id is flagged (``id_generated``).
"""
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt"}
CANONICAL_COLUMNS = ["transcript_id", "call_type", "call_date", "transcript_text",
                     "customer_id", "source_file", "data_origin"]
FLAG_COLUMNS = ["call_type_inferred", "id_generated"]

_ALIASES = {
    "transcript_id": ["transcript_id", "id", "call_id", "conversation_id"],
    "call_type": ["call_type", "type", "category", "call_category"],
    "call_date": ["call_date", "date", "timestamp", "created_at", "call_time"],
    "transcript_text": ["transcript_text", "transcript", "text", "content", "body", "conversation"],
    "customer_id": ["customer_id", "account_id", "client_id"],
    "data_origin": ["data_origin", "origin"],
}
_CALL_TYPE_MAP = {
    "support": "support", "customer support": "support", "customer_support": "support", "support call": "support",
    "external": "external", "external call": "external", "client": "external", "sales": "external",
    "account management": "external",
    "internal": "internal", "internal call": "internal", "engineering": "internal", "eng": "internal",
}


def normalize_call_type(value) -> str | None:
    """Map free-text call types to support/external/internal. None = missing, 'unknown' = unrecognized."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    key = str(value).strip().lower().replace("-", " ")
    if not key:
        return None
    return _CALL_TYPE_MAP.get(key, "unknown")


def infer_call_type_from_path(path: Path) -> str | None:
    """Infer a call type from folder / file name tokens (flagged as inferred by the caller)."""
    for part in reversed(path.with_suffix("").parts):
        low = part.lower()
        for token in ("support", "external", "internal"):
            if token in low:
                return token
    return None


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                  and not p.name.lower().startswith("readme"))


def _pick(rec: dict, field: str):
    lowered = {str(k).strip().lower(): v for k, v in rec.items()}
    for alias in _ALIASES[field]:
        value = lowered.get(alias)
        if value not in (None, ""):
            return value
    return None


def _coerce_text(value) -> str | None:
    """Accept plain strings or lists of {speaker, text} turn dicts; keep speaker labels."""
    if value is None:
        return None
    if isinstance(value, list):
        lines = []
        for turn in value:
            if isinstance(turn, dict):
                who = turn.get("speaker") or turn.get("role") or ""
                said = turn.get("text") or turn.get("utterance") or turn.get("content") or ""
                lines.append(f"{who}: {said}" if who else str(said))
            else:
                lines.append(str(turn))
        return "\n".join(lines)
    return str(value)


def _load_csv(path: Path) -> list:
    return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")


def _load_json(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(data, dict):
        for key in ("transcripts", "records", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return data


def _load_jsonl(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append(line)  # surfaces as a malformed (non-dict) record
    return out


def _load_txt(path: Path) -> list:
    return [{"transcript_id": path.stem, "transcript_text": path.read_text(encoding="utf-8", errors="replace")}]


_LOADERS = {".csv": _load_csv, ".json": _load_json, ".jsonl": _load_jsonl, ".txt": _load_txt}


def load_transcripts(path, default_origin: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Load transcripts from a file or folder. Returns (DataFrame, malformed_records)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    rows: list[dict] = []
    malformed: list[dict] = []
    for f in discover_files(path):
        origin = default_origin or ("synthetic" if "synthetic" in [p.lower() for p in f.parts] else "original")
        rel = f.relative_to(path) if path.is_dir() else Path(f.name)
        path_ct = infer_call_type_from_path(rel)
        try:
            records = _LOADERS[f.suffix.lower()](f)
        except Exception as exc:  # unreadable / invalid file: report, keep going
            malformed.append({"source_file": str(f), "error": f"{type(exc).__name__}: {exc}"})
            continue
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                malformed.append({"source_file": str(f), "index": i, "error": "record is not an object"})
                continue
            tid = _pick(rec, "transcript_id")
            id_generated = tid is None
            if id_generated:
                tid = f"{f.stem}-{i}"
            ct = normalize_call_type(_pick(rec, "call_type"))
            inferred = False
            if ct in (None, "unknown"):
                if path_ct:
                    ct, inferred = path_ct, True
                elif ct is None:
                    ct = "unknown"
            cust = _pick(rec, "customer_id")
            rows.append({
                "transcript_id": str(tid), "call_type": ct, "call_date": _pick(rec, "call_date"),
                "transcript_text": _coerce_text(_pick(rec, "transcript_text")),
                "customer_id": None if cust is None else str(cust),
                "source_file": str(f), "data_origin": _pick(rec, "data_origin") or origin,
                "call_type_inferred": inferred, "id_generated": id_generated,
            })
    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS + FLAG_COLUMNS)
    df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    logger.info("Loaded %d records from %s (%d malformed)", len(df), path, len(malformed))
    return df, malformed
