"""Central configuration. Values come from environment variables / a local .env file."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _float_or_none(name: str) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Construct via :func:`get_settings` so env changes are picked up."""

    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_temperature: float | None = field(default_factory=lambda: _float_or_none("OPENAI_TEMPERATURE"))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("OPENAI_TIMEOUT", "60")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_RETRIES", "4")))
    max_transcript_chars: int = 24000
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    synthetic_dir: Path = ROOT / "data" / "synthetic"
    taxonomy_path: Path = ROOT / "config" / "taxonomy.json"


def get_settings() -> Settings:
    return Settings()
