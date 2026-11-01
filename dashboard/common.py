"""Shared dashboard helpers."""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Context:
    df: pd.DataFrame            # filtered records (one data origin)
    full_df: pd.DataFrame       # unfiltered records of the selected origin
    meta: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    clusters: dict | None = None
    failures: list = field(default_factory=list)
    taxonomy: list = field(default_factory=list)
    run_dir: str = ""
