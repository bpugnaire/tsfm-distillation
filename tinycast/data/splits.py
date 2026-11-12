"""Load configs/data/splits.yaml as typed timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Splits:
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    fresh_start: pd.Timestamp
    model_releases: dict[str, pd.Timestamp]
    gallery_seed_offset: int


def load_splits(path: Path | None = None) -> Splits:
    d = yaml.safe_load((path or ROOT / "configs" / "data" / "splits.yaml").read_text())
    ts = {k: pd.Timestamp(d[k]) for k in ("train_end", "val_start", "val_end", "fresh_start")}
    rel = {k: pd.Timestamp(v, tz="UTC") for k, v in d["model_releases"].items()}
    return Splits(**ts, model_releases=rel, gallery_seed_offset=int(d["gallery_seed_offset"]))
