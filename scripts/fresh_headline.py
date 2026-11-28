"""Pre-registered headline across the fresh datasets: the site student fine-tuned with replay on 1 year of history
(3 seeds, per-window scores averaged over seeds) vs every model zero-shot, paired ratios with a bootstrap over series.

    uv run python scripts/fresh_headline.py  -> results/fresh_headline.json

Electricity (Stage F) was scored on the refreshed 494-window suite; pedestrians from Stage G; solar, wind, bikes from
Stage H. Only the pre-registered arm is reported here; the other arms are in each study's table.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from tinycast.eval.benchmark import ROOT
from tinycast.eval.metrics import paired_ratio_ci

sys.path.insert(0, str(ROOT / "scripts"))
from study import per_series  # noqa: E402

SETS = [  # (label, study, suite, arm)
    ("Electricity consumption (FR)", "stage_f_finetune", "fresh_fr_power", "replay_fr_1y"),
    ("Solar production (FR)", "stage_h_fresh_multi", "fresh_fr_solar", "replay_solar_1y"),
    ("Wind production (FR)", "stage_h_fresh_multi", "fresh_fr_wind", "replay_wind_1y"),
    ("Bike counters (Paris)", "stage_h_fresh_multi", "fresh_paris_bike", "replay_bike_1y"),
    ("Pedestrian counts (Melbourne)", "stage_g_fresh_ped", "fresh_mel_ped", "replay_mel_1y"),
]
ZOO = ["timestral-5m", "toto-2.0-4m", "toto-2.0-22m", "timestral-28m", "chronos-bolt-tiny", "chronos-bolt-mini",
       "chronos-2-small", "toto-2.0-313m", "timestral-209m", "t0-gamma", "chronos-2"]  # fmt: skip


def main() -> None:
    out = {}
    for label, study, suite, arm in SETS:
        sdir = ROOT / "results" / "study" / study / suite
        meta = json.loads((sdir / "_suite.json").read_text())
        groups = [m.get("series_id", i) for m, i in zip(meta["meta"], meta["ids"], strict=True)]
        a = per_series(sdir, arm)
        zs = per_series(sdir, "zero_shot")
        sn = np.array(
            [
                np.nan if v is None else v
                for v in json.loads((sdir / "seasonal_naive.json").read_text())["per_series"]["wql"]
            ]
        )
        row = {"n_windows": len(meta["ids"]), "n_series": len(set(groups)),
               "rel_wql": float(np.exp(np.nanmean(np.log(a / sn)))), "vs_zero_shot_student": list(paired_ratio_ci(a, zs, groups=groups))}  # fmt: skip
        for z in ZOO:
            f = ROOT / "results" / suite / f"{z}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            if r["ids"] != meta["ids"]:
                raise ValueError(f"{z} on {suite}: windows differ from the study's")
            b = np.array([np.nan if v is None else v for v in r["per_series"]["wql"]])
            row[z] = list(paired_ratio_ci(a, b, groups=groups))
        out[label] = row
        print(f"\n{label}: {row['n_windows']} windows, {row['n_series']} series; fine-tuned student {row['rel_wql']:.3f}; "
              f"vs untouched student {row['vs_zero_shot_student'][0]:.3f}")  # fmt: skip
        for z in ZOO:
            if z in row:
                r = row[z]
                v = "WIN " if r[2] < 1 else "LOSS" if r[1] > 1 else "tie "  # noqa: E702
                print(f"   {v} vs {z:18s} {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]")
    (ROOT / "results" / "fresh_headline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
