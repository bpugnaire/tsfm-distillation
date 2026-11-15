"""results/long_horizon/*@H -> results/long_horizon/summary.json + table.md (paired ratios on the same windows)."""

import json

import numpy as np

from tinycast.eval.benchmark import ROOT
from tinycast.eval.metrics import paired_ratio_ci

D = ROOT / "results" / "long_horizon"
PAIRS = [
    ("student-rollout-paths", "student-rollout-median", "sampled paths vs median feedback"),
    ("student-rollout-paths", "timestral-28m", "student (paths) vs its teacher"),
    ("student-rollout-paths", "toto-2.0-4m", "student (paths) vs Toto-2.0-4m"),
    ("student-rollout-paths", "chronos-bolt-tiny", "student (paths) vs Chronos-Bolt-tiny"),
]
out, md = (
    {},
    [
        "# Long horizons: student rolled out in 64-step blocks vs native long decoding (512-step context)",
        "",
    ],
)
for suite in ("shapes", "fresh_fr_power", "multidomain_test"):
    md += [f"## {suite}", "", "| H | windows | " + " | ".join(p[2] for p in PAIRS) + " | cov80 paths / median / teacher |",
           "|---|---|" + "---|" * len(PAIRS) + "---|"]  # fmt: skip
    for H in (64, 128, 256, 512):
        d = D / f"{suite}@{H}"
        if not (d / "summary.json").exists():
            continue
        runs = {
            p.stem: json.loads(p.read_text())
            for p in d.glob("*.json")
            if not p.name.startswith(("_", "summary"))
        }
        base = np.array(
            [np.nan if v is None else v for v in runs["seasonal_naive"]["per_series"]["wql"]]
        )
        groups = np.array(runs["seasonal_naive"]["groups"])
        cells, rec = [], {"n_windows": len(base), "pairs": {}}
        for a, b, label in PAIRS:
            wa = np.array([np.nan if v is None else v for v in runs[a]["per_series"]["wql"]])
            wb = np.array([np.nan if v is None else v for v in runs[b]["per_series"]["wql"]])
            ok = np.isfinite(wa) & np.isfinite(wb) & (wa > 0) & (wb > 0)
            r = paired_ratio_ci(wa[ok], wb[ok], groups[ok])
            rec["pairs"][label] = r
            cells.append(f"{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]")
        cov = [
            np.nanmean(runs[m]["per_series"]["cov80"])
            for m in ("student-rollout-paths", "student-rollout-median", "timestral-28m")
        ]
        rec["cov80"] = dict(zip(("paths", "median", "teacher"), cov, strict=True))
        out[f"{suite}@{H}"] = rec
        md.append(
            f"| {H} | {len(base)} | "
            + " | ".join(cells)
            + f" | {cov[0]:.2f} / {cov[1]:.2f} / {cov[2]:.2f} |"
        )
    md.append("")
(D / "summary.json").write_text(json.dumps(out, indent=1))
(D / "table.md").write_text("\n".join(md))
print("\n".join(md))
