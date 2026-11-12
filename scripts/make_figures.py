"""make figures: regenerate every figure in results/figures from results/*.json"""

import json

from tinycast.data.shapes import SHAPE_NAMES, gallery
from tinycast.eval.plots import ROOT, gallery_figure, shapes_grid


def load_runs(suite_dir):
    return {
        json.loads(p.read_text())["model"]: json.loads(p.read_text())
        for p in sorted(suite_dir.glob("*.json"))
        if not p.name.startswith(("_", "summary"))
    }


if __name__ == "__main__":
    figs = ROOT / "results" / "figures"
    shapes_grid(gallery(), figs / "shapes_grid.png")
    sd = ROOT / "results" / "shapes"
    if (sd / "_suite.json").exists():
        suite = json.loads((sd / "_suite.json").read_text())
        runs = load_runs(sd)
        order = [
            "seasonal_naive",
            "ets",
            "toto-2.0-4m",
            "toto-2.0-22m",
            "toto-2.0-313m",
            "t0-gamma",
            "chronos-2-small",
            "chronos-2",
        ]
        order += sorted(m for m in runs if m.startswith("student"))
        models = [m for m in order if m in runs]
        half = len(SHAPE_NAMES) // 2
        gallery_figure(suite, runs, models, list(SHAPE_NAMES[:half]), figs / "gallery_a.png")
        gallery_figure(suite, runs, models, list(SHAPE_NAMES[half:]), figs / "gallery_b.png")
    print("figures ->", figs)
