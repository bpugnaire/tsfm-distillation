"""What does a teacher *do* on different kinds of input? -> results/input_families.json

Motivation (owner question 2026-11-13): can we distil on teacher forecasts of pure random noise?
In distillation the student learns f_teacher(context) only where it sees contexts, so the
question becomes: on which inputs is the teacher's function informative?

For each input family we feed 200 contexts to teachers and measure, in each context's own robust
scale (median / mean-abs-deviation, as the student sees it):
  * movement:  std over the horizon of the median forecast  / mean 80% interval width
               ~0 = "flat line + constant band": nothing to learn beyond "predict the level"
  * diversity: mean distance between normalised median curves of different contexts
               ~0 = every context gets the same answer: one function, not a mapping
  * width:     mean 80% interval width in context-scale units
Plus 3 example (context, quantiles) per family for the website.
"""

import json

import numpy as np

from tinycast.config import QUANTILE_LEVELS, ModelEntry
from tinycast.data.synthetic import kernel_synth
from tinycast.eval.benchmark import ROOT, git_sha
from tinycast.models import build_model

N, T, H = 200, 512, 64
TEACHERS = [
    ("toto", "toto-2.0-22m", "Datadog/Toto-2.0-22m"),
    ("t0", "t0-gamma", "theforecastingcompany/t0-gamma"),
]


def families(rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {
        "white_noise": rng.standard_normal((N, T)),
        "random_walk": np.cumsum(rng.standard_normal((N, T)), axis=1),
        "kernelsynth": np.stack(
            [kernel_synth(np.random.default_rng([s, 99]), T) for s in range(N)]
        ),
    }
    from tinycast.data.shapes import SHAPE_NAMES, make_shape

    out["random_shapes"] = np.stack(
        [
            make_shape(SHAPE_NAMES[s % len(SHAPE_NAMES)], 900_000 + s, T, H, randomize=True).context
            for s in range(N)
        ]
    )
    try:
        from tinycast.data.corpus import CorpusSpec, build_corpus

        out["real_electricity"] = build_corpus(CorpusSpec("val", N, 0, 0, T, H)).context
    except FileNotFoundError:
        pass
    return {k: v.astype(np.float32) for k, v in out.items()}


def robust_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loc = np.nanmedian(x, axis=1, keepdims=True)
    scale = np.nanmean(np.abs(x - loc), axis=1, keepdims=True) + 1e-6
    return loc, scale


def stats(ctx: np.ndarray, q: np.ndarray) -> dict:
    lv = list(QUANTILE_LEVELS)
    loc, scale = robust_scale(ctx)
    qn = (q - loc[:, :, None]) / scale[:, :, None]
    med = qn[:, lv.index(0.5)]
    width = (qn[:, lv.index(0.9)] - qn[:, lv.index(0.1)]).mean(1)
    movement = med.std(1) / width
    centred = med - med.mean(1, keepdims=True)
    idx = np.random.default_rng(0).integers(0, len(med), (2000, 2))
    diversity = np.abs(centred[idx[:, 0]] - centred[idx[:, 1]]).mean()
    return {
        "movement_median": float(np.median(movement)),
        "diversity": float(diversity),
        "width80_median": float(np.median(width)),
    }


if __name__ == "__main__":
    fam = families(np.random.default_rng(2026))
    res = {"n_per_family": N, "git_sha": git_sha(), "teachers": {}, "examples": {}}
    for kind, name, ck in TEACHERS:
        m = build_model(ModelEntry(kind=kind, name=name, checkpoint=ck, batch_size=100))
        res["teachers"][name] = {}
        for fname, x in fam.items():
            q = m.forecast(x, H)
            res["teachers"][name][fname] = stats(x, q)
            print(f"{name:14s} {fname:17s} {res['teachers'][name][fname]}", flush=True)
            if name == "toto-2.0-22m":
                show = [0, 1, 2]
                res["examples"][fname] = [
                    {
                        "context": [round(float(v), 4) for v in x[i, -192:]],
                        "q": [
                            [round(float(v), 4) for v in q[i, lv]] for lv in (0, 2, 4, 6, 8, 10, 12)
                        ],
                    }
                    for i in show
                ]
    res["example_levels"] = [QUANTILE_LEVELS[i] for i in (0, 2, 4, 6, 8, 10, 12)]
    (ROOT / "results" / "input_families.json").write_text(json.dumps(res))
