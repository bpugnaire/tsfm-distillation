""" "Your data": a target domain's own recent history, for light fine-tuning (study Stage D, RQ6).

A user with `history` steps of data has ONLY that period: windows (contexts included) are cut to it, so short
histories give short, left-padded contexts. The history ends right before the domain's test period:
  fr_power    hourly RTE series, history ends 1 h before splits.fresh_start (the fresh test starts there)
  pedestrian  hourly Melbourne pedestrian counts (held out of distillation), history ends at 85% of each series
              (where pedestrian_test targets start)
The last 20% of history origins are a fine-tuning validation set (early stopping), never the test.
"""

from __future__ import annotations

import numpy as np

from tinycast.data.corpus import Corpus
from tinycast.data.windows import window_id

DOMAINS = ("fr_power", "pedestrian")


def _domain_series(domain: str, end_time=None) -> list[tuple[str, np.ndarray, int]]:
    """(series id, values, index where the history ends = first test index). `end_time` (fr_power only) moves the
    end of the history earlier, e.g. to build "the same calendar month one year before" for diagnostics."""
    if domain == "fr_power":
        from tinycast.data.sources.rte_eco2mix import series_arrays
        from tinycast.data.splits import load_splits

        fresh = load_splits().fresh_start if end_time is None else end_time
        if fresh > load_splits().fresh_start:
            raise ValueError("fine-tuning history may not extend into the fresh test period")
        out = []
        for sid, (ts, v) in sorted(series_arrays().items()):
            out.append((sid, v, int(np.searchsorted(ts, fresh))))
        return out
    if domain == "pedestrian":
        from tinycast.data.sources import hf_datasets as H

        return [
            (s.series_id, s.values, int(0.85 * len(s.values)))
            for s in H.load("monash_pedestrian_counts")
        ]
    raise ValueError(f"unknown domain {domain}; expected one of {DOMAINS}")


def history_origins(
    end: int, history: int, min_context: int, horizon: int
) -> tuple[int, np.ndarray]:
    """(history start, forecast origins): every target [i, i + horizon) ends at or before `end` (the test start)."""
    start = max(0, end - history)
    return start, np.arange(start + min_context, end - horizon + 1)


def history_corpora(
    domain: str,
    history: int,
    context_length: int = 512,
    horizon: int = 64,
    min_context: int = 32,
    max_windows: int = 20000,
    seed: int = 0,
    mode: str = "cold",
    end_time=None,
) -> tuple[Corpus, Corpus]:
    """mode "cold": you only have `history` steps (contexts AND targets inside it; the pilot's setting).
    mode "warm": you have a longer past; contexts may use it (up to context_length), but every *target* used for
    fine-tuning lies inside the last `history` steps (the amount of supervision is what varies)."""
    if mode not in ("cold", "warm"):
        raise ValueError(mode)
    if mode == "cold" and history < min_context + horizon + 8:
        raise ValueError(f"history {history} too short for horizon {horizon}")
    rng = np.random.default_rng([seed, history, len(domain)])
    tr, va = ([], [], [], []), ([], [], [], [])
    for sid, v, end in _domain_series(domain, end_time):
        start, origins = history_origins(end, history, min_context, horizon)
        if (
            mode == "warm"
        ):  # targets [i, i + H) inside the last `history` steps; context from the full past
            origins = np.arange(max(min_context, end - history), end - horizon + 1)
            start = 0
        if len(origins) == 0:
            continue
        cut = int(0.8 * len(origins))
        for part, idx in ((tr, origins[:cut]), (va, origins[cut:])):
            for i in idx:
                ctx = np.full(context_length, np.nan, dtype=np.float32)
                c = v[max(start, i - context_length) : i]
                ctx[context_length - len(c) :] = c
                fut = v[i : i + horizon]
                if np.isfinite(fut).mean() < 0.9 or np.isfinite(c).sum() < min_context:
                    continue
                part[0].append(window_id(domain, sid, f"{mode}{history}:{i}"))
                part[1].append(domain)
                part[2].append(ctx)
                part[3].append(fut.astype(np.float32))

    def to_corpus(p, cap):
        keep = rng.permutation(len(p[0]))[:cap] if len(p[0]) > cap else np.arange(len(p[0]))
        return Corpus(
            np.array(p[0])[keep], np.array(p[1])[keep], np.stack(p[2])[keep], np.stack(p[3])[keep]
        )

    return to_corpus(tr, max_windows), to_corpus(va, max(500, max_windows // 5))
