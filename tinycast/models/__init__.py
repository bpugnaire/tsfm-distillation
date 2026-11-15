"""Model zoo factory: build any Forecaster from a config entry."""

from __future__ import annotations

from tinycast.config import ModelEntry


def build_model(entry: ModelEntry):
    kind = entry.kind
    if kind == "seasonal_naive":
        from tinycast.models.baselines import SeasonalNaive

        return SeasonalNaive(entry.name, season=entry.season)
    if kind == "ets":
        from tinycast.models.baselines import ETS

        return ETS(entry.name, season=entry.season)
    if kind in ("toto", "t0", "chronos2", "chronos_bolt", "timestral"):
        from tinycast.models import teachers as T

        cls = {
            "toto": T.TotoAdapter,
            "t0": T.T0Adapter,
            "chronos2": T.Chronos2Adapter,
            "chronos_bolt": T.ChronosBoltAdapter,
            "timestral": T.TimestralAdapter,
        }[kind]
        return cls(entry.name, entry.checkpoint, device=entry.device, batch_size=entry.batch_size)
    if kind == "student":
        from tinycast.models.student import StudentForecaster

        return StudentForecaster.from_run(entry.checkpoint, name=entry.name, device=entry.device)
    if kind == "precomputed":
        from tinycast.models.precomputed import PrecomputedForecaster

        return PrecomputedForecaster(entry.name, entry.checkpoint)
    if kind == "student_onnx":
        from tinycast.export.onnx import OnnxForecaster

        return OnnxForecaster(entry.checkpoint, name=entry.name)
    raise ValueError(f"unknown model kind {kind!r}")
