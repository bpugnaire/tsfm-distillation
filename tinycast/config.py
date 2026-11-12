"""Typed configs loaded from YAML.

No Hydra: a config is a plain (frozen-ish) dataclass tree, and `load(path, Cls)` builds it
recursively from YAML, *failing on unknown keys* so a typo never silently falls back to a
default.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

# Project-wide quantile grid (CLAUDE.md section 7).
QUANTILE_LEVELS: tuple[float, ...] = (
    0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
)  # fmt: skip
# Levels shared by every model (Toto 2.0 only emits these). Headline WQL uses them (ADR 0002).
CORE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ConfigError(ValueError):
    pass


def _build(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(
                f"{where}: expected a mapping for {tp.__name__}, got {type(value).__name__}"
            )
        hints = typing.get_type_hints(tp)
        fields = {f.name: f for f in dataclasses.fields(tp)}
        unknown = set(value) - set(fields)
        if unknown:
            raise ConfigError(f"{where}: unknown keys {sorted(unknown)} for {tp.__name__}")
        kwargs = {k: _build(hints[k], v, f"{where}.{k}") for k, v in value.items()}
        missing = [
            n
            for n, f in fields.items()
            if n not in kwargs
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ]
        if missing:
            raise ConfigError(f"{where}: missing required keys {missing} for {tp.__name__}")
        return tp(**kwargs)
    if origin in (types.UnionType, typing.Union):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        return _build(non_none[0], value, where)
    if origin in (list, tuple):
        args = typing.get_args(tp)
        inner = args[0] if args else Any
        items = [_build(inner, v, f"{where}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        _, vt = typing.get_args(tp)
        return {k: _build(vt, v, f"{where}.{k}") for k, v in value.items()}
    if tp is float and isinstance(value, int):
        return float(value)
    if tp in (int, float, str, bool) and not isinstance(value, tp):
        raise ConfigError(f"{where}: expected {tp.__name__}, got {value!r}")
    return value


def from_dict(cls: type[T], data: dict[str, Any], where: str = "config") -> T:
    return _build(cls, data, where)


def load(path: str | Path, cls: type[T]) -> T:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return from_dict(cls, data, where=str(path))


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StudentConfig:
    arch: str = "patch_transformer"  # or "mlp"
    context_length: int = 512
    horizon: int = 64
    patch_size: int = 16
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    use_asinh: bool = True
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS


@dataclasses.dataclass
class LossConfig:
    alpha: float = 0.0  # L = (1-alpha) L_gt + alpha L_kd
    kd: str = "pinball_teacher"  # pinball_teacher | quantile_huber
    horizon_weighting: str = "uniform"  # uniform | linear
    kd_levels: str = (
        "native"  # native: KD only on teacher's native levels; all: also extrapolated tails
    )


@dataclasses.dataclass
class DataConfig:
    suite: str = "synthetic"  # synthetic | fr_power | mix
    n_train_windows: int = 20000
    n_val_windows: int = 2000
    synthetic_share: float = 1.0
    seed: int = 0
    teacher: str | None = None  # name of a teacher cache, required when alpha > 0


@dataclasses.dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 0.05
    batch_size: int = 256
    max_steps: int = 2000
    warmup_steps: int = 100
    grad_clip: float = 1.0
    grad_accum: int = 1
    eval_every: int = 200
    patience: int = 5  # evals without val improvement before early stopping
    amp: bool = True  # only used on CUDA


@dataclasses.dataclass
class TrainConfig:
    name: str
    seed: int = 0
    device: str = "auto"
    student: StudentConfig = dataclasses.field(default_factory=StudentConfig)
    loss: LossConfig = dataclasses.field(default_factory=LossConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    wandb: bool = False


@dataclasses.dataclass
class ModelEntry:
    kind: str  # seasonal_naive | ets | toto | t0 | chronos2 | student | timestral
    name: str
    checkpoint: str | None = None
    batch_size: int = 32
    device: str = "cpu"


@dataclasses.dataclass
class BenchConfig:
    suite: str
    horizon: int = 64
    context_length: int = 512
    seed: int = (
        1_000_000  # gallery seeds are disjoint from training seeds (< 1e6), see test_leakage
    )
    n_seeds: int = 1  # instances per shape (shapes suite); gallery shows the first
    models: list[ModelEntry] = dataclasses.field(default_factory=list)
    latency: bool = True
