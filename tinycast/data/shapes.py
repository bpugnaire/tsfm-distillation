"""Canonical toy shapes for the gallery (and, randomised, for training).

Every generator is a pure function of a `numpy.random.Generator` and returns the full
noiseless signal and the observed (noisy) signal over `context_length + horizon` steps.
`make_shape` splits them into context / future and attaches metadata.

Two modes:
* `randomize=False` (gallery): canonical parameters, only the noise depends on the seed.
* `randomize=True` (training): parameters are drawn from ranges around the canonical ones.

Gallery seeds are >= 1_000_000; training seeds are < 1_000_000 (tests/test_leakage.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

GALLERY_SEED_OFFSET = 1_000_000


@dataclass
class Shape:
    name: str
    context: np.ndarray  # (T,) float32, may contain NaN (missing)
    future: np.ndarray  # (H,) float32, observed future, never NaN
    future_noiseless: np.ndarray  # (H,) float32, conditional mean of the future
    meta: dict = field(default_factory=dict)


# Each generator: (rng, n, randomize) -> (noiseless, observed, period, params)
Gen = Callable[[np.random.Generator, int, bool], tuple[np.ndarray, np.ndarray, int | None, dict]]
_REGISTRY: dict[str, tuple[Gen, str]] = {}


def _register(name: str, description: str):
    def deco(fn: Gen) -> Gen:
        _REGISTRY[name] = (fn, description)
        return fn

    return deco


def _u(rng, lo, hi, randomize, default):
    return float(rng.uniform(lo, hi)) if randomize else default


def _pick(rng, options, randomize, default):
    return int(rng.choice(options)) if randomize else default


@_register("sine", "Single sinusoid, period 24, light noise")
def _sine(rng, n, r):
    p = _pick(rng, [12, 24, 48, 96], r, 24)
    a, lvl, s = _u(rng, 1, 5, r, 3.0), _u(rng, 5, 20, r, 10.0), _u(rng, 0.05, 0.3, r, 0.15)
    t = np.arange(n)
    f = lvl + a * np.sin(2 * np.pi * t / p)
    return f, f + rng.normal(0, s, n), p, dict(period=p, amp=a, level=lvl, noise=s)


@_register("two_seasonalities", "Daily (24) + weekly (168) seasonality, like electricity load")
def _two_seas(rng, n, r):
    a1, a2, s = _u(rng, 1, 4, r, 3.0), _u(rng, 0.5, 3, r, 1.5), _u(rng, 0.1, 0.4, r, 0.2)
    t = np.arange(n)
    f = 20 + a1 * np.sin(2 * np.pi * t / 24) + a2 * np.sin(2 * np.pi * t / 168 + 1.0)
    return f, f + rng.normal(0, s, n), 24, dict(a_daily=a1, a_weekly=a2, noise=s)


@_register("trend_seasonality", "Linear trend + seasonality")
def _trend_seas(rng, n, r):
    slope, a, s = _u(rng, 0.005, 0.05, r, 0.02), _u(rng, 1, 4, r, 2.0), _u(rng, 0.1, 0.4, r, 0.2)
    t = np.arange(n)
    f = 10 + slope * t + a * np.sin(2 * np.pi * t / 24)
    return f, f + rng.normal(0, s, n), 24, dict(slope=slope, amp=a, noise=s)


@_register("multiplicative_seasonality", "Seasonal amplitude grows with the level")
def _mult_seas(rng, n, r):
    g, s = _u(rng, 0.001, 0.004, r, 0.0025), _u(rng, 0.01, 0.04, r, 0.02)
    t = np.arange(n)
    level = 5 * np.exp(g * t)
    f = level * (1 + 0.4 * np.sin(2 * np.pi * t / 24))
    return f, f * (1 + rng.normal(0, s, n)), 24, dict(growth=g, noise_rel=s)


@_register("damped_trend", "Trend that flattens out (damped)")
def _damped(rng, n, r):
    phi, s = _u(rng, 0.98, 0.997, r, 0.993), _u(rng, 0.1, 0.5, r, 0.3)
    t = np.arange(n)
    f = 10 + 20 * (1 - phi**t)
    return f, f + rng.normal(0, s, n), None, dict(phi=phi, noise=s)


@_register("exponential_growth", "Exponential growth")
def _expo(rng, n, r):
    g, s = _u(rng, 0.002, 0.008, r, 0.005), _u(rng, 0.01, 0.04, r, 0.02)
    t = np.arange(n)
    f = 2 * np.exp(g * t)
    return f, f * (1 + rng.normal(0, s, n)), None, dict(growth=g, noise_rel=s)


@_register("logistic", "Logistic saturation (S-curve) reaching its plateau")
def _logistic(rng, n, r):
    k, mid, s = _u(rng, 0.01, 0.04, r, 0.02), _u(rng, 0.3, 0.7, r, 0.45), _u(rng, 0.1, 0.4, r, 0.2)
    t = np.arange(n)
    f = 5 + 20 / (1 + np.exp(-k * (t - mid * n)))
    return f, f + rng.normal(0, s, n), None, dict(rate=k, midpoint=mid, noise=s)


@_register("level_shift", "Seasonal series with a step change late in the context")
def _level_shift(rng, n, r):
    jump, at, s = _u(rng, -8, 8, r, 6.0), _u(rng, 0.6, 0.85, r, 0.75), _u(rng, 0.1, 0.4, r, 0.2)
    t = np.arange(n)
    f = 15 + 2 * np.sin(2 * np.pi * t / 24) + jump * (t >= int(at * n))
    return f, f + rng.normal(0, s, n), 24, dict(jump=jump, at=at, noise=s)


@_register("trend_break", "Flat, then a trend that starts late in the context")
def _trend_break(rng, n, r):
    slope, at, s = (
        _u(rng, 0.02, 0.1, r, 0.05),
        _u(rng, 0.5, 0.8, r, 0.65),
        _u(rng, 0.1, 0.4, r, 0.2),
    )
    t = np.arange(n)
    f = 10 + slope * np.maximum(t - at * n, 0)
    return f, f + rng.normal(0, s, n), None, dict(slope=slope, at=at, noise=s)


@_register("spikes", "Seasonal series with large outliers in the context only")
def _spikes(rng, n, r):
    rate, mag = _u(rng, 0.005, 0.03, r, 0.015), _u(rng, 5, 15, r, 10.0)
    t = np.arange(n)
    f = 10 + 2 * np.sin(2 * np.pi * t / 24)
    y = f + rng.normal(0, 0.2, n)
    spikes = (rng.uniform(size=n) < rate) * rng.choice([-1, 1], n) * mag
    spikes[-128:] = 0  # never in the future (and not right before it)
    return f, y + spikes, 24, dict(rate=rate, magnitude=mag)


@_register("heteroscedastic", "Noise level that grows over time")
def _hetero(rng, n, r):
    s0, s1 = _u(rng, 0.05, 0.2, r, 0.1), _u(rng, 0.8, 2.5, r, 1.5)
    t = np.arange(n)
    f = 10 + 2 * np.sin(2 * np.pi * t / 24)
    sd = s0 + (s1 - s0) * t / n
    return f, f + rng.normal(0, 1, n) * sd, 24, dict(noise_start=s0, noise_end=s1)


@_register("missing_gaps", "Seasonal series with gaps (NaN) in the context")
def _gaps(rng, n, r):
    f = 10 + 3 * np.sin(2 * np.pi * np.arange(n) / 24)
    y = f + rng.normal(0, 0.2, n)
    n_gaps = _pick(rng, [2, 3, 4, 5], r, 3)
    # gaps are applied to the context only, in make_shape (meta["gaps"])
    gaps = [
        (int(rng.integers(0, 0.8 * (n - 64))), int(rng.integers(10, 40))) for _ in range(n_gaps)
    ]
    return f, y, 24, dict(gaps=gaps)


@_register("random_walk", "Gaussian random walk (future mean = last value)")
def _rw(rng, n, r):
    s = _u(rng, 0.1, 1.0, r, 0.5)
    y = 50 + np.cumsum(rng.normal(0, s, n))
    return y.copy(), y, None, dict(step_sd=s, conditional_mean="last_value")


@_register("ar1", "Mean-reverting AR(1)")
def _ar1(rng, n, r):
    phi, s = _u(rng, 0.7, 0.98, r, 0.9), _u(rng, 0.3, 1.0, r, 0.5)
    y = np.empty(n)
    y[0] = 20 + rng.normal(0, s / np.sqrt(1 - phi**2))
    for i in range(1, n):
        y[i] = 20 + phi * (y[i - 1] - 20) + rng.normal(0, s)
    return y.copy(), y, None, dict(phi=phi, noise=s, mean=20.0, conditional_mean="ar1")


@_register("intermittent", "Intermittent demand: mostly zeros, occasional positive orders")
def _intermittent(rng, n, r):
    p, mu = _u(rng, 0.05, 0.3, r, 0.15), _u(rng, 2, 10, r, 5.0)
    occur = rng.uniform(size=n) < p
    size = rng.gamma(2.0, mu / 2.0, n)
    y = occur * np.ceil(size)
    return np.full(n, p * (mu + 0.5)), y, None, dict(prob=p, mean_size=mu)


@_register("sawtooth", "Sawtooth (slow ramp, instant reset)")
def _saw(rng, n, r):
    p, s = _pick(rng, [24, 48, 72], r, 48), _u(rng, 0.05, 0.3, r, 0.1)
    t = np.arange(n)
    f = 5 + 10 * ((t % p) / p)
    return f, f + rng.normal(0, s, n), p, dict(period=p, noise=s)


@_register("square_wave", "Square wave (on/off regime)")
def _square(rng, n, r):
    p, s = _pick(rng, [24, 48, 96], r, 48), _u(rng, 0.05, 0.3, r, 0.1)
    t = np.arange(n)
    f = 10 + 4 * np.sign(np.sin(2 * np.pi * t / p) + 1e-9)
    return f, f + rng.normal(0, s, n), p, dict(period=p, noise=s)


@_register("chirp", "Sinusoid whose frequency increases over time")
def _chirp(rng, n, r):
    f0, f1 = _u(rng, 1 / 96, 1 / 64, r, 1 / 80), _u(rng, 1 / 24, 1 / 12, r, 1 / 16)
    t = np.arange(n)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * n))
    f = 10 + 3 * np.sin(phase)
    return f, f + rng.normal(0, 0.15, n), None, dict(f_start=f0, f_end=f1)


@_register("impulse_decay", "Impulse followed by exponential decay back to baseline")
def _impulse(rng, n, r):
    at, tau = _u(rng, 0.75, 0.9, r, 0.85), _u(rng, 10, 40, r, 20.0)
    t = np.arange(n)
    t0 = int(at * n)
    f = 5 + 15 * np.exp(-(t - t0) / tau) * (t >= t0)
    return f, f + rng.normal(0, 0.2, n), None, dict(at=at, decay=tau)


@_register("constant_noise", "Constant level + white noise (the forecast should be flat)")
def _const(rng, n, r):
    lvl, s = _u(rng, 5, 50, r, 10.0), _u(rng, 0.2, 2.0, r, 1.0)
    return np.full(n, lvl), lvl + rng.normal(0, s, n), None, dict(level=lvl, noise=s)


SHAPE_NAMES: tuple[str, ...] = tuple(_REGISTRY)


def describe(name: str) -> str:
    return _REGISTRY[name][1]


def make_shape(
    name: str,
    seed: int,
    context_length: int = 512,
    horizon: int = 64,
    randomize: bool = False,
) -> Shape:
    fn, desc = _REGISTRY[name]
    # Independent stream per (name, seed): adding a shape never changes the others.
    name_key = int.from_bytes(name.encode(), "little") % (2**32)
    rng = np.random.default_rng([seed, name_key])
    n = context_length + horizon
    f, y, period, params = fn(rng, n, randomize)
    ctx = y[:context_length].astype(np.float32).copy()
    fut = y[context_length:].astype(np.float32).copy()
    fut_mean = f[context_length:].astype(np.float32).copy()
    if params.get("conditional_mean") == "last_value":
        fut_mean[:] = y[context_length - 1]
    elif params.get("conditional_mean") == "ar1":
        mu, phi = params["mean"], params["phi"]
        fut_mean = (mu + (y[context_length - 1] - mu) * phi ** np.arange(1, horizon + 1)).astype(
            np.float32
        )
    for start, length in params.get("gaps", []):
        ctx[start : start + length] = np.nan
    meta = dict(name=name, description=desc, seed=seed, period=period, params=params)
    return Shape(name, ctx, fut, fut_mean, meta)


def gallery(context_length: int = 512, horizon: int = 64, seed: int = 0) -> list[Shape]:
    """The fixed gallery set: one canonical instance per shape, seeds >= GALLERY_SEED_OFFSET."""
    return [
        make_shape(n, GALLERY_SEED_OFFSET + seed, context_length, horizon, randomize=False)
        for n in SHAPE_NAMES
    ]
