import numpy as np
import pytest

from tinycast.data import cache as C


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ROOT", tmp_path)
    return tmp_path


def _fn(calls):
    def f(ctx):
        calls.append(len(ctx))
        base = ctx[:, -1:, None].repeat(64, -1)  # (n, 1, 64)
        return base + np.linspace(-1, 1, 13)[None, :, None]

    return f


def test_roundtrip_resume_and_alignment(tmp_root):
    ids = np.array([f"w{i}" for i in range(10)])
    ctx = np.arange(10 * 8, dtype=np.float32).reshape(10, 8)
    meta = {"checkpoint": "x", "levels": [0.0] * 13, "horizon": 64}
    calls = []
    C.write_cache("t", "k", ids, ctx, _fn(calls), meta, shard_size=4, log=lambda s: None)
    assert calls == [4, 4, 2]
    C.write_cache("t", "k", ids, ctx, _fn(calls), meta, shard_size=4, log=lambda s: None)
    assert calls == [4, 4, 2]  # resumed: nothing recomputed
    q, man = C.read_cache("t", "k", ids[::-1])
    assert q.shape == (10, 13, 64)
    np.testing.assert_allclose(q[0, 6], ctx[-1, -1], rtol=1e-3)  # aligned to requested order
    assert C.check_cache("t", "k", 10)["nan"] == 0
    with pytest.raises(KeyError):
        C.read_cache("t", "k", np.array(["nope"]))
    with pytest.raises(ValueError):  # different checkpoint -> refuse to mix caches
        C.write_cache(
            "t", "k", ids, ctx, _fn(calls), {**meta, "checkpoint": "y"}, log=lambda s: None
        )
    with pytest.raises(ValueError):  # same key, different corpus content -> refuse to mix
        C.write_cache("t", "k", ids[::-1], ctx, _fn(calls), meta, log=lambda s: None)


def test_large_values_survive_encoding():
    """Regression: v1 float16 overflowed above 65504 (Stage A). Values up to 1e9 must round-trip finitely and precisely."""
    rng = np.random.default_rng(0)
    base = rng.choice([1e-3, 1.0, 7e4, 3e6, 1e9], size=(20, 1, 1))
    q = np.sort(base * (1 + 0.1 * rng.standard_normal((20, 13, 64))), axis=1)
    out = C._decode(C._encode(q), 13, 64)
    assert np.isfinite(out).all()
    spread = q.max(axis=(1, 2), keepdims=True) - q.min(axis=(1, 2), keepdims=True)
    assert (np.abs(out - q) <= 1e-3 * spread + 1e-6 * np.abs(q)).all()


def test_teacher_caching_never_uses_mps(monkeypatch):
    """Regression: Toto-2.0-313m returned non-finite quantiles on Apple MPS for a constant context (Stage B)."""
    import inspect

    src = inspect.getsource(C.cache_teacher_on_corpus)
    assert '"mps"' not in src
