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
