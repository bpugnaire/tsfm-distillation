import numpy as np
import pytest

from tinycast.data import shapes as S


@pytest.mark.parametrize("name", S.SHAPE_NAMES)
@pytest.mark.parametrize("randomize", [False, True])
def test_deterministic_per_seed(name, randomize):
    a = S.make_shape(name, seed=7, randomize=randomize)
    b = S.make_shape(name, seed=7, randomize=randomize)
    np.testing.assert_array_equal(a.context, b.context)
    np.testing.assert_array_equal(a.future, b.future)
    c = S.make_shape(name, seed=8, randomize=randomize)
    assert not np.array_equal(np.nan_to_num(a.context), np.nan_to_num(c.context))


@pytest.mark.parametrize("name", S.SHAPE_NAMES)
def test_shapes_dtypes_and_meta(name):
    s = S.make_shape(name, seed=1, context_length=512, horizon=64)
    assert (
        s.context.shape == (512,) and s.future.shape == (64,) and s.future_noiseless.shape == (64,)
    )
    assert s.context.dtype == s.future.dtype == np.float32
    assert np.isfinite(s.future).all() and np.isfinite(s.future_noiseless).all()
    assert "period" in s.meta and "params" in s.meta and s.meta["name"] == name
    assert np.abs(s.future).sum() > 0 or name == "intermittent"


def test_only_gap_shape_has_nan():
    for name in S.SHAPE_NAMES:
        s = S.make_shape(name, seed=3)
        assert np.isnan(s.context).any() == (name == "missing_gaps"), name


def test_gallery_size_and_seeds():
    g = S.gallery()
    assert len(g) >= 18
    assert all(s.meta["seed"] >= S.GALLERY_SEED_OFFSET for s in g)


def test_synthetic_deterministic_and_finite_future():
    import time

    from tinycast.data.synthetic import synthetic_batch

    seeds = np.arange(64)
    t0 = time.time()
    c1, f1 = synthetic_batch(seeds)
    elapsed = time.time() - t0
    c2, f2 = synthetic_batch(seeds)
    np.testing.assert_array_equal(np.nan_to_num(c1), np.nan_to_num(c2))
    assert c1.shape == (64, 512) and f1.shape == (64, 64)
    assert np.isfinite(f1).all()
    assert elapsed < 10, f"synthetic generation too slow: {elapsed:.1f}s / 64"


def test_parallel_synthetic_identical_to_serial():
    from tinycast.data.synthetic import synthetic_batch, synthetic_series

    seeds = np.arange(5000, 10000)
    c, f = synthetic_batch(seeds)  # parallel path (>= 5000)
    for k in (0, 1234, 4999):
        np.testing.assert_array_equal(
            np.nan_to_num(np.concatenate([c[k], f[k]])),
            np.nan_to_num(synthetic_series(int(seeds[k]))),
        )
