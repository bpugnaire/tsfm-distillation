import numpy as np
import pytest

from tinycast.eval import contamination as C


def _series(seed, n=2000):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 50 + 10 * np.sin(2 * np.pi * t / 24) + np.cumsum(rng.normal(0, 1, n))


def test_detects_copied_rescaled_and_shifted_series():
    train = {"ours": [_series(0), _series(1)]}
    fp = C.Fingerprint.build(train)
    copied = _series(0)
    rescaled = (
        _series(1)[300:1500] * 1000 + 7
    )  # different units, offset, cropped at an arbitrary position
    unrelated = _series(42)
    assert C.check_dataset(fp, [copied])["seen_share"] == pytest.approx(1.0)
    r = C.check_dataset(fp, [rescaled])
    assert r["seen_share"] > 0.95 and r["matched_training_datasets"] == {"ours": r["n_chunks"]}
    assert C.check_dataset(fp, [unrelated])["seen_share"] == 0.0


def test_flat_and_missing_chunks_ignored():
    fp = C.Fingerprint.build({"zeros": [np.zeros(500)]})
    assert len(fp.hashes) == 0  # flat chunks carry no evidence
    x = _series(3)
    x[::7] = np.nan
    assert len(C.chunk_hashes(x)) == 0  # every chunk has a NaN


def test_tail_share_only_counts_the_end():
    a, b = _series(5), _series(6)
    fp = C.Fingerprint.build({"ours": [a[:1000]]})  # we trained on the first half only
    r = C.check_dataset(fp, [np.concatenate([a[:1000], b[:1000]])])
    assert r["seen_share"] == pytest.approx(0.5, abs=0.05)
    assert r["seen_share_tail"] == pytest.approx(0.0)


def test_sparse_spike_chunks_do_not_match_unrelated_series():
    """Regression (GIFT-Eval car_parts): a lone spike on a flat line looks the same in any series once z-normalised."""
    rng = np.random.default_rng(0)
    train = np.zeros(5000)
    train[rng.choice(5000, 40, replace=False)] = rng.uniform(1, 50, 40)  # occupancy-like blips
    bench = np.zeros(3000)
    bench[rng.choice(3000, 30, replace=False)] = 1.0  # intermittent demand counts
    fp = C.Fingerprint.build({"ours": [train, _series(0)]})
    assert C.check_dataset(fp, [bench])["seen_share"] == 0.0
    assert C.check_dataset(fp, [_series(0)])["seen_share"] == pytest.approx(
        1.0
    )  # real content still detected
