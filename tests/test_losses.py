import numpy as np
import pytest
import torch
from scipy.stats import norm

from tinycast.config import QUANTILE_LEVELS
from tinycast.distill import losses as L


def test_level_mass_sums_to_one():
    w = L.level_mass(QUANTILE_LEVELS)
    assert float(w.sum()) == pytest.approx(1.0)
    assert float(w[0]) == pytest.approx(0.03)  # [0, (0.01+0.05)/2]
    assert float(w[6]) == pytest.approx(0.1)  # median: [0.45, 0.55]


def test_horizon_weights_mean_one():
    for kind in ("uniform", "linear"):
        w = L.horizon_weights(64, kind)
        assert float(w.mean()) == pytest.approx(1.0)
    w = L.horizon_weights(64, "linear")
    assert w[-1] > w[0]


def test_pinball_gt_hand_computed():
    # one level tau=0.9, y=10, q=8 -> rho = 1.8; times 2 * weight 1
    q = torch.tensor([[[8.0]]])
    y = torch.tensor([[10.0]])
    out = L.pinball_gt(q, y, [0.9], level_w=torch.tensor([1.0]))
    assert float(out) == pytest.approx(3.6)


def test_pinball_gt_mask_ignores_missing():
    q = torch.zeros(1, 1, 2)
    y = torch.tensor([[1.0, 1000.0]])
    mask = torch.tensor([[True, False]])
    out = L.pinball_gt(q, y, [0.5], mask=mask, level_w=torch.tensor([1.0]))
    assert float(out) == pytest.approx(1.0)  # 2 * 0.5 * 1


def test_kd_pinball_teacher_minimiser_is_teacher_quantile():
    """Optimising free student quantiles against a teacher's quantile set recovers the
    teacher's quantiles (the sample-pinball property)."""
    t_levels = np.linspace(0.005, 0.995, 199)
    qt = torch.tensor(norm.ppf(t_levels), dtype=torch.float32)[None, :, None]
    levels = (0.1, 0.5, 0.9)
    q = torch.zeros(1, 3, 1, requires_grad=True)
    opt = torch.optim.Adam([q], lr=0.02)
    for _ in range(1500):
        opt.zero_grad()
        L.kd_pinball_teacher(q, qt, levels, tuple(t_levels)).backward()
        opt.step()
    np.testing.assert_allclose(q.detach().numpy().ravel(), norm.ppf(levels), atol=0.03)


def test_kd_pinball_teacher_respects_level_mask():
    t_levels = (0.1, 0.5, 0.9)
    qt = torch.tensor([[[-1.0], [0.0], [1000.0]]])  # absurd 0.9 value, masked out
    q = torch.zeros(1, 1, 1)
    mask = torch.tensor([True, True, False])
    out = L.kd_pinball_teacher(q, qt, (0.5,), t_levels, teacher_level_mask=mask)
    assert float(out) < 10


def test_kd_quantile_huber_zero_when_equal():
    qt = torch.randn(2, 13, 4)
    out = L.kd_quantile_huber(qt.clone(), qt, list(range(13)))
    assert float(out) == pytest.approx(0.0)


def test_mixed_objective():
    a, b = torch.tensor(1.0), torch.tensor(3.0)
    assert float(L.mixed_objective(a, b, 0.5)) == pytest.approx(2.0)
    assert float(L.mixed_objective(a, None, 0.0)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        L.mixed_objective(a, None, 0.5)


def test_manual_asinh_sinh_match_torch():
    from tinycast.models.student import _asinh, _sinh

    z = torch.cat([torch.linspace(-1e4, 1e4, 2001), torch.linspace(-3, 3, 601)])
    torch.testing.assert_close(_asinh(z), torch.asinh(z), rtol=1e-6, atol=1e-6)
    w = torch.linspace(-15, 15, 601)
    torch.testing.assert_close(_sinh(w), torch.sinh(w), rtol=1e-5, atol=1e-6)
