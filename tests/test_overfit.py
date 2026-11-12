"""The student must overfit 64 windows (catches loss/scaler/head bugs cheaply)."""

import numpy as np
import pytest
import torch

from tinycast.config import StudentConfig
from tinycast.data.synthetic import synthetic_batch
from tinycast.distill import losses as L
from tinycast.distill.train import scaled_targets
from tinycast.models.student import Student


@pytest.mark.slow
def test_student_overfits_64_windows():
    torch.manual_seed(0)
    ctx, fut = synthetic_batch(np.arange(64))
    x, y = torch.tensor(ctx), torch.tensor(fut)
    m = Student(StudentConfig(d_model=64, n_layers=2, n_heads=4, dropout=0.0))
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    levels = m.c.quantile_levels
    first = None
    for _ in range(600):
        zq, loc, scale = m.forward_scaled(x)
        loss = L.pinball_gt(zq, scaled_targets(m, y, loc, scale), levels)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.1 * first, (first, float(loss))
