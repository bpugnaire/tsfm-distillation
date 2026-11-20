import pytest
import torch

from tinycast.config import StudentConfig
from tinycast.models.student import Student


def test_multitoken_shapes_and_monotone():
    m = Student(StudentConfig(arch="patch_transformer_mt")).eval()
    x = torch.randn(4, 512).cumsum(1) + 10
    x[0, :300] = float("nan")
    q = m(x)
    assert q.shape == (4, 13, 64)
    assert torch.isfinite(q).all() and (q.diff(dim=1) >= 0).all()


def test_context_tokens_do_not_see_queries():
    """Two-stream mask: the context rows of the encoder output ignore the query tokens."""
    net = Student(StudentConfig(arch="patch_transformer_mt", dropout=0.0)).net.eval()
    z, mask = torch.randn(2, 512), torch.ones(2, 512, dtype=torch.bool)

    def context_out():
        zp, mp = z.view(2, net.n_patches, 16), mask.view(2, net.n_patches, 16).float()
        tok = (
            torch.cat([net.embed(torch.cat([zp * mp, mp], -1)), net.queries.expand(2, -1, -1)], 1)
            + net.pos
        )
        return net.encoder(tok, mask=net.attn_mask)[:, : net.n_patches]

    with torch.no_grad():
        a = context_out()
        net.queries.add_(1.0)
        b = context_out()
    assert torch.allclose(a, b, atol=1e-5)


def test_horizon_must_be_multiple_of_out_patch():
    with pytest.raises(ValueError):
        Student(StudentConfig(arch="patch_transformer_mt", horizon=60))
