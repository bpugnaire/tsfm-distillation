"""All figures for README / articles. One style, defined in configs/style.yaml."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@cache
def style() -> dict:
    return yaml.safe_load((ROOT / "configs" / "style.yaml").read_text())


def role_of(model: str) -> str:
    m = model.lower()
    for prefix, role in style()["model_roles"].items():
        if m.startswith(prefix):
            return role
    raise KeyError(f"no role for model {model!r}: add it to configs/style.yaml")


def color(model_or_role: str, mode: str = "light") -> str:
    roles = style()["roles"]
    role = model_or_role if model_or_role in roles else role_of(model_or_role)
    return roles[role][mode]


def ink(name: str, mode: str = "light") -> str:
    return style()["ink"][name][mode]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": ink("surface"),
            "axes.facecolor": ink("surface"),
            "axes.edgecolor": ink("axis"),
            "axes.labelcolor": ink("secondary"),
            "axes.titlecolor": ink("primary"),
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": ink("grid"),
            "grid.linewidth": 0.5,
            "xtick.color": ink("muted"),
            "ytick.color": ink("muted"),
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.5,
            "font.family": "sans-serif",
            "legend.frameon": False,
            "legend.fontsize": 7,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def fan(ax, context, future, q, levels, model: str, show_context: int = 192) -> None:
    """Context (ink), truth (ink, dashed), median + quantile bands in the model's role colour."""
    T = len(context)
    t_ctx = np.arange(T - show_context, T)
    t_fut = np.arange(T, T + len(future))
    ax.plot(t_ctx, context[-show_context:], color=ink("secondary"), lw=1.0)
    ax.plot(t_fut, future, color=ink("primary"), lw=1.0, ls="--")
    c = color(model)
    lv = list(np.round(levels, 4))
    for lo, hi, op in style()["bands"]:
        if lo in lv and hi in lv:
            ax.fill_between(t_fut, q[lv.index(lo)], q[lv.index(hi)], color=c, alpha=op, lw=0)
    ax.plot(t_fut, q[lv.index(0.5)], color=c, lw=1.5)
    ax.axvline(T - 0.5, color=ink("axis"), lw=0.6)


def shapes_grid(shape_list, path: Path, ncols: int = 5) -> None:
    apply_style()
    n = len(shape_list)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 1.7 * nrows))
    for ax, s in zip(axes.flat, shape_list, strict=False):
        T = len(s.context)
        ax.plot(np.arange(T), s.context, color=ink("secondary"), lw=0.8)
        ax.plot(np.arange(T, T + len(s.future)), s.future, color=ink("primary"), lw=0.8, ls="--")
        ax.plot(np.arange(T, T + len(s.future)), s.future_noiseless, color=color("student"), lw=1.2)
        ax.axvline(T - 0.5, color=ink("axis"), lw=0.6)
        ax.set_title(s.name, loc="left")
        ax.set_xticks([])
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(
        "Gallery shapes: context (grey), observed future (dashed), conditional mean (blue)",
        fontsize=9,
        color=ink("primary"),
        x=0.01,
        ha="left",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
