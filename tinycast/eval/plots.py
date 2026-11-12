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


def gallery_figure(
    suite: dict, runs: dict, models: list[str], shapes: list[str], path: Path, seed: int = 0
) -> None:
    """Small multiples: rows = shapes, columns = models; WQL under each cell (e-means style)."""
    apply_style()
    ids = suite["ids"]
    fig, axes = plt.subplots(
        len(shapes), len(models), figsize=(1.9 * len(models), 1.15 * len(shapes)), sharex=True
    )
    suite_levels = runs[models[0]]["quantile_levels"]
    for i, sh in enumerate(shapes):
        k = ids.index(f"{sh}#{seed}")
        ctx = np.array([np.nan if v is None else v for v in suite["context"][k]])
        fut = np.array(suite["future"][k])
        for j, m in enumerate(models):
            ax = axes[i, j]
            q = np.array(runs[m]["forecasts_f16"][k], dtype=float)
            fan(ax, ctx, fut, q, suite_levels, m, show_context=128)
            # y-limits from the data, not the (possibly extrapolated) 1%/99% tails
            obs = np.concatenate([ctx[-128:][np.isfinite(ctx[-128:])], fut])
            lo, hi = obs.min(), obs.max()
            pad = 0.35 * (hi - lo + 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
            w = runs[m]["per_series"]["wql"][k]
            ax.text(
                0.02,
                0.04,
                f"WQL {w:.3f}",
                transform=ax.transAxes,
                fontsize=6,
                color=ink("secondary"),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(m, fontsize=7)
            if j == 0:
                ax.set_ylabel(sh, fontsize=6, rotation=0, ha="right", va="center")
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
