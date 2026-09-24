#!/usr/bin/env python3
"""Assemble four half-experiment/half-theory Fermi-surface comparisons.

Panels (a,c) compare experiment (left) with stripe DFT+U (right); panels
(b,d) compare the same experiment with A-AFM DFT+U.  Stripe maps are passed
explicitly because the two materials have separate calculation lineages.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from fermi_surface_helpers import (
    LABEL_FS,
    OUTDIR,
    PANEL_FS,
    ROOT,
    SHARED_NORM,
    TICK_FS,
    TITLE_FS,
    load_bz_vertices,
    load_dft_fermi_surface_wannier,
    load_exp_fermi_surface_conbs2,
    load_exp_fermi_surface_cotas2,
    overlay_bz,
    rescale_intensity,
)


def load_map(path: Path) -> tuple[np.ndarray, list[float]]:
    """Load the common Wannier-FS NPZ schema used by the A-AFM maps."""

    data = np.load(path)
    required = {"intensity", "kx", "ky"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{path} is missing required arrays: {', '.join(missing)}")
    intensity = np.asarray(data["intensity"])
    kx = np.asarray(data["kx"])
    ky = np.asarray(data["ky"])
    if intensity.shape != (len(ky), len(kx)):
        raise ValueError(
            f"{path}: intensity shape {intensity.shape} does not match "
            f"ky/kx lengths {(len(ky), len(kx))}"
        )
    extent = [float(kx[0]), float(kx[-1]), float(ky[0]), float(ky[-1])]
    return intensity, extent


def translate_map_x(
    data: tuple[np.ndarray, list[float]],
    shift: float,
) -> tuple[np.ndarray, list[float]]:
    """Translate a map horizontally without resampling its intensity grid."""

    intensity, extent = data
    shifted_extent = [extent[0] + shift, extent[1] + shift, extent[2], extent[3]]
    return intensity, shifted_extent


def visible_window(
    exp_extent: list[float],
    verts_prim: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return the square view used by the existing Fig. 3 panels."""

    x_min = min(exp_extent[0], float(np.min(verts_prim[:, 0])))
    x_max = max(exp_extent[1], float(np.max(verts_prim[:, 0])))
    y_min = min(exp_extent[2], float(np.min(verts_prim[:, 1])))
    y_max = max(exp_extent[3], float(np.max(verts_prim[:, 1])))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min)
    half = 0.5 * span + 0.03 * span
    return x_center - half, x_center + half, y_center - half, y_center + half


def add_supercell_k_marker(
    ax: plt.Axes,
    verts_super: np.ndarray,
) -> None:
    """Mark the visible pocket at upper-left K of the supercell BZ."""

    upper_left = verts_super[
        (verts_super[:, 0] < 0)
        & (verts_super[:, 1] > 0.25 * np.max(verts_super[:, 1]))
    ]
    if len(upper_left) != 1:
        raise RuntimeError("Could not identify the upper-left supercell K vertex")
    tip = upper_left[0, :2]
    ax.annotate(
        r"$K$",
        xy=(tip[0], tip[1]),
        xytext=(tip[0] - 0.42, tip[1] - 0.23),
        fontsize=PANEL_FS,
        color="white",
        ha="left",
        va="center",
        zorder=8,
        arrowprops=dict(
            arrowstyle="->",
            color="white",
            lw=1.3,
            mutation_scale=8,
            shrinkA=4,
            shrinkB=3,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
        ),
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
    )


def style_split_panel(
    ax: plt.Axes,
    exp_data: tuple[np.ndarray, list[float]],
    theory_data: tuple[np.ndarray, list[float]],
    title: str,
    phase_label: str,
    panel_label: str,
    verts_prim: np.ndarray,
    verts_super: np.ndarray,
    mark_k: bool = False,
    show_manual_labels: bool = True,
) -> None:
    """Draw experiment for kx<0 and one theory map for kx>0."""

    x_min, x_max, y_min, y_max = visible_window(exp_data[1], verts_prim)
    split_x = 0.0

    exp_image = ax.imshow(
        rescale_intensity(exp_data[0]),
        origin="lower",
        extent=exp_data[1],
        aspect="equal",
        cmap="inferno",
        norm=SHARED_NORM,
        interpolation="bicubic",
        zorder=1,
    )
    theory_image = ax.imshow(
        rescale_intensity(theory_data[0]),
        origin="lower",
        extent=theory_data[1],
        aspect="equal",
        cmap="inferno",
        norm=SHARED_NORM,
        interpolation="bicubic",
        zorder=1,
    )
    exp_image.set_clip_path(
        Rectangle((x_min, y_min), split_x - x_min, y_max - y_min, transform=ax.transData)
    )
    theory_image.set_clip_path(
        Rectangle((split_x, y_min), x_max - split_x, y_max - y_min, transform=ax.transData)
    )

    overlay_bz(ax, verts_prim, verts_super)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)", fontsize=LABEL_FS)
    ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.text(
        -0.20,
        1.02,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_FS,
        color="black",
        zorder=8,
    )

    if show_manual_labels:
        half_label_style = dict(
            transform=ax.transAxes,
            va="top",
            fontsize=12.5,
            zorder=8,
        )
        ax.text(0.03, 0.97, "Exp.", ha="left", color="black", **half_label_style)
        ax.text(
            0.97,
            0.97,
            phase_label,
            ha="right",
            color="white",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            **half_label_style,
        )
        if mark_k:
            add_supercell_k_marker(ax, verts_super)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conb-stripe", type=Path, default=ROOT / "data" / "fig3" / "Co1_3NbS2_stripe_fermi_surface.npz")
    parser.add_argument("--cota-stripe", type=Path, default=ROOT / "data" / "fig3" / "Co1_3TaS2_stripe_fermi_surface.npz")
    parser.add_argument(
        "--aafm-shift",
        type=float,
        default=0.0,
        help="Energy shift in eV for both existing A-AFM maps. Default: %(default)s",
    )
    parser.add_argument(
        "--conb-exp-kx-shift",
        type=float,
        default=0.08,
        help=(
            "Horizontal translation of the Co1/3NbS2 experimental map in "
            "inverse angstrom. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--outstem",
        default="fig3_fermi_surfaces",
        help="Output stem under output/fig3. Default: %(default)s",
    )
    parser.add_argument(
        "--omit-manual-labels",
        action="store_true",
        help=(
            "Omit the Exp., stripe, A-AFM, and K annotations so they can be "
            "added manually. Panel letters and material titles are retained."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    materials = [
        {
            "title": r"Co$_{1/3}$NbS$_2$",
            "key": "Co1_3NbS2",
            "exp": translate_map_x(
                load_exp_fermi_surface_conbs2(), args.conb_exp_kx_shift
            ),
            "stripe": load_map(args.conb_stripe),
            "aafm": load_dft_fermi_surface_wannier("Co1_3NbS2", args.aafm_shift),
        },
        {
            "title": r"Co$_{1/3}$TaS$_2$",
            "key": "Co1_3TaS2",
            "exp": load_exp_fermi_surface_cotas2(),
            "stripe": load_map(args.cota_stripe),
            "aafm": load_dft_fermi_surface_wannier("Co1_3TaS2", args.aafm_shift),
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4), constrained_layout=True)
    panel_labels = [
        [r"$\mathrm{(a)}$", r"$\mathrm{(b)}$"],
        [r"$\mathrm{(c)}$", r"$\mathrm{(d)}$"],
    ]
    for row, material in enumerate(materials):
        verts_prim, verts_super = load_bz_vertices(material["key"])
        style_split_panel(
            axes[row, 0],
            material["exp"],
            material["stripe"],
            material["title"],
            "stripe",
            panel_labels[row][0],
            verts_prim,
            verts_super,
            mark_k=(row == 0),
            show_manual_labels=not args.omit_manual_labels,
        )
        style_split_panel(
            axes[row, 1],
            material["exp"],
            material["aafm"],
            material["title"],
            "A-AFM",
            panel_labels[row][1],
            verts_prim,
            verts_super,
            show_manual_labels=not args.omit_manual_labels,
        )

    out_pdf = OUTDIR / f"{args.outstem}.pdf"
    out_png = OUTDIR / f"{args.outstem}.png"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
