#!/usr/bin/env python3
"""Candidate Fig. 2(b): matched-cell surface/bulk magnetic-energy U scan.

The plotted convention matches the manuscript:
    Delta E = E(A-AFM) - E(stripe), in meV per Fe.

Negative values favor A-AFM; positive values favor stripe. The script reads
absolute E0 totals and performs the per-Fe normalization explicitly so the
source data remain auditable.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]


DATA = REPO / "data" / "fig2" / "u_scan_e0.dat"
OUT_DIR = REPO / "output" / "fig2"
OUT_PDF = OUT_DIR / "fig2b_u_scan.pdf"
OUT_PNG = OUT_DIR / "fig2b_u_scan.png"

N_FE_SLAB = 8
N_FE_BULK = 4


def load_scan() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.loadtxt(DATA)
    u = values[:, 0]
    slab = (values[:, 1] - values[:, 2]) * 1000.0 / N_FE_SLAB
    bulk = (values[:, 3] - values[:, 4]) * 1000.0 / N_FE_BULK
    return u, slab, bulk


def main() -> None:
    u, slab, bulk = load_scan()

    panel_fs = 16.5
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": panel_fs,
            "axes.labelsize": 18,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.5,
            "figure.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(3.65, 3.25))
    ax.axhline(0.0, color="black", linewidth=1.2, zorder=1)

    # Okabe-Ito blue/orange provides color-vision-deficiency-safe separation;
    # square/circle markers retain a second, non-color encoding.
    bulk_color = "#0072B2"
    surface_color = "#D55E00"
    ax.plot(
        u,
        bulk,
        color=bulk_color,
        marker="s",
        markersize=6.8,
        linewidth=2.0,
        label="bulk",
        zorder=3,
    )
    ax.plot(
        u,
        slab,
        color=surface_color,
        marker="o",
        markersize=6.8,
        linewidth=2.0,
        label="surface",
        zorder=3,
    )

    ax.set_xlabel(r"$U$ (eV)")
    ax.set_ylabel(
        r"$E_{\mathrm{A\! -\! AFM}} - E_{\mathrm{stripe}}$"
        "\n"
        r"$\mathrm{(meV/Fe)}$"
    )
    # Keep points at their true numeric U values while using an even tick grid.
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"])
    ax.set_xlim(-0.08, 2.08)
    ax.set_yticks([-5, 0, 5, 10, 15, 20, 25])
    ax.set_ylim(-5.5, 26.0)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.47),
        fontsize=13,
        handlelength=1.5,
    )
    ax.tick_params(direction="out", length=4, width=1.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.25)

    params = {
        "delta_e_convention": "E(A-AFM)-E(stripe)",
        "energy_column": "E0",
        "n_fe_slab": N_FE_SLAB,
        "n_fe_bulk": N_FE_BULK,
        "u_eV": u.tolist(),
        "slab_meV_per_Fe": np.round(slab, 6).tolist(),
        "bulk_meV_per_Fe": np.round(bulk, 6).tolist(),
        "y_ticks_meV_per_Fe": [-5, 0, 5, 10, 15, 20, 25],
        "inset": False,
        "method": "matched-cell PBE+U matrix",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print("U(eV)  slab(meV/Fe)  bulk(meV/Fe)")
    for u_i, slab_i, bulk_i in zip(u, slab, bulk):
        print(f"{u_i:4.1f}  {slab_i:+12.3f}  {bulk_i:+12.3f}")
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
