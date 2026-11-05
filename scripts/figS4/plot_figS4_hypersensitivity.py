#!/usr/bin/env python3
"""SI Fig. S4: A-AFM Gamma-K0 bands on the ARPES cut, as-received vs symmetrized structure.

Panel (a) uses the as-received (experimental) structure, the same unfolded spectral weight as
Fig. 1(c); panel (b) uses the structure symmetrized to P6_322. Each panel is rendered by
scripts/fig1/plot_fig1c_arpes_overlay.plot_arpes_overlay with that calculation's own
ground-state Fermi level, then the two are placed side by side.

    python3 scripts/figS4/plot_figS4_hypersensitivity.py   # from the repository root

Writes output/fig1/appendix_aafm_{unsym_split,sym_degenerate}.png (the panels) and
output/figS4/figS4_hypersensitivity.{pdf,png}.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "fig1"))

from plot_fig1c_arpes_overlay import OUT_DIR, plot_arpes_overlay  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

DATA = REPO / "data"
PANEL_UNSYM = "appendix_aafm_unsym_split"
PANEL_SYM = "appendix_aafm_sym_degenerate"
OUT = REPO / "output" / "figS4" / "figS4_hypersensitivity"


def render_panels() -> None:
    # (a) as-received structure, with its own ground-state E_F (same value as Fig. 1(c)).
    plot_arpes_overlay(
        dft_path=DATA / "fig1" / "aafm_unfolded_sw.npy",
        out_name=PANEL_UNSYM,
        e_fermi=7.730112,
        show_colorbar=False,
    )
    # (b) structure symmetrized to P6_322, with that run's ground-state E_F.
    plot_arpes_overlay(
        dft_path=DATA / "figS4" / "aafm_symmetrized_unfolded_sw.npy",
        out_name=PANEL_SYM,
        e_fermi=7.7670,
        show_colorbar=False,
    )


def assemble() -> None:
    # plot_fig1c_arpes_overlay sets large publication rcParams at import; reset for the composite.
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    pt = 1 / 72  # inches per point
    box_w, box_h, y0 = 300.96, 361.44, 7.2
    fig_w, fig_h = 633.9030113636, box_h + 2 * y0  # no panel titles
    panels = (
        (7.2, PANEL_UNSYM, "(a)"),
        (325.8, PANEL_SYM, "(b)"),
    )

    fig = plt.figure(figsize=(fig_w * pt, fig_h * pt), dpi=100)
    for x0, name, label in panels:
        ax = fig.add_axes((x0 / fig_w, y0 / fig_h, box_w / fig_w, box_h / fig_h))
        ax.imshow(Image.open(OUT_DIR / f"{name}.png"), aspect="auto")
        ax.set_axis_off()
        ax.text(0.03, 0.99, label, transform=ax.transAxes, fontsize=15, ha="left", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.pdf", dpi=100)
    fig.savefig(f"{OUT}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT}.pdf and .png")


if __name__ == "__main__":
    render_panels()
    assemble()
