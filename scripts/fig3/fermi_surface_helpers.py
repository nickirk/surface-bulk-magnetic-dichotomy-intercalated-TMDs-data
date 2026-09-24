from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import PowerNorm

# Match the manuscript body + Fig 1 / Fig 2 font: serif with Computer Modern math
# for consistency across figures. Panel letters are rendered as CM
# mathtext below so they match Fig 1's cmr10 panel letters.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


# Font sizes raised for print legibility.
TITLE_FS = 17
LABEL_FS = 16
TICK_FS = 14
# Panel-letter size matched to Fig 1's baseline (assemble_fig1 LABEL_SZ=96 px on a
# 3450 px / 11.5" composite = 2.78% of the included width). Fig 3 is matplotlib
# (7.4" figure) included at 0.9\columnwidth, so PANEL_FS = 0.0278 * 7.4*72 / 0.9
# ≈ 16.5 pt gives the same on-page label size. Column width cancels.
PANEL_FS = 16.5


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "output" / "fig3"

DATA = ROOT / "data" / "fig3"
AAFM_MAPS = {m: DATA / f"{m}_aafm_fermi_surface.npz" for m in ("Co1_3NbS2", "Co1_3TaS2")}
# Experimental maps are not redistributed: build them from the published sources with the
# scripts in scripts/fig3/experiment/ (see README), which write to these paths.
EXP_CONBS2 = ROOT / "output" / "fig3_experiment" / "popcevic_fig6" / "Fig6_c.npz"
EXP_COTAS2 = ROOT / "output" / "fig3_experiment" / "sas_fig1" / "panel_b" / "interpolated_data.npz"


def read_poscar(filename: Path) -> np.ndarray:
    with filename.open() as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    lattice = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)], dtype=float)
    return lattice * scale


def reciprocal_lattice(lattice: np.ndarray) -> np.ndarray:
    a1, a2, a3 = lattice
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2.0 * np.pi * np.cross(a2, a3) / volume
    b2 = 2.0 * np.pi * np.cross(a3, a1) / volume
    b3 = 2.0 * np.pi * np.cross(a1, a2) / volume
    return np.array([b1, b2, b3])


def get_hex_bz_vertices(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    v1 = (2.0 * b1 - b2) / 3.0
    v2 = (b1 + b2) / 3.0
    v3 = (-b1 + 2.0 * b2) / 3.0
    return np.array([v1, v2, v3, -v1, -v2, -v3])


def load_bz_vertices(material: str) -> tuple[np.ndarray, np.ndarray]:
    prim_lat = read_poscar(DATA / f"{material}_POSCAR_PRIM")
    super_lat = read_poscar(DATA / f"{material}_POSCAR_SUPER")
    b_prim = reciprocal_lattice(prim_lat)
    b_super = reciprocal_lattice(super_lat)
    verts_prim = get_hex_bz_vertices(b_prim[0], b_prim[1])
    verts_super = get_hex_bz_vertices(b_super[0], b_super[1])
    return verts_prim, verts_super


def load_dft_fermi_surface_wannier(
    material: str,
    shift_ev: float = 0.0,
) -> tuple[np.ndarray, list[float]]:
    if shift_ev != 0.0:
        raise ValueError("only the E_F map (shift 0) is included in this data set")
    path = AAFM_MAPS[material]
    data = np.load(path)
    intensity = data["intensity"]
    extent = [float(data["kx"][0]), float(data["kx"][-1]), float(data["ky"][0]), float(data["ky"][-1])]
    return intensity, extent


def load_exp_fermi_surface_conbs2() -> tuple[np.ndarray, list[float]]:
    # Use the paper-oriented rendering of Fig. 6(c).
    data = np.load(EXP_CONBS2)
    intensity = np.rot90(np.flipud(data["intensity"].T), k=1)
    x = np.linspace(-1.1, 1.1, intensity.shape[1])
    y = np.linspace(-1.1, 1.1, intensity.shape[0])
    extent = [x.min(), x.max(), y.min(), y.max()]
    return intensity, extent


def load_exp_fermi_surface_cotas2() -> tuple[np.ndarray, list[float]]:
    data = np.load(EXP_COTAS2)
    intensity = data["intensity"].copy()
    intensity[~data["support_mask"]] = np.nan
    extent = [data["x"].min(), data["x"].max(), data["y"].min(), data["y"].max()]
    return intensity, extent


def overlay_bz(ax: plt.Axes, verts_prim: np.ndarray, verts_super: np.ndarray) -> None:
    vp = np.vstack([verts_prim[:, :2], verts_prim[0, :2]])
    vs = np.vstack([verts_super[:, :2], verts_super[0, :2]])
    ax.plot(vp[:, 0], vp[:, 1], color="black", linewidth=1.5, zorder=3)
    ax.plot(vs[:, 0], vs[:, 1], color="#7fe7ff", linewidth=1.2, linestyle=(0, (3.0, 2.0)), zorder=3)


SHARED_NORM = PowerNorm(gamma=0.7, vmin=0.0, vmax=1.0)


def rescale_intensity(intensity: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] using per-dataset p1/p99 so the same PowerNorm applies to all panels."""
    vmin = np.nanpercentile(intensity, 1)
    vmax = np.nanpercentile(intensity, 99)
    scaled = (intensity - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0)


def style_panel(
    ax: plt.Axes,
    intensity: np.ndarray,
    extent: list[float],
    title: str,
    verts_prim: np.ndarray,
    verts_super: np.ndarray,
    panel_label: str = "",
    mark_k: bool = False,
) -> None:
    ax.imshow(
        rescale_intensity(intensity),
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="inferno",
        norm=SHARED_NORM,
        interpolation="bicubic",
    )
    overlay_bz(ax, verts_prim, verts_super)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)", fontsize=LABEL_FS)
    ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    if mark_k:
        # The pocket discussed in the text is at K of the sqrt(3)xsqrt(3)
        # supercell BZ (cyan dashed hexagon), not at primitive-zone K0.  Use
        # the upper-left supercell vertex so the marker remains on the
        # experimental half when the final panels are split at kx=0.
        upper_left = verts_super[
            (verts_super[:, 0] < 0)
            & (verts_super[:, 1] > 0.25 * np.max(verts_super[:, 1]))
        ]
        if len(upper_left) != 1:
            raise RuntimeError("Could not identify the upper-left supercell K vertex")
        tip = upper_left[0, :2]
        ax.annotate(
            r"$K$", xy=(tip[0], tip[1]), xytext=(tip[0] - 0.42, tip[1] + 0.23),
            fontsize=PANEL_FS, color="white", ha="left", va="center", zorder=6,
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2,
                            shrinkA=6, shrinkB=3,
                            path_effects=[pe.withStroke(linewidth=4, foreground="black")]),
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
        )
    if panel_label:
        # top-left subfigure label; black glyph on a subtle white box so it reads
        # on both the white corner margins (Exp panels — box is invisible there)
        # and the dark data (DFT panels). A white *halo* traced the thin CM parens
        # and read as doubled "((b))" on the dark panels, so we use a box instead.
        # Placed outside the axes box: sits above the top-left
        # corner, left of the centred title. Outside the data area there is
        # nothing to read against, so the white backing box is dropped.
        ax.text(
            -0.20, 1.02, panel_label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=PANEL_FS, color="black", zorder=5,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot side-by-side experimental and DFT Fermi-surface maps for Co materials.")
    parser.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help="Energy shift in eV applied to both DFT Wannier FS datasets. Default: %(default)s",
    )
    parser.add_argument(
        "--outstem",
        default="fig3_co_fermi_surface_side_by_side",
        help="Output file stem written under manuscripts/figures. Default: %(default)s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        (
            r"Co$_{1/3}$NbS$_2$",
            "Co1_3NbS2",
            load_exp_fermi_surface_conbs2(),
            load_dft_fermi_surface_wannier("Co1_3NbS2", shift_ev=args.shift),
        ),
        (
            r"Co$_{1/3}$TaS$_2$",
            "Co1_3TaS2",
            load_exp_fermi_surface_cotas2(),
            load_dft_fermi_surface_wannier("Co1_3TaS2", shift_ev=args.shift),
        ),
    ]

    # Slightly smaller canvas + larger fonts => tighter plot area.
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4), constrained_layout=True)

    # CM mathtext so panel letters match Fig 1's cmr10.
    panel_labels = [[r"$\mathrm{(a)}$", r"$\mathrm{(b)}$"],
                    [r"$\mathrm{(c)}$", r"$\mathrm{(d)}$"]]
    for row, (label, material_key, exp_data, dft_data) in enumerate(datasets):
        verts_prim, verts_super = load_bz_vertices(material_key)
        style_panel(axes[row, 0], exp_data[0], exp_data[1], f"{label} Exp.", verts_prim, verts_super,
                    panel_label=panel_labels[row][0],
                    mark_k=(row == 0))
        style_panel(axes[row, 1], dft_data[0], dft_data[1], f"{label} DFT+U", verts_prim, verts_super,
                    panel_label=panel_labels[row][1])

        # Match the visible reciprocal-space window between experiment and DFT
        # within each material row, but expand it to fully contain the large BZ.
        exp_extent = exp_data[1]
        x_min = min(exp_extent[0], np.min(verts_prim[:, 0]))
        x_max = max(exp_extent[1], np.max(verts_prim[:, 0]))
        y_min = min(exp_extent[2], np.min(verts_prim[:, 1]))
        y_max = max(exp_extent[3], np.max(verts_prim[:, 1]))
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        span = max(x_max - x_min, y_max - y_min)
        pad = 0.03 * span
        half = 0.5 * span + pad
        for col in range(2):
            axes[row, col].set_xlim(x_center - half, x_center + half)
            axes[row, col].set_ylim(y_center - half, y_center + half)

    out_png = OUTDIR / f"{args.outstem}.png"
    out_pdf = OUTDIR / f"{args.outstem}.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
