"""Render the Fig. 4(b,c) spin-contrast maps and their colour bars.

sigma = 5 meV Gaussian at each calculation's ground-state E_F; the colour limit of each panel is the
99.5th percentile of |eta| inside |k| < 1.3 1/A; 128-level #205493 -> #ffffff -> #c04000 colour map;
bicubic interpolation; the maps are cropped to the primitive hexagon on its physical bounding box.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Polygon


SIGMA_E = 0.005
EF_BULK = 7.7670
EF_SLAB = 2.7011
MAP_DPI = 720
MAP_HEIGHT_PX = 2412
# Nearest integer width to the exact regular-hexagon ratio sqrt(3)/2.
MAP_WIDTH_PX = 2089
CBAR_WIDTH_PX = 540
CBAR_HEIGHT_PX = 2500

A_PRIM = np.array(
    [
        [2.87945, 1.66245123, 0.0],
        [-2.87945, 1.66245123, 0.0],
        [0.0, 0.0, 12.1539],
    ]
)


def make_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "bwr_alt",
        [(0, "#205493"), (0.5, "#ffffff"), (1, "#c04000")],
        N=128,
    )
    cmap.set_over("#3d0000")
    cmap.set_under("#001a3d")
    return cmap


def recip(lattice: np.ndarray) -> np.ndarray:
    a1, a2, a3 = lattice
    volume = np.dot(a1, np.cross(a2, a3))
    return np.array(
        [
            2 * np.pi * np.cross(a2, a3) / volume,
            2 * np.pi * np.cross(a3, a1) / volume,
            2 * np.pi * np.cross(a1, a2) / volume,
        ]
    )


def hex_vertices() -> np.ndarray:
    b1, b2 = recip(A_PRIM)[:2, :2]
    return np.array(
        [
            (2 * b1 - b2) / 3,
            (b1 + b2) / 3,
            (-b1 + 2 * b2) / 3,
            (-2 * b1 + b2) / 3,
            (-b1 - b2) / 3,
            (b1 - 2 * b2) / 3,
        ]
    )


def gauss(energies: np.ndarray, weights: np.ndarray, e0: float) -> np.ndarray:
    return (
        weights
        * np.exp(-0.5 * ((energies - e0) / SIGMA_E) ** 2)
        / (np.sqrt(2 * np.pi) * SIGMA_E)
    ).sum(axis=1)


def load_grid(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    kx = data["KX"] if "KX" in data.files else data["kx"]
    ky = data["KY"] if "KY" in data.files else data["ky"]
    if kx.shape != (401, 401) or ky.shape != (401, 401):
        raise ValueError(f"expected 401x401 grid, got {kx.shape} and {ky.shape}")
    return kx, ky


def bulk_eta(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        kx, ky = load_grid(data)
        up = gauss(data["E1"], data["P1"], EF_BULK).reshape(kx.shape)
    if not np.allclose(kx, -kx[::-1, ::-1]) or not np.allclose(
        ky, -ky[::-1, ::-1]
    ):
        raise ValueError("bulk k grid is not origin-symmetric")
    return kx, ky, up - up[::-1, ::-1]


def slab_eta(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        kx, ky = load_grid(data)
        up = gauss(data["E1"], data["P1"], EF_SLAB)
        down = gauss(data["E2"], data["P2"], EF_SLAB)
    return kx, ky, (up - down).reshape(kx.shape)


def robust_vmax(kx: np.ndarray, ky: np.ndarray, field: np.ndarray) -> float:
    return float(np.percentile(np.abs(field[np.hypot(kx, ky) < 1.3]), 99.5))


def render_map(
    output: Path,
    kx: np.ndarray,
    ky: np.ndarray,
    field: np.ndarray,
    vmax: float,
) -> None:
    vertices = hex_vertices()
    xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
    ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
    physical_ratio = (xmax - xmin) / (ymax - ymin)
    pixel_ratio = MAP_WIDTH_PX / MAP_HEIGHT_PX
    if abs(pixel_ratio / physical_ratio - 1) > 1e-4:
        raise ValueError(
            f"pixel aspect {pixel_ratio:.9f} mismatches physical {physical_ratio:.9f}"
        )

    fig = plt.figure(
        figsize=(MAP_WIDTH_PX / MAP_DPI, MAP_HEIGHT_PX / MAP_DPI),
        dpi=MAP_DPI,
        facecolor="none",
    )
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    image = ax.imshow(
        field,
        origin="lower",
        extent=[kx.min(), kx.max(), ky.min(), ky.max()],
        cmap=make_cmap(),
        vmin=-vmax,
        vmax=vmax,
        interpolation="bicubic",
        aspect="equal",
    )
    clip = Polygon(vertices, closed=True, transform=ax.transData)
    image.set_clip_path(clip)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=MAP_DPI,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close(fig)


def render_colorbar(output: Path, vmax: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(
        figsize=(CBAR_WIDTH_PX / MAP_DPI, CBAR_HEIGHT_PX / MAP_DPI),
        dpi=MAP_DPI,
        facecolor="white",
    )
    cax = fig.add_axes([0.16, 0.035, 0.22, 0.93])
    cbar = ColorbarBase(
        cax,
        cmap=make_cmap(),
        norm=Normalize(vmin=-vmax, vmax=vmax),
        orientation="vertical",
        ticks=[-vmax, 0, vmax],
    )
    rounded = int(round(vmax))
    cbar.ax.set_yticklabels([rf"$-{rounded}$", "$0$", rf"${rounded}$"])
    cbar.ax.tick_params(labelsize=13, width=0.9, length=3.5)
    cbar.outline.set_linewidth(0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=MAP_DPI, facecolor="white", edgecolor="none")
    plt.close(fig)


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "fig4"
BULK_FIELD = DATA / "eta_bulk_kz5_16_nk401.npz"
SLAB_FIELD = DATA / "eta_slab_asym_nk401.npz"


def save_field(path: Path, kx: np.ndarray, ky: np.ndarray, field: np.ndarray, **meta) -> None:
    """Store the eta map on its 1-D axes (the grid is a regular meshgrid, xy indexing)."""
    assert np.allclose(kx, kx[:1, :]) and np.allclose(ky, ky[:, :1])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, kx=kx[0], ky=ky[:, 0], eta=field, sigma_eV=SIGMA_E, **meta)


def load_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        kx, ky = np.meshgrid(data["kx"], data["ky"], indexing="xy")
        return kx, ky, data["eta"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Fig. 4(b,c). By default reads the eta fields in data/fig4; "
        "--from-caches first rebuilds those fields from Wannier-unfolded (E, P) caches "
        "(see scripts/unfolding/wannier/), which are too large to include here."
    )
    parser.add_argument("--from-caches", nargs=2, type=Path, metavar=("BULK_CACHE", "SLAB_CACHE"))
    parser.add_argument("--output-dir", type=Path, default=REPO / "output" / "fig4")
    args = parser.parse_args()

    if args.from_caches:
        kx, ky, field = bulk_eta(args.from_caches[0])
        save_field(BULK_FIELD, kx, ky, field, e_fermi_eV=EF_BULK,
                   definition="A_up(k) - A_up(-k), bulk A-AFM, k_z = 5/16 (2 pi/c)")
        kx, ky, field = slab_eta(args.from_caches[1])
        save_field(SLAB_FIELD, kx, ky, field, e_fermi_eV=EF_SLAB,
                   definition="A_up(k) - A_dn(k), asymmetric Fe-terminated slab")

    bulk_kx, bulk_ky, bulk_field = load_field(BULK_FIELD)
    bulk_vmax = robust_vmax(bulk_kx, bulk_ky, bulk_field)
    render_map(args.output_dir / "fig4b_eta_map.png", bulk_kx, bulk_ky, bulk_field, bulk_vmax)
    render_colorbar(args.output_dir / "fig4b_colorbar.png", bulk_vmax)
    del bulk_kx, bulk_ky, bulk_field
    gc.collect()

    slab_kx, slab_ky, slab_field = load_field(SLAB_FIELD)
    slab_vmax = robust_vmax(slab_kx, slab_ky, slab_field)
    render_map(args.output_dir / "fig4c_eta_map.png", slab_kx, slab_ky, slab_field, slab_vmax)
    render_colorbar(args.output_dir / "fig4c_colorbar.png", slab_vmax)

    # The published colour limits; a mismatch means a wrong input, Fermi level,
    # spin convention or broadening.
    if not np.isclose(bulk_vmax, 23.224439, atol=5e-6):
        raise ValueError(f"unexpected bulk vmax {bulk_vmax:.9f}")
    if not np.isclose(slab_vmax, 68.854526, atol=5e-6):
        raise ValueError(f"unexpected slab vmax {slab_vmax:.9f}")
    print(f"bulk_vmax={bulk_vmax:.6f}; slab_vmax={slab_vmax:.6f}; "
          f"map_pixels={MAP_WIDTH_PX}x{MAP_HEIGHT_PX}")


if __name__ == "__main__":
    main()
