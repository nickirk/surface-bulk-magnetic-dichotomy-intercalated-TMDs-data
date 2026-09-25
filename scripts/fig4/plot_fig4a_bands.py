"""Render Fig. 4(a) on the primitive M0--Gamma--K0 path.

Reads the raw spin-polarized VASP EIGENVAL of the band calculation
(data/fig4/panel_a_bulk_M0GK0_kz5_16_EIGENVAL; inputs in inputs/fig4/a_bulk_bands_M0GK0_kz5_16):

    python3 scripts/fig4/plot_fig4a_bands.py data/fig4/panel_a_bulk_M0GK0_kz5_16_EIGENVAL output/fig4/fig4a_bands.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


A_SC = np.array(
    [
        [5.7589, 0.0, 0.0],
        [-2.87945, 4.987353697854, 0.0],
        [0.0, 0.0, 12.1539],
    ]
)
B_SC = 2 * np.pi * np.linalg.inv(A_SC).T
EF = 7.7670
YLIM = (-2.54, 0.26)
UP, DN = "#205493", "#c04000"


def read_eigenval(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    _, nk, nb = [int(value) for value in lines[5].split()]
    kf = np.zeros((nk, 3))
    eu = np.zeros((nk, nb))
    ed = np.zeros((nk, nb))
    row = 6
    for ik in range(nk):
        while not lines[row].strip():
            row += 1
        kf[ik] = [float(value) for value in lines[row].split()[:3]]
        row += 1
        for ib in range(nb):
            fields = lines[row].split()
            eu[ik, ib] = float(fields[1])
            ed[ik, ib] = float(fields[2])
            row += 1
    return kf, eu, ed


def validate_path(kf: np.ndarray) -> int:
    expected_start = np.array([0.5, 0.5, 5.0 / 16.0])
    expected_end = np.array([0.0, 1.0, 5.0 / 16.0])
    if not np.allclose(kf[0], expected_start, atol=2e-6):
        raise ValueError(f"unexpected M0 endpoint: {kf[0]}")
    if not np.allclose(kf[-1], expected_end, atol=2e-6):
        raise ValueError(f"unexpected K0 endpoint: {kf[-1]}")
    if not np.allclose(kf[:, 2], 5.0 / 16.0, atol=2e-6):
        raise ValueError("kz is not fixed at 5/16")
    igamma = int(np.argmin(np.linalg.norm(kf[:, :2], axis=1)))
    if np.linalg.norm(kf[igamma, :2]) > 2e-6:
        raise ValueError(f"path does not pass through Gamma: {kf[igamma]}")
    return igamma


def render(eigenval: Path, output: Path) -> None:
    kf, eu, ed = read_eigenval(eigenval)
    # This band run started from a separately converged ground state whose spin
    # channels are labelled opposite to the Wannier run behind Fig. 4(b): its
    # channel 2 reproduces the Wannier channel 1 to 0.01 meV along M0-Gamma.
    # Exchange the columns so both panels use the same "up".
    eu, ed = ed, eu
    igamma = validate_path(kf)
    kc = kf @ B_SC
    kdist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))]
    ticks = [kdist[0], kdist[igamma], kdist[-1]]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 19,
            "axes.labelsize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    )
    # Exact overlay footprint used by the 735 x 617.39 bp full-page compositor.
    fig = plt.figure(figsize=(367.0 / 72.0, 292.0 / 72.0), facecolor="white")
    ax = fig.add_axes([0.215, 0.12, 0.60, 0.76])

    for energies, colour, zorder in [(ed, DN, 1), (eu, UP, 2)]:
        relative = energies - EF
        for ib in range(energies.shape[1]):
            if (relative[:, ib] > YLIM[0] - 0.3).any() and (
                relative[:, ib] < YLIM[1] + 0.3
            ).any():
                ax.plot(
                    kdist,
                    relative[:, ib],
                    color=colour,
                    lw=1.1,
                    alpha=0.85,
                    zorder=zorder,
                )

    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.axvline(ticks[1], color="0.6", lw=0.7, ls=":")
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$M_0$", r"$\Gamma$", r"$K_0$"])
    ax.set_xlim(kdist[0], kdist[-1])
    ax.set_ylim(*YLIM)
    ax.set_ylabel(r"$E-E_F$ (eV)")
    ax.legend(
        handles=[
            Line2D([], [], color=UP, lw=2, label=r"$\uparrow$"),
            Line2D([], [], color=DN, lw=2, label=r"$\downarrow$"),
        ],
        fontsize=15,
        loc="lower left",
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=False)
    fig.savefig(output.with_suffix(".png"), dpi=200, transparent=False)
    plt.close(fig)

    def max_split(section: slice) -> float:
        delta = np.abs(eu[section] - ed[section])
        near = np.abs(eu[section] - EF) < 1.5
        return 1000 * np.max(np.where(near, delta, 0.0))

    print(
        f"saved {output}; nk={len(kf)}, nb={eu.shape[1]}, "
        f"ticks={np.round(ticks, 6)}, "
        f"max split M0-Gamma={max_split(slice(0, igamma + 1)):.3f} meV, "
        f"Gamma-K0={max_split(slice(igamma, len(kf))):.3f} meV"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eigenval", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.eigenval, args.output)


if __name__ == "__main__":
    main()
