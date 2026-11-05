"""SI Fig. S3, slab band structures: raw spin-polarized VASP EIGENVAL, drawn with the Fig. 4(a) style.

Three panels, each selected by name:
    asym_M0GK0  asymmetric Fe-terminated slab, M0-Gamma-K0 (the Fig. 4(a) in-plane path)
    sym_M0GK0   top-bottom symmetric slab, same path (nodal: up and down coincide)
    sym_M0GP    symmetric slab, M0-Gamma-P with P = (5/12, 1/6, 0) (primitive), the midpoint of the
                M0-K0 zone edge; Gamma-P crosses the region where the symmetric slab is spin split

    python3 scripts/figS3/plot_figS3_slab_bands.py sym_M0GP data/figS3/slab_sym_M0GP_EIGENVAL output/figS3/slab_sym_bands_M0GP.pdf

Spin channel 1 is plotted as up (red), as in the Fig. 4(c) and Fig. S2 maps of the same slabs.
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
        [0.0, 0.0, 1.0]  # replaced per panel in main(),
    ]
)
B_SC = 2 * np.pi * np.linalg.inv(A_SC).T
EF = 0.0  # set per panel in main()
END = (0.0, 1.0, 0.0)
END_LABEL = r"$K_0$"
YLIM = (-2.54, 0.26)
UP, DN = "#d62728", "#1f77b4"  # red up, blue down, as the structure arrows and eta maps


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
    expected_start = np.array([0.5, 0.5, 0.0])
    expected_end = np.array(END)
    if not np.allclose(kf[0], expected_start, atol=2e-6):
        raise ValueError(f"unexpected M0 endpoint: {kf[0]}")
    if not np.allclose(kf[-1], expected_end, atol=2e-6):
        raise ValueError(f"unexpected K0 endpoint: {kf[-1]}")
    if not np.allclose(kf[:, 2], 0.0, atol=2e-6):
        raise ValueError("kz is not 0")
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
    # slab runs: spin channel 1 = up, no exchange (same ground states as the Fig. 4(c)/S1 maps)
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
    ax.set_xticklabels([r"$M_0$", r"$\Gamma$", END_LABEL])
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


PANELS = {
    "asym_M0GK0": (2.7011, 22.110968859, (0.0, 1.0, 0.0), r"$K_0$"),
    "sym_M0GK0": (3.5207, 26.714987718, (0.0, 1.0, 0.0), r"$K_0$"),
    "sym_M0GP": (3.5207, 26.714987718, (0.25, 0.75, 0.0), r"$P$"),
}


def main() -> None:
    global EF, END, END_LABEL, A_SC, B_SC
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", choices=sorted(PANELS))
    parser.add_argument("eigenval", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    EF, c_slab, END, END_LABEL = PANELS[args.panel]
    A_SC = np.array([[5.7589, 0.0, 0.0], [-2.87945, 4.987353697854, 0.0], [0.0, 0.0, c_slab]])
    B_SC = 2 * np.pi * np.linalg.inv(A_SC).T
    render(args.eigenval, args.output)


if __name__ == "__main__":
    main()
