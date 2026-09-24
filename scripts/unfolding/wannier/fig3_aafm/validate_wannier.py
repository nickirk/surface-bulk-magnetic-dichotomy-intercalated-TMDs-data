"""
Validate the Wannier tight-binding model against the original DFT eigenvalues
on the coarse k-mesh (read from wannier90.N.eig), especially around E_F.

For each k in the coarse mesh:
    1. Interpolate H(K) from Wannier H(R) → eigenvalues W_nk
    2. Read DFT eigenvalues D_nk from wannier90.N.eig (192 bands)
    3. Compare in several energy windows
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from wannier_unfolding import (
    WIN_FILE, HR_FILES, HR_DIR, WOUT_FILES, EFERMI,
    parse_win, parse_wf_centres, parse_hr_dat,
    assign_wannier_to_prim, interpolate_H_batch, M_MATRIX,
)

# Frozen window (from win file): dis_froz_min = 2.0, dis_froz_max = 9.0 [eV]
FROZ_MIN, FROZ_MAX = 2.0, 9.0
DIS_MIN, DIS_MAX = 1.0, 13.0


def parse_kpoints_from_win(win_file: Path):
    """Return (nk, 3) SC-fractional k-mesh from the begin kpoints / end kpoints block."""
    lines = [ln.strip() for ln in win_file.read_text().splitlines()]
    i0 = next(i for i, ln in enumerate(lines) if ln.lower() == "begin kpoints")
    i1 = next(i for i, ln in enumerate(lines) if ln.lower() == "end kpoints" and i > i0)
    kpts = []
    for ln in lines[i0 + 1: i1]:
        parts = ln.split()
        if len(parts) >= 3:
            kpts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(kpts)


def parse_eig(eig_file: Path, n_bands: int, n_k: int) -> np.ndarray:
    """Read Wannier .eig file → (nk, nbands) eigenvalue array (eV)."""
    arr = np.zeros((n_k, n_bands))
    with open(eig_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            ib = int(parts[0]) - 1
            ik = int(parts[1]) - 1
            ev = float(parts[2])
            arr[ik, ib] = ev
    return arr


def compare_spin(spin: int, struct):
    R_vecs, wR, H_R, n_wann = parse_hr_dat(HR_FILES[spin])
    K_sc = parse_kpoints_from_win(WIN_FILE)         # (nk, 3)
    n_k = len(K_sc)

    # Wannier-interpolated eigenvalues at the coarse k-mesh
    H_K = interpolate_H_batch(K_sc, R_vecs, wR, H_R)   # (nk, nw, nw)
    W_eigs = np.zeros((n_k, n_wann))
    for i in range(n_k):
        W_eigs[i] = np.linalg.eigvalsh(H_K[i])

    # DFT eigenvalues — assume 192 bands (matches win file num_bands)
    eig_file = HR_DIR / f"wannier90.{spin}.eig"
    n_dft_bands = 192
    D_eigs = parse_eig(eig_file, n_dft_bands, n_k)

    # For each k, find the best match between each Wannier eigenvalue W_nk and
    # nearest DFT eigenvalue D_mk in the disentanglement window. Report RMS.
    diffs_froz = []
    diffs_dis = []
    matched_pairs = []
    for ik in range(n_k):
        D_k = D_eigs[ik]
        W_k = W_eigs[ik]
        # Restrict to bands in the disentanglement window
        D_in_dis = D_k[(D_k >= DIS_MIN) & (D_k <= DIS_MAX)]
        W_in_dis = W_k[(W_k >= DIS_MIN) & (W_k <= DIS_MAX)]
        # Best one-to-one match: sort both and pair by index
        nmin = min(len(D_in_dis), len(W_in_dis))
        D_sorted = np.sort(D_in_dis)[:nmin]
        W_sorted = np.sort(W_in_dis)[:nmin]
        diff = W_sorted - D_sorted
        diffs_dis.extend(diff.tolist())
        for e in diff[(D_sorted >= FROZ_MIN) & (D_sorted <= FROZ_MAX)]:
            diffs_froz.append(e)
        matched_pairs.append((D_sorted, W_sorted))

    diffs_dis = np.array(diffs_dis)
    diffs_froz = np.array(diffs_froz)

    print(f"\n=== Spin {spin} ===")
    print(f"Coarse k-mesh: {n_k} k-points  |  Wannier bands: {n_wann}")
    print(f"Frozen window [{FROZ_MIN:.2f}, {FROZ_MAX:.2f}] eV "
          f"(E_F = {EFERMI:.3f}, i.e. E_F-{EFERMI-FROZ_MIN:.2f} to "
          f"E_F+{FROZ_MAX-EFERMI:.2f}):")
    print(f"  N matched pairs   : {len(diffs_froz)}")
    print(f"  mean(|ΔE|)        : {np.mean(np.abs(diffs_froz))*1000:.2f} meV")
    print(f"  RMS ΔE            : {np.sqrt(np.mean(diffs_froz**2))*1000:.2f} meV")
    print(f"  max(|ΔE|)         : {np.max(np.abs(diffs_froz))*1000:.2f} meV")
    print(f"Disentanglement window [{DIS_MIN:.2f}, {DIS_MAX:.2f}] eV:")
    print(f"  RMS ΔE            : {np.sqrt(np.mean(diffs_dis**2))*1000:.2f} meV")
    print(f"  max(|ΔE|)         : {np.max(np.abs(diffs_dis))*1000:.2f} meV")

    return K_sc, D_eigs, W_eigs, matched_pairs


def main():
    struct = parse_win(WIN_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for spin_idx, spin in enumerate((1, 2)):
        K_sc, D_eigs, W_eigs, pairs = compare_spin(spin, struct)
        n_k = len(K_sc)
        ax = axes[spin_idx]
        for ik in range(n_k):
            D_k = D_eigs[ik]
            W_k = W_eigs[ik]
            # Plot DFT as grey lines, Wannier as red dots, relative to E_F
            ax.plot(np.full_like(D_k, ik), D_k - EFERMI, "_", color="grey",
                    ms=3, alpha=0.4, mew=0.5)
            ax.plot(np.full_like(W_k, ik), W_k - EFERMI, "o", color="C3",
                    ms=2, alpha=0.8, mew=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(FROZ_MIN - EFERMI, color="b", lw=0.5, ls="--", label="frozen window")
        ax.axhline(FROZ_MAX - EFERMI, color="b", lw=0.5, ls="--")
        ax.axhline(DIS_MIN - EFERMI, color="g", lw=0.5, ls=":", label="dis. window")
        ax.axhline(DIS_MAX - EFERMI, color="g", lw=0.5, ls=":")
        ax.set_xlabel("coarse k-mesh index")
        ax.set_title(f"spin {spin}")
        ax.set_ylim(-3, 3)
        if spin_idx == 0:
            ax.set_ylabel("E - E$_F$ (eV)")
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("DFT (grey dashes) vs Wannier-interpolated (red dots)")
    fig.tight_layout()
    out = HR_DIR.parent / "figures" / "wannier_vs_dft_coarse_mesh.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"\nSaved comparison figure → {out}")


if __name__ == "__main__":
    main()
