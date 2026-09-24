"""
Unfold the Wannier-interpolated Fermi surface onto the primitive 2H-NbS2 BZ.

Samples a *Cartesian* kx, ky grid (not a primitive-fractional grid) so the
resulting plot has a square k-range like a typical ARPES FS map.

Produces a 2D spectral-weight map integrated over a Gaussian centered at
E_F + shift (shift=0 → Fermi surface).

Overlays two hexagonal BZs:
    • primitive NbS2 BZ — black, solid  (|Γ-K|≈1.262 Å⁻¹)
    • √3×√3 SC BZ      — white, dashed (|Γ-K|≈0.729 Å⁻¹)

Usage:
    python plot_fermi_surface.py --shift 0.0 --krange 1.4 --nk 241
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from wannier_unfolding import (
    EFERMI, HR_FILES, WOUT_FILES, WIN_FILE, WORK_ROOT,
    parse_win, parse_wf_centres, parse_hr_dat,
    assign_wannier_to_prim, compute_spectral_weight,
    M_MATRIX, N_SC,
)


def gaussian_integrate(E: np.ndarray, P: np.ndarray,
                       E_target: float, sigma: float) -> np.ndarray:
    delta = (E - E_target) / sigma
    kernel = np.exp(-0.5 * delta ** 2) / (np.sqrt(2 * np.pi) * sigma)
    return (P * kernel).sum(axis=1)


def hex_corners(G_to_K: float, rotation_deg: float = 30.0) -> np.ndarray:
    """
    Return (6, 2) Cartesian corners of a regular hexagon of Γ-to-K radius.
    rotation_deg = angle of the FIRST corner (0° = flat top, 30° = pointy top).
    """
    angles = np.deg2rad(rotation_deg + np.arange(6) * 60)
    return np.stack([G_to_K * np.cos(angles), G_to_K * np.sin(angles)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shift", type=float, default=0.0,
                    help="Energy offset from E_F (eV). 0 = Fermi surface.")
    ap.add_argument("--nk", type=int, default=241,
                    help="Grid samples per axis.")
    ap.add_argument("--sigma", type=float, default=0.05,
                    help="Gaussian broadening in energy (eV).")
    ap.add_argument("--kz", type=float, default=0.0,
                    help="kz fractional (primitive recip).")
    ap.add_argument("--krange", type=float, default=1.4,
                    help="Half-range in Cartesian kx, ky [Å⁻¹].")
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--outdir", type=Path,
                    default=Path("unfolding_output"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Structure and β assignment (shared across spins)
    struct = parse_win(WIN_FILE)
    centres = parse_wf_centres(WOUT_FILES[1], 76)
    beta = assign_wannier_to_prim(centres, struct)

    # Primitive reciprocal lattice (rows = recip vectors, with 2π factor, 1/Å)
    A_prim = struct.A_prim
    B_prim = 2 * np.pi * np.linalg.inv(A_prim).T    # shape (3, 3)

    # Build a *Cartesian* kx, ky grid
    kx = np.linspace(-args.krange, args.krange, args.nk)
    ky = np.linspace(-args.krange, args.krange, args.nk)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")     # shapes (nk, nk)
    k_cart = np.stack([KX.ravel(), KY.ravel(), np.zeros(KX.size)], axis=-1)

    # Convert Cartesian k  → primitive-fractional k:
    #     k_cart = k_frac @ B_prim   →   k_frac = k_cart @ B_prim^{-1}
    B_prim_inv = np.linalg.inv(B_prim)
    k_frac_prim = k_cart @ B_prim_inv
    # Override kz to the requested fractional value
    k_frac_prim[:, 2] = args.kz
    print(f"k-grid: {args.nk}×{args.nk} Cartesian, |k| ≤ {args.krange:.2f} Å⁻¹  "
          f"→ {len(k_frac_prim)} points, kz_frac = {args.kz}")

    # Accumulate spectral weight over both spins
    A_total = np.zeros(len(k_frac_prim))
    for spin in (1, 2):
        print(f"  computing spin {spin}: reading H(R) ...", flush=True)
        R_vecs, wR, H_R, n_wann = parse_hr_dat(HR_FILES[spin])
        E, P = compute_spectral_weight(k_frac_prim, H_R, R_vecs, wR, beta, batch=256)
        A = gaussian_integrate(E, P, EFERMI + args.shift, args.sigma)
        print(f"    spin {spin} A range: [{A.min():.3g}, {A.max():.3g}]")
        A_total += A
    A2D = A_total.reshape(KX.shape)

    # ── Plot (style matched to scripts/plot_co_fermi_surface_side_by_side.py) ──
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    vmin = float(np.nanpercentile(A2D, 5.0))
    vmax = float(np.nanpercentile(A2D, 99.0))
    im = ax.imshow(A2D, origin="lower", aspect="equal", cmap=args.cmap,
                   extent=[-args.krange, args.krange, -args.krange, args.krange],
                   interpolation="bicubic", vmin=vmin, vmax=vmax)

    # Primitive BZ (pristine TaS₂): solid black
    K_prim_cart = 2 * np.pi * np.array([2 / 3, 1 / 3, 0]) @ np.linalg.inv(A_prim).T
    r_prim = np.linalg.norm(K_prim_cart[:2])
    ang0_prim = np.rad2deg(np.arctan2(K_prim_cart[1], K_prim_cart[0]))
    ax.add_patch(Polygon(hex_corners(r_prim, ang0_prim), closed=True, fill=False,
                         edgecolor="black", linewidth=1.5, zorder=3))

    # √3×√3 SC BZ: light-cyan dashed
    K_sc_cart = 2 * np.pi * np.array([1 / 3, 1 / 3, 0]) @ np.linalg.inv(A_prim).T
    r_sc = np.linalg.norm(K_sc_cart[:2])
    ang0_sc = np.rad2deg(np.arctan2(K_sc_cart[1], K_sc_cart[0]))
    ax.add_patch(Polygon(hex_corners(r_sc, ang0_sc), closed=True, fill=False,
                         edgecolor="#7fe7ff", linewidth=1.2,
                         linestyle=(0, (3.0, 2.0)), zorder=3))

    ax.set_xlim(-args.krange, args.krange)
    ax.set_ylim(-args.krange, args.krange)
    ax.set_xlabel(r"$k_x$ (Å$^{-1}$)", fontsize=18)
    ax.set_ylabel(r"$k_y$ (Å$^{-1}$)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_title(r"Co$_{1/3}$TaS$_2$ DFT", fontsize=18)

    fig.tight_layout()
    out = args.outdir / (f"unfolded_fs_shift{args.shift:+.2f}eV"
                          f"_kz{args.kz:.2f}_nk{args.nk}_cart.png")
    fig.savefig(out, dpi=200)
    print(f"\nSaved → {out}")
    # Also save raw FS for the multi-panel assembly script
    npz_out = out.with_suffix(".npz")
    kx_ax = np.linspace(-args.krange, args.krange, args.nk)
    np.savez(npz_out, kx=kx_ax, ky=kx_ax, intensity=A2D,
              shift=args.shift, sigma=args.sigma, kz=args.kz, efermi=EFERMI)
    print(f"Saved → {npz_out}")


if __name__ == "__main__":
    main()
