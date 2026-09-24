"""
PZ-unfolded sharp Fermi surface + η_z(k) analysis for the SYMMETRIC 2-Fe slab.

This is the control: with σ_h symmetry restored about the slab midplane and
[C₂⊥||i] exact, A↑(k) ≡ A↓(k) is symmetry-forced — η_z(k) should vanish to
numerical precision.

Same in-plane cell + M-matrix as the asymmetric hex task; only data paths
and num_wann differ.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))            # scripts/unfolding/wannier
import wannier_unfolding as wu
# W90_DIR: directory with wannier90.{1,2}.win, wannier90.{1,2}.wout and wannier90.{1,2}_hr.dat
DATA = Path(os.environ["W90_DIR"])
OUTDIR = Path(os.environ.get("OUT_DIR", "unfolding_output"))
OUTDIR.mkdir(exist_ok=True, parents=True)

# Hex √3×√3 → 1×1 NbS2 M-matrix (det = 3)
wu.M_MATRIX = np.array([[1, -1, 0], [1, 2, 0], [0, 0, 1]], dtype=int)
wu.N_SC = 3

wu.HR_DIR = DATA
wu.WIN_FILE = DATA / "wannier90.1.win"
wu.WOUT_FILES = {1: DATA / "wannier90.1.wout", 2: DATA / "wannier90.2.wout"}
wu.HR_FILES = {1: DATA / "wannier90.1_hr.dat", 2: DATA / "wannier90.2_hr.dat"}
wu.EFERMI = 3.5207  # ground-state SCF Fermi level of this slab

NK = int(os.environ.get("NK", 401))   # the published map uses 401
KRANGE = 1.4
SIGMA_SHARP = 0.002   # 2 meV
SIGMA_LABEL = f"{SIGMA_SHARP*1e3:.0f} meV"

CACHE = OUTDIR / "spectra_cache.npz"


def gaussian_integrate(E, P, E0, sigma):
    d = (E - E0) / sigma
    return (P * np.exp(-0.5 * d * d) / (np.sqrt(2*np.pi) * sigma)).sum(axis=1)


def compute_or_load_spectra():
    struct = wu.parse_win(wu.WIN_FILE)
    if CACHE.exists():
        d = np.load(CACHE)
        if int(d["nk"]) == NK and abs(float(d["krange"]) - KRANGE) < 1e-6:
            print(f"[cache] loaded {CACHE.name}")
            return d["KX"], d["KY"], {1: d["E1"], 2: d["E2"]}, {1: d["P1"], 2: d["P2"]}, struct
        print("[cache] grid mismatch; recomputing")

    kx = np.linspace(-KRANGE, KRANGE, NK)
    ky = np.linspace(-KRANGE, KRANGE, NK)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k_cart = np.stack([KX.ravel(), KY.ravel(), np.zeros(KX.size)], axis=-1)
    B_prim = 2*np.pi * np.linalg.inv(struct.A_prim).T
    k_frac_prim = k_cart @ np.linalg.inv(B_prim)
    k_frac_prim[:, 2] = 0.0

    E_per, P_per = {}, {}
    for spin in (1, 2):
        print(f"[unfold] spin {spin} …")
        R_vecs, wR, H_R, n_wann = wu.parse_hr_dat(wu.HR_FILES[spin])
        centres = wu.parse_wf_centres(wu.WOUT_FILES[spin], n_wann)
        beta = wu.assign_wannier_to_prim(centres, struct)
        E, P = wu.compute_spectral_weight(k_frac_prim, H_R, R_vecs, wR, beta,
                                          batch=128, n_workers=None)
        E_per[spin] = E
        P_per[spin] = P

    print(f"[cache] saving spectra to {CACHE.name}")
    np.savez_compressed(CACHE, KX=KX, KY=KY,
                        E1=E_per[1], P1=P_per[1], E2=E_per[2], P2=P_per[2],
                        nk=NK, krange=KRANGE)
    return KX, KY, E_per, P_per, struct


def hex_vertices(b1, b2):
    return np.array([
        (2*b1 - b2)/3.0, (b1 + b2)/3.0, (-b1 + 2*b2)/3.0,
        (-2*b1 + b2)/3.0, (-b1 - b2)/3.0, (b1 - 2*b2)/3.0,
    ])


def reciprocal(L):
    a1, a2, a3 = L
    V = np.dot(a1, np.cross(a2, a3))
    return np.array([
        2*np.pi*np.cross(a2, a3)/V,
        2*np.pi*np.cross(a3, a1)/V,
        2*np.pi*np.cross(a1, a2)/V,
    ])


def plot_sharp_three_panel(KX, KY, A_up, A_dn, struct):
    delta = A_up - A_dn
    B_prim_no2pi = np.linalg.inv(struct.A_prim).T
    hex_xy = hex_vertices(B_prim_no2pi[0]*2*np.pi, B_prim_no2pi[1]*2*np.pi)[:, :2]
    closed_hex = np.vstack([hex_xy, hex_xy[0]])

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2))
    vmax_each = np.percentile(np.maximum(A_up, A_dn), 99.9)
    vmax_d = max(np.percentile(np.abs(delta), 99.9), vmax_each * 1e-3)
    extent = [KX.min(), KX.max(), KY.min(), KY.max()]

    panels = [
        (axes[0], A_up,  "magma",     0,        vmax_each, r"$A_\uparrow$"),
        (axes[1], A_dn,  "magma",     0,        vmax_each, r"$A_\downarrow$"),
        (axes[2], delta, "coolwarm", -vmax_d,   vmax_d,    r"$\Delta A = A_\uparrow - A_\downarrow$"),
    ]
    for ax, dat, cmap, vmin, vmax, title in panels:
        im = ax.imshow(dat, origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="bicubic")
        hex_color = "white" if cmap == "magma" else "black"
        ax.plot(closed_hex[:, 0], closed_hex[:, 1], color=hex_color, lw=1.2)
        ax.set_aspect("equal"); ax.set_xlim(-KRANGE, KRANGE); ax.set_ylim(-KRANGE, KRANGE)
        ax.set_xlabel(r"$k_x$ (Å$^{-1}$)"); ax.set_ylabel(r"$k_y$ (Å$^{-1}$)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        rf"SYMMETRIC slab — PZ unfolded FS at $E_F$={wu.EFERMI:.3f} eV, "
        rf"σ = {SIGMA_LABEL}.  $[C_{{2\perp}}\|i]$-protected: expect A↑ ≡ A↓.",
        fontsize=12)
    fig.tight_layout()
    out = OUTDIR / "unfolded_FS_pz_sharp.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)

    rms_up = np.sqrt(np.mean(A_up**2))
    rms_dn = np.sqrt(np.mean(A_dn**2))
    rms_d = np.sqrt(np.mean(delta**2))
    print(f"\n  RMS A↑           = {rms_up:.4f}")
    print(f"  RMS A↓           = {rms_dn:.4f}")
    print(f"  RMS ΔA           = {rms_d:.4f}")
    print(f"  RMS ΔA / RMS (½(A↑+A↓)) = {rms_d/(0.5*(rms_up+rms_dn)):.6f}")
    print(f"  max|ΔA|          = {np.abs(delta).max():.4f}")
    print(f"  max(A↑+A↓)       = {(A_up+A_dn).max():.4f}")
    print(f"  max|ΔA|/max(sum) = {np.abs(delta).max()/(A_up+A_dn).max():.6f}")
    print(f"  net pol.         = {(A_up.sum()-A_dn.sum())/(A_up.sum()+A_dn.sum()):+.6f}")


def plot_eta_field(KX, KY, A_up, A_dn, struct):
    eta = A_up - A_dn
    B_prim_no2pi = np.linalg.inv(struct.A_prim).T
    hex_xy = hex_vertices(B_prim_no2pi[0]*2*np.pi, B_prim_no2pi[1]*2*np.pi)[:, :2]
    closed_hex = np.vstack([hex_xy, hex_xy[0]])

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    extent = [KX.min(), KX.max(), KY.min(), KY.max()]
    vmax = max(np.percentile(np.abs(eta), 99.5), 1e-3)
    cmap = LinearSegmentedColormap.from_list(
        "bwr_alt", [(0, "#205493"), (0.5, "#ffffff"), (1, "#c04000")], N=128)
    im = ax.imshow(eta, origin="lower", extent=extent, cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="bicubic")
    ax.plot(closed_hex[:, 0], closed_hex[:, 1], color="black", lw=1.3)

    b = reciprocal(struct.A_prim)
    K1 = (2*b[0, :2] - b[1, :2])/3.0
    for n in range(6):
        th = n*np.pi/3
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Kn = R @ K1
        ax.plot(Kn[0], Kn[1], "k+", markersize=14, markeredgewidth=1.6)

    ax.set_aspect("equal"); ax.set_xlim(-KRANGE, KRANGE); ax.set_ylim(-KRANGE, KRANGE)
    ax.set_xlabel(r"$k_x$ (Å$^{-1}$)"); ax.set_ylabel(r"$k_y$ (Å$^{-1}$)")
    ax.set_title(rf"$\eta_z(\mathbf{{k}}) = A_\uparrow - A_\downarrow$  "
                 rf"(SYM slab, sharp σ = {SIGMA_LABEL})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\eta_z$ (arb.)")
    fig.tight_layout()
    out = OUTDIR / "eta_field.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main():
    print(f"=== sharp PZ-unfolded FS for SYMMETRIC slab ===")
    print(f"  E_F = {wu.EFERMI} eV, σ = {SIGMA_LABEL}\n")
    KX, KY, E_per, P_per, struct = compute_or_load_spectra()

    print(f"\nbroadening …")
    A_up = gaussian_integrate(E_per[1], P_per[1], wu.EFERMI, SIGMA_SHARP).reshape(KX.shape)
    A_dn = gaussian_integrate(E_per[2], P_per[2], wu.EFERMI, SIGMA_SHARP).reshape(KX.shape)

    np.savez_compressed(OUTDIR / "unfolded_FS_pz_sharp.npz",
                        kx=KX, ky=KY, A_up=A_up, A_dn=A_dn,
                        delta=A_up - A_dn,
                        fermi_energy=wu.EFERMI, sigma=SIGMA_SHARP,
                        nk=NK, krange=KRANGE)

    plot_sharp_three_panel(KX, KY, A_up, A_dn, struct)
    plot_eta_field(KX, KY, A_up, A_dn, struct)


if __name__ == "__main__":
    main()
