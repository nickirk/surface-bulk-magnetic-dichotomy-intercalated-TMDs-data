"""
Sharp (≈ unbroadened) PZ-unfolded Fermi surface and η_z(k) order-parameter
diagnostics for the hex 2-Fe all-A-AFM slab.

Produces:
  figures/unfolded_FS_pz_sharp.{png,pdf}           — A↑, A↓, ΔA at σ ≈ 1 meV
  figures/eta_ring_fit.{png,pdf}                   — η_z(θ) at |k|=|K| + cos(6θ) fit
  figures/eta_sign_map.{png,pdf}                   — sign(η_z) over the BZ, with the 3 nodal lines
  figures/unfolded_FS_pz_sharp.npz                 — cached A↑, A↓ at sharp σ, plus E, P spectra

Recomputes the (E_n, P_n) spectrum once and caches it so future σ sweeps
are cheap.
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
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))            # scripts/unfolding/wannier
import wannier_unfolding as wu
# W90_DIR: directory with wannier90.{1,2}.win, wannier90.{1,2}.wout and wannier90.{1,2}_hr.dat
DATA = Path(os.environ["W90_DIR"])
OUTDIR = Path(os.environ.get("OUT_DIR", "unfolding_output"))
OUTDIR.mkdir(exist_ok=True, parents=True)

# Hex cell M-matrix (det = 3)
wu.M_MATRIX = np.array([[1, -1, 0], [1, 2, 0], [0, 0, 1]], dtype=int)
wu.N_SC = 3

wu.HR_DIR = DATA
wu.WIN_FILE = DATA / "wannier90.1.win"
wu.WOUT_FILES = {1: DATA / "wannier90.1.wout", 2: DATA / "wannier90.2.wout"}
wu.HR_FILES = {1: DATA / "wannier90.1_hr.dat", 2: DATA / "wannier90.2_hr.dat"}
wu.EFERMI = 2.7011

# Grid parameters — same as before
NK = int(os.environ.get("NK", 401))   # the published maps use 401
KRANGE = 1.4
SIGMA_SHARP = 0.002   # 2 meV — effectively the Fermi contour
SIGMA_LABEL = f"{SIGMA_SHARP*1e3:.0f} meV"

CACHE = OUTDIR / "spectra_cache.npz"


def gaussian_integrate(E, P, E0, sigma):
    d = (E - E0) / sigma
    return (P * np.exp(-0.5 * d * d) / (np.sqrt(2*np.pi) * sigma)).sum(axis=1)


def compute_or_load_spectra():
    """Return (KX, KY, E_per_spin, P_per_spin, struct)."""
    struct = wu.parse_win(wu.WIN_FILE)

    if CACHE.exists():
        d = np.load(CACHE)
        if int(d["nk"]) == NK and abs(float(d["krange"]) - KRANGE) < 1e-6:
            print(f"[cache] loaded {CACHE.name}")
            KX, KY = d["KX"], d["KY"]
            E_per = {1: d["E1"], 2: d["E2"]}
            P_per = {1: d["P1"], 2: d["P2"]}
            return KX, KY, E_per, P_per, struct
        print(f"[cache] grid mismatch; recomputing")

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
                        E1=E_per[1], P1=P_per[1],
                        E2=E_per[2], P2=P_per[2],
                        nk=NK, krange=KRANGE)
    return KX, KY, E_per, P_per, struct


def reciprocal(L):
    a1, a2, a3 = L
    V = np.dot(a1, np.cross(a2, a3))
    return np.array([
        2*np.pi * np.cross(a2, a3) / V,
        2*np.pi * np.cross(a3, a1) / V,
        2*np.pi * np.cross(a1, a2) / V,
    ])


def hex_vertices(b1, b2):
    return np.array([
        (2*b1 - b2)/3.0, (b1 + b2)/3.0, (-b1 + 2*b2)/3.0,
        (-2*b1 + b2)/3.0, (-b1 - b2)/3.0, (b1 - 2*b2)/3.0,
    ])


def plot_sharp_three_panel(KX, KY, A_up, A_dn, struct):
    delta = A_up - A_dn
    B_prim_no2pi = np.linalg.inv(struct.A_prim).T
    hex_xy = hex_vertices(B_prim_no2pi[0]*2*np.pi, B_prim_no2pi[1]*2*np.pi)[:, :2]
    closed_hex = np.vstack([hex_xy, hex_xy[0]])

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2))
    vmax_each = np.percentile(np.maximum(A_up, A_dn), 99.9)
    vmax_d = np.percentile(np.abs(delta), 99.9)
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
        ax.set_aspect("equal")
        ax.set_xlim(-KRANGE, KRANGE)
        ax.set_ylim(-KRANGE, KRANGE)
        ax.set_xlabel(r"$k_x$ (Å$^{-1}$)")
        ax.set_ylabel(r"$k_y$ (Å$^{-1}$)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        rf"Fe$_{{1/3}}$NbS$_2$ 2-Fe slab (HEX √3×√3) — PZ unfolded FS at "
        rf"$E_F$={wu.EFERMI:.3f} eV, σ = {SIGMA_LABEL} (near unbroadened)",
        fontsize=13)
    fig.tight_layout()
    out = OUTDIR / "unfolded_FS_pz_sharp.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def plot_eta_ring(KX, KY, A_up, A_dn, struct):
    """η_z(θ) on the ring |k|=|K|, with a cos(6θ) fit overlay."""
    eta = A_up - A_dn
    b = reciprocal(struct.A_prim)
    K_pt = (2*b[0, :2] - b[1, :2]) / 3.0
    Kmag = float(np.linalg.norm(K_pt))

    n_theta = 720
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    dk = KX[0, 1] - KX[0, 0]
    rows = (Kmag*np.sin(theta) - KY.min()) / dk
    cols = (Kmag*np.cos(theta) - KX.min()) / dk
    eta_ring = map_coordinates(eta, np.stack([rows, cols]), order=3,
                                mode="constant", cval=0)
    Asum_ring = map_coordinates(A_up + A_dn,
                                  np.stack([rows, cols]), order=3, mode="constant", cval=0)

    # Fourier coefficients
    coeffs = np.fft.rfft(eta_ring) / n_theta
    a0 = coeffs[0].real
    # cos(6θ) component amplitude (real-cos) and phase
    a6, b6 = 2 * coeffs[6].real, -2 * coeffs[6].imag
    amp6 = np.hypot(a6, b6)
    phi6 = np.arctan2(b6, a6)
    fit6 = a0 + amp6 * np.cos(6*theta - phi6)

    # Find K-point angular locations: argmax along ring at first 60° sectors
    # 6 K-points: at θ = 90° + n·60°  (top, ...)  -- depends on basis convention
    angle_K = np.degrees(np.arctan2(K_pt[1], K_pt[0]))
    K_angles = (angle_K + 60.0*np.arange(6)) % 360

    # Fit harmonic powers
    p = np.zeros(13)
    for l in range(13):
        p[l] = abs(coeffs[l])**2 if l == 0 else 2*abs(coeffs[l])**2
    p_total_nonconst = p[1:].sum() + 1e-30

    fig, (ax, axp) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                   gridspec_kw=dict(width_ratios=[2.2, 1]))

    ax.plot(np.degrees(theta), eta_ring,  color="#1f4e79", lw=1.4, label=r"$\eta_z(\theta)$")
    ax.plot(np.degrees(theta), fit6,      color="#c04000", lw=1.8, ls="--",
            label=rf"$a_0 + a_6\,\cos(6\theta-\phi_6)$,  $a_6$={amp6:.2f}")
    ax.axhline(0, color="grey", lw=0.6, alpha=0.6)
    for ka in K_angles:
        ax.axvline(ka, color="grey", lw=0.6, alpha=0.4)
    ax.set_xlim(0, 360)
    ax.set_xlabel(r"$\theta$ around the $|k|=|K|$ ring (deg)")
    ax.set_ylabel(r"$\eta_z(\theta) = A_\uparrow - A_\downarrow$")
    ax.set_title(rf"$\eta_z(\theta)$ on the K-point ring  ($|k|={Kmag:.3f}$ Å$^{{-1}}$)")
    ax.legend(loc="upper right", fontsize=10)

    # Power spectrum bar
    ls = np.arange(13)
    bars = axp.bar(ls, p / p_total_nonconst * 100, color="#1f4e79", alpha=0.85)
    # highlight l=6
    bars[6].set_color("#c04000")
    # label key harmonics
    annot = {0: "const", 2: "d", 3: "f", 4: "g", 6: "i (l=6)", 12: "i² (l=12)"}
    for l, lab in annot.items():
        if p[l] / p_total_nonconst * 100 < 0.5:
            continue
        axp.text(l, p[l]/p_total_nonconst*100 + 1.5, lab,
                 ha="center", va="bottom", fontsize=9)
    axp.set_xticks(np.arange(13))
    axp.set_xlabel(r"angular harmonic $l$")
    axp.set_ylabel(r"power $|c_l|^2 / \Sigma_{l>0} |c_l|^2$ (%)")
    axp.set_title("Fourier decomposition of $\\eta_z(\\theta)$")
    axp.set_ylim(0, max(20.0, 1.05*p.max()/p_total_nonconst*100))

    fig.suptitle(
        rf"Order parameter $\eta_z(\theta)$ on $|k|=|K|$ — i-wave ($l=6$) signature",
        fontsize=12)
    fig.tight_layout()
    out = OUTDIR / "eta_ring_fit.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)
    return phi6


def plot_eta_sign_map(KX, KY, A_up, A_dn, struct, phi6):
    """sign(η_z) over the full plane, with the 3 nodal lines through Γ."""
    eta = A_up - A_dn
    Asum = A_up + A_dn

    # mask out the low-spectral-weight regions (where signs are noise)
    thresh = np.percentile(Asum, 60)
    sig = np.where(Asum > thresh, np.sign(eta), 0.0)

    # nodal lines of cos(6θ - φ6) at θ such that 6θ - φ6 = π/2 + nπ
    n_lines = 3
    nodal_thetas = (np.pi/2 + np.pi*np.arange(2*n_lines) + phi6) / 6.0

    B_prim_no2pi = np.linalg.inv(struct.A_prim).T
    hex_xy = hex_vertices(B_prim_no2pi[0]*2*np.pi, B_prim_no2pi[1]*2*np.pi)[:, :2]
    closed_hex = np.vstack([hex_xy, hex_xy[0]])

    cmap = LinearSegmentedColormap.from_list(
        "blueredgrey",
        [(0, "#205493"), (0.5, "#e0e0e0"), (1, "#c04000")], N=64)
    extent = [KX.min(), KX.max(), KY.min(), KY.max()]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    im = ax.imshow(sig, origin="lower", extent=extent, cmap=cmap,
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.plot(closed_hex[:, 0], closed_hex[:, 1], color="black", lw=1.3)

    for t in nodal_thetas:
        r = KRANGE * 1.2
        ax.plot([-r*np.cos(t), r*np.cos(t)],
                [-r*np.sin(t), r*np.sin(t)],
                color="black", lw=1.0, ls="--", alpha=0.8)

    ax.set_aspect("equal")
    ax.set_xlim(-KRANGE, KRANGE)
    ax.set_ylim(-KRANGE, KRANGE)
    ax.set_xlabel(r"$k_x$ (Å$^{-1}$)")
    ax.set_ylabel(r"$k_y$ (Å$^{-1}$)")
    ax.set_title(
        r"sign($\eta_z(\mathbf{k})$): 3 nodal lines, 6 alternating sectors  (i-wave $l=6$)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
    cbar.set_ticklabels([r"$A_\downarrow > A_\uparrow$",
                          "below threshold",
                          r"$A_\uparrow > A_\downarrow$"])
    fig.tight_layout()
    out = OUTDIR / "eta_sign_map.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main():
    print(f"=== sharp (σ={SIGMA_LABEL}) FS + η_z analysis ===")
    KX, KY, E_per, P_per, struct = compute_or_load_spectra()

    print(f"broadening at σ = {SIGMA_LABEL} …")
    A_up = gaussian_integrate(E_per[1], P_per[1], wu.EFERMI, SIGMA_SHARP).reshape(KX.shape)
    A_dn = gaussian_integrate(E_per[2], P_per[2], wu.EFERMI, SIGMA_SHARP).reshape(KX.shape)

    np.savez_compressed(OUTDIR / "unfolded_FS_pz_sharp.npz",
                        kx=KX, ky=KY, A_up=A_up, A_dn=A_dn,
                        delta=A_up - A_dn,
                        fermi_energy=wu.EFERMI, sigma=SIGMA_SHARP,
                        nk=NK, krange=KRANGE)

    plot_sharp_three_panel(KX, KY, A_up, A_dn, struct)
    phi6 = plot_eta_ring(KX, KY, A_up, A_dn, struct)
    plot_eta_sign_map(KX, KY, A_up, A_dn, struct, phi6)


if __name__ == "__main__":
    main()
