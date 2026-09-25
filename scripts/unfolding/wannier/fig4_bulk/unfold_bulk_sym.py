"""Wannier unfolding of bulk A-AFM Fe1/3NbS2 onto the 1x1 NbS2 zone, for Fig. 4(b).

Writes the (E_n, P_n) cache on a regular in-plane grid at a fixed k_z; Fig. 4(b) uses
k_z = (5/16)(2 pi/c) on a 401 x 401 grid over +/-1.4 1/A:

    W90_DIR=<wannier dir> KZ_FRAC=0.3125 NK=401 python3 unfold_bulk_sym.py

The eta field is then A_up(k) - A_dn(k) (up = spin channel 1) at E_F = 7.7670 eV with a 5 meV Gaussian
(scripts/fig4/plot_fig4bc_eta_maps.py --from-caches).
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/unfolding/wannier
import wannier_unfolding as wu
DATA = Path(os.environ["W90_DIR"])
OUT = Path(os.environ.get("OUT_DIR", "unfolding_output"))
OUT.mkdir(exist_ok=True, parents=True)

# Hex √3×√3 → 1×1 NbS2 in-plane unfolding (same as the original)
wu.M_MATRIX = np.array([[1, -1, 0], [1, 2, 0], [0, 0, 1]], dtype=int)
wu.N_SC = 3

wu.HR_DIR = DATA
wu.WIN_FILE = DATA / "wannier90.1.win"
wu.WOUT_FILES = {1: DATA / "wannier90.1.wout", 2: DATA / "wannier90.2.wout"}
wu.HR_FILES = {1: DATA / "wannier90.1_hr.dat", 2: DATA / "wannier90.2_hr.dat"}
wu.EFERMI = 7.7670

NK = int(os.environ.get("NK", 401))
KRANGE = 1.4
SIGMA_E = 0.015
KZ_FRAC = float(os.environ.get("KZ_FRAC", 0.0))
KZ_LABEL = f"kz{int(round(KZ_FRAC*100)):03d}"
CACHE = OUT / f"spectra_cache_v2_{KZ_LABEL}.npz"


def compute_spectra():
    struct = wu.parse_win(wu.WIN_FILE)
    if CACHE.exists():
        d = np.load(CACHE)
        if (int(d["nk"]) == NK and abs(float(d["krange"]) - KRANGE) < 1e-6
                and abs(float(d["kz_frac"]) - KZ_FRAC) < 1e-6):
            print(f"[cache] loaded {CACHE.name}")
            return
    kx = np.linspace(-KRANGE, KRANGE, NK)
    ky = np.linspace(-KRANGE, KRANGE, NK)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k_cart = np.stack([KX.ravel(), KY.ravel(), np.zeros(KX.size)], axis=-1)
    B_prim = 2 * np.pi * np.linalg.inv(struct.A_prim).T
    k_frac_prim = k_cart @ np.linalg.inv(B_prim)
    k_frac_prim[:, 2] = KZ_FRAC

    E_per, P_per = {}, {}
    for spin in (1, 2):
        print(f"[unfold] spin {spin} at k_z_frac={KZ_FRAC} …")
        R_vecs, wR, H_R, n_wann = wu.parse_hr_dat(wu.HR_FILES[spin])
        centres = wu.parse_wf_centres(wu.WOUT_FILES[spin], n_wann)
        beta = wu.assign_wannier_to_prim(centres, struct)
        E, P = wu.compute_spectral_weight(k_frac_prim, H_R, R_vecs, wR, beta,
                                          batch=128, n_workers=None)
        E_per[spin], P_per[spin] = E, P

    np.savez_compressed(CACHE, KX=KX, KY=KY,
                        E1=E_per[1], P1=P_per[1], E2=E_per[2], P2=P_per[2],
                        nk=NK, krange=KRANGE, kz_frac=KZ_FRAC)
    print(f"[cache] saved {CACHE}")


if __name__ == "__main__":
    compute_spectra()
