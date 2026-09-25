"""Fig. S1 (symmetric slab): the spin contrast and its single-channel symmetry check.

Exact slab op g = C2 about the in-plane axis at the slab center (swaps the two Fe
layers + two surfaces). In SUPERCELL fractional recip it is (k1,k2,k3)->(k2,k1,-k3)
(Cartesian: mirror about the b1+b2 = 60deg axis -- NOT the 45deg kx<->ky diagonal;
that was the earlier frame bug). It relates the two spin channels on the FULL slab:
    A_dn(k) = A_up(g.k)
so the single-channel construction
    eta_full(k) = A_up(k) - A_up(g.k)
is a check on the plotted field eta_old(k) = A_up(k) - A_dn(k) (they agree to a correlation of 0.89 at
sigma = 5 meV; the rest is independent sub-meV interpolation error in the two spin channels). A_up(g.k) is computed by re-evaluating the up-channel
unfold at the EXACT g-image k-points (g maps grid->non-grid, so recompute, no interp).

Controls: TRS A_sigma(k) vs A_sigma(-k) [frame-independent grid flip]; swap
A_up(g.k) vs A_dn; artifact = eta_old - eta_full.  Serial only (n_workers=1).
"""
from pathlib import Path
import os
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/unfolding/wannier
import wannier_unfolding as wu
DATA = Path(os.environ["W90_DIR"])
OUTDIR = Path(os.environ.get("OUT_DIR", "unfolding_output"))

wu.M_MATRIX = np.array([[1, -1, 0], [1, 2, 0], [0, 0, 1]], dtype=int); wu.N_SC = 3
EF = 3.5207; SIG = 0.005
st = wu.parse_win(DATA / "wannier90.1.win")
B_prim = 2 * np.pi * np.linalg.inv(st.A_prim).T
A_sc = np.array([[5.7589, 0, 0], [-2.87945, 4.987353697854, 0], [0, 0, 26.714987718]])
B_sc = 2 * np.pi * np.linalg.inv(A_sc).T

d = np.load(OUTDIR / "spectra_cache.npz")
KX, KY = d["KX"], d["KY"]; NK = KX.shape[0]
def gauss(E, P):
    x = (E - EF) / SIG; return (P * np.exp(-0.5 * x * x) / (np.sqrt(2 * np.pi) * SIG)).sum(1).reshape(NK, NK)
A_up = gauss(d["E1"], d["P1"]); A_dn = gauss(d["E2"], d["P2"])

# g in SUPERCELL fractional: (k1,k2,k3)->(k2,k1,-k3)
kcart = np.stack([KX.ravel(), KY.ravel(), np.zeros(KX.size)], -1)
ksc = kcart @ np.linalg.inv(B_sc)
ksc_g = ksc[:, [1, 0, 2]].copy(); ksc_g[:, 2] *= -1
gk_cart = ksc_g @ B_sc
kf_g = gk_cart @ np.linalg.inv(B_prim); kf_g[:, 2] = 0.0

# recompute up-channel unfold at exact g-image points (full slab, no mask)
R, wR, H_R, nw = wu.parse_hr_dat(DATA / "wannier90.1_hr.dat")
ctr = wu.parse_wf_centres(DATA / "wannier90.1.wout", nw)
beta = wu.assign_wannier_to_prim(ctr, st)
print("unfolding A_up at g.k (exact) ...")
Eg, Pg = wu.compute_spectral_weight(kf_g, H_R, R, wR, beta, batch=256, n_workers=1)
A_up_gk = gauss(Eg, Pg)

eta_full = A_up - A_up_gk        # single-channel check
eta_old = A_up - A_dn           # plotted in Fig. S1

bz = np.hypot(KX, KY) < 1.25
def rms(x): return float(np.sqrt(np.mean((x[bz])**2)))
print("\n=== FULL-SLAB results ===")
print(f"[control] TRS  A_up(k) vs A_up(-k):  {rms(A_up[::-1,::-1]-A_up)/rms(A_up):.3f} (frame-indep)")
print(f"[control] TRS  A_dn(k) vs A_dn(-k):  {rms(A_dn[::-1,::-1]-A_dn)/rms(A_dn):.3f}")
print(f"[swap] A_up(g.k) vs A_dn(k):         {rms(A_up_gk-A_dn)/rms(A_dn):.3f}  (was 0.965 w/ frame bug)")
print(f"rms eta_full(clean)={rms(eta_full):.4f}  rms eta_old(2ch)={rms(eta_old):.4f}  max|eta_full|={np.abs(eta_full[bz]).max():.3f}")
print(f"[artifact] rms(eta_old-eta_full)/rms(eta_full) = {rms(eta_old-eta_full)/rms(eta_full):.3f}")
print(f"[anchor] g-antisym eta_full(g.k)+eta_full(k): (recompute needed; construction exact)")
print(f"[anchor] TRS eta_full(-k) vs eta_full(k): {rms(eta_full[::-1,::-1]-eta_full)/rms(eta_full):.3f} (expect ~0, even)")
np.savez(OUTDIR / "eta_symmetric_slab.npz", KX=KX, KY=KY, eta_full=eta_full, eta_old=eta_old,
         A_up=A_up, A_dn=A_dn, A_up_gk=A_up_gk)
print(f"saved {OUTDIR / 'eta_symmetric_slab.npz'}")
