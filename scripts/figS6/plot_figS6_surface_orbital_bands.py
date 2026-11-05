"""SI Fig. S6: surface-orbital-projected bands of the relaxed A-AFM Fe-terminated slab on the ARPES cut.

Reads data/figS6/slab_aafm_unfolded_sw.npy (plane-wave unfolded weights on the primitive NbS2 cell,
Gamma -> K0) and data/figS6/slab_aafm_toplayer_orbital_weights.npz (PROCAR projections onto the top
layer: Fe d_z2, d_xz, d_yz; S p_z; Nb d_z2, d_xy, d_x2-y2), both made by
scripts/unfolding/planewave/figS6_slab_unfold_proj.py. Ribbon half-width = 0.1 x unfolded weight x
top-layer projection; no weight cutoff. ARPES processing and axes follow scripts/fig1. Energies use
the slab ground-state Fermi level (4.7672 eV), with no rigid shift.

    python3 scripts/figS6/plot_figS6_surface_orbital_bands.py   # from the repository root

Writes output/figS6/figS6_surface_orbital_bands.{pdf,png}.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "fig1"))
import plot_fig1c_arpes_overlay as f1  # noqa: E402  (style + ARPES loader)

E_F = 4.7672
DATA = REPO / "data" / "figS6"
OUT = REPO / "output" / "figS6"
UP, DN = "#d62728", "#1f77b4"
SCALE = 0.1

As = np.array([[5.7588999999999997, 0, 0], [0, 9.9747073957084069, 0], [0, 0, 34.3078]])
Au = np.array([[2.8794499999999998, 1.6624512326180678, 0],
               [-2.8794499999999994, 1.6624512326180678, 0], [0, 0, 34.3078]])
M = np.round(np.linalg.solve(Au.T, As.T).T).astype(int)
cell = np.linalg.inv(M) @ As


def kdist_of(nk):
    kf = np.array([[1. / 3 * i / (nk - 1), 1. / 3 * i / (nk - 1), 0.0] for i in range(nk)])
    kc = kf @ np.linalg.inv(cell).T * 2 * np.pi
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))]


sw = np.load(DATA / "slab_aafm_unfolded_sw.npy")     # (nspin, nk, nb, 2): energy, unfolded weight
o = np.load(DATA / "slab_aafm_toplayer_orbital_weights.npz")
weight = sw[..., 1] * (o["fe_d"] + o["s_pz"] + o["nb_d"])

k_vals, energies, inten = f1._load_arpes()
vmax = np.percentile(inten, 97.)
processed = np.clip(inten, 0, vmax) ** 0.9
fig, ax = plt.subplots()
fig.subplots_adjust(left=f1.AX_LEFT, right=f1.AX_RIGHT, top=f1.AX_TOP, bottom=f1.AX_BOTTOM)
ax.imshow(processed.T, origin="lower", extent=[k_vals.min(), k_vals.max(), energies.min(), energies.max()],
          cmap="Greys", aspect="auto", interpolation="bicubic")
a0 = 3.3249
k_K = 4 * np.pi / (3 * a0)
for kp in (0, k_K / 2, k_K):
    ax.axvline(kp, color="white", linestyle="--", linewidth=1.5, alpha=0.6)
kd = kdist_of(sw.shape[1])
assert abs(kd[-1] - k_K) < 2e-3, (kd[-1], k_K)
for ispin in (0, 1):
    col = (UP, DN)[ispin]
    for ib in range(sw.shape[2]):
        E = sw[ispin, :, ib, 0] - E_F
        W = weight[ispin, :, ib]
        if W.max() > 0.0 and E.max() > -5 and E.min() < 2:
            env = W * SCALE
            ax.fill_between(kd, E - env, E + env, where=W > 0.0, color=col, alpha=0.9, zorder=2, linewidth=0)
ax.set_xlim(0, k_K)
ax.set_xticks([0, k_K / 2, k_K])
ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K_{\rm 0}$"])
ax.get_xticklabels()[0].set_horizontalalignment("left")
ax.set_xlabel(r"Momentum $k$ ($\mathrm{\mathring{A}}^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_ylim(energies.min(), energies.max())
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figS6_surface_orbital_bands.{ext}", dpi=300)
plt.close(fig)
print("wrote", OUT / "figS6_surface_orbital_bands.{pdf,png}")
