"""Fig. S6: unfold the relaxed A-AFM slab bands onto the primitive NbS2 cell along Gamma -> K0 and
project them onto the top layer.

Run in the band-calculation directory (inputs/figS6/slab_aafm_bands: WAVECAR, PROCAR, KPOINTS) with
VaspBandUnfolding (github.com/QijingZheng/VaspBandUnfolding) importable. Writes
slab_aafm_unfolded_sw.npy (spin, k, band, [E, weight]) and slab_aafm_toplayer_orbital_weights.npz
(spin, k, band) as in data/figS6/.
"""
import numpy as np
from procar import procar
from unfold import unfold, make_kpath, find_K_from_k

As = np.array([[5.7588999999999997, 0, 0], [0, 9.9747073957084069, 0], [0, 0, 34.3078]])
Au = np.array([[2.8794499999999998, 1.6624512326180678, 0],
               [-2.8794499999999994, 1.6624512326180678, 0], [0, 0, 34.3078]])
M = np.round(np.linalg.solve(Au.T, As.T).T).astype(int)
kpath = make_kpath([[0, 0, 0], [1. / 3, 1. / 3, 0]], nseg=60)
K_in_sup = [find_K_from_k(k, M)[0] for k in kpath]
# PROCAR rows follow the KPOINTS order, so map each path point to its KPOINTS row by value, modulo
# supercell reciprocal lattice vectors ((0, +0.5, 0) and (0, -0.5, 0) are the same point).
kp = np.loadtxt("KPOINTS", skiprows=3)[:, :3]


def same_k(a, b):
    d = a - b
    return np.abs(d - np.round(d)).max(axis=-1)


kmap = []
for K in K_in_sup:
    d = same_k(kp, K)
    assert d.min() < 1e-6, ("path point not in KPOINTS", K)
    kmap.append(int(d.argmin()))

# Top layer, 1-based VASP indices of POSCAR -> 0-based.
fe = [i - 1 for i in (6, 8)]                       # surface Fe layer, z = 23.96 A
s = [i - 1 for i in (63, 64, 65, 75, 76, 77)]      # top S layer, z = 22.88 A
nb = [i - 1 for i in (21, 22, 25, 27, 28, 31)]     # top Nb layer, z = 21.27 A
# PROCAR orbital order: s py pz px dxy dyz dz2 dxz x2-y2
p = procar()
np.savez("slab_aafm_toplayer_orbital_weights.npz",
         fe_d=p.get_pw(atoms=fe, spd=[6, 5, 7])[:, kmap, :],     # d_z2, d_yz, d_xz
         s_pz=p.get_pw(atoms=s, spd=[2])[:, kmap, :],
         nb_d=p.get_pw(atoms=nb, spd=[6, 4, 8])[:, kmap, :])     # d_z2, d_xy, d_x2-y2

sw = unfold(M=M, wavecar="WAVECAR").spectral_weight(kpath)
np.save("slab_aafm_unfolded_sw.npy", sw)
print("M", M.tolist(), "sw shape", sw.shape)
