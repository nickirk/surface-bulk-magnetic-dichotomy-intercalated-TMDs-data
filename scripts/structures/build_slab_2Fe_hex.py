"""
Build a 2-Fe-layer Fe1/3NbS2 fe-on-top slab in the HEX √3×√3 primitive cell
(NO in-plane doubling).  Same vertical structure as the rectangular-cell version at
../slab_2Fe_aAFM_u03_wannier_fs but in the natively C₃-symmetric cell.

Primitive (1×1 hex √3×√3 of Fe1/3NbS2) cell parameters:
    a₁ = (5.7589,  0.0,   0.0)  Å
    a₂ = (-2.879, 4.987, 0.0)   Å
    γ(a₁,a₂) = 120°
Atoms: 2 Fe + 6 Nb + 12 S = 20 atoms total.

Layer stacking (after origin shift), bottom → top:
    S - Nb - S - Fe - S - Nb - S - Fe
    same 8-layer pattern as the rectangular slab, but with HALF the atoms
    per layer (3 S, 3 Nb, 1 Fe per layer in hex vs 6, 6, 2 in rect).

A-AFM ordering: bottom Fe ↓, top Fe ↑.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import json

REPO = Path(__file__).resolve().parents[2]
WORK_DIR = Path(".")   # output directory: POSCAR and build_summary.json are written to ./inputs/
BULK = REPO / "inputs" / "fig4" / "b_bulk_aAFM_wannier" / "POSCAR"


def read_poscar(path):
    lines = path.read_text().splitlines()
    L = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * float(lines[1])
    sp_line = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    species = [s for s, n in zip(sp_line, counts) for _ in range(n)]
    i = 7
    if lines[i].strip().lower().startswith("s"):
        i += 1
    coord_mode = lines[i].strip().lower(); i += 1
    n = sum(counts)
    coords = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in range(n)])
    if coord_mode.startswith("c") or coord_mode.startswith("k"):
        coords = coords @ np.linalg.inv(L)
    return L, species, coords - np.floor(coords)


def write_poscar_frac(path, L, species, coords_frac, comment, magmom=None,
                      selective_dynamics=False):
    sp_unique, counts = [], []
    for s in species:
        if not sp_unique or s != sp_unique[-1]:
            sp_unique.append(s); counts.append(0)
        counts[-1] += 1
    with open(path, "w") as f:
        f.write(comment + "\n")
        f.write("1.0\n")
        for i in range(3):
            f.write(f"  {L[i,0]:18.12f} {L[i,1]:18.12f} {L[i,2]:18.12f}\n")
        f.write("  " + "  ".join(sp_unique) + "\n")
        f.write("  " + "  ".join(str(c) for c in counts) + "\n")
        if selective_dynamics:
            f.write("Selective dynamics\n")
        f.write("Direct\n")
        for c in coords_frac:
            line = f"  {c[0]:18.12f} {c[1]:18.12f} {c[2]:18.12f}"
            if selective_dynamics:
                line += "  F  F  F"
            f.write(line + "\n")


def main():
    Lh, sph, fh = read_poscar(BULK)
    print(f"Bulk hex cell:  a={np.linalg.norm(Lh[0]):.4f}  "
          f"b={np.linalg.norm(Lh[1]):.4f}  c={np.linalg.norm(Lh[2]):.4f}  "
          f"γ={np.degrees(np.arccos(np.dot(Lh[0],Lh[1])/(np.linalg.norm(Lh[0])*np.linalg.norm(Lh[1])))):.2f}°")
    print(f"Bulk atoms: Fe={sph.count('Fe')} Nb={sph.count('Nb')} S={sph.count('S')}")

    # ── Step 1: shift origin so the (S layer formerly at bulk z = 10.588 Å)
    # becomes the bottom of the cell.  Same fractional shift as in the
    # rectangular build script.
    c_bulk = Lh[2, 2]                           # 12.154 Å
    z_S_top_bulk = 10.588 / c_bulk              # ≈ 0.8712
    f_shifted = fh.copy()
    f_shifted[:, 2] = (fh[:, 2] - z_S_top_bulk) % 1.0
    print(f"\nOrigin shift Δz = -{z_S_top_bulk:.4f} (frac):  "
          f"z_min = {f_shifted[:,2].min():.4f}, z_max = {f_shifted[:,2].max():.4f}")

    # ── Step 2: add vacuum, keeping Cartesian z fixed and pinning the bottom-
    # most atom at z ≈ 1.43 Å (matches rectangular slab convention).
    cart = f_shifted @ Lh
    z_min_abs = cart[:, 2].min()
    z_max_abs = cart[:, 2].max()
    bottom_pad = 1.43
    top_vacuum = 10.0
    cart[:, 2] += bottom_pad - z_min_abs
    c_new = (z_max_abs - z_min_abs) + bottom_pad + top_vacuum
    L_slab = Lh.copy()
    L_slab[2, 2] = c_new
    f_slab = cart @ np.linalg.inv(L_slab)
    f_slab[:, 2] = f_slab[:, 2] % 1.0

    print(f"\nSlab cell:  {np.linalg.norm(L_slab[0]):.4f} × "
          f"{np.linalg.norm(L_slab[1]):.4f} × {L_slab[2,2]:.4f} Å")
    print(f"Slab z range (Å):  [{cart[:,2].min():.3f}, {cart[:,2].max():.3f}]")
    print(f"Vacuum on top: {c_new - cart[:,2].max():.3f} Å,  "
          f"on bottom: {cart[:,2].min():.3f} Å")

    # ── Step 3: sort atoms (Fe first, then Nb, S) and report layers ──
    order = np.lexsort((np.round(f_slab[:, 0], 4),
                        np.round(f_slab[:, 1], 4),
                        np.round(cart[:, 2], 3),
                        [{"Fe": 0, "Nb": 1, "S": 2}[s] for s in sph]))
    sp_sorted = [sph[i] for i in order]
    f_sorted  = f_slab[order]
    cart_sorted = cart[order]

    print("\nLayer stacking (bottom → top):")
    seen = []
    for s, c in zip(sp_sorted, cart_sorted):
        z = round(c[2], 3)
        if not seen or abs(z - seen[-1][1]) > 0.05:
            seen.append([s, z, 1])
        else:
            seen[-1][2] += 1
    for s, z, n in seen:
        print(f"  z = {z:6.3f} Å    {n}× {s}")

    # ── Step 4: assign A-AFM MAGMOM (bottom Fe ↓, top Fe ↑) ──
    fe_idx = [i for i, s in enumerate(sp_sorted) if s == "Fe"]
    z_fe_unique = sorted(set(round(cart_sorted[i, 2], 2) for i in fe_idx))
    z_bot = z_fe_unique[0]
    magmom_fe = []
    print("\nFe atoms in POSCAR order:")
    for k, i in enumerate(fe_idx):
        z = round(cart_sorted[i, 2], 2)
        m = -4 if z == z_bot else +4
        magmom_fe.append(m)
        c = cart_sorted[i]
        print(f"  Fe{k+1}  ({c[0]:7.4f}, {c[1]:7.4f}, {c[2]:7.4f}) Å   MAGMOM = {m:+d}")

    n_Nb = sp_sorted.count("Nb")
    n_S  = sp_sorted.count("S")
    magmom_str = " ".join(f"{m:+d}" for m in magmom_fe) + f"   {n_Nb}*0.0   {n_S}*0.0"
    print(f"\nMAGMOM string:  {magmom_str}")

    # ── Step 5: write POSCAR + JSON summary ──
    out = WORK_DIR / "inputs" / "POSCAR"
    write_poscar_frac(out, L_slab, sp_sorted, f_sorted,
                      "Fe1/3NbS2 2-Fe-layer fe-on-top slab, all-A-AFM, hex √3×√3 cell, unrelaxed",
                      selective_dynamics=True)
    print(f"\nWrote {out}")

    summary = {
        "n_atoms": len(sp_sorted),
        "n_Fe": sp_sorted.count("Fe"),
        "n_Nb": n_Nb,
        "n_S":  n_S,
        "lattice_AA": L_slab.tolist(),
        "slab_thickness_AA": float(cart_sorted[:, 2].max() - cart_sorted[:, 2].min()),
        "bottom_pad_AA": bottom_pad,
        "top_vacuum_AA": top_vacuum,
        "z_layers_AA": [{"species": s, "z": z, "count": n} for s, z, n in seen],
        "magmom_Fe": magmom_fe,
        "magmom_string": magmom_str,
        "comment": "Hex √3×√3 cell of Fe1/3NbS2; primitive cell for the magnetic structure. "
                   "Companion to ../slab_2Fe_aAFM_u03_wannier_fs/ which uses the rectangular 1×2 cell."
    }
    with open(WORK_DIR / "inputs" / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
