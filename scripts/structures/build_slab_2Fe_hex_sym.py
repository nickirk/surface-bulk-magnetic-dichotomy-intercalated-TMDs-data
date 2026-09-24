"""
Build a SYMMETRIC 2-Fe-layer Fe1/3NbS2 slab in the HEX √3×√3 primitive cell.

Layer stacking (bottom → top):
    S – Nb – S – Fe(↓) – S – Nb – S – Fe(↑) – S – Nb – S

This is one full c-period of bulk Fe1/3NbS2 PLUS one extra NbS2 sandwich on top,
so the slab is inversion-symmetric about the middle Nb layer.  The combined
operation [C₂⊥||i] (inversion about midplane × spin flip) is an exact slab
symmetry, putting this geometry in the i-wave (SD-SAM) class of the Lange et
al. (arXiv 2602.08773) surface-altermagnet classification.

Expected behaviour: A↑(k) ≡ A↓(k) on the unfolded Fermi surface, i.e.
η_z(k) = 0 to numerical precision.  This is the control that isolates whether
the η_z signal in the asymmetric 2-Fe-on-top slab is genuine altermagnetism
(it isn't, per the symmetric-slab vs asymmetric-slab comparison) or simply
environmental-asymmetry-driven ferrimagnetism.

Atoms: 2 Fe + 9 Nb + 18 S = 29 atoms total.

The hex √3×√3 in-plane cell is identical to the asymmetric slab.  c is grown
to accommodate the extra NbS2 sandwich + vacuum.
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


def write_poscar_frac(path, L, species, coords_frac, comment, selective_dynamics=False):
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
    c_bulk = Lh[2, 2]                           # ≈ 12.154 Å (one A-AFM period)
    cart_bulk = fh @ Lh

    print(f"Bulk hex cell:  a={np.linalg.norm(Lh[0]):.4f}  "
          f"c={c_bulk:.4f}  γ={np.degrees(np.arccos(np.dot(Lh[0],Lh[1])/(np.linalg.norm(Lh[0])*np.linalg.norm(Lh[1])))):.2f}°")
    print(f"Bulk atoms: Fe={sph.count('Fe')} Nb={sph.count('Nb')} S={sph.count('S')}\n")

    # ── Step 1: replicate one bulk c-period and identify the "layers" ──
    # Sort all bulk atoms by z, group by similar z, and identify the structure.
    z_sort = np.argsort(cart_bulk[:, 2])
    sp_sorted = [sph[i] for i in z_sort]
    z_sorted = cart_bulk[z_sort, 2]
    xy_sorted = cart_bulk[z_sort, :2]

    print("Bulk c-period layers (bottom → top):")
    layers = []     # list of (species, z, count, list of (x,y))
    for s, z, xy in zip(sp_sorted, z_sorted, xy_sorted):
        if not layers or abs(z - layers[-1][1]) > 0.05 or s != layers[-1][0]:
            layers.append([s, z, 1, [xy]])
        else:
            layers[-1][2] += 1
            layers[-1][3].append(xy)
    for s, z, n, xys in layers:
        print(f"   z = {z:7.4f} Å    {n}× {s}")

    # Per the README, the c-period contains: Nb(z=0)-S-Fe-S-Nb-S-Fe-S-[Nb(z=c)].
    # We want: bottom S, then NbS2 sandwich, Fe, NbS2 sandwich, Fe, NbS2 sandwich, top S.
    # Equivalently: take one c-period (Nb-S-Fe-S-Nb-S-Fe-S) and append one more
    # NbS2 sandwich (S-Nb-S) on top so that the bottom-S and top-S layers
    # bracket the structure symmetrically.

    # ── Step 2: assemble 1.5 c-periods, with a sandwich on each side ──
    # The atomic structure of one c-period is: layers 1-8 (Nb-S-Fe-S-Nb-S-Fe-S).
    # The slab we want, schematically:
    #   ┌ bottom vacuum
    #   ├ S
    #   ├ Nb
    #   ├ S
    #   ├ Fe↓
    #   ├ S
    #   ├ Nb
    #   ├ S
    #   ├ Fe↑
    #   ├ S
    #   ├ Nb
    #   ├ S
    #   └ top vacuum
    #
    # That's 11 atomic layers.  The bulk has 9 such layers if we wrap (Nb-S-Fe-
    # S-Nb-S-Fe-S-Nb) but only 8 distinct without the wrap.  To build the
    # symmetric slab, we'll take the 8 layers of one c-period and add 3 more
    # layers (S-Nb-S) translated by exactly one c-period.

    # First, find the proper sandwich-bottom S to put at the bottom of the
    # slab.  A "sandwich-bottom S" is an S layer with the Nb of its sandwich
    # *directly above* it (no Fe between them).  In Fe1/3NbS2 bulk these are
    # the S layers at z = 4.511 and z = 10.588 — each has its sandwich Nb just
    # ~1.55 Å above and a Fe further below.
    z_Nb = sorted(set(round(z, 3) for s, z in zip(sp_sorted, z_sorted) if s == "Nb"))
    z_S  = sorted(set(round(z, 3) for s, z in zip(sp_sorted, z_sorted) if s == "S"))
    sandwich_bot_S = []
    for zs in z_S:
        # closest Nb above this S (within c_bulk)
        gaps = [(zn - zs) % c_bulk for zn in z_Nb]
        d_to_Nb_above = min(g for g in gaps if g > 0.5)
        # closest Fe above
        z_Fe = sorted(set(round(z, 3) for s, z in zip(sp_sorted, z_sorted) if s == "Fe"))
        gaps_Fe = [(zf - zs) % c_bulk for zf in z_Fe]
        d_to_Fe_above = min(g for g in gaps_Fe if g > 0.5) if gaps_Fe else 99
        if d_to_Nb_above < d_to_Fe_above:
            sandwich_bot_S.append(zs)
    print(f"\nS layers identified as sandwich-bottoms (Nb directly above, no Fe between): "
          f"{sandwich_bot_S}")
    assert len(sandwich_bot_S) == 2, f"expected 2 sandwich-bottom S layers, got {sandwich_bot_S}"
    z_S_bottom = sandwich_bot_S[0]
    print(f"Shifting origin so the lower sandwich-bottom S (z = {z_S_bottom:.4f} Å) "
          f"goes to z = 0.")

    # Translate all bulk atoms by -z_S_bottom (wrapping modulo c)
    cart_shifted = cart_bulk.copy()
    cart_shifted[:, 2] = (cart_bulk[:, 2] - z_S_bottom) % c_bulk
    f_shifted = cart_shifted @ np.linalg.inv(Lh)
    f_shifted[:, 2] = f_shifted[:, 2] % 1.0
    cart_shifted = f_shifted @ Lh

    # Verify the resulting c-period starts S-Nb-S-Fe-…
    print("\nShifted c-period layers (bottom → top):")
    layers_shift = []
    z_sort2 = np.argsort(cart_shifted[:, 2])
    for i in z_sort2:
        s, z = sph[i], cart_shifted[i, 2]
        if not layers_shift or abs(z - layers_shift[-1][1]) > 0.05 or s != layers_shift[-1][0]:
            layers_shift.append([s, z, 1])
        else:
            layers_shift[-1][2] += 1
    for s, z, n in layers_shift:
        print(f"   z = {z:7.4f} Å    {n}× {s}")
    expected = ["S", "Nb", "S", "Fe", "S", "Nb", "S", "Fe"]
    actual = [layers_shift[k][0] for k in range(min(len(layers_shift), len(expected)))]
    assert actual == expected, (
        f"After origin shift, c-period order is wrong. Expected {expected}, got {actual}.")

    # ── Step 3: append a top S-Nb-S sandwich (3 layers, 9 atoms) translated
    # up by one c-period.  These atoms come from the shifted bulk's lowest
    # S-Nb-S sandwich (z ∈ [0, ~3.5]) — now safely S-Nb-S because we shifted
    # to a sandwich bottom.
    sandwich_idx = np.where(cart_shifted[:, 2] < 3.5)[0]
    sandwich_sp = [sph[i] for i in sandwich_idx]
    sandwich_cart = cart_shifted[sandwich_idx].copy()

    print(f"\nTop sandwich to be appended ({len(sandwich_idx)} atoms):")
    for s, c in zip(sandwich_sp, sandwich_cart):
        print(f"   {s}   ({c[0]:7.4f}, {c[1]:7.4f}, {c[2]:7.4f}) Å")
    species_set = set(sandwich_sp)
    assert species_set == {"S", "Nb"}, (
        f"top sandwich should contain only S+Nb (= S-Nb-S layers), got species {species_set}")

    # Translate the sandwich up by one c-period
    sandwich_top = sandwich_cart.copy()
    sandwich_top[:, 2] += c_bulk

    # Stack: bulk c-period + translated sandwich
    full_sp = list(sph) + sandwich_sp
    full_cart = np.vstack([cart_shifted, sandwich_top])

    # ── Step 4: build slab cell with vacuum, pinning bottom-S at z = bottom_pad ──
    bottom_pad = 1.43       # Å — same as the asymmetric slab
    top_vacuum = 10.0
    z_min_abs = full_cart[:, 2].min()
    z_max_abs = full_cart[:, 2].max()
    full_cart[:, 2] += bottom_pad - z_min_abs
    c_new = (z_max_abs - z_min_abs) + bottom_pad + top_vacuum
    L_slab = Lh.copy()
    L_slab[2, 2] = c_new
    f_slab = full_cart @ np.linalg.inv(L_slab)
    f_slab[:, 2] = f_slab[:, 2] % 1.0

    print(f"\nSlab cell:  {np.linalg.norm(L_slab[0]):.4f} × "
          f"{np.linalg.norm(L_slab[1]):.4f} × {L_slab[2,2]:.4f} Å")
    print(f"Slab z range: [{full_cart[:,2].min():.3f}, {full_cart[:,2].max():.3f}] Å")
    print(f"Vacuum:  top = {c_new - full_cart[:,2].max():.3f} Å,  "
          f"bottom = {full_cart[:,2].min():.3f} Å")

    # ── Step 5: sort atoms in Fe, Nb, S order; identify Fe layers ──
    order = np.lexsort((
        np.round(f_slab[:, 0], 4),
        np.round(f_slab[:, 1], 4),
        np.round(full_cart[:, 2], 3),
        [{"Fe": 0, "Nb": 1, "S": 2}[s] for s in full_sp],
    ))
    sp_sorted = [full_sp[i] for i in order]
    f_sorted = f_slab[order]
    cart_sorted = full_cart[order]

    print("\nSlab stacking (bottom → top), with verification of σ_h symmetry:")
    # iterate atoms in z-order, NOT POSCAR-order, so layer_groups reflects the
    # actual stacking
    z_order = np.argsort(cart_sorted[:, 2])
    layer_groups = []
    for i in z_order:
        s, c = sp_sorted[i], cart_sorted[i]
        z = round(c[2], 3)
        if not layer_groups or abs(z - layer_groups[-1][1]) > 0.05 or s != layer_groups[-1][0]:
            layer_groups.append([s, z, 1, [c[:2]]])
        else:
            layer_groups[-1][2] += 1
            layer_groups[-1][3].append(c[:2])

    # Check inversion-symmetry pairs about midplane
    z_midplane = 0.5 * (cart_sorted[:, 2].min() + cart_sorted[:, 2].max())
    print(f"\n  midplane z = {z_midplane:.4f} Å")
    n_layers = len(layer_groups)
    for k in range(n_layers // 2):
        s_bot, z_bot, n_bot, _ = layer_groups[k]
        s_top, z_top, n_top, _ = layer_groups[-(k+1)]
        z_bot_inv = 2*z_midplane - z_bot
        match = "✓" if (s_bot == s_top and n_bot == n_top and abs(z_bot_inv - z_top) < 0.05) else "✗"
        print(f"    L{k+1:>2}: {n_bot}× {s_bot:<2} z={z_bot:7.4f}  ↔  "
              f"L{n_layers-k:>2}: {n_top}× {s_top:<2} z={z_top:7.4f}   "
              f"(2*mid - z_bot = {z_bot_inv:.4f})  {match}")
    if n_layers % 2 == 1:
        s_m, z_m, n_m, _ = layer_groups[n_layers // 2]
        print(f"    L{n_layers//2 + 1:>2}: {n_m}× {s_m:<2} z={z_m:7.4f}  (midplane atom; should be on z_mid)")

    # ── Step 6: assign A-AFM MAGMOM with bottom Fe ↓, top Fe ↑ ──
    fe_idx = [i for i, s in enumerate(sp_sorted) if s == "Fe"]
    z_fe = sorted(set(round(cart_sorted[i, 2], 2) for i in fe_idx))
    z_bot_fe = z_fe[0]
    magmom_fe = []
    print("\nFe atoms in POSCAR order:")
    for k, i in enumerate(fe_idx):
        z = round(cart_sorted[i, 2], 2)
        m = -4 if z == z_bot_fe else +4
        magmom_fe.append(m)
        c = cart_sorted[i]
        print(f"  Fe{k+1}  ({c[0]:7.4f}, {c[1]:7.4f}, {c[2]:7.4f}) Å   MAGMOM = {m:+d}")

    n_Nb = sp_sorted.count("Nb")
    n_S = sp_sorted.count("S")
    n_Fe = sp_sorted.count("Fe")
    magmom_str = " ".join(f"{m:+d}" for m in magmom_fe) + f"   {n_Nb}*0.0   {n_S}*0.0"
    print(f"\nMAGMOM string:  {magmom_str}")
    print(f"Total atoms: {len(sp_sorted)}  (Fe={n_Fe}, Nb={n_Nb}, S={n_S})")

    # ── Step 7: write POSCAR + summary ──
    out = WORK_DIR / "inputs" / "POSCAR"
    write_poscar_frac(out, L_slab, sp_sorted, f_sorted,
                      "Fe1/3NbS2 2-Fe SYMMETRIC slab (S-Nb-S-Fe-S-Nb-S-Fe-S-Nb-S), "
                      "hex √3×√3 cell, all-A-AFM, unrelaxed",
                      selective_dynamics=True)
    print(f"\nWrote {out}")

    summary = {
        "n_atoms": len(sp_sorted),
        "n_Fe": n_Fe, "n_Nb": n_Nb, "n_S": n_S,
        "lattice_AA": L_slab.tolist(),
        "slab_thickness_AA": float(cart_sorted[:, 2].max() - cart_sorted[:, 2].min()),
        "bottom_pad_AA": bottom_pad,
        "top_vacuum_AA": top_vacuum,
        "midplane_z_AA": z_midplane,
        "layer_groups": [{"species": s, "z": z, "count": n} for s, z, n, _ in layer_groups],
        "magmom_Fe": magmom_fe,
        "magmom_string": magmom_str,
        "expected_symmetry": "[C2⊥||i] about midplane forces A↑(k) = A↓(k); η_z(k) = 0.",
        "comment": "Symmetric slab control for slab_2Fe_aAFM_u03_hex_wannier_fs.",
    }
    with open(WORK_DIR / "inputs" / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {WORK_DIR / 'inputs' / 'build_summary.json'}")


if __name__ == "__main__":
    main()
