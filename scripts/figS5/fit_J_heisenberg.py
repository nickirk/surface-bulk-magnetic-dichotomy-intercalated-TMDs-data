"""Heisenberg-J fit pipeline for Fe1/3 NbS2.

STEP 1 – validate the coefficient matrix:
    fit J's to Weber's published energies (PRB 103 214439, Table II) and
    confirm we recover Wu Table III (= Weber Table III).

STEP 2 – fit J's to our own Weber-settings DFT:
    same matrix, our energies → our J's.

Model (Wu eq. 4 / Weber eq. 1):
    H = E0 + S² Σ_α J_α Σ_<ij>∈α s_i s_j
α ∈ {J1, J2, J1', J2', J3'};  unprimed = intra-layer, primed = inter-layer.
S = 2 (Fe²⁺).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
# Weber Table II (meV/Fe relative to lowest phase), re-referenced to FM = 0:
#   U=0.3 eV:  ground-state = a-stripe, FM=+35.7   → ΔE_phase = T2_phase - 35.7
#   U=0.9 eV:  ground-state = a-zigzag, FM=+19.7   → ΔE_phase = T2_phase - 19.7
WEBER = {
    "U=0.3": {  # ΔE relative to FM (meV/Fe), from Weber Table II
        "a_fm":     27.8 - 35.7,    # -7.9
        "a_stripe":  0.0 - 35.7,    # -35.7
        "f_stripe": 15.2 - 35.7,    # -20.5
        "a_zigzag":  0.9 - 35.7,    # -34.8
        "f_zigzag": 14.1 - 35.7,    # -21.6
    },
    "U=0.9": {
        "a_fm":     22.5 - 19.7,    # +2.8
        "a_stripe":  2.5 - 19.7,    # -17.2
        "f_stripe": 12.0 - 19.7,    # -7.7
        "a_zigzag":  0.0 - 19.7,    # -19.7
        "f_zigzag": 10.1 - 19.7,    # -9.6
    },
}

# Wu Table III (= Weber Table III), J in meV.
# Column order: (J1, J2, J1', J2', J3')  -- i.e., the cumulative order used in
# Wu's panel-(a) cumulative fit (intra-NN → intra-2NN → inter-NN → inter-2NN → inter-3NN)
WU_J = {
    "U=0.3": np.array([+0.76, -0.006, +0.49, -0.20, -0.07]),
    "U=0.9": np.array([+0.57, -0.14,  +0.28, -0.16, -0.09]),
}

# Our DFT (Weber settings: ISMEAR=-5, PREC=Accurate, U=0.3; inputs in inputs/figS5/heisenberg_fit/)
# ΔE per Fe vs FM (meV); a_zigzag and f_zigzag from last DAV/CGA step of un-terminated
# but well-converged (dE < 0.2 meV/cell) runs.
OUR_DFT = {
    "a_fm":     -8.195,
    "a_stripe": -57.155,
    "f_stripe": -44.838,
    "a_zigzag": -59.388,
    "f_zigzag": -53.905,
}

PHASES   = ["a_fm", "a_stripe", "f_stripe", "a_zigzag", "f_zigzag"]
J_NAMES  = ["J1", "J2", "J1'", "J2'", "J3'"]
S_EFF    = 2.0
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "inputs" / "figS5" / "heisenberg_fit"

# Reference distances for shell classification (Å).
# Computed from a=5.76, c=12.14 hex primitive, P6_322, Fe at 2c:
#   J1   intra 1NN = a               = 5.76
#   J2   intra 2NN = a*sqrt(3)       = 9.98
#   J1'  inter 1NN = sqrt(a^2/3 + (c/2)^2)             ≈ 6.92
#   J2'  inter 2NN = sqrt(a^2 + (c/2)^2 - ...)         ≈ 9.00
#   J3'  inter 3NN ≈ 10.69
J_DIST  = [5.76, 9.98, 6.92, 9.00, 10.69]
J_INTRA = [True, True, False, False, False]

# ---------------------------------------------------------------------------
# POSCAR + INCAR parsing
# ---------------------------------------------------------------------------
def parse_poscar(path: Path):
    lines = path.read_text().splitlines()
    scale = float(lines[1])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    species = lines[5].split()
    counts  = [int(x) for x in lines[6].split()]
    direct  = lines[7].strip().lower().startswith("d")
    pos = []
    for j in range(8, 8 + sum(counts)):
        pos.append([float(x) for x in lines[j].split()[:3]])
    pos = np.array(pos)
    if direct:
        pos = pos @ lat
    sp_list = []
    for sp, n in zip(species, counts):
        sp_list += [sp] * n
    return lat, np.array(sp_list), pos

def parse_magmom(incar_path: Path) -> np.ndarray:
    for line in incar_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("#"): continue
        if not s.upper().startswith("MAGMOM"): continue
        rhs = s.split("=", 1)[1].split("#")[0].split("!")[0]
        out = []
        for tok in rhs.replace(",", " ").split():
            if "*" in tok:
                k, v = tok.split("*"); out += [float(v)] * int(k)
            else:
                out.append(float(tok))
        return np.array(out)
    raise ValueError(f"No MAGMOM in {incar_path}")

def fe_pos_and_spin(phase: str):
    lat, sp, pos = parse_poscar(DATA_DIR / phase / "POSCAR")
    mm = parse_magmom(DATA_DIR / phase / "INCAR")
    if mm.size == pos.shape[0]:
        per_atom = mm
    elif mm.size == 3 * pos.shape[0]:
        per_atom = mm.reshape(-1, 3)[:, 2]
    else:
        raise ValueError(f"{phase}: MAGMOM size {mm.size} vs {pos.shape[0]} atoms")
    fe_mask = sp == "Fe"
    return lat, pos[fe_mask], np.sign(per_atom[fe_mask]).astype(int)

# ---------------------------------------------------------------------------
# Per-phase coefficient (uses ±2 in-plane cell replication so no NN folds back)
# ---------------------------------------------------------------------------
def coeff_phase(phase: str, verbose: bool = False) -> np.ndarray:
    lat, fe_pos, spin = fe_pos_and_spin(phase)
    n_c = len(fe_pos)
    inv_lat = np.linalg.inv(lat)
    frac_z = (fe_pos @ inv_lat)[:, 2] % 1.0
    layer = (frac_z > 0.5).astype(int)

    # Build a 5×5×3 replicated cell
    rep_pos, rep_spin, rep_layer = [], [], []
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            for iz in range(-1, 2):
                shift = ix*lat[0] + iy*lat[1] + iz*lat[2]
                for k in range(n_c):
                    rep_pos.append(fe_pos[k] + shift)
                    rep_spin.append(spin[k])
                    rep_layer.append(layer[k])
    rep_pos   = np.array(rep_pos)
    rep_spin  = np.array(rep_spin)
    rep_layer = np.array(rep_layer)

    coeff = np.zeros(5)
    counts = np.zeros(5, dtype=int)
    for i in range(n_c):
        d = np.linalg.norm(rep_pos - fe_pos[i], axis=1)
        same_layer = rep_layer == layer[i]
        for k in range(5):
            in_shell = (np.abs(d - J_DIST[k]) < 0.15) & (d > 0.1)
            mask = in_shell & (same_layer if J_INTRA[k] else ~same_layer)
            coeff[k] += spin[i] * np.sum(rep_spin[mask])
            counts[k] += np.sum(mask)
    # Per-Fe in Wu/Weber convention: directed-pair sum / N_Fe  (NO /2).
    # Each bond is counted from both endpoints — same convention Wu uses
    # in Eq. 2 to derive E_stripe - E_zigzag = 4 J_2c S^2 - 4 J_2 S^2 - 8 J_3c S^2.
    coeff = coeff / n_c
    if verbose:
        print(f"  {phase}: bonds/Fe = {counts/n_c}, raw Σss/N = {coeff}")
    return S_EFF**2 * coeff

# ---------------------------------------------------------------------------
# Build A matrix
# ---------------------------------------------------------------------------
print("Computing FM-reference and per-phase coefficients...")
A_fm = coeff_phase("fm", verbose=True)
print(f"  expected for FM (all s_i s_j=+1): z_α*S²/2 where z = bonds/Fe")
print(f"  → for triangular intra: z=6, J1 J2 → 12, 12")
print(f"  → for bilayer inter:    z_J1'=6, z_J2'=?, z_J3'=?")
print()

A = np.zeros((len(PHASES), 5))
for i, ph in enumerate(PHASES):
    A[i] = coeff_phase(ph, verbose=True) - A_fm

print("\nCoefficient matrix A  (ΔE/Fe = A · J):")
print("              " + "  ".join(f"{n:>7}" for n in J_NAMES))
for ph, row in zip(PHASES, A):
    print(f"  {ph:>9}: " + "  ".join(f"{v:+7.3f}" for v in row))

# Wu/Weber stripe-zigzag formula: E_a_stripe - E_a_zigzag = -16 J2 + 16 J2' - 32 J3'
diff = A[PHASES.index("a_stripe")] - A[PHASES.index("a_zigzag")]
print(f"\nE(a_stripe) − E(a_zigzag) per-Fe coefficients on (J1, J2, J1', J2', J3'):")
print(f"  Wu/Weber expected: (0, -16, 0, +16, -32)")
print(f"  Our matrix:        ({diff[0]:+.0f}, {diff[1]:+.0f}, {diff[2]:+.0f}, "
      f"{diff[3]:+.0f}, {diff[4]:+.0f})")

# ---------------------------------------------------------------------------
# STEP 1: Plug Wu's published J's into our matrix; should reproduce Weber Table II
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 1 — sanity check:  A · J_Wu  vs  Weber Table II energies")
print("=" * 72)
for U in ("U=0.3", "U=0.9"):
    pred = A @ WU_J[U]
    web  = np.array([WEBER[U][p] for p in PHASES])
    print(f"\n  {U}  (J_Wu = {WU_J[U]}):")
    print(f"  {'phase':>9}    A·J     Weber    diff")
    for ph, p, w in zip(PHASES, pred, web):
        print(f"  {ph:>9}  {p:+7.2f}  {w:+7.2f}  {p-w:+5.2f}")

# ---------------------------------------------------------------------------
# STEP 1b: fit J from Weber Table II directly; compare to Wu Table III
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 1b — fit J from Weber's energies; compare with Wu Table III")
print("=" * 72)
J_fit_weber = {}
for U in ("U=0.3", "U=0.9"):
    web = np.array([WEBER[U][p] for p in PHASES])
    J_fit, *_ = np.linalg.lstsq(A, web, rcond=None)
    J_fit_weber[U] = J_fit
    print(f"\n  {U}:")
    print(f"  {'J':>4}        fit       Wu      diff")
    for n, jf, jw in zip(J_NAMES, J_fit, WU_J[U]):
        print(f"  {n:>4}   {jf:+8.4f}  {jw:+8.4f}  {jf-jw:+7.4f}")
    pred = A @ J_fit
    rmse = np.sqrt(np.mean((pred - web) ** 2))
    print(f"  fit RMSE vs Weber data = {rmse:.3f} meV/Fe")

# ---------------------------------------------------------------------------
# STEP 2: fit J from our DFT data
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("STEP 2 — fit J from OUR DFT (Weber settings)")
print("=" * 72)
our = np.array([OUR_DFT[p] for p in PHASES])
print(f"  Our ΔE/Fe (meV): {dict(zip(PHASES, our))}")
J_ours, *_ = np.linalg.lstsq(A, our, rcond=None)
print(f"\n  {'J':>4}        fit    (compare to Wu Table III U=0.3)")
for n, jf, jw in zip(J_NAMES, J_ours, WU_J["U=0.3"]):
    print(f"  {n:>4}   {jf:+8.4f}     {jw:+8.4f}")
pred_ours = A @ J_ours
print(f"  fit RMSE vs our data = {np.sqrt(np.mean((pred_ours-our)**2)):.3f} meV/Fe")

# Cumulative J fits  (J^N = lstsq(A[:, :N], data))
def cumulative_fit(data):
    J_cum = np.full((5, 5), np.nan)
    pred_cum = np.full((5, 5), np.nan)
    for N in range(1, 6):
        sol, *_ = np.linalg.lstsq(A[:, :N], data, rcond=None)
        J_cum[N-1, :N] = sol
        pred_cum[N-1, :] = A[:, :N] @ sol
    return J_cum, pred_cum

J_cum_ours, pred_cum_ours = cumulative_fit(our)
J_cum_w03,  pred_cum_w03  = cumulative_fit(np.array([WEBER["U=0.3"][p] for p in PHASES]))

print("\nOur cumulative J fits (meV):")
print("  N      " + "  ".join(f"{n:>8}" for n in J_NAMES))
for N in range(1, 6):
    cells = [f"{J_cum_ours[N-1,k]:+8.3f}" if not np.isnan(J_cum_ours[N-1,k]) else "    -- "
             for k in range(5)]
    print(f"  N={N}:  " + "  ".join(cells))

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
out_dir = REPO / "output" / "figS5"
out_dir.mkdir(parents=True, exist_ok=True)
np.savez(out_dir / "J_fit_results.npz",
         A=A, our_dE=our,
         weber_dE_U03=np.array([WEBER["U=0.3"][p] for p in PHASES]),
         weber_dE_U09=np.array([WEBER["U=0.9"][p] for p in PHASES]),
         J_fit_weber_U03=J_fit_weber["U=0.3"],
         J_fit_weber_U09=J_fit_weber["U=0.9"],
         J_ours=J_ours,
         J_cum_ours=J_cum_ours, pred_cum_ours=pred_cum_ours,
         J_cum_w03=J_cum_w03,   pred_cum_w03=pred_cum_w03,
         phases=np.array(PHASES), j_names=np.array(J_NAMES))
print(f"\nWrote {out_dir / 'J_fit_results.npz'}  (the committed copy is data/figS5/J_fit_results.npz)")
