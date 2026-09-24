"""
Wannier-based unfolding for Fe_{1/3}NbS2 a-AFM SURFACE slab onto the primitive
hexagonal NbS2 slab BZ.

Popescu–Zunger spectral weight:
    P_n(k) = (1/N_sc) * Σ_β | Σ_{α ∈ β} c_α^n(K) exp(i 2π k_frac_prim · r_α) |^2

where:
    - K is the SC BZ image of the primitive-BZ k-point: K_sc = M @ k_prim (mod 1)
    - β groups WFs with the same (primitive-atom type, orbital-within-atom)
    - r_α is the primitive-lattice integer vector of the SC atom hosting WF α
    - N_sc = det(M)

This follows the fcdmft/utils/unfold.py approach:
    - Compute fractional primitive coords of every SC atom:  f_prim = f_sc @ M
    - Identify primitive-atom types by (species, f_prim mod 1) within tolerance
    - r_α = floor(f_prim) of the host SC atom

Supercell ↔ primitive transformation for the Fe surface slab:
    AS_CELL rows = [(5.7589, 0, 0), (0, 9.9747, 0), (0, 0, 34.308)]
    AU_CELL rows = [(2.879, 1.662, 0), (-2.879, 1.662, 0), (0, 0, 34.308)]
    M = [[1, -1, 0], [3, 3, 0], [0, 0, 1]]   →   det(M) = 6
    (AS_CELL = M @ AU_CELL:  a1_sc = a1_prim - a2_prim,  a2_sc = 3 a1_prim + 3 a2_prim)

Slab geometry: we unfold only in-plane (kz fixed). The c-axis transformation
is trivial (M_zz = 1) so kz is unchanged between SC and primitive BZs.
"""
from __future__ import annotations

import re
import os
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

# ── Project constants ───────────────────────────────────────────────────────
WORK_ROOT = Path(".")
HR_DIR = Path(os.environ.get("W90_DIR", "."))   # drivers override these module constants
WIN_FILE = HR_DIR / "wannier90.1.win"
WOUT_FILES = {1: HR_DIR / "wannier90.1.wout", 2: HR_DIR / "wannier90.2.wout"}
HR_FILES = {1: HR_DIR / "wannier90.1_hr.dat", 2: HR_DIR / "wannier90.2_hr.dat"}

# Fermi level from the GS calculation (not NSCF).  User directive: use GS E_F.
EFERMI = 4.7785  # eV

# Supercell → primitive integer transformation (row convention: A_sc = M @ A_prim)
# 2D in-plane transformation has det = 6 (rectangular SC from hexagonal primitive)
M_MATRIX = np.array([[1, -1, 0], [3, 3, 0], [0, 0, 1]], dtype=int)
N_SC = int(round(np.linalg.det(M_MATRIX)))   # == 6


# ── Structure / Wannier I/O ─────────────────────────────────────────────────
@dataclass
class Structure:
    """Row convention throughout:  A[i, :] = ith lattice vector (in Å)."""
    A_sc: np.ndarray
    A_prim: np.ndarray
    species: list
    pos_cart: np.ndarray
    pos_sc_frac: np.ndarray
    pos_prim_frac: np.ndarray


def parse_win(win_file: Path) -> Structure:
    """Parse unit_cell_cart and atoms_cart blocks from a Wannier90 .win file."""
    lines = [ln.strip() for ln in win_file.read_text().splitlines()]

    def block(begin_tag: str, end_tag: str) -> list[str]:
        i0 = next(i for i, ln in enumerate(lines) if ln.lower() == begin_tag)
        i1 = next(i for i, ln in enumerate(lines) if ln.lower() == end_tag and i > i0)
        return [ln for ln in lines[i0 + 1:i1] if ln and ln.lower() != "ang"]

    uc = block("begin unit_cell_cart", "end unit_cell_cart")
    A_sc = np.array([[float(x) for x in ln.split()] for ln in uc[:3]])

    at = block("begin atoms_cart", "end atoms_cart")
    species, positions = [], []
    for ln in at:
        parts = ln.split()
        species.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pos_cart = np.array(positions)

    # A_prim rows:  A_sc = M @ A_prim  →  A_prim = M^{-1} @ A_sc
    A_prim = np.linalg.solve(M_MATRIX.astype(float), A_sc)
    pos_sc_frac = pos_cart @ np.linalg.inv(A_sc)
    pos_prim_frac = pos_cart @ np.linalg.inv(A_prim)
    assert np.allclose(pos_prim_frac, pos_sc_frac @ M_MATRIX, atol=1e-6)

    return Structure(A_sc=A_sc, A_prim=A_prim, species=species,
                     pos_cart=pos_cart, pos_sc_frac=pos_sc_frac,
                     pos_prim_frac=pos_prim_frac)


def parse_wf_centres(wout_file: Path, n_wann: int) -> np.ndarray:
    txt = wout_file.read_text().splitlines()
    i0 = next(i for i, ln in enumerate(txt) if "Final State" in ln)
    pat = re.compile(
        r"WF centre and spread\s+(\d+)\s*\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)")
    centres = np.zeros((n_wann, 3))
    for ln in txt[i0:]:
        m = pat.search(ln)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        centres[idx] = [float(m.group(2)), float(m.group(3)), float(m.group(4))]
        if idx == n_wann - 1:
            break
    return centres


def parse_hr_dat(hr_file: Path):
    """Read Wannier90 _hr.dat. Returns (R_vecs, wR, H_R, n_wann)."""
    with open(hr_file) as f:
        f.readline()
        n_wann = int(f.readline().split()[0])
        n_R = int(f.readline().split()[0])
        wR = []
        while len(wR) < n_R:
            wR.extend(int(x) for x in f.readline().split())
        wR = np.array(wR[:n_R], dtype=int)

        R_set: dict[tuple[int, int, int], int] = {}
        H_R = np.zeros((n_R, n_wann, n_wann), dtype=complex)
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            Rx, Ry, Rz = int(parts[0]), int(parts[1]), int(parts[2])
            i = int(parts[3]) - 1
            j = int(parts[4]) - 1
            H_R[R_set.setdefault((Rx, Ry, Rz), len(R_set)), i, j] = \
                float(parts[5]) + 1j * float(parts[6])

    R_vecs = np.zeros((len(R_set), 3), dtype=int)
    for k, v in R_set.items():
        R_vecs[v] = k
    return R_vecs, wR, H_R, n_wann


# ── WF → (primitive atom, orbital) assignment ────────────────────────────────
@dataclass
class BetaAssignment:
    beta_id: np.ndarray
    r_prim_lat: np.ndarray
    beta_labels: list
    n_copies_per_beta: np.ndarray


def _identify_prim_atom_types(structure: Structure, tol: float = 1.5e-2):
    species = structure.species
    f_prim = structure.pos_prim_frac

    r_prim_lat = np.floor(f_prim + 1e-6).astype(int)
    frac_within = f_prim - r_prim_lat

    prim_types: list[tuple[str, tuple]] = []
    prim_type_of_atom = -np.ones(len(species), dtype=int)

    for ia, (sp, fw) in enumerate(zip(species, frac_within)):
        match = -1
        for tid, (sp2, fw2) in enumerate(prim_types):
            if sp2 != sp:
                continue
            delta = fw - np.array(fw2)
            # wrap only in-plane dims (z is slab-direction, no lattice periodicity
            # meaningful within slab thickness)
            delta[:2] -= np.round(delta[:2])
            if np.linalg.norm(delta) < tol:
                match = tid
                break
        if match < 0:
            prim_types.append((sp, tuple(fw.tolist())))
            match = len(prim_types) - 1
        prim_type_of_atom[ia] = match

    return prim_type_of_atom, prim_types, r_prim_lat


def assign_wannier_to_prim(centres: np.ndarray, structure: Structure) -> BetaAssignment:
    n_wann = len(centres)

    # Step 1: find nearest SC atom (search ±2 in-plane, ±0 z images — slab)
    sc_atom_of_wf = np.full(n_wann, -1, dtype=int)
    for iw in range(n_wann):
        c = centres[iw]
        best_d = np.inf
        best_a = -1
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                shift = n1 * structure.A_sc[0] + n2 * structure.A_sc[1]
                d = np.linalg.norm(structure.pos_cart + shift - c, axis=1)
                idx = int(np.argmin(d))
                if d[idx] < best_d:
                    best_d = float(d[idx])
                    best_a = idx
        sc_atom_of_wf[iw] = best_a

    # Step 2: orbital-within-atom index (sort WFs on each SC atom by WF index)
    orb_idx = np.zeros(n_wann, dtype=int)
    wfs_per_atom = defaultdict(list)
    for iw in range(n_wann):
        wfs_per_atom[sc_atom_of_wf[iw]].append(iw)
    for ia, wfs in wfs_per_atom.items():
        for k, iw in enumerate(sorted(wfs)):
            orb_idx[iw] = k

    # Step 3: prim atom types & r_prim_lat per SC atom
    prim_type_of_atom, prim_types, r_prim_lat_atom = _identify_prim_atom_types(structure)

    # Step 4: β id = unique (prim_type, orb_idx)
    prim_type_of_wf = prim_type_of_atom[sc_atom_of_wf]
    pair = np.stack([prim_type_of_wf, orb_idx], axis=1)
    _, beta_id = np.unique(pair, axis=0, return_inverse=True)
    r_prim_lat_wf = r_prim_lat_atom[sc_atom_of_wf]

    beta_labels: list = []
    unique_pairs, _ = np.unique(pair, axis=0, return_index=True)
    for (pt, oi) in unique_pairs:
        sp, fw = prim_types[pt]
        beta_labels.append((sp, oi, fw))

    n_copies = np.bincount(beta_id)
    return BetaAssignment(beta_id=beta_id, r_prim_lat=r_prim_lat_wf,
                          beta_labels=beta_labels, n_copies_per_beta=n_copies)


# ── H(K) interpolation ───────────────────────────────────────────────────────
def interpolate_H_batch(K_frac_sc: np.ndarray, R_vecs: np.ndarray, wR: np.ndarray,
                        H_R: np.ndarray) -> np.ndarray:
    phases = np.exp(2j * np.pi * K_frac_sc @ R_vecs.T)
    weights = phases / wR[None, :]
    return np.einsum("kr,rij->kij", weights, H_R, optimize=True)


# ── Parallel Popescu–Zunger ─────────────────────────────────────────────────
def _compute_chunk(k_chunk, H_R, R_vecs, wR, r_prim_lat, beta_mat,
                   M_mat, inv_N_sc, orbital_mask=None):
    K_cart = k_chunk @ M_mat.T
    K_frac_sc = K_cart - np.round(K_cart)
    phase_wf = np.exp(2j * np.pi * k_chunk @ r_prim_lat.T.astype(float))
    phases = np.exp(2j * np.pi * K_frac_sc @ R_vecs.T)
    weights = phases / wR[None, :]
    H_K = np.einsum("kr,rij->kij", weights, H_R, optimize=True)
    w, U = np.linalg.eigh(H_K)
    c_weighted = U * phase_wf[:, :, None]
    proj = np.matmul(beta_mat, c_weighted)
    P = (proj.real ** 2 + proj.imag ** 2).sum(axis=1) * inv_N_sc
    if orbital_mask is not None:
        # |c_α^n|² summed over α in the subset → (B, n_wann)
        U_sel = U[:, orbital_mask, :]
        P_proj = (U_sel.real ** 2 + U_sel.imag ** 2).sum(axis=1)
        return w, P, P_proj
    return w, P, None


def compute_spectral_weight(k_frac_prim: np.ndarray,
                            H_R: np.ndarray, R_vecs: np.ndarray, wR: np.ndarray,
                            beta: BetaAssignment,
                            batch: int = 256,
                            n_workers: int | None = None,
                            orbital_mask: np.ndarray | None = None):
    """Parallel Popescu–Zunger across k-point chunks.

    If ``orbital_mask`` (bool array of length n_wann) is given, also returns the
    orbital-projection weight P_proj[k, n] = Σ_{α ∈ mask} |c_α^n|² alongside
    the unfolded weight.
    """
    from concurrent.futures import ProcessPoolExecutor

    n_k = len(k_frac_prim)
    n_wann = H_R.shape[1]
    n_beta = int(beta.beta_id.max()) + 1
    beta_mat = np.zeros((n_beta, n_wann), dtype=np.float64)
    beta_mat[beta.beta_id, np.arange(n_wann)] = 1.0
    inv_N_sc = 1.0 / N_SC
    want_proj = orbital_mask is not None

    chunks = [k_frac_prim[i:i + batch] for i in range(0, n_k, batch)]
    n_chunks = len(chunks)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_workers = min(n_workers, n_chunks)

    E = np.zeros((n_k, n_wann))
    P = np.zeros((n_k, n_wann))
    Pp = np.zeros((n_k, n_wann)) if want_proj else None

    def _collect(fut, i0, n):
        w, Pc, Ppc = fut.result() if hasattr(fut, "result") else fut
        E[i0:i0 + n] = w
        P[i0:i0 + n] = Pc
        if want_proj:
            Pp[i0:i0 + n] = Ppc

    if n_workers <= 1:
        for ic, chunk in enumerate(chunks):
            i0 = ic * batch
            res = _compute_chunk(chunk, H_R, R_vecs, wR, beta.r_prim_lat,
                                 beta_mat, M_MATRIX, inv_N_sc, orbital_mask)
            _collect(res, i0, len(chunk))
        return (E, P, Pp) if want_proj else (E, P)

    worker_env = {k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    old = {k: os.environ.get(k) for k in worker_env}
    os.environ.update(worker_env)
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_compute_chunk, chunk, H_R, R_vecs, wR,
                                 beta.r_prim_lat, beta_mat, M_MATRIX, inv_N_sc,
                                 orbital_mask)
                       for chunk in chunks]
            for ic, fut in enumerate(futures):
                _collect(fut, ic * batch, len(chunks[ic]))
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return (E, P, Pp) if want_proj else (E, P)


def build_orbital_mask_by_z(structure, centres, z_min=16.0, z_max=None):
    """
    Build a bool mask over Wannier functions selecting those whose SC-atom host
    has z-coordinate in [z_min, z_max).  Uses the same nearest-atom assignment
    as assign_wannier_to_prim.

    Returns (mask (n_wann,), atom_indices_selected, wf_count_per_species)
    """
    from collections import Counter
    n_wann = len(centres)
    mask = np.zeros(n_wann, dtype=bool)
    species_count = Counter()
    # Same nearest-SC-atom search as in assign_wannier_to_prim
    for iw in range(n_wann):
        c = centres[iw]
        best_d = np.inf
        best_a = -1
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                shift = n1 * structure.A_sc[0] + n2 * structure.A_sc[1]
                d = np.linalg.norm(structure.pos_cart + shift - c, axis=1)
                idx = int(np.argmin(d))
                if d[idx] < best_d:
                    best_d = float(d[idx]); best_a = idx
        z = structure.pos_cart[best_a, 2]
        inside = (z >= z_min) and (z_max is None or z < z_max)
        if inside:
            mask[iw] = True
            species_count[structure.species[best_a]] += 1
    return mask, species_count


# ── Self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    struct = parse_win(WIN_FILE)
    print(f"Parsed {len(struct.species)} SC atoms from {WIN_FILE.name}")
    print("A_sc (rows = SC lattice vectors, Å):")
    print(struct.A_sc)
    print("A_prim (derived from M^-1):")
    print(struct.A_prim)
    print(f"N_sc = det(M) = {N_SC}")

    pt, ptypes, r_prim = _identify_prim_atom_types(struct)
    print(f"\n{len(ptypes)} primitive atom types detected:")
    by_species = defaultdict(list)
    for tid, (sp, fw) in enumerate(ptypes):
        n = int(np.sum(pt == tid))
        by_species[sp].append((tid, n, fw))
    for sp, infos in by_species.items():
        print(f"  {sp} : {len(infos)} distinct prim positions "
              f"(total {sum(n for _, n, _ in infos)} atoms)")

    R_vecs, wR, H_R, n_wann = parse_hr_dat(HR_FILES[1])
    print(f"\nSpin-1 hr.dat:  n_wann = {n_wann}   n_R = {len(R_vecs)}")

    centres = parse_wf_centres(WOUT_FILES[1], n_wann)
    beta = assign_wannier_to_prim(centres, struct)
    print(f"\nβ assignment:  {len(beta.beta_labels)} groups across {n_wann} WFs")
    sizes = np.bincount(beta.beta_id)
    print(f"  β-group sizes: unique = {sorted(set(sizes.tolist()))}  "
          f"(min {sizes.min()}, max {sizes.max()})")

    # sum rule at Γ should give n_wann / N_sc
    E, P = compute_spectral_weight(np.zeros((1, 3)), H_R, R_vecs, wR, beta, n_workers=1)
    print(f"  Γ sum rule: Σ_β Σ_n P = {P.sum():.3f}   expect {n_wann}/{N_SC} = {n_wann/N_SC:.3f}")
