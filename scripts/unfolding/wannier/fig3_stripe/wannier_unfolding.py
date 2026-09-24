"""
Wannier-based unfolding for Co_{1/3}NbS2 a-AFM onto the primitive NbS2 BZ.

Popescu–Zunger spectral weight:
    P_n(k) = (1/N_sc) * Σ_β | Σ_{α ∈ β} c_α^n(K) exp(i 2π k_frac_prim · r_α) |^2

where:
    - K is the SC BZ image of the primitive-BZ k-point: K = M @ k_frac_prim (mod 1)
    - α runs over the Wannier functions in the supercell
    - β groups WFs with the same (primitive-atom type, orbital index within atom)
    - r_α is the primitive-lattice integer vector of the SC atom hosting WF α
    - N_sc = det(M)

This follows the approach in fcdmft/utils/unfold.py:
    - Compute fractional primitive coords of every SC atom:  f_prim = f_sc @ M
    - Identify primitive-atom types by (species, f_prim mod 1) within tolerance
    - r_α = floor(f_prim) of the host SC atom
    - Atoms at the same f_prim-mod-1 are copies of the same primitive atom at
      different primitive lattice translations (includes Co here).

Supercell:  √3×√3 (in-plane) of the 2H-NbS2 primitive cell (c unchanged) with
    M = [[2,-1,0], [-1,2,0], [0,0,1]]   → det M = 3.
2H-NbS2 primitive cell has c = c_sc = 12.134 Å, containing 2 Nb layers
(z_frac = 0 and 0.5, trigonal prismatic stacking).
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import os
import numpy as np

sys.path.append(str(Path(__file__).parent / "shim"))   # find_K_from_k from VaspBandUnfolding
from unfold import find_K_from_k  # noqa: E402

# ── Project constants ───────────────────────────────────────────────────────
WORK_ROOT = Path(os.environ.get("W90_WORK_ROOT", "."))
HR_DIR = Path(os.environ["W90_DIR"])   # per-material Wannier directory
WIN_FILE = HR_DIR / "wannier90.1.win"
WOUT_FILES = {1: HR_DIR / "wannier90.1.wout", 2: HR_DIR / "wannier90.2.wout"}
HR_FILES = {1: HR_DIR / "wannier90.1_hr.dat", 2: HR_DIR / "wannier90.2_hr.dat"}

EFERMI = float(os.environ["W90_EFERMI"])   # ground-state SCF E_F, set by fs_worker320.py (Co1/3NbS2 7.8760, Co1/3TaS2 6.4456)

# Supercell → primitive integer transformation (row-convention A_sc = M @ A_prim)
M_MATRIX = np.array([[1, 1, 0], [-3, 3, 0], [0, 0, 1]], dtype=int)   # stripe (det 6), verified: inv(M)@A_sc gives |a1|=|a2|=a0, gamma=60.000
N_SC = int(round(np.linalg.det(M_MATRIX)))      # 3


# ── Structure / Wannier I/O ─────────────────────────────────────────────────
@dataclass
class Structure:
    """Row convention throughout:  A[i, :] = ith lattice vector (in Å)."""
    A_sc: np.ndarray            # (3,3)
    A_prim: np.ndarray          # (3,3)
    species: list               # len = n_sc_atoms
    pos_cart: np.ndarray        # (n_sc_atoms, 3) Å
    pos_sc_frac: np.ndarray     # (n_sc_atoms, 3) in SC fractional
    pos_prim_frac: np.ndarray   # (n_sc_atoms, 3) in PRIM fractional (may exceed [0,1))


def parse_win(win_file: Path) -> Structure:
    """Parse unit_cell_cart and atoms_cart blocks from a Wannier90 .win file."""
    lines = [ln.strip() for ln in win_file.read_text().splitlines()]

    def block(begin_tag: str, end_tag: str) -> list[str]:
        i0 = next(i for i, ln in enumerate(lines) if ln.lower() == begin_tag)
        i1 = next(i for i, ln in enumerate(lines) if ln.lower() == end_tag and i > i0)
        return [ln for ln in lines[i0 + 1:i1] if ln and ln.lower() != "ang"]

    uc = block("begin unit_cell_cart", "end unit_cell_cart")
    A_sc = np.array([[float(x) for x in ln.split()] for ln in uc[:3]])   # rows

    at = block("begin atoms_cart", "end atoms_cart")
    species, positions = [], []
    for ln in at:
        parts = ln.split()
        species.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pos_cart = np.array(positions)

    # Derive primitive lattice rows:  A_sc = M @ A_prim    →   A_prim = M^{-1} @ A_sc
    A_prim = np.linalg.solve(M_MATRIX.astype(float), A_sc)

    # fractional coords — rows of A are lattice vectors, so
    #    pos_cart = frac @ A    →    frac = pos_cart @ A^{-1}
    pos_sc_frac = pos_cart @ np.linalg.inv(A_sc)
    pos_prim_frac = pos_cart @ np.linalg.inv(A_prim)

    # sanity check: pos_prim_frac should equal pos_sc_frac @ M_MATRIX
    assert np.allclose(pos_prim_frac, pos_sc_frac @ M_MATRIX, atol=1e-6)

    return Structure(A_sc=A_sc, A_prim=A_prim, species=species,
                     pos_cart=pos_cart, pos_sc_frac=pos_sc_frac,
                     pos_prim_frac=pos_prim_frac)


def parse_wf_centres(wout_file: Path, n_wann: int) -> np.ndarray:
    """Parse the 'Final State' block; return (n_wann, 3) Cartesian centres."""
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
        f.readline()                          # comment
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
            hre = float(parts[5])
            him = float(parts[6])
            key = (Rx, Ry, Rz)
            if key not in R_set:
                R_set[key] = len(R_set)
            ri = R_set[key]
            H_R[ri, i, j] = hre + 1j * him

    R_vecs = np.zeros((len(R_set), 3), dtype=int)
    for k, v in R_set.items():
        R_vecs[v] = k
    return R_vecs, wR, H_R, n_wann


# ── WF → (primitive atom, orbital) assignment ────────────────────────────────
@dataclass
class BetaAssignment:
    beta_id: np.ndarray          # (n_wann,) β group index
    r_prim_lat: np.ndarray       # (n_wann, 3) primitive lattice translation of host SC atom
    beta_labels: list            # list of (species, orb_idx, prim_frac_within) per β
    n_copies_per_beta: np.ndarray  # (n_beta,) number of WFs in each β (should be same per prim atom)


def _identify_prim_atom_types(structure: Structure, tol: float = 1.5e-2):
    """
    Group SC atoms by (species, fractional prim coord mod 1) → prim atom type.

    Returns
    -------
    prim_type_of_atom : (n_sc_atoms,) int
    prim_types        : list of (species, frac_within_prim_tuple)
    r_prim_lat        : (n_sc_atoms, 3) integer primitive translations = floor(pos_prim_frac)
    """
    species = structure.species
    f_prim = structure.pos_prim_frac

    # Integer primitive translation of each SC atom
    r_prim_lat = np.floor(f_prim + 1e-6).astype(int)
    frac_within = f_prim - r_prim_lat            # in [0, 1)

    prim_types: list[tuple[str, tuple]] = []
    prim_type_of_atom = -np.ones(len(species), dtype=int)

    for ia, (sp, fw) in enumerate(zip(species, frac_within)):
        match = -1
        for tid, (sp2, fw2) in enumerate(prim_types):
            if sp2 != sp:
                continue
            delta = fw - np.array(fw2)
            delta -= np.round(delta)
            if np.linalg.norm(delta) < tol:
                match = tid
                break
        if match < 0:
            prim_types.append((sp, tuple(fw.tolist())))
            match = len(prim_types) - 1
        prim_type_of_atom[ia] = match

    return prim_type_of_atom, prim_types, r_prim_lat


def assign_wannier_to_prim(centres: np.ndarray, structure: Structure) -> BetaAssignment:
    """
    Map each Wannier function to a β group (primitive-atom type, orbital-within-atom).

    Returns a BetaAssignment. Each β contributes a single term to the
    Popescu–Zunger amplitude per SC copy of that primitive atom; WFs within the
    same β sum coherently with phase exp(i 2π k · r_α).
    """
    n_wann = len(centres)

    # Step 1: find nearest SC atom for each WF (scan ±2 SC images for WFs that wandered)
    sc_atom_of_wf = np.full(n_wann, -1, dtype=int)
    for iw in range(n_wann):
        c = centres[iw]
        best_d = np.inf
        best_a = -1
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                for n3 in range(-1, 2):
                    shift = n1 * structure.A_sc[0] + n2 * structure.A_sc[1] + n3 * structure.A_sc[2]
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

    # Step 3: identify prim atom type for each SC atom
    prim_type_of_atom, prim_types, r_prim_lat_atom = _identify_prim_atom_types(structure)

    # Step 4: β id per WF  =  unique (prim_type, orb_idx)
    prim_type_of_wf = prim_type_of_atom[sc_atom_of_wf]
    pair = np.stack([prim_type_of_wf, orb_idx], axis=1)
    _, beta_id = np.unique(pair, axis=0, return_inverse=True)

    # Translate r_prim_lat per WF
    r_prim_lat_wf = r_prim_lat_atom[sc_atom_of_wf]

    # Build β labels for reporting
    beta_labels: list = []
    unique_pairs, first_idx = np.unique(pair, axis=0, return_index=True)
    for (pt, oi) in unique_pairs:
        sp, fw = prim_types[pt]
        beta_labels.append((sp, oi, fw))

    n_copies = np.bincount(beta_id)
    return BetaAssignment(beta_id=beta_id, r_prim_lat=r_prim_lat_wf,
                          beta_labels=beta_labels, n_copies_per_beta=n_copies)


# ── H(K) interpolation ───────────────────────────────────────────────────────
def interpolate_H_batch(K_frac_sc: np.ndarray, R_vecs: np.ndarray, wR: np.ndarray,
                        H_R: np.ndarray) -> np.ndarray:
    """
    H(K) = Σ_R (1/w_R) H(R) exp(i 2π K · R)
    K_frac_sc : (nk, 3) in SC fractional coords.
    Returns H of shape (nk, n_wann, n_wann).
    """
    phases = np.exp(2j * np.pi * K_frac_sc @ R_vecs.T)     # (nk, nR)
    weights = phases / wR[None, :]                         # (nk, nR)
    H_K = np.einsum("kr,rij->kij", weights, H_R)
    return H_K


# ── Spectral weight computation ──────────────────────────────────────────────
# Serial worker: processes a chunk of k-points end-to-end so we can farm chunks
# out to separate processes (LAPACK's eigh on 76×76 matrices is single-threaded,
# so the only way to saturate cores is to run multiple eigh's in parallel).
def _compute_chunk(k_chunk: np.ndarray,
                   H_R: np.ndarray, R_vecs: np.ndarray, wR: np.ndarray,
                   r_prim_lat: np.ndarray, beta_mat: np.ndarray,
                   M_mat: np.ndarray, inv_N_sc: float):
    # fold k → K_sc
    K_cart = k_chunk @ M_mat.T
    K_frac_sc = K_cart - np.round(K_cart)
    # phases
    phase_wf = np.exp(2j * np.pi * k_chunk @ r_prim_lat.T.astype(float))
    # H(K) — (B, n, n)
    phases = np.exp(2j * np.pi * K_frac_sc @ R_vecs.T)
    weights = phases / wR[None, :]
    H_K = np.einsum("kr,rij->kij", weights, H_R, optimize=True)
    # eigendecomposition — still serial per matrix (LAPACK limit at this N)
    w, U = np.linalg.eigh(H_K)
    # weighted coefficients and β reduction
    c_weighted = U * phase_wf[:, :, None]
    proj = np.matmul(beta_mat, c_weighted)
    P = (proj.real ** 2 + proj.imag ** 2).sum(axis=1) * inv_N_sc
    return w, P


def compute_spectral_weight(k_frac_prim: np.ndarray,
                            H_R: np.ndarray, R_vecs: np.ndarray, wR: np.ndarray,
                            beta: BetaAssignment,
                            batch: int = 512,
                            n_workers: int | None = None):
    """
    Popescu-Zunger spectral weight, parallelised across k-point chunks.

    For each k: fold K = M @ k (mod 1), interpolate H(K), diagonalise, compute
        P_n(k) = (1/N_sc) Σ_β | Σ_{α ∈ β} c_α^n · exp(i 2π k · r_α) |²

    Uses multiprocessing because `np.linalg.eigh` on 76×76 matrices is
    single-threaded inside LAPACK — BLAS threading does not help at this size,
    so we farm out chunks of k-points to worker processes instead.

    Parameters
    ----------
    batch : int
        Number of k-points per worker chunk (trade-off between overhead and
        memory). Default 512 works well for ~200 MB H_R tensors.
    n_workers : int or None
        Number of worker processes (default: os.cpu_count()-1, clipped to the
        number of chunks). Set to 1 or 0 to disable multiprocessing.

    Returns
    -------
    E : (nk, n_wann) float, eigenvalues [eV]
    P : (nk, n_wann) float, spectral weights per band
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    n_k = len(k_frac_prim)
    n_wann = H_R.shape[1]

    # β reduction matrix  β_mat[β, α] = 1 if α ∈ β else 0
    n_beta = int(beta.beta_id.max()) + 1
    beta_mat = np.zeros((n_beta, n_wann), dtype=np.float64)
    beta_mat[beta.beta_id, np.arange(n_wann)] = 1.0
    inv_N_sc = 1.0 / N_SC

    # Build chunks
    chunks = [k_frac_prim[i:i + batch] for i in range(0, n_k, batch)]
    n_chunks = len(chunks)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)
    n_workers = min(n_workers, n_chunks)

    if n_workers <= 1:
        # Serial path (useful for debugging; also avoids spawn overhead on small jobs)
        E_all = np.zeros((n_k, n_wann))
        P_all = np.zeros((n_k, n_wann))
        for ic, chunk in enumerate(chunks):
            i0 = ic * batch
            w, P = _compute_chunk(chunk, H_R, R_vecs, wR, beta.r_prim_lat,
                                  beta_mat, M_MATRIX, inv_N_sc)
            E_all[i0:i0 + len(chunk)] = w
            P_all[i0:i0 + len(chunk)] = P
        return E_all, P_all

    # Parallel path: cap BLAS threads in each worker to 1 to avoid over-subscription
    worker_env = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    # Apply to current env so worker processes inherit (ProcessPoolExecutor forks)
    old_env = {k: os.environ.get(k) for k in worker_env}
    os.environ.update(worker_env)
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(_compute_chunk, chunk, H_R, R_vecs, wR,
                          beta.r_prim_lat, beta_mat, M_MATRIX, inv_N_sc)
                for chunk in chunks
            ]
            E_all = np.zeros((n_k, n_wann))
            P_all = np.zeros((n_k, n_wann))
            for ic, fut in enumerate(futures):
                i0 = ic * batch
                w, P = fut.result()
                E_all[i0:i0 + len(chunks[ic])] = w
                P_all[i0:i0 + len(chunks[ic])] = P
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return E_all, P_all


# ── Self-test / diagnostic ──────────────────────────────────────────────────
if __name__ == "__main__":
    struct = parse_win(WIN_FILE)
    print(f"Parsed {len(struct.species)} SC atoms from {WIN_FILE.name}")
    print("A_sc (rows = SC lattice vectors, Å):")
    print(struct.A_sc)
    print("A_prim (rows = primitive lattice vectors, Å):")
    print(struct.A_prim)
    print(f"N_sc = det(M) = {N_SC}")

    print("\nPer-atom prim-fractional coords and r_prim_lat:")
    pt, ptypes, r_prim = _identify_prim_atom_types(struct)
    for ia, sp in enumerate(struct.species):
        fw = struct.pos_prim_frac[ia] - r_prim[ia]
        print(f"  {ia:2d} {sp:<3s}  f_prim = {struct.pos_prim_frac[ia].round(4)!s:<30s}"
              f"  r_prim_lat = {r_prim[ia]}   type_id = {pt[ia]}")
    print(f"\n{len(ptypes)} primitive atom types detected:")
    for tid, (sp, fw) in enumerate(ptypes):
        n = int(np.sum(pt == tid))
        print(f"  type {tid}:  {sp:<3s}  frac_within = {np.array(fw).round(4)}   copies = {n}")

    R_vecs, wR, H_R, n_wann = parse_hr_dat(HR_FILES[1])
    print(f"\nSpin-1 hr.dat:  n_wann = {n_wann}   n_R = {len(R_vecs)}")

    centres = parse_wf_centres(WOUT_FILES[1], n_wann)
    beta = assign_wannier_to_prim(centres, struct)
    print(f"\nβ assignment:  {len(beta.beta_labels)} groups across {n_wann} WFs")
    sizes = np.bincount(beta.beta_id)
    print(f"  β-group sizes: {sorted(set(sizes.tolist()))}  (min {sizes.min()}, max {sizes.max()})")
    print("  β-group summary:")
    for bi, (sp, oi, fw) in enumerate(beta.beta_labels):
        print(f"    β{bi:2d}:  {sp:<3s}  orb = {oi}  frac_within = {np.array(fw).round(4)}  n = {sizes[bi]}")
