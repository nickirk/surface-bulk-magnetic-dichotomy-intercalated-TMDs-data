# Unfolding: methods, workflow and checks

Two unfolding methods are used.

| Figure | Quantity | Method |
|---|---|---|
| Fig. 1(c), Fig. S2 | band structures on the ARPES cut | plane-wave unfolding of the supercell wavefunctions (VaspBandUnfolding; Popescu–Zunger weight) |
| Fig. 3 | Fermi surfaces of Co₁/₃NbS₂ and Co₁/₃TaS₂ | Wannier-orbital unfolding |
| Fig. 4(b,c), Fig. S1 | spin contrast η_z at E_F | Wannier-orbital unfolding |
| Fig. 4(a) | bulk bands on the primitive path | none: raw VASP eigenvalues |

## Plane-wave unfolding (Fig. 1(c), Fig. S2)

Drivers in `scripts/unfolding/planewave/`, run in the band-calculation directory next to its WAVECAR,
with VaspBandUnfolding (github.com/QijingZheng/VaspBandUnfolding) importable. Each builds the k-path,
writes the supercell KPOINTS used by the band run, and saves `sw.npy` (spin, k, band, [E, weight]).

| Data | Driver | Band run | Ground-state E_F used in the figure |
|---|---|---|---|
| `data/fig1/aafm_unfolded_sw.npy` | `fig1c_aafm_unfold_calc.py` | `inputs/fig1/c_aafm_bands` (SCF: `c_aafm_ground_state`) | 7.730112 eV |
| `data/fig1/stripe_unfolded_sw.npy` | `fig1c_stripe_unfold_calc.py` | `inputs/fig1/c_stripe_bands` (SCF: `c_stripe_ground_state`) | 8.66907 eV |
| `data/figS2/aafm_symmetrized_unfolded_sw.npy` | `figS2_symmetrized_unfold_calc.py` | `inputs/figS2/symmetrized_bands` (SCF: `symmetrized_ground_state`) | 7.7670 eV |

The `eref` inside each driver only sets its own preview plot; the figures use the Fermi levels above.

## Wannier-orbital unfolding

1. **Ground-state SCF.** Every map uses the Fermi level of this self-consistent run. A
   non-self-consistent `ICHARG = 11` run recomputes E_F on its own mesh, which here differs by about
   28 meV and visibly changes a Fermi surface.
2. **Wannier preparation.** NSCF with `LWANNIER90 = .TRUE.` and `ISYM = 0`, from the SCF charge
   density. Inputs: `inputs/fig4/b_bulk_aAFM_wannier/`, `inputs/fig4/c_slab_asym_Fe_terminated_hex/`,
   `inputs/figS1/slab_symmetric_hex/`.
3. **wannier90.** Maximal localisation with disentanglement; windows are set relative to E_F. One
   Hamiltonian per spin, `wannier90.{1,2}_hr.dat`.
4. **Validation.** Interpolated bands reproduce the DFT eigenvalues inside the frozen window
   (RMS ≈ 5×10⁻⁶ eV, no state off by more than 1 meV). States are matched by nearest neighbour,
   not by index.
5. **Unfolding** (`scripts/unfolding/wannier/wannier_unfolding.py`). Diagonalise H(K) built from
   `_hr.dat` and weight each eigenstate by

   P_n(k) = (1/N_sc) Σ_β | Σ_{α∈β} c_α^n(K) · exp(i 2π k·r_α) |²

   where K = M·k mod 1, α runs over the supercell Wannier functions, β groups Wannier functions of
   the same primitive-atom type and orbital, r_α is the primitive-lattice translation of the atom
   hosting α, and N_sc = det M.
6. **Broadening at E_F.** Gaussian in energy. σ = 0.05 eV for Fig. 3, spins summed. σ = 5 meV for
   the η_z maps: η_z = A↑ − A↓ for the asymmetric slab; A↑(k) − A↑(−k) for the bulk; A↑(k) − A↑(g·k)
   for the symmetric slab, with g the C₂ about the in-plane axis at the slab centre.

### Drivers

| Data | Driver | Run as |
|---|---|---|
| `data/fig3/*_aafm_fermi_surface.npz` | `fig3_aafm/plot_fermi_surface.py` | `W90_DIR=<dir> W90_EFERMI=<E_F> python3 plot_fermi_surface.py --shift 0 --sigma 0.05 --nk 181 --krange 1.35` |
| `data/fig3/*_stripe_fermi_surface.npz` | `fig3_stripe/fs_worker320.py` (Slurm: `run_fs320.sh`) writes (E, P) on the nk = 181 grid; the map is the σ = 0.05 eV Gaussian of the spin-summed weight at E_F, as for A-AFM | 320-band Hamiltonians |
| `data/fig4/eta_bulk_kz5_16_nk401.npz` | `fig4_bulk/unfold_bulk_sym.py` | `W90_DIR=<dir> KZ_FRAC=0.3125 NK=401` |
| `data/fig4/eta_slab_asym_nk401.npz` | `fig4_slab_asym/eta_and_sharp_fs.py` | `W90_DIR=<dir> NK=401` |
| `data/figS1/eta_symmetric_slab_nk401.npz` | `figS1_slab_sym/eta_and_sharp_fs.py`, then `rebuild_eta_full.py` | `W90_DIR=<dir> NK=401` |

The Fig. 4 bulk and slab caches are turned into the stored η_z maps by
`scripts/fig4/plot_fig4bc_eta_maps.py --from-caches BULK_CACHE SLAB_CACHE`.

### Supercell matrices (rows: A_sc = M · A_prim)

| Cell | M | det M |
|---|---|---|
| A-AFM √3×√3 bulk (Fig. 3) | [[2,−1,0],[−1,2,0],[0,0,1]] | 3 |
| stripe, orthorhombic (Fig. 3) | [[1,1,0],[−3,3,0],[0,0,1]] | 6 |
| hexagonal bulk and slabs (Fig. 4, Fig. S1) | [[1,−1,0],[1,2,0],[0,0,1]] | 3 |

To verify an M, derive A_prim = M⁻¹·A_sc and check M·A_prim = A_sc and that A_prim is a 1×1 cell
(|a1| = |a2|, γ = 60° or 120°). After unfolding, the mean weight equals 1/det M.

### Fermi levels (ground-state SCF, eV)

| Calculation | E_F |
|---|---|
| A-AFM Co₁/₃NbS₂ | 7.83264 |
| A-AFM Co₁/₃TaS₂ | 6.3915 |
| stripe Co₁/₃NbS₂ | 7.8760 |
| stripe Co₁/₃TaS₂ | 6.4456 |
| bulk A-AFM Fe₁/₃NbS₂ (Fig. 4) | 7.7670 |
| asymmetric slab (Fig. 4(c)) | 2.7011 |
| symmetric slab (Fig. S1) | 3.5207 |

### Reproduction checks

- Rerunning the Fig. 3 A-AFM Co₁/₃NbS₂ driver reproduces the stored map to 1.4×10⁻¹² (peak 17.2).
- The Fig. S1 map at 201×201 reproduces an independent earlier calculation to 3×10⁻¹¹. The 401 grid
  contains the 201 grid, and the two agree to the same precision at all 40,401 shared points.
- The Fig. 4 bulk 401 map is identical to a 201 calculation at the shared points; the slab agrees to
  1.5×10⁻¹⁴.
