# Data for Surface–Bulk Magnetic Dichotomy in Intercalated Transition Metal Dichalcogenides

Data, plotting scripts and VASP inputs for the figures of the manuscript and its Supplemental
Material. `data/` and `scripts/` are organised by figure, in the order the figures appear. The
figures themselves are not included; they are published with the article.

    data/<figure>/       the numbers each figure is drawn from
    scripts/<figure>/    the scripts that draw them
    scripts/unfolding/   plane-wave and Wannier-orbital unfolding used to produce the band, Fermi-surface and spin-contrast data
    scripts/structures/  builders for the slab structures used in Fig. 4 and Fig. S3
    inputs/<figure>/     VASP inputs (INCAR, KPOINTS, POSCAR; wannier90.win) for every calculation behind a figure
    docs/                conventions.md (signs, normalisation, energies); unfolding.md (methods)

POTCAR files are not included (VASP licence). Use the PAW potentials named in each POSCAR.

## Reproducing the figures

Run from the repository root with Python 3, numpy, scipy and matplotlib. Output goes to
`output/<figure>/`. Each script draws the data panels of its figure; structure images and the
final arrangement of panels were composed separately.

| Figure | Panel | Command | Data |
|---|---|---|---|
| Fig. 1 | (b) | `python3 scripts/fig1/plot_fig1b_phase_energies.py --from-csv data/fig1/phase_energies.csv --out-dir output/fig1` | `phase_energies.csv` |
| | (c) | `python3 scripts/fig1/plot_fig1c_arpes_overlay.py` | ARPES cut `arpes_FNS030n2_88combine.txt`; unfolded bands `aafm_unfolded_sw.npy`, `stripe_unfolded_sw.npy` |
| Fig. 2 | (b) | `python3 scripts/fig2/plot_fig2b_u_scan.py` | `u_scan_e0.dat` (tables: `bulk_useries.tsv`, `slab_useries.tsv`, `bulk_u03_reseed.tsv`) |
| Fig. 3 | (a–d) | first build the experimental maps (below), then `python3 scripts/fig3/plot_fig3_fermi_surfaces.py` | calculated maps `*_fermi_surface.npz`; cells `*_POSCAR_PRIM/SUPER`. Surface energetics: `cobalt_*.tsv` |
| Fig. 4 | (a) | `python3 scripts/fig4/plot_fig4a_bands.py data/fig4/panel_a_bulk_M0GK0_kz5_16_EIGENVAL output/fig4/fig4a_bands.pdf` | raw VASP EIGENVAL |
| | (b), (c) | `python3 scripts/fig4/plot_fig4bc_eta_maps.py` | spin-contrast maps `eta_bulk_kz5_16_nk401.npz`, `eta_slab_asym_nk401.npz` |
| Fig. S2 | (b) | `python3 scripts/figS2/plot_figS2b_J_fit.py` | `J_fit_results.npz`, produced by `scripts/figS2/fit_J_heisenberg.py` from the magnetic configurations in `inputs/figS2/heisenberg_fit/` |
| Fig. S3 | | `python3 scripts/figS3/plot_figS3_symmetric_slab.py` | `eta_symmetric_slab_nk401.npz` |
| Fig. S4 | (a), (b), (c) | `python3 scripts/figS4/plot_figS4_slab_bands.py {sym_M0GK0,asym_M0GK0,sym_M0GP} data/figS4/<EIGENVAL> output/figS4/<name>.pdf` | raw VASP EIGENVAL of the slab band runs: `slab_sym_M0GK0_EIGENVAL`, `slab_asym_M0GK0_EIGENVAL`, `slab_sym_M0GP_EIGENVAL` (inputs in `inputs/figS4/`) |
| Fig. S5 | (a), (b) | `python3 scripts/figS5/plot_figS5_hypersensitivity.py` | `data/fig1/aafm_unfolded_sw.npy` (as-received structure), `data/figS5/aafm_symmetrized_unfolded_sw.npy` |
| Fig. S6 | | `python3 scripts/figS6/plot_figS6_surface_orbital_bands.py` | ARPES cut as in Fig. 1(c); `slab_aafm_unfolded_sw.npy`, `slab_aafm_toplayer_orbital_weights.npz` |

Every script above was run on this repository as released and compared pixel by pixel with the
renders of the scripts that produced the manuscript figures: 0 differing pixels in every panel, except
Fig. S2(b). Its script plots the same fitted values as the published panel with a different layout.

## How the derived data were produced

- **Band structures, Fig. 1(c), Fig. S5 and Fig. S6:** the `*_sw.npy` files hold the plane-wave unfolded
  spectral weight of the supercell wavefunctions (VaspBandUnfolding; Popescu & Zunger,
  Phys. Rev. B 85, 085201 (2012)), made by the drivers in `scripts/unfolding/planewave/` from the
  band calculations in `inputs/fig1/c_*`, `inputs/figS5/` and `inputs/figS6/`. Each array is (spin, k-point, band,
  [energy, weight]). For Fig. S6 the weights are multiplied by PROCAR projections onto the top-layer
  Fe d_z2, d_xz, d_yz; S p_z; and Nb d_z2, d_xy, d_x2-y2 orbitals (same driver). Its ground state is
  `inputs/fig2/slab_U0.3_aafm`.
- **Fermi surfaces, Fig. 3, and spin-contrast maps, Fig. 4(b,c) and Fig. S3:** Wannier-orbital
  unfolding (Ku, Berlijn & Lee, Phys. Rev. Lett. 104, 216401 (2010)). The Wannier Hamiltonians are too
  large to include; `docs/unfolding.md` gives the workflow, the supercell matrices, the Fermi levels
  and the checks, and `scripts/unfolding/wannier/` holds the code. The maps in `data/` are the output
  of that code; `scripts/fig4/plot_fig4bc_eta_maps.py --from-caches` rebuilds the Fig. 4 maps from
  the unfolded spectra.
- **Slab band structures, Fig. S4:** raw spin-polarized eigenvalues from non-self-consistent runs on the
  ground-state charge densities of the Fig. 4(c) and Fig. S3 slabs (`inputs/figS4/`); spin channel 1 is plotted
  as up. P = (5/12, 1/6, 0) in primitive reciprocal coordinates, the midpoint of the M0-K0 zone edge.
- **Energies, Figs. 1(b) and 2(b):** `E0` from the final OSZICAR line of each calculation; see
  `docs/conventions.md` for signs and normalisation.

## Notes

- **Experimental maps for Fig. 3 are not redistributed.** Build them from the published sources:

      # Co1/3NbS2: Figure6.xlsx from the dataset of Popcevic et al., Phys. Rev. B 105, 155114 (2022)
      POPCEVIC_FIG6_XLSX=<path>/Figure6.xlsx python3 scripts/fig3/experiment/replot_popcevic_fig6_from_xlsx.py
      # Co1/3TaS2: an image of Fig. 1 of Sas et al. (cited in the Fig. 3 caption)
      SAS_FIG1_PNG=<path>/fig1.png python3 scripts/fig3/experiment/extract_sas_fig1_data.py
      SAS_FIG1_PNG=<path>/fig1.png python3 scripts/fig3/experiment/interpolate_sas_fig1_masks.py

  They write to `output/fig3_experiment/`, where the Fig. 3 script reads them. The Sas extraction
  expects the figure image used for the manuscript (pixel bounds are recorded in the script).
- The Fig. 2(b) slab energies use one fixed slab geometry for all U. See `docs/conventions.md`.
