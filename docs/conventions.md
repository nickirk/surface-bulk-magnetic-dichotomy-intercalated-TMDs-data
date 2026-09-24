# Conventions

## Sign
Tables in `data/` use **ΔE = E(stripe) − E(A-AFM)**: negative means the stripe phase is lower,
positive means A-AFM is lower. **The manuscript plots the opposite sign, E(A-AFM) − E(stripe)**, as
does `scripts/fig2/plot_fig2b_u_scan.py`.

## Normalisation
The bulk cell contains **4 Fe** and the slab **8 Fe**, so raw ΔE in meV is not comparable between
them; compare the per-Fe (per-Co) columns.

## Which energy
`E0` is the `E0` column of the final OSZICAR line, the energy extrapolated to zero smearing. VASP
also prints `TOTEN` and `E_no_entropy`; across these calculations the three differ by up to ~41 meV,
so a single energy should always state which one it is.

## Convergence
- Static calculations (`NSW = 0`): converged when the OUTCAR contains `aborting loop because EDIFF is reached`.
- Relaxations (`NSW > 0`): converged when the OUTCAR contains `reached required accuracy`. The EDIFF
  line is printed at every ionic step and says nothing about whether the relaxation finished.

## Slab geometry
The Fig. 2(b) slab U-series is a **fixed-geometry scan**: every U uses one relaxed slab geometry
(Gaussian σ = 0.05 eV). Slabs relaxed separately at each U can give a different sign (at U = 0.9 eV,
−3.45 meV against +11.47 meV here), so a statement about which phase the slab favours must say which
geometry it refers to.
