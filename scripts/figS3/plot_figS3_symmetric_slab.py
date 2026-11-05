#!/usr/bin/env python3
"""SI Fig. S3: spin contrast of the top-bottom symmetric slab (control for Fig. 4(c)).

Reads data/figS3/eta_symmetric_slab_nk401.npz and renders it with exactly the conventions of
Fig. 4(b,c) (scripts/fig4/plot_fig4bc_eta_maps.py): sigma = 5 meV at the slab's own E_F,
bicubic interpolation, primitive-hexagon crop, and this panel's own symmetric colour limit
(99.5th percentile of |eta| inside |k| < 1.3 1/A).

The field is eta(k) = A_up(k) - A_dn(k), the same definition as Fig. 4(c). It is built from the
Wannier Hamiltonians by scripts/unfolding/wannier/figS3_slab_sym/ (the spin-resolved spectra of
eta_and_sharp_fs.py; rebuild_eta_full.py also returns this field as eta_old, together with the
single-channel check A_up(k) - A_up(g.k), g = C2 about the in-plane axis at the slab centre).

    python3 scripts/figS3/plot_figS3_symmetric_slab.py   # from the repository root
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "fig4"))
from plot_fig4bc_eta_maps import load_field, render_colorbar, render_map, robust_vmax  # noqa: E402

OUT = REPO / "output" / "figS3"

kx, ky, eta = load_field(REPO / "data" / "figS3" / "eta_symmetric_slab_nk401.npz")
vmax = robust_vmax(kx, ky, eta)
render_map(OUT / "figS3_eta_map.png", kx, ky, eta, vmax)
render_colorbar(OUT / "figS3_colorbar.png", vmax)
print(f"vmax = {vmax:.6f} (colour bar +/-{round(vmax)})")
