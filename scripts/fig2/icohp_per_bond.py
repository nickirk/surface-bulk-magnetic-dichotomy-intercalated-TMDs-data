"""Fig. 2(d) and Table S9: -ICOHP per Fe-S and Fe-Nb bond from LOBSTER.

Reads data/fig2/icohp/<system>/ICOHPLIST.lobster (LOBSTER input: inputs/fig2/icohp/lobsterin) for the
40-atom bulk cell and the unrelaxed and relaxed 80-atom Fe-terminated slabs, all A-AFM at U = 0.3 eV.
For each Fe site, every Fe-S and Fe-Nb pair listed by LOBSTER is one bond; -ICOHP per bond is
-(spin 1 + spin 2), averaged over the bonds of the selected Fe (bulk: Fe1, Fe2; slab: the two
surface Fe, Fe6 and Fe8). The d_z2-d_z2 column is the Fe 3d_z2 - Nb 4d_z2 orbital-pair part of the
Fe-Nb value (z along c).

    python3 scripts/fig2/icohp_per_bond.py      # from the repository root

Writes data/fig2/fig2d_icohp_per_bond.tsv (the committed copy is the one Fig. 2(d) is drawn from).
"""
import re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "fig2" / "icohp"
SYSTEMS = [("bulk", "bulk_aafm", (1, 2)), ("surface_unrelaxed", "unrelaxed_aafm", (6, 8)),
           ("surface_relaxed", "relaxed_aafm", (6, 8))]
LAB = re.compile(r"^([A-Z][a-z]?)(\d+)(?:_(.+))?$")


def read_icohp(p):
    out = []
    for line in Path(p).read_text().splitlines()[2:]:
        f = line.split()
        if len(f) < 8 or not f[0].isdigit():
            continue
        ma, mb = LAB.match(f[1]), LAB.match(f[2])
        out.append(dict(idx=int(f[0]), ea=ma[1], a=int(ma[2]), oa=ma[3], eb=mb[1], b=int(mb[2]), ob=mb[3],
                        d=float(f[3]), up=float(f[7]), dn=float(f[8]) if len(f) > 8 else 0.0))
    return out


rows = []
for label, key, fes in SYSTEMS:
    ic = read_icohp(SRC / key / "ICOHPLIST.lobster")
    tot = [r for r in ic if r["oa"] is None]
    orb = [r for r in ic if r["oa"] is not None]
    for partner in ("S", "Nb"):
        bonds = [r for r in tot for fe in fes
                 if (r["ea"], r["a"], r["eb"]) == ("Fe", fe, partner) or (r["eb"], r["b"], r["ea"]) == ("Fe", fe, partner)]
        icohp = [-(r["up"] + r["dn"]) for r in bonds]
        dz = [sum(-(o["up"] + o["dn"]) for o in orb if o["idx"] == r["idx"] and {o["oa"], o["ob"]} == {"3d_z^2", "4d_z^2"})
              for r in bonds]
        rows.append((label, "Fe-" + partner, len(bonds) // len(fes), np.mean([r["d"] for r in bonds]), np.mean(icohp),
                     np.mean(dz) if partner == "Nb" else np.nan))

out = REPO / "data" / "fig2" / "fig2d_icohp_per_bond.tsv"
with open(out, "w") as fh:
    fh.write("system\tbond\tbonds_per_Fe\td_mean_A\tminus_ICOHP_per_bond_eV\tdz2_dz2_part_eV\n")
    for r in rows:
        fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.3f}\t{r[4]:.3f}\t" + ("" if np.isnan(r[5]) else f"{r[5]:.3f}") + "\n")
print(out.read_text())
