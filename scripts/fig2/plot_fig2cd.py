"""Fig. 2(c) surface-order energies and Fig. 2(d) -ICOHP per bond, drawn as separate panels.

(c) reads data/fig2/fig2c_surface_order_energies.tsv and plots E - E(stripe) in meV/Fe (eight Fe per slab).
(d) reads data/fig2/fig2d_icohp_per_bond.tsv (made by scripts/fig2/icohp_per_bond.py); hatching marks the
    Fe 3d_z2 - Nb 4d_z2 part of the Fe-Nb bar.

    python3 scripts/fig2/plot_fig2cd.py      # from the repository root

Writes output/fig2/fig2c_surface_orders.{pdf,png} and output/fig2/fig2d_icohp.{pdf,png}.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
DATA, OUT = REPO / "data" / "fig2", REPO / "output" / "fig2"
EN = r"$\endash$"
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 6.8, "axes.labelsize": 8.2,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "axes.linewidth": 0.75,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "hatch.linewidth": 0.5})
BULK, SURF_U, SURF_R = "#0072B2", "#f2b880", "#D55E00"
ORD = {"A-AFM": "black", "stripe": "#009E73", "FM1": "#E69F00", "FM2": "#CC79A7"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def surface_order_energies():
    rows = [r for r in csv.reader(open(DATA / "fig2c_surface_order_energies.tsv"), delimiter="\t")
            if r and not r[0].startswith("#")][1:]
    tab = {(b, o): (float(v), u) for b, o, v, u in rows}
    blocks = ["Fe-term_unrelaxed", "Fe-term_relaxed", "Fe-term_r2SCAN", "NbS2-term_relaxed"]
    data = {k: [] for k in ("A-AFM", "FM1", "FM2")}
    for b in blocks:
        st, unit = tab[(b, "stripe")]
        to_mev = 1000.0 if unit == "eV_E0" else 1.0
        for k in data:
            data[k].append((tab[(b, k)][0] - st) * to_mev / 8)
    return data


# ---------------------------------------------------------------- (c)
data = surface_order_energies()
fig, ax = plt.subplots(figsize=(2.6, 2.0))
half = 0.30
ax.axhline(0, color="0.75", lw=0.5, zorder=0)
for i in range(4):
    ax.plot([i - half, i + half], [0, 0], color=ORD["stripe"], lw=1.8, solid_capstyle="butt")
for name, pts in data.items():
    for i, y in enumerate(pts):
        ax.plot([i - half, i + half], [y, y], color=ORD[name], lw=1.8, solid_capstyle="butt")
        if 0 < i < 3:
            ax.plot([i - 1 + half, i - half], [pts[i - 1], y], color=ORD[name], lw=0.6, ls=":")
ax.set_xlim(-0.55, 3.55); ax.set_ylim(-4.0, 31.0); ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xticks(range(4), ["surf.\nunrelax.", "surf.\nrelax.", r"r$^2$SCAN", "NbS$_2$\nterm."])
ax.set_ylabel(r"$E - E_{\mathrm{stripe}}$ (meV/Fe)", labelpad=1)
ax.legend([Line2D([], [], color=ORD[k], lw=1.8) for k in ("A-AFM", "stripe", "FM2", "FM1")],
          ["A-AFM", "stripe", "FM2", "FM1"], loc="upper left", ncol=2, frameon=False, handlelength=1.0,
          handletextpad=0.3, labelspacing=0.2, columnspacing=0.8, borderaxespad=0.3)
fig.tight_layout()
save(fig, "fig2c_surface_orders")

# ---------------------------------------------------------------- (d)
rows = list(csv.DictReader(open(DATA / "fig2d_icohp_per_bond.tsv"), delimiter="\t"))
val = {(r["system"], r["bond"]): r for r in rows}
fig, ax = plt.subplots(figsize=(2.2, 2.0))
wbar, x0 = 0.26, np.arange(2)
for j, (sysname, lab, col) in enumerate((("bulk", "bulk", BULK), ("surface_unrelaxed", "surf. unrelax.", SURF_U),
                                         ("surface_relaxed", "surf. relax.", SURF_R))):
    xs = x0 + (j - 1) * wbar
    ax.bar(xs, [float(val[(sysname, b)]["minus_ICOHP_per_bond_eV"]) for b in ("Fe-S", "Fe-Nb")], wbar,
           color=col, ec="k", lw=0.5, label=lab)
    ax.bar(xs[1], float(val[(sysname, "Fe-Nb")]["dz2_dz2_part_eV"]), wbar, color="none", ec="white", lw=0, hatch="///")
ax.bar([np.nan], [np.nan], color="0.85", ec="k", lw=0.5, hatch="///", label=r"Fe$\,d_{z^2}$" + EN + r"Nb$\,d_{z^2}$")
ax.set_xticks(x0, ["Fe" + EN + "S", "Fe" + EN + "Nb"]); ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0, 4); ax.set_yticks([0, 1, 2, 3, 4])
ax.set_ylabel(r"$-$ICOHP per bond (eV)", labelpad=1)
ax.legend(frameon=False, handlelength=1.0, handletextpad=0.3, borderaxespad=0.25, labelspacing=0.2, loc="upper left")
fig.tight_layout()
save(fig, "fig2d_icohp")
