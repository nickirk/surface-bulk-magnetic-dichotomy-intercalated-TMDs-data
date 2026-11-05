"""SI Fig. S5(b): pairwise Heisenberg fit for bulk Fe1/3NbS2.

Single Heisenberg fit using all five J's and ten DFT+U configurations.
  - Left y-axis: fitted J values (meV) with ±1σ error bars (one point per J).
  - Right y-axis: cumulative RMSE as each J is added in the order
    J_1, J_1', J_2, J_2', J_3' (1NN → 2NN → 3NN), showing the model-floor
    plateau.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- publication style ---
rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size":        20,
    "axes.labelsize":   22,
    "axes.titlesize":   22,
    "xtick.labelsize":  19,
    "ytick.labelsize":  19,
    "legend.fontsize":  15,
    "axes.linewidth":    1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size":  5,
    "ytick.major.size":  5,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "lines.linewidth":   2.4,
})

# --- data: 10-config joint LSQ in the new cumulative order ---
saved = np.load(Path(__file__).resolve().parents[2] / "data" / "figS5" / "J_fit_results.npz", allow_pickle=True)
M_known    = saved["A"]
dE_known   = saved["our_dE"]
E_FM = -294.68026247
ROW_CAND = {
    "cand1": np.array([-32, -16, -24, -24, -48], dtype=float),
    "cand2": np.array([-28, -16, -28, -20, -40], dtype=float),
    "cand3": np.array([-28, -16, -20, -28, -56], dtype=float),
    "cand4": np.array([-24, -16, -32, -32, -48], dtype=float),
    "cand5": np.array([-24, -16, -24, -40, -64], dtype=float),
}
E_CAND = {"cand1": -589.73380, "cand2": -589.70830, "cand3": -589.68701,
          "cand4": -589.84744, "cand5": -589.64236}
dE_cand = {n: (E/8 - E_FM/4) * 1000 for n, E in E_CAND.items()}
A_orig = np.vstack([M_known] + [ROW_CAND[n] for n in ROW_CAND])
y      = np.concatenate([dE_known, [dE_cand[n] for n in ROW_CAND]])

# Reorder columns: original [J1, J2, J1', J2', J3'] → new [J1, J1', J2, J2', J3']
new_order = [0, 2, 1, 3, 4]
A = A_orig[:, new_order]
J_LABELS = [r"$J_1$", r"$J_1'$", r"$J_2$", r"$J_2'$", r"$J_3'$"]

# Single 5-J fit using all 10 configs  (Cov from σ²(AᵀA)⁻¹ with σ²=χ²/dof)
sol, *_ = np.linalg.lstsq(A, y, rcond=None)
res     = A @ sol - y
chi2    = float(np.sum(res ** 2))
dof     = A.shape[0] - A.shape[1]
sigma2  = chi2 / dof
cov     = sigma2 * np.linalg.inv(A.T @ A)
J_vals  = sol
J_se    = np.sqrt(np.maximum(np.diag(cov), 0))
print("5-J fit using all 10 configs:")
for label, j, s in zip(J_LABELS, J_vals, J_se):
    print(f"  {label} = {j:+.3f} ± {s:.3f}")

# Cumulative RMSE (still useful: shows the plateau behaviour)
N = np.arange(1, 6)
rmse = np.zeros(5)
for k in N:
    A_k = A[:, :k]
    sol_k, *_ = np.linalg.lstsq(A_k, y, rcond=None)
    res_k = A_k @ sol_k - y
    rmse[k-1] = float(np.sqrt(np.mean(res_k ** 2)))
print("Cumulative RMSE:", rmse.round(3))

# --- figure ---
# Aspect ratio of the former main-text Fig. 2 panel; the published Fig. S5(b)
# shows the same values on a wider canvas.
fig, axL = plt.subplots(figsize=(6.48, 5.0))
axR = axL.twinx()

# J value points (left y-axis)  -- single 5-J fit, one point per J
J_color = "#1f4e79"   # dark blue
j_pts = axL.errorbar(N, J_vals, yerr=J_se, fmt="o", color=J_color,
                     ms=14, lw=0, capsize=7, capthick=1.6, elinewidth=1.8,
                     markeredgecolor="black", markeredgewidth=0.9,
                     label=r"$J$ value", zorder=5)
axL.axhline(0, color="0.5", lw=0.7, ls="--", alpha=0.7)
axL.set_ylabel(r"$J$ (meV)", color=J_color)
axL.tick_params(axis="y", labelcolor=J_color, colors=J_color)
axL.spines["left"].set_color(J_color)
axL.set_xticks(N)
axL.set_xticklabels(J_LABELS)
axL.set_xlim(0.5, 5.5)
axL.set_ylim(-1.2, 2.5)

# --- Cumulative RMSE on right y-axis ---
rmse_line, = axR.plot(N, rmse, "-s", color="#b03030", ms=12, lw=2.6,
                      markerfacecolor="#b03030", markeredgecolor="black",
                      markeredgewidth=1.0, zorder=4, label="RMSE")
axR.set_ylabel("fit RMSE (meV/Fe)", color="#b03030")
axR.tick_params(axis="y", labelcolor="#b03030", colors="#b03030")
axR.spines["right"].set_color("#b03030")
axR.set_ylim(0, 11)

# Legend
axL.legend([j_pts, rmse_line], [r"$J$ (5-fit, 10 configs)", "cumulative RMSE"],
           loc="upper right", bbox_to_anchor=(0.98, 0.98),
           frameon=False, handlelength=1.4, handletextpad=0.4)

fig.subplots_adjust(left=0.17, right=0.83, bottom=0.16, top=0.95)

# --- save ---
out_dir = Path(__file__).resolve().parents[2] / "output" / "figS5"
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    out = out_dir / f"figS5b_J_fit.{ext}"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
