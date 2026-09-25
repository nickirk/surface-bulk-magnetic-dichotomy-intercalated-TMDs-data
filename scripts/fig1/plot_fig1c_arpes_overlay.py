"""
Overlay plotter for ARPES (greys) + optional DFT ribbons (red up / blue down; orange when spin-summed),
restored from the e2b7a12 version of scripts/plot_fns_arpes.py and
extended with two toggles needed by Fig. 1's bottom row:

    show_ylabel  — draw the 'E - E_F (eV)' label and tick labels
    show_colorbar — draw the Lo/Hi intensity colorbar on the right

All other aspects of the plot (greyscale cmap, high-sym dashed lines,
ribbon style, k-path mapping) are preserved byte-for-byte so that the
three bottom-row panels remain visually consistent.

Output goes to manuscripts/figures/Fe1_3NbS2/<out_name>.{pdf,png}.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import os

# Publication style — same settings that produced the existing overlay PDFs.
plt.style.use("default")
plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 42,
    "axes.labelsize" : 48,
    "axes.titlesize" : 48,
    "xtick.labelsize": 42,
    "ytick.labelsize": 42,
    "legend.fontsize": 42,
    "legend.frameon" : False,
    "figure.figsize" : (10, 12),
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
    "mathtext.fontset": "dejavuserif",
    "xtick.major.size":  10,
    "xtick.major.width": 2.5,
    "ytick.major.size":  10,
    "ytick.major.width": 2.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.linewidth":    2.2,
})

REPO = Path(__file__).resolve().parents[2]
EXP_FILE = REPO / "data" / "fig1" / "arpes_FNS030n2_88combine.txt"
OUT_DIR  = REPO / "output" / "fig1"

# ---------------------------------------------------------------------------
# Data loading (identical to plot_fns_arpes.py)
# ---------------------------------------------------------------------------
def _load_arpes():
    with open(EXP_FILE, "r", encoding="latin1") as f:
        lines = [l.replace("\r", "").strip("\n") for l in f.readlines()]
    energies = np.array([float(x) for x in lines[2].split("\t") if x.strip()])
    n_e = len(energies)
    ks, Is = [], []
    for line in lines[3:]:
        parts = line.split("\t")
        meaningful = [p for p in parts if p.strip()]
        if not meaningful:
            continue
        ks.append(float(meaningful[0]))
        row = [float(x) if x.strip() else 0.0 for x in parts]
        Is.append(row[-n_e:])
    k = np.array(ks); I = np.array(Is)
    if k[0] > k[-1]:
        k = k[::-1]; I = I[::-1, :]
    return k, energies, I

# ---------------------------------------------------------------------------
# Main overlay routine
# ---------------------------------------------------------------------------
# Fixed axes placement so (c), (d), (e) share identical geometry and
# the colorbar in (d) does NOT change the axes box.
AX_LEFT, AX_RIGHT, AX_TOP, AX_BOTTOM = 0.32, 0.94, 0.96, 0.15

def plot_arpes_overlay(dft_path=None, out_name="arpes_overlay",
                      e_fermi=7.730112, scale=0.05, combine_spin=False,
                      show_ylabel_text=True, show_colorbar=True,
                      show_yticks=True, show_xlabel_text=True,
                      spin_channels=None, single_spin_color=None):
    """Render an ARPES grey image with optional spin-coloured DFT ribbons.

    dft_path         Path to sw.npy, or None for ARPES-only.
    out_name         File stem (saved to manuscripts/figures/Fe1_3NbS2/).
    combine_spin     Draw both spin contributions in the same colour. Sum and
                     average differ only by an overall normalization here; use
                     the same ``scale`` as comparison panels so ribbon widths
                     retain a common visual encoding.
    show_ylabel_text If False: drop the 'E - E_F (eV)' axis label. The
                     numeric tick labels are ALWAYS kept so all three
                     panels share an identical axes box + aspect.
    show_colorbar    If True: draw an INSET Lo/Hi colorbar inside the
                     axes so the outer figure geometry is unchanged.
    """
    k_vals, energies, intensities = _load_arpes()

    # Contrast-enhanced intensities (same as original)
    gamma = 0.9
    vmax  = np.percentile(intensities, 97.)
    processed = np.clip(intensities, 0, vmax) ** gamma

    fig, ax = plt.subplots()
    # Fixed axes placement → identical geometry across (c), (d), (e).
    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT,
                        top=AX_TOP,   bottom=AX_BOTTOM)
    extent = [k_vals.min(), k_vals.max(), energies.min(), energies.max()]
    img = ax.imshow(processed.T, origin="lower", extent=extent,
                    cmap="Greys", aspect="auto", interpolation="bicubic")

    # High-sym lines
    a0  = 3.3249
    k_K = (4 * np.pi) / (3 * a0)
    k_M = k_K / 2.0
    for kp in [0, k_M, k_K]:
        ax.axvline(kp, color="white", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xlim(0, k_K)
    ax.set_xticks([0, k_M, k_K])
    ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K_{\rm 0}$"])
    # Pull Γ inward (left-aligned → sits right of its tick).
    # K₀ stays centred — the crop padding on the right is wide enough
    # that the subscript "0" barely touches the neighbouring Γ.
    xtl = ax.get_xticklabels()
    xtl[0].set_horizontalalignment("left")
    xtl[2].set_horizontalalignment("center")

    # --- DFT overlay (ribbon style, identical to e2b7a12) ---
    if dft_path is not None:
        dft_path = Path(dft_path)
        if dft_path.exists():
            sw_dft = np.load(dft_path)
            if "Co1_3NbS2" in dft_path.parts and "stripe" in dft_path.parts:
                As = np.array([
                    [5.7518315133822995, 0.0, 0.0],
                    [0.0, 9.9624644177539317, 0.0],
                    [0.0, 0.0, 12.1389822762154491],
                ])
            elif "stripe" in dft_path.parts:
                As = np.array([
                    [5.7588999999999997, 0.0, 0.0],
                    [0.0, 9.9747073957084069, 0.0],
                    [0.0, 0.0, 12.1539000000000000],
                ])
            else:
                As = np.array([
                    [ 5.7588999999999997, 0.0, 0.0],
                    [-2.8794499999999990, 4.9873536978542035, 0.0],
                    [ 0.0, 0.0, 12.1539000000000000],
                ])
            Au = np.array([
                [ 2.8794499999999998, 1.6624512326180678, 0.0],
                [-2.8794499999999994, 1.6624512326180678, 0.0],
                [ 0.0, 0.0, 12.1539000000000000],
            ])
            M_mat = np.round(np.linalg.solve(Au.T, As.T).T).astype(int)
            cell  = np.linalg.inv(M_mat) @ As
            nseg  = sw_dft.shape[1] - 1
            kf0 = np.array([0.0, 0.0, 0.125])
            kf1 = np.array([1./3, 1./3, 0.125])
            kpath_frac = np.array([kf0 + (kf1 - kf0) / nseg * i for i in range(sw_dft.shape[1])])
            kpt_c = kpath_frac @ np.linalg.inv(cell).T * 2 * np.pi
            kdist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(kpt_c, axis=0), axis=1))]

            colors = (["xkcd:bright orange", "xkcd:bright orange"]
                      if combine_spin else ["#d62728", "#1f77b4"])  # red up, blue down
            # spin_channels: which spin indices to draw (VASP order, 0 = up).
            # Used for Fig 1(c) A-AFM, where only one channel is shown so the
            # structural (not altermagnetic) Gamma-M splitting is not on display.
            # single_spin_color overrides the per-channel colour for legacy
            # one-spin renders. Spin-integrated renders use combine_spin.
            draw_spins = (range(sw_dft.shape[0]) if spin_channels is None
                          else [s for s in spin_channels if s < sw_dft.shape[0]])
            for ispin in draw_spins:
                for ib in range(sw_dft.shape[2]):
                    E = sw_dft[ispin, :, ib, 0] - e_fermi
                    W = sw_dft[ispin, :, ib, 1]
                    if np.max(W) > 0.0001 and np.max(E) > -5 and np.min(E) < 2:
                        env = W * scale
                        valid = (W > 0.0001)
                        if not np.any(valid):
                            continue
                        col = single_spin_color or colors[ispin]
                        ax.fill_between(kdist, E - env, E + env, where=valid,
                                        color=col, alpha=0.9,
                                        zorder=2, linewidth=0)
                        ax.plot(kdist, E, color=col,
                                alpha=0.25, linewidth=1, zorder=1)

    # --- Colorbar (inset so it does not change the axes box) ---
    if show_colorbar:
        cax = ax.inset_axes([0.88, 0.04, 0.05, 0.30])   # [x,y,w,h] ax-frac
        cbar = fig.colorbar(img, cax=cax)
        vmin_, vmax_ = img.get_clim()
        cbar.set_ticks([vmin_, vmax_])
        cbar.set_ticklabels(["Lo", "Hi"])
        cbar.ax.tick_params(labelsize=24)

    # --- Axis labels ---
    if show_xlabel_text:
        ax.set_xlabel(r"Momentum $k$ ($\mathrm{\mathring{A}}^{-1}$)")
    if show_ylabel_text:
        ax.set_ylabel(r"$E - E_F$ (eV)")
    if not show_yticks:
        ax.tick_params(axis="y", which="both",
                       left=False, right=False,
                       labelleft=False, labelright=False)
    ax.set_ylim(energies.min(), energies.max())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{out_name}.pdf"
    png = OUT_DIR / f"{out_name}.png"
    # NO bbox_inches='tight' — we want identical pixel dimensions across
    # panels so the composite tiles cleanly.
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  wrote {png}  ({int(fig.get_size_inches()[0]*300)}x{int(fig.get_size_inches()[1]*300)} px)")

# ---------------------------------------------------------------------------
# Driver — regenerate the three bottom-row panels of Fig. 1
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA = REPO / "data" / "fig1"
    # Fig-1(c) A-AFM ribbon: as-received (experimental neutron-diffraction) structure
    # for the like-for-like ARPES comparison. The small Γ–M splitting
    # is structural (0.58 mÅ hypersensitivity, Appendix), not the Fig-4 altermagnetic split.
    AAFM   = DATA / "aafm_unfolded_sw.npy"
    STRIPE = DATA / "stripe_unfolded_sw.npy"

    print("Rendering Fig. 1 bottom-row panels …")
    # New (c): three side-by-side sub-panels — exp, A-AFM, stripe.
    # Only exp keeps the y-axis label and tick labels/marks; the two
    # overlays have their y-axis ticks + labels blanked so they can be
    # packed tight against the exp panel in the composite.
    plot_arpes_overlay(
        dft_path=None,
        out_name="arpes_experimental_fns",
        show_ylabel_text=True, show_yticks=True, show_colorbar=False,
        show_xlabel_text=False)
    plot_arpes_overlay(
        dft_path=AAFM,
        out_name="arpes_experimental_fns_with_dft_aAFM_spin_average",
        e_fermi=7.730112,  # ground-state E_F of the as-received A-AFM calculation;
                           # not the symmetrized structure's 7.7670 (each panel uses its own run's E_F)
        scale=0.05, combine_spin=True,  # equal-weight spin integration;
                                         # same ribbon scale as Stripe
        show_ylabel_text=False, show_yticks=False, show_colorbar=False,
        show_xlabel_text=False)
    plot_arpes_overlay(
        dft_path=STRIPE,
        out_name="arpes_experimental_fns_with_dft_stripe_cyan_orange",
        e_fermi=8.66907,
        show_ylabel_text=False, show_yticks=False, show_colorbar=False,
        show_xlabel_text=False)
    print("Done.")
