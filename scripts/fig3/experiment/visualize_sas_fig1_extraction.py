from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from extract_sas_fig1_data import OUTPUT_DIR, SOURCE, colorbar_lookup, crop_with_bounds, load_image


def intensity_to_rgb(intensity: np.ndarray, colors: np.ndarray, values: np.ndarray) -> np.ndarray:
    n = len(values)
    idx = np.rint((1.0 - np.clip(intensity, 0.0, 1.0)) * (n - 1)).astype(int)
    rgb = colors[idx]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def panel_extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if y[0] <= y[-1]:
        return float(x[0]), float(x[-1]), float(y[-1]), float(y[0])
    return float(x[0]), float(x[-1]), float(y[0]), float(y[-1])


def y_limits(y: np.ndarray) -> tuple[float, float]:
    if y[0] <= y[-1]:
        return float(y[-1]), float(y[0])
    return float(y[0]), float(y[-1])


def load_panel(panel_name: str) -> dict[str, np.ndarray]:
    path = OUTPUT_DIR / f"panel_{panel_name}" / "data.npz"
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def plot_panel_comparison(
    panel_name: str,
    panel: dict[str, np.ndarray],
    colors: np.ndarray,
    values: np.ndarray,
    outdir: Path,
) -> None:
    x = panel["x"]
    y = panel["y"]
    intensity = panel["intensity"]
    masked = panel["masked"]
    rgb = panel["rgb"]
    reconstructed_rgb = intensity_to_rgb(intensity, colors, values)

    extent = panel_extent(x, y)
    ylim = y_limits(y)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    axes[0].imshow(rgb, extent=extent, aspect="auto")
    axes[0].set_title(f"Original panel ({panel_name})")

    axes[1].imshow(intensity, extent=extent, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Extracted intensity")

    axes[2].imshow(reconstructed_rgb, extent=extent, aspect="auto")
    axes[2].set_title("Recolored from extracted intensity")

    for ax in axes:
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$k_{\Gamma M_0}$ ($\mathrm{\AA}^{-1}$)" if panel_name == "a" else r"$k_{\Gamma M_0}$ ($\mathrm{\AA}^{-1}$)")
        if panel_name == "a":
            ax.set_ylabel("Binding energy (eV)")
        else:
            ax.set_ylabel(r"$k_{\Gamma K_0}$ ($\mathrm{\AA}^{-1}$)")
        ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.6)
        ax.axvline(0.0, color="white", linewidth=0.6, alpha=0.6)

    fig.savefig(outdir / f"panel_{panel_name}_comparison_axes.png", dpi=200)
    plt.close(fig)

    diff = np.abs(rgb.astype(float) - reconstructed_rgb.astype(float)).mean(axis=2)
    diff_masked = np.ma.array(diff, mask=masked)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    axes[0].imshow(masked, extent=extent, cmap="gray_r", aspect="auto")
    axes[0].set_title(f"Masked pixels ({masked.mean():.1%})")
    axes[1].imshow(diff_masked, extent=extent, cmap="magma", aspect="auto")
    axes[1].set_title("RGB residual (unmasked only)")
    for ax in axes:
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$k_{\Gamma M_0}$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel("Binding energy (eV)" if panel_name == "a" else r"$k_{\Gamma K_0}$ ($\mathrm{\AA}^{-1}$)")
    fig.savefig(outdir / f"panel_{panel_name}_mask_and_residual.png", dpi=200)
    plt.close(fig)


def main() -> None:
    vis_dir = OUTPUT_DIR / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    colorbar_rgb = load_image(OUTPUT_DIR / "colorbar_crop.png")
    colors, values = colorbar_lookup(colorbar_rgb)

    for panel_name in ("a", "b"):
        plot_panel_comparison(panel_name, load_panel(panel_name), colors, values, vis_dir)


if __name__ == "__main__":
    main()
