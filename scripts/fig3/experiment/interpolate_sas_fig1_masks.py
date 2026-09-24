from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from extract_sas_fig1_data import OUTPUT_DIR
from visualize_sas_fig1_extraction import intensity_to_rgb, load_panel, panel_extent, y_limits


def estimate_support(intensity: np.ndarray, masked: np.ndarray) -> np.ndarray:
    clean = intensity.copy()
    clean[masked] = 0.0
    seed = clean > 0.01
    closed = ndimage.binary_closing(seed, structure=np.ones((9, 9), dtype=bool), iterations=2)
    filled = ndimage.binary_fill_holes(closed)
    labels, count = ndimage.label(filled)
    if count == 0:
        return np.zeros_like(intensity, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def nearest_seed(values: np.ndarray, known_mask: np.ndarray) -> np.ndarray:
    # distance_transform_edt uses zeros as seed locations, so invert known_mask.
    indices = ndimage.distance_transform_edt(~known_mask, return_distances=False, return_indices=True)
    return values[tuple(indices)]


def diffuse_inpaint(
    intensity: np.ndarray,
    target_mask: np.ndarray,
    support_mask: np.ndarray,
    sigma: float = 2.0,
    iterations: int = 80,
) -> np.ndarray:
    known_mask = support_mask & ~target_mask
    filled = intensity.copy()
    filled[target_mask] = nearest_seed(intensity, known_mask)[target_mask]

    support_f = support_mask.astype(float)
    eps = 1e-12
    for _ in range(iterations):
        weighted = ndimage.gaussian_filter(filled * support_f, sigma=sigma, mode="nearest")
        norm = ndimage.gaussian_filter(support_f, sigma=sigma, mode="nearest")
        proposal = weighted / np.maximum(norm, eps)
        filled[target_mask] = proposal[target_mask]
        filled[known_mask] = intensity[known_mask]

    filled[~support_mask] = 0.0
    return filled


def save_preview(panel_dir: Path, name: str, rgb: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(panel_dir / name)


def masked_preview_rgb(
    intensity: np.ndarray,
    support_mask: np.ndarray,
    colors: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    rgb = intensity_to_rgb(intensity, colors, values)
    rgb = rgb.copy()
    rgb[~support_mask] = 0
    return rgb


def support_debug_image(support_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    img = np.zeros(support_mask.shape + (3,), dtype=np.uint8)
    img[support_mask] = [255, 255, 255]
    img[target_mask] = [255, 0, 0]
    return img


def expand_target_mask(panel_name: str, target_mask: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    if panel_name == "b":
        expanded = ndimage.binary_dilation(
            target_mask,
            structure=np.ones((5, 5), dtype=bool),
            iterations=2,
        )
        return expanded & support_mask
    return target_mask


def zero_known_outside_artifacts(
    panel_name: str,
    x: np.ndarray,
    y: np.ndarray,
    intensity: np.ndarray,
    support_mask: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if panel_name != "b":
        return intensity, support_mask, target_mask

    xx, yy = np.meshgrid(x, y)
    # The remaining M0 label ghost sits below the octagonal support near the
    # lower-right corner. It is outside the experimental data area, so force it to zero.
    cleanup = (xx > 0.54) & (xx < 0.77) & (yy < -1.05)

    cleaned_intensity = intensity.copy()
    cleaned_support = support_mask.copy()
    cleaned_target = target_mask.copy()

    cleaned_intensity[cleanup] = 0.0
    cleaned_support[cleanup] = False
    cleaned_target[cleanup] = False
    return cleaned_intensity, cleaned_support, cleaned_target


def plot_interpolation_comparison(
    panel_name: str,
    x: np.ndarray,
    y: np.ndarray,
    original: np.ndarray,
    interpolated: np.ndarray,
    support_mask: np.ndarray,
    target_mask: np.ndarray,
    panel_dir: Path,
) -> None:
    extent = panel_extent(x, y)
    ylim = y_limits(y)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].imshow(original, extent=extent, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Original extracted intensity")
    axes[1].imshow(interpolated, extent=extent, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Interpolated intensity")
    delta = np.where(target_mask, interpolated - original, 0.0)
    vmax = np.max(np.abs(delta[target_mask])) if np.any(target_mask) else 1.0
    axes[2].imshow(delta, extent=extent, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Change on masked region")

    xlabel = r"$k_{\Gamma M_0}$ ($\mathrm{\AA}^{-1}$)"
    ylabel = "Binding energy (eV)" if panel_name == "a" else r"$k_{\Gamma K_0}$ ($\mathrm{\AA}^{-1}$)"
    for ax in axes:
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.savefig(panel_dir / "interpolation_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    axes[0].imshow(support_mask, extent=extent, cmap="gray_r", aspect="auto")
    axes[0].set_title("Support region")
    axes[1].imshow(target_mask, extent=extent, cmap="gray_r", aspect="auto")
    axes[1].set_title("Interpolated pixels only")
    for ax in axes:
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.savefig(panel_dir / "support_and_target_masks.png", dpi=200)
    plt.close(fig)


def process_panel(panel_name: str, colors: np.ndarray, values: np.ndarray) -> None:
    panel_dir = OUTPUT_DIR / f"panel_{panel_name}"
    data = load_panel(panel_name)
    x = data["x"]
    y = data["y"]
    intensity = data["intensity"]
    masked = data["masked"]

    support_mask = estimate_support(intensity, masked)
    target_mask = support_mask & masked
    target_mask = expand_target_mask(panel_name, target_mask, support_mask)
    interpolated = diffuse_inpaint(intensity, target_mask, support_mask)
    interpolated, support_mask, target_mask = zero_known_outside_artifacts(
        panel_name, x, y, interpolated, support_mask, target_mask
    )

    np.savez_compressed(
        panel_dir / "interpolated_data.npz",
        x=x,
        y=y,
        intensity=interpolated,
        support_mask=support_mask,
        target_mask=target_mask,
    )

    save_preview(panel_dir, "support_debug.png", support_debug_image(support_mask, target_mask))
    save_preview(panel_dir, "interpolated_preview.png", masked_preview_rgb(interpolated, support_mask, colors, values))
    plot_interpolation_comparison(panel_name, x, y, intensity, interpolated, support_mask, target_mask, panel_dir)


def main() -> None:
    colorbar = np.array(plt.imread(OUTPUT_DIR / "colorbar_crop.png"))
    if colorbar.dtype != np.uint8:
        colorbar = (255 * colorbar[..., :3]).astype(np.uint8)
    colors = np.median(colorbar[:, colorbar.shape[1] // 3 : colorbar.shape[1] * 2 // 3], axis=1)
    values = np.linspace(1.0, 0.0, len(colors), dtype=float)

    for panel_name in ("a", "b"):
        process_panel(panel_name, colors, values)


if __name__ == "__main__":
    main()
