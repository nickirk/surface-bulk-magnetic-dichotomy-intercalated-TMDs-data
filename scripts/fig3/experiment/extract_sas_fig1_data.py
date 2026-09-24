from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


import os
# Source: an image of Fig. 1 of the Co1/3TaS2 ARPES paper by Sas et al. cited in Fig. 3.
SOURCE = Path(os.environ.get("SAS_FIG1_PNG", "sas_fig1_source.png"))
OUTPUT_DIR = Path(os.environ.get("OUT_DIR", "output/fig3_experiment/sas_fig1"))


@dataclass(frozen=True)
class Bounds:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_image(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def component_bounds(rgb: np.ndarray, threshold: int = 245) -> list[Bounds]:
    mask = np.any(rgb < threshold, axis=2)
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    bounds = []
    for idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        area = int((labels[slc] == idx).sum())
        if area < 1_000:
            continue
        y0, y1 = slc[0].start, slc[0].stop - 1
        x0, x1 = slc[1].start, slc[1].stop - 1
        bounds.append((area, Bounds(x0=x0, y0=y0, x1=x1, y1=y1)))
    bounds.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in bounds]


def detect_main_panels(rgb: np.ndarray) -> dict[str, Bounds]:
    components = component_bounds(rgb)
    large = [b for b in components if b.width * b.height > 1_000_000]
    large.sort(key=lambda b: b.x0)
    if len(large) < 2:
        raise RuntimeError("Could not detect the two main figure panels.")
    return {"a": large[0], "b": large[1]}


def detect_colorbar(rgb: np.ndarray, panel_a: Bounds) -> Bounds:
    for bounds in component_bounds(rgb):
        if bounds.x0 > panel_a.x1 and bounds.y0 > panel_a.y0:
            if 20_000 < bounds.width * bounds.height < 120_000:
                return bounds
    raise RuntimeError("Could not detect the figure colorbar.")


def threshold_groups(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    active = np.flatnonzero(values >= threshold)
    if active.size == 0:
        return []
    groups = []
    start = int(active[0])
    prev = int(active[0])
    for value in active[1:]:
        value = int(value)
        if value - prev > 1:
            groups.append((start, prev))
            start = value
        prev = value
    groups.append((start, prev))
    return groups


def detect_frame(crop: np.ndarray, panel_name: str) -> Bounds:
    black = np.all(crop < 50, axis=2)
    row_counts = black.sum(axis=1)
    col_counts = black.sum(axis=0)

    row_groups = threshold_groups(row_counts, 0.98 * row_counts.max())
    col_groups = threshold_groups(col_counts, 0.98 * col_counts.max())

    if len(row_groups) < 2 or len(col_groups) < 2:
        raise RuntimeError(f"Could not detect frame lines for panel {panel_name}.")

    top = row_groups[0][0]
    bottom = row_groups[-1][1]
    left = col_groups[0][0]
    right = col_groups[-1][1]

    # Leave a tiny inset so the saved data exclude the axis line pixels.
    inset = 7 if panel_name == "a" else 6
    return Bounds(
        x0=left + inset,
        y0=top + inset,
        x1=right - inset,
        y1=bottom - inset,
    )


def crop_with_bounds(rgb: np.ndarray, bounds: Bounds) -> np.ndarray:
    return rgb[bounds.y0 : bounds.y1 + 1, bounds.x0 : bounds.x1 + 1].copy()


def build_annotation_mask(rgb: np.ndarray, panel_name: str) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    near_black = (r < 55) & (g < 55) & (b < 55)
    near_white = (r > 235) & (g > 235) & (b > 235)
    cyan = (r < 120) & (g > 150) & (b > 150)
    green = (r < 190) & (g > 145) & (b < 140)
    red = (r > 190) & (g < 130) & (b < 130)

    mask = near_black | near_white | cyan | green | red

    if panel_name == "a":
        # The colorbar belongs to panel (a), but not the data grid itself.
        return mask
    return mask


def detect_colorbar_inner_frame(colorbar_crop: np.ndarray) -> Bounds:
    black = np.all(colorbar_crop < 70, axis=2)
    row_counts = black.sum(axis=1)
    col_counts = black.sum(axis=0)

    row_groups = threshold_groups(row_counts, 0.8 * row_counts.max())
    col_groups = threshold_groups(col_counts, 0.8 * col_counts.max())
    if len(row_groups) < 2 or len(col_groups) < 2:
        raise RuntimeError("Could not detect the colorbar frame.")

    return Bounds(
        x0=col_groups[0][0] + 2,
        y0=row_groups[0][0] + 2,
        x1=col_groups[-1][1] - 2,
        y1=row_groups[-1][1] - 2,
    )


def colorbar_lookup(colorbar_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds = detect_colorbar_inner_frame(colorbar_rgb)
    inner = crop_with_bounds(colorbar_rgb, bounds)
    left = inner.shape[1] // 3
    right = inner.shape[1] - left
    stripe = inner[:, left:right]
    colors = np.median(stripe, axis=1)
    values = np.linspace(1.0, 0.0, inner.shape[0], dtype=float)
    return colors.astype(float), values


def map_rgb_to_intensity(rgb: np.ndarray, colors: np.ndarray, values: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(float)
    lookup = colors.astype(float)
    tree = cKDTree(lookup)
    _, indices = tree.query(flat, k=1)
    return values[indices].reshape(rgb.shape[:2])


def make_debug_masked(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[mask] = np.array([0, 255, 255], dtype=np.uint8)
    return out


def normalized_to_u8(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    scaled = np.round(255 * clipped).astype(np.uint8)
    return np.repeat(scaled[..., None], 3, axis=2)


def panel_axes(panel_name: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    if panel_name == "a":
        x = np.linspace(-1.0, 1.0, width)
        y = np.linspace(0.0, 4.0, height)
    elif panel_name == "b":
        x = np.linspace(-1.25, 1.25, width)
        y = np.linspace(1.25, -1.25, height)
    else:
        raise ValueError(f"Unknown panel {panel_name}")
    return x, y


def write_panel(
    panel_name: str,
    source_rgb: np.ndarray,
    panel_bounds: Bounds,
    frame_bounds: Bounds,
    colors: np.ndarray,
    values: np.ndarray,
) -> dict[str, object]:
    panel_crop = crop_with_bounds(source_rgb, panel_bounds)
    plot_rgb = crop_with_bounds(panel_crop, frame_bounds)
    mask = build_annotation_mask(plot_rgb, panel_name)
    intensity = map_rgb_to_intensity(plot_rgb, colors, values)
    intensity[mask] = 0.0

    x_axis, y_axis = panel_axes(panel_name, intensity.shape)

    panel_dir = OUTPUT_DIR / f"panel_{panel_name}"
    panel_dir.mkdir(parents=True, exist_ok=True)

    save_image(panel_dir / "panel_crop.png", panel_crop)
    save_image(panel_dir / "plot_region.png", plot_rgb)
    save_image(panel_dir / "mask_debug.png", make_debug_masked(plot_rgb, mask))
    save_image(panel_dir / "intensity_preview.png", normalized_to_u8(intensity))

    np.savez_compressed(
        panel_dir / "data.npz",
        x=x_axis,
        y=y_axis,
        intensity=intensity,
        masked=mask,
        rgb=plot_rgb,
        panel_bounds=np.array([panel_bounds.x0, panel_bounds.y0, panel_bounds.x1, panel_bounds.y1]),
        frame_bounds=np.array([frame_bounds.x0, frame_bounds.y0, frame_bounds.x1, frame_bounds.y1]),
    )

    return {
        "panel_bounds": [panel_bounds.x0, panel_bounds.y0, panel_bounds.x1, panel_bounds.y1],
        "frame_bounds": [frame_bounds.x0, frame_bounds.y0, frame_bounds.x1, frame_bounds.y1],
        "shape": list(intensity.shape),
        "x_min": float(x_axis[0]),
        "x_max": float(x_axis[-1]),
        "y_min": float(y_axis.min()),
        "y_max": float(y_axis.max()),
        "masked_fraction": float(mask.mean()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_rgb = load_image(SOURCE)
    panels = detect_main_panels(source_rgb)
    colorbar_bounds = detect_colorbar(source_rgb, panels["a"])
    colorbar_rgb = crop_with_bounds(source_rgb, colorbar_bounds)
    save_image(OUTPUT_DIR / "colorbar_crop.png", colorbar_rgb)
    colors, values = colorbar_lookup(colorbar_rgb)

    summary = {
        "source": str(SOURCE),
        "colorbar_bounds": [colorbar_bounds.x0, colorbar_bounds.y0, colorbar_bounds.x1, colorbar_bounds.y1],
        "panels": {},
    }

    for name in ("a", "b"):
        panel_rgb = crop_with_bounds(source_rgb, panels[name])
        frame = detect_frame(panel_rgb, name)
        summary["panels"][name] = write_panel(name, source_rgb, panels[name], frame, colors, values)

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
