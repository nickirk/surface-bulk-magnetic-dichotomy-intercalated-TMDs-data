from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


import os
# Source: Figure6.xlsx from the dataset published with Popcevic et al., Phys. Rev. B 105, 155114 (2022).
XLSX_PATH = Path(os.environ.get("POPCEVIC_FIG6_XLSX", "Figure6.xlsx"))
OUTDIR = Path(os.environ.get("OUT_DIR", "output/fig3_experiment/popcevic_fig6"))
FIG6C_XLIM = (-1.30, 1.30)
FIG6C_YLIM = (-1.12, 1.12)
NBS2_A_ANGSTROM = 3.349
CO13NBS2_SUPERCELL_A_ANGSTROM = 5.792
FIG6C_LARGE_HEX_RADIUS = 4.0 * np.pi / (3.0 * NBS2_A_ANGSTROM)
FIG6C_SMALL_HEX_RADIUS = 4.0 * np.pi / (3.0 * CO13NBS2_SUPERCELL_A_ANGSTROM)
FIG6C_LARGE_HEX_GAMMA_M = np.sqrt(3.0) * FIG6C_LARGE_HEX_RADIUS / 2.0
FIG6C_SMALL_HEX_GAMMA_M = np.sqrt(3.0) * FIG6C_SMALL_HEX_RADIUS / 2.0
FIG6C_NATIVE_OUTER_ROTATION_DEG = 90.0
FIG6C_NATIVE_INNER_ROTATION_DEG = FIG6C_NATIVE_OUTER_ROTATION_DEG + 30.0


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass
class SheetData:
    name: str
    x: np.ndarray
    y: np.ndarray
    intensity: np.ndarray
    row_start: int
    row_end: int


def col_to_num(col: str) -> int:
    n = 0
    for c in col:
        n = n * 26 + ord(c.upper()) - 64
    return n


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.findall(".//a:t", NS)) for si in root.findall("a:si", NS)]


def workbook_targets(zf: zipfile.ZipFile) -> dict[str, str]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("pr:Relationship", NS)}
    targets = {}
    for sheet in wb.find("a:sheets", NS):
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        targets[sheet.attrib["name"]] = "xl/" + rid_to_target[rid]
    return targets


def parse_sheet(zf: zipfile.ZipFile, shared: list[str], target: str, name: str) -> SheetData:
    root = ET.fromstring(zf.read(target))
    cells: dict[tuple[int, int], str] = {}
    rows_present = set()
    cols_present = set()

    for cell in root.findall(".//a:sheetData/a:row/a:c", NS):
        ref = cell.attrib.get("r", "")
        m = re.match(r"([A-Z]+)(\d+)", ref)
        if not m:
            continue
        col_letters, row_str = m.groups()
        row = int(row_str)
        col = col_to_num(col_letters)
        t = cell.attrib.get("t")
        v = cell.find("a:v", NS)
        value = v.text if v is not None else ""
        if t == "s" and value:
            value = shared[int(value)]
        elif t == "inlineStr":
            value = "".join(tn.text or "" for tn in cell.findall(".//a:t", NS))
        cells[(row, col)] = value
        rows_present.add(row)
        cols_present.add(col)

    data_rows = sorted(r for r in rows_present if r >= 3)
    ranges = []
    if data_rows:
        start = prev = data_rows[0]
        for r in data_rows[1:]:
            if r != prev + 1:
                ranges.append((start, prev))
                start = r
            prev = r
        ranges.append((start, prev))
    if len(ranges) != 1:
        raise RuntimeError(f"Expected one contiguous data block in {name}, got {ranges}")
    row_start, row_end = ranges[0]

    x_cols = sorted(cols_present)
    x = np.array([float(cells[(2, c)]) for c in x_cols], dtype=float)
    nrows = row_end - row_start + 1
    # First-pass y scale from the screenshot axes: all panels visually span about +/-1.1 A^-1.
    y = np.linspace(1.1, -1.1, nrows)
    intensity = np.full((nrows, len(x_cols)), np.nan, dtype=float)
    for r in range(row_start, row_end + 1):
        for c in x_cols:
            value = cells.get((r, c), "")
            if value == "":
                continue
            try:
                intensity[r - row_start, c - 1] = float(value)
            except ValueError:
                continue

    return SheetData(name=name, x=x, y=y, intensity=intensity, row_start=row_start, row_end=row_end)


def load_all_sheets(path: Path) -> list[SheetData]:
    with zipfile.ZipFile(path) as zf:
        shared = load_shared_strings(zf)
        targets = workbook_targets(zf)
        sheets = []
        for name in ("Fig6(a)", "Fig6(b)", "Fig6(c)", "Fig6(d)"):
            sheets.append(parse_sheet(zf, shared, targets[name], name))
        return sheets


def save_sheet_summary(sheets: list[SheetData]) -> None:
    lines = []
    for s in sheets:
        valid = np.isfinite(s.intensity)
        lines.append(
            f"{s.name}: shape={s.intensity.shape}, x=[{s.x.min():.6f}, {s.x.max():.6f}], "
            f"y=[{s.y.min():.3f}, {s.y.max():.3f}], valid_fraction={valid.mean():.3f}, "
            f"rows={s.row_start}-{s.row_end}"
        )
    (OUTDIR / "sheet_summary.txt").write_text("\n".join(lines) + "\n")


def plot_single_sheet(sheet: SheetData) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="white", alpha=0.0)
    ax.imshow(
        sheet.intensity,
        extent=[sheet.x.min(), sheet.x.max(), sheet.y.min(), sheet.y.max()],
        origin="upper",
        aspect="auto",
        cmap=cmap,
    )
    ax.set_title(sheet.name)
    ax.set_xlabel(r"$k_{\Gamma K_0}$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$k_{\Gamma M_0}$ ($\mathrm{\AA}^{-1}$)")
    fig.savefig(OUTDIR / f"{sheet.name.replace('(', '_').replace(')', '')}_single.png", dpi=200)
    plt.close(fig)


def plot_four_panel(sheets: list[SheetData]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), constrained_layout=True)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="white", alpha=0.0)

    for ax, sheet in zip(axes.ravel(), sheets):
        im = ax.imshow(
            sheet.intensity,
            extent=[sheet.x.min(), sheet.x.max(), sheet.y.min(), sheet.y.max()],
            origin="upper",
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(sheet.name)
        ax.set_xlabel(r"$k_{\Gamma K_0}$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel(r"$k_{\Gamma M_0}$ ($\mathrm{\AA}^{-1}$)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Intensity (arb. units)")
    fig.savefig(OUTDIR / "figure6_four_panel_firstpass.png", dpi=220)
    plt.close(fig)


def plot_four_panel_transposed(sheets: list[SheetData]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), constrained_layout=True)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="white", alpha=0.0)

    for ax, sheet in zip(axes.ravel(), sheets):
        arr = sheet.intensity.T
        x = np.linspace(-1.1, 1.1, arr.shape[1])
        y = np.linspace(1.1, -1.1, arr.shape[0])
        im = ax.imshow(
            arr,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="upper",
            aspect="auto",
            cmap=cmap,
        )
        ax.set_title(f"{sheet.name} (transposed)")
        ax.set_xlabel(r"$k_{\Gamma K_0}$ ($\mathrm{\AA}^{-1}$)")
        ax.set_ylabel(r"$k_{\Gamma M_0}$ ($\mathrm{\AA}^{-1}$)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Intensity (arb. units)")
    fig.savefig(OUTDIR / "figure6_four_panel_transposed.png", dpi=220)
    plt.close(fig)


def save_npz(sheets: list[SheetData]) -> None:
    for sheet in sheets:
        np.savez_compressed(
            OUTDIR / f"{sheet.name.replace('(', '_').replace(')', '')}.npz",
            x=sheet.x,
            y=sheet.y,
            intensity=sheet.intensity,
        )


def hexagon_vertices(radius: float, rotation_deg: float = 0.0) -> np.ndarray:
    angles = np.deg2rad(np.arange(0, 361, 60) + rotation_deg)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def fig6c_native_frame(sheet: SheetData) -> tuple[np.ndarray, list[float], np.ndarray, np.ndarray]:
    arr = sheet.intensity
    mask = np.isfinite(arr)
    populated_cols = np.flatnonzero(mask.any(axis=0))
    if populated_cols.size == 0:
        raise RuntimeError("Fig6(c) sheet has no populated columns")

    first_col = populated_cols[0]
    col_rows = np.flatnonzero(mask[:, first_col])
    if col_rows.size == 0:
        raise RuntimeError("Leftmost populated Fig6(c) column has no finite rows")

    # Anchor the native workbook frame so the first populated column sits on the
    # left vertical edge of the rotated outer BZ, and the midpoint of that
    # column defines y = 0.
    x_shift = -FIG6C_LARGE_HEX_GAMMA_M - sheet.x[first_col]
    x_coords = sheet.x + x_shift

    row_mid = 0.5 * (col_rows.min() + col_rows.max())
    max_row_offset = max(row_mid, arr.shape[0] - 1 - row_mid)
    y_scale = FIG6C_LARGE_HEX_RADIUS / max_row_offset
    y_coords = -(np.arange(arr.shape[0]) - row_mid) * y_scale

    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    return arr, extent, x_coords, y_coords


def plot_fig6c_with_hexagons(sheet: SheetData) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 5.0), constrained_layout=True)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="white", alpha=0.0)

    arr, extent, x_coords, y_coords = fig6c_native_frame(sheet)
    ax.imshow(arr, extent=extent, origin="upper", aspect="equal", cmap=cmap)

    big_hex = hexagon_vertices(FIG6C_LARGE_HEX_RADIUS, rotation_deg=FIG6C_NATIVE_OUTER_ROTATION_DEG)
    # Keep the inner BZ in the same native frame by applying the same base
    # rotation as the outer zone plus the 30-degree relative rotation.
    small_hex = hexagon_vertices(FIG6C_SMALL_HEX_RADIUS, rotation_deg=FIG6C_NATIVE_INNER_ROTATION_DEG)
    ax.plot(big_hex[:, 0], big_hex[:, 1], color="black", linewidth=1.2, zorder=3)
    ax.plot(
        small_hex[:, 0],
        small_hex[:, 1],
        color="#66d9e8",
        linewidth=1.0,
        linestyle=(0, (2.0, 2.0)),
        zorder=3,
    )
    ax.axvline(0.0, color="black", linewidth=0.9, zorder=4)
    ax.axhline(0.0, color="black", linewidth=0.9, zorder=4)

    x_pad = 0.08
    y_pad = 0.08
    ax.set_xlim(min(x_coords.min(), -FIG6C_LARGE_HEX_GAMMA_M) - x_pad, max(x_coords.max(), FIG6C_LARGE_HEX_GAMMA_M) + x_pad)
    ax.set_ylim(y_coords.min() - y_pad, y_coords.max() + y_pad)
    ax.set_title(r"Fig. 6(c) native workbook frame")
    ax.set_xlabel(r"stored x ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"aligned row coordinate ($\mathrm{\AA}^{-1}$)")
    fig.savefig(OUTDIR / "Fig6_c_with_hexagons.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sheets = load_all_sheets(XLSX_PATH)
    save_sheet_summary(sheets)
    save_npz(sheets)
    plot_four_panel(sheets)
    plot_four_panel_transposed(sheets)
    for sheet in sheets:
        plot_single_sheet(sheet)
    plot_fig6c_with_hexagons(next(sheet for sheet in sheets if sheet.name == "Fig6(c)"))
    print("Saved outputs to", OUTDIR)


if __name__ == "__main__":
    main()
