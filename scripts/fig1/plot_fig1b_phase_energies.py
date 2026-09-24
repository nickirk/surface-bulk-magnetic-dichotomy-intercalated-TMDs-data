#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


MATERIALS = ["Fe1_3NbS2", "Co1_3NbS2", "Co1_3TaS2", "Ni1_3NbS2"]
MATERIAL_LABELS = {
    "Fe1_3NbS2": r"$\mathrm{Fe}_{1/3}\mathrm{Nb}\mathrm{S}_2$",
    "Co1_3NbS2": r"$\mathrm{Co}_{1/3}\mathrm{Nb}\mathrm{S}_2$",
    "Co1_3TaS2": r"$\mathrm{Co}_{1/3}\mathrm{Ta}\mathrm{S}_2$",
    "Ni1_3NbS2": r"$\mathrm{Ni}_{1/3}\mathrm{Nb}\mathrm{S}_2$",
}
PHASES = ["aAFM", "FM", "astrip", "zigzag"]
PHASE_MARKERS = {"aAFM": "*", "FM": "s", "astrip": "^", "zigzag": "D"}
PHASE_LABELS = {"aAFM": "A-AFM", "FM": "FM", "astrip": "Stripe", "zigzag": "Zigzag"}
PHASE_COLORS = {"aAFM": "#d62728", "FM": "#1f77b4", "astrip": "#2ca02c", "zigzag": "#ff7f0e"}
PHASE_MULTIPLICITY = {"aAFM": 1, "FM": 1, "astrip": 2, "zigzag": 4}
INTERCALANT_ATOMS_PER_UNIT = 2
OSZICAR_RE = re.compile(r"F=\s*([-+0-9.Ee]+)\s+E0=\s*([-+0-9.Ee]+).*mag=\s*([-+0-9.Ee]+)")


def material_phase_dir(base: Path, material: str, phase: str) -> Path:
    return base / material / phase / "pbe"


def parse_oszicar(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    for line in reversed(path.read_text(errors="ignore").splitlines()):
        match = OSZICAR_RE.search(line)
        if match:
            return {
                "F": float(match.group(1)),
                "E0": float(match.group(2)),
                "mag": float(match.group(3)),
                "last_line": line.strip(),
            }
    raise ValueError(f"Could not parse final energy from {path}")


def parse_outcar(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(errors="ignore")
    return {
        "normal_termination": "General timing and accounting informations for this job" in text,
        "ediff_reached": "aborting loop because EDIFF is reached" in text,
        "psmaxn_warning": "PSMAXN for non-local potential too small" in text,
        "zhegv_failed": "ZHEGV failed" in text,
        "stale_file_handle": "Stale file handle" in text,
        "bad_termination": "BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES" in text,
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase Energy Summary",
        "",
        "| Material | Phase | E0 (eV) | E0 / M atom (eV) | Delta (meV / M atom) | Mag | Normal Termination | EDIFF | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        notes = []
        if row["psmaxn_warning"]:
            notes.append("PSMAXN")
        if row["stale_file_handle"]:
            notes.append("stale-file")
        if row["bad_termination"]:
            notes.append("bad-termination")
        if row["zhegv_failed"]:
            notes.append("ZHEGV")
        lines.append(
            f"| {row['material']} | {row['phase']} | {row['E0']:.6f} | {row['E0_per_m_atom']:.6f} | {row['delta_meV']:.3f} | "
            f"{row['mag']:.4f} | {row['normal_termination']} | {row['ediff_reached']} | {', '.join(notes)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            # Axis and tick-label sizes raised 20% for legibility.
            "font.size":       28,
            "axes.labelsize":  41,   # was 34
            "axes.titlesize":  41,   # was 34
            "xtick.labelsize": 34,   # was 28
            "ytick.labelsize": 36,   # was 30
            "legend.fontsize": 26,
            "xtick.major.size":  8,
            "xtick.major.width": 2.0,
            "ytick.major.size":  8,
            "ytick.major.width": 2.0,
            "xtick.direction":   "out",
            "ytick.direction":   "out",
            "axes.linewidth":    2.0,
        }
    )
    x_positions = {material: idx for idx, material in enumerate(MATERIALS)}
    # Tall aspect for Fig. 1 panel (b): the composite uses a stacked
    # structure column on the left, so this panel should match that height
    # instead of forcing the structure column to shrink.
    fig, ax = plt.subplots(figsize=(11.0, 9.0))

    for phase in PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase]
        xs = [x_positions[row["material"]] for row in phase_rows]
        ys = [row["delta_meV"] for row in phase_rows]
        if phase == "aAFM":
            ax.scatter(
                xs,
                ys,
                s=560,
                marker=PHASE_MARKERS[phase],
                label=PHASE_LABELS[phase],
                c=PHASE_COLORS[phase],
                linewidths=1.0,
                zorder=4,
            )
        else:
            ax.scatter(
                xs,
                ys,
                s=420,
                marker=PHASE_MARKERS[phase],
                label=PHASE_LABELS[phase],
                facecolors="none",
                edgecolors=PHASE_COLORS[phase],
                linewidths=2.0,
                zorder=3,
            )

    ax.set_xticks(
        list(x_positions.values()),
        [MATERIAL_LABELS[material] for material in MATERIALS],
    )
    ax.set_ylabel(r"$\Delta E$ (meV/mag. ion)")
    ax.axhline(0.0, color="0.75", linewidth=1)
    # mode="expand" stretched the row to the full axes width, which is what
    # opened the gap between each marker and its text. Pack naturally and
    # centre instead; handletextpad closes the marker-to-text gap directly.
    ax.legend(frameon=False, ncols=4,
              handletextpad=0.25, columnspacing=1.4, borderpad=0.0,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize fetched phase energies and make a comparison plot")
    parser.add_argument("--fetched-dir")
    parser.add_argument("--from-csv", help="Reuse an existing phase_energies.csv instead of reparsing VASP outputs")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()

    rows = []
    if args.from_csv:
        with Path(args.from_csv).expanduser().resolve().open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["delta_meV"] = float(row["delta_meV"])
    else:
        if not args.fetched_dir:
            parser.error("--fetched-dir is required unless --from-csv is used")
        fetched_dir = Path(args.fetched_dir).expanduser().resolve()
        for material in MATERIALS:
            for phase in PHASES:
                run_dir = material_phase_dir(fetched_dir, material, phase)
                osz = parse_oszicar(run_dir / "OSZICAR")
                out = parse_outcar(run_dir / "OUTCAR")
                rows.append(
                    {
                        "material": material,
                        "phase": phase,
                        "run_dir": str(run_dir),
                        "E0": osz["E0"],
                        "E0_per_m_atom": osz["E0"] / (PHASE_MULTIPLICITY[phase] * INTERCALANT_ATOMS_PER_UNIT),
                        "mag": osz["mag"],
                        "last_oszicar_line": osz["last_line"],
                        **out,
                    }
                )

        base_energies = {
            material: min(row["E0_per_m_atom"] for row in rows if row["material"] == material)
            for material in MATERIALS
        }
        for row in rows:
            row["delta_meV"] = (row["E0_per_m_atom"] - base_energies[row["material"]]) * 1000.0

    rows.sort(key=lambda row: (MATERIALS.index(row["material"]), PHASES.index(row["phase"])))

    if not args.from_csv:
        write_csv(out_dir / "phase_energies.csv", rows)
        write_markdown(out_dir / "phase_energies.md", rows)
    make_plot(out_dir / "phase_energy_differences.png", rows)

    print(f"[ok] wrote {out_dir / 'phase_energies.csv'}")
    print(f"[ok] wrote {out_dir / 'phase_energies.md'}")
    print(f"[ok] wrote {out_dir / 'phase_energy_differences.png'}")


if __name__ == "__main__":
    main()
