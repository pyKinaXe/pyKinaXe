"""Compare pyKinaXe's and KRSA's kinase-family Z-scores per dataset.

Both methods score kinase families in KRSA's vocabulary (pyKinaXe's family
analysis runs in KRSA mapping mode), and results_KRSA/ is produced from
pyKinaXe's own image analysis, so both sides start from the same spot values
and the comparison measures the scoring. See NOTE.txt for the folder layout,
the reference run and the interpretation.

Input per dataset:
    results_pykinaxe/results_<dataset>_data_set/results_kinase_families/
        families_<control>_<test>.csv               pyKinaXe, both arrays in one file
    results_KRSA/results_<dataset>_data_set/<PTK|STK>/tables/
        <test>_vs_<control>_KRSA_acrossChip.txt     KRSA, one file per array

Output per dataset, under output/<dataset>/:
    family_z_comparison.csv   one row per (array, comparison, family) with both
                              Z-scores, their difference and the call
    overlap_summary.csv       significant-family overlap per array and pooled;
                              only scope == "array" rows count families, the
                              pooled rows count (array, family) instances
    waterfall/                paired family-Z waterfalls per array and comparison

pyKinaXe's Z_Score is the mean over the LFC cutoffs 0.2/0.3/0.4, so the KRSA
counterpart is acrossChip.AvgZ, not Zscores_primary. Z_THRESHOLD applies to
both sides. Families are keyed by (array, family), because PTK and STK are
independent arrays with disjoint peptide sets. pyKinaXe names comparisons
<control>_vs_<test>, KRSA <test>_vs_<control>; the output uses pyKinaXe's order.

Run:
    python validation_paper/compare_pyKinaXe_results_with_KRSA/compare_pykinaxe_krsa.py
    ...                                                        --datasets rat
    ...                                                        --z-threshold 1.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


# ============================================================================
# CONFIG
# ============================================================================
COMPARE_DIR = Path(__file__).resolve().parent
RESULTS_PYKINAXE = COMPARE_DIR / "results_pykinaxe"
RESULTS_KRSA = COMPARE_DIR / "results_KRSA"
OUTPUT_DIR = COMPARE_DIR / "output"

# The family waterfall is styled from the SAME file as the pipeline's own peptide
# waterfall, so both plots stay visually consistent and a style change is made in
# one place. Missing file => the built-in defaults of that config.
WATERFALL_STYLE_PATH = COMPARE_DIR.parents[1] / "config" / "waterfall_plot_config.yaml"

# Common absolute Z-score threshold applied to BOTH methods. 2.0 is simultaneously
# KRSA's own hit convention (run_krsa.R: z_hit_threshold) and pyKinaXe's configured
# family threshold (config/upstream_kinase_families_analysis.yaml:
# default_kpea_zscore_threshold). Lowering it to 1.5 roughly triples the call count
# on both sides and drops the Jaccard to ~0.5 -- threshold jitter, not disagreement:
# the family-Z distribution is dense around 1.5. Report r and RMS, which are
# threshold-free, alongside any Jaccard.
Z_THRESHOLD = 2.0

# Arrays to compare. KRSA stores one table per array; pyKinaXe puts both in one
# file and distinguishes them in its Type column.
ARRAYS = ["PTK", "STK"]

# The datasets. The input folders are fixed here rather than globbed, so the
# reported numbers refer to one specific pair of runs. `pykinaxe` holds one
# families_<control>_<test>.csv per comparison; `krsa` is the folder
# run_KRSA_on_pykinaxe_input.py wrote.
DATASETS: dict[str, dict[str, str]] = {
    "benchmarking": {
        "title": "benchmarking (HDV constructs)",
        "pykinaxe": "results_pykinaxe/results_benchmarking_data_set/results_kinase_families",
        "krsa": "results_KRSA/results_benchmarking_data_set",
        "output_subdir": "benchmarking",
    },
    "rat": {
        "title": "CDRL vwr-rats-kinome",
        "pykinaxe": "results_pykinaxe/results_CDRL_vwr-rats-kinome_data_set/results_kinase_families",
        "krsa": "results_KRSA/results_CDRL_vwr-rats-kinome_data_set",
        "output_subdir": "rat",
    },
}

# KRSA Z-score table naming and columns. acrossChip/AvgZ is the structural
# counterpart to pyKinaXe's cutoff-averaged Z_Score -- see the module docstring.
KRSA_TABLE_TEMPLATE = "{test}_vs_{control}_KRSA_acrossChip.txt"
KRSA_KINASE_COL = "Kinase"
KRSA_Z_COL = "AvgZ"

# pyKinaXe family-output columns.
PYK_FAMILY_COL = "Kinase_Family"
PYK_Z_COL = "Z_Score"
PYK_TYPE_COL = "Type"
PYK_COMPARISON_COL = "Comparison"

# --- print geometry --------------------------------------------------------
# The waterfalls are sized for PAPER, not for the screen, and for ONE specific
# page: THREE comparisons side by side on A4, each column carrying the STK panel
# above the PTK panel of the same comparison.
#
#     210 mm - 2 x 8 mm margin - 2 x 5 mm gutter = 3 x 61.3 mm wide
#     297 mm - 2 x 8 mm margin - 4 mm between the two panels
#             - 12 mm legend strip - 4 mm gap                = 261 mm per column
#
# Each PNG is built at exactly that width and saved WITHOUT a tight bounding box,
# so the width is real at PLOT_DPI. Placed at 100 % on the page, every font size
# below IS its point size on paper -- nothing is scaled afterwards, which is what
# makes print text unreadable.
#
# FONT_Y_TICK is ONE number for every panel, PTK and STK alike: they are read
# above one another, so their family labels must match. It is the largest size at
# which the WORST column still fits: 82 STK + 24 PTK families = 106 rows share the
# 221 mm of row space left once both panels' titles and axes are paid for, which
# is a 5.9 pt pitch and therefore 5.1 pt of type. It cannot be raised without
# either dropping families or giving up the single page -- the script reports
# every column that does not fit instead of silently shrinking anything.
PANEL_WIDTH_MM = 61.3
COLUMN_HEIGHT_MM = 261.0       # STK panel + PTK panel of one comparison, stacked
FONT_TITLE = 9.0
FONT_AXIS_LABEL = 8.0
FONT_X_TICK = 7.0
FONT_Y_TICK = 5.1              # the same on PTK and STK -- see the note above
FONT_LEGEND = 7.5
ROW_SPACING = 1.15             # row height = y-tick font size x this
PLOT_DPI = 300

# The x axis is IDENTICAL on every panel -- same limits, same ticks, same labels --
# so two panels can be read against each other and a marker at the same distance
# from zero means the same Z everywhere. Letting each panel autoscale (PTK to
# +/-2, STK to +/-4) made that silently false. 4.0 clears the data: the largest
# |Z| measured on either dataset is 3.85. The step of 2 puts a tick exactly on the
# significance threshold and keeps the labels apart on a 61 mm panel. A value
# outside the range would be CLIPPED, so the script says so instead, per panel.
X_LIMIT = 4.0
X_TICK_STEP = 2.0
X_TICK_DECIMALS = 0

# The legend is NOT drawn on the panels -- it covered data on the tall ones and it
# is redundant when several panels are composed into one figure. It is written
# once per dataset as its own PNG, in ONE ROW, cropped to its content, at the same
# point sizes as the panels, so it fits as a strip under the three columns.
LEGEND_FILENAME = "waterfall_legend.png"
LABEL_PYKINAXE = "pyKinaXe (Z_Score)"
LABEL_KRSA = "KRSA (AvgZ)"

CSV_FAMILY_Z = "family_z_comparison.csv"
CSV_OVERLAP = "overlap_summary.csv"
WATERFALL_SUBDIR = "waterfall"
# ============================================================================


def _split_comparison(label: str) -> tuple[str, str]:
    """Split a pyKinaXe 'Comparison' label into (control, test).

    pyKinaXe writes ``<control>_vs_<test>`` ("mock_vs_pLHDAg",
    "HPC_CTL_vs_STR_Exer"), i.e. the control first. Since 2026-08-25 both halves
    ARE the sample names, so no translation table is needed -- the KRSA file name
    for the same comparison is the two halves in the other order.
    """
    text = str(label).strip()
    if "_vs_" not in text:
        raise ValueError(
            f"Cannot read a control/test pair from Comparison = '{text}'. "
            "Expected '<control>_vs_<test>'. Family tables written before "
            "2026-08-25 carried positional labels (Control_vs_Test1) and are not "
            "supported here; re-run the pipeline on that dataset."
        )
    control, test = text.split("_vs_", 1)
    return control.strip(), test.strip()


def load_pykinaxe_families(family_dir: Path) -> pd.DataFrame:
    """Read every per-comparison family CSV of one dataset into one tidy frame."""
    if not family_dir.is_dir():
        raise FileNotFoundError(
            f"pyKinaXe family folder not found: {family_dir}\n"
            "Copy the pyKinaXe family results of that dataset there (see NOTE.txt)."
        )
    files = sorted(family_dir.glob("families_*.csv"))
    if not files:
        raise FileNotFoundError(f"No families_*.csv in {family_dir} (see NOTE.txt).")

    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pairs = raw[PYK_COMPARISON_COL].map(_split_comparison)

    out = pd.DataFrame(
        {
            "Array": raw[PYK_TYPE_COL].astype(str).str.strip(),
            "Comparison": raw[PYK_COMPARISON_COL].astype(str).str.strip(),
            "Control": [p[0] for p in pairs],
            "Test": [p[1] for p in pairs],
            "Kinase_Family": raw[PYK_FAMILY_COL].astype(str).str.strip(),
            "Z_pyKinaXe": pd.to_numeric(raw[PYK_Z_COL], errors="coerce"),
        }
    )
    if "NumSubstrates" in raw.columns:
        out["NumSubstrates"] = pd.to_numeric(raw["NumSubstrates"], errors="coerce")
    print(f"  pyKinaXe: {len(files)} file(s), {len(out)} family rows  <- {family_dir}")
    return out


def load_krsa_families(krsa_dir: Path, array: str, control: str,
                       test: str) -> pd.DataFrame | None:
    """Return every KRSA family AvgZ for one array and comparison, or None."""
    path = (krsa_dir / array / "tables"
            / KRSA_TABLE_TEMPLATE.format(test=test, control=control))
    if not path.exists():
        print(f"  WARNING: missing KRSA table {path} -- array dropped from BOTH "
              "sides for this comparison.")
        return None
    df = pd.read_csv(path, sep="\t")
    # acrossChip holds one row per kinase x LFC cutoff, but AvgZ is constant within
    # a kinase (it IS the mean over those rows). Keeping one row per kinase is a
    # statement of intent, not a fix for double counting.
    df = df.drop_duplicates(subset=KRSA_KINASE_COL)
    return pd.DataFrame(
        {
            "Array": array,
            "Control": control,
            "Test": test,
            "Kinase_Family": df[KRSA_KINASE_COL].astype(str).str.strip(),
            "Z_KRSA": pd.to_numeric(df[KRSA_Z_COL], errors="coerce"),
        }
    )


def build_family_z_table(
    pyk: pd.DataFrame, krsa_dir: Path, z_threshold: float
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Pair both methods' family Z-scores into one tidy table.

    An OUTER join, so a family scored by only one of the two stays visible with a
    missing Z on the other side rather than disappearing.

    Returns:
        tuple: ``(table, available)`` -- the paired table and the
        ``(array, comparison)`` pairs for which a KRSA table was actually found.
        Everything counted downstream is restricted to those pairs: counting the
        pyKinaXe side against a missing KRSA table would look like total
        disagreement rather than missing input.
    """
    comparisons = (
        pyk[["Comparison", "Control", "Test"]]
        .drop_duplicates()
        .sort_values("Comparison")
        .itertuples(index=False)
    )

    krsa_frames: list[pd.DataFrame] = []
    available: list[tuple[str, str]] = []
    for comp in comparisons:
        for array in ARRAYS:
            frame = load_krsa_families(krsa_dir, array, comp.Control, comp.Test)
            if frame is None:
                continue
            krsa_frames.append(frame)
            available.append((array, comp.Comparison))

    krsa = (
        pd.concat(krsa_frames, ignore_index=True)
        if krsa_frames
        else pd.DataFrame(columns=["Array", "Control", "Test", "Kinase_Family",
                                   "Z_KRSA"])
    )

    merged = pyk.merge(krsa, on=["Array", "Control", "Test", "Kinase_Family"],
                       how="outer")
    # A family only KRSA scores arrives without the pyKinaXe-side label columns.
    merged["Comparison"] = merged["Comparison"].fillna(
        merged["Control"].astype(str) + "_vs_" + merged["Test"].astype(str)
    )

    sig_pyk = merged["Z_pyKinaXe"].abs() >= z_threshold
    sig_krsa = merged["Z_KRSA"].abs() >= z_threshold
    merged["Call"] = "none"
    merged.loc[sig_krsa & ~sig_pyk, "Call"] = "KRSA_only"
    merged.loc[sig_pyk & ~sig_krsa, "Call"] = "pyKinaXe_only"
    merged.loc[sig_pyk & sig_krsa, "Call"] = "both"
    merged["Z_difference"] = merged["Z_pyKinaXe"] - merged["Z_KRSA"]

    keep = set(available)
    merged = merged[
        [(a, c) in keep for a, c in zip(merged["Array"], merged["Comparison"])]
    ]

    columns = ["Array", "Comparison", "Kinase_Family", "Z_pyKinaXe", "Z_KRSA",
               "Z_difference", "Call"]
    if "NumSubstrates" in merged.columns:
        columns.insert(-1, "NumSubstrates")
    table = (
        merged[columns]
        .sort_values(["Array", "Comparison", "Z_pyKinaXe"],
                     ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return table, available


def _overlap_row(scope: str, array: str, comparison: str, unit: str,
                 krsa: set, pyk: set) -> dict:
    """Build one summary row from two comparable sets.

    ``unit`` records what the members are, because only ``scope == "array"`` rows
    count families -- the pooled scopes count array- or array/comparison-qualified
    instances, and their counts must not be re-quoted as family counts.
    """
    overlap, union = krsa & pyk, krsa | pyk
    return {
        "Scope": scope,
        "Array": array,
        "Comparison": comparison,
        "Unit": unit,
        "n_KRSA": len(krsa),
        "n_pyKinaXe": len(pyk),
        "n_overlap": len(overlap),
        "n_KRSA_only": len(krsa - pyk),
        "n_pyKinaXe_only": len(pyk - krsa),
        "n_union": len(union),
        "Jaccard": round(len(overlap) / len(union), 3) if union else 0.0,
    }


def build_overlap_summary(family_z: pd.DataFrame,
                          available: list[tuple[str, str]]) -> pd.DataFrame:
    """Aggregate the paired table into the per-array and pooled overlap rows."""
    rows: list[dict] = []
    krsa_all: set[tuple[str, str, str]] = set()
    pyk_all: set[tuple[str, str, str]] = set()

    for comparison in sorted({c for _array, c in available}):
        arrays = [a for a, c in available if c == comparison]
        krsa_cond: set[tuple[str, str]] = set()
        pyk_cond: set[tuple[str, str]] = set()
        for array in ARRAYS:
            if array not in arrays:
                continue
            sub = family_z[(family_z["Array"] == array)
                           & (family_z["Comparison"] == comparison)]
            krsa_a = set(sub.loc[sub["Call"].isin(["both", "KRSA_only"]),
                                 "Kinase_Family"])
            pyk_a = set(sub.loc[sub["Call"].isin(["both", "pyKinaXe_only"]),
                                "Kinase_Family"])
            # Per-array rows are the only ones whose unit is "families".
            rows.append(_overlap_row("array", array, comparison, "families",
                                     krsa_a, pyk_a))
            krsa_cond |= {(array, fam) for fam in krsa_a}
            pyk_cond |= {(array, fam) for fam in pyk_a}
            krsa_all |= {(array, comparison, fam) for fam in krsa_a}
            pyk_all |= {(array, comparison, fam) for fam in pyk_a}

        rows.append(_overlap_row("comparison", "+".join(ARRAYS), comparison,
                                 "(array,family) instances", krsa_cond, pyk_cond))

    if rows:
        rows.append(_overlap_row("all", "+".join(ARRAYS), "ALL_comparisons",
                                 "(array,comparison,family) instances",
                                 krsa_all, pyk_all))
    return pd.DataFrame(rows)


def agreement_stats(family_z: pd.DataFrame) -> dict[str, float]:
    """Threshold-free agreement of the two Z-scores, over families BOTH scored."""
    both = family_z.dropna(subset=["Z_pyKinaXe", "Z_KRSA"])
    if len(both) < 3:
        return {"n": float(len(both))}
    return {
        "n": float(len(both)),
        "r": float(both["Z_pyKinaXe"].corr(both["Z_KRSA"])),
        "RMS": float((both["Z_difference"] ** 2).mean() ** 0.5),
        "max_abs_diff": float(both["Z_difference"].abs().max()),
    }


def _load_waterfall_style() -> dict:
    """Load the shared waterfall style, falling back to its documented defaults."""
    try:
        with WATERFALL_STYLE_PATH.open() as handle:
            cfg = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        print(f"  WARNING: {WATERFALL_STYLE_PATH} not found -- using built-in style.")
        cfg = {}
    return cfg


def _draw_family_waterfall(df: pd.DataFrame, title: str, out_path: Path,
                           z_threshold: float) -> float:
    """Draw a paired waterfall of the family Z-scores of both methods.

    One row per kinase family, ranked by the pyKinaXe Z so the plot reads as a
    waterfall of OUR result, with the KRSA value for the same family drawn on the
    same row. A connector marks the gap between the two, which is what the eye
    should be drawn to. Dashed verticals sit at the shared +/-z_threshold.

    The figure is built at the PRINT size defined in the CONFIG block (one of
    three columns on A4) and saved without a tight bounding box, so the point
    sizes there are the point sizes on paper -- the SAME ones on every panel, PTK
    and STK alike; a longer family list gets a taller panel, not smaller type. It
    carries NO legend -- that is a file of its own, see _draw_legend_png(). The x
    axis is the same on every panel (see X_LIMIT), so panels can be read against
    each other; only the family list and the panel height differ.

    COLOURS come from config/waterfall_plot_config.yaml, the same file the
    pipeline's peptide waterfall reads, so the two plots stay recognisably one
    family. Its marker colours are reused for the two METHODS here (up-colour =
    pyKinaXe, down-colour = KRSA) rather than for a direction. Geometry and font
    sizes are not taken from it: that config sizes a screen figure that is
    scaled afterwards, which makes print text unreadable.
    """
    ranked = (
        df.dropna(subset=["Z_pyKinaXe", "Z_KRSA"], how="all")
        .sort_values("Z_pyKinaXe", ascending=True, na_position="first")
        .reset_index(drop=True)
    )
    if ranked.empty:
        return 0.0

    cfg = _load_waterfall_style()
    marker = cfg.get("marker", {})
    stem = cfg.get("stem", {})
    lines = cfg.get("threshold_lines", {})
    axes = cfg.get("axes", {})

    color_pyk = marker.get("color_up", "red")
    color_krsa = marker.get("color_down", "blue")
    marker_size = marker.get("size_significant", 18)
    marker_alpha = marker.get("alpha_significant", 0.9)

    stem_color = stem.get("color", "gray")
    stem_width = stem.get("linewidth", 0.8)
    stem_alpha_strong = stem.get("alpha_significant", 0.9)
    stem_alpha_soft = stem.get("alpha_not_significant", 0.45)

    n = len(ranked)
    positions = list(range(n))

    # Geometry in inches. The width is fixed; the height follows the row count,
    # and the y-tick size is the free variable that keeps the labels from
    # colliding once the page height is exhausted (see the CONFIG block).
    fig_width = PANEL_WIDTH_MM / 25.4
    margin_top = FONT_TITLE * 2.4 / 72
    margin_bottom = (FONT_X_TICK + FONT_AXIS_LABEL) * 2.3 / 72
    margin_right = 0.08

    # The type size is fixed, so the panel HEIGHT follows the family count. That
    # is the whole trick: a tall STK panel and a short PTK one then carry the same
    # type, which is what lets them sit in one column and be read as one figure.
    y_font = FONT_Y_TICK
    row_height = y_font * ROW_SPACING / 72
    fig_height = margin_top + margin_bottom + n * row_height

    longest_label = max(len(str(f)) for f in ranked["Kinase_Family"])
    margin_left = longest_label * y_font * 0.62 / 72 + 0.10

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for y, row in zip(positions, ranked.itertuples(index=False)):
        z_pyk, z_krsa = row.Z_pyKinaXe, row.Z_KRSA
        # Faint stem from zero for each value, plus a strong connector spanning the
        # gap between the two methods.
        for value in (z_pyk, z_krsa):
            if pd.notna(value):
                ax.hlines(y, 0, value, color=stem_color, linewidth=stem_width,
                          alpha=stem_alpha_soft, zorder=1)
        if pd.notna(z_pyk) and pd.notna(z_krsa):
            ax.hlines(y, min(z_pyk, z_krsa), max(z_pyk, z_krsa), color=stem_color,
                      linewidth=stem_width * 2, alpha=stem_alpha_strong, zorder=2)

    ax.scatter(ranked["Z_KRSA"], positions, s=marker_size, color=color_krsa,
               alpha=marker_alpha, linewidths=0, zorder=3)
    ax.scatter(ranked["Z_pyKinaXe"], positions, s=marker_size, color=color_pyk,
               alpha=marker_alpha, linewidths=0, zorder=3)

    ax.axvline(0, color=lines.get("center_color", "black"),
               linestyle=lines.get("center_linestyle", "-"),
               linewidth=lines.get("center_linewidth", 0.9), zorder=1)
    for cutoff in (z_threshold, -z_threshold):
        ax.axvline(cutoff, color=lines.get("cutoff_color", "black"),
                   linestyle=lines.get("cutoff_linestyle", "--"),
                   linewidth=lines.get("cutoff_linewidth", 1.4), zorder=1)

    extreme = max(ranked[["Z_pyKinaXe", "Z_KRSA"]].abs().max(skipna=True))
    if extreme > X_LIMIT:
        print(f"  NOTE: {out_path.name} holds |Z| = {extreme:.2f}, outside the "
              f"shared x range of +/-{X_LIMIT:g} -- it is CLIPPED. Raise X_LIMIT "
              "(it applies to every panel, which is the point).")
    n_ticks = int(round(2 * X_LIMIT / X_TICK_STEP)) + 1
    x_ticks = [-X_LIMIT + i * X_TICK_STEP for i in range(n_ticks)]
    ax.set_xlim(-X_LIMIT, X_LIMIT)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t:.{X_TICK_DECIMALS}f}" for t in x_ticks])
    ax.set_ylim(-0.75, n - 0.25)
    ax.set_yticks(positions)
    ax.set_yticklabels(ranked["Kinase_Family"].tolist(), fontsize=y_font)
    ax.tick_params(axis="y", length=2, pad=1.5)
    ax.tick_params(axis="x", labelsize=FONT_X_TICK)
    ax.set_xlabel("Z-Score", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.grid(axis="x", alpha=axes.get("grid_alpha", 0.25))
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.subplots_adjust(
        left=margin_left / fig_width,
        right=1.0 - margin_right / fig_width,
        top=1.0 - margin_top / fig_height,
        bottom=margin_bottom / fig_height,
    )
    # No bbox_inches="tight": cropping would change the physical width and undo
    # the print sizing above.
    fig.savefig(out_path, dpi=PLOT_DPI, format="png")
    plt.close(fig)
    return fig_height * 25.4


def _draw_legend_png(out_path: Path, z_threshold: float) -> None:
    """Write the waterfall legend as a standalone PNG, cropped to its content.

    The panels carry no legend, so this file is what explains them. It is drawn at
    the same point sizes as the panels, so panel and legend can be placed side by
    side at 100 % without either being rescaled. It also documents the dashed
    verticals, which the panel titles no longer mention.

    Laid out in ONE ROW, so it reads as a strip under (or over) a row of panels.
    """
    cfg = _load_waterfall_style()
    marker = cfg.get("marker", {})
    lines = cfg.get("threshold_lines", {})
    color_pyk = marker.get("color_up", "red")
    color_krsa = marker.get("color_down", "blue")
    marker_size = marker.get("size_significant", 18)

    # scatter() sizes in points^2, Line2D in points of diameter: the sqrt makes the
    # key the SAME dot that is on the panels, not a legend-sized approximation.
    key_size = marker_size ** 0.5
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=color_pyk,
                   markersize=key_size, label=LABEL_PYKINAXE),
        plt.Line2D([], [], marker="o", linestyle="none", color=color_krsa,
                   markersize=key_size, label=LABEL_KRSA),
        plt.Line2D([], [], color=lines.get("cutoff_color", "black"),
                   linestyle=lines.get("cutoff_linestyle", "--"),
                   linewidth=lines.get("cutoff_linewidth", 1.4),
                   label=f"|Z| = {z_threshold:g} (significance threshold)"),
    ]

    # One row, so the legend sits as a strip above or below a row of panels rather
    # than as a column beside them. The canvas is oversized; the tight bounding
    # box below crops it to the legend itself.
    fig = plt.figure(figsize=(9.0, 1.0))
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="center", ncol=len(handles), fontsize=FONT_LEGEND,
               framealpha=1.0, edgecolor="black", handletextpad=0.6,
               borderpad=0.6, columnspacing=1.8)
    fig.savefig(out_path, dpi=PLOT_DPI, format="png", bbox_inches="tight",
                transparent=True)
    plt.close(fig)


def run_dataset(key: str, spec: dict[str, str], z_threshold: float) -> bool:
    """Compare one dataset end to end. Returns False if it could not be read."""
    print(f"\n{'=' * 78}")
    print(f"DATASET  {key}  --  {spec['title']}")
    print(f"{'=' * 78}")

    family_dir = COMPARE_DIR / spec["pykinaxe"]
    krsa_dir = COMPARE_DIR / spec["krsa"]
    out_dir = OUTPUT_DIR / spec["output_subdir"]

    try:
        pyk = load_pykinaxe_families(family_dir)
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        return False
    if not krsa_dir.is_dir():
        print(f"  [ERROR] KRSA folder not found: {krsa_dir}\n"
              "  Produce it with results_KRSA/run_KRSA_on_pykinaxe_input.py.")
        return False
    print(f"  KRSA    : {krsa_dir}")

    family_z, available = build_family_z_table(pyk, krsa_dir, z_threshold)
    if family_z.empty:
        print("  [ERROR] no (array, comparison) pair had results on both sides.")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    waterfall_dir = out_dir / WATERFALL_SUBDIR
    waterfall_dir.mkdir(exist_ok=True)
    # Clear stale plots of an earlier run so the folder cannot mix two runs.
    for stale in waterfall_dir.glob("waterfall_*.png"):
        stale.unlink()

    family_z.to_csv(out_dir / CSV_FAMILY_Z, index=False)
    summary = build_overlap_summary(family_z, available)
    summary.to_csv(out_dir / CSV_OVERLAP, index=False)

    column_height: dict[str, float] = {}
    for (array, comparison), sub in family_z.groupby(["Array", "Comparison"]):
        # Title carries the comparison and the array and nothing else: these panels
        # are meant to be composed into figures, where a repeated "Kinase-family Z
        # (|Z|>=2)" on every panel is noise. The threshold is in the legend PNG.
        height_mm = _draw_family_waterfall(
            sub,
            f"{comparison.replace('_vs_', ' vs ')}  {array}",
            waterfall_dir / f"waterfall_{comparison}_{array}.png",
            z_threshold,
        )
        column_height[comparison] = column_height.get(comparison, 0.0) + height_mm
    _draw_legend_png(waterfall_dir / LEGEND_FILENAME, z_threshold)

    # One COLUMN of the intended page is the STK panel above the PTK panel of the
    # same comparison. Check that stack against the page budget here rather than
    # letting the reader discover it in a layout program.
    tallest = max(column_height.items(), key=lambda kv: kv[1])
    if tallest[1] > COLUMN_HEIGHT_MM:
        print(f"  NOTE: the tallest column ({tallest[0]}) is {tallest[1]:.0f} mm, "
              f"over the {COLUMN_HEIGHT_MM:.0f} mm an A4 column has. Lower "
              "FONT_Y_TICK or put fewer comparisons on the page.")

    stats = agreement_stats(family_z)
    print(f"\n  Significant-family overlap (|Z| >= {z_threshold:g}) -- only "
          "scope=array rows count families:")
    print(summary.to_string(index=False))
    if "r" in stats:
        print(f"\n  Continuous agreement over the {int(stats['n'])} families both "
              f"methods score: r = {stats['r']:.3f}, RMS = {stats['RMS']:.3f}, "
              f"max |diff| = {stats['max_abs_diff']:.3f}")
        print("  r and RMS are threshold-free; the Jaccard above is not.")
    print(f"\n  Wrote {out_dir / CSV_FAMILY_Z} ({len(family_z)} family instances)")
    print(f"  Wrote {out_dir / CSV_OVERLAP}")
    n_panels = len([f for f in waterfall_dir.glob("waterfall_*.png")
                    if f.name != LEGEND_FILENAME])
    print(f"  Wrote {n_panels} waterfall(s) + {LEGEND_FILENAME} to {waterfall_dir}")
    print(f"  Panels are {PANEL_WIDTH_MM:.0f} mm wide at {FONT_Y_TICK:g} pt "
          f"family labels; tallest STK+PTK column {tallest[1]:.0f} mm of "
          f"{COLUMN_HEIGHT_MM:.0f} mm.")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), metavar="NAME",
        default=sorted(DATASETS),
        help=f"which datasets to compare (default: all -- {', '.join(sorted(DATASETS))})",
    )
    parser.add_argument(
        "--z-threshold", type=float, default=Z_THRESHOLD,
        help=f"absolute Z threshold applied to BOTH methods (default: {Z_THRESHOLD:g})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"pyKinaXe vs KRSA, kinase families, |Z| >= {args.z_threshold:g}  "
          f"-- {len(args.datasets)} dataset(s): {', '.join(args.datasets)}")
    ok = [run_dataset(key, DATASETS[key], args.z_threshold) for key in args.datasets]
    failed = [key for key, good in zip(args.datasets, ok) if not good]
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        return 1
    print(f"\nAll output under {OUTPUT_DIR}/<dataset>/ "
          f"-- {CSV_FAMILY_Z}, {CSV_OVERLAP} and {WATERFALL_SUBDIR}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
