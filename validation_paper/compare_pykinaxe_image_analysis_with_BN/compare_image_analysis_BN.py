"""Compare BioNavigator's and pyKinaXe's spot intensities on the same chips.

The script pairs the individual spot images of the same physical chips before
any peptide statistic, kinase mapping or enrichment score exists, for both
datasets in one run (the CDRL vwr-rats-kinome chips as "rat", the
benchmarking/HDV chips as "benchmarking"). NOTE.txt describes the inputs and
how to produce them.

Output per dataset, under output/<dataset>/:
    spot_intensities.csv                 one row per spot image: Array, Barcode,
                                         ArrayRow, Exposure_Time, Cycle, spotRow,
                                         spotCol, Peptide, I_BioNavigator
                                         (Median_SigmBg), I_pyKinaXe (I_median),
                                         Difference, Difference_relative
    spot_intensity_agreement_<ARRAY>.png one figure per array: BioNavigator on x
                                         against pyKinaXe on y, linear axes with a
                                         shared range, and an inset for the low
                                         range (plot.zoom_range or
                                         plot.zoom_percentile)

The agreement statistics (Pearson, Spearman, Lin's CCC, OLS slope, median
ratio, share within the tolerance factor) are printed to the terminal.

Inputs are configured per dataset in config_compare_image_analysis.yaml: the
BioNavigator crosstab exports (signal column Median_SigmBg, columns prefixed
ds0./ds1./ds2../js0.; PTK as prewash plus afterwash) and pyKinaXe's
<timestamp>_Export_image_analysis_<ARRAY>_bn.csv (signal column I_median). Both
ship in this folder under exports_BN/ and exports_pykinaxe/. Relative paths are
resolved against this folder first and the repository root second, and an
entry may contain a wildcard. With inputs.pykinaxe.source: run_engine the
pyKinaXe side is computed live from the raw chip images named under
inputs.pykinaxe.engine instead of read from the export.

Spots are paired on (Barcode, Row, spotRow, spotCol, Exposure Time, Cycle, ID)
with an outer join, so a spot recorded by one side only stays in the list with
an empty value. The script also pairs on position alone, once as exported and
once with one side's spotRow/spotCol swapped, and exits non-zero if the swapped
pairing correlates better; this catches a transposed spot grid, which no other
check in the repository detects, because both exports label every position
with the same peptide.

Run:
    python validation_paper/compare_pykinaxe_image_analysis_with_BN/compare_image_analysis_BN.py
    --config <file>   another config with the same format (a `datasets:` mapping)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is fixed)
import numpy as np
import pandas as pd
import yaml

COMPARE_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMPARE_DIR.parents[1]

CONFIG_PATH = COMPARE_DIR / "config_compare_image_analysis.yaml"

BN_SIGNAL_COLUMN = "Median_SigmBg"
PYK_SIGNAL_COLUMN = "I_median"

OUTPUT_COLUMNS = [
    "Array", "Barcode", "ArrayRow", "Exposure_Time", "Cycle",
    "spotRow", "spotCol", "Peptide",
    "I_BioNavigator", "I_pyKinaXe", "Difference", "Difference_relative",
]
SORT_COLUMNS = ["Array", "Barcode", "ArrayRow", "Exposure_Time", "Cycle",
                "spotRow", "spotCol"]

# Defaults for ONE entry of `datasets:`. Only the inputs have to be named: the
# output subfolder falls back to the dataset's own key and the title to nothing.
DATASET_DEFAULTS: dict = {
    "title": None,
    "output_subdir": None,
    "inputs": {"bn": {}, "pykinaxe": {}},
}

# Everything except `datasets` has a usable default; `datasets` cannot, because
# no input path can be guessed. `output` is the ROOT folder -- each dataset gets
# a subfolder of it.
CONFIG_DEFAULTS: dict = {
    "datasets": {},
    "output": "output",
    "pairing": {
        "arrays": ["PTK", "STK"],
        "keys": ["Barcode", "Row", "spotRow", "spotCol", "Exposure Time", "Cycle", "ID"],
        "drop_ids": ["#REF"],
        "duplicate_handling": "mean",
    },
    "report": {"check_orientation": True},
    "plot": {
        "enabled": True,
        "filename_pattern": "spot_intensity_agreement_{array}.png",
        "dpi": 300,
        "figsize": [6.3, 6.3],
        "inset_bounds": [0.07, 0.55, 0.40, 0.40],
        "margin": 0.03,
        "zoom_range": [-100.0, 150.0],
        "zoom_percentile": 95.0,
        "tolerance_factor": 2.0,
        "shift": 10.0,
        "point_size": 4.0,
        "point_alpha": 0.15,
        "font_size": 11.0,
    },
}


def _merge_config(defaults: dict, override: dict) -> dict:
    """Return ``defaults`` with ``override`` merged in, one level deep per section."""
    merged = {key: (dict(value) if isinstance(value, dict) else value)
              for key, value in defaults.items()}
    for section, values in (override or {}).items():
        if section in merged and isinstance(values, dict) and isinstance(merged[section], dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


def load_config() -> dict:
    """Load the config on top of CONFIG_DEFAULTS and check that it names datasets.

    A missing or dataset-less file is not usable: the defaults carry the pairing,
    the report and the plot settings, but they cannot carry input paths, and
    without ``datasets:`` there is nothing to compare. The old single-dataset
    format (a top-level ``inputs:``) is named explicitly in the message, because
    that is what every config in this folder looked like before.
    """
    try:
        with CONFIG_PATH.open() as handle:
            user_config = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        raise SystemExit(
            f"Config not found: {CONFIG_PATH}\n"
            f"It names the input paths of every dataset, and nothing can guess those."
        )
    config = _merge_config(CONFIG_DEFAULTS, user_config)
    if not config["datasets"]:
        hint = ""
        if "inputs" in user_config:
            hint = ("\n\nThis file is in the OLD single-dataset format (a top-level "
                    "'inputs:').\nEvery dataset now lives under 'datasets:', which is "
                    "what lets one run\nproduce all of them into subfolders of "
                    "'output:':\n\n"
                    "    datasets:\n"
                    "      rat:\n"
                    "        output_subdir: \"rat\"\n"
                    "        inputs:\n"
                    "          bn: {...}\n"
                    "          pykinaxe: {...}\n")
        raise SystemExit(f"{CONFIG_PATH.name} defines no datasets.{hint}")
    return config


def dataset_config(name: str, config: dict) -> tuple[dict, str]:
    """Return (run config, output subfolder) for ONE dataset.

    A dataset entry carries only its own inputs; pairing, report and plot are
    shared by all of them. Folding the shared sections in here is what lets every
    function below keep taking a single ``config`` and stay unaware that there is
    more than one dataset -- the multi-dataset loop lives in ``main`` alone.
    """
    try:
        entry = config["datasets"][name] or {}
    except KeyError:
        raise SystemExit(
            f"{CONFIG_PATH.name} has no dataset '{name}'. It defines: "
            f"{', '.join(config['datasets']) or '(none)'}."
        )
    dataset = _merge_config(DATASET_DEFAULTS, entry)
    inputs = dataset["inputs"] or {}
    if not inputs.get("bn") or not inputs.get("pykinaxe"):
        raise SystemExit(
            f"Dataset '{name}' needs both inputs.bn and inputs.pykinaxe in "
            f"{CONFIG_PATH.name}."
        )
    run_config = {
        "title": str(dataset["title"] or name),
        "inputs": inputs,
        "pairing": config["pairing"],
        "report": config["report"],
        "plot": config["plot"],
    }
    return run_config, str(dataset["output_subdir"] or name)


def _resolve_inputs(pattern: str) -> list[Path]:
    """Resolve one configured input entry to the file(s) it names.

    BOTH sides ship inside this folder -- the BioNavigator exports under
    exports_BN/ and pyKinaXe's own under exports_pykinaxe/, one subfolder per
    dataset -- so a relative entry is looked up in THIS folder first and against
    the repository root second. The second lookup is what still allows pointing
    the config straight at a run output under results/ without copying it here.
    An absolute path is used as given, and a relative path that exists in
    neither place is returned as the
    repository-root candidate, so the not-found message names the location the
    config most likely meant.

    An entry containing a wildcard is expanded and may match SEVERAL files,
    sorted by name. That is what lets the config name pyKinaXe's export without
    repeating the timestamp of the run that wrote it: drop a fresh run's chip
    folder into that dataset's folder under exports_pykinaxe/ and the pattern
    finds it. A wildcard matching nothing returns an empty list, which load_side
    reports.
    """
    candidate = Path(pattern)
    if any(char in pattern for char in "*?["):
        if candidate.is_absolute():
            anchor = Path(candidate.anchor)
            return sorted(anchor.glob(str(candidate.relative_to(anchor))))
        return sorted(COMPARE_DIR.glob(pattern)) or sorted(REPO_ROOT.glob(pattern))
    if candidate.is_absolute():
        return [candidate]
    local = COMPARE_DIR / candidate
    return [local if local.exists() else REPO_ROOT / candidate]


def _strip_bn_prefix(columns) -> list[str]:
    """Drop the BioNavigator ``dsN./jsN.`` export prefixes from column names.

    The prefix is everything up to the last dot; BN also writes doubled dots
    (``ds2..spotRow``), which this handles for free. No plain BN column name
    contains a dot.
    """
    return [str(col).rsplit(".", 1)[-1].strip() for col in columns]


def load_side(paths, signal_column: str, label: str, config: dict) -> pd.DataFrame:
    """Read and normalise one side (BN or pyKinaXe) for one array.

    Raises when the signal column is absent, because that almost always means the
    two sides were swapped in the config -- the file names do not distinguish
    them, only this column does.
    """
    pairing = config["pairing"]
    keys = list(pairing["keys"])

    frames = []
    for entry in paths:
        matches = _resolve_inputs(entry)
        if not matches:
            raise FileNotFoundError(
                f"{label} input matched no file: {entry}\n"
                f"Fix inputs in {CONFIG_PATH.name} (see NOTE.txt). Wildcards are "
                f"expanded in {COMPARE_DIR.name}/ first, then in the repository "
                f"root."
            )
        for full in matches:
            if not full.exists():
                raise FileNotFoundError(
                    f"{label} input not found: {full}\n"
                    f"Fix inputs in {CONFIG_PATH.name} (see NOTE.txt). Relative "
                    f"paths are looked up in {COMPARE_DIR.name}/ first, then in "
                    f"the repository root."
                )
            frame = pd.read_csv(full, low_memory=False)
            frame.columns = _strip_bn_prefix(frame.columns)
            if signal_column not in frame.columns:
                raise KeyError(
                    f"{full.name} has no '{signal_column}' column, so it is not "
                    f"a {label} export. BioNavigator writes "
                    f"'{BN_SIGNAL_COLUMN}', pyKinaXe writes "
                    f"'{PYK_SIGNAL_COLUMN}'. Check {CONFIG_PATH.name}: the "
                    f"file NAME does not tell them apart."
                )
            missing = [key for key in keys if key not in frame.columns]
            if missing:
                raise KeyError(f"{full.name} is missing pairing keys {missing}.")
            frames.append(frame[keys + [signal_column]])
            print(f"    {label}: {full.name}  ({len(frame)} rows)")

    merged = pd.concat(frames, ignore_index=True)
    return _normalize_side(merged, signal_column, label, config)


def _normalize_side(frame: pd.DataFrame, signal_column: str, label: str,
                    config: dict) -> pd.DataFrame:
    """Normalise one side's raw table into the paired ``keys + [label]`` form.

    Shared by the file-reading path (``load_side``) and the live-engine path
    (``run_pykinaxe_engine``): ``frame`` only has to carry the pairing keys and
    ``signal_column``. Drops the configured control IDs, coerces the keys to
    numbers (ID stays a stripped string), renames the signal to ``label`` and
    collapses any rows sharing a full key so the outer join stays one-to-one.
    """
    pairing = config["pairing"]
    keys = list(pairing["keys"])

    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise KeyError(f"{label} data is missing pairing keys {missing}.")
    if signal_column not in frame.columns:
        raise KeyError(
            f"{label} data has no '{signal_column}' column. BioNavigator writes "
            f"'{BN_SIGNAL_COLUMN}', pyKinaXe writes '{PYK_SIGNAL_COLUMN}'."
        )

    merged = frame[keys + [signal_column]].copy()
    merged = merged[~merged["ID"].astype(str).isin(pairing["drop_ids"])]
    for key in keys:
        if key == "ID":
            merged[key] = merged[key].astype(str).str.strip()
        else:
            merged[key] = pd.to_numeric(merged[key], errors="coerce")
    merged = merged.dropna(subset=keys).rename(columns={signal_column: label})

    # Replicate rows sharing the full key are collapsed; leaving them in would
    # make the pairing a many-to-many join and silently multiply the row count.
    duplicated = int(merged.duplicated(keys).sum())
    if duplicated:
        if pairing["duplicate_handling"] == "first":
            merged = merged.drop_duplicates(keys, keep="first")
        else:
            merged = merged.groupby(keys, as_index=False)[label].mean()
        print(f"      {duplicated} rows shared a full key; collapsed "
              f"({pairing['duplicate_handling']}).")
    return merged


def _resolve_engine_path(path_str: str) -> Path:
    """Resolve a raw-data path for the engine (absolute, or relative to repo root)."""
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def run_pykinaxe_engine(array: str, config: dict) -> pd.DataFrame:
    """Run pyKinaXe's image-analysis engine on one array's raw chip.

    Returns the engine's ``final_output_bn`` -- the same BN-format per-spot table
    the pipeline writes as ``*_Export_image_analysis_<ARRAY>_bn.csv`` -- so the
    y-axis of the comparison reflects the CURRENT engine rather than a stored
    export. Only the pyKinaXe side is computed here; BioNavigator stays
    file-read. The heavy engine imports are done lazily so ``read_export`` mode
    -- which is what both shipped datasets use -- needs none of them, and the
    engine writes its usual export under ``results/`` as any run does (the
    comparison uses the in-memory table regardless).
    """
    engine_cfg = config["inputs"]["pykinaxe"].get("engine", {})
    if array not in engine_cfg:
        raise KeyError(
            f"pykinaxe.engine has no entry for array '{array}'. Add "
            f"engine.{array}.data_dir and engine.{array}.subfolder, or set "
            f"pykinaxe.source: read_export."
        )
    spec = engine_cfg[array]
    data_dir = _resolve_engine_path(spec["data_dir"])
    subfolder = str(spec["subfolder"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"pyKinaXe engine data_dir for {array} not found: {data_dir}\n"
            f"Fix inputs.pykinaxe.engine.{array}.data_dir in the config."
        )

    # Import the engine only when actually running it (numba etc. are heavy).
    for import_dir in (REPO_ROOT, REPO_ROOT / "src"):
        if str(import_dir) not in sys.path:
            sys.path.insert(0, str(import_dir))
    from kx_data_importer import DataLoader
    from kx_image_processor import ImageProcessor

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"    pyKinaXe engine [{array}]: analysing {data_dir.name} / {subfolder}")
    loader = DataLoader(
        data_dir=str(data_dir),
        experiment_name=data_dir.name,
        subfolder=subfolder,
        timestamp=timestamp,
    )
    loader.load_data(
        annotation_verbose=False, layout_verbose=False, images_verbose=False
    )
    processor = ImageProcessor(loader)
    processor.process()

    frame = processor.final_output_bn
    if frame is None or len(frame) == 0:
        raise RuntimeError(
            f"pyKinaXe engine produced an empty final_output_bn for {array}."
        )
    print(f"    pyKinaXe engine [{array}]: {len(frame):,} spot rows computed.")
    return frame


def pair_array(array: str, config: dict):
    """Outer-join one array's spots; returns (paired, bn, pyk).

    BioNavigator (x) is always read from files. The pyKinaXe side (y) is either
    read from a stored ``_bn`` export or computed live by the engine, per
    ``inputs.pykinaxe.source``.
    """
    keys = list(config["pairing"]["keys"])
    bn = load_side(config["inputs"]["bn"][array], BN_SIGNAL_COLUMN, "BN", config)

    pyk_cfg = config["inputs"]["pykinaxe"]
    source = str(pyk_cfg.get("source", "read_export")).lower()
    if source == "run_engine":
        pyk = _normalize_side(run_pykinaxe_engine(array, config),
                              PYK_SIGNAL_COLUMN, "pyKinaXe", config)
    elif source == "read_export":
        pyk = load_side(pyk_cfg[array], PYK_SIGNAL_COLUMN, "pyKinaXe", config)
    else:
        raise ValueError(
            f"Unknown inputs.pykinaxe.source '{source}'. "
            f"Use 'run_engine' or 'read_export'."
        )

    paired = bn.merge(pyk, on=keys, how="outer")
    paired.insert(0, "Array", array)
    return paired, bn, pyk


def build_list(paired: pd.DataFrame) -> pd.DataFrame:
    """Turn one array's paired spots into the exported row format."""
    listing = pd.DataFrame({
        "Array": paired["Array"],
        "Barcode": paired["Barcode"].astype("Int64"),
        "ArrayRow": paired["Row"].astype("Int64"),
        "Exposure_Time": paired["Exposure Time"].astype("Int64"),
        "Cycle": paired["Cycle"].astype("Int64"),
        "spotRow": paired["spotRow"].astype("Int64"),
        "spotCol": paired["spotCol"].astype("Int64"),
        "Peptide": paired["ID"],
        "I_BioNavigator": paired["BN"],
        "I_pyKinaXe": paired["pyKinaXe"],
    })
    listing["Difference"] = listing["I_BioNavigator"] - listing["I_pyKinaXe"]
    # Relative to pyKinaXe, floored at one count: the median spot carries only a
    # few counts, so an unfloored ratio would explode in the noise floor.
    listing["Difference_relative"] = (
        listing["Difference"] / listing["I_pyKinaXe"].abs().clip(lower=1.0)
    )
    return listing[OUTPUT_COLUMNS]


def check_orientation(array: str, bn: pd.DataFrame, pyk: pd.DataFrame,
                      config: dict) -> bool:
    """Verify that the two sides agree on the ORIENTATION of the spot grid.

    Pairs on POSITION alone, as exported and with one side's spotRow/spotCol
    swapped, and compares the two correlations. "as exported" must win. See the
    module docstring for why this check exists and why nothing else catches it.
    """
    position_keys = [key for key in config["pairing"]["keys"] if key != "ID"]
    direct = bn.merge(pyk, on=position_keys, suffixes=("_bn", "_pyk"))
    swapped = bn.rename(columns={"spotRow": "spotCol", "spotCol": "spotRow"})
    transposed = swapped.merge(pyk, on=position_keys, suffixes=("_bn", "_pyk"))

    if len(direct) < 3 or len(transposed) < 3:
        print(f"    {array}: too few paired spots for the orientation check.")
        return True

    r_direct = float(direct["BN"].corr(direct["pyKinaXe"]))
    r_transposed = float(transposed["BN"].corr(transposed["pyKinaXe"]))
    on_diagonal = direct["spotRow"] == direct["spotCol"]
    r_diagonal = float(direct.loc[on_diagonal, "BN"].corr(direct.loc[on_diagonal, "pyKinaXe"]))
    r_off = float(direct.loc[~on_diagonal, "BN"].corr(direct.loc[~on_diagonal, "pyKinaXe"]))

    ok = r_direct > r_transposed
    print(f"    orientation {'OK' if ok else '*** TRANSPOSED ***'}: "
          f"as exported r={r_direct:.4f}, grid transposed r={r_transposed:.4f} "
          f"(diagonal {r_diagonal:.4f}, off-diagonal {r_off:.4f})")
    if not ok:
        print("        One side reads the pixels of the mirrored spot. Both label\n"
              "        the positions identically, so every row still pairs and the\n"
              "        list is silently wrong off the diagonal. See NOTE.txt.")
    return ok


# ---------------------------------------------------------------------------
# Agreement plot: BioNavigator (x) against pyKinaXe (y)
# ---------------------------------------------------------------------------

def agreement_stats(bn_values: np.ndarray, pyk_values: np.ndarray,
                    plot_config: dict) -> dict:
    """Quantify how close the two tools' values are on one array.

    Correlation alone is not enough here: two tools can correlate perfectly and
    still sit on a line other than the identity. Reported together are

        pearson       linear agreement on the raw counts, dominated by the
                      brightest spots
        pearson_log   the same on log10 of the counts, floored at one count, so
                      the three orders of magnitude below saturation count too
        spearman      rank agreement; depressed by BN's integer rounding in the
                      noise floor, so read it next to the other two
        ccc           Lin's concordance correlation coefficient -- correlation
                      penalised for any departure from the identity line, which
                      is the "are the VALUES the same" number
        slope/offset  ordinary least squares fit pyKinaXe = slope*BN + offset
        median_log2   median of log2((pyK+shift)/(BN+shift)); 0 means neither
                      tool reads systematically higher
        within        share of spots inside the tolerance band

    The ratio is taken on values shifted by ``plot.shift`` counts and floored at
    one count. Without that, the noise floor -- where BN's rounding to integers
    can produce any ratio, including a division by zero -- would decide both the
    median and the share.
    """
    shift = float(plot_config["shift"])
    factor = float(plot_config["tolerance_factor"])

    bn_mean, pyk_mean = bn_values.mean(), pyk_values.mean()
    bn_var, pyk_var = bn_values.var(), pyk_values.var()
    covariance = float(np.mean((bn_values - bn_mean) * (pyk_values - pyk_mean)))
    denominator = bn_var + pyk_var + (bn_mean - pyk_mean) ** 2
    ccc = 2.0 * covariance / denominator if denominator > 0 else float("nan")

    log_bn = np.log10(np.clip(bn_values, 1.0, None))
    log_pyk = np.log10(np.clip(pyk_values, 1.0, None))
    slope, offset = np.polyfit(bn_values, pyk_values, 1)

    log2_ratio = np.log2(np.clip(pyk_values + shift, 1.0, None)
                         / np.clip(bn_values + shift, 1.0, None))
    within = float(np.mean(np.abs(log2_ratio) <= np.log2(factor)))

    return {
        "n": int(bn_values.size),
        "pearson": float(np.corrcoef(bn_values, pyk_values)[0, 1]),
        "pearson_log": float(np.corrcoef(log_bn, log_pyk)[0, 1]),
        "spearman": float(pd.Series(bn_values).corr(pd.Series(pyk_values),
                                                    method="spearman")),
        "ccc": float(ccc),
        "slope": float(slope),
        "offset": float(offset),
        "median_log2": float(np.median(log2_ratio)),
        "within": within,
        "log2_ratio": log2_ratio,
    }


def _panel_limits(bn_values: np.ndarray, pyk_values: np.ndarray,
                  plot_config: dict, percentile: float | None) -> tuple[float, float]:
    """Return one shared (low, high) range for both axes of a panel.

    Both axes get the SAME range so the identity line is the 45-degree diagonal
    and the eye can read a departure from it as a departure between the tools.
    ``percentile`` None means the full range of the array; a number cuts the
    upper end at that percentile of both tools' values, which is what makes the
    crowded low range readable on a linear axis. The lower end is always the
    smallest value either tool reported -- BN's negative noise-floor values are
    part of the comparison, not an outlier to be hidden.
    """
    low = float(min(bn_values.min(), pyk_values.min()))
    if percentile is None:
        high = float(max(bn_values.max(), pyk_values.max()))
    else:
        high = float(max(np.percentile(bn_values, percentile),
                         np.percentile(pyk_values, percentile)))
    return _with_margin(low, high, plot_config)


def _with_margin(low: float, high: float, plot_config: dict) -> tuple[float, float]:
    """Pad a (low, high) range by plot.margin on each side."""
    if high <= low:
        high = low + 1.0
    margin = float(plot_config.get("margin", 0.03)) * (high - low)
    return low - margin, high + margin


def _zoom_limits(bn_values: np.ndarray, pyk_values: np.ndarray,
                 plot_config: dict) -> tuple[float, float] | None:
    """Return the inset's (low, high), or None when no inset is wanted.

    plot.zoom_range gives the SAME window to every array, which is what makes
    two arrays' insets comparable at a glance -- read one, and the other's
    numbers mean the same thing. plot.zoom_percentile instead picks the window
    from each array's own distribution, so the insets cover the same FRACTION of
    the spots but different values; it is the fallback when no range is set.
    Either way plot.margin is added, and the window is not clipped to the data:
    a fixed range says what it shows even if one array has nothing at its edge.
    """
    window = plot_config.get("zoom_range")
    if window:
        return _with_margin(float(window[0]), float(window[1]), plot_config)
    percentile = plot_config.get("zoom_percentile")
    if percentile:
        return _panel_limits(bn_values, pyk_values, plot_config, float(percentile))
    return None


def _draw_scatter(axis, bn_values, pyk_values, plot_config: dict,
                  limits: tuple[float, float], label_axes: bool = True) -> None:
    """Draw the spots and the identity line into one axes.

    Both axes are LINEAR and span ``limits``, the same range on x and y, so the
    identity line is the 45-degree diagonal and a departure between the tools is
    a departure from that line. Points outside the range are clipped by the axes
    rather than dropped, so no panel hides a spot that exists -- it only stops
    showing where exactly it sits.
    """
    base = float(plot_config.get("font_size", 11.0))
    low, high = limits
    # The identity line and the grid are tied to the font size so that raising
    # it for a half-width placement does not leave hairlines under large type.
    axis.scatter(bn_values, pyk_values, s=float(plot_config["point_size"]),
                 alpha=float(plot_config["point_alpha"]), color="#1f4e79",
                 linewidths=0, rasterized=True, clip_on=True)
    axis.plot([low, high], [low, high], color="black", lw=base * 0.09)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25, lw=base * 0.045)
    # The inset is a fraction of the panel and its tick labels sit INSIDE it, so
    # they take a smaller size than the main panel's, which sit outside. Both are
    # derived from plot.font_size; see the config for what that means on paper.
    axis.tick_params(labelsize=base - 1 if label_axes else base - 5)
    if label_axes:
        axis.set_xlabel("Spot intensities BioNavigator", fontsize=base)
        axis.set_ylabel("Spot intensities pyKinaXe", fontsize=base)


def _draw_array_figure(array: str, bn_values: np.ndarray, pyk_values: np.ndarray,
                       stats: dict, plot_config: dict):
    """Build the figure for ONE array; returns (figure, inset limits or None).

    One panel, because that is the comparison: BN on x, pyKinaXe on y, the whole
    range of the array. The inset exists because the axes are linear -- the few
    hundred bright spots set the range, so the tens of thousands of dim ones,
    which is where the two tools actually differ, would otherwise be one blob in
    the corner. Same data, same axes, only the range differs; the rectangle in
    the main panel marks what the inset shows.
    """
    width, height = plot_config["figsize"]
    figure, axis = plt.subplots(figsize=(width, height))

    full = _panel_limits(bn_values, pyk_values, plot_config, None)
    _draw_scatter(axis, bn_values, pyk_values, plot_config, full)
    # The array and nothing else. The dataset is the output folder, the spot
    # count is in the terminal and in the CSV, and that the axes are linear over
    # one shared range is what the axes themselves say.
    axis.set_title(array,
                   fontsize=float(plot_config.get("font_size", 11.0)) + 1)

    limits = _zoom_limits(bn_values, pyk_values, plot_config)
    if limits:
        inset = axis.inset_axes(list(plot_config["inset_bounds"]),
                                xlim=limits, ylim=limits)
        _draw_scatter(inset, bn_values, pyk_values, plot_config, limits,
                      label_axes=False)
        # The connector lines are what makes the inset readable as a zoom rather
        # than as a second, unrelated panel.
        axis.indicate_inset_zoom(inset, edgecolor="#555555", lw=0.8, alpha=0.9)

    figure.tight_layout()
    return figure, limits


def write_agreement_plots(listing: pd.DataFrame, output_dir: Path,
                          config: dict) -> tuple[dict, dict]:
    """Write ONE figure per array; returns (path per array, stats per array).

    One file per array rather than one shared figure: PTK and STK are different
    chips with different ranges and different agreement, and putting them side by
    side invited reading one array's picture as the other's.

    Only spots BOTH sides recorded can be drawn: the CSV keeps the one-sided
    rows, the figure cannot use them. That count and the agreement NUMBERS are
    printed to the terminal, not written into the image, which carries the array
    name and the spots and nothing else.
    """
    plot_config = config["plot"]
    if not plot_config.get("enabled", True):
        print("\nAgreement plots: disabled (plot.enabled = false).")
        return {}, {}

    arrays = [array for array in config["pairing"]["arrays"]
              if array in set(listing["Array"])]
    both = listing["I_BioNavigator"].notna() & listing["I_pyKinaXe"].notna()
    usable = [array for array in arrays
              if int((both & (listing["Array"] == array)).sum()) >= 3]
    if not usable:
        print("\nAgreement plots: skipped, no array has enough two-sided spots.")
        return {}, {}

    paths, stats_per_array = {}, {}
    print()
    for array in usable:
        subset = listing[both & (listing["Array"] == array)]
        bn_values = subset["I_BioNavigator"].to_numpy(dtype=float)
        pyk_values = subset["I_pyKinaXe"].to_numpy(dtype=float)
        stats = agreement_stats(bn_values, pyk_values, plot_config)
        stats_per_array[array] = stats

        figure, zoom = _draw_array_figure(array, bn_values, pyk_values, stats,
                                          plot_config)
        out_path = output_dir / plot_config["filename_pattern"].format(array=array)
        figure.savefig(out_path, dpi=int(plot_config["dpi"]))
        plt.close(figure)
        paths[array] = out_path

        print(f"Wrote: {out_path}")
        print(f"  {array}: n={stats['n']:,}  Pearson {stats['pearson']:.4f} "
              f"(log {stats['pearson_log']:.4f})  Spearman {stats['spearman']:.4f}  "
              f"CCC {stats['ccc']:.4f}  slope {stats['slope']:.3f}  "
              f"median ratio {2 ** stats['median_log2']:.3f}x  "
              f"within {float(plot_config['tolerance_factor']):g}x "
              f"{stats['within'] * 100:.1f}%")
        if zoom:
            # A fixed inset window covers a different share of each array, so
            # say which -- otherwise the inset looks equally representative of
            # both when it is not.
            inside = float(np.mean((bn_values >= zoom[0]) & (bn_values <= zoom[1])
                                   & (pyk_values >= zoom[0]) & (pyk_values <= zoom[1])))
            print(f"    inset {zoom[0]:.0f} to {zoom[1]:.0f}: "
                  f"{inside * 100:.1f}% of the spots")
    return paths, stats_per_array


def process_dataset(name: str, config: dict, output_dir: Path) -> bool:
    """Run the whole comparison for ONE dataset; returns the orientation verdict.

    Writes exactly three files into ``output_dir``: the per-spot list and the PTK
    and STK agreement plots. ``config`` is the dataset's own run config -- its
    inputs plus the shared pairing/report/plot sections -- so nothing below this
    function knows that there is more than one dataset.

    The orientation verdict is RETURNED rather than raised on, so that a second
    dataset still runs and gets its plots when the first one fails the check;
    ``main`` exits non-zero at the end.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}")
    print(f"DATASET '{name}': {config['title']}")
    print(f"  output: {output_dir}")
    print("=" * 78)

    listings = []
    orientation_ok = True
    for array in config["pairing"]["arrays"]:
        print(f"  {array}:")
        paired, bn, pyk = pair_array(array, config)
        print(f"    {len(paired)} rows; "
              f"{int((paired['BN'].notna() & paired['pyKinaXe'].notna()).sum())} on both "
              f"sides, {int(paired['pyKinaXe'].isna().sum())} BN only, "
              f"{int(paired['BN'].isna().sum())} pyKinaXe only")
        if config["report"].get("check_orientation", True):
            orientation_ok &= check_orientation(array, bn, pyk, config)
        listings.append(build_list(paired))

    listing = (pd.concat(listings, ignore_index=True)
               .sort_values(SORT_COLUMNS, kind="mergesort")
               .reset_index(drop=True))

    out_path = output_dir / "spot_intensities.csv"
    listing.to_csv(out_path, index=False, float_format="%.4f")

    both = listing["Difference"].notna()
    print(f"\nWrote: {out_path}")
    print(f"  {len(listing):,} rows ({out_path.stat().st_size / 1e6:.1f} MB), "
          f"{int(both.sum()):,} of them with both values")
    print(f"  median difference {listing.loc[both, 'Difference'].median():+.2f} counts, "
          f"median relative {listing.loc[both, 'Difference_relative'].median():+.3f}")
    print("  columns: " + ", ".join(OUTPUT_COLUMNS))

    write_agreement_plots(listing, output_dir, config)
    return bool(orientation_ok)


def main() -> None:
    """Compare EVERY dataset in the config, each into a subfolder of its own.

    One invocation covers both shipped datasets -- that is the point of the
    ``datasets:`` mapping: the rat and the benchmarking chips are always compared
    with the same pairing keys and the same plot settings, and reading two
    outputs that were produced by two different invocations of two different
    configs was the way to get that silently wrong.

    ``--config <file>`` runs the same comparison over another config. The path is
    rebound on the module rather than threaded through as an argument so that
    every error message keeps naming the file actually being read.
    """
    global CONFIG_PATH
    argv = sys.argv[1:]
    if "--config" in argv:
        given = Path(argv[argv.index("--config") + 1])
        if not given.is_absolute():
            given = (COMPARE_DIR / given) if (COMPARE_DIR / given).exists() \
                    else (REPO_ROOT / given)
        CONFIG_PATH = given
        print(f"Config: {CONFIG_PATH}\n")
    config = load_config()
    root_dir = COMPARE_DIR / config["output"]
    names = list(config["datasets"])

    print("Spot intensities: BioNavigator (Median_SigmBg) vs pyKinaXe (I_median).")
    print(f"  paired on: {', '.join(config['pairing']['keys'])}")
    print(f"  arrays:    {', '.join(config['pairing']['arrays'])}")
    print(f"  datasets:  {', '.join(names)}  ->  {root_dir}/<dataset>/")

    transposed = []
    for name in names:
        run_config, subdir = dataset_config(name, config)
        if not process_dataset(name, run_config, root_dir / subdir):
            transposed.append(name)

    written = ["spot_intensities.csv"]
    if config["plot"].get("enabled", True):
        written += [config["plot"]["filename_pattern"].format(array=array)
                    for array in config["pairing"]["arrays"]]
    print(f"\n{'=' * 78}")
    print(f"Done: {len(names)} dataset(s) written to {root_dir}/")
    for name in names:
        print(f"  {dataset_config(name, config)[1]}/: " + ", ".join(written))

    if transposed:
        raise SystemExit(
            f"\nORIENTATION CHECK FAILED for: {', '.join(transposed)}; those lists "
            f"are wrong off the grid diagonal.\nFix the spot-grid orientation before "
            f"using them (see NOTE.txt)."
        )


if __name__ == "__main__":
    main()
