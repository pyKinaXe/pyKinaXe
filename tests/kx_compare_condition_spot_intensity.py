#!/usr/bin/env python3
"""
Compare control-normalized spot signals between Test1, Test2, and Test3.

The script reads the exported PTK/STK spot-level image analysis tables from a
Kerthi result directory, merges them with the matching data_enrichment table,
and writes a small report with:

- one combined spot-level table for the selected conditions
- sample-level mean relative intensities
- condition-level summaries
- exposure/cycle summaries
- sample-level noise estimates
- pairwise Welch t-tests on sample-level means
- a quick visualization of sample-level mean relative intensities

By default the script uses the latest paired PTK/STK result directories that
correspond to the HDV Aug24 raw-data folders and the full
`*_Export_image_analysis_PTK.csv` / `*_Export_image_analysis_STK.csv` tables.
It uses `I_median` by default and removes:

- `#REF` rows
- STK pre-phosphorylated peptides (`ID` starts with `p`)

Use `--use-filtered` to restrict the analysis to the filtered export tables that
are closer to the downstream peptide workflow.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path
import sys
import tempfile


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-pykinaxe"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache-pykinaxe"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_EXPERIMENT_SUFFIX = "Kerthi_HDV_Aug24"
DEFAULT_RAW_DATA_ROOT = REPO_ROOT / "data/HDV/09.2024 L, SHDAg and full genome/Raw data and measurements"
DEFAULT_CHIP_RAW_DIRS = {
    "PTK": DEFAULT_RAW_DATA_ROOT / "PTK 2 Aug24",
    "STK": DEFAULT_RAW_DATA_ROOT / "STK 2 Aug24",
}
DEFAULT_CONDITIONS = ("Test1", "Test2", "Test3")
DEFAULT_METRIC = "I_median"
CHIP_TYPES = ("PTK", "STK")


def find_latest_result_dir(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    experiment_suffix: str = DEFAULT_EXPERIMENT_SUFFIX,
) -> Path:
    """Find latest result dir.
    
    Args:
        results_root (Path): Path-like value for results root.
        experiment_suffix (str): Experiment suffix used by this function.
    
    Returns:
        Path: Found latest result dir.
    """
    candidates = sorted(
        path
        for path in results_root.glob(f"*_{experiment_suffix}")
        if path.is_dir() and any(path.glob("**/*_Export_image_analysis_PTK.csv"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No result directories matching '*_{experiment_suffix}' were found under {results_root}."
        )
    return candidates[-1]


def find_latest_chip_result_dirs(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    chip_raw_dirs: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Find latest chip result dirs.
    
    Args:
        results_root (Path): Path-like value for results root.
        chip_raw_dirs (dict[str, Path] | None): Path-like value for chip raw dirs.
    
    Returns:
        dict[str, Path]: Found latest chip result dirs.
    """
    chip_raw_dirs = chip_raw_dirs or DEFAULT_CHIP_RAW_DIRS
    resolved: dict[str, Path] = {}

    for chip_type in CHIP_TYPES:
        raw_dir = chip_raw_dirs.get(chip_type)
        if raw_dir is None:
            raise FileNotFoundError(f"No raw-data directory was configured for chip type {chip_type}.")

        result_suffix = raw_dir.name
        candidates = sorted(
            path
            for path in results_root.glob(f"*_{result_suffix}")
            if path.is_dir() and any(path.glob(f"**/*_Export_image_analysis_{chip_type}.csv"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No result directories matching '*_{result_suffix}' were found under {results_root} "
                f"for raw-data directory {raw_dir}."
            )
        resolved[chip_type] = candidates[-1]

    return resolved


def infer_timestamp(results_dir: Path) -> str:
    """Infer the timestamp encoded in a result path or filename.
    
    Args:
        results_dir (Path): Directory containing or receiving the results.
    
    Returns:
        str: Inferred timestamp.
    """
    ukas = sorted(results_dir.glob("*_UKA_output_Control_Test*.xlsx"))
    if ukas:
        return ukas[0].name.split("_UKA_output_", 1)[0]

    exports = sorted(results_dir.glob("**/*_Export_image_analysis_*.csv"))
    if not exports:
        raise FileNotFoundError(
            f"Could not infer timestamp from UKA or export-image files in {results_dir}."
        )
    return exports[0].name.split("_Export_image_analysis_", 1)[0]


def resolve_export_path(results_dir: Path, chip_type: str, use_filtered: bool) -> Path:
    """Resolve export path.
    
    Args:
        results_dir (Path): Directory containing or receiving the results.
        chip_type (str): Chip type used by this function.
        use_filtered (bool): Boolean flag controlling whether to use filtered.
    
    Returns:
        Path: Resolved export path.
    """
    suffix = f"_Export_image_analysis_{chip_type}_filtered.csv" if use_filtered else f"_Export_image_analysis_{chip_type}.csv"
    matches = sorted(path for path in results_dir.glob(f"**/*{suffix}") if "_bn.csv" not in path.name)
    if not matches:
        raise FileNotFoundError(
            f"Could not find {chip_type} export-image file with suffix '{suffix}' in {results_dir}."
        )
    if len(matches) > 1:
        print(
            f"WARNING: found multiple {chip_type} export files in {results_dir}; using {matches[0].name}",
            file=sys.stderr,
        )
    return matches[0]


def resolve_enrichment_path(export_path: Path) -> Path:
    """Resolve enrichment path.
    
    Args:
        export_path (Path): Path to the export.
    
    Returns:
        Path: Resolved enrichment path.
    """
    sibling_matches = sorted(export_path.parent.glob("*_data_enrichment.csv"))
    if sibling_matches:
        return sibling_matches[0]

    root_matches = sorted(export_path.parents[1].glob("**/*_data_enrichment.csv"))
    if root_matches:
        return root_matches[0]

    raise FileNotFoundError(f"Could not find a data_enrichment.csv next to {export_path}.")


def load_spot_table(
    export_path: Path,
    chip_type: str,
    metric: str,
    conditions_to_keep: tuple[str, ...],
) -> pd.DataFrame:
    """Load spot table.
    
    Args:
        export_path (Path): Path to the export.
        chip_type (str): Chip type used by this function.
        metric (str): Metric used by this function.
        conditions_to_keep (tuple[str, ...]): Conditions to keep processed by this function.
    
    Returns:
        pd.DataFrame: Loaded spot table.
    """
    enrichment_path = resolve_enrichment_path(export_path)

    df_export = pd.read_csv(export_path)
    df_enrichment = pd.read_csv(enrichment_path)

    metric_aliases = {
        "I_median": ("I_median", "Median_SigmBg", "Intensity"),
        "Median_SigmBg": ("Median_SigmBg", "I_median", "Intensity"),
        "Intensity": ("Intensity",),
        "MaxIntensity": ("MaxIntensity",),
    }
    metric_candidates = metric_aliases.get(metric, (metric,))
    resolved_metric = next((candidate for candidate in metric_candidates if candidate in df_export.columns), None)
    if resolved_metric is None:
        raise ValueError(
            f"Metric '{metric}' not found in {export_path.name}. Available metric-like columns: "
            f"{[col for col in ('I_median', 'Median_SigmBg', 'Intensity', 'MaxIntensity') if col in df_export.columns]}"
        )

    enrichment_cols = [
        "Barcode",
        "Row",
        "Sample name",
        "Construct",
        "Type",
        "Supergroup",
        "Test Condition",
        "Biological Replicate",
        "Technical Replicate",
    ]
    available_enrichment_cols = [col for col in enrichment_cols if col in df_enrichment.columns]

    df = df_export.merge(
        df_enrichment[available_enrichment_cols].drop_duplicates(subset=["Barcode", "Row"]),
        on=["Barcode", "Row"],
        how="left",
        validate="many_to_one",
    )

    before_filters = len(df)
    ref_removed = int((df["ID"].astype(str) == "#REF").sum())
    df = df[df["ID"].astype(str) != "#REF"].copy()

    prephospho_removed = 0
    if chip_type == "STK":
        is_prephospho = df["ID"].astype(str).str.startswith("p", na=False)
        prephospho_removed = int(is_prephospho.sum())
        df = df[~is_prephospho].copy()

    df["ChipType"] = chip_type
    df["Metric"] = pd.to_numeric(df[resolved_metric], errors="coerce")
    df["MetricColumn"] = resolved_metric
    df = df[df["Test Condition"].isin(conditions_to_keep)].copy()
    df = df[df["Metric"].notna()].copy()

    print(
        f"{chip_type}: kept {len(df)}/{before_filters} rows after filtering "
        f"({ref_removed} #REF rows removed, {prephospho_removed} pre-phosphorylated rows removed). "
        f"Using metric column '{resolved_metric}'."
    )

    return df


def subtract_control_intensities(
    df_spots: pd.DataFrame,
    test_conditions: tuple[str, ...],
) -> pd.DataFrame:
    """Convert absolute spot intensities into test-minus-control intensities.
    
    Control baselines are matched within the same chip type, peptide/spot
    context, and replicate context when available.
    
    Args:
        df_spots (pd.DataFrame): Input pandas DataFrame containing spots.
        test_conditions (tuple[str, ...]): Test conditions processed by this function.
    
    Returns:
        pd.DataFrame: Subtract control intensities.
    """
    control_rows = df_spots[df_spots["Test Condition"] == "Control"].copy()
    test_rows = df_spots[df_spots["Test Condition"].isin(test_conditions)].copy()

    if control_rows.empty:
        raise ValueError(
            "No 'Control' rows were found, so relative-to-control intensities cannot be computed."
        )

    key_candidates = [
        "ChipType",
        "ID",
        "Exposure Time",
        "Cycle",
        "Biological Replicate",
        "Technical Replicate",
    ]
    join_keys = [col for col in key_candidates if col in df_spots.columns]
    if not join_keys:
        raise ValueError(
            "Could not identify any columns for matching tests to control intensities."
        )

    control_lookup = (
        control_rows.groupby(join_keys, dropna=False)
        .agg(
            control_metric=("Metric", "mean"),
            n_control_matches=("Metric", "size"),
        )
        .reset_index()
    )

    merged = test_rows.merge(
        control_lookup,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )

    missing_control = int(merged["control_metric"].isna().sum())
    if missing_control:
        missing_preview = (
            merged.loc[merged["control_metric"].isna(), join_keys]
            .drop_duplicates()
            .head(10)
        )
        raise ValueError(
            "Some test rows could not be matched to corresponding control intensities. "
            f"Missing rows: {missing_control}. Example keys:\n{missing_preview.to_string(index=False)}"
        )

    merged["AbsoluteMetric"] = merged["Metric"]
    merged["Metric"] = merged["Metric"] - merged["control_metric"]
    merged["ControlMetric"] = merged["control_metric"]
    merged["MetricType"] = "relative_to_control"
    merged = merged.drop(columns=["control_metric"])
    return merged


def build_mean_sample_difference_summary(
    df_spots_absolute: pd.DataFrame,
    test_conditions: tuple[str, ...],
) -> pd.DataFrame:
    """Average test and control spot intensities across the two samples first,
    then subtract mean(test) - mean(control), and finally integrate by chip.
    
    Args:
        df_spots_absolute (pd.DataFrame): Input pandas DataFrame containing spots absolute.
        test_conditions (tuple[str, ...]): Test conditions processed by this function.
    
    Returns:
        pd.DataFrame: Constructed mean sample difference summary.
    """
    key_candidates = [
        "ChipType",
        "ID",
        "Exposure Time",
        "Cycle",
    ]
    join_keys = [col for col in key_candidates if col in df_spots_absolute.columns]
    if not join_keys:
        raise ValueError(
            "Could not identify spot-defining columns for mean-sample difference summary."
        )

    control_rows = df_spots_absolute[df_spots_absolute["Test Condition"] == "Control"].copy()
    test_rows = df_spots_absolute[df_spots_absolute["Test Condition"].isin(test_conditions)].copy()

    control_means = (
        control_rows.groupby(join_keys, dropna=False)
        .agg(mean_control_metric=("Metric", "mean"))
        .reset_index()
    )

    test_means = (
        test_rows.groupby(join_keys + ["Test Condition"], dropna=False)
        .agg(mean_test_metric=("Metric", "mean"))
        .reset_index()
    )

    merged = test_means.merge(
        control_means,
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    if merged["mean_control_metric"].isna().any():
        raise ValueError("Some mean test spot intensities could not be matched to mean control spot intensities.")

    merged["mean_difference"] = merged["mean_test_metric"] - merged["mean_control_metric"]
    merged["abs_mean_difference"] = merged["mean_difference"].abs()

    chip_summary = (
        merged.groupby(["ChipType", "Test Condition"], dropna=False)
        .agg(
            n_spot_keys=("mean_difference", "size"),
            integrated_mean_difference=("mean_difference", "sum"),
            integrated_abs_mean_difference=("abs_mean_difference", "sum"),
            mean_of_mean_differences=("mean_difference", "mean"),
            median_of_mean_differences=("mean_difference", "median"),
        )
        .reset_index()
    )

    all_summary = (
        chip_summary.groupby("Test Condition", dropna=False)
        .agg(
            n_spot_keys=("n_spot_keys", "sum"),
            integrated_mean_difference=("integrated_mean_difference", "sum"),
            integrated_abs_mean_difference=("integrated_abs_mean_difference", "sum"),
            mean_of_mean_differences=("mean_of_mean_differences", "sum"),
            median_of_mean_differences=("median_of_mean_differences", "sum"),
        )
        .reset_index()
    )
    all_summary["ChipType"] = "ALL"

    summary = pd.concat([chip_summary, all_summary], ignore_index=True)
    chip_order = {"ALL": 0, "PTK": 1, "STK": 2}
    condition_order = {condition: idx for idx, condition in enumerate(test_conditions)}
    summary["chip_order"] = summary["ChipType"].map(chip_order).fillna(99).astype(int)
    summary["condition_order"] = summary["Test Condition"].map(condition_order).fillna(999).astype(int)
    summary = summary.sort_values(["chip_order", "condition_order"]).drop(
        columns=["chip_order", "condition_order"]
    )
    return summary.reset_index(drop=True)


def build_sample_summary(df_spots: pd.DataFrame) -> pd.DataFrame:
    """Build sample summary.
    
    Args:
        df_spots (pd.DataFrame): Input pandas DataFrame containing spots.
    
    Returns:
        pd.DataFrame: Constructed sample summary.
    """
    group_cols = [
        "ChipType",
        "Test Condition",
        "Barcode",
        "Row",
        "Sample name",
        "Construct",
        "Biological Replicate",
        "Technical Replicate",
    ]
    available_group_cols = [col for col in group_cols if col in df_spots.columns]

    summary = (
        df_spots.groupby(available_group_cols, dropna=False)
        .agg(
            n_spots=("Metric", "size"),
            n_unique_peptides=("ID", "nunique"),
            integrated_intensity=("Metric", "sum"),
            integrated_abs_intensity=("Metric", lambda s: np.abs(s.to_numpy(dtype=float)).sum()),
            integrated_positive_intensity=("Metric", lambda s: s[s > 0].sum()),
            integrated_negative_intensity=("Metric", lambda s: s[s < 0].sum()),
            mean_intensity=("Metric", "mean"),
            median_intensity=("Metric", "median"),
            std_intensity=("Metric", "std"),
            min_intensity=("Metric", "min"),
            max_intensity=("Metric", "max"),
        )
        .reset_index()
    )
    return summary


def build_condition_summary(
    df_spots: pd.DataFrame,
    sample_summary: pd.DataFrame,
    conditions: tuple[str, ...],
) -> pd.DataFrame:
    """Build condition summary.
    
    Args:
        df_spots (pd.DataFrame): Input pandas DataFrame containing spots.
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
        conditions (tuple[str, ...]): Conditions processed by this function.
    
    Returns:
        pd.DataFrame: Constructed condition summary.
    """
    spot_summary = (
        df_spots.groupby(["ChipType", "Test Condition"], dropna=False)
        .agg(
            spot_n=("Metric", "size"),
            spot_mean=("Metric", "mean"),
            spot_median=("Metric", "median"),
            spot_std=("Metric", "std"),
        )
        .reset_index()
    )

    sample_level = (
        sample_summary.groupby(["ChipType", "Test Condition"], dropna=False)
        .agg(
            sample_n=("integrated_intensity", "size"),
            sample_mean_of_integrals=("integrated_intensity", "mean"),
            sample_median_of_integrals=("integrated_intensity", "median"),
            sample_std_of_integrals=("integrated_intensity", "std"),
            sample_mean_of_abs_integrals=("integrated_abs_intensity", "mean"),
            sample_median_of_abs_integrals=("integrated_abs_intensity", "median"),
            sample_std_of_abs_integrals=("integrated_abs_intensity", "std"),
            sample_mean_of_positive_integrals=("integrated_positive_intensity", "mean"),
            sample_median_of_positive_integrals=("integrated_positive_intensity", "median"),
            sample_std_of_positive_integrals=("integrated_positive_intensity", "std"),
            sample_mean_of_negative_integrals=("integrated_negative_intensity", "mean"),
            sample_median_of_negative_integrals=("integrated_negative_intensity", "median"),
            sample_std_of_negative_integrals=("integrated_negative_intensity", "std"),
            sample_mean_of_medians=("median_intensity", "mean"),
        )
        .reset_index()
    )

    summary = spot_summary.merge(
        sample_level,
        on=["ChipType", "Test Condition"],
        how="outer",
    )
    summary["sample_sem_of_integrals"] = summary["sample_std_of_integrals"] / np.sqrt(
        summary["sample_n"].clip(lower=1)
    )

    combined_spots = df_spots.copy()
    combined_spots["ChipType"] = "ALL"
    combined_samples = sample_summary.copy()
    combined_samples["ChipType"] = "ALL"

    combined = build_condition_summary_single_scope(combined_spots, combined_samples)
    summary = pd.concat([summary, combined], ignore_index=True)

    order = {condition: idx for idx, condition in enumerate(conditions)}
    summary["condition_order"] = summary["Test Condition"].map(order).fillna(999).astype(int)
    summary["chip_order"] = summary["ChipType"].map({"ALL": 0, "PTK": 1, "STK": 2}).fillna(99).astype(int)
    summary = summary.sort_values(["chip_order", "condition_order"]).drop(columns=["condition_order", "chip_order"])
    return summary.reset_index(drop=True)


def build_condition_summary_single_scope(
    df_spots: pd.DataFrame,
    sample_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build condition summary single scope.
    
    Args:
        df_spots (pd.DataFrame): Input pandas DataFrame containing spots.
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
    
    Returns:
        pd.DataFrame: Constructed condition summary single scope.
    """
    spot_summary = (
        df_spots.groupby(["ChipType", "Test Condition"], dropna=False)
        .agg(
            spot_n=("Metric", "size"),
            spot_mean=("Metric", "mean"),
            spot_median=("Metric", "median"),
            spot_std=("Metric", "std"),
        )
        .reset_index()
    )
    sample_level = (
        sample_summary.groupby(["ChipType", "Test Condition"], dropna=False)
        .agg(
            sample_n=("integrated_intensity", "size"),
            sample_mean_of_integrals=("integrated_intensity", "mean"),
            sample_median_of_integrals=("integrated_intensity", "median"),
            sample_std_of_integrals=("integrated_intensity", "std"),
            sample_mean_of_abs_integrals=("integrated_abs_intensity", "mean"),
            sample_median_of_abs_integrals=("integrated_abs_intensity", "median"),
            sample_std_of_abs_integrals=("integrated_abs_intensity", "std"),
            sample_mean_of_positive_integrals=("integrated_positive_intensity", "mean"),
            sample_median_of_positive_integrals=("integrated_positive_intensity", "median"),
            sample_std_of_positive_integrals=("integrated_positive_intensity", "std"),
            sample_mean_of_negative_integrals=("integrated_negative_intensity", "mean"),
            sample_median_of_negative_integrals=("integrated_negative_intensity", "median"),
            sample_std_of_negative_integrals=("integrated_negative_intensity", "std"),
            sample_mean_of_medians=("median_intensity", "mean"),
        )
        .reset_index()
    )
    summary = spot_summary.merge(sample_level, on=["ChipType", "Test Condition"], how="outer")
    summary["sample_sem_of_integrals"] = summary["sample_std_of_integrals"] / np.sqrt(
        summary["sample_n"].clip(lower=1)
    )
    return summary


def build_exposure_cycle_summary(df_spots: pd.DataFrame) -> pd.DataFrame:
    """Build exposure cycle summary.
    
    Args:
        df_spots (pd.DataFrame): Input pandas DataFrame containing spots.
    
    Returns:
        pd.DataFrame: Constructed exposure cycle summary.
    """
    group_cols = ["ChipType", "Test Condition"]
    if "Exposure Time" in df_spots.columns:
        group_cols.append("Exposure Time")
    if "Cycle" in df_spots.columns:
        group_cols.append("Cycle")

    return (
        df_spots.groupby(group_cols, dropna=False)
        .agg(
            spot_n=("Metric", "size"),
            mean_intensity=("Metric", "mean"),
            median_intensity=("Metric", "median"),
            std_intensity=("Metric", "std"),
        )
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def build_sample_noise_summary(sample_summary: pd.DataFrame) -> pd.DataFrame:
    """Build sample noise summary.
    
    Args:
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
    
    Returns:
        pd.DataFrame: Constructed sample noise summary.
    """
    columns = [
        "ChipType",
        "Test Condition",
        "Barcode",
        "Row",
        "Sample name",
        "Biological Replicate",
        "Technical Replicate",
        "n_spots",
        "std_intensity",
        "mean_intensity",
        "integrated_intensity",
        "integrated_abs_intensity",
        "integrated_positive_intensity",
        "integrated_negative_intensity",
    ]
    available_columns = [col for col in columns if col in sample_summary.columns]
    noise_summary = sample_summary[available_columns].copy()
    noise_summary = noise_summary.rename(
        columns={"std_intensity": "noise_std", "mean_intensity": "noise_center_mean"}
    )
    return noise_summary


def select_monotonic_spots(
    df_spots: pd.DataFrame,
    conditions: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select monotonic spots.
    
    Args:
        df_spots (pd.DataFrame): Input pandas DataFrame containing spots.
        conditions (tuple[str, ...]): Conditions processed by this function.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Selected monotonic spots.
    """
    if len(conditions) != 3:
        raise ValueError(
            "Monotonic trend selection currently expects exactly three conditions."
        )

    monotonic_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    spot_key_cols = ["ID", "Exposure Time", "Cycle"]

    for chip_type in CHIP_TYPES:
        chip_df = df_spots[df_spots["ChipType"] == chip_type].copy()
        condition_means = (
            chip_df.groupby(spot_key_cols + ["Test Condition"], dropna=False)
            .agg(mean_metric=("Metric", "mean"))
            .reset_index()
        )
        wide = condition_means.pivot_table(
            index=spot_key_cols,
            columns="Test Condition",
            values="mean_metric",
        )
        wide = wide.dropna(subset=list(conditions))

        monotonic_mask = (
            (wide[conditions[0]] <= wide[conditions[1]])
            & (wide[conditions[1]] <= wide[conditions[2]])
        )
        monotonic_keys = wide.loc[monotonic_mask].reset_index()[spot_key_cols]
        monotonic_chip_df = chip_df.merge(monotonic_keys, on=spot_key_cols, how="inner")
        monotonic_frames.append(monotonic_chip_df)

        summary_rows.append(
            {
                "ChipType": chip_type,
                "n_total_spot_keys": int(len(wide)),
                "n_monotonic_increasing_spot_keys": int(monotonic_mask.sum()),
                "fraction_monotonic_increasing": float(monotonic_mask.mean()) if len(wide) else np.nan,
            }
        )

    monotonic_spots = pd.concat(monotonic_frames, ignore_index=True)
    monotonic_summary = pd.DataFrame(summary_rows)
    return monotonic_spots, monotonic_summary


def build_pairwise_tests(sample_summary: pd.DataFrame, conditions: tuple[str, ...]) -> pd.DataFrame:
    """Build pairwise tests.
    
    Args:
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
        conditions (tuple[str, ...]): Conditions processed by this function.
    
    Returns:
        pd.DataFrame: Constructed pairwise tests.
    """
    rows: list[dict[str, object]] = []

    scope_frames = [("ALL", sample_summary)]
    for chip_type in CHIP_TYPES:
        scope_frames.append((chip_type, sample_summary[sample_summary["ChipType"] == chip_type].copy()))

    for scope_name, scope_df in scope_frames:
        vectors = {
            condition: scope_df.loc[
                scope_df["Test Condition"] == condition, "integrated_intensity"
            ].dropna().astype(float).to_numpy()
            for condition in conditions
        }

        valid_vectors = [vector for vector in vectors.values() if len(vector) >= 2]
        anova_stat = np.nan
        anova_p = np.nan
        if len(valid_vectors) >= 2:
            anova_stat, anova_p = f_oneway(*valid_vectors)

        for cond_a, cond_b in itertools.combinations(conditions, 2):
            values_a = vectors[cond_a]
            values_b = vectors[cond_b]
            t_stat = np.nan
            p_value = np.nan
            if len(values_a) >= 2 and len(values_b) >= 2:
                t_stat, p_value = ttest_ind(values_a, values_b, equal_var=False)

            rows.append(
                {
                    "Scope": scope_name,
                    "Condition_A": cond_a,
                    "Condition_B": cond_b,
                    "n_samples_A": len(values_a),
                    "n_samples_B": len(values_b),
                    "mean_A": float(np.mean(values_a)) if len(values_a) else np.nan,
                    "mean_B": float(np.mean(values_b)) if len(values_b) else np.nan,
                    "mean_difference_A_minus_B": (
                        float(np.mean(values_a) - np.mean(values_b))
                        if len(values_a) and len(values_b)
                        else np.nan
                    ),
                    "welch_t_stat": float(t_stat) if not pd.isna(t_stat) else np.nan,
                    "welch_p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                    "anova_f_stat_scope": float(anova_stat) if not pd.isna(anova_stat) else np.nan,
                    "anova_p_value_scope": float(anova_p) if not pd.isna(anova_p) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def plot_sample_means(
    sample_summary: pd.DataFrame,
    metric_label: str,
    conditions: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot sample means.
    
    Args:
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
        metric_label (str): Metric label used by this function.
        conditions (tuple[str, ...]): Conditions processed by this function.
        output_path (Path): Path to the output.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    all_scope = build_cross_chip_sample_sums(
        sample_summary=sample_summary,
        value_column="integrated_intensity",
    )
    scopes = [
        ("ALL", all_scope),
        ("PTK", sample_summary[sample_summary["ChipType"] == "PTK"].copy()),
        ("STK", sample_summary[sample_summary["ChipType"] == "STK"].copy()),
    ]

    colors = {
        "Test1": "#1f4e79",
        "Test2": "#c0392b",
        "Test3": "#4d4d4d",
    }

    fig, axes = plt.subplots(len(scopes), 1, figsize=(10, 12), constrained_layout=True)

    for ax, (scope_name, scope_df) in zip(np.ravel(axes), scopes):
        means = []
        sems = []
        for idx, condition in enumerate(conditions):
            values = scope_df.loc[
                scope_df["Test Condition"] == condition, "integrated_intensity"
            ].dropna().astype(float).to_numpy()
            if len(values):
                means.append(float(np.mean(values)))
                sems.append(float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0)
            else:
                means.append(np.nan)
                sems.append(np.nan)

            if len(values):
                if len(values) == 1:
                    jitter = np.array([0.0])
                else:
                    jitter = np.linspace(-0.08, 0.08, len(values))
                ax.scatter(
                    np.full(len(values), idx, dtype=float) + jitter,
                    values,
                    s=45,
                    color=colors.get(condition, "#333333"),
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3,
                )

        x = np.arange(len(conditions), dtype=float)
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[colors.get(condition, "#777777") for condition in conditions],
            alpha=0.35,
            edgecolor="black",
            linewidth=1.0,
            capsize=4,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(f"Integrated {metric_label} difference")
        if scope_name == "ALL":
            ax.set_title(f"{scope_name}: sample-level integrated relative intensity (PTK + STK sum)")
        else:
            ax.set_title(f"{scope_name}: sample-level integrated relative intensity")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(f"Condition comparison using integrated {metric_label} relative to control", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sample_noise(
    sample_summary: pd.DataFrame,
    metric_label: str,
    conditions: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot sample noise.
    
    Args:
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
        metric_label (str): Metric label used by this function.
        conditions (tuple[str, ...]): Conditions processed by this function.
        output_path (Path): Path to the output.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    scopes = [
        ("ALL", sample_summary),
        ("PTK", sample_summary[sample_summary["ChipType"] == "PTK"].copy()),
        ("STK", sample_summary[sample_summary["ChipType"] == "STK"].copy()),
    ]

    colors = {
        "Test1": "#1f4e79",
        "Test2": "#c0392b",
        "Test3": "#4d4d4d",
    }

    fig, axes = plt.subplots(len(scopes), 1, figsize=(10, 12), constrained_layout=True)

    for ax, (scope_name, scope_df) in zip(np.ravel(axes), scopes):
        means = []
        sems = []
        for idx, condition in enumerate(conditions):
            values = scope_df.loc[
                scope_df["Test Condition"] == condition, "std_intensity"
            ].dropna().astype(float).to_numpy()
            if len(values):
                means.append(float(np.mean(values)))
                sems.append(float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0)
            else:
                means.append(np.nan)
                sems.append(np.nan)

            if len(values):
                jitter = np.array([0.0]) if len(values) == 1 else np.linspace(-0.08, 0.08, len(values))
                ax.scatter(
                    np.full(len(values), idx, dtype=float) + jitter,
                    values,
                    s=45,
                    color=colors.get(condition, "#333333"),
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3,
                )

        x = np.arange(len(conditions), dtype=float)
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[colors.get(condition, "#777777") for condition in conditions],
            alpha=0.35,
            edgecolor="black",
            linewidth=1.0,
            capsize=4,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(f"{metric_label} noise (SD)")
        ax.set_title(f"{scope_name}: sample-level noise after control subtraction")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Noise estimate from spot-level {metric_label} differences relative to control",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_cross_chip_sample_sums(
    sample_summary: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Build cross chip sample sums.
    
    Args:
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
        value_column (str): Value column used by this function.
    
    Returns:
        pd.DataFrame: Constructed cross chip sample sums.
    """
    group_cols = [
        "Test Condition",
        "Sample name",
        "Biological Replicate",
        "Technical Replicate",
    ]
    available_group_cols = [col for col in group_cols if col in sample_summary.columns]
    if not available_group_cols:
        raise ValueError("Could not identify grouping columns for cross-chip sample summation.")

    all_scope = (
        sample_summary.groupby(available_group_cols, dropna=False)
        .agg(
            ChipType=("ChipType", lambda _: "ALL"),
            value=(value_column, "sum"),
        )
        .reset_index()
        .rename(columns={"value": value_column})
    )
    return all_scope


def plot_sample_abs_integrals(
    sample_summary: pd.DataFrame,
    metric_label: str,
    conditions: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot sample abs integrals.
    
    Args:
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
        metric_label (str): Metric label used by this function.
        conditions (tuple[str, ...]): Conditions processed by this function.
        output_path (Path): Path to the output.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    all_scope = build_cross_chip_sample_sums(
        sample_summary=sample_summary,
        value_column="integrated_abs_intensity",
    )
    scopes = [
        ("ALL", all_scope),
        ("PTK", sample_summary[sample_summary["ChipType"] == "PTK"].copy()),
        ("STK", sample_summary[sample_summary["ChipType"] == "STK"].copy()),
    ]

    colors = {
        "Test1": "#1f4e79",
        "Test2": "#c0392b",
        "Test3": "#4d4d4d",
    }

    fig, axes = plt.subplots(len(scopes), 1, figsize=(10, 12), constrained_layout=True)

    for ax, (scope_name, scope_df) in zip(np.ravel(axes), scopes):
        means = []
        sems = []
        for idx, condition in enumerate(conditions):
            values = scope_df.loc[
                scope_df["Test Condition"] == condition, "integrated_abs_intensity"
            ].dropna().astype(float).to_numpy()
            if len(values):
                means.append(float(np.mean(values)))
                sems.append(float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0)
            else:
                means.append(np.nan)
                sems.append(np.nan)

            if len(values):
                jitter = np.array([0.0]) if len(values) == 1 else np.linspace(-0.08, 0.08, len(values))
                ax.scatter(
                    np.full(len(values), idx, dtype=float) + jitter,
                    values,
                    s=45,
                    color=colors.get(condition, "#333333"),
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3,
                )

        x = np.arange(len(conditions), dtype=float)
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[colors.get(condition, "#777777") for condition in conditions],
            alpha=0.35,
            edgecolor="black",
            linewidth=1.0,
            capsize=4,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(f"Integrated |{metric_label} difference|")
        if scope_name == "ALL":
            ax.set_title(f"{scope_name}: sample-level integrated absolute relative intensity (PTK + STK sum)")
        else:
            ax.set_title(f"{scope_name}: sample-level integrated absolute relative intensity")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Condition comparison using integrated absolute {metric_label} differences relative to control",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sample_signed_component(
    sample_summary: pd.DataFrame,
    metric_label: str,
    conditions: tuple[str, ...],
    output_path: Path,
    value_column: str,
    title_suffix: str,
    ylabel: str,
) -> None:
    """Plot sample signed component.
    
    Args:
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
        metric_label (str): Metric label used by this function.
        conditions (tuple[str, ...]): Conditions processed by this function.
        output_path (Path): Path to the output.
        value_column (str): Value column used by this function.
        title_suffix (str): Title suffix used by this function.
        ylabel (str): Ylabel used by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    all_scope = build_cross_chip_sample_sums(
        sample_summary=sample_summary,
        value_column=value_column,
    )
    scopes = [
        ("ALL", all_scope),
        ("PTK", sample_summary[sample_summary["ChipType"] == "PTK"].copy()),
        ("STK", sample_summary[sample_summary["ChipType"] == "STK"].copy()),
    ]

    colors = {
        "Test1": "#1f4e79",
        "Test2": "#c0392b",
        "Test3": "#4d4d4d",
    }

    fig, axes = plt.subplots(len(scopes), 1, figsize=(10, 12), constrained_layout=True)

    for ax, (scope_name, scope_df) in zip(np.ravel(axes), scopes):
        means = []
        sems = []
        for idx, condition in enumerate(conditions):
            values = scope_df.loc[
                scope_df["Test Condition"] == condition, value_column
            ].dropna().astype(float).to_numpy()
            if len(values):
                means.append(float(np.mean(values)))
                sems.append(float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0)
            else:
                means.append(np.nan)
                sems.append(np.nan)

            if len(values):
                jitter = np.array([0.0]) if len(values) == 1 else np.linspace(-0.08, 0.08, len(values))
                ax.scatter(
                    np.full(len(values), idx, dtype=float) + jitter,
                    values,
                    s=45,
                    color=colors.get(condition, "#333333"),
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3,
                )

        x = np.arange(len(conditions), dtype=float)
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[colors.get(condition, "#777777") for condition in conditions],
            alpha=0.35,
            edgecolor="black",
            linewidth=1.0,
            capsize=4,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(ylabel)
        if scope_name == "ALL":
            ax.set_title(f"{scope_name}: {title_suffix} (PTK + STK sum)")
        else:
            ax.set_title(f"{scope_name}: {title_suffix}")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(f"{metric_label} relative to control: {title_suffix}", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_monotonic_intensity(
    sample_summary: pd.DataFrame,
    metric_label: str,
    conditions: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot monotonic intensity.
    
    Args:
        sample_summary (pd.DataFrame): Pandas DataFrame containing sample summary.
        metric_label (str): Metric label used by this function.
        conditions (tuple[str, ...]): Conditions processed by this function.
        output_path (Path): Path to the output.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    all_scope = build_cross_chip_sample_sums(
        sample_summary=sample_summary,
        value_column="integrated_intensity",
    )
    scopes = [
        ("ALL", all_scope),
        ("PTK", sample_summary[sample_summary["ChipType"] == "PTK"].copy()),
        ("STK", sample_summary[sample_summary["ChipType"] == "STK"].copy()),
    ]

    colors = {
        "Test1": "#1f4e79",
        "Test2": "#c0392b",
        "Test3": "#4d4d4d",
    }

    fig, axes = plt.subplots(len(scopes), 1, figsize=(10, 12), constrained_layout=True)

    for ax, (scope_name, scope_df) in zip(np.ravel(axes), scopes):
        means = []
        sems = []
        for idx, condition in enumerate(conditions):
            values = scope_df.loc[
                scope_df["Test Condition"] == condition, "integrated_intensity"
            ].dropna().astype(float).to_numpy()
            if len(values):
                means.append(float(np.mean(values)))
                sems.append(float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0)
            else:
                means.append(np.nan)
                sems.append(np.nan)

            if len(values):
                jitter = np.array([0.0]) if len(values) == 1 else np.linspace(-0.08, 0.08, len(values))
                ax.scatter(
                    np.full(len(values), idx, dtype=float) + jitter,
                    values,
                    s=45,
                    color=colors.get(condition, "#333333"),
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3,
                )

        x = np.arange(len(conditions), dtype=float)
        ax.bar(
            x,
            means,
            yerr=sems,
            color=[colors.get(condition, "#777777") for condition in conditions],
            alpha=0.35,
            edgecolor="black",
            linewidth=1.0,
            capsize=4,
            zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(f"Integrated {metric_label} difference")
        if scope_name == "ALL":
            ax.set_title(f"{scope_name}: monotonic-increasing spot subset (PTK + STK sum)")
        else:
            ax.set_title(f"{scope_name}: monotonic-increasing spot subset")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Integrated {metric_label} relative to control using only monotonic-increasing spots",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mean_sample_difference_summary(
    mean_summary: pd.DataFrame,
    metric_label: str,
    conditions: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot mean sample difference summary.
    
    Args:
        mean_summary (pd.DataFrame): Pandas DataFrame containing mean summary.
        metric_label (str): Metric label used by this function.
        conditions (tuple[str, ...]): Conditions processed by this function.
        output_path (Path): Path to the output.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    scopes = [
        ("ALL", mean_summary[mean_summary["ChipType"] == "ALL"].copy()),
        ("PTK", mean_summary[mean_summary["ChipType"] == "PTK"].copy()),
        ("STK", mean_summary[mean_summary["ChipType"] == "STK"].copy()),
    ]

    colors = {
        "Test1": "#1f4e79",
        "Test2": "#c0392b",
        "Test3": "#4d4d4d",
    }

    fig, axes = plt.subplots(len(scopes), 1, figsize=(10, 11), constrained_layout=True)

    for ax, (scope_name, scope_df) in zip(np.ravel(axes), scopes):
        values = [
            float(
                scope_df.loc[scope_df["Test Condition"] == condition, "integrated_mean_difference"].iloc[0]
            )
            if (scope_df["Test Condition"] == condition).any()
            else np.nan
            for condition in conditions
        ]
        x = np.arange(len(conditions), dtype=float)
        ax.bar(
            x,
            values,
            color=[colors.get(condition, "#777777") for condition in conditions],
            alpha=0.45,
            edgecolor="black",
            linewidth=1.0,
            zorder=2,
        )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel(f"Integrated mean {metric_label} difference")
        if scope_name == "ALL":
            ax.set_title(f"{scope_name}: PTK + STK sum after mean(test) - mean(control)")
        else:
            ax.set_title(f"{scope_name}: mean(test) - mean(control) integrated over spots")
        ax.grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Replicate-mean comparison using integrated {metric_label} differences",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script.
    
    Args:
        None.
    
    Returns:
        argparse.Namespace: Parse args.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare test-minus-control spot intensities between Test1, Test2, and Test3 "
            "using PTK/STK export-image analysis tables."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing result folders. Default: results/",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Explicit combined result directory containing both PTK and STK exports. "
            "If omitted, the script first looks for the latest combined folder and "
            "then falls back to the latest separate PTK/STK result folders that match "
            "the configured raw-data directories."
        ),
    )
    parser.add_argument(
        "--experiment-suffix",
        default=DEFAULT_EXPERIMENT_SUFFIX,
        help=f"Result directory suffix to search for. Default: {DEFAULT_EXPERIMENT_SUFFIX}",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Conditions to include. Default: Test1 Test2 Test3",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Metric column from the export-image tables. Default: I_median",
    )
    parser.add_argument(
        "--use-filtered",
        action="store_true",
        help="Use *_Export_image_analysis_<chip>_filtered.csv instead of the full export tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <combined-results-dir>/spot_intensity_condition_analysis "
            "or, for separate PTK/STK result folders, to "
            "results/<timestamp>_spot_intensity_condition_analysis"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the module as a command-line entry point.
    
    Args:
        None.
    
    Returns:
        None: The command-line entry point runs for its side effects.
    """
    args = parse_args()

    combined_results_dir: Path | None = None
    chip_result_dirs: dict[str, Path]
    if args.results_dir is not None:
        combined_results_dir = args.results_dir
        chip_result_dirs = {chip_type: combined_results_dir for chip_type in CHIP_TYPES}
    else:
        try:
            combined_results_dir = find_latest_result_dir(
                results_root=args.results_root,
                experiment_suffix=args.experiment_suffix,
            )
            chip_result_dirs = {chip_type: combined_results_dir for chip_type in CHIP_TYPES}
        except FileNotFoundError:
            chip_result_dirs = find_latest_chip_result_dirs(results_root=args.results_root)

    for chip_type, results_dir in chip_result_dirs.items():
        if not results_dir.exists():
            raise FileNotFoundError(f"Result directory for {chip_type} does not exist: {results_dir}")

    timestamps = {
        chip_type: infer_timestamp(results_dir)
        for chip_type, results_dir in chip_result_dirs.items()
    }
    timestamp = max(timestamps.values())
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else (
            combined_results_dir / "spot_intensity_condition_analysis"
            if combined_results_dir is not None
            else args.results_root / f"{timestamp}_spot_intensity_condition_analysis"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = tuple(args.conditions)

    if combined_results_dir is not None:
        print(f"Using combined result directory: {combined_results_dir}")
    else:
        for chip_type in CHIP_TYPES:
            print(f"Using {chip_type} result directory: {chip_result_dirs[chip_type]}")
    print(f"Timestamp: {timestamp}")
    print(f"Metric: {args.metric}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Using filtered export tables: {args.use_filtered}")

    spot_tables = []
    export_manifest_rows = []
    for chip_type in CHIP_TYPES:
        results_dir = chip_result_dirs[chip_type]
        export_path = resolve_export_path(results_dir, chip_type, use_filtered=args.use_filtered)
        enrichment_path = resolve_enrichment_path(export_path)
        export_manifest_rows.append(
            {
                "ChipType": chip_type,
                "results_dir": str(results_dir),
                "timestamp": timestamps[chip_type],
                "export_path": str(export_path),
                "enrichment_path": str(enrichment_path),
            }
        )
        spot_tables.append(
            load_spot_table(
                export_path=export_path,
                chip_type=chip_type,
                metric=args.metric,
                conditions_to_keep=("Control",) + conditions,
            )
        )

    spots_absolute = pd.concat(spot_tables, ignore_index=True)
    spots = subtract_control_intensities(spots_absolute, test_conditions=conditions)
    mean_sample_difference_summary = build_mean_sample_difference_summary(
        spots_absolute,
        test_conditions=conditions,
    )
    if spots.empty:
        raise ValueError(
            f"No spot rows were found for conditions {conditions} in the resolved PTK/STK result directories."
        )

    sample_summary = build_sample_summary(spots)
    sample_noise_summary = build_sample_noise_summary(sample_summary)
    monotonic_spots, monotonic_spot_summary = select_monotonic_spots(spots, conditions=conditions)
    monotonic_sample_summary = build_sample_summary(monotonic_spots)
    condition_summary = build_condition_summary(spots, sample_summary, conditions=conditions)
    exposure_cycle_summary = build_exposure_cycle_summary(spots)
    pairwise_tests = build_pairwise_tests(sample_summary, conditions)
    export_manifest = pd.DataFrame(export_manifest_rows)

    spots_path = output_dir / f"{timestamp}_spot_intensity_conditions_spots.csv"
    samples_path = output_dir / f"{timestamp}_spot_intensity_conditions_sample_summary.csv"
    noise_samples_path = output_dir / f"{timestamp}_spot_intensity_conditions_sample_noise_summary.csv"
    abs_samples_path = output_dir / f"{timestamp}_spot_intensity_conditions_absolute_sample_summary.csv"
    positive_samples_path = output_dir / f"{timestamp}_spot_intensity_conditions_positive_sample_summary.csv"
    negative_samples_path = output_dir / f"{timestamp}_spot_intensity_conditions_negative_sample_summary.csv"
    mean_sample_difference_path = output_dir / f"{timestamp}_spot_intensity_conditions_mean_sample_difference_summary.csv"
    monotonic_spot_summary_path = output_dir / f"{timestamp}_spot_intensity_conditions_monotonic_spot_summary.csv"
    monotonic_samples_path = output_dir / f"{timestamp}_spot_intensity_conditions_monotonic_sample_summary.csv"
    condition_path = output_dir / f"{timestamp}_spot_intensity_conditions_summary.csv"
    exposure_cycle_path = output_dir / f"{timestamp}_spot_intensity_conditions_exposure_cycle_summary.csv"
    stats_path = output_dir / f"{timestamp}_spot_intensity_conditions_pairwise_tests.csv"
    manifest_path = output_dir / f"{timestamp}_spot_intensity_conditions_inputs.csv"
    figure_path = output_dir / f"{timestamp}_spot_intensity_conditions_relative_to_control_plot.png"
    noise_figure_path = output_dir / f"{timestamp}_spot_intensity_conditions_noise_plot.png"
    abs_figure_path = output_dir / f"{timestamp}_spot_intensity_conditions_absolute_relative_to_control_plot.png"
    positive_figure_path = output_dir / f"{timestamp}_spot_intensity_conditions_positive_relative_to_control_plot.png"
    negative_figure_path = output_dir / f"{timestamp}_spot_intensity_conditions_negative_relative_to_control_plot.png"
    mean_sample_difference_figure_path = output_dir / f"{timestamp}_spot_intensity_conditions_mean_sample_difference_plot.png"
    monotonic_figure_path = output_dir / f"{timestamp}_spot_intensity_conditions_monotonic_increasing_plot.png"

    spots.to_csv(spots_path, index=False)
    sample_summary.to_csv(samples_path, index=False)
    sample_noise_summary.to_csv(noise_samples_path, index=False)
    sample_summary.to_csv(abs_samples_path, index=False)
    sample_summary.to_csv(positive_samples_path, index=False)
    sample_summary.to_csv(negative_samples_path, index=False)
    mean_sample_difference_summary.to_csv(mean_sample_difference_path, index=False)
    monotonic_spot_summary.to_csv(monotonic_spot_summary_path, index=False)
    monotonic_sample_summary.to_csv(monotonic_samples_path, index=False)
    condition_summary.to_csv(condition_path, index=False)
    exposure_cycle_summary.to_csv(exposure_cycle_path, index=False)
    pairwise_tests.to_csv(stats_path, index=False)
    export_manifest.to_csv(manifest_path, index=False)

    plot_sample_means(
        sample_summary=sample_summary,
        metric_label=args.metric,
        conditions=conditions,
        output_path=figure_path,
    )
    plot_sample_noise(
        sample_summary=sample_summary,
        metric_label=args.metric,
        conditions=conditions,
        output_path=noise_figure_path,
    )
    plot_sample_abs_integrals(
        sample_summary=sample_summary,
        metric_label=args.metric,
        conditions=conditions,
        output_path=abs_figure_path,
    )
    plot_sample_signed_component(
        sample_summary=sample_summary,
        metric_label=args.metric,
        conditions=conditions,
        output_path=positive_figure_path,
        value_column="integrated_positive_intensity",
        title_suffix="sample-level integrated positive relative intensity",
        ylabel=f"Integrated positive {args.metric} difference",
    )
    plot_sample_signed_component(
        sample_summary=sample_summary,
        metric_label=args.metric,
        conditions=conditions,
        output_path=negative_figure_path,
        value_column="integrated_negative_intensity",
        title_suffix="sample-level integrated negative relative intensity",
        ylabel=f"Integrated negative {args.metric} difference",
    )
    plot_mean_sample_difference_summary(
        mean_summary=mean_sample_difference_summary,
        metric_label=args.metric,
        conditions=conditions,
        output_path=mean_sample_difference_figure_path,
    )
    plot_monotonic_intensity(
        sample_summary=monotonic_sample_summary,
        metric_label=args.metric,
        conditions=conditions,
        output_path=monotonic_figure_path,
    )

    print("\nWrote:")
    for path in (
        spots_path,
        samples_path,
        noise_samples_path,
        abs_samples_path,
        positive_samples_path,
        negative_samples_path,
        mean_sample_difference_path,
        monotonic_spot_summary_path,
        monotonic_samples_path,
        condition_path,
        exposure_cycle_path,
        stats_path,
        manifest_path,
        figure_path,
        noise_figure_path,
        abs_figure_path,
        positive_figure_path,
        negative_figure_path,
        mean_sample_difference_figure_path,
        monotonic_figure_path,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
