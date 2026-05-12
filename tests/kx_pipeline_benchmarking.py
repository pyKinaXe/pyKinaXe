"""
Benchmark the image-analysis pipeline against the Kerthi reference datasets.

This script:
1. runs the kinase extraction pipeline
2. runs the staged peptide/kinase/pathway analysis on the processed PTK and STK tables
3. benchmarks selected kinase-analysis conditions against the Kerthi reference kinase sets
"""

from pathlib import Path
import sys
import time

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"

for import_dir in (SRC_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))


from kx_benchmarking import Benchmarking_Heatmap
from kx_kinase_extraction_pipeline import (
    DEFAULT_UKA_KPEA_PARAMS,
    run_image_analysis_pipeline,
    run_uka_analysis,
)


KERTHI_BENCHMARK_SPECS = [
    ("Test1", "results_lhdag.csv"),
    ("Test2", "results_shdag.csv"),
    ("Test3", "results_psvld3.csv"),
]

BENCHMARK_UKA_PARAMS = {
    **DEFAULT_UKA_KPEA_PARAMS,
    "volcano_plot": False,
    "heatmap_plot": False,
    "debugging_print": False,
}

BENCHMARK_ACTIVITY_COLUMNS = [
    "MedianPeptideStatistic",
    "KinaseStatistic",
    "MeanPeptideStatistic",
    "KinaseChange",
]


def _coalesce_benchmark_activity(df_uka):
    """Prefer median kinase statistic values, with fallbacks for older result files.
    
    Args:
        df_uka: UKA result DataFrame processed by this helper.
    
    Returns:
        tuple: Coalesced benchmark activity.
    """
    available_cols = [col for col in BENCHMARK_ACTIVITY_COLUMNS if col in df_uka.columns]
    if not available_cols:
        raise ValueError(
            "Could not find a usable kinase activity column. Expected one of: "
            f"{BENCHMARK_ACTIVITY_COLUMNS}"
        )

    activity = pd.Series(pd.NA, index=df_uka.index, dtype="Float64")
    activity_metric = pd.Series(pd.NA, index=df_uka.index, dtype="object")

    for col in available_cols:
        values = pd.to_numeric(df_uka[col], errors="coerce")
        fill_mask = activity.isna() & values.notna()
        activity.loc[fill_mask] = values.loc[fill_mask]
        activity_metric.loc[fill_mask] = col

    if activity.notna().sum() == 0:
        raise ValueError(
            "Activity columns were present, but all values were missing/non-numeric: "
            f"{available_cols}"
        )

    return activity.astype(float), activity_metric


def prepare_benchmark_input(df_uka):
    """Prepare significant kinase rows for benchmarking, preserving duplicates.
    
    Args:
        df_uka: UKA result DataFrame processed by this helper.
    
    Returns:
        object: Prepared benchmark input.
    """
    rank_score_col = "KRSA_AbsMeanZ" if "KRSA_AbsMeanZ" in df_uka.columns else "KPEA_AbsDominantZ"
    activity, activity_metric = _coalesce_benchmark_activity(df_uka)

    ranked = (
        df_uka.assign(
            Activity=activity,
            ActivityMetric=activity_metric,
            abs_stat=lambda df: df["Activity"].abs(),
        )
        .sort_values(
            ["NegLog10EmpiricalP", rank_score_col, "abs_stat"],
            ascending=[False, False, False],
        )
    )

    output_cols = ["Kinase", "Activity", "ActivityMetric"]
    for col in ["Type", "NumSubstrates", "KPEA_SelectedCutoff"]:
        if col in ranked.columns:
            output_cols.append(col)

    return ranked[ranked["Significant"] & ranked["Activity"].notna()][output_cols]


def _find_condition_key(results, target_suffix):
    """Resolve a target condition such as Test1 from the UKA results dictionary.
    
    Args:
        results: Results processed by this function.
        target_suffix: Target suffix processed by this function.
    
    Returns:
        object: Found condition key.
    """
    matches = [
        key for key in results
        if str(key) == target_suffix or str(key).endswith(f"_{target_suffix}")
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Could not resolve a unique UKA condition for '{target_suffix}'. "
            f"Available conditions: {list(results.keys())}"
        )
    return matches[0]


def run_pipeline_benchmarking(results, analysis_timestamp, experiment_name):
    """Benchmark the staged kinase-analysis output for the configured Kerthi conditions.
    
    Args:
        results: Results processed by this function.
        analysis_timestamp: Timestamp string assigned to the current analysis run.
        experiment_name: Experiment name processed by this function.
    
    Returns:
        object: Run result for pipeline benchmarking.
    """
    benchmark_results = {}
    results_dir = REPO_ROOT / "results" / f"{analysis_timestamp}_{experiment_name}"
    benchmark_data_dir = REPO_ROOT / "data/raw/Kerthi_HDV_Aug24"

    for target_suffix, benchmark_file in KERTHI_BENCHMARK_SPECS:
        condition_key = _find_condition_key(results, target_suffix)
        print(f"Benchmarking condition '{condition_key}' against '{benchmark_file}'")

        test_results = prepare_benchmark_input(results[condition_key]["all_kinases"])
        control_results = pd.read_csv(benchmark_data_dir / benchmark_file)

        benchmark = Benchmarking_Heatmap(
            input_results=test_results,
            input_benchmarking=control_results,
            label_results="pyKinaXe",
            label_benchmarking="Kerthi reference",
            save_dir=results_dir / f"{analysis_timestamp}_benchmarking_{target_suffix}",
            heatmap=True,
            venn_plot=True,
            dpi=150,
            debugging_print=False,
            input_results_all=results[condition_key]["all_kinases"],
            activity_label="Median Kinase Statistic",
        )
        benchmark_results[target_suffix] = benchmark.run()

    return benchmark_results


def main():
    """Run image analysis, UKA, and Kerthi benchmarking as a test workflow.
    
    Args:
        None.
    
    Returns:
        None: The command-line entry point runs for its side effects.
    """
    (
        _loader_PTK,
        _loader_STK,
        enricher,
        processor_PTK,
        processor_STK,
        analysis_timestamp,
        pipeline_duration,
        experiment_name,
        results_parent_relpath,
    ) = run_image_analysis_pipeline()

    results, _uka_analysis, uka_duration = run_uka_analysis(
        enricher=enricher,
        processor_PTK=processor_PTK,
        processor_STK=processor_STK,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=BENCHMARK_UKA_PARAMS,
    )

    if not isinstance(results, dict):
        raise ValueError(
            "Kerthi benchmarking expects multi-condition UKA output as a results dictionary."
        )

    print("\n" + "=" * 80)
    print("Starting Kerthi benchmarking...")
    print("=" * 80)
    tic_benchmark = time.time()

    benchmark_results = run_pipeline_benchmarking(
        results=results,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
    )

    benchmark_duration = time.time() - tic_benchmark
    total_duration = pipeline_duration + uka_duration + benchmark_duration

    print("\n" + "=" * 80)
    print("Kerthi benchmarking COMPLETED")
    print("=" * 80)
    print(f"  Image Analysis Pipeline: {pipeline_duration:.2f}s ({pipeline_duration / 60:.2f} min)")
    print(f"  Upstream Kinase Analysis: {uka_duration:.2f}s ({uka_duration / 60:.2f} min)")
    print(f"  Kerthi Benchmarking: {benchmark_duration:.2f}s ({benchmark_duration / 60:.2f} min)")
    print(f"  TOTAL TIME: {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    print("=" * 80 + "\n")

    return benchmark_results


if __name__ == "__main__":
    main()
