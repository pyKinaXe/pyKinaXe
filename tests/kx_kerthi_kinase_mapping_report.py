#!/usr/bin/env python3
"""
Build a Kerthi kinase mapping report from existing pyKinaXe UKA outputs.

The report helps diagnose why Kerthi reference kinases are not identified as
significant by pyKinaXe. It reuses pyKinaXe's active BLAST + PTM mapping logic and
writes one Excel workbook with:

- kinase_summary: one row per condition and kinase
- mapping_details: one row per kinase-peptide-substrate-source mapping
- parameters: active mapping parameters and input paths
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"

for import_dir in (SRC_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))


from kx_kinase_extraction_pipeline import DEFAULT_UKA_KPEA_PARAMS
from kx_upstream_kinase_analysis import KinaseActivityAnalysis


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_EXPERIMENT_SUFFIX = "Kerthi_HDV_Aug24"
DEFAULT_BENCHMARK_DATA_DIR = REPO_ROOT / "data/raw/Kerthi_HDV_Aug24"

KERTHI_BENCHMARK_SPECS = {
    "Test1": "results_lhdag.csv",
    "Test2": "results_shdag.csv",
    "Test3": "results_psvld3.csv",
}

SUMMARY_NUMERIC_COLS = [
    "kerthi_activity",
    "pykinaxe_num_substrates",
    "pykinaxe_kinase_change",
    "pykinaxe_kinase_statistic",
    "pykinaxe_z_score",
    "pykinaxe_p_value",
    "pykinaxe_fdr",
]

PEPTIDE_VALUE_COLS = [
    "ID",
    "Type",
    "GeneName",
    "UniprotAccession",
    "Sequence",
    "mean_control",
    "mean_treatment",
    "peptide_change",
    "peptide_statistic",
    "p_value",
    "logp_value",
]


def find_latest_kerthi_result_dir(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    experiment_suffix: str = DEFAULT_EXPERIMENT_SUFFIX,
) -> Path:
    """Find latest kerthi result dir.
    
    Args:
        results_root (Path): Path-like value for results root.
        experiment_suffix (str): Experiment suffix used by this function.
    
    Returns:
        Path: Found latest kerthi result dir.
    """
    candidates = []
    for path in results_root.glob(f"*_{experiment_suffix}"):
        if not path.is_dir():
            continue
        if all(
            list(path.glob(f"*_UKA_output_Control_{condition}.xlsx"))
            for condition in KERTHI_BENCHMARK_SPECS
        ):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No Kerthi result directory with Test1-3 UKA workbooks found under {results_root}."
        )

    return sorted(candidates, key=lambda p: p.name)[-1]


def infer_timestamp(results_dir: Path) -> str:
    """Infer the timestamp encoded in a result path or filename.
    
    Args:
        results_dir (Path): Directory containing or receiving the results.
    
    Returns:
        str: Inferred timestamp.
    """
    uka_files = sorted(results_dir.glob("*_UKA_output_Control_Test*.xlsx"))
    if not uka_files:
        raise FileNotFoundError(f"No UKA output workbooks found in {results_dir}.")
    return uka_files[0].name.split("_UKA_output_", 1)[0]


def load_condition_workbook(
    results_dir: Path,
    timestamp: str,
    condition: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load condition workbook.
    
    Args:
        results_dir (Path): Directory containing or receiving the results.
        timestamp (str): Timestamp string associated with the current analysis run.
        condition (str): Condition used by this function.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Loaded condition workbook.
    """
    workbook = results_dir / f"{timestamp}_UKA_output_Control_{condition}.xlsx"
    if not workbook.exists():
        raise FileNotFoundError(f"Missing pyKinaXe UKA workbook: {workbook}")

    peptides = pd.read_excel(workbook, sheet_name="Peptide_Statistics")
    py_all = pd.read_excel(workbook, sheet_name="Kinases_All")
    py_sig = pd.read_excel(workbook, sheet_name="Kinases_Significant")

    for df in (py_all, py_sig):
        if "Kinase" in df.columns:
            df["Kinase"] = df["Kinase"].astype(str).str.strip()

    return peptides, py_all, py_sig


def load_kerthi_reference(
    benchmark_data_dir: Path,
    condition: str,
) -> pd.DataFrame:
    """Load kerthi reference.
    
    Args:
        benchmark_data_dir (Path): Directory containing or receiving the benchmark data.
        condition (str): Condition used by this function.
    
    Returns:
        pd.DataFrame: Loaded kerthi reference.
    """
    if condition not in KERTHI_BENCHMARK_SPECS:
        raise KeyError(f"Unknown Kerthi condition '{condition}'.")

    path = benchmark_data_dir / KERTHI_BENCHMARK_SPECS[condition]
    if not path.exists():
        raise FileNotFoundError(f"Missing Kerthi benchmark file: {path}")

    df = pd.read_csv(path)
    if "Kinase" not in df.columns:
        raise ValueError(f"Kerthi benchmark file must contain a 'Kinase' column: {path}")

    df = df[df["Kinase"].notna()].copy()
    df["Kinase"] = df["Kinase"].astype(str).str.strip()
    if "Activity" in df.columns:
        df["Activity"] = pd.to_numeric(df["Activity"], errors="coerce")

    return df.drop_duplicates(subset="Kinase", keep="first")


def make_analysis(args: argparse.Namespace) -> KinaseActivityAnalysis:
    """Create analysis.
    
    Args:
        args (argparse.Namespace): Args processed by this function.
    
    Returns:
        KinaseActivityAnalysis: Created analysis.
    """
    params = dict(DEFAULT_UKA_KPEA_PARAMS)

    return KinaseActivityAnalysis(
        path_file_enrichment_peptides=params["path_file_enrichment_peptides"],
        BLAST_threshold=args.blast_threshold,
        input_stk_ptm_path=params["input_stk_ptm_path"],
        input_ptk_ptm_path=params["input_ptk_ptm_path"],
        use_verified_interactions_only=args.use_verified_interactions_only,
        verified_evidence_levels=params["verified_evidence_levels"],
        verified_min_score=args.verified_min_score,
        verified_min_references=args.verified_min_references,
        require_known_ptm_site=args.require_known_ptm_site,
        volcano_plot=False,
        volcano_plot_GUI=False,
        volcano_plot_output=None,
        uka_visualization_metric=params["uka_visualization_metric"],
        path_output_peptide_statistic=None,
        debugging_print=args.debugging_print,
        n_permutations=max(100, int(params["n_permutations"])),
        kpea_lfc_cutoffs=params["kpea_lfc_cutoffs"],
        kpea_cutoff_mode=params["kpea_cutoff_mode"],
        kpea_primary_lfc_cutoff=params["kpea_primary_lfc_cutoff"],
        kpea_substrate_cutoff=args.kpea_substrate_cutoff,
        kpea_zscore_threshold=params["kpea_zscore_threshold"],
        kpea_z_cap=params["kpea_z_cap"],
        kpea_significance_method=params["kpea_significance_method"],
        kpea_empirical_p_threshold=params["kpea_empirical_p_threshold"],
        kpea_fdr_threshold=params["kpea_fdr_threshold"],
    )


def load_mapping_resources(
    analysis: KinaseActivityAnalysis,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load mapping resources.
    
    Args:
        analysis (KinaseActivityAnalysis): Analysis processed by this function.
    
    Returns:
        tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]: Loaded mapping resources.
    """
    peptide_enrichment = pd.read_csv(analysis.path_file_enrichment_peptides)
    df_blast = analysis._load_BLAST(df_peptide_enrichment=peptide_enrichment)  # noqa: SLF001

    raw_ptm = {
        "PTK": pd.read_csv(analysis.input_ptk_ptm_path),
        "STK": pd.read_csv(analysis.input_stk_ptm_path),
    }
    filtered_ptm = {
        "PTK": analysis._filter_ptm_interactions_for_uka(raw_ptm["PTK"], array_type="PTK"),  # noqa: SLF001
        "STK": analysis._filter_ptm_interactions_for_uka(raw_ptm["STK"], array_type="STK"),  # noqa: SLF001
    }
    return df_blast, raw_ptm, filtered_ptm


def deduplicate_kinase_table(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate the kinase result table before reporting.
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame used by this function.
    
    Returns:
        pd.DataFrame: Deduplicate kinase table.
    """
    if df is None or df.empty or "Kinase" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["Kinase"] = df["Kinase"].astype(str).str.strip()

    sort_cols = []
    ascending = []
    for col in ["Significant", "KRSA_AbsMeanZ", "NegLog10EmpiricalP", "NumSubstrates"]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)
    return df.drop_duplicates(subset="Kinase", keep="first").set_index("Kinase")


def peptide_values_for_mapping(df_collapsed: pd.DataFrame) -> pd.DataFrame:
    """Collect peptide-level values used in the mapping report.
    
    Args:
        df_collapsed (pd.DataFrame): Input pandas DataFrame containing collapsed.
    
    Returns:
        pd.DataFrame: Peptide values for mapping.
    """
    available = [col for col in PEPTIDE_VALUE_COLS if col in df_collapsed.columns]
    peptide_values = df_collapsed[available].copy()
    return peptide_values.drop_duplicates(subset="ID", keep="first")


def build_mapping_details_for_condition(
    analysis: KinaseActivityAnalysis,
    peptides: pd.DataFrame,
    df_blast: pd.DataFrame,
    filtered_ptm: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build mapping details for condition.
    
    Args:
        analysis (KinaseActivityAnalysis): Analysis processed by this function.
        peptides (pd.DataFrame): Pandas DataFrame containing peptides.
        df_blast (pd.DataFrame): Input pandas DataFrame containing BLAST.
        filtered_ptm (dict[str, pd.DataFrame]): Pandas DataFrame containing filtered PTM.
    
    Returns:
        pd.DataFrame: Constructed mapping details for condition.
    """
    detail_frames = []

    for array_type in ("PTK", "STK"):
        if "Type" not in peptides.columns:
            raise ValueError("Peptide_Statistics sheet must contain a 'Type' column.")

        df_array = peptides[peptides["Type"].astype(str).str.upper() == array_type].copy()
        if df_array.empty:
            continue

        df_collapsed = analysis._collapse_duplicate_peptides_for_uka(df_array)  # noqa: SLF001
        _kinase_to_peptides, mapping_df, _peptide_order = analysis._map_kinases_to_peptides(  # noqa: SLF001
            df_pooled=df_collapsed,
            df_ptm=filtered_ptm[array_type],
            df_BLAST=df_blast,
        )

        if mapping_df.empty:
            continue

        mapping_df = mapping_df.copy()
        mapping_df["Array_Type"] = array_type
        peptide_values = peptide_values_for_mapping(df_collapsed)
        mapping_df = mapping_df.merge(
            peptide_values,
            left_on="Peptide_ID",
            right_on="ID",
            how="left",
            suffixes=("", "_peptide"),
        )
        if "ID" in mapping_df.columns:
            mapping_df.drop(columns=["ID"], inplace=True)

        detail_frames.append(mapping_df)

    if not detail_frames:
        return pd.DataFrame()

    details = pd.concat(detail_frames, ignore_index=True)
    preferred_cols = [
        "condition",
        "Kinase_UniprotID",
        "Array_Type",
        "Peptide_ID",
        "peptide_change",
        "peptide_statistic",
        "mean_control",
        "mean_treatment",
        "p_value",
        "logp_value",
        "Sequence",
        "GeneName",
        "UniprotAccession",
        "Substrate_BLAST",
        "Peptide_CandidateSites",
        "Matched_PTM_Sites",
        "Source_Database",
        "Peptide_UniprotName",
        "Peptide_UniprotID",
    ]
    return details[[col for col in preferred_cols if col in details.columns]]


def ptm_availability_for_kinase(
    kinase: str,
    raw_ptm: dict[str, pd.DataFrame],
    filtered_ptm: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Summarize PTM evidence availability for one kinase.
    
    Args:
        kinase (str): Kinase used by this function.
        raw_ptm (dict[str, pd.DataFrame]): Pandas DataFrame containing raw PTM.
        filtered_ptm (dict[str, pd.DataFrame]): Pandas DataFrame containing filtered PTM.
    
    Returns:
        dict[str, object]: PTM availability for kinase.
    """
    out: dict[str, object] = {}
    raw_arrays = []
    filtered_arrays = []

    for array_type in ("PTK", "STK"):
        raw_rows = raw_ptm[array_type][
            raw_ptm[array_type]["ptm_enzyme"].astype(str) == kinase
        ]
        filtered_rows = filtered_ptm[array_type][
            filtered_ptm[array_type]["ptm_enzyme"].astype(str) == kinase
        ]

        if not raw_rows.empty:
            raw_arrays.append(array_type)
        if not filtered_rows.empty:
            filtered_arrays.append(array_type)

        out[f"raw_ptm_{array_type}_substrate_count"] = int(
            raw_rows["uniprot_id"].nunique()
        )
        out[f"filtered_ptm_{array_type}_substrate_count"] = int(
            filtered_rows["uniprot_id"].nunique()
        )

    out["active_ptm_raw_arrays"] = ";".join(raw_arrays)
    out["active_ptm_filtered_arrays"] = ";".join(filtered_arrays)
    return out


def _format_number(value, digits: int = 4) -> str:
    """Format number.
    
    Args:
        value: Input value processed by this helper.
        digits (int): Digits used by this function.
    
    Returns:
        str: Formatted number.
    """
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def format_peptide_change_table(group: pd.DataFrame) -> str:
    """Format peptide change table.
    
    Args:
        group (pd.DataFrame): Pandas DataFrame containing group.
    
    Returns:
        str: Formatted peptide change table.
    """
    if group.empty:
        return ""

    rows = (
        group.sort_values(["Peptide_ID", "Array_Type", "Substrate_BLAST"])
        .drop_duplicates(subset=["Peptide_ID", "Array_Type"])
        .itertuples(index=False)
    )

    formatted = []
    for row in rows:
        row_dict = row._asdict()
        formatted.append(
            "{peptide} [{array_type}] change={change}, statistic={stat}".format(
                peptide=row_dict.get("Peptide_ID", ""),
                array_type=row_dict.get("Array_Type", ""),
                change=_format_number(row_dict.get("peptide_change")),
                stat=_format_number(row_dict.get("peptide_statistic")),
            )
        )
    return " | ".join(formatted)


def classify_mapping_status(
    in_kerthi: bool,
    in_pykinaxe_all: bool,
    in_pykinaxe_significant: bool,
    mapped_peptide_count: int,
    active_filtered_arrays: str,
    substrate_cutoff: int,
) -> str:
    """Classify the mapping-support status for one kinase.
    
    Args:
        in_kerthi (bool): In kerthi used by this function.
        in_pykinaxe_all (bool): In pykinaxe all used by this function.
        in_pykinaxe_significant (bool): In pykinaxe significant used by this function.
        mapped_peptide_count (int): Mapped peptide count used by this function.
        active_filtered_arrays (str): Active filtered arrays used by this function.
        substrate_cutoff (int): Substrate cutoff used by this function.
    
    Returns:
        str: Classification for mapping status.
    """
    if in_kerthi and in_pykinaxe_significant:
        return "shared_significant"
    if in_pykinaxe_significant:
        return "pykinaxe_only_significant"
    if in_pykinaxe_all:
        return "mapped_and_scored_not_significant"
    if not active_filtered_arrays:
        return "not_available_after_active_ptm_filters"
    if mapped_peptide_count == 0:
        return "available_in_ptm_but_no_blast_site_peptide_mapping"
    if mapped_peptide_count < substrate_cutoff:
        return "mapped_below_substrate_cutoff"
    return "mapped_but_not_scored_check_array_or_filters"


def select_kinases_for_scope(
    scope: str,
    kerthi_ids: set[str],
    pykinaxe_sig_ids: set[str],
) -> set[str]:
    """Select kinases for scope.
    
    Args:
        scope (str): Scope used by this function.
        kerthi_ids (set[str]): Kerthi IDs processed by this function.
        pykinaxe_sig_ids (set[str]): Pykinaxe sig IDs processed by this function.
    
    Returns:
        set[str]: Selected kinases for scope.
    """
    if scope == "kerthi":
        return set(kerthi_ids)
    if scope == "missing":
        return set(kerthi_ids) - set(pykinaxe_sig_ids)
    if scope == "union":
        return set(kerthi_ids) | set(pykinaxe_sig_ids)
    raise ValueError(f"Unknown scope: {scope}")


def build_condition_report(
    condition: str,
    kerthi_df: pd.DataFrame,
    py_all: pd.DataFrame,
    py_sig: pd.DataFrame,
    mapping_details: pd.DataFrame,
    raw_ptm: dict[str, pd.DataFrame],
    filtered_ptm: dict[str, pd.DataFrame],
    analysis: KinaseActivityAnalysis,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build condition report.
    
    Args:
        condition (str): Condition used by this function.
        kerthi_df (pd.DataFrame): Input pandas DataFrame containing kerthi.
        py_all (pd.DataFrame): Pandas DataFrame containing py all.
        py_sig (pd.DataFrame): Pandas DataFrame containing py sig.
        mapping_details (pd.DataFrame): Pandas DataFrame containing mapping details.
        raw_ptm (dict[str, pd.DataFrame]): Pandas DataFrame containing raw PTM.
        filtered_ptm (dict[str, pd.DataFrame]): Pandas DataFrame containing filtered PTM.
        analysis (KinaseActivityAnalysis): Analysis processed by this function.
        scope (str): Scope used by this function.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Constructed condition report.
    """
    kerthi_by_kinase = kerthi_df.drop_duplicates(subset="Kinase").set_index("Kinase")
    py_all_by_kinase = deduplicate_kinase_table(py_all)
    py_sig_ids = set(py_sig["Kinase"].dropna().astype(str)) if not py_sig.empty else set()
    py_all_ids = set(py_all["Kinase"].dropna().astype(str)) if not py_all.empty else set()
    kerthi_ids = set(kerthi_df["Kinase"].dropna().astype(str))

    kinases = sorted(select_kinases_for_scope(scope, kerthi_ids, py_sig_ids))

    details = mapping_details.copy()
    if not details.empty:
        details.insert(0, "condition", condition)
        details["in_kerthi_reference"] = details["Kinase_UniprotID"].isin(kerthi_ids)
        details["in_pykinaxe_all"] = details["Kinase_UniprotID"].isin(py_all_ids)
        details["in_pykinaxe_significant"] = details["Kinase_UniprotID"].isin(py_sig_ids)
        details = details[details["Kinase_UniprotID"].isin(kinases)].copy()

    summary_rows = []
    for kinase in kinases:
        group = (
            details[details["Kinase_UniprotID"] == kinase]
            if not details.empty
            else pd.DataFrame()
        )
        peptide_group = (
            group.drop_duplicates(subset=["Peptide_ID", "Array_Type"])
            if not group.empty
            else group
        )

        in_kerthi = kinase in kerthi_ids
        in_pykinaxe_all = kinase in py_all_ids
        in_pykinaxe_sig = kinase in py_sig_ids

        ptm_availability = ptm_availability_for_kinase(kinase, raw_ptm, filtered_ptm)
        py_row = py_all_by_kinase.loc[kinase] if in_pykinaxe_all else {}
        kerthi_row = kerthi_by_kinase.loc[kinase] if in_kerthi else {}

        mapped_peptides = (
            sorted(peptide_group["Peptide_ID"].dropna().astype(str).unique())
            if not peptide_group.empty
            else []
        )
        mapped_substrates = (
            sorted(group["Substrate_BLAST"].dropna().astype(str).unique())
            if not group.empty and "Substrate_BLAST" in group.columns
            else []
        )

        summary_rows.append(
            {
                "condition": condition,
                "kinase": kinase,
                "in_kerthi_reference": in_kerthi,
                "in_pykinaxe_all": in_pykinaxe_all,
                "in_pykinaxe_significant": in_pykinaxe_sig,
                "mapping_status": classify_mapping_status(
                    in_kerthi=in_kerthi,
                    in_pykinaxe_all=in_pykinaxe_all,
                    in_pykinaxe_significant=in_pykinaxe_sig,
                    mapped_peptide_count=len(mapped_peptides),
                    active_filtered_arrays=ptm_availability["active_ptm_filtered_arrays"],
                    substrate_cutoff=analysis.kpea_substrate_cutoff,
                ),
                "mapped_peptide_count": len(mapped_peptides),
                "mapped_substrate_count": len(mapped_substrates),
                "mapped_peptides": ";".join(mapped_peptides),
                "mapped_peptide_change_values": format_peptide_change_table(group),
                "mapped_substrates": ";".join(mapped_substrates),
                "mapped_arrays": (
                    ";".join(sorted(group["Array_Type"].dropna().astype(str).unique()))
                    if not group.empty and "Array_Type" in group.columns
                    else ""
                ),
                "matched_ptm_sites": (
                    ";".join(sorted(group["Matched_PTM_Sites"].dropna().astype(str).unique()))
                    if not group.empty and "Matched_PTM_Sites" in group.columns
                    else ""
                ),
                "mapping_sources": (
                    ";".join(sorted(group["Source_Database"].dropna().astype(str).unique()))
                    if not group.empty and "Source_Database" in group.columns
                    else ""
                ),
                "kerthi_activity": kerthi_row.get("Activity", np.nan),
                "pykinaxe_type": py_row.get("Type", ""),
                "pykinaxe_num_substrates": py_row.get("NumSubstrates", np.nan),
                "pykinaxe_kinase_change": py_row.get("KinaseChange", np.nan),
                "pykinaxe_kinase_statistic": py_row.get("KinaseStatistic", np.nan),
                "pykinaxe_z_score": py_row.get("Z_Score", np.nan),
                "pykinaxe_p_value": py_row.get("p_value", np.nan),
                "pykinaxe_fdr": py_row.get("FDR", np.nan),
                **ptm_availability,
            }
        )

    summary = pd.DataFrame(summary_rows)
    for col in SUMMARY_NUMERIC_COLS:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    return summary, details


def build_parameter_table(
    args: argparse.Namespace,
    results_dir: Path,
    output_path: Path,
    analysis: KinaseActivityAnalysis,
) -> pd.DataFrame:
    """Build parameter table.
    
    Args:
        args (argparse.Namespace): Args processed by this function.
        results_dir (Path): Directory containing or receiving the results.
        output_path (Path): Path to the output.
        analysis (KinaseActivityAnalysis): Analysis processed by this function.
    
    Returns:
        pd.DataFrame: Constructed parameter table.
    """
    values = {
        "results_dir": results_dir,
        "benchmark_data_dir": args.benchmark_data_dir,
        "output_path": output_path,
        "scope": args.scope,
        "conditions": ",".join(args.conditions),
        "BLAST_threshold": analysis.BLAST_threshold,
        "require_known_ptm_site": analysis.require_known_ptm_site,
        "use_verified_interactions_only": analysis.use_verified_interactions_only,
        "verified_evidence_levels": ";".join(sorted(analysis.verified_evidence_levels)),
        "verified_min_score": analysis.verified_min_score,
        "verified_min_references": analysis.verified_min_references,
        "input_ptk_ptm_path": analysis.input_ptk_ptm_path,
        "input_stk_ptm_path": analysis.input_stk_ptm_path,
        "path_file_enrichment_peptides": analysis.path_file_enrichment_peptides,
        "kpea_substrate_cutoff": analysis.kpea_substrate_cutoff,
    }
    return pd.DataFrame(
        [{"parameter": key, "value": str(value)} for key, value in values.items()]
    )


def write_report(
    output_path: Path,
    summary: pd.DataFrame,
    details: pd.DataFrame,
    parameters: pd.DataFrame,
) -> None:
    """Write report.
    
    Args:
        output_path (Path): Path to the output.
        summary (pd.DataFrame): Pandas DataFrame containing summary.
        details (pd.DataFrame): Pandas DataFrame containing details.
        parameters (pd.DataFrame): Pandas DataFrame containing parameters.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="kinase_summary", index=False)
        details.to_excel(writer, sheet_name="mapping_details", index=False)
        parameters.to_excel(writer, sheet_name="parameters", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build arg parser.
    
    Args:
        None.
    
    Returns:
        argparse.ArgumentParser: Constructed arg parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build an Excel report showing how Kerthi reference kinases map to "
            "pyKinaXe peptides through BLAST and PTM interactions."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="pyKinaXe result directory. Defaults to the latest Kerthi_HDV_Aug24 result directory.",
    )
    parser.add_argument(
        "--benchmark-data-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DATA_DIR,
        help="Directory containing Kerthi results_lhdag/results_shdag/results_psvld3 CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .xlsx path. Defaults to <results-dir>/<timestamp>_kerthi_kinase_mapping_report.xlsx.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(KERTHI_BENCHMARK_SPECS),
        choices=list(KERTHI_BENCHMARK_SPECS),
        help="Kerthi test conditions to include.",
    )
    parser.add_argument(
        "--scope",
        choices=("kerthi", "missing", "union"),
        default="kerthi",
        help=(
            "kinases to report: all Kerthi kinases, Kerthi kinases missing from "
            "pyKinaXe significant output, or Kerthi/pyKinaXe-significant union."
        ),
    )
    parser.add_argument(
        "--blast-threshold",
        type=float,
        default=DEFAULT_UKA_KPEA_PARAMS["BLAST_threshold"],
        help="BLAST Positives(%%) threshold used for peptide-protein matches.",
    )
    parser.add_argument(
        "--kpea-substrate-cutoff",
        type=int,
        default=DEFAULT_UKA_KPEA_PARAMS["kpea_substrate_cutoff"],
        help="Minimum mapped peptide count used to classify below-cutoff kinases.",
    )
    parser.add_argument(
        "--require-known-ptm-site",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_UKA_KPEA_PARAMS["require_known_ptm_site"],
        help="Require informative PTM sites and peptide-site compatibility.",
    )
    parser.add_argument(
        "--use-verified-interactions-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_UKA_KPEA_PARAMS["use_verified_interactions_only"],
        help="Apply the active verified-interaction filter to OmniPath PTM rows.",
    )
    parser.add_argument(
        "--verified-min-score",
        type=float,
        default=DEFAULT_UKA_KPEA_PARAMS["verified_min_score"],
        help="Optional minimum PTM score when verified filtering is active.",
    )
    parser.add_argument(
        "--verified-min-references",
        type=int,
        default=DEFAULT_UKA_KPEA_PARAMS["verified_min_references"],
        help="Optional minimum reference count when verified filtering is active.",
    )
    parser.add_argument(
        "--debugging-print",
        action="store_true",
        help="Print verbose pyKinaXe mapping diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    """Run the module as a command-line entry point.
    
    Args:
        argv (list[str] | None): Argv processed by this function.
    
    Returns:
        None: The command-line entry point runs for its side effects.
    """
    args = build_arg_parser().parse_args(argv)

    results_dir = args.results_dir or find_latest_kerthi_result_dir()
    timestamp = infer_timestamp(results_dir)
    output_path = args.output or (
        results_dir / f"{timestamp}_kerthi_kinase_mapping_report.xlsx"
    )

    analysis = make_analysis(args)
    df_blast, raw_ptm, filtered_ptm = load_mapping_resources(analysis)

    summary_frames = []
    detail_frames = []
    for condition in args.conditions:
        print(f"Building Kerthi mapping report for {condition}...")
        kerthi_df = load_kerthi_reference(args.benchmark_data_dir, condition)
        peptides, py_all, py_sig = load_condition_workbook(
            results_dir=results_dir,
            timestamp=timestamp,
            condition=condition,
        )
        mapping_details = build_mapping_details_for_condition(
            analysis=analysis,
            peptides=peptides,
            df_blast=df_blast,
            filtered_ptm=filtered_ptm,
        )
        summary, details = build_condition_report(
            condition=condition,
            kerthi_df=kerthi_df,
            py_all=py_all,
            py_sig=py_sig,
            mapping_details=mapping_details,
            raw_ptm=raw_ptm,
            filtered_ptm=filtered_ptm,
            analysis=analysis,
            scope=args.scope,
        )
        summary_frames.append(summary)
        detail_frames.append(details)

    summary_report = pd.concat(summary_frames, ignore_index=True)
    detail_report = (
        pd.concat(detail_frames, ignore_index=True)
        if any(not frame.empty for frame in detail_frames)
        else pd.DataFrame()
    )
    parameter_table = build_parameter_table(args, results_dir, output_path, analysis)

    write_report(
        output_path=output_path,
        summary=summary_report,
        details=detail_report,
        parameters=parameter_table,
    )

    print(f"\nWrote Kerthi kinase mapping report: {output_path}")
    print("\nMapping status summary:")
    print(
        summary_report.groupby(["condition", "mapping_status"])
        .size()
        .reset_index(name="n")
        .to_string(index=False)
    )
    return output_path


if __name__ == "__main__":
    main()
