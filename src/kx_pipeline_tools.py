"""Orchestration of the pyKinaXe analysis stages for the terminal and web workflows.

The scientific code lives in the other kx_* modules. This module discovers the
PTK/STK runs, builds the DataLoader instances, lays out the results
directories, resolves the default parameters, and runs the image pipeline, the
peptide, kinase, family and pathway stages and the optional figures. It is used
by scripts/kx_kinase_extraction_pipeline.py and by
webapp/backend/kx_web_kinase_extraction_pipeline.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from config.pipeline_defaults import (  # noqa: E402
    CREATE_PROCESSING_STAGE_FIGURES,
    CREATE_PUBLICATION_FIGURES,
    DEFAULT_UKA_KPEA_PARAMS,
    NUM_REPRESENTATIVE_IMAGES,
    PROCESSING_STAGE_FIGURE_IMAGE_LIMIT,
    PROCESSING_STAGE_FIGURES_ALL_IMAGES,
    PROCESSING_STAGE_FIGURES_FRACTION,
)
from config.upstream_kinase_families_analysis import (  # noqa: E402
    UPSTREAM_KINASE_FAMILIES_ANALYSIS_DEFAULTS as FAMILY_DEFAULTS,
)
from kx_data_enricher import DataEnricher  # noqa: E402
from kx_data_importer import DataLoader  # noqa: E402
from kx_image_processor import ImageProcessor  # noqa: E402
from kx_pathway_enrichment_analysis import (  # noqa: E402
    PathwayEnrichmentAnalysis,
    gprofiler_gene_symbols,
)
from kx_peptide_analysis import PeptideStatistics  # noqa: E402
from kx_upstream_kinase_analysis import KinaseActivityAnalysis  # noqa: E402


def build_results_dir(
    analysis_timestamp: str,
    experiment_name: str,
    results_parent_relpath: Optional[Path] = None,
) -> Path:
    """Build the shared non-timestamped experiment results directory.
    
    The terminal pipeline writes under ``results/`` by default, but the web
    backend can override the root via ``PYKINAXE_RESULTS_ROOT`` so each web job
    stays isolated inside its own output sandbox.
    
    Args:
        analysis_timestamp (str): Timestamp string assigned to the current analysis run.
    
    Returns:
        Path: Constructed experiment-level results dir.
    """
    override_root = os.environ.get("PYKINAXE_RESULTS_ROOT")
    if override_root:
        results_root = Path(override_root).expanduser().resolve()
    else:
        results_root = REPO_ROOT / "results"
    parent_relpath = Path(results_parent_relpath or ".")
    return results_root / parent_relpath / experiment_name


def resolve_shared_results_context(loader_ptk, loader_stk) -> dict[str, Path | str]:
    """Compute the shared result-folder context for a PTK/STK pair.
    
    PTK and STK runs are often siblings under a common experiment parent. This
    helper preserves that relationship in the results tree so downstream files
    stay grouped together in a way that mirrors the input data layout.
    """
    base_data_dir = loader_ptk.base_data_dir.resolve()
    ptk_experiment_dir = loader_ptk.experiment_dir.resolve()
    stk_experiment_dir = loader_stk.experiment_dir.resolve()
    common_parent_dir = Path(
        os.path.commonpath(
            [
                str(loader_ptk.data_dir.resolve()),
                str(loader_stk.data_dir.resolve()),
            ]
        )
    )

    if common_parent_dir != base_data_dir:
        results_structure_root = common_parent_dir.parent
    else:
        results_structure_root = base_data_dir

    results_parent_relpath = results_structure_root.relative_to(base_data_dir)

    image_processing_relroot = (
        Path(common_parent_dir.name) / f"{loader_ptk.timestamp}_image_processing"
    )

    for loader in (loader_ptk, loader_stk):
        loader.results_parent_relpath = results_parent_relpath
    loader_ptk.results_experiment_relpath = (
        image_processing_relroot / ptk_experiment_dir.name
    )
    loader_stk.results_experiment_relpath = (
        image_processing_relroot / stk_experiment_dir.name
    )

    return {
        "results_parent_relpath": results_parent_relpath,
        "experiment_name": common_parent_dir.name,
    }


def resolve_base_data_dir(base_data_dir=None) -> Path:
    """Resolve the base data directory used for folder discovery."""
    if base_data_dir is None:
        possible_paths = [
            Path("./data"),
            Path("../data"),
            Path("../../data"),
            Path(os.getcwd()) / "data",
            Path(os.getcwd()).parent / "data",
            Path(os.getcwd()).parent.parent / "data",
            Path.home() / "data",
        ]

        for path in possible_paths:
            if path.exists() and path.is_dir():
                return path

        raise FileNotFoundError(
            "Could not find data directory. Tried:\n"
            + "\n".join(f"  - {p}" for p in possible_paths)
            + "\n\nPlease specify base_data_dir explicitly."
        )

    base_path = Path(base_data_dir)
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {base_path}")
    return base_path


def discover_valid_data_folders(base_data_dir=None):
    """Discover experimental folders that contain PTK or STK runs."""
    base_path = resolve_base_data_dir(base_data_dir)
    valid_folders_map = {}

    for root, dirs, _files in os.walk(base_path):
        root_path = Path(root)
        for dir_name in dirs:
            folder_upper = dir_name.upper()
            has_datetime = bool(DataLoader.FOLDER_DATETIME_PATTERN.search(dir_name))
            has_run = DataLoader.FOLDER_RUN_KEYWORD in folder_upper
            has_peptide_type = any(
                peptide in folder_upper for peptide in DataLoader.VALID_PEPTIDE_TYPES
            )

            if not (has_datetime and has_run and has_peptide_type):
                continue

            dir_path = root_path / dir_name
            relative_parent = root_path.relative_to(base_path)
            parent_key = (
                str(relative_parent) if str(relative_parent) != "." else root_path.name
            )
            valid_folders_map.setdefault(parent_key, []).append(dir_path)

    return base_path, valid_folders_map, sorted(valid_folders_map)


def resolve_data_selection(
    experiment_index: int,
    run_index: int,
    base_data_dir=None,
    expected_peptide_type: Optional[str] = None,
):
    """Resolve a numeric folder selection into concrete PTK/STK inputs."""
    base_path, valid_folders_map, experiment_keys = discover_valid_data_folders(
        base_data_dir=base_data_dir
    )

    if not experiment_keys:
        raise FileNotFoundError(f"No valid experiment folders found in {base_path}")

    if not 1 <= experiment_index <= len(experiment_keys):
        raise ValueError(
            f"Experiment index {experiment_index} is out of range 1-{len(experiment_keys)}."
        )

    experiment_key = experiment_keys[experiment_index - 1]
    run_folders = sorted(valid_folders_map[experiment_key])

    if not 1 <= run_index <= len(run_folders):
        raise ValueError(
            f"Run index {run_index} is out of range 1-{len(run_folders)} "
            f"for experiment '{experiment_key}'."
        )

    selected_folder = run_folders[run_index - 1]
    peptides_type = next(
        (
            peptide_type
            for peptide_type in DataLoader.VALID_PEPTIDE_TYPES
            if peptide_type in selected_folder.name.upper()
        ),
        None,
    )

    if expected_peptide_type is not None:
        normalized_expected = expected_peptide_type.upper()
        if peptides_type != normalized_expected:
            raise ValueError(
                f"Selection {experiment_index}-{run_index} resolved to '{selected_folder.name}' "
                f"({peptides_type}), expected {normalized_expected}."
            )

    experiment_dir = (
        base_path if experiment_key == base_path.name else base_path / experiment_key
    )
    return {
        "base_data_dir": base_path,
        "experiment_key": experiment_key,
        "experiment_dir": experiment_dir,
        "experiment_name": experiment_dir.name,
        "subfolder_name": selected_folder.name,
        "data_dir": selected_folder,
        "peptides_type": peptides_type,
    }


def build_loader_from_selection(
    selection: Tuple[int, int],
    timestamp: str,
    expected_peptide_type: str,
    base_data_dir=None,
):
    """Build and configure a data loader from the chosen inputs.
    
    Args:
        timestamp (str): Timestamp string associated with the current analysis run.
    """
    experiment_index, run_index = selection
    resolved = resolve_data_selection(
        experiment_index=experiment_index,
        run_index=run_index,
        base_data_dir=base_data_dir,
        expected_peptide_type=expected_peptide_type,
    )

    print(
        f"Using {expected_peptide_type} selection "
        f"{experiment_index}-{run_index}: "
        f"{resolved['experiment_key']} / {resolved['subfolder_name']}"
    )

    loader = DataLoader(
        data_dir=resolved["base_data_dir"],
        experiment_name=resolved["experiment_key"],
        subfolder=resolved["subfolder_name"],
        timestamp=timestamp,
    )
    loader.experiment_name = resolved["experiment_name"]
    return loader


def resolve_uka_params(
    analysis_timestamp: str,
    experiment_name: str,
    uka_params=None,
    results_parent_relpath: Optional[Path] = None,
):
    """Resolve the downstream UKA/KPEA parameter dictionary.
    
    This function starts from the YAML-backed defaults in ``config/`` and then
    applies any caller overrides. It also fills in timestamped output paths so
    later stages can assume that every required output location already exists
    in the parameter bundle.
    
    Args:
        analysis_timestamp (str): Timestamp string assigned to the current analysis run.
    """
    resolved = dict(DEFAULT_UKA_KPEA_PARAMS)
    if uka_params is not None:
        resolved.update(uka_params)

    custom_keys = set(uka_params or {})
    results_dir = build_results_dir(
        analysis_timestamp,
        experiment_name,
        results_parent_relpath=results_parent_relpath,
    )
    uka_results_dir = results_dir / f"{analysis_timestamp}_downstream_analysis"
    uka_results_dir.mkdir(parents=True, exist_ok=True)

    # One folder per result topic, each with its own plots/ subfolder, so the
    # tables and the figures that describe them stay together.
    def _resolve_results_dir(key, folder_name):
        """Resolve one per-topic results directory.
        
        Args:
            key: Parameter key holding the optional override.
            folder_name: Folder created inside the downstream directory by default.
        """
        value = resolved.get(key)
        resolved[key] = (
            Path(value) if value is not None else uka_results_dir / folder_name
        )
        return resolved[key]

    peptide_dir = _resolve_results_dir("peptide_results_output", "results_peptides")
    kinase_dir = _resolve_results_dir(
        "kinase_results_output", "results_individual_kinases"
    )
    _resolve_results_dir("family_results_output", "results_kinase_families")
    pathway_dir = _resolve_results_dir("pathway_results_output", "results_pathways")

    resolved["log_output"] = (
        Path(resolved["log_output"])
        if resolved["log_output"] is not None
        else uka_results_dir / "logs"
    )
    # Filename PREFIX (not a directory) for the optional debug tables of the
    # peptide stage, so they join the logs instead of littering the run root.
    resolved["path_output_peptide_statistic"] = (
        Path(resolved["path_output_peptide_statistic"])
        if resolved["path_output_peptide_statistic"] is not None
        else resolved["log_output"] / analysis_timestamp
    )

    global_waterfall = resolved["waterfall_plot"]
    global_heatmap = resolved["heatmap_plot"]

    if "peptide_waterfall_plot" not in custom_keys:
        resolved["peptide_waterfall_plot"] = global_waterfall

    if "peptide_heatmap_plot" not in custom_keys:
        resolved["peptide_heatmap_plot"] = global_heatmap
    if "pathway_heatmap_plot" not in custom_keys:
        resolved["pathway_heatmap_plot"] = global_heatmap

    def _resolve_stage_output(key, default_dir):
        """Resolve an optional stage-figure output destination."""
        value = resolved.get(key)
        resolved[key] = default_dir if value is None else Path(value)

    _resolve_stage_output("peptide_waterfall_plot_output", peptide_dir / "plots")
    _resolve_stage_output("peptide_heatmap_plot_output", peptide_dir / "plots")
    _resolve_stage_output("pathway_heatmap_plot_output", pathway_dir / "plots")
    _resolve_stage_output("kinase_venn_plot_output", kinase_dir / "plots")
    _resolve_stage_output("pathway_venn_plot_output", pathway_dir / "plots")

    # Fail here rather than after the whole analysis has run.
    resolved["output_mode"] = resolve_output_mode(resolved["output_mode"])

    resolved["input_stk_ptm_path"] = Path(resolved["input_stk_ptm_path"])
    resolved["input_ptk_ptm_path"] = Path(resolved["input_ptk_ptm_path"])
    return resolved


RUN_CONFIG_FILENAME = "run_config.txt"


def write_run_config(downstream_dir, resolved_params):
    """Write the thresholds, hit cutoffs and backgrounds of a downstream run.

    The single-kinase and pathway values are the resolved ones (config defaults
    plus caller overrides). The family values come from
    config/upstream_kinase_families_analysis.yaml, which the family stage reads
    directly.

    Args:
        downstream_dir: The run's ``<timestamp>_downstream_analysis`` folder.
        resolved_params: Output of :func:`resolve_uka_params`.

    Returns:
        Path: The written ``run_config.txt``.
    """

    def _cutoffs(values):
        """Return LFC cutoffs as ``0.2, 0.3, 0.4``."""
        return ", ".join(str(float(value)) for value in values)

    def _cutoff_mode_lines(mode, cutoffs, primary, key_prefix=""):
        """Return the cutoff-mode line, plus the used cutoff in primary mode."""
        lines = [f"  {key_prefix}kpea_cutoff_mode: {mode}"]
        if str(mode).lower() == "primary":
            used = cutoffs[0] if primary is None else primary
            lines.append(f"  {key_prefix}kpea_primary_lfc_cutoff: {float(used)}")
        return lines

    kinase_cutoffs = list(resolved_params["kpea_lfc_cutoffs"])
    family_cutoffs = list(FAMILY_DEFAULTS["default_kpea_lfc_cutoffs"])
    lines = [
        "pyKinaXe downstream analysis: thresholds, hit cutoffs and backgrounds",
        f"run folder: {Path(downstream_dir).name}",
        f"written:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Individual kinases (UKA/KPEA; config/pipeline_defaults.yaml)",
        f"  kpea_zscore_threshold: {float(resolved_params['kpea_zscore_threshold'])}",
        f"  kpea_lfc_cutoffs: {_cutoffs(kinase_cutoffs)}",
        *_cutoff_mode_lines(
            resolved_params["kpea_cutoff_mode"],
            kinase_cutoffs,
            resolved_params["kpea_primary_lfc_cutoff"],
        ),
        f"  kpea_substrate_cutoff: {resolved_params['kpea_substrate_cutoff']}",
        f"  kpea_background_universe: {resolved_params['kpea_background_universe']}",
        f"  BLAST_threshold: {resolved_params['BLAST_threshold']}",
        "",
        "Kinase families (config/upstream_kinase_families_analysis.yaml)",
        f"  default_kpea_zscore_threshold: {float(FAMILY_DEFAULTS['default_kpea_zscore_threshold'])}",
        f"  default_kpea_lfc_cutoffs: {_cutoffs(family_cutoffs)}",
        *_cutoff_mode_lines(
            FAMILY_DEFAULTS["default_kpea_cutoff_mode"],
            family_cutoffs,
            None,
            key_prefix="default_",
        ),
        f"  default_kpea_background_universe: {FAMILY_DEFAULTS['default_kpea_background_universe']}",
        f"  family_mapping_source: {FAMILY_DEFAULTS['family_mapping_source']}"
        + (
            " (KRSA mapping table; BLAST_threshold does not apply)"
            if str(FAMILY_DEFAULTS["family_mapping_source"]).lower() == "krsa"
            else ""
        ),
        "",
        "Pathway enrichment (config/pipeline_defaults.yaml)",
        f"  pathway_background: {resolved_params['pathway_background']}",
        f"  significance_level_pathways: {float(resolved_params['significance_level_pathways'])}"
        " (g:SCS-corrected)",
        "",
    ]
    output_path = Path(downstream_dir) / RUN_CONFIG_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_timed_uka_params(uka_params=None):
    """Disable expensive plot/export options for the timed analysis phase.
    
    The repository reports the scientific analysis duration separately from the
    post-processing visualization work. This helper keeps the timed portion
    focused on statistics and enrichment while the heavier plotting steps can
    run afterwards without skewing the headline runtime summary.
    """
    timed_params = dict(uka_params or {})
    timed_params.update(
        {
            "waterfall_plot": False,
            "heatmap_plot": False,
            "peptide_waterfall_plot": False,
            "peptide_heatmap_plot": False,
            "pathway_heatmap_plot": False,
            "venn_plot": False,
            "venn_plot_tables": False,
        }
    )
    return timed_params


USER_OUTPUT_MODE = "user"
DEVELOPER_OUTPUT_MODE = "developer"
ALLOWED_OUTPUT_MODES = (USER_OUTPUT_MODE, DEVELOPER_OUTPUT_MODE)

# Columns that are, BY CONSTRUCTION, an exact copy of another column in the same
# kinase/family table -- verified against kx_upstream_kinase_analysis.py, not just
# against one run:
#   KRSA_MeanZ, Z_Score                      both float(mean_z[i])
#   KRSA_AbsMeanZ, KPEA_AbsDominantZ         both float(abs_mean_z[i]) = |Z_Score|
#   MeanSubstrate, KinaseStatistic           both = mean_peptide_statistic
#   Significant_ZScore / _SelectedMethod /
#   SelectedForReport / Significant          all assigned from one mask
# They are dropped from BOTH output modes; the surviving name of each group is the
# one kept in the user-mode lists below. The in-memory frames keep every column,
# because _rank_kinase_results ranks on KRSA_AbsMeanZ and KinaseStatistic.
KINASE_DUPLICATE_COLUMNS = (
    "MeanSubstrate",
    "KinaseStatistic",
    "KRSA_MeanZ",
    "KRSA_AbsMeanZ",
    "KPEA_AbsDominantZ",
    "Significant_ZScore",
    "Significant_SelectedMethod",
    "SelectedForReport",
)

# Exact column sets for the user mode -- the short, publication-facing tables.
USER_MODE_COLUMNS = {
    "kinases": (
        "Comparison",
        "Kinase",
        "Kinase_Name",
        "NumSubstrates",
        "KinaseChange",
        "MeanPeptideStatistic",
        "MedianPeptideStatistic",
        "Z_Score",
        "Significant",
        "Type",
    ),
    "families": (
        "Comparison",
        "Kinase_Family",
        "NumSubstrates",
        "KinaseChange",
        "MeanPeptideStatistic",
        "MedianPeptideStatistic",
        "Z_Score",
        "Significant",
        "Type",
    ),
    "pathways": (
        "source",
        "native",
        "name",
        "p_value",
        "significant",
        "description",
        "precision",
        "recall",
        "parents",
        "intersections",
    ),
    # The peptide table carries one value and one label column per sample, so the
    # per-sample block is expanded at write time for however many replicates the
    # run has. "<samples>" marks where it goes.
    "peptides": (
        "ID",
        "UniprotAccession",
        "GeneName",
        "UniprotAccessions",
        "GeneNames",
        "n_mapped_proteins",
        "Sequence",
        "<samples>",
        "average_expression",
        "t_statistic",
        "p_value",
        "logp_value",
        "mean_control",
        "mean_treatment",
        "SD_control",
        "SD_treatment",
        "peptide_change",
        "Type",
        "ControlCondition",
        "Condition",
        "Construct",
    ),
}

PEPTIDE_SAMPLE_BLOCK_MARKER = "<samples>"


def resolve_output_mode(value):
    """Validate the configured output mode.
    
    Args:
        value: Configured mode name.
    
    Returns:
        str: Either ``'user'`` or ``'developer'``.
    """
    resolved = str(value).strip().lower()
    if resolved not in ALLOWED_OUTPUT_MODES:
        raise ValueError(
            f"Unknown output_mode '{value}'. Use one of {list(ALLOWED_OUTPUT_MODES)}."
        )
    return resolved


def _peptide_sample_columns(df):
    """List the per-sample peptide columns, value and label paired, by index.
    
    Args:
        df: The peptide table.
    
    Returns:
        list[str]: ``control_sample_1, control_label_1, ..., treatment_sample_1, ...``
    """
    columns = []
    for group in ("control", "treatment"):
        indices = sorted(
            {
                int(col.rsplit("_", 1)[1])
                for col in df.columns
                if col.startswith(f"{group}_sample_")
                and not col.endswith("_zscore")
                and col.rsplit("_", 1)[1].isdigit()
            }
        )
        for index in indices:
            for prefix in (f"{group}_sample_{index}", f"{group}_label_{index}"):
                if prefix in df.columns:
                    columns.append(prefix)
    return columns


def collapse_peptide_rows_for_export(df):
    """Collapse the peptide table to one row per (array, peptide) for export.

    ``peptide_statistics`` carries one row per (peptide, mapped protein): the
    enrichment table repeats a peptide once for every UniProt accession it maps
    to, with identical statistics (verified: the duplicate groups of a run differ
    only in ``UniprotAccession`` / ``GeneName``). A peptide printed on BOTH arrays
    however has two real measurements, so the key is ``(Type, ID)``, not ``ID``.

    The dedup rule is the one the waterfall/heatmap path uses
    (sort by ``p_value`` ascending, keep the first row per ``(Type, ID)``), so the
    published table and the figures of the same run describe the same peptide set.
    The protein multiplicity that the long form carried implicitly is kept
    explicitly in three added columns.

    Args:
        df: The raw ``peptide_statistics`` frame of one comparison.

    Returns:
        pd.DataFrame: One row per ``(Type, ID)`` plus ``UniprotAccessions``,
        ``GeneNames`` (``';'``-joined, sorted, unique) and ``n_mapped_proteins``.
    """
    if df is None or df.empty or not {"Type", "ID"}.issubset(df.columns):
        return df

    def _joined(group, column):
        """Sorted, unique, ``';'``-joined values of one column of one group."""
        if column not in group.columns:
            return ""
        return ";".join(
            sorted(
                {
                    str(value).strip()
                    for value in group[column].dropna()
                    if str(value).strip()
                }
            )
        )

    ordered = df.sort_values("p_value", ascending=True) if "p_value" in df.columns else df

    accessions, gene_names, protein_counts = {}, {}, {}
    for key, group in ordered.groupby(["Type", "ID"], sort=False):
        joined_accessions = _joined(group, "UniprotAccession")
        accessions[key] = joined_accessions
        gene_names[key] = _joined(group, "GeneName")
        protein_counts[key] = len([v for v in joined_accessions.split(";") if v]) or len(group)

    collapsed = ordered.drop_duplicates(subset=["Type", "ID"], keep="first").copy()
    keys = list(zip(collapsed["Type"], collapsed["ID"]))
    collapsed["UniprotAccessions"] = [accessions.get(key, "") for key in keys]
    collapsed["GeneNames"] = [gene_names.get(key, "") for key in keys]
    collapsed["n_mapped_proteins"] = [protein_counts.get(key, 1) for key in keys]
    return collapsed.reset_index(drop=True)


def project_output_columns(df, table, output_mode):
    """Reduce one result table to the columns its output mode exports.
    
    Developer mode keeps everything except the exact duplicates listed in
    ``KINASE_DUPLICATE_COLUMNS``; user mode keeps the short, fixed column set of
    ``USER_MODE_COLUMNS``. Requested columns the table does not have are skipped
    rather than raising, so a run without pathway annotations or with a different
    replicate count still writes.
    
    Args:
        df: The table to project.
        table: One of ``'peptides'``, ``'kinases'``, ``'families'``, ``'pathways'``.
        output_mode: ``'user'`` or ``'developer'``.
    
    Returns:
        pd.DataFrame: The table with only the exported columns, in export order.
    """
    if df is None or df.empty:
        return df

    if output_mode == DEVELOPER_OUTPUT_MODE:
        if table in ("kinases", "families"):
            drop = [col for col in KINASE_DUPLICATE_COLUMNS if col in df.columns]
            return df.drop(columns=drop)
        return df

    wanted = []
    for name in USER_MODE_COLUMNS[table]:
        if name == PEPTIDE_SAMPLE_BLOCK_MARKER:
            wanted.extend(_peptide_sample_columns(df))
        elif name in df.columns:
            wanted.append(name)
    return df[wanted]


def rank_by_absolute_z(df):
    """Order one kinase/family table by descending |Z_Score|.

    ``Significant`` is exactly ``|Z_Score| >= threshold``, so this single sort key
    also puts every significant row above every non-significant one -- which is why
    a separate significance file is unnecessary: filter the ``Significant`` column,
    or just read from the top.
    
    Args:
        df: A kinase or family table.
    
    Returns:
        pd.DataFrame: The table, most extreme Z first.
    """
    if df is None or df.empty or "Z_Score" not in df.columns:
        return df
    return (
        df.assign(_abs_z=lambda frame: frame["Z_Score"].abs())
        .sort_values(["_abs_z", "NumSubstrates"], ascending=[False, False])
        .drop(columns="_abs_z")
        .reset_index(drop=True)
    )


def save_condition_results_to_csv(
    condition_results,
    peptide_output,
    kinase_output,
    pathway_output,
    control,
    condition,
    output_mode=USER_OUTPUT_MODE,
):
    """Write the downstream tables of one comparison as CSVs.

    Each table goes into the results folder of its own topic. One file per
    comparison, and for the kinases one file per comparison AND array:

    - ``peptide_statistics_<control>_<condition>.csv``
    - ``kinases_<PTK|STK>_<control>_<condition>.csv``
    - ``pathways_<source>_<control>_<condition>.csv``

    The kinase tables come from the UN-deduplicated result, so a kinase with
    substrates on both arrays keeps both measurements with their own substrate
    counts. Splitting by array makes the old ``all`` / ``all_raw`` distinction
    vanish: within one array every kinase appears exactly once, so there is
    nothing left to deduplicate. There is no separate significance file either --
    the ``Significant`` column carries that, and the rows are ordered by |Z|.

    Empty tables are skipped rather than written as header-only files.
    
    Args:
        peptide_output: Directory receiving the peptide table.
        kinase_output: Directory receiving the kinase tables.
        pathway_output: Directory receiving the pathway tables.
        output_mode: ``'user'`` for the short tables, ``'developer'`` for the full
            ones (see :func:`project_output_columns`).
    
    Returns:
        list[Path]: The files written, in write order.
    """
    output_mode = resolve_output_mode(output_mode)
    comparison = f"{control}_{condition}"

    tables = [
        (
            peptide_output,
            f"peptide_statistics_{comparison}",
            "peptides",
            collapse_peptide_rows_for_export(condition_results["peptide_statistics"]),
        )
    ]

    # One kinase file per array. UKA_raw holds one row per (kinase, array); the
    # ranked all_kinases / significant_kinases stay in memory for the pathway and
    # venn stages, which work on the combined set.
    #
    # The comparison goes INTO the table, not only into the file name: the file
    # name joins both labels with '_' and the labels contain '_' themselves
    # ('kinases_PTK_STR_CTL_STR_Exer.csv'), so it cannot be split back into
    # control and test. The family table has carried this column from the start;
    # the kinase table did not, which left the reference arm unrecorded.
    comparison_label = f"{control}_vs_{condition}"

    def _with_comparison(frame):
        """Return the frame with 'Comparison' as its first column."""
        if frame is None or frame.empty:
            return frame
        out = frame.copy()
        if "Comparison" in out.columns:
            out = out.drop(columns=["Comparison"])
        out.insert(0, "Comparison", comparison_label)
        return out

    df_kinases = condition_results["UKA_raw"]
    if df_kinases is not None and not df_kinases.empty and "Type" in df_kinases.columns:
        for array, group in df_kinases.groupby("Type", sort=True):
            tables.append(
                (
                    kinase_output,
                    f"kinases_{array}_{comparison}",
                    "kinases",
                    _with_comparison(rank_by_absolute_z(group)),
                )
            )
    else:
        tables.append(
            (
                kinase_output,
                f"kinases_{comparison}",
                "kinases",
                _with_comparison(rank_by_absolute_z(df_kinases)),
            )
        )

    tables.extend(
        [
            (pathway_output, f"pathways_KEGG_{comparison}", "pathways", condition_results["pathways_KEGG"]),
            (pathway_output, f"pathways_WikiPathways_{comparison}", "pathways", condition_results["pathways_WP"]),
            (pathway_output, f"pathways_Reactome_{comparison}", "pathways", condition_results["pathways_REAC"]),
        ]
    )

    written = []
    for output_dir, name, table, df in tables:
        if df is None or df.empty:
            continue
        exported = project_output_columns(df, table, output_mode)
        output_path = Path(output_dir) / f"{name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported.to_csv(output_path, index=False)
        written.append(output_path)
    return written


def plot_control_condition_venn_diagrams(
    all_results,
    kinase_analysis,
    pathway_analysis,
    kinase_output_path,
    pathway_output_path,
    save_tables=True,
):
    """Generate venn-diagram summaries for control-condition overlaps.
    
    Args:
        kinase_output_path: Directory receiving the kinase overlap outputs.
        pathway_output_path: Directory receiving the pathway overlap outputs.
    """
    if not all_results:
        return {}

    print("[4]  Plotting overlap diagrams for control-vs-condition comparisons...")
    kinase_output_path = Path(kinase_output_path)
    pathway_output_path = Path(pathway_output_path)
    kinase_output_path.mkdir(parents=True, exist_ok=True)
    pathway_output_path.mkdir(parents=True, exist_ok=True)

    venn_outputs = {}
    kinase_outputs = kinase_analysis.plot_kinase_overlap_venn(
        kinase_results_by_condition=all_results,
        output_path=kinase_output_path,
        save_tables=save_tables,
    )
    if kinase_outputs:
        venn_outputs["kinases"] = kinase_outputs

    pathway_outputs = pathway_analysis.plot_pathway_overlap_venn(
        pathway_results_by_condition=all_results,
        output_path=pathway_output_path,
        save_tables=save_tables,
    )
    if pathway_outputs:
        venn_outputs["pathways"] = pathway_outputs

    if venn_outputs:
        print(
            f"     Kinase overlap outputs saved to: {kinase_output_path}\n"
            f"     Pathway overlap outputs saved to: {pathway_output_path}"
        )
    else:
        print("     No non-empty kinase or pathway groups found for Venn/overlap diagrams.")
    print("\n=====================================================================================\n")
    return venn_outputs


def render_uka_outputs_excluded_from_timing(
    all_results,
    peptide_analysis,
    kinase_analysis,
    pathway_analysis,
    resolved_params,
):
    """Render optional downstream outputs excluded from the main timing block."""
    should_render_stage_outputs = any(
        (
            resolved_params["peptide_waterfall_plot"],
            resolved_params["peptide_heatmap_plot"],
            resolved_params["pathway_heatmap_plot"],
        )
    )
    should_render_venn = resolved_params["venn_plot"]

    if not should_render_stage_outputs and not should_render_venn:
        return {}

    print("\n" + "=" * 80)
    print("Generating plots and overlap outputs (excluded from timed analysis)...")
    print("=" * 80)
    tic_outputs = time.time()

    for comparison_key, condition_results in all_results.items():
        control_condition = condition_results["control_condition"]
        condition = condition_results["condition"]

        if (
            resolved_params["peptide_waterfall_plot"]
            or resolved_params["peptide_heatmap_plot"]
        ):
            # Dedup on (Type, ID) -- see PeptideStatistics._compute_condition_result:
            # a peptide repeats per mapped protein, but one printed on BOTH arrays
            # has two real measurements and each belongs in its own heatmap panel.
            df_peptides_plot = (
                condition_results["peptide_statistics"]
                .sort_values("p_value", ascending=True)
                .drop_duplicates(subset=["Type", "ID"], keep="first")
                .reset_index(drop=True)
            )

            if resolved_params["peptide_waterfall_plot"]:
                peptide_analysis._plot_waterfall(
                    df_peptides=df_peptides_plot,
                    output_path=resolved_params["peptide_waterfall_plot_output"],
                    control=control_condition,
                    condition=condition,
                )
            if resolved_params["peptide_heatmap_plot"]:
                peptide_analysis._plot_heatmap(
                    df_peptides=df_peptides_plot,
                    output_path=resolved_params["peptide_heatmap_plot_output"],
                    control=control_condition,
                    condition=condition,
                )

        if resolved_params["pathway_heatmap_plot"]:
            pathway_analysis._plot_heatmap(
                significant_kinases=condition_results["significant_kinases"],
                pathways={
                    "KEGG": condition_results["pathways_KEGG"],
                    "WP": condition_results["pathways_WP"],
                    "REAC": condition_results["pathways_REAC"],
                },
                output_path=resolved_params["pathway_heatmap_plot_output"],
                control=control_condition,
                condition=condition,
            )

    # All comparisons of the run in ONE heatmap per pathway source (union of
    # pathways and kinases, shared colorbar); nothing to compare with one.
    if resolved_params["pathway_heatmap_plot"] and len(all_results) > 1:
        pathway_analysis._plot_comparison_heatmaps(
            all_results=all_results,
            output_path=resolved_params["pathway_heatmap_plot_output"],
        )

    venn_diagrams = {}
    if should_render_venn:
        venn_diagrams = plot_control_condition_venn_diagrams(
            all_results=all_results,
            kinase_analysis=kinase_analysis,
            pathway_analysis=pathway_analysis,
            kinase_output_path=resolved_params["kinase_venn_plot_output"],
            pathway_output_path=resolved_params["pathway_venn_plot_output"],
            save_tables=resolved_params["venn_plot_tables"],
        )

    output_duration = time.time() - tic_outputs
    print(
        f"Plots and overlap outputs completed in {output_duration:.2f} seconds "
        "(excluded from timed analysis)."
    )
    print("=" * 80 + "\n")
    return venn_diagrams


def run_image_analysis_pipeline(
    ptk_selection: Optional[Tuple[int, int]] = None,
    stk_selection: Optional[Tuple[int, int]] = None,
    base_data_dir=None,
):
    """Run the PTK/STK import, enrichment, and image-processing stages.
    
    This is the bridge between the user-facing pipeline entry points and the
    image-centric core of pyKinaXe. It returns the live loader/enricher/
    processor objects because later stages need both their outputs and some of
    their metadata and path context.
    """
    analysis_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"Analysis session timestamp: {analysis_timestamp}")

    # Build or auto-discover the paired PTK and STK runs that define one
    # complete PamGene analysis session.
    if ptk_selection is None:
        loader_PTK = DataLoader(timestamp=analysis_timestamp)
    else:
        loader_PTK = build_loader_from_selection(
            selection=ptk_selection,
            timestamp=analysis_timestamp,
            expected_peptide_type="PTK",
            base_data_dir=base_data_dir,
        )

    if stk_selection is None:
        loader_STK = DataLoader(timestamp=analysis_timestamp)
    else:
        loader_STK = build_loader_from_selection(
            selection=stk_selection,
            timestamp=analysis_timestamp,
            expected_peptide_type="STK",
            base_data_dir=base_data_dir,
        )

    # Import both chips before deciding where the shared output tree should
    # live. The loaders resolve the true experiment/data directories during
    # ``load_data()``.
    loader_PTK.load_data()
    loader_STK.load_data()
    shared_results_context = resolve_shared_results_context(loader_PTK, loader_STK)

    print("\n" + "=" * 80)
    print("Starting Image Analysis Pipeline...")
    print("=" * 80)
    tic_pipeline = time.time()

    # The enricher turns the two raw annotation tables into one consistent
    # experimental design table that all downstream analyses use.
    enricher = DataEnricher(loader_PTK, loader_STK)
    enricher.enrich_data()

    # Each chip type is processed independently but within the same timestamped
    # session so their outputs remain comparable and co-located.
    processor_PTK = ImageProcessor(loader_PTK)
    processor_STK = ImageProcessor(loader_STK)
    processor_PTK.process()
    processor_STK.process()

    pipeline_duration = time.time() - tic_pipeline
    experiment_name = shared_results_context["experiment_name"]
    results_parent_relpath = shared_results_context["results_parent_relpath"]

    print("\n" + "=" * 80)
    print(
        f"Image Analysis Pipeline COMPLETED in {pipeline_duration:.2f} seconds "
        f"({pipeline_duration / 60:.2f} minutes)"
    )
    print("=" * 80 + "\n")

    return (
        loader_PTK,
        loader_STK,
        enricher,
        processor_PTK,
        processor_STK,
        analysis_timestamp,
        pipeline_duration,
        experiment_name,
        results_parent_relpath,
    )


def run_peptide_statistics_analysis(
    enricher,
    processor_PTK,
    processor_STK,
    analysis_timestamp,
    experiment_name,
    results_parent_relpath=None,
    uka_params=None,
):
    """Run stage 1 of the downstream analysis: peptide-level statistics.
    
    Args:
        analysis_timestamp: Timestamp string assigned to the current analysis run.
    """
    resolved_params = resolve_uka_params(
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=uka_params,
    )

    peptide_analysis = PeptideStatistics(
        file_enrichment=enricher.enriched_table,
        df_ptk=processor_PTK.final_output_bn,
        df_stk=processor_STK.final_output_bn,
        path_file_enrichment_peptides=resolved_params["path_file_enrichment_peptides"],
        waterfall_plot=resolved_params["peptide_waterfall_plot"],
        waterfall_plot_output=resolved_params["peptide_waterfall_plot_output"],
        heatmap_plot=resolved_params["peptide_heatmap_plot"],
        heatmap_plot_output=resolved_params["peptide_heatmap_plot_output"],
        waterfall_lfc_cutoffs=resolved_params["kpea_lfc_cutoffs"],
        waterfall_cutoff_mode=resolved_params["kpea_cutoff_mode"],
        waterfall_primary_lfc_cutoff=resolved_params["kpea_primary_lfc_cutoff"],
        path_output_peptide_statistic=resolved_params["path_output_peptide_statistic"],
        log_output=resolved_params["log_output"],
        use_limma=resolved_params["use_limma"],
        debugging_print=resolved_params["debugging_print"],
        log2_slope_mode=resolved_params["log2_slope_mode"],
        batch_correction=resolved_params["batch_correction"],
        batch_correction_method=resolved_params["batch_correction_method"],
        batch_column=resolved_params["batch_column"],
        array_normalization=resolved_params["array_normalization"],
        array_normalization_method=resolved_params["array_normalization_method"],
        qc_mode=resolved_params["qc_mode"],
        qc_krsa_signal_threshold=resolved_params["qc_krsa_signal_threshold"],
        qc_krsa_r2_threshold=resolved_params["qc_krsa_r2_threshold"],
        # Must be forwarded explicitly: the pipeline passes every parameter, so
        # a missing kwarg here silently pins the class default and makes the
        # config setting a no-op (same trap as qc_mode had, see the BUGFIX note
        # in PeptideStatistics._build_condition_worker).
        contrast_mode=resolved_params["contrast_mode"],
        contrasts=resolved_params["contrasts"],
    )
    return peptide_analysis.run_peptide_statistics(), peptide_analysis


def run_kinase_analysis(
    peptide_results,
    analysis_timestamp,
    experiment_name,
    results_parent_relpath=None,
    uka_params=None,
):
    """Run stage 2 of the downstream analysis: upstream kinase scoring.
    
    Args:
        analysis_timestamp: Timestamp string assigned to the current analysis run.
    """
    resolved_params = resolve_uka_params(
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=uka_params,
    )

    def _build_kinase_analysis():
        """Build the kinase-analysis object for one condition comparison."""
        return KinaseActivityAnalysis(
            path_file_enrichment_peptides=resolved_params["path_file_enrichment_peptides"],
            BLAST_threshold=resolved_params["BLAST_threshold"],
            input_stk_ptm_path=resolved_params["input_stk_ptm_path"],
            input_ptk_ptm_path=resolved_params["input_ptk_ptm_path"],
            use_verified_interactions_only=resolved_params["use_verified_interactions_only"],
            verified_evidence_levels=resolved_params["verified_evidence_levels"],
            verified_min_score=resolved_params["verified_min_score"],
            verified_min_references=resolved_params["verified_min_references"],
            require_known_ptm_site=resolved_params["require_known_ptm_site"],
            path_output_peptide_statistic=resolved_params["path_output_peptide_statistic"],
            debugging_print=resolved_params["debugging_print"],
            kpea_lfc_cutoffs=resolved_params["kpea_lfc_cutoffs"],
            kpea_cutoff_mode=resolved_params["kpea_cutoff_mode"],
            kpea_primary_lfc_cutoff=resolved_params["kpea_primary_lfc_cutoff"],
            kpea_substrate_cutoff=resolved_params["kpea_substrate_cutoff"],
            kpea_zscore_threshold=resolved_params["kpea_zscore_threshold"],
            kpea_z_cap=resolved_params["kpea_z_cap"],
            kpea_background_universe=resolved_params["kpea_background_universe"],
            kpea_chip_numbers=resolved_params["kpea_chip_numbers"],
        )

    def _prime_kinase_worker(base_worker, worker):
        """Warm up one kinase-analysis worker before parallel execution.
        
        Returns:
            object: Primed kinase worker.
        """
        worker._peptide_enrichment_cache = base_worker._peptide_enrichment_cache
        worker._blast_cache = base_worker._blast_cache
        worker._blast_peptide_to_proteins_cache = (
            base_worker._blast_peptide_to_proteins_cache
        )
        worker._ptm_cache = base_worker._ptm_cache
        worker._substrate_lookup_cache = dict(base_worker._substrate_lookup_cache)
        # Shared by REFERENCE (not copied): one UniProt lookup per run instead
        # of one per comparison, and one consistent name everywhere.
        worker._kinase_name_cache = base_worker._kinase_name_cache
        worker._kinase_name_cache_lock = base_worker._kinase_name_cache_lock
        worker._kinase_name_unresolved = base_worker._kinase_name_unresolved
        return worker

    def _resolve_run_chip_numbers(base_worker):
        """Resolve one PamChip version per array for the whole run.

        The chip version decides the KPEA null population. Left to
        ``KinaseActivityAnalysis._resolve_chip_number`` it is inferred from the
        QC-passed peptides of ONE comparison, so a comparison that lost the few
        peptides separating two chip generations is scored against a different M
        and K than its siblings -- silently, and depending on which comparison the
        thread pool happens to score first. The union of every comparison's
        peptide ids is a property of the RUN and is order-independent.

        Args:
            base_worker: The primed analysis object; its enrichment cache is reused.

        Returns:
            dict: ``{"PTK": "86412", ...}``. Explicit pins are kept unchanged;
                arrays that cannot be resolved are left out, so the worker warns
                and falls back exactly as before.
        """
        pinned = {
            str(key).strip().upper(): str(value).strip()
            for key, value in (resolved_params["kpea_chip_numbers"] or {}).items()
            if value not in (None, "")
        }
        enrichment = base_worker._get_peptide_enrichment()
        if enrichment is None or "family" not in enrichment.columns:
            return pinned
        measured = {"PTK": set(), "STK": set()}
        for payload in peptide_results.values():
            df_peptides = (
                payload.get("peptide_statistics")
                if isinstance(payload, dict)
                else payload
            )
            if df_peptides is None or "Type" not in df_peptides.columns:
                continue
            for array_type in measured:
                subset = df_peptides[df_peptides["Type"] == array_type]
                measured[array_type] |= {str(pid) for pid in subset["ID"]}
        family = enrichment["family"].astype(str).str.strip().str.upper()
        for array_type, measured_ids in measured.items():
            if array_type in pinned or not measured_ids:
                continue
            chip_reference = enrichment[family == array_type]
            if chip_reference.empty:
                continue
            chip_number = base_worker._resolve_chip_number(
                array_type, measured_ids, chip_reference
            )
            if chip_number is not None:
                pinned[array_type] = str(chip_number)
        return pinned

    def _run_condition_kinase_analysis(condition, payload, base_worker):
        """Run kinase analysis for one condition comparison."""
        worker = _prime_kinase_worker(base_worker, _build_kinase_analysis())
        # `condition` is the comparison KEY (e.g. "mock_vs_pSHDAg"); the label
        # the analysis puts into file names comes from the payload, so a run with
        # a single control produces exactly the same names as before.
        result = worker.run_kinase_analysis(
            peptide_statistics=payload,
            control=payload["control_condition"],
            condition=payload["condition"],
        )
        return condition, result

    # Prime one base analysis object with external resources once, then clone
    # those cached resources into per-condition workers to avoid repeating the
    # most expensive lookup and parsing steps.
    kinase_analysis = _build_kinase_analysis()
    kinase_analysis._get_peptide_enrichment()
    df_blast = kinase_analysis._get_blast_data()
    kinase_analysis._get_peptide_to_proteins(df_blast)
    df_ptm_stk, df_ptm_ptk = kinase_analysis._get_ptm_data()
    kinase_analysis._get_substrate_to_kinase_lookup(df_ptm_stk, cache_key="STK")
    kinase_analysis._get_substrate_to_kinase_lookup(df_ptm_ptk, cache_key="PTK")

    # Resolve the chip version ONCE, from every comparison of this run, and pin it
    # so that all per-condition workers score against the same null population.
    # `_build_kinase_analysis` reads `resolved_params` at call time, so the pins
    # reach every worker created below.
    resolved_params["kpea_chip_numbers"] = _resolve_run_chip_numbers(kinase_analysis)
    kinase_analysis.kpea_chip_numbers = dict(resolved_params["kpea_chip_numbers"])
    print(
        "     KPEA chip versions used for this run: "
        f"{resolved_params['kpea_chip_numbers'] or 'none resolved'}"
    )

    kinase_results = {}
    condition_items = list(peptide_results.items())
    if len(condition_items) > 1:
        max_workers = min(len(condition_items), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_condition_kinase_analysis,
                    condition,
                    payload,
                    kinase_analysis,
                )
                for condition, payload in condition_items
            ]
            for future in futures:
                condition, result = future.result()
                kinase_results[condition] = result
    else:
        for condition, payload in condition_items:
            _, result = _run_condition_kinase_analysis(
                condition,
                payload,
                kinase_analysis,
            )
            kinase_results[condition] = result

    return kinase_results, kinase_analysis


def build_mappable_kinase_background(kinase_analysis, peptide_results, kinase_results):
    """Collect the kinases of the ``'mappable_kinases'`` pathway background.

    For each array the run measured, every kinase the PTM data maps to at least
    ``kpea_substrate_cutoff`` peptides printed on the run's chip version
    (:meth:`KinaseActivityAnalysis.mappable_kinases`). The kinases scored in the
    run belong to that set by construction; they are added explicitly, so the
    significant kinases always lie inside the background, also when a chip-wide
    mapping cannot be built.

    Args:
        kinase_analysis: The run's kinase-analysis object, chip versions pinned.
        peptide_results: Stage-1 payloads; they decide which arrays were measured.
        kinase_results: Stage-2 payloads with their ``all_kinases_raw`` tables.

    Returns:
        pd.DataFrame: One row per kinase and array: ``Kinase``, ``Gene_Symbol``
            (the id sent to g:Profiler), ``Type`` and ``NumChipSubstrates`` (NaN
            for a scored kinase missing from the chip mapping).
    """
    arrays = set()
    for payload in peptide_results.values():
        df_peptides = (
            payload.get("peptide_statistics") if isinstance(payload, dict) else payload
        )
        if df_peptides is not None and "Type" in df_peptides.columns:
            arrays |= set(df_peptides["Type"].dropna().astype(str))
    scored = {}
    for result in kinase_results.values():
        df_kinases = result.get("all_kinases_raw")
        if df_kinases is None or df_kinases.empty or "Type" not in df_kinases.columns:
            continue
        for array_type, group in df_kinases.groupby("Type"):
            scored.setdefault(str(array_type), set()).update(group["Kinase"].astype(str))

    rows = []
    for array_type in sorted(arrays | set(scored)):
        on_chip = kinase_analysis.mappable_kinases(array_type)
        if on_chip is None:
            print(
                f"     WARNING: no chip-wide kinase mapping for {array_type}; the "
                f"pathway background holds only the {array_type} kinases scored "
                "in this run."
            )
            on_chip = {}
        for kinase in sorted(set(on_chip) | scored.get(array_type, set())):
            rows.append(
                {
                    "Kinase": kinase,
                    "Type": array_type,
                    "NumChipSubstrates": on_chip.get(kinase, math.nan),
                }
            )
    df_background = pd.DataFrame(rows, columns=["Kinase", "Type", "NumChipSubstrates"])
    gene_symbols = gprofiler_gene_symbols(df_background["Kinase"])
    df_background.insert(1, "Gene_Symbol", df_background["Kinase"].map(gene_symbols))
    return df_background


def run_pathway_enrichment(
    kinase_results,
    peptide_results,
    analysis_timestamp,
    experiment_name,
    results_parent_relpath=None,
    uka_params=None,
    kinase_analysis=None,
):
    """Run stage 3 of the downstream analysis: pathway enrichment.
    
    Args:
        analysis_timestamp: Timestamp string assigned to the current analysis run.
        kinase_analysis: The run's kinase-analysis object; needed only for
            ``pathway_background = 'mappable_kinases'``.
    """
    resolved_params = resolve_uka_params(
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=uka_params,
    )

    # One background for every comparison of the run, so their pathway results
    # stay comparable. 'annotated' needs no list: g:Profiler's own domain.
    background_kinases = None
    if str(resolved_params["pathway_background"]).lower() == "mappable_kinases":
        if kinase_analysis is None:
            raise ValueError(
                "pathway_background 'mappable_kinases' needs the run's "
                "KinaseActivityAnalysis (kinase_analysis)."
            )
        df_background = build_mappable_kinase_background(
            kinase_analysis, peptide_results, kinase_results
        )
        background_kinases = df_background["Kinase"].drop_duplicates().tolist()
        background_path = (
            Path(resolved_params["pathway_results_output"])
            / "pathway_background_kinases.csv"
        )
        background_path.parent.mkdir(parents=True, exist_ok=True)
        df_background.to_csv(background_path, index=False)
        per_array = ", ".join(
            f"{array_type} {count}"
            for array_type, count in df_background.groupby("Type")["Kinase"]
            .nunique()
            .items()
        )
        chip_versions = kinase_analysis.kpea_chip_numbers or "unresolved"
        print(
            f"     Pathway background: {len(background_kinases)} mappable kinases "
            f"({per_array}), chip versions {chip_versions}; list saved to "
            f"{background_path}"
        )

    def _build_pathway_analysis():
        """Build the pathway-analysis object for one condition comparison."""
        return PathwayEnrichmentAnalysis(
            significance_level_pathways=resolved_params["significance_level_pathways"],
            heatmap_plot=resolved_params["pathway_heatmap_plot"],
            heatmap_plot_output=resolved_params["pathway_heatmap_plot_output"],
            uka_visualization_metric=resolved_params["uka_visualization_metric"],
            pathway_background=resolved_params["pathway_background"],
            background_kinases=background_kinases,
            debugging_print=resolved_params["debugging_print"],
        )

    pathway_analysis = _build_pathway_analysis()

    def _analyze_pathway_condition(condition, payload):
        """Run pathway enrichment for one condition comparison."""
        worker = _build_pathway_analysis()
        control_condition = None
        condition_label = condition
        peptide_payload = peptide_results.get(condition)
        if peptide_payload is not None:
            control_condition = peptide_payload.get("control_condition")
            condition_label = peptide_payload.get("condition", condition)
        result = worker.run_pathway_enrichment(
            significant_kinases=payload["significant_kinases"],
            control=control_condition,
            condition=condition_label,
        )
        return condition, result

    pathway_results = {}
    condition_items = list(kinase_results.items())
    if len(condition_items) > 1:
        max_workers = min(len(condition_items), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_analyze_pathway_condition, condition, payload)
                for condition, payload in condition_items
            ]
            for future in futures:
                condition, result = future.result()
                pathway_results[condition] = result
    else:
        for condition, payload in condition_items:
            _, result = _analyze_pathway_condition(condition, payload)
            pathway_results[condition] = result

    return pathway_results, pathway_analysis


def run_uka_analysis(
    enricher,
    processor_PTK,
    processor_STK,
    analysis_timestamp,
    experiment_name,
    results_parent_relpath=None,
    uka_params=None,
):
    """Run the full staged downstream analysis and save the per-topic CSV tables.
    
    The returned ``analysis_bundle`` keeps the live stage objects around for
    optional post-processing such as venn diagrams or custom visualizations.
    
    Args:
        analysis_timestamp: Timestamp string assigned to the current analysis run.
    """
    resolved_params = resolve_uka_params(
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=uka_params,
    )
    timed_uka_params = build_timed_uka_params(uka_params=uka_params)

    # Written first, so a run that fails later still records its settings.
    run_config_path = write_run_config(
        build_results_dir(
            analysis_timestamp,
            experiment_name,
            results_parent_relpath=results_parent_relpath,
        )
        / f"{analysis_timestamp}_downstream_analysis",
        resolved_params,
    )
    print(f"Run settings written to {run_config_path}")

    print("\n" + "=" * 80)
    print("Starting staged UKA/KPEA analysis...")
    print("=" * 80)
    tic_uka = time.time()

    # Execute the three conceptual downstream stages in order. Each stage
    # builds on the results of the previous one.
    peptide_results, peptide_analysis = run_peptide_statistics_analysis(
        enricher=enricher,
        processor_PTK=processor_PTK,
        processor_STK=processor_STK,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=timed_uka_params,
    )
    kinase_results, kinase_analysis = run_kinase_analysis(
        peptide_results=peptide_results,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=timed_uka_params,
    )
    pathway_results, pathway_analysis = run_pathway_enrichment(
        kinase_results=kinase_results,
        peptide_results=peptide_results,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=timed_uka_params,
        kinase_analysis=kinase_analysis,
    )

    # Repackage per-condition stage outputs into one export-oriented structure
    # and persist the canonical CSV tables in their per-topic results folders.
    all_results = {}
    for condition, peptide_payload in peptide_results.items():
        control_condition = peptide_payload["control_condition"]
        # The dict key identifies the comparison; the label that goes into file
        # names and plot titles is the test condition of the pair.
        condition_label = peptide_payload["condition"]
        combined_result = {
            "control_condition": control_condition,
            "condition": condition_label,
            "construct": peptide_payload["construct"],
            "peptide_statistics": peptide_payload["peptide_statistics"],
            "kinase_analysis": kinase_results[condition],
            "pathway_enrichment": pathway_results[condition],
            "peptides": peptide_payload["peptide_statistics"],
            "all_kinases": kinase_results[condition]["all_kinases"],
            "significant_kinases": kinase_results[condition]["significant_kinases"],
            "UKA": kinase_results[condition]["all_kinases"],
            "UKA_raw": kinase_results[condition]["all_kinases_raw"],
            "pathways_KEGG": pathway_results[condition]["pathways_KEGG"],
            "pathways_WP": pathway_results[condition]["pathways_WP"],
            "pathways_REAC": pathway_results[condition]["pathways_REAC"],
        }
        save_condition_results_to_csv(
            condition_results=combined_result,
            peptide_output=resolved_params["peptide_results_output"],
            kinase_output=resolved_params["kinase_results_output"],
            pathway_output=resolved_params["pathway_results_output"],
            control=control_condition,
            condition=condition_label,
            output_mode=resolved_params["output_mode"],
        )
        all_results[condition] = combined_result

    # Extra, self-contained kinase-FAMILY analysis (KRSA-like, at the family
    # level). It runs inside the timed block, so the runtime summary covers both
    # KPEA layers (families and single kinases). It writes its own CSV tables,
    # and its results are not fed into any downstream stage (pathways/plots)
    # here. Wrapped so a failure never aborts the main single-kinase pipeline.
    try:
        from kx_upstream_kinase_families_analysis import run_kinase_families_analysis

        run_kinase_families_analysis(
            peptide_results,
            resolved_params["family_results_output"],
            output_mode=resolved_params["output_mode"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: kinase-family analysis failed (skipped): {exc}")

    uka_duration = time.time() - tic_uka
    print("\n" + "=" * 80)
    print(
        f"Staged UKA/KPEA analysis COMPLETED in {uka_duration:.2f} seconds "
        f"({uka_duration / 60:.2f} minutes)"
    )
    print("=" * 80 + "\n")

    # Plot-heavy overlap outputs are generated after the timed core analysis so
    # that the runtime summaries reflect the analysis itself.
    venn_diagrams = render_uka_outputs_excluded_from_timing(
        all_results=all_results,
        peptide_analysis=peptide_analysis,
        kinase_analysis=kinase_analysis,
        pathway_analysis=pathway_analysis,
        resolved_params=resolved_params,
    )

    analysis_bundle = {
        "peptide_statistics": peptide_analysis,
        "kinase_analysis": kinase_analysis,
        "pathway_enrichment": pathway_analysis,
        "results_by_condition": all_results,
        "venn_diagrams": venn_diagrams,
    }

    if len(all_results) == 1:
        single_condition = next(iter(all_results.values()))
        return (
            (
                single_condition["UKA"],
                single_condition["pathways_KEGG"],
                single_condition["pathways_WP"],
                single_condition["pathways_REAC"],
            ),
            analysis_bundle,
            uka_duration,
        )

    return all_results, analysis_bundle, uka_duration


def create_publication_figures(
    processor_PTK,
    processor_STK,
    num_representative_images=1,
):
    """Render representative publication-style QC figures for PTK and STK."""
    print("\n" + "=" * 80)
    print("Creating publication-quality figures...")
    print("=" * 80)

    for representative_image_idx in range(num_representative_images):
        print(f"\nGenerating publication figure for PTK image {representative_image_idx}...")
        fig_ptk, _ = processor_PTK.create_publication_figure(
            image_idx=representative_image_idx,
            figsize=(16, 16),
            dpi=300,
            save_images=True,
            cross_section_type="horizontal",
            cross_section_offset=-80,
        )
        plt.close(fig_ptk)

        print(f"\nGenerating publication figure for STK image {representative_image_idx}...")
        fig_stk, _ = processor_STK.create_publication_figure(
            image_idx=representative_image_idx,
            figsize=(16, 16),
            dpi=300,
            save_images=True,
            cross_section_type="horizontal",
            cross_section_offset=-80,
        )
        plt.close(fig_stk)

    print("=" * 80)
    print("Publication figures created successfully!")
    print("=" * 80)


def create_processing_stage_figures(
    processor_PTK,
    processor_STK,
    all_images=True,
    fraction=None,
    image_limit=None,
):
    """Render per-image intermediate processing figures for QC review.

    Args:
        all_images: When True (and no ``fraction`` below 1.0 is given), render
            figures for every image; when False, render only the first image.
        fraction: Optional float. When set below 1.0 it takes precedence over
            ``all_images`` and renders a contiguous prefix of ``ceil(n * fraction)``
            images (at least 1) per processor. ``None`` or ``>= 1.0`` defers to
            ``all_images``. Example: 0.3333 renders roughly the first third.
        image_limit: Optional hard cap on the number of images rendered.
    """
    print("\n" + "=" * 80)
    print("Creating per-image processing-stage figures...")
    print("=" * 80)

    def _image_indices(processor):
        """Normalize the requested image-index selection into a list of indices."""
        n_images = int(processor.original_images.sizes["image_idx"])
        if n_images <= 0:
            return range(0)
        frac = None if fraction is None else float(fraction)
        if frac is not None and frac < 1.0:
            # Render a contiguous prefix covering the requested fraction of the
            # run (at least one image). This takes precedence over all_images.
            limit = max(1, math.ceil(n_images * frac))
            if image_limit is not None:
                limit = min(limit, int(image_limit))
        elif all_images:
            # Every image. ``fraction`` is None or >= 1.0 here, which per the
            # documented contract defers to ``all_images``.
            limit = n_images if image_limit is None else min(n_images, int(image_limit))
        else:
            limit = 1 if image_limit is None else min(n_images, int(image_limit))
        return range(min(limit, n_images))

    for image_idx in _image_indices(processor_PTK):
        print(f"\nGenerating processing-stage figures for PTK image {image_idx}...")
        processor_PTK.visualize_processing_stages(
            image_idx=image_idx,
            save_images=True,
            show_spot_grid=True,
        )
        plt.close("all")

    for image_idx in _image_indices(processor_STK):
        print(f"\nGenerating processing-stage figures for STK image {image_idx}...")
        processor_STK.visualize_processing_stages(
            image_idx=image_idx,
            save_images=True,
            show_spot_grid=True,
        )
        plt.close("all")

    print("=" * 80)
    print("Per-image processing-stage figures created successfully!")
    print("=" * 80)


def run_terminal_pipeline(
    *,
    ptk_selection: Optional[Tuple[int, int]] = None,
    stk_selection: Optional[Tuple[int, int]] = None,
    base_data_dir=None,
    create_publication_figures_flag: bool = CREATE_PUBLICATION_FIGURES,
    num_representative_images: int = NUM_REPRESENTATIVE_IMAGES,
    create_processing_stage_figures_flag: bool = CREATE_PROCESSING_STAGE_FIGURES,
    processing_stage_figures_all_images: bool = PROCESSING_STAGE_FIGURES_ALL_IMAGES,
    processing_stage_figures_fraction: Optional[float] = PROCESSING_STAGE_FIGURES_FRACTION,
    processing_stage_figure_image_limit=PROCESSING_STAGE_FIGURE_IMAGE_LIMIT,
):
    """Convenience wrapper that executes the full terminal workflow.
    
    The script entry point now spells the top-level stages out directly in its
    own ``main()``, but this wrapper remains useful for tests, notebooks, or
    local helper code that wants one callable for the default terminal flow.
    
    Args:
        processing_stage_figures_fraction (Optional[float]): Fraction of images (0 < f < 1) to render as a contiguous prefix; takes precedence over all_images when set below 1.0.
    """
    (
        loader_PTK,
        loader_STK,
        enricher,
        processor_PTK,
        processor_STK,
        analysis_timestamp,
        pipeline_duration,
        experiment_name,
        results_parent_relpath,
    ) = run_image_analysis_pipeline(
        ptk_selection=ptk_selection,
        stk_selection=stk_selection,
        base_data_dir=base_data_dir,
    )

    # Render processing-stage figures right after image analysis and BEFORE the
    # downstream analysis, so image-QC figures are still produced even if
    # UKA/KPEA later fails. The processors are already fully processed above.
    if create_processing_stage_figures_flag:
        create_processing_stage_figures(
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            all_images=processing_stage_figures_all_images,
            fraction=processing_stage_figures_fraction,
            image_limit=processing_stage_figure_image_limit,
        )

    results, analysis_bundle, uka_duration = run_uka_analysis(
        enricher=enricher,
        processor_PTK=processor_PTK,
        processor_STK=processor_STK,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
    )

    total_duration = pipeline_duration + uka_duration
    print("\n" + "=" * 80)
    print("TOTAL ANALYSIS TIME SUMMARY")
    print("=" * 80)
    print(
        f"  Image Analysis Pipeline: {pipeline_duration:.2f}s "
        f"({pipeline_duration / 60:.2f} min)"
    )
    print(
        f"  Staged UKA/KPEA Analysis: {uka_duration:.2f}s "
        f"({uka_duration / 60:.2f} min)"
    )
    print(f"  TOTAL TIME: {total_duration:.2f}s ({total_duration / 60:.2f} min)")
    print("=" * 80 + "\n")

    # Processing-stage figures are rendered earlier, before UKA/KPEA.
    if create_publication_figures_flag:
        create_publication_figures(
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            num_representative_images=num_representative_images,
        )

    return {
        "loaders": (loader_PTK, loader_STK),
        "enricher": enricher,
        "processors": (processor_PTK, processor_STK),
        "analysis_bundle": analysis_bundle,
        "uka_results": results,
    }
