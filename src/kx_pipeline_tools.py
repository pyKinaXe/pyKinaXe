"""Shared orchestration helpers for the terminal and web pyKinaXe workflows.

This module does not implement the scientific algorithms themselves. Instead,
it coordinates the main analysis stages exposed by the lower-level modules:

- ``kx_data_importer`` for input discovery and TIFF/layout loading
- ``kx_data_enricher`` for PTK/STK sample-design enrichment
- ``kx_image_processor`` for geometry detection and intensity extraction
- ``kx_peptide_analysis`` for peptide statistics
- ``kx_upstream_kinase_analysis`` for kinase-level scoring
- ``kx_pathway_enrichment_analysis`` for pathway enrichment

In practice, this file is the reusable backbone behind:

- the terminal entry point in ``scripts/kx_kinase_extraction_pipeline.py``
- the non-interactive web workflow in
  ``webapp/backend/kx_web_kinase_extraction_pipeline.py``

Keeping the orchestration here makes the entry points easier to read while
keeping the scientific stages reusable.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
)
from kx_data_enricher import DataEnricher  # noqa: E402
from kx_data_importer import DataLoader  # noqa: E402
from kx_image_processor import ImageProcessor  # noqa: E402
from kx_pathway_enrichment_analysis import PathwayEnrichmentAnalysis  # noqa: E402
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
        experiment_name (str): Experiment name used by this function.
        results_parent_relpath (Optional[Path]): Path-like value for results parent relpath.
    
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
    
    Args:
        loader_ptk: Loader PTK processed by this function.
        loader_stk: Loader STK processed by this function.
    
    Returns:
        dict[str, Path | str]: Resolved shared results context.
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
    """Resolve the base data directory used for folder discovery.
    
    Args:
        base_data_dir: Directory containing or receiving the base data.
    
    Returns:
        Path: Resolved base data dir.
    """
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
    """Discover experimental folders that contain PTK or STK runs.
    
    Args:
        base_data_dir: Directory containing or receiving the base data.
    
    Returns:
        tuple: Discover valid data folders.
    """
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
    """Resolve a numeric folder selection into concrete PTK/STK inputs.
    
    Args:
        experiment_index (int): Index selecting the experiment.
        run_index (int): Index selecting the run.
        base_data_dir: Directory containing or receiving the base data.
        expected_peptide_type (Optional[str]): Expected peptide type processed by this function.
    
    Returns:
        dict: Resolved data selection.
    """
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
        selection (Tuple[int, int]): Selection processed by this function.
        timestamp (str): Timestamp string associated with the current analysis run.
        expected_peptide_type (str): Expected peptide type used by this function.
        base_data_dir: Directory containing or receiving the base data.
    
    Returns:
        object: Constructed loader from selection.
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
        experiment_name (str): Experiment name used by this function.
        uka_params: UKA params processed by this function.
        results_parent_relpath (Optional[Path]): Path-like value for results parent relpath.
    
    Returns:
        object: Resolved UKA params.
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

    resolved["path_output"] = (
        Path(resolved["path_output"])
        if resolved["path_output"] is not None
        else uka_results_dir / f"{analysis_timestamp}_UKA_output"
    )
    resolved["path_output_peptide_statistic"] = (
        Path(resolved["path_output_peptide_statistic"])
        if resolved["path_output_peptide_statistic"] is not None
        else uka_results_dir / analysis_timestamp
    )

    shared_volcano_dir = (
        Path(resolved["volcano_plot_output"])
        if resolved["volcano_plot_output"] is not None
        else uka_results_dir / f"{analysis_timestamp}_volcano_plots"
    )
    shared_heatmap_dir = (
        Path(resolved["heatmap_plot_output"])
        if resolved["heatmap_plot_output"] is not None
        else uka_results_dir / f"{analysis_timestamp}_heatmap_plots"
    )
    shared_venn_dir = (
        Path(resolved["venn_plot_output"])
        if resolved["venn_plot_output"] is not None
        else uka_results_dir / f"{analysis_timestamp}_venn_plots"
    )

    global_volcano = resolved["volcano_plot"]
    global_volcano_gui = resolved["volcano_plot_GUI"]
    global_heatmap = resolved["heatmap_plot"]
    global_heatmap_gui = resolved["heatmap_plot_GUI"]

    if "peptide_volcano_plot" not in custom_keys:
        resolved["peptide_volcano_plot"] = global_volcano
    if "kinase_volcano_plot" not in custom_keys:
        resolved["kinase_volcano_plot"] = global_volcano
    if "peptide_volcano_plot_GUI" not in custom_keys:
        resolved["peptide_volcano_plot_GUI"] = global_volcano_gui
    if "kinase_volcano_plot_GUI" not in custom_keys:
        resolved["kinase_volcano_plot_GUI"] = global_volcano_gui

    if "peptide_heatmap_plot" not in custom_keys:
        resolved["peptide_heatmap_plot"] = global_heatmap
    if "pathway_heatmap_plot" not in custom_keys:
        resolved["pathway_heatmap_plot"] = global_heatmap
    if "peptide_heatmap_plot_GUI" not in custom_keys:
        resolved["peptide_heatmap_plot_GUI"] = global_heatmap_gui
    if "pathway_heatmap_plot_GUI" not in custom_keys:
        resolved["pathway_heatmap_plot_GUI"] = global_heatmap_gui

    def _resolve_stage_output(key, default_dir):
        """Resolve an optional stage-figure output destination.
        
        Args:
            key: Key processed by this function.
            default_dir: Directory containing or receiving the default.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        value = resolved.get(key)
        resolved[key] = default_dir if value is None else Path(value)

    _resolve_stage_output("peptide_volcano_plot_output", shared_volcano_dir)
    _resolve_stage_output("kinase_volcano_plot_output", shared_volcano_dir)
    _resolve_stage_output("peptide_heatmap_plot_output", shared_heatmap_dir)
    _resolve_stage_output("pathway_heatmap_plot_output", shared_heatmap_dir)
    _resolve_stage_output("venn_plot_output", shared_venn_dir)

    resolved["input_stk_ptm_path"] = Path(resolved["input_stk_ptm_path"])
    resolved["input_ptk_ptm_path"] = Path(resolved["input_ptk_ptm_path"])
    return resolved


def build_timed_uka_params(uka_params=None):
    """Disable expensive plot/export options for the timed analysis phase.
    
    The repository reports the scientific analysis duration separately from the
    post-processing visualization work. This helper keeps the timed portion
    focused on statistics and enrichment while the heavier plotting steps can
    run afterwards without skewing the headline runtime summary.
    
    Args:
        uka_params: UKA params processed by this function.
    
    Returns:
        object: Constructed timed UKA params.
    """
    timed_params = dict(uka_params or {})
    timed_params.update(
        {
            "volcano_plot": False,
            "heatmap_plot": False,
            "peptide_volcano_plot": False,
            "peptide_heatmap_plot": False,
            "kinase_volcano_plot": False,
            "pathway_heatmap_plot": False,
            "venn_plot": False,
            "venn_plot_tables": False,
        }
    )
    return timed_params


def save_results_to_excel(condition_results, output_base, control, condition):
    """Write the main downstream analysis tables into an Excel workbook.
    
    Args:
        condition_results: Condition results processed by this function.
        output_base: Output location or value for base.
        control: Control processed by this function.
        condition: Condition processed by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    output_path = Path(f"{output_base}_{control}_{condition}.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        condition_results["peptide_statistics"].to_excel(
            writer,
            sheet_name="Peptide_Statistics",
            index=False,
        )
        condition_results["all_kinases"].to_excel(
            writer,
            sheet_name="Kinases_All",
            index=False,
        )
        condition_results["significant_kinases"].to_excel(
            writer,
            sheet_name="Kinases_Significant",
            index=False,
        )

        if not condition_results["UKA_raw"].empty:
            condition_results["UKA_raw"].to_excel(
                writer,
                sheet_name="Kinases_All_Raw",
                index=False,
            )
        if not condition_results["pathways_KEGG"].empty:
            condition_results["pathways_KEGG"].to_excel(
                writer,
                sheet_name="Pathways_KEGG",
                index=False,
            )
        if not condition_results["pathways_WP"].empty:
            condition_results["pathways_WP"].to_excel(
                writer,
                sheet_name="Pathways_WikiPathways",
                index=False,
            )
        if not condition_results["pathways_REAC"].empty:
            condition_results["pathways_REAC"].to_excel(
                writer,
                sheet_name="Pathways_Reactome",
                index=False,
            )


def plot_control_condition_venn_diagrams(
    all_results,
    kinase_analysis,
    pathway_analysis,
    output_path,
    save_tables=True,
):
    """Generate venn-diagram summaries for control-condition overlaps.
    
    Args:
        all_results: All results processed by this function.
        kinase_analysis: Kinase analysis processed by this function.
        pathway_analysis: Pathway analysis processed by this function.
        output_path: Path to the output.
        save_tables: Save tables processed by this function.
    
    Returns:
        object: Plot output for control condition venn diagrams.
    """
    if not all_results:
        return {}

    print("[4]  Plotting overlap diagrams for control-vs-condition comparisons...")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    venn_outputs = {}
    kinase_outputs = kinase_analysis.plot_kinase_overlap_venn(
        kinase_results_by_condition=all_results,
        output_path=output_path,
        save_tables=save_tables,
    )
    if kinase_outputs:
        venn_outputs["kinases"] = kinase_outputs

    pathway_outputs = pathway_analysis.plot_pathway_overlap_venn(
        pathway_results_by_condition=all_results,
        output_path=output_path,
        save_tables=save_tables,
    )
    if pathway_outputs:
        venn_outputs["pathways"] = pathway_outputs

    if venn_outputs:
        print(f"     Venn/overlap diagrams saved to: {output_path}")
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
    """Render optional downstream outputs excluded from the main timing block.
    
    Args:
        all_results: All results processed by this function.
        peptide_analysis: Peptide analysis processed by this function.
        kinase_analysis: Kinase analysis processed by this function.
        pathway_analysis: Pathway analysis processed by this function.
        resolved_params: Resolved params processed by this function.
    
    Returns:
        object: Render UKA outputs excluded from timing.
    """
    should_render_stage_outputs = any(
        (
            resolved_params["peptide_volcano_plot"],
            resolved_params["peptide_heatmap_plot"],
            resolved_params["kinase_volcano_plot"],
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

    for condition, condition_results in all_results.items():
        control_condition = condition_results["control_condition"]

        if (
            resolved_params["peptide_volcano_plot"]
            or resolved_params["peptide_heatmap_plot"]
        ):
            df_peptides_plot = (
                condition_results["peptide_statistics"]
                .sort_values("p_value", ascending=True)
                .drop_duplicates(subset="ID", keep="first")
                .reset_index(drop=True)
            )

            if resolved_params["peptide_volcano_plot"]:
                peptide_analysis._plot_volcano(
                    df_peptides=df_peptides_plot,
                    output_path=resolved_params["peptide_volcano_plot_output"],
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

        if resolved_params["kinase_volcano_plot"]:
            kinase_analysis._plot_volcano(
                df_uka=condition_results["all_kinases"],
                output_path=resolved_params["kinase_volcano_plot_output"],
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

    venn_diagrams = {}
    if should_render_venn:
        venn_diagrams = plot_control_condition_venn_diagrams(
            all_results=all_results,
            kinase_analysis=kinase_analysis,
            pathway_analysis=pathway_analysis,
            output_path=resolved_params["venn_plot_output"],
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
    
    Args:
        ptk_selection (Optional[Tuple[int, int]]): PTK selection processed by this function.
        stk_selection (Optional[Tuple[int, int]]): STK selection processed by this function.
        base_data_dir: Directory containing or receiving the base data.
    
    Returns:
        tuple: Run result for image analysis pipeline.
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
        enricher: Enricher processed by this function.
        processor_PTK: Processor PTK processed by this function.
        processor_STK: Processor STK processed by this function.
        analysis_timestamp: Timestamp string assigned to the current analysis run.
        experiment_name: Experiment name processed by this function.
        results_parent_relpath: Results parent relpath processed by this function.
        uka_params: UKA params processed by this function.
    
    Returns:
        tuple: Run result for peptide statistics analysis.
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
        significance_level_peptides=resolved_params["significance_level_peptides"],
        volcano_plot=resolved_params["peptide_volcano_plot"],
        volcano_plot_GUI=resolved_params["peptide_volcano_plot_GUI"],
        volcano_plot_output=resolved_params["peptide_volcano_plot_output"],
        heatmap_plot=resolved_params["peptide_heatmap_plot"],
        heatmap_plot_GUI=resolved_params["peptide_heatmap_plot_GUI"],
        heatmap_plot_output=resolved_params["peptide_heatmap_plot_output"],
        path_output_peptide_statistic=resolved_params["path_output_peptide_statistic"],
        use_limma=resolved_params["use_limma"],
        debugging_print=resolved_params["debugging_print"],
        log2_slope_mode=resolved_params["log2_slope_mode"],
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
        peptide_results: Peptide results processed by this function.
        analysis_timestamp: Timestamp string assigned to the current analysis run.
        experiment_name: Experiment name processed by this function.
        results_parent_relpath: Results parent relpath processed by this function.
        uka_params: UKA params processed by this function.
    
    Returns:
        object: Run result for kinase analysis.
    """
    resolved_params = resolve_uka_params(
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=uka_params,
    )

    def _build_kinase_analysis():
        """Build the kinase-analysis object for one condition comparison.
        
        Args:
            None.
        
        Returns:
            object: Constructed kinase analysis.
        """
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
            volcano_plot=resolved_params["kinase_volcano_plot"],
            volcano_plot_GUI=resolved_params["kinase_volcano_plot_GUI"],
            volcano_plot_output=resolved_params["kinase_volcano_plot_output"],
            uka_visualization_metric=resolved_params["uka_visualization_metric"],
            path_output_peptide_statistic=resolved_params["path_output_peptide_statistic"],
            debugging_print=resolved_params["debugging_print"],
            n_permutations=resolved_params["n_permutations"],
            kpea_lfc_cutoffs=resolved_params["kpea_lfc_cutoffs"],
            kpea_cutoff_mode=resolved_params["kpea_cutoff_mode"],
            kpea_primary_lfc_cutoff=resolved_params["kpea_primary_lfc_cutoff"],
            kpea_substrate_cutoff=resolved_params["kpea_substrate_cutoff"],
            kpea_zscore_threshold=resolved_params["kpea_zscore_threshold"],
            kpea_z_cap=resolved_params["kpea_z_cap"],
            kpea_significance_method=resolved_params["kpea_significance_method"],
            kpea_empirical_p_threshold=resolved_params["kpea_empirical_p_threshold"],
            kpea_fdr_threshold=resolved_params["kpea_fdr_threshold"],
        )

    def _prime_kinase_worker(base_worker, worker):
        """Warm up one kinase-analysis worker before parallel execution.
        
        Args:
            base_worker: Base worker processed by this function.
            worker: Worker processed by this function.
        
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
        worker._kinase_name_cache = dict(base_worker._kinase_name_cache)
        return worker

    def _run_condition_kinase_analysis(condition, payload, base_worker):
        """Run kinase analysis for one condition comparison.
        
        Args:
            condition: Condition processed by this function.
            payload: Payload processed by this function.
            base_worker: Base worker processed by this function.
        
        Returns:
            tuple: Run result for condition kinase analysis.
        """
        worker = _prime_kinase_worker(base_worker, _build_kinase_analysis())
        result = worker.run_kinase_analysis(
            peptide_statistics=payload,
            control=payload["control_condition"],
            condition=condition,
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


def run_pathway_enrichment(
    kinase_results,
    peptide_results,
    analysis_timestamp,
    experiment_name,
    results_parent_relpath=None,
    uka_params=None,
):
    """Run stage 3 of the downstream analysis: pathway enrichment.
    
    Args:
        kinase_results: Kinase results processed by this function.
        peptide_results: Peptide results processed by this function.
        analysis_timestamp: Timestamp string assigned to the current analysis run.
        experiment_name: Experiment name processed by this function.
        results_parent_relpath: Results parent relpath processed by this function.
        uka_params: UKA params processed by this function.
    
    Returns:
        object: Run result for pathway enrichment.
    """
    resolved_params = resolve_uka_params(
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=uka_params,
    )

    def _build_pathway_analysis():
        """Build the pathway-analysis object for one condition comparison.
        
        Args:
            None.
        
        Returns:
            object: Constructed pathway analysis.
        """
        return PathwayEnrichmentAnalysis(
            significance_level_pathways=resolved_params["significance_level_pathways"],
            heatmap_plot=resolved_params["pathway_heatmap_plot"],
            heatmap_plot_GUI=resolved_params["pathway_heatmap_plot_GUI"],
            heatmap_plot_output=resolved_params["pathway_heatmap_plot_output"],
            uka_visualization_metric=resolved_params["uka_visualization_metric"],
            debugging_print=resolved_params["debugging_print"],
        )

    pathway_analysis = _build_pathway_analysis()

    def _analyze_pathway_condition(condition, payload):
        """Run pathway enrichment for one condition comparison.
        
        Args:
            condition: Condition processed by this function.
            payload: Payload processed by this function.
        
        Returns:
            tuple: Analyze pathway condition.
        """
        worker = _build_pathway_analysis()
        control_condition = None
        if condition in peptide_results:
            control_condition = peptide_results[condition].get("control_condition")
        result = worker.run_pathway_enrichment(
            significant_kinases=payload["significant_kinases"],
            control=control_condition,
            condition=condition,
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
    """Run the full staged downstream analysis and save condition workbooks.
    
    The returned ``analysis_bundle`` keeps the live stage objects around for
    optional post-processing such as venn diagrams or custom visualizations.
    
    Args:
        enricher: Enricher processed by this function.
        processor_PTK: Processor PTK processed by this function.
        processor_STK: Processor STK processed by this function.
        analysis_timestamp: Timestamp string assigned to the current analysis run.
        experiment_name: Experiment name processed by this function.
        results_parent_relpath: Results parent relpath processed by this function.
        uka_params: UKA params processed by this function.
    
    Returns:
        tuple: Run result for UKA analysis.
    """
    resolved_params = resolve_uka_params(
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
        uka_params=uka_params,
    )
    timed_uka_params = build_timed_uka_params(uka_params=uka_params)

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
    )

    # Repackage per-condition stage outputs into one export-oriented structure
    # and persist the canonical Excel workbooks expected by users and tests.
    all_results = {}
    for condition, peptide_payload in peptide_results.items():
        control_condition = peptide_payload["control_condition"]
        combined_result = {
            "control_condition": control_condition,
            "condition": condition,
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
        save_results_to_excel(
            condition_results=combined_result,
            output_base=resolved_params["path_output"],
            control=control_condition,
            condition=condition,
        )
        all_results[condition] = combined_result

    uka_duration = time.time() - tic_uka
    print("\n" + "=" * 80)
    print(
        f"Staged UKA/KPEA analysis COMPLETED in {uka_duration:.2f} seconds "
        f"({uka_duration / 60:.2f} minutes)"
    )
    print("=" * 80 + "\n")

    # Plot-heavy overlap outputs are intentionally generated after the timed
    # core analysis so runtime summaries reflect the analysis itself.
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
    """Render representative publication-style QC figures for PTK and STK.
    
    Args:
        processor_PTK: Processor PTK processed by this function.
        processor_STK: Processor STK processed by this function.
        num_representative_images: Num representative images processed by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
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
    image_limit=None,
):
    """Render per-image intermediate processing figures for QC review.
    
    Args:
        processor_PTK: Processor PTK processed by this function.
        processor_STK: Processor STK processed by this function.
        all_images: All images processed by this function.
        image_limit: Image limit processed by this function.
    
    Returns:
        object: Created processing stage figures.
    """
    print("\n" + "=" * 80)
    print("Creating per-image processing-stage figures...")
    print("=" * 80)

    def _image_indices(processor):
        """Normalize the requested image-index selection into a list of indices.
        
        Args:
            processor: Processor processed by this function.
        
        Returns:
            object: Image indices.
        """
        n_images = int(processor.original_images.sizes["image_idx"])
        if all_images:
            limit = n_images if image_limit is None else min(n_images, int(image_limit))
        else:
            limit = 1 if image_limit is None else min(n_images, int(image_limit))
        return range(limit)

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
    processing_stage_figure_image_limit=PROCESSING_STAGE_FIGURE_IMAGE_LIMIT,
):
    """Convenience wrapper that executes the full terminal workflow.
    
    The script entry point now spells the top-level stages out directly in its
    own ``main()``, but this wrapper remains useful for tests, notebooks, or
    local helper code that wants one callable for the default terminal flow.
    
    Args:
        ptk_selection (Optional[Tuple[int, int]]): PTK selection processed by this function.
        stk_selection (Optional[Tuple[int, int]]): STK selection processed by this function.
        base_data_dir: Directory containing or receiving the base data.
        create_publication_figures_flag (bool): Boolean flag controlling whether to create publication figures flag.
        num_representative_images (int): Num representative images used by this function.
        create_processing_stage_figures_flag (bool): Boolean flag controlling whether to create processing stage figures flag.
        processing_stage_figures_all_images (bool): Processing stage figures all images used by this function.
        processing_stage_figure_image_limit: Processing stage figure image limit processed by this function.
    
    Returns:
        dict: Run result for terminal pipeline.
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

    if create_publication_figures_flag:
        create_publication_figures(
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            num_representative_images=num_representative_images,
        )

    if create_processing_stage_figures_flag:
        create_processing_stage_figures(
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            all_images=processing_stage_figures_all_images,
            image_limit=processing_stage_figure_image_limit,
        )

    return {
        "loaders": (loader_PTK, loader_STK),
        "enricher": enricher,
        "processors": (processor_PTK, processor_STK),
        "analysis_bundle": analysis_bundle,
        "uka_results": results,
    }
