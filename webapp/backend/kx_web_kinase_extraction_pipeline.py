"""Backend-side web pipeline for non-interactive pyKinaXe runs.

This module is the bridge between the Flask runtime in ``webapp/pykinaxe_webapp.py``
and the scientific pipeline code in ``src/``. It adapts uploaded PTK/STK job
folders into the same objects the terminal workflow uses, while forcing all
outputs into job-scoped directories suitable for the web app.

It contains both:

- reusable backend helpers used by the Flask API
- an optional CLI entry point for running the same web-oriented workflow from a shell

Keeping the implementation in one backend file makes the web stack easier to
follow without changing the behavior exposed to the frontend.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Callable
import zipfile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib-pykinaxe-web"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "xdg-cache-pykinaxe-web"),
)

for import_dir in (REPO_ROOT, SRC_DIR, SCRIPTS_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))


from kx_data_importer import DataLoader  # noqa: E402
from kx_data_enricher import DataEnricher  # noqa: E402
from kx_image_processor import ImageProcessor  # noqa: E402
from kx_pipeline_tools import (  # noqa: E402
    create_processing_stage_figures as render_processing_stage_figures,
    create_publication_figures as render_publication_figures,
    run_uka_analysis,
)


VALID_PEPTIDE_TYPES = ("PTK", "STK")


@dataclass
class WebPipelineRun:
    """Container for the key directories and outputs produced by one web pipeline run."""
    job_id: str
    analysis_timestamp: str
    ptk_data_dir: Path
    stk_data_dir: Path
    output_root: Path
    image_processing_dir: Path
    downstream_output_dir: Path


def unzip_archive(archive_path: Path, destination_dir: Path) -> None:
    """Unpack a ZIP archive into the requested destination directory.
    
    Args:
        archive_path (Path): Path to the archive.
        destination_dir (Path): Directory containing or receiving the destination.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(destination_dir)


def _is_valid_uploaded_run_dir(path: Path, peptide_type: str) -> bool:
    """Return whether valid uploaded run dir.
    
    Args:
        path (Path): Path value processed by this helper.
        peptide_type (str): Peptide type used by this function.
    
    Returns:
        bool: Is valid uploaded run dir.
    """
    path_name = path.name.upper()
    if peptide_type not in path_name:
        return False
    if "RUN" not in path_name:
        return False
    if not path.is_dir():
        return False
    if not (path / "ImageResults").exists():
        return False
    return True


def resolve_uploaded_data_dir(extracted_root: Path, peptide_type: str) -> Path:
    """Resolve the single valid uploaded PTK or STK run directory.
    
    Args:
        extracted_root (Path): Path-like value for extracted root.
        peptide_type (str): Peptide type used by this function.
    
    Returns:
        Path: Resolved uploaded data dir.
    """
    candidates = sorted(
        path
        for path in extracted_root.rglob("*")
        if _is_valid_uploaded_run_dir(path, peptide_type=peptide_type)
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a valid {peptide_type} uploaded run directory under {extracted_root}."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Found multiple candidate {peptide_type} run directories under {extracted_root}: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def _ensure_dir(path: Path) -> Path:
    """Create the directory if needed and return its path.
    
    Args:
        path (Path): Path value processed by this helper.
    
    Returns:
        Path: Ensured dir.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _temporary_results_root(results_root: Path):
    """Temporarily override the global results root for one web run.
    
    Args:
        results_root (Path): Path-like value for results root.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    env_key = "PYKINAXE_RESULTS_ROOT"
    previous = os.environ.get(env_key)
    os.environ[env_key] = str(Path(results_root).resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


def _write_source_info(
    results_dir: Path,
    timestamp: str,
    source_data_path: Path,
    label: str,
) -> None:
    # Mirror the provenance notes written by the terminal pipeline so every web
    # job still records exactly which source data folder produced the outputs.
    """Write a provenance note describing the source data directory.
    
    Args:
        results_dir (Path): Directory containing or receiving the results.
        timestamp (str): Timestamp string associated with the current analysis run.
        source_data_path (Path): Path to the source data.
        label (str): Label used by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    info_path = results_dir / f"{timestamp}_source_data_path.txt"
    info_path.write_text(
        "\n".join(
            [
                f"Source Data Path ({label}):",
                str(source_data_path),
                "",
                f"Analysis Timestamp: {timestamp}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_loader_for_uploaded_dir(data_dir: Path, timestamp: str) -> DataLoader:
    """Create a ``DataLoader`` from a concrete uploaded run directory.
    
    Args:
        data_dir (Path): Directory containing or receiving the data.
        timestamp (str): Timestamp string associated with the current analysis run.
    
    Returns:
        DataLoader: Constructed loader for uploaded dir.
    """
    data_dir = Path(data_dir).resolve()
    experiment_dir = data_dir.parent
    return DataLoader(
        data_dir=experiment_dir,
        experiment_name=experiment_dir.name,
        subfolder=data_dir.name,
        timestamp=timestamp,
    )


def _override_loader_outputs(
    loader: DataLoader,
    processor: ImageProcessor,
    image_processing_dir: Path,
    label: str,
) -> None:
    # The terminal pipeline writes into the shared ``results/`` tree. The web
    # pipeline rewires those locations so each job stays fully self-contained.
    """Redirect loader and processor outputs into the web-job result tree.
    
    Args:
        loader (DataLoader): Loader processed by this function.
        processor (ImageProcessor): Processor processed by this function.
        image_processing_dir (Path): Directory containing or receiving the image-processing outputs.
        label (str): Label used by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    results_dir = _ensure_dir(image_processing_dir / label / loader.subfolder_name)
    processor.results_dir = results_dir
    processor._enrichment_file = (
        REPO_ROOT / "data" / "external" / "PamGene" / "enrichment_peptides.csv"
    )
    _write_source_info(results_dir, loader.timestamp, loader.data_dir, label=label)


def _override_enricher_outputs(
    enricher: DataEnricher,
    image_processing_dir: Path,
) -> None:
    # Keep the enriched PTK/STK metadata next to each job's image-analysis
    # outputs instead of the repository-level default results location.
    """Redirect enricher outputs into the web-job result tree.
    
    Args:
        enricher (DataEnricher): Enricher processed by this function.
        image_processing_dir (Path): Directory containing or receiving the image-processing outputs.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    enricher.results_dir_1 = _ensure_dir(
        image_processing_dir / "PTK" / enricher.subfolder_name_1
    )
    enricher.results_dir_2 = _ensure_dir(
        image_processing_dir / "STK" / enricher.subfolder_name_2
    )
    _write_source_info(
        enricher.results_dir_1,
        enricher.timestamp_1,
        enricher.source_data_path_1,
        label="PTK",
    )
    _write_source_info(
        enricher.results_dir_2,
        enricher.timestamp_2,
        enricher.source_data_path_2,
        label="STK",
    )


def build_web_uka_params(
    analysis_timestamp: str,
    output_root: Path,
) -> dict[str, Path | bool | int | float | str]:
    """Build web-job-scoped downstream output parameters.
    
    The terminal workflow stores outputs under the shared project results tree.
    The web backend instead redirects every downstream directory under one
    job-local ``output_root``.
    
    Args:
        analysis_timestamp (str): Timestamp string assigned to the current analysis run.
        output_root (Path): Output location or value for root.
    
    Returns:
        dict[str, Path | bool | int | float | str]: Constructed web UKA params.
    """
    downstream_dir = _ensure_dir(output_root / f"{analysis_timestamp}_downstream_analysis")
    return {
        "path_output": downstream_dir / f"{analysis_timestamp}_UKA_output",
        "path_output_peptide_statistic": downstream_dir / analysis_timestamp,
        "volcano_plot_output": downstream_dir / f"{analysis_timestamp}_volcano_plots",
        "heatmap_plot_output": downstream_dir / f"{analysis_timestamp}_heatmap_plots",
        "venn_plot_output": downstream_dir / f"{analysis_timestamp}_venn_plots",
        "peptide_heatmap_plot": True,
        "peptide_volcano_plot": True,
        "kinase_volcano_plot": True,
        "pathway_heatmap_plot": True,
        "venn_plot": True,
        "venn_plot_tables": True,
    }


def run_web_pipeline(
    *,
    job_id: str,
    ptk_data_dir: Path,
    stk_data_dir: Path,
    output_root: Path,
    create_processing_stage_figures: bool = False,
    processing_stage_figure_image_limit: int | None = None,
    create_publication_figures: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> WebPipelineRun:
    """Run the full non-interactive PTK/STK workflow for one web job.
    
    Args:
        job_id (str): Unique identifier of the web-analysis job.
        ptk_data_dir (Path): Directory containing or receiving the PTK data.
        stk_data_dir (Path): Directory containing or receiving the STK data.
        output_root (Path): Output location or value for root.
        create_processing_stage_figures (bool): Boolean flag controlling whether to create processing stage figures.
        processing_stage_figure_image_limit (int | None): Processing stage figure image limit processed by this function.
        create_publication_figures (bool): Boolean flag controlling whether to create publication figures.
        progress_callback (Callable[[str], None] | None): Optional callback used to receive progress messages.
    
    Returns:
        WebPipelineRun: Run result for web pipeline.
    """
    def _progress(message: str) -> None:
        """Forward one progress message to the optional callback.
        
        Args:
            message (str): Status or log message to record.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if progress_callback is not None:
            progress_callback(message)

    analysis_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_root = _ensure_dir(Path(output_root).resolve())
    image_processing_dir = _ensure_dir(output_root / f"{analysis_timestamp}_image_processing")
    downstream_dir = _ensure_dir(output_root / f"{analysis_timestamp}_downstream_analysis")

    _progress("Initializing PTK/STK loaders.")
    with _temporary_results_root(output_root):
        # From here on, the web workflow intentionally mirrors the terminal
        # scientific pipeline. The main differences are non-interactive input
        # handling and job-local output directories.
        loader_PTK = _build_loader_for_uploaded_dir(ptk_data_dir, timestamp=analysis_timestamp)
        loader_STK = _build_loader_for_uploaded_dir(stk_data_dir, timestamp=analysis_timestamp)
        _progress("Loading PTK data.")
        loader_PTK.load_data()
        _progress("Loading STK data.")
        loader_STK.load_data()

        _progress("Preparing image processing.")
        enricher = DataEnricher(loader_PTK, loader_STK)
        _override_enricher_outputs(enricher, image_processing_dir=image_processing_dir)
        enricher.enrich_data()

        _progress("Starting PTK image processing.")
        processor_PTK = ImageProcessor(loader_PTK)
        processor_STK = ImageProcessor(loader_STK)
        _override_loader_outputs(
            loader_PTK,
            processor_PTK,
            image_processing_dir=image_processing_dir,
            label="PTK",
        )
        _override_loader_outputs(
            loader_STK,
            processor_STK,
            image_processing_dir=image_processing_dir,
            label="STK",
        )
        processor_PTK.process()
        _progress("Starting STK image processing.")
        processor_STK.process()

        _progress("Running downstream analysis.")
        web_uka_params = build_web_uka_params(
            analysis_timestamp=analysis_timestamp,
            output_root=output_root,
        )
        run_uka_analysis(
            enricher=enricher,
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            analysis_timestamp=analysis_timestamp,
            experiment_name=f"web_job_{job_id}",
            uka_params=web_uka_params,
        )
        _progress("Downstream analysis completed.")

        if create_processing_stage_figures:
            _progress("Generating intermediate image-processing figures.")
            render_processing_stage_figures(
                processor_PTK=processor_PTK,
                processor_STK=processor_STK,
                all_images=processing_stage_figure_image_limit is None,
                image_limit=processing_stage_figure_image_limit,
            )

        if create_publication_figures:
            _progress("Generating publication figures.")
            render_publication_figures(
                processor_PTK=processor_PTK,
                processor_STK=processor_STK,
                num_representative_images=1,
            )

    return WebPipelineRun(
        job_id=job_id,
        analysis_timestamp=analysis_timestamp,
        ptk_data_dir=Path(ptk_data_dir).resolve(),
        stk_data_dir=Path(stk_data_dir).resolve(),
        output_root=output_root,
        image_processing_dir=image_processing_dir,
        downstream_output_dir=downstream_dir,
    )


def collect_frontend_results(run: WebPipelineRun) -> dict[str, object]:
    """Summarize completed outputs into the compact payload the frontend needs.
    
    Args:
        run (WebPipelineRun): Run processed by this function.
    
    Returns:
        dict[str, object]: Collect frontend results.
    """
    workbooks = sorted(run.downstream_output_dir.glob("*_UKA_output_*.xlsx"))
    kinase_outputs: list[dict[str, object]] = []

    for workbook in workbooks:
        try:
            df_kinases = pd.read_excel(workbook, sheet_name="Kinases_Significant")
        except ValueError:
            continue

        kinase_col = "Kinase_Name" if "Kinase_Name" in df_kinases.columns else "Kinase"
        if kinase_col not in df_kinases.columns:
            continue

        kinase_names = (
            df_kinases[kinase_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        comparison = workbook.stem.split("_UKA_output_", 1)[-1]
        if "_" in comparison:
            comparison = comparison.split("_", 1)[-1]
        kinase_outputs.append(
            {
                "workbook_name": workbook.name,
                "comparison": comparison,
                "kinase_names": kinase_names,
                "kinase_csv": ", ".join(kinase_names),
                "relative_path": str(workbook.relative_to(run.output_root)),
            }
        )

    heatmap_paths = sorted(
        run.downstream_output_dir.glob("*_heatmap_plots/peptides_heatmap_*.png")
    )
    heatmaps = [
        {
            "filename": path.name,
            "relative_path": str(path.relative_to(run.output_root)),
        }
        for path in heatmap_paths
    ]

    return {
        "job_id": run.job_id,
        "analysis_timestamp": run.analysis_timestamp,
        "output_root": str(run.output_root),
        "ptk_data_dir": str(run.ptk_data_dir),
        "stk_data_dir": str(run.stk_data_dir),
        "kinase_outputs": kinase_outputs,
        "heatmaps": heatmaps,
    }


def _copy_export_tree(source_dir: Path, export_root: Path) -> None:
    """Copy one top-level result tree into the export bundle root.

    Args:
        source_dir (Path): Existing directory to copy into the archive bundle.
        export_root (Path): Root directory of the curated archive bundle.

    Returns:
        None: This function is used for side effects and does not return a value.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Expected export directory is missing: {source_dir}")
    shutil.copytree(source_dir, export_root / source_dir.name)


def _build_job_info_text(
    run: WebPipelineRun,
    job_metadata: dict[str, object] | None = None,
) -> str:
    """Build the user-facing metadata file included in the download archive.

    Args:
        run (WebPipelineRun): Completed run being exported.
        job_metadata (dict[str, object] | None): Optional job metadata from the Flask runtime.

    Returns:
        str: Formatted archive metadata text.
    """
    metadata = dict(job_metadata or {})
    lines = [
        "pyKinaXe Web Job Info",
        "=====================",
        "",
        f"Job ID: {run.job_id}",
        f"Analysis Timestamp: {run.analysis_timestamp}",
    ]

    created_at = str(metadata.get("created_at") or "").strip()
    finished_at = str(metadata.get("finished_at") or "").strip()
    if created_at:
        lines.append(f"Job Created At: {created_at}")
    if finished_at:
        lines.append(f"Analysis Finished At: {finished_at}")

    return "\n".join(lines) + "\n"


def create_results_archive(
    run: WebPipelineRun,
    job_metadata: dict[str, object] | None = None,
) -> Path:
    """Zip one job's output tree into a single browser-downloadable archive.
    
    Args:
        run (WebPipelineRun): Run processed by this function.
        job_metadata (dict[str, object] | None): Optional job metadata from the Flask runtime.
    
    Returns:
        Path: Created results archive.
    """
    archive_dir = _ensure_dir(run.output_root.parent / "downloads")
    archive_base = archive_dir / f"{run.analysis_timestamp}_pyKinaXe_results"
    with tempfile.TemporaryDirectory(
        dir=archive_dir,
        prefix=f"{run.analysis_timestamp}_export_",
    ) as temp_dir:
        export_root = Path(temp_dir) / "bundle"
        export_root.mkdir(parents=True, exist_ok=True)
        _copy_export_tree(run.image_processing_dir, export_root)
        _copy_export_tree(run.downstream_output_dir, export_root)
        (export_root / "job_info.txt").write_text(
            _build_job_info_text(run, job_metadata=job_metadata),
            encoding="utf-8",
        )
        archive_path = shutil.make_archive(
            base_name=str(archive_base),
            format="zip",
            root_dir=export_root,
        )
    return Path(archive_path)


def build_parser() -> argparse.ArgumentParser:
    """Build parser.
    
    Args:
        None.
    
    Returns:
        argparse.ArgumentParser: Constructed parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the non-interactive pyKinaXe kinase extraction pipeline "
            "for explicit PTK/STK input directories. "
            "This is the frontend-oriented pipeline entry point."
        )
    )
    parser.add_argument(
        "--job-id",
        default="manual",
        help="Job identifier used in output metadata.",
    )
    parser.add_argument(
        "--ptk-dir",
        required=True,
        type=Path,
        help="Path to the PTK run directory.",
    )
    parser.add_argument(
        "--stk-dir",
        required=True,
        type=Path,
        help="Path to the STK run directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where all outputs will be written.",
    )
    parser.add_argument(
        "--processing-stage-figures",
        action="store_true",
        help="Generate intermediate image-processing figures for uploaded images.",
    )
    parser.add_argument(
        "--publication-figures",
        action="store_true",
        help="Generate publication-style figures in addition to the standard outputs.",
    )
    return parser


def main() -> None:
    """Run the module as a command-line entry point.
    
    Args:
        None.
    
    Returns:
        None: The command-line entry point runs for its side effects.
    """
    args = build_parser().parse_args()
    run = run_web_pipeline(
        job_id=args.job_id,
        ptk_data_dir=args.ptk_dir,
        stk_data_dir=args.stk_dir,
        output_root=args.output_dir,
        create_processing_stage_figures=args.processing_stage_figures,
        create_publication_figures=args.publication_figures,
    )
    results = collect_frontend_results(run)
    print(f"Analysis timestamp: {run.analysis_timestamp}")
    print(f"Output directory: {run.output_root}")
    print(f"Found {len(results['kinase_outputs'])} kinase workbook summaries.")
    print(f"Found {len(results['heatmaps'])} peptide heatmap images.")


if __name__ == "__main__":
    main()
