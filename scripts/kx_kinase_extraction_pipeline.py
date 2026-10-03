"""Terminal entry point of the pyKinaXe pipeline.

The script adds the repository root and ``src/`` to the import path, selects
the headless matplotlib backend and runs the stages implemented in
``src/kx_pipeline_tools.py``: run selection, PTK/STK import and enrichment,
image processing, peptide statistics, kinase and pathway analysis and the
optional figures. Defaults come from ``config/pipeline_defaults.yaml``.

    python scripts/kx_kinase_extraction_pipeline.py

``run_image_analysis_pipeline`` and ``run_uka_analysis`` are re-exported here
because other scripts in the repository import them from this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


# The terminal pipeline is expected to run headlessly in many environments
# (remote sessions, batch jobs, CI, servers), so we force a non-GUI backend.
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Allow this file to be run directly from the repository root without
# requiring the project to be installed as a package first.
for import_dir in (REPO_ROOT, SRC_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))


from config.pipeline_defaults import (  # noqa: E402
    # Entry-point flags stay in config so this script remains a thin launcher.
    CREATE_PROCESSING_STAGE_FIGURES,
    CREATE_PUBLICATION_FIGURES,
    DEFAULT_KINASE_ANALYSIS_PARAMS,
    DEFAULT_PATHWAY_ENRICHMENT_PARAMS,
    DEFAULT_PEPTIDE_STATISTICS_PARAMS,
    DEFAULT_UKA_KPEA_PARAMS,
    DEFAULT_VENN_PLOT_PARAMS,
    NUM_REPRESENTATIVE_IMAGES,
    PROCESSING_STAGE_FIGURE_IMAGE_LIMIT,
    PROCESSING_STAGE_FIGURES_ALL_IMAGES,
    PROCESSING_STAGE_FIGURES_FRACTION,
)
from kx_pipeline_tools import (  # noqa: E402
    # Re-export commonly used helpers from the consolidated pipeline tools
    # module so existing imports from this script continue to work.
    create_processing_stage_figures,
    create_publication_figures,
    resolve_data_selection,
    run_image_analysis_pipeline,
    run_kinase_analysis,
    run_pathway_enrichment,
    run_peptide_statistics_analysis,
    run_uka_analysis,
)


__all__ = [
    # Public compatibility surface for local tests and helper scripts.
    "CREATE_PROCESSING_STAGE_FIGURES",
    "CREATE_PUBLICATION_FIGURES",
    "DEFAULT_KINASE_ANALYSIS_PARAMS",
    "DEFAULT_PATHWAY_ENRICHMENT_PARAMS",
    "DEFAULT_PEPTIDE_STATISTICS_PARAMS",
    "DEFAULT_UKA_KPEA_PARAMS",
    "DEFAULT_VENN_PLOT_PARAMS",
    "NUM_REPRESENTATIVE_IMAGES",
    "PROCESSING_STAGE_FIGURE_IMAGE_LIMIT",
    "PROCESSING_STAGE_FIGURES_ALL_IMAGES",
    "PROCESSING_STAGE_FIGURES_FRACTION",
    "create_processing_stage_figures",
    "create_publication_figures",
    "resolve_data_selection",
    "run_image_analysis_pipeline",
    "run_kinase_analysis",
    "run_pathway_enrichment",
    "run_peptide_statistics_analysis",
    "run_uka_analysis",
]


def main():
    """Run the default terminal workflow.
    
    All behavior toggles used here come from ``config/pipeline_defaults.py``
    and the underlying ``config/pipeline_defaults.yaml`` data file.
    The stages are spelled out here so that the script reads as an overview of
    the pipeline:
    
    - load and enrich PTK/STK data
    - process images
    - run downstream UKA/KPEA analysis
    - optionally generate figures
    """
    # Step 1: run the image-analysis portion of the PTK/STK pipeline.
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
    ) = run_image_analysis_pipeline()

    # Step 2: render intermediate processing-stage figures right after image
    # analysis and BEFORE the downstream analysis, so image-QC figures are still
    # produced even if UKA/KPEA later fails. The processors are already fully
    # processed by run_image_analysis_pipeline() above.
    if CREATE_PROCESSING_STAGE_FIGURES:
        create_processing_stage_figures(
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            all_images=PROCESSING_STAGE_FIGURES_ALL_IMAGES,
            fraction=PROCESSING_STAGE_FIGURES_FRACTION,
            image_limit=PROCESSING_STAGE_FIGURE_IMAGE_LIMIT,
        )

    # Step 3: run the downstream peptide/kinase/pathway analysis.
    results, analysis_bundle, uka_duration = run_uka_analysis(
        enricher=enricher,
        processor_PTK=processor_PTK,
        processor_STK=processor_STK,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
    )

    # Step 4: report the total runtime across the main analysis stages.
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

    # Step 5: optionally render publication figures controlled from config.
    # (Processing-stage figures are rendered earlier, before UKA/KPEA.)
    if CREATE_PUBLICATION_FIGURES:
        create_publication_figures(
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            num_representative_images=NUM_REPRESENTATIVE_IMAGES,
        )

    return {
        "loaders": (loader_PTK, loader_STK),
        "enricher": enricher,
        "processors": (processor_PTK, processor_STK),
        "analysis_bundle": analysis_bundle,
        "uka_results": results,
    }


if __name__ == "__main__":
    # Running this file directly should behave like the canonical terminal
    # pipeline command.
    main()
