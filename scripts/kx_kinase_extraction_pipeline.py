"""Run the staged pyKinaXe kinase extraction pipeline from the terminal.

This script is intentionally lightweight, but it still shows the main
terminal workflow explicitly.

The lower-level implementation lives outside ``scripts/``:

- configuration and user-adjustable defaults live in ``config/``
- reusable pipeline functions live in ``src/kx_pipeline_tools.py``

This file exists as the human-facing command-line entry point for the
standard PTK/STK workflow. Its responsibilities are deliberately limited to:

1. making sure the repository's import paths are available
2. forcing a non-interactive matplotlib backend suitable for batch runs
3. re-exporting a few commonly used pipeline symbols for older imports/tests
4. showing the high-level pipeline steps in ``main()``

This keeps the script readable as a high-level
"what happens when I run the pipeline?" overview, while the detailed helper
logic remains in reusable functions shared by the web and batch entry points.

Typical usage:

    python scripts/kx_kinase_extraction_pipeline.py

The script still exposes several imported symbols such as
``run_image_analysis_pipeline`` and ``run_uka_analysis`` because tests and
other local tools in this repository import them directly from here.
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
    This function intentionally spells out the top-level pipeline stages so the
    script itself remains a readable orchestration layer:
    
    - load and enrich PTK/STK data
    - process images
    - run downstream UKA/KPEA analysis
    - optionally generate figures
    
    Args:
        None.
    
    Returns:
        None: The command-line entry point runs for its side effects.
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

    # Step 2: run the downstream peptide/kinase/pathway analysis.
    results, analysis_bundle, uka_duration = run_uka_analysis(
        enricher=enricher,
        processor_PTK=processor_PTK,
        processor_STK=processor_STK,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
    )

    # Step 3: report the total runtime across the main analysis stages.
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

    # Step 4: optionally render post-analysis figures controlled from config.
    if CREATE_PUBLICATION_FIGURES:
        create_publication_figures(
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            num_representative_images=NUM_REPRESENTATIVE_IMAGES,
        )

    if CREATE_PROCESSING_STAGE_FIGURES:
        create_processing_stage_figures(
            processor_PTK=processor_PTK,
            processor_STK=processor_STK,
            all_images=PROCESSING_STAGE_FIGURES_ALL_IMAGES,
            image_limit=PROCESSING_STAGE_FIGURE_IMAGE_LIMIT,
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
