"""Load default pipeline parameters and flags from YAML configuration."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_suffix(".yaml")


def _load_config() -> dict:
    """Load the module configuration from its YAML companion file."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _optional_repo_path(value):
    """Resolve an optional repository-relative path value."""
    if value is None:
        return None
    return REPO_ROOT / str(value)


def _resolve_path_fields(mapping: dict, keys: tuple[str, ...]) -> dict:
    """Resolve configured path-like fields against the repository root."""
    resolved = dict(mapping)
    for key in keys:
        if key in resolved:
            resolved[key] = _optional_repo_path(resolved[key])
    return resolved


_CONFIG = _load_config()


DEFAULT_PEPTIDE_STATISTICS_PARAMS = _resolve_path_fields(
    _CONFIG["default_peptide_statistics_params"],
    (
        "path_file_enrichment_peptides",
        "path_output_peptide_statistic",
        "peptide_waterfall_plot_output",
        "peptide_heatmap_plot_output",
    ),
)

DEFAULT_KINASE_ANALYSIS_PARAMS = _resolve_path_fields(
    _CONFIG["default_kinase_analysis_params"],
    (
        "input_stk_ptm_path",
        "input_ptk_ptm_path",
    ),
)
DEFAULT_KINASE_ANALYSIS_PARAMS["verified_evidence_levels"] = tuple(
    DEFAULT_KINASE_ANALYSIS_PARAMS["verified_evidence_levels"]
)

DEFAULT_PATHWAY_ENRICHMENT_PARAMS = _resolve_path_fields(
    _CONFIG["default_pathway_enrichment_params"],
    ("pathway_heatmap_plot_output",),
)

DEFAULT_VENN_PLOT_PARAMS = _resolve_path_fields(
    _CONFIG["default_venn_plot_params"],
    ("kinase_venn_plot_output", "pathway_venn_plot_output"),
)

DEFAULT_UKA_KPEA_PARAMS = {
    **DEFAULT_PEPTIDE_STATISTICS_PARAMS,
    **DEFAULT_KINASE_ANALYSIS_PARAMS,
    **DEFAULT_PATHWAY_ENRICHMENT_PARAMS,
    **DEFAULT_VENN_PLOT_PARAMS,
    **{
        key: _optional_repo_path(_CONFIG["default_uka_kpea_params"][key])
        for key in (
            "peptide_results_output",
            "kinase_results_output",
            "family_results_output",
            "pathway_results_output",
            "log_output",
        )
    },
    "output_mode": _CONFIG["default_uka_kpea_params"]["output_mode"],
    "waterfall_plot": _CONFIG["default_uka_kpea_params"]["waterfall_plot"],
    "heatmap_plot": _CONFIG["default_uka_kpea_params"]["heatmap_plot"],
}

CREATE_PUBLICATION_FIGURES = bool(_CONFIG["runtime_flags"]["create_publication_figures"])
NUM_REPRESENTATIVE_IMAGES = int(_CONFIG["runtime_flags"]["num_representative_images"])
CREATE_PROCESSING_STAGE_FIGURES = bool(
    _CONFIG["runtime_flags"]["create_processing_stage_figures"]
)
PROCESSING_STAGE_FIGURES_ALL_IMAGES = bool(
    _CONFIG["runtime_flags"]["processing_stage_figures_all_images"]
)
_PROCESSING_STAGE_FIGURES_FRACTION = _CONFIG["runtime_flags"].get(
    "processing_stage_figures_fraction"
)
PROCESSING_STAGE_FIGURES_FRACTION = (
    None
    if _PROCESSING_STAGE_FIGURES_FRACTION is None
    else float(_PROCESSING_STAGE_FIGURES_FRACTION)
)
PROCESSING_STAGE_FIGURE_IMAGE_LIMIT = _CONFIG["runtime_flags"][
    "processing_stage_figure_image_limit"
]
