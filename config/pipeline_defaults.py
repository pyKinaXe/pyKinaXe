"""Load default pipeline parameters and flags from YAML configuration."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_suffix(".yaml")


def _load_config() -> dict:
    """Load the module configuration from its YAML companion file.
    
    Args:
        None.
    
    Returns:
        dict: Loaded config.
    """
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _optional_repo_path(value):
    """Resolve an optional repository-relative path value.
    
    Args:
        value: Input value processed by this helper.
    
    Returns:
        object: Optional repo path.
    """
    if value is None:
        return None
    return REPO_ROOT / str(value)


def _resolve_path_fields(mapping: dict, keys: tuple[str, ...]) -> dict:
    """Resolve configured path-like fields against the repository root.
    
    Args:
        mapping (dict): Mapping processed by this function.
        keys (tuple[str, ...]): Keys processed by this function.
    
    Returns:
        dict: Resolved path fields.
    """
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
        "peptide_volcano_plot_output",
        "peptide_heatmap_plot_output",
    ),
)

DEFAULT_KINASE_ANALYSIS_PARAMS = _resolve_path_fields(
    _CONFIG["default_kinase_analysis_params"],
    (
        "input_stk_ptm_path",
        "input_ptk_ptm_path",
        "kinase_volcano_plot_output",
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
    ("venn_plot_output",),
)

DEFAULT_UKA_KPEA_PARAMS = {
    **DEFAULT_PEPTIDE_STATISTICS_PARAMS,
    **DEFAULT_KINASE_ANALYSIS_PARAMS,
    **DEFAULT_PATHWAY_ENRICHMENT_PARAMS,
    **DEFAULT_VENN_PLOT_PARAMS,
    "path_output": _optional_repo_path(_CONFIG["default_uka_kpea_params"]["path_output"]),
    "volcano_plot": _CONFIG["default_uka_kpea_params"]["volcano_plot"],
    "volcano_plot_GUI": _CONFIG["default_uka_kpea_params"]["volcano_plot_GUI"],
    "volcano_plot_output": _optional_repo_path(
        _CONFIG["default_uka_kpea_params"]["volcano_plot_output"]
    ),
    "heatmap_plot": _CONFIG["default_uka_kpea_params"]["heatmap_plot"],
    "heatmap_plot_GUI": _CONFIG["default_uka_kpea_params"]["heatmap_plot_GUI"],
    "heatmap_plot_output": _optional_repo_path(
        _CONFIG["default_uka_kpea_params"]["heatmap_plot_output"]
    ),
}

CREATE_PUBLICATION_FIGURES = bool(_CONFIG["runtime_flags"]["create_publication_figures"])
NUM_REPRESENTATIVE_IMAGES = int(_CONFIG["runtime_flags"]["num_representative_images"])
CREATE_PROCESSING_STAGE_FIGURES = bool(
    _CONFIG["runtime_flags"]["create_processing_stage_figures"]
)
PROCESSING_STAGE_FIGURES_ALL_IMAGES = bool(
    _CONFIG["runtime_flags"]["processing_stage_figures_all_images"]
)
PROCESSING_STAGE_FIGURE_IMAGE_LIMIT = _CONFIG["runtime_flags"][
    "processing_stage_figure_image_limit"
]
