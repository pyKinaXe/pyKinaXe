"""YAML-backed defaults for ``kx_data_enricher.py``."""

from __future__ import annotations

from config._loader import load_yaml_config, resolve_repo_path


_CONFIG = load_yaml_config(__file__)

DATA_ENRICHER_DEFAULTS = _CONFIG["data_enricher"]
UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS = _CONFIG[
    "uniprot_blast_api_data_collector"
]
UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["input_path"] = resolve_repo_path(
    UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["input_path"]
)
UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["output_path"] = resolve_repo_path(
    UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["output_path"]
)
OMNIPATH_PTM_EXTRACTOR_DEFAULTS = {
    **_CONFIG["omnipath_ptm_extractor"],
    "curated_source_labels": tuple(
        _CONFIG["omnipath_ptm_extractor"]["curated_source_labels"]
    ),
    "default_uniprot_stk_path": resolve_repo_path(
        _CONFIG["omnipath_ptm_extractor"]["default_uniprot_stk_path"]
    ),
    "default_uniprot_ptk_path": resolve_repo_path(
        _CONFIG["omnipath_ptm_extractor"]["default_uniprot_ptk_path"]
    ),
    "output_dir": resolve_repo_path(_CONFIG["omnipath_ptm_extractor"]["output_dir"]),
}
KINASE_LIVER_EXTRACTOR_DEFAULTS = {
    **_CONFIG["kinase_liver_extractor"],
    "output_path": resolve_repo_path(_CONFIG["kinase_liver_extractor"]["output_path"]),
}
