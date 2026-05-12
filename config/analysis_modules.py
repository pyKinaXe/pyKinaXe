"""YAML-backed defaults for analysis modules in ``src/``."""

from __future__ import annotations

from pathlib import Path

from config._loader import load_yaml_config, resolve_repo_path


_CONFIG = load_yaml_config(__file__)

PEPTIDE_ANALYSIS_DEFAULTS = {
    **_CONFIG["peptide_analysis"],
    "path_file_enrichment_peptides": resolve_repo_path(
        _CONFIG["peptide_analysis"]["path_file_enrichment_peptides"]
    ),
    "allowed_log2_slope_modes": tuple(
        _CONFIG["peptide_analysis"]["allowed_log2_slope_modes"]
    ),
}

PATHWAY_ENRICHMENT_DEFAULTS = _CONFIG["pathway_enrichment_analysis"]

UPSTREAM_KINASE_ANALYSIS_DEFAULTS = {
    **_CONFIG["upstream_kinase_analysis"],
    "default_verified_evidence_levels": tuple(
        _CONFIG["upstream_kinase_analysis"]["default_verified_evidence_levels"]
    ),
    "default_blast_results_relative_path": Path(
        _CONFIG["upstream_kinase_analysis"]["default_blast_results_relative_path"]
    ),
    "path_file_enrichment_peptides": resolve_repo_path(
        _CONFIG["upstream_kinase_analysis"]["path_file_enrichment_peptides"]
    ),
    "input_stk_ptm_path": resolve_repo_path(
        _CONFIG["upstream_kinase_analysis"]["input_stk_ptm_path"]
    ),
    "input_ptk_ptm_path": resolve_repo_path(
        _CONFIG["upstream_kinase_analysis"]["input_ptk_ptm_path"]
    ),
    "candidate_data_paths": tuple(
        _CONFIG["upstream_kinase_analysis"]["candidate_data_paths"]
    ),
    "allowed_kpea_significance_methods": tuple(
        _CONFIG["upstream_kinase_analysis"]["allowed_kpea_significance_methods"]
    ),
    "allowed_kpea_cutoff_modes": tuple(
        _CONFIG["upstream_kinase_analysis"]["allowed_kpea_cutoff_modes"]
    ),
    "default_kpea_lfc_cutoffs": tuple(
        _CONFIG["upstream_kinase_analysis"]["default_kpea_lfc_cutoffs"]
    ),
}
