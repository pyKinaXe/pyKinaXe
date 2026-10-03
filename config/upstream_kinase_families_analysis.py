"""YAML-backed defaults for the kinase-FAMILY upstream analysis.

Mirrors :mod:`config.analysis_modules`' ``UPSTREAM_KINASE_ANALYSIS_DEFAULTS`` but
reads from ``upstream_kinase_families_analysis.yaml`` so the family-level analysis
can carry its own (higher) significance threshold independently of the
single-kinase UKA.
"""

from __future__ import annotations

from pathlib import Path

from config._loader import load_yaml_config, resolve_repo_path


_CONFIG = load_yaml_config(__file__)

UPSTREAM_KINASE_FAMILIES_ANALYSIS_DEFAULTS = {
    **_CONFIG["upstream_kinase_families_analysis"],
    "default_verified_evidence_levels": tuple(
        _CONFIG["upstream_kinase_families_analysis"]["default_verified_evidence_levels"]
    ),
    "default_blast_results_relative_path": Path(
        _CONFIG["upstream_kinase_families_analysis"]["default_blast_results_relative_path"]
    ),
    "path_file_enrichment_peptides": resolve_repo_path(
        _CONFIG["upstream_kinase_families_analysis"]["path_file_enrichment_peptides"]
    ),
    "input_stk_ptm_path": resolve_repo_path(
        _CONFIG["upstream_kinase_families_analysis"]["input_stk_ptm_path"]
    ),
    "input_ptk_ptm_path": resolve_repo_path(
        _CONFIG["upstream_kinase_families_analysis"]["input_ptk_ptm_path"]
    ),
    "candidate_data_paths": tuple(
        _CONFIG["upstream_kinase_families_analysis"]["candidate_data_paths"]
    ),
    "allowed_kpea_background_universes": tuple(
        _CONFIG["upstream_kinase_families_analysis"]["allowed_kpea_background_universes"]
    ),
    "allowed_kpea_cutoff_modes": tuple(
        _CONFIG["upstream_kinase_families_analysis"]["allowed_kpea_cutoff_modes"]
    ),
    "default_kpea_lfc_cutoffs": tuple(
        _CONFIG["upstream_kinase_families_analysis"]["default_kpea_lfc_cutoffs"]
    ),
    "allowed_family_mapping_sources": tuple(
        _CONFIG["upstream_kinase_families_analysis"]["allowed_family_mapping_sources"]
    ),
    "krsa_ptk_mapping_path": resolve_repo_path(
        _CONFIG["upstream_kinase_families_analysis"]["krsa_ptk_mapping_path"]
    ),
    "krsa_stk_mapping_path": resolve_repo_path(
        _CONFIG["upstream_kinase_families_analysis"]["krsa_stk_mapping_path"]
    ),
}
