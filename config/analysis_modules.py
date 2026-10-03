"""YAML-backed defaults for the analysis modules in ``src/``.

Two config files feed the analysis classes, at different levels:

* ``analysis_modules.yaml`` (this module) holds what only the class needs:
  validation whitelists (``allowed_*``), resource locations, column names and
  hard floors.
* ``pipeline_defaults.yaml`` holds what the pipeline passes when it starts a
  run.

Parameters that both levels need used to be written out in both files, with
nothing keeping the copies in sync, so a value edited in one place could have
no effect on real runs (the pipeline passes every parameter explicitly, so its
value wins over the class default). Those parameters now live only in
``pipeline_defaults.yaml`` and are re-exported here under the ``default_*``
names the classes expect.
"""

from __future__ import annotations

from pathlib import Path

from config._loader import load_yaml_config, resolve_repo_path
from config.data_enricher import OMNIPATH_PTM_EXTRACTOR_DEFAULTS as _ENRICHER_PTM
from config.pipeline_defaults import (
    DEFAULT_KINASE_ANALYSIS_PARAMS as _PIPELINE_KINASE,
    DEFAULT_PATHWAY_ENRICHMENT_PARAMS as _PIPELINE_PATHWAY,
    DEFAULT_PEPTIDE_STATISTICS_PARAMS as _PIPELINE_PEPTIDE,
)


_CONFIG = load_yaml_config(__file__)

PEPTIDE_ANALYSIS_DEFAULTS = {
    **_CONFIG["peptide_analysis"],
    "path_file_enrichment_peptides": resolve_repo_path(
        _CONFIG["peptide_analysis"]["path_file_enrichment_peptides"]
    ),
    "allowed_log2_slope_modes": tuple(
        _CONFIG["peptide_analysis"]["allowed_log2_slope_modes"]
    ),
    "allowed_batch_correction_methods": tuple(
        _CONFIG["peptide_analysis"]["allowed_batch_correction_methods"]
    ),
    "allowed_array_normalization_methods": tuple(
        _CONFIG["peptide_analysis"]["allowed_array_normalization_methods"]
    ),
    "allowed_qc_modes": tuple(_CONFIG["peptide_analysis"]["allowed_qc_modes"]),
    "allowed_contrast_modes": tuple(
        _CONFIG["peptide_analysis"]["allowed_contrast_modes"]
    ),
    # Single source of truth: pipeline_defaults.yaml.
    "default_log2_slope_mode": _PIPELINE_PEPTIDE["log2_slope_mode"],
    "default_batch_correction": _PIPELINE_PEPTIDE["batch_correction"],
    "default_batch_correction_method": _PIPELINE_PEPTIDE["batch_correction_method"],
    "default_array_normalization": _PIPELINE_PEPTIDE["array_normalization"],
    "default_array_normalization_method": _PIPELINE_PEPTIDE[
        "array_normalization_method"
    ],
    "default_qc_mode": _PIPELINE_PEPTIDE["qc_mode"],
    "default_contrast_mode": _PIPELINE_PEPTIDE["contrast_mode"],
    "default_qc_krsa_signal_threshold": _PIPELINE_PEPTIDE["qc_krsa_signal_threshold"],
    "default_qc_krsa_r2_threshold": _PIPELINE_PEPTIDE["qc_krsa_r2_threshold"],
    "batch_column": _PIPELINE_PEPTIDE["batch_column"],
    # The peptide waterfall draws the KPEA log2 fold change cutoffs, so it reads
    # them from default_kinase_analysis_params rather than carrying its own copy.
    "default_waterfall_lfc_cutoffs": tuple(_PIPELINE_KINASE["kpea_lfc_cutoffs"]),
    "default_waterfall_cutoff_mode": _PIPELINE_KINASE["kpea_cutoff_mode"],
    "default_waterfall_primary_lfc_cutoff": _PIPELINE_KINASE[
        "kpea_primary_lfc_cutoff"
    ],
}

PATHWAY_ENRICHMENT_DEFAULTS = {
    **_CONFIG["pathway_enrichment_analysis"],
    "allowed_pathway_backgrounds": tuple(
        _CONFIG["pathway_enrichment_analysis"]["allowed_pathway_backgrounds"]
    ),
    # Single source of truth: pipeline_defaults.yaml.
    "default_pathway_background": _PIPELINE_PATHWAY["pathway_background"],
    # Accession -> primary gene symbol of the UniProt release the PTM tables were
    # built from (config/data_enricher.yaml); the pathway query sends the symbols.
    "gene_symbols_path": _ENRICHER_PTM["default_uniprot_gene_symbols_path"],
}

UPSTREAM_KINASE_ANALYSIS_DEFAULTS = {
    **_CONFIG["upstream_kinase_analysis"],
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
    "allowed_kpea_background_universes": tuple(
        _CONFIG["upstream_kinase_analysis"]["allowed_kpea_background_universes"]
    ),
    "allowed_kpea_cutoff_modes": tuple(
        _CONFIG["upstream_kinase_analysis"]["allowed_kpea_cutoff_modes"]
    ),
    # Single source of truth: pipeline_defaults.yaml.
    "default_verified_evidence_levels": tuple(
        _PIPELINE_KINASE["verified_evidence_levels"]
    ),
    "default_kpea_lfc_cutoffs": tuple(_PIPELINE_KINASE["kpea_lfc_cutoffs"]),
    "default_kpea_cutoff_mode": _PIPELINE_KINASE["kpea_cutoff_mode"],
    "default_kpea_substrate_cutoff": _PIPELINE_KINASE["kpea_substrate_cutoff"],
    "default_kpea_zscore_threshold": _PIPELINE_KINASE["kpea_zscore_threshold"],
    "default_kpea_z_cap": _PIPELINE_KINASE["kpea_z_cap"],
    "default_kpea_background_universe": _PIPELINE_KINASE["kpea_background_universe"],
}
