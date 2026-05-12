"""Upstream kinase activity analysis for the pyKinaXe downstream workflow.

This module implements stage 2 of the downstream pipeline. It starts from the
peptide-level statistics produced by ``kx_peptide_analysis.py`` and combines
them with curated peptide-enrichment tables, BLAST-derived mapping data, and
PTM/kinase-substrate resources to estimate kinase activity changes.

Key responsibilities:

- map PamChip peptides to proteins and candidate kinases
- apply evidence filters for verified kinase-substrate interactions
- compute per-kinase activity summaries
- estimate significance through z-score and permutation-based procedures
- export kinase-level tables and volcano/venn-ready result sets

The main public class is ``KinaseActivityAnalysis``. Its outputs feed directly
into ``kx_pathway_enrichment_analysis.py`` and the workbook export logic in
``kx_pipeline_tools.py``.
"""

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests
from statsmodels.stats.multitest import multipletests
from numba import njit

from config.analysis_modules import UPSTREAM_KINASE_ANALYSIS_DEFAULTS
from kx_plot_results import VolcanoPlot, VennDiagramPlot

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - optional GUI dependency
    tk = None


@njit(cache=False, nogil=True)
def _counts_to_z_numba_1d(counts, mean_null, std_null, z_cap):
    """Return counts to z Numba 1d.
    
    Args:
        counts: Counts processed by this function.
        mean_null: Mean null processed by this function.
        std_null: Std null processed by this function.
        z_cap: Z cap processed by this function.
    
    Returns:
        object: Counts to z Numba 1d.
    """
    n = counts.shape[0]
    z = np.empty(n, dtype=np.float64)
    for i in range(n):
        std_value = std_null[i]
        diff = counts[i] - mean_null[i]
        if std_value > 1e-12:
            z_value = diff / std_value
        elif abs(diff) > 0.5:
            z_value = z_cap if diff > 0.0 else -z_cap
        else:
            z_value = 0.0

        if z_value > z_cap:
            z_value = z_cap
        elif z_value < -z_cap:
            z_value = -z_cap
        z[i] = z_value
    return z


@njit(cache=False, nogil=True)
def _accumulate_counts_to_z_sum_numba_2d(null_counts, mean_null, std_null, z_cap, out):
    """Return accumulate counts to z sum Numba 2d.
    
    Args:
        null_counts: Null counts processed by this function.
        mean_null: Mean null processed by this function.
        std_null: Std null processed by this function.
        z_cap: Z cap processed by this function.
        out: Out processed by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    n_permutations, n_kinases = null_counts.shape
    for perm_idx in range(n_permutations):
        for kinase_idx in range(n_kinases):
            std_value = std_null[kinase_idx]
            diff = null_counts[perm_idx, kinase_idx] - mean_null[kinase_idx]
            if std_value > 1e-12:
                z_value = diff / std_value
            elif abs(diff) > 0.5:
                z_value = z_cap if diff > 0.0 else -z_cap
            else:
                z_value = 0.0

            if z_value > z_cap:
                z_value = z_cap
            elif z_value < -z_cap:
                z_value = -z_cap
            out[perm_idx, kinase_idx] += z_value


@njit(cache=False, nogil=True)
def _summarize_kinase_arrays_numba(substrate_masks, peptide_stats, peptide_changes):
    """Return summarize kinase arrays Numba.
    
    Args:
        substrate_masks: Substrate masks processed by this function.
        peptide_stats: Peptide stats processed by this function.
        peptide_changes: Peptide changes processed by this function.
    
    Returns:
        tuple: Summarize kinase arrays Numba.
    """
    n_kinases, n_peptides = substrate_masks.shape
    num_substrates = np.zeros(n_kinases, dtype=np.int64)
    mean_stats = np.empty(n_kinases, dtype=np.float64)
    median_stats = np.empty(n_kinases, dtype=np.float64)
    mean_changes = np.empty(n_kinases, dtype=np.float64)

    for kinase_idx in range(n_kinases):
        stats_buffer = np.empty(n_peptides, dtype=np.float64)
        changes_buffer = np.empty(n_peptides, dtype=np.float64)
        substrate_count = 0
        stats_count = 0
        changes_count = 0
        stats_sum = 0.0
        changes_sum = 0.0

        for peptide_idx in range(n_peptides):
            if not substrate_masks[kinase_idx, peptide_idx]:
                continue

            substrate_count += 1

            stat_value = peptide_stats[peptide_idx]
            if not np.isnan(stat_value):
                stats_buffer[stats_count] = stat_value
                stats_sum += stat_value
                stats_count += 1

            change_value = peptide_changes[peptide_idx]
            if not np.isnan(change_value):
                changes_buffer[changes_count] = change_value
                changes_sum += change_value
                changes_count += 1

        num_substrates[kinase_idx] = substrate_count

        if stats_count > 0:
            mean_stats[kinase_idx] = stats_sum / stats_count
            sorted_stats = np.sort(stats_buffer[:stats_count])
            middle = stats_count // 2
            if stats_count % 2 == 0:
                median_stats[kinase_idx] = (
                    sorted_stats[middle - 1] + sorted_stats[middle]
                ) / 2.0
            else:
                median_stats[kinase_idx] = sorted_stats[middle]
        else:
            mean_stats[kinase_idx] = np.nan
            median_stats[kinase_idx] = np.nan

        if changes_count > 0:
            mean_changes[kinase_idx] = changes_sum / changes_count
        else:
            mean_changes[kinase_idx] = np.nan

    return num_substrates, mean_stats, median_stats, mean_changes


class KinaseActivityAnalysis:
    """Stage 2 of the UKA/KPEA workflow: kinase mapping and scoring."""

    DEFAULT_VERIFIED_EVIDENCE_LEVELS = tuple(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["default_verified_evidence_levels"]
    )
    DEFAULT_BLAST_RESULTS_RELATIVE_PATH = Path(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["default_blast_results_relative_path"]
    )
    UKA_METRIC_OPTIONS = dict(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["uka_metric_options"]
    )
    ALLOWED_KPEA_SIGNIFICANCE_METHODS = tuple(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["allowed_kpea_significance_methods"]
    )
    ALLOWED_KPEA_CUTOFF_MODES = tuple(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["allowed_kpea_cutoff_modes"]
    )
    MINIMUM_N_PERMUTATIONS = int(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["minimum_n_permutations"]
    )

    def __init__(
        self,
        path_file_enrichment_peptides=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "path_file_enrichment_peptides"
        ],
        BLAST_threshold=70,
        use_verified_interactions_only=False,
        require_known_ptm_site=False,
        verified_evidence_levels=None,
        verified_min_score=None,
        verified_min_references=None,
        volcano_plot=False,
        volcano_plot_GUI=False,
        volcano_plot_output=None,
        uka_visualization_metric=None,
        uka_volcano_x_axis=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "default_uka_volcano_x_axis"
        ],
        path_output_peptide_statistic=None,
        debugging_print=True,
        n_permutations=5000,
        kpea_lfc_cutoffs=None,
        kpea_cutoff_mode=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "default_kpea_cutoff_mode"
        ],
        kpea_primary_lfc_cutoff=None,
        kpea_substrate_cutoff=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "default_kpea_substrate_cutoff"
        ],
        kpea_zscore_threshold=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "default_kpea_zscore_threshold"
        ],
        kpea_z_cap=UPSTREAM_KINASE_ANALYSIS_DEFAULTS["default_kpea_z_cap"],
        kpea_significance_method=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "default_kpea_significance_method"
        ],
        kpea_empirical_p_threshold=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "default_kpea_empirical_p_threshold"
        ],
        kpea_fdr_threshold=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "default_kpea_fdr_threshold"
        ],
        input_stk_ptm_path=UPSTREAM_KINASE_ANALYSIS_DEFAULTS["input_stk_ptm_path"],
        input_ptk_ptm_path=UPSTREAM_KINASE_ANALYSIS_DEFAULTS["input_ptk_ptm_path"],
    ):
        """Initialize the KinaseActivityAnalysis instance.
        
        Args:
            path_file_enrichment_peptides: Path to the file enrichment peptides.
            BLAST_threshold: Threshold value used to filter, classify, or flag results.
            use_verified_interactions_only: Boolean flag controlling whether to use verified interactions only.
            require_known_ptm_site: Boolean flag controlling whether to require known PTM site.
            verified_evidence_levels: Verified evidence levels processed by this function.
            verified_min_score: Verified min score processed by this function.
            verified_min_references: Verified min references processed by this function.
            volcano_plot: Volcano plot processed by this function.
            volcano_plot_GUI: Volcano plot GUI processed by this function.
            volcano_plot_output: Volcano plot output processed by this function.
            uka_visualization_metric: UKA visualization metric processed by this function.
            uka_volcano_x_axis: UKA volcano x axis processed by this function.
            path_output_peptide_statistic: Path to the output peptide statistic.
            debugging_print: Whether to print additional debug information.
            n_permutations: Number of permutations used by this function.
            kpea_lfc_cutoffs: KPEA lfc cutoffs processed by this function.
            kpea_cutoff_mode: KPEA cutoff mode processed by this function.
            kpea_primary_lfc_cutoff: KPEA primary lfc cutoff processed by this function.
            kpea_substrate_cutoff: KPEA substrate cutoff processed by this function.
            kpea_zscore_threshold: Threshold value used to filter, classify, or flag results.
            kpea_z_cap: KPEA z cap processed by this function.
            kpea_significance_method: KPEA significance method processed by this function.
            kpea_empirical_p_threshold: Threshold value used to filter, classify, or flag results.
            kpea_fdr_threshold: Threshold value used to filter, classify, or flag results.
            input_stk_ptm_path: Path to the input STK PTM.
            input_ptk_ptm_path: Path to the input PTK PTM.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        self.path_file_enrichment_peptides = Path(path_file_enrichment_peptides)
        self.BLAST_threshold = BLAST_threshold
        self.use_verified_interactions_only = use_verified_interactions_only
        self.require_known_ptm_site = require_known_ptm_site
        if verified_evidence_levels is None:
            verified_evidence_levels = self.DEFAULT_VERIFIED_EVIDENCE_LEVELS
        self.verified_evidence_levels = {
            str(level).strip().lower()
            for level in verified_evidence_levels
            if str(level).strip()
        }
        self.verified_min_score = (
            None if verified_min_score is None else float(verified_min_score)
        )
        self.verified_min_references = (
            None if verified_min_references is None else int(verified_min_references)
        )
        if self.verified_min_score is not None and self.verified_min_score < 0:
            raise ValueError("verified_min_score must be >= 0.")
        if self.verified_min_references is not None and self.verified_min_references < 0:
            raise ValueError("verified_min_references must be >= 0.")
        self.volcano_plot = volcano_plot
        self.volcano_plot_GUI = volcano_plot_GUI
        self.volcano_plot_output = (
            Path(volcano_plot_output) if volcano_plot_output is not None else None
        )
        self.path_output_peptide_statistic = (
            Path(path_output_peptide_statistic)
            if path_output_peptide_statistic is not None
            else None
        )
        self.debugging_print = debugging_print

        self.df_blast_path = self._resolve_data_path(
            self.DEFAULT_BLAST_RESULTS_RELATIVE_PATH
        )
        self.input_stk_ptm_path = Path(input_stk_ptm_path)
        self.input_ptk_ptm_path = Path(input_ptk_ptm_path)
        self._peptide_enrichment_cache = None
        self._blast_cache = None
        self._blast_peptide_to_proteins_cache = None
        self._ptm_cache = None
        self._substrate_lookup_cache = {}
        self._kinase_name_cache = {}

        resolved_significance_method = str(kpea_significance_method).lower()
        if resolved_significance_method not in self.ALLOWED_KPEA_SIGNIFICANCE_METHODS:
            raise ValueError(
                f"Unknown kpea_significance_method '{kpea_significance_method}'. "
                "Use 'z_score', 'p_value', or 'fdr'."
            )
        resolved_cutoff_mode = str(kpea_cutoff_mode).lower()
        if resolved_cutoff_mode not in self.ALLOWED_KPEA_CUTOFF_MODES:
            raise ValueError(
                f"Unknown kpea_cutoff_mode '{kpea_cutoff_mode}'. "
                "Use 'average' or 'primary'."
            )

        resolved_uka_metric = uka_visualization_metric
        if resolved_uka_metric is None:
            resolved_uka_metric = uka_volcano_x_axis
        resolved_uka_metric = str(resolved_uka_metric).lower()
        if resolved_uka_metric not in self.UKA_METRIC_OPTIONS:
            raise ValueError(
                f"Unknown uka_visualization_metric '{resolved_uka_metric}'. "
                "Use 'kinase_change' or 'kinase_statistic'."
            )

        self.uka_visualization_metric = resolved_uka_metric
        metric_config = self.UKA_METRIC_OPTIONS[resolved_uka_metric]
        self.uka_visualization_column = metric_config["column"]
        self.uka_visualization_label = metric_config["label"]

        self.n_permutations = int(n_permutations)
        if self.n_permutations < self.MINIMUM_N_PERMUTATIONS:
            raise ValueError(
                f"n_permutations must be >= {self.MINIMUM_N_PERMUTATIONS} "
                f"for a stable empirical null; got {self.n_permutations}."
            )

        self.kpea_lfc_cutoffs = tuple(
            float(cutoff)
            for cutoff in (
                UPSTREAM_KINASE_ANALYSIS_DEFAULTS["default_kpea_lfc_cutoffs"]
                if kpea_lfc_cutoffs is None
                else kpea_lfc_cutoffs
            )
        )
        if not self.kpea_lfc_cutoffs:
            raise ValueError("kpea_lfc_cutoffs must contain at least one cutoff.")
        if any((not np.isfinite(cutoff)) or cutoff < 0 for cutoff in self.kpea_lfc_cutoffs):
            raise ValueError("kpea_lfc_cutoffs must contain only finite values >= 0.")

        self.kpea_cutoff_mode = resolved_cutoff_mode
        self.kpea_primary_lfc_cutoff = float(
            self.kpea_lfc_cutoffs[0]
            if kpea_primary_lfc_cutoff is None
            else kpea_primary_lfc_cutoff
        )
        if (not np.isfinite(self.kpea_primary_lfc_cutoff)) or self.kpea_primary_lfc_cutoff < 0:
            raise ValueError("kpea_primary_lfc_cutoff must be a finite value >= 0.")

        self.kpea_substrate_cutoff = int(kpea_substrate_cutoff)
        self.kpea_zscore_threshold = float(kpea_zscore_threshold)
        self.kpea_z_cap = float(kpea_z_cap)
        self.kpea_significance_method = resolved_significance_method
        self.kpea_empirical_p_threshold = float(kpea_empirical_p_threshold)
        self.kpea_fdr_threshold = float(kpea_fdr_threshold)

    def _dprint(self, *args, **kwargs):
        """Print a message only when debug logging is enabled.
        
        Args:
            *args: Additional positional arguments forwarded by this function.
            **kwargs: Additional keyword arguments forwarded by this function.
        
        Returns:
            None: Debug output is emitted only for its side effects.
        """
        if self.debugging_print:
            print(*args, **kwargs)

    @staticmethod
    def _candidate_data_dirs():
        """Return candidate data dirs.
        
        Args:
            None.
        
        Returns:
            list: Candidate data dirs.
        """
        cwd = Path(os.getcwd())
        script_root_data_dir = Path(__file__).resolve().parent.parent / "data"
        configured_paths = [
            Path(path_value).expanduser()
            for path_value in UPSTREAM_KINASE_ANALYSIS_DEFAULTS["candidate_data_paths"]
        ]
        return [script_root_data_dir, *configured_paths]

    @classmethod
    def _resolve_data_path(cls, relative_path):
        """Resolve data path.
        
        Args:
            relative_path: Path to the relative.
        
        Returns:
            object: Resolved data path.
        """
        path = Path(relative_path)
        if path.is_absolute():
            return path

        if path.parts and path.parts[0] == "data":
            relative_to_data_dir = Path(*path.parts[1:])
        else:
            relative_to_data_dir = path

        for data_dir in cls._candidate_data_dirs():
            candidate = data_dir / relative_to_data_dir
            if candidate.exists():
                return candidate

        return Path("data") / relative_to_data_dir

    @staticmethod
    def _safe_filename_token(value):
        """Return a safe filename token.
        
        Args:
            value: Input value processed by this helper.
        
        Returns:
            object: Safe filename token.
        """
        token = "NA" if value is None else str(value).strip()
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token).strip("_")
        return token or "NA"

    @staticmethod
    def _comparison_label(condition, payload):
        """Return comparison label.
        
        Args:
            condition: Condition processed by this function.
            payload: Payload processed by this function.
        
        Returns:
            object: Comparison label.
        """
        control = None
        if isinstance(payload, dict):
            control = payload.get("control_condition")
        if control is None:
            return str(condition)
        return f"{control} vs {condition}"

    @staticmethod
    def _extract_kinase_items(payload):
        """Extract kinase items.
        
        Args:
            payload: Payload processed by this function.
        
        Returns:
            object: Extracted kinase items.
        """
        candidate = payload
        if isinstance(payload, dict):
            candidate = payload.get("significant_kinases")
            if candidate is None and isinstance(payload.get("kinase_analysis"), dict):
                candidate = payload["kinase_analysis"].get("significant_kinases")

        if candidate is None:
            return []
        if isinstance(candidate, pd.DataFrame):
            if candidate.empty or "Kinase" not in candidate.columns:
                return []
            return candidate["Kinase"].dropna().astype(str).tolist()
        return list(candidate)

    def plot_kinase_overlap_venn(
        self,
        kinase_results_by_condition,
        output_path,
        save_tables=True,
    ):
        """Plot overlaps of significant kinases across condition comparisons.
        
        Args:
            kinase_results_by_condition: Kinase results by condition processed by this function.
            output_path: Path to the output.
            save_tables: Save tables processed by this function.
        
        Returns:
            dict: Plot output for kinase overlap venn.
        """
        if not kinase_results_by_condition:
            return {}

        groups = {
            self._comparison_label(condition, payload): self._extract_kinase_items(payload)
            for condition, payload in kinase_results_by_condition.items()
        }
        if not any(groups.values()):
            self._dprint("     No significant kinases found for overlap plotting.")
            return {}

        output_path = Path(output_path)
        save_path = output_path / "kinases_significant_overlap.png"
        table_dir = output_path / "kinases_significant_overlap_tables" if save_tables else None

        plotter = VennDiagramPlot(
            groups=groups,
            title="Significant kinase overlap",
            item_label="kinases",
            save_path=save_path,
            save_tables_dir=table_dir,
            debugging_print=self.debugging_print,
        )
        fig = plotter.plot()
        import matplotlib.pyplot as plt

        plt.close(fig)

        print(f"     Significant kinase overlap diagram saved: {save_path}")
        return {
            "plot": save_path,
            "tables": table_dir,
            "group_sizes": {
                group_name: len(group_values)
                for group_name, group_values in plotter.group_sets.items()
            },
        }

    def _write_debug_csv(self, df, suffix):
        """Write debug CSV.
        
        Args:
            df: Input pandas DataFrame used by this function.
            suffix: Suffix processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self.path_output_peptide_statistic is None:
            return
        output_path = Path(f"{self.path_output_peptide_statistic}_{suffix}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def _counts_to_z(self, counts, mean_null, std_null):
        """Return counts to z.
        
        Args:
            counts: Counts processed by this function.
            mean_null: Mean null processed by this function.
            std_null: Std null processed by this function.
        
        Returns:
            object: Counts to z.
        """
        counts, mean_null, std_null = np.broadcast_arrays(
            np.asarray(counts, dtype=float),
            np.asarray(mean_null, dtype=float),
            np.asarray(std_null, dtype=float),
        )

        z = np.zeros_like(counts, dtype=float)
        valid = std_null > 1e-12
        z[valid] = (counts[valid] - mean_null[valid]) / std_null[valid]

        constant = ~valid
        if np.any(constant):
            diff = counts[constant] - mean_null[constant]
            z[constant] = np.where(
                np.abs(diff) > 0.5,
                np.sign(diff) * self.kpea_z_cap,
                0.0,
            )

        return np.clip(z, -self.kpea_z_cap, self.kpea_z_cap)

    def _sample_null_counts(
        self,
        hit_indices,
        n_background_peptides,
        n_permutations,
        substrate_mask_matrix,
        rng,
    ):
        """Return sample null counts.
        
        Args:
            hit_indices: Hit indices processed by this function.
            n_background_peptides: Number of background peptides used by this function.
            n_permutations: Number of permutations used by this function.
            substrate_mask_matrix: Substrate mask matrix processed by this function.
            rng: Rng processed by this function.
        
        Returns:
            tuple: Sample null counts.
        """
        n_hits = int(len(hit_indices))
        n_kinases = substrate_mask_matrix.shape[1]
        if n_hits == 0:
            zero_obs = np.zeros(n_kinases, dtype=np.float64)
            zero_null = np.zeros((n_permutations, n_kinases), dtype=np.int64)
            return zero_obs, zero_null, n_hits

        observed_counts = substrate_mask_matrix[hit_indices].sum(axis=0).astype(np.float64)
        if n_hits >= n_background_peptides:
            full_counts = substrate_mask_matrix.sum(axis=0).astype(np.int64, copy=False)
            null_counts = np.repeat(full_counts[np.newaxis, :], n_permutations, axis=0)
        else:
            substrate_counts = substrate_mask_matrix.sum(axis=0).astype(np.int64, copy=False)
            null_counts = rng.hypergeometric(
                ngood=substrate_counts,
                nbad=n_background_peptides - substrate_counts,
                nsample=n_hits,
                size=(n_permutations, n_kinases),
            )
        return observed_counts, null_counts, n_hits

    def _zscore_significance_mask(self, df_kpea):
        """Return z-score significance mask.
        
        Args:
            df_kpea: Input pandas DataFrame containing KPEA.
        
        Returns:
            bool: Boolean result of the evaluated condition.
        """
        return df_kpea["Z_Score"].abs().fillna(0.0) >= self.kpea_zscore_threshold

    def _kpea_score_stat_label(self):
        """Return KPEA score stat label.
        
        Args:
            None.
        
        Returns:
            object: KPEA score stat label.
        """
        if self.kpea_cutoff_mode == "primary":
            return f"KRSA |Z| at cutoff {self.kpea_primary_lfc_cutoff:g}"
        return "KRSA |mean Z|"

    def _kpea_score_basis_label(self):
        """Return KPEA score basis label.
        
        Args:
            None.
        
        Returns:
            object: KPEA score basis label.
        """
        if self.kpea_cutoff_mode == "primary":
            return f"primary cutoff {self.kpea_primary_lfc_cutoff:g}"
        return f"mean across cutoffs {list(self.kpea_lfc_cutoffs)}"

    def _resolve_kpea_scoring_cutoffs(self, lfc_cutoffs):
        """Resolve KPEA scoring cutoffs.
        
        Args:
            lfc_cutoffs: Lfc cutoffs processed by this function.
        
        Returns:
            tuple: Resolved KPEA scoring cutoffs.
        """
        normalized_cutoffs = tuple(float(cutoff) for cutoff in lfc_cutoffs)
        if not normalized_cutoffs:
            raise ValueError("At least one KPEA cutoff is required for scoring.")
        if any((not np.isfinite(cutoff)) or cutoff < 0 for cutoff in normalized_cutoffs):
            raise ValueError("KPEA scoring cutoffs must be finite values >= 0.")

        if self.kpea_cutoff_mode == "primary":
            return (float(self.kpea_primary_lfc_cutoff),), float(
                self.kpea_primary_lfc_cutoff
            )
        return normalized_cutoffs, np.nan

    def _active_significance_label(self):
        """Return active significance label.
        
        Args:
            None.
        
        Returns:
            object: Active significance label.
        """
        if self.kpea_significance_method == "z_score":
            return f"{self._kpea_score_stat_label()} >= {self.kpea_zscore_threshold}"
        if self.kpea_significance_method == "p_value":
            return (
                f"empirical p ({self._kpea_score_basis_label()}) < "
                f"{self.kpea_empirical_p_threshold}"
            )
        return f"FDR ({self._kpea_score_basis_label()}) < {self.kpea_fdr_threshold}"

    def _annotate_significance_columns(self, df_kpea):
        """Annotate significance columns.
        
        Args:
            df_kpea: Input pandas DataFrame containing KPEA.
        
        Returns:
            object: Annotated significance columns.
        """
        if df_kpea is None or df_kpea.empty:
            return df_kpea

        df_kpea = df_kpea.copy()
        zscore_sig = self._zscore_significance_mask(df_kpea)
        empirical_sig = df_kpea["p_value"] < self.kpea_empirical_p_threshold
        fdr_sig = df_kpea["FDR"] < self.kpea_fdr_threshold

        df_kpea["Significant_ZScore"] = zscore_sig
        df_kpea["Significant_EmpiricalPValue"] = empirical_sig
        df_kpea["Significant_FDR"] = fdr_sig

        if self.kpea_significance_method == "z_score":
            significant_selected_method = zscore_sig
        elif self.kpea_significance_method == "p_value":
            significant_selected_method = empirical_sig
        else:
            significant_selected_method = fdr_sig

        df_kpea["Significant_SelectedMethod"] = significant_selected_method
        df_kpea["SelectedForReport"] = significant_selected_method
        df_kpea["Significant"] = df_kpea["Significant_SelectedMethod"]
        return df_kpea

    @staticmethod
    def _split_delimited_values(raw_value):
        """Split delimited values.
        
        Args:
            raw_value: Raw value processed by this function.
        
        Returns:
            object: Split delimited values.
        """
        if pd.isna(raw_value):
            return set()
        return {
            str(token).strip()
            for token in re.split(r"[;,]", str(raw_value))
            if str(token).strip()
        }

    @staticmethod
    def _is_known_ptm_site(site):
        """Return whether known PTM site.
        
        Args:
            site: Site processed by this function.
        
        Returns:
            bool: True when is known PTM site is satisfied, otherwise False.
        """
        if pd.isna(site):
            return False

        site_str = str(site).strip()
        if not site_str:
            return False

        normalized_site = site_str.lower()
        if normalized_site in {"unknown", "na", "nan", "none"}:
            return False

        if ";" in site_str:
            return any(
                KinaseActivityAnalysis._is_known_ptm_site(part)
                for part in site_str.split(";")
            )

        return bool(re.fullmatch(r"[YST]\d+", site_str.upper()))

    @staticmethod
    def _extract_known_ptm_site_tokens(raw_site):
        """Extract known PTM site tokens.
        
        Args:
            raw_site: Raw site processed by this function.
        
        Returns:
            object: Extracted known PTM site tokens.
        """
        if pd.isna(raw_site):
            return tuple()

        tokens = []
        for token in re.split(r"[;,]", str(raw_site)):
            normalized = str(token).strip().upper()
            if KinaseActivityAnalysis._is_known_ptm_site(normalized):
                tokens.append(normalized)
        return tuple(sorted(set(tokens)))

    @staticmethod
    def _is_generic_ptm_site(site):
        """Return whether generic PTM site.
        
        Args:
            site: Site processed by this function.
        
        Returns:
            bool: True when is generic PTM site is satisfied, otherwise False.
        """
        normalized = str(site).strip().upper()
        return normalized in {"Y", "S/T"}

    @classmethod
    def _extract_informative_ptm_site_tokens(cls, raw_site):
        """Extract informative PTM site tokens.
        
        Args:
            raw_site: Raw site processed by this function.
        
        Returns:
            object: Extracted informative PTM site tokens.
        """
        if pd.isna(raw_site):
            return tuple()

        tokens = []
        for token in re.split(r"[;,]", str(raw_site)):
            normalized = str(token).strip().upper()
            if cls._is_known_ptm_site(normalized) or cls._is_generic_ptm_site(normalized):
                tokens.append(normalized)
        return tuple(sorted(set(tokens)))

    @classmethod
    def _has_informative_ptm_site(cls, raw_site):
        """Return whether informative PTM site.
        
        Args:
            raw_site: Raw site processed by this function.
        
        Returns:
            bool: True when has informative PTM site is satisfied, otherwise False.
        """
        return bool(cls._extract_informative_ptm_site_tokens(raw_site))

    def _site_matches_array_type(self, raw_site, array_type):
        """Return site matches array type.
        
        Args:
            raw_site: Raw site processed by this function.
            array_type: Array type processed by this function.
        
        Returns:
            object: Site matches array type.
        """
        informative_sites = self._extract_informative_ptm_site_tokens(raw_site)
        if not informative_sites:
            return False

        array_type = str(array_type).upper()
        if array_type == "PTK":
            return all(site.startswith("Y") for site in informative_sites)
        if array_type == "STK":
            return all(site.startswith(("S", "T")) for site in informative_sites)
        return True

    def _ptm_sites_match_peptide_sites(self, peptide_sites, ptm_sites):
        """Return PTM sites match peptide sites.
        
        Args:
            peptide_sites: Peptide sites processed by this function.
            ptm_sites: PTM sites processed by this function.
        
        Returns:
            object: PTM sites match peptide sites.
        """
        normalized_peptide_sites = {
            str(site).strip().upper()
            for site in peptide_sites
            if self._is_known_ptm_site(site)
        }
        normalized_ptm_sites = {
            str(site).strip().upper()
            for site in ptm_sites
            if self._is_known_ptm_site(site)
        }
        if not normalized_peptide_sites or not normalized_ptm_sites:
            return False
        return not normalized_peptide_sites.isdisjoint(normalized_ptm_sites)

    def _substrate_matches_peptide(
        self,
        candidate_sites,
        direct_uniprots,
        substrate,
        positioned_ptm_sites,
        has_generic_ptm_site,
    ):
        """Return substrate matches peptide.
        
        Args:
            candidate_sites: Candidate sites processed by this function.
            direct_uniprots: Direct uniprots processed by this function.
            substrate: Substrate processed by this function.
            positioned_ptm_sites: Positioned PTM sites processed by this function.
            has_generic_ptm_site: Has generic PTM site processed by this function.
        
        Returns:
            object: Substrate matches peptide.
        """
        if self.require_known_ptm_site:
            if positioned_ptm_sites:
                return self._ptm_sites_match_peptide_sites(
                    candidate_sites,
                    positioned_ptm_sites,
                )
            return has_generic_ptm_site

        if substrate in direct_uniprots and candidate_sites and positioned_ptm_sites:
            return not candidate_sites.isdisjoint(positioned_ptm_sites)
        return True

    @staticmethod
    def _normalize_evidence_level(level):
        """Normalize evidence level.
        
        Args:
            level: Level processed by this function.
        
        Returns:
            object: Normalized evidence level.
        """
        if pd.isna(level):
            return ""
        return str(level).strip().lower()

    def _verified_interaction_mask(self, df_ptm):
        """Return verified interaction mask.
        
        Args:
            df_ptm: Input pandas DataFrame containing PTM.
        
        Returns:
            object: Verified interaction mask.
        """
        if df_ptm is None or df_ptm.empty:
            return pd.Series(dtype=bool)

        mask = pd.Series(True, index=df_ptm.index)

        if self.verified_evidence_levels:
            if "evidence_level" not in df_ptm.columns:
                raise ValueError(
                    "use_verified_interactions_only=True requires the OmniPath PTM "
                    "files to contain an 'evidence_level' column."
                )
            evidence_levels = df_ptm["evidence_level"].apply(
                self._normalize_evidence_level
            )
            mask &= evidence_levels.isin(self.verified_evidence_levels)

        if self.verified_min_score is not None:
            if "score" not in df_ptm.columns:
                raise ValueError(
                    "verified_min_score requires the OmniPath PTM files to contain "
                    "a 'score' column."
                )
            mask &= (
                pd.to_numeric(df_ptm["score"], errors="coerce").fillna(-np.inf)
                >= self.verified_min_score
            )

        if self.verified_min_references is not None:
            if "n_references" not in df_ptm.columns:
                raise ValueError(
                    "verified_min_references requires the OmniPath PTM files to "
                    "contain an 'n_references' column."
                )
            mask &= (
                pd.to_numeric(df_ptm["n_references"], errors="coerce").fillna(-1)
                >= self.verified_min_references
            )

        return mask

    def _verified_filter_description(self):
        """Return verified filter description.
        
        Args:
            None.
        
        Returns:
            object: Verified filter description.
        """
        parts = []
        if self.verified_evidence_levels:
            parts.append(
                "evidence_level in "
                f"{sorted(self.verified_evidence_levels)}"
            )
        if self.verified_min_score is not None:
            parts.append(f"score >= {self.verified_min_score:g}")
        if self.verified_min_references is not None:
            parts.append(f"n_references >= {self.verified_min_references}")
        return ", ".join(parts) if parts else "no additional criteria"

    @staticmethod
    def _extract_candidate_sites_from_peptide(peptide_id, sequence):
        """Extract candidate sites from peptide.
        
        Args:
            peptide_id: Peptide ID processed by this function.
            sequence: Sequence processed by this function.
        
        Returns:
            object: Extracted candidate sites from peptide.
        """
        if pd.isna(peptide_id) or pd.isna(sequence):
            return set()

        match = re.search(r"_(\d+)_(\d+)$", str(peptide_id))
        if match is None:
            return set()

        start_pos = int(match.group(1))
        raw_sequence = str(sequence)
        normalized_sequence = (
            raw_sequence.replace("(pY)", "Y")
            .replace("(pS)", "S")
            .replace("(pT)", "T")
            .replace("pY", "Y")
            .replace("pS", "S")
            .replace("pT", "T")
        )

        explicit_sites = set()
        sequence_cursor = 0
        token_pattern = re.compile(r"\(p([YST])\)|([A-Z])")
        for token in token_pattern.finditer(raw_sequence):
            modified_residue = token.group(1)
            plain_residue = token.group(2)
            residue = modified_residue or plain_residue
            if modified_residue:
                explicit_sites.add(f"{modified_residue}{start_pos + sequence_cursor}")
            if residue:
                sequence_cursor += 1

        if explicit_sites:
            return explicit_sites

        candidate_sites = set()
        for offset, residue in enumerate(normalized_sequence):
            if residue in {"Y", "S", "T"}:
                candidate_sites.add(f"{residue}{start_pos + offset}")
        return candidate_sites

    def _collapse_duplicate_peptides_for_uka(self, df_pooled):
        """Collapse duplicate peptides for UKA.
        
        Args:
            df_pooled: Input pandas DataFrame containing pooled.
        
        Returns:
            object: Collapsed duplicate peptides for UKA.
        """
        if df_pooled.empty or "ID" not in df_pooled.columns:
            return df_pooled

        duplicated_mask = df_pooled["ID"].duplicated(keep=False)
        if not duplicated_mask.any():
            return df_pooled.copy()

        collapsed_rows = []
        for _, group in df_pooled.groupby("ID", sort=False):
            representative = group.iloc[0].copy()

            direct_uniprots = sorted(
                {
                    str(value).strip()
                    for value in group.get(
                        "UniprotAccession", pd.Series(dtype=object)
                    ).dropna()
                    if str(value).strip()
                }
            )
            direct_gene_names = sorted(
                {
                    str(value).strip()
                    for value in group.get("GeneName", pd.Series(dtype=object)).dropna()
                    if str(value).strip()
                }
            )

            if direct_uniprots:
                representative["UniprotAccession"] = direct_uniprots[0]
            if direct_gene_names:
                representative["GeneName"] = direct_gene_names[0]

            representative["DirectUniprotAccessions"] = ";".join(direct_uniprots)
            representative["DirectGeneNames"] = ";".join(direct_gene_names)
            collapsed_rows.append(representative)

        df_collapsed = pd.DataFrame(collapsed_rows).reset_index(drop=True)
        self._dprint(
            "     Collapsed duplicate peptide annotations for UKA: "
            f"{len(df_pooled)} -> {len(df_collapsed)} rows."
        )
        return df_collapsed

    def _filter_ptm_interactions_for_uka(self, df_ptm, array_type="PTM"):
        """Filter PTM interactions for UKA.
        
        Args:
            df_ptm: Input pandas DataFrame containing PTM.
            array_type: Array type processed by this function.
        
        Returns:
            object: Filtered PTM interactions for UKA.
        """
        if df_ptm is None or df_ptm.empty:
            return df_ptm

        df_ptm_filtered = df_ptm.copy()
        if "source_database" not in df_ptm_filtered.columns:
            if "source" in df_ptm_filtered.columns:
                df_ptm_filtered["source_database"] = df_ptm_filtered["source"]
            else:
                df_ptm_filtered["source_database"] = "unknown"

        if "site" not in df_ptm_filtered.columns:
            df_ptm_filtered["site"] = "unknown"

        before_total = len(df_ptm_filtered)
        if self.use_verified_interactions_only:
            df_ptm_filtered = df_ptm_filtered[
                self._verified_interaction_mask(df_ptm_filtered)
            ].copy()
            self._dprint(
                f"     {array_type}: kept {len(df_ptm_filtered)}/{before_total} "
                "OmniPath verified interactions "
                f"({self._verified_filter_description()})"
            )
            before_total = len(df_ptm_filtered)

        if self.require_known_ptm_site:
            df_ptm_filtered = df_ptm_filtered[
                df_ptm_filtered["site"].apply(self._has_informative_ptm_site)
            ].copy()
            before_total = len(df_ptm_filtered)
            df_ptm_filtered = df_ptm_filtered[
                df_ptm_filtered["site"].apply(
                    lambda raw_site: self._site_matches_array_type(raw_site, array_type)
                )
            ].copy()

        df_ptm_filtered = df_ptm_filtered.drop_duplicates(
            subset=["uniprot_id", "ptm_enzyme", "site"]
        ).reset_index(drop=True)
        return df_ptm_filtered

    def _load_BLAST(self, df_peptide_enrichment):
        """Load BLAST.
        
        Args:
            df_peptide_enrichment: Input pandas DataFrame containing peptide enrichment.
        
        Returns:
            object: Loaded BLAST.
        """
        if not self.df_blast_path.exists():
            tried_paths = [
                data_dir / self.DEFAULT_BLAST_RESULTS_RELATIVE_PATH
                for data_dir in self._candidate_data_dirs()
            ]
            raise FileNotFoundError(
                "BLAST results file not found. Tried:\n"
                + "\n".join(f"  - {path}" for path in tried_paths)
                + "\n\nPass an explicit path or place the file under a detected data directory."
            )

        df_BLAST = pd.read_csv(self.df_blast_path)
        df_BLAST.rename(columns={"source_uniprot_id": "source_peptide_id"}, inplace=True)

        df_BLAST_merged = df_BLAST.merge(
            df_peptide_enrichment,
            left_on="source_peptide_id",
            right_on="ID",
            how="left",
        )

        df_BLAST_merged["is_direct_chip_match"] = (
            df_BLAST_merged["Accession"]
            .fillna("")
            .astype(str)
            .eq(df_BLAST_merged["PepProtein_UniprotID"].fillna("").astype(str))
        )
        df_BLAST_merged.rename(
            columns={"PepProtein_UniprotID": "source_uniprot_id"},
            inplace=True,
        )
        df_BLAST_merged["subject_uniprot_id"] = df_BLAST_merged["Accession"]

        df_BLAST_filtered = df_BLAST_merged.query(
            f"`Positives(%)` >= {self.BLAST_threshold}"
        )

        sort_cols = [
            "source_peptide_id",
            "is_direct_chip_match",
            "Positives(%)",
            "Identities(%)",
        ]
        ascending = [True, False, False, False]
        if "Score(Bits)" in df_BLAST_filtered.columns:
            sort_cols.append("Score(Bits)")
            ascending.append(False)
        if "Hit" in df_BLAST_filtered.columns:
            sort_cols.append("Hit")
            ascending.append(True)

        df_BLAST_filtered = (
            df_BLAST_filtered.sort_values(sort_cols, ascending=ascending)
            .drop_duplicates(subset=["source_peptide_id", "subject_uniprot_id"], keep="first")
        )

        self._dprint(
            "     BLAST data contains "
            f"{len(df_BLAST_filtered)} connections between peptides and proteins."
        )
        return df_BLAST_filtered

    def _load_peptide_enrichment(self):
        """Load peptide enrichment.
        
        Args:
            None.
        
        Returns:
            object: Loaded peptide enrichment.
        """
        return pd.read_csv(self.path_file_enrichment_peptides)

    def _get_peptide_enrichment(self):
        """Get peptide enrichment.
        
        Args:
            None.
        
        Returns:
            object: Requested peptide enrichment.
        """
        if self._peptide_enrichment_cache is None:
            self._peptide_enrichment_cache = self._load_peptide_enrichment()
        return self._peptide_enrichment_cache

    def _get_blast_data(self):
        """Get BLAST data.
        
        Args:
            None.
        
        Returns:
            object: Requested BLAST data.
        """
        if self._blast_cache is None:
            self._blast_cache = self._load_BLAST(
                df_peptide_enrichment=self._get_peptide_enrichment()
            )
        return self._blast_cache

    def _get_peptide_to_proteins(self, df_BLAST=None):
        """Get peptide to proteins.
        
        Args:
            df_BLAST: Input pandas DataFrame containing BLAST.
        
        Returns:
            object: Requested peptide to proteins.
        """
        if df_BLAST is None or df_BLAST is self._blast_cache:
            if self._blast_peptide_to_proteins_cache is None:
                cached_blast = self._get_blast_data()
                self._blast_peptide_to_proteins_cache = (
                    cached_blast.drop_duplicates(
                        subset=["source_peptide_id", "subject_uniprot_id"]
                    )
                    .groupby("source_peptide_id")["subject_uniprot_id"]
                    .apply(list)
                    .to_dict()
                )
            return self._blast_peptide_to_proteins_cache

        return (
            df_BLAST.drop_duplicates(subset=["source_peptide_id", "subject_uniprot_id"])
            .groupby("source_peptide_id")["subject_uniprot_id"]
            .apply(list)
            .to_dict()
        )

    def _load_and_merge_ptm_data(self):
        """Load and merge PTM data.
        
        Args:
            None.
        
        Returns:
            tuple: Loaded and merge PTM data.
        """
        missing_paths = [
            path
            for path in (self.input_stk_ptm_path, self.input_ptk_ptm_path)
            if not path.exists()
        ]
        if missing_paths:
            missing_str = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(
                "Missing PTM input file(s): "
                f"{missing_str}. Run `python src/kx_data_enricher.py omnipath` first, "
                "or pass input_stk_ptm_path/input_ptk_ptm_path explicitly."
            )

        ptm_stk = pd.read_csv(self.input_stk_ptm_path)
        ptm_ptk = pd.read_csv(self.input_ptk_ptm_path)

        ptm_stk = self._filter_ptm_interactions_for_uka(ptm_stk, array_type="STK")
        ptm_ptk = self._filter_ptm_interactions_for_uka(ptm_ptk, array_type="PTK")
        return ptm_stk, ptm_ptk

    def _get_ptm_data(self):
        """Get PTM data.
        
        Args:
            None.
        
        Returns:
            object: Requested PTM data.
        """
        if self._ptm_cache is None:
            self._ptm_cache = self._load_and_merge_ptm_data()
        return self._ptm_cache

    def _resolve_kinase_names_uniprot(self, uniprot_ids, batch_size=100):
        """Resolve kinase names UniProt.
        
        Args:
            uniprot_ids: UniProt IDs processed by this function.
            batch_size: Batch size processed by this function.
        
        Returns:
            object: Resolved kinase names UniProt.
        """
        id_to_name = {}
        unique_ids = list(dict.fromkeys(str(uid).strip() for uid in uniprot_ids if str(uid).strip()))
        if not unique_ids:
            return id_to_name

        uncached_ids = [
            uid for uid in unique_ids if uid not in self._kinase_name_cache
        ]
        if not uncached_ids:
            return {uid: self._kinase_name_cache[uid] for uid in unique_ids}

        self._dprint(
            f"     Resolving {len(uncached_ids)} kinase names via UniProt API..."
        )
        for i in range(0, len(uncached_ids), batch_size):
            batch = uncached_ids[i : i + batch_size]
            query = " OR ".join(f"accession:{uid}" for uid in batch)
            url = UPSTREAM_KINASE_ANALYSIS_DEFAULTS["uniprot_name_lookup_url"]
            params = {
                "query": query,
                "fields": "accession,gene_primary,protein_name",
                "format": "tsv",
                "size": str(len(batch)),
            }

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                lines = response.text.strip().split("\n")
                if len(lines) < 2:
                    continue

                header = lines[0].split("\t")
                for line in lines[1:]:
                    fields = line.split("\t")
                    row_dict = dict(zip(header, fields))
                    accession = row_dict.get("Entry", "").strip()
                    gene_name = row_dict.get("Gene Names (primary)", "").strip()
                    protein_name = row_dict.get("Protein names", "").strip()

                    name = gene_name if gene_name else protein_name
                    if accession:
                        self._kinase_name_cache[accession] = name if name else accession
            except requests.RequestException as exc:
                self._dprint(
                    "     WARNING: UniProt API request failed for batch "
                    f"{i // batch_size + 1}: {exc}"
                )
                for uid in batch:
                    self._kinase_name_cache.setdefault(uid, uid)

        for uid in uncached_ids:
            self._kinase_name_cache.setdefault(uid, uid)

        return {uid: self._kinase_name_cache[uid] for uid in unique_ids}

    def _build_substrate_to_kinase_lookup(self, df_ptm):
        """Build substrate to kinase lookup.
        
        Args:
            df_ptm: Input pandas DataFrame containing PTM.
        
        Returns:
            object: Constructed substrate to kinase lookup.
        """
        substrate_lookup = defaultdict(
            lambda: defaultdict(lambda: {"databases": set(), "ptm_sites": set()})
        )

        has_source_database = "source_database" in df_ptm.columns
        has_source = "source" in df_ptm.columns
        has_site = "site" in df_ptm.columns

        for row in df_ptm.itertuples(index=False):
            substrate = row.uniprot_id
            kinase = row.ptm_enzyme

            if has_source_database:
                raw_db = getattr(row, "source_database", "unknown")
            elif has_source:
                raw_db = getattr(row, "source", "unknown")
            else:
                raw_db = "unknown"
            database = str(raw_db).strip() if pd.notna(raw_db) else ""
            if not database:
                database = "unknown"

            site_value = getattr(row, "site", "") if has_site else ""
            entry = substrate_lookup[substrate][kinase]
            entry["databases"].add(database)
            entry["ptm_sites"].update(
                site_token.upper()
                for site_token in self._extract_informative_ptm_site_tokens(site_value)
            )

        normalized_lookup = {}
        for substrate, kinase_map in substrate_lookup.items():
            entries = []
            for kinase, payload in kinase_map.items():
                ptm_sites = frozenset(payload["ptm_sites"])
                positioned_ptm_sites = frozenset(
                    site for site in ptm_sites if self._is_known_ptm_site(site)
                )
                entries.append(
                    {
                        "kinase": kinase,
                        "databases": tuple(sorted(payload["databases"])),
                        "ptm_sites": ptm_sites,
                        "positioned_ptm_sites": positioned_ptm_sites,
                        "has_generic_ptm_site": any(
                            self._is_generic_ptm_site(site) for site in ptm_sites
                        ),
                    }
                )
            normalized_lookup[substrate] = entries

        return normalized_lookup

    def _get_substrate_to_kinase_lookup(self, df_ptm, cache_key=None):
        """Get substrate to kinase lookup.
        
        Args:
            df_ptm: Input pandas DataFrame containing PTM.
            cache_key: Cache key processed by this function.
        
        Returns:
            object: Requested substrate to kinase lookup.
        """
        if cache_key is not None and cache_key in self._substrate_lookup_cache:
            return self._substrate_lookup_cache[cache_key]

        lookup = self._build_substrate_to_kinase_lookup(df_ptm)
        if cache_key is not None:
            self._substrate_lookup_cache[cache_key] = lookup
        return lookup

    def _map_kinases_to_peptides(
        self,
        df_pooled,
        df_ptm,
        df_BLAST,
        include_mapping_rows=True,
        substrate_to_kinase_entries=None,
        peptide_to_proteins=None,
    ):
        """Map kinases to peptides.
        
        Args:
            df_pooled: Input pandas DataFrame containing pooled.
            df_ptm: Input pandas DataFrame containing PTM.
            df_BLAST: Input pandas DataFrame containing BLAST.
            include_mapping_rows: Boolean flag controlling whether to include mapping rows.
            substrate_to_kinase_entries: Substrate to kinase entries processed by this function.
            peptide_to_proteins: Peptide to proteins processed by this function.
        
        Returns:
            tuple: Mapped kinases to peptides.
        """
        mapping_columns = [
            "Peptide_ID",
            "Peptide_UniprotName",
            "Peptide_UniprotID",
            "Peptide_CandidateSites",
            "Substrate_BLAST",
            "Matched_PTM_Sites",
            "Kinase_UniprotID",
            "Kinase_UniprotName",
            "Source_Database",
        ]

        if substrate_to_kinase_entries is None:
            substrate_to_kinase_entries = self._build_substrate_to_kinase_lookup(df_ptm)
        if peptide_to_proteins is None:
            peptide_to_proteins = self._get_peptide_to_proteins(df_BLAST)

        gene_name_col = "DirectGeneNames" if "DirectGeneNames" in df_pooled.columns else "GeneName"
        direct_uniprot_col = (
            "DirectUniprotAccessions"
            if "DirectUniprotAccessions" in df_pooled.columns
            else "UniprotAccession"
        )

        peptide_info = {}
        quantitative_rows = {}
        peptide_order = []
        for peptide_row in df_pooled.itertuples(index=False):
            peptide_id = peptide_row.ID
            if peptide_id not in peptide_info:
                peptide_info[peptide_id] = {
                    "GeneNames": set(),
                    "DirectUniprotAccessions": set(),
                    "candidate_sites": set(),
                }
                quantitative_rows[peptide_id] = {
                    "mean_control": peptide_row.mean_control,
                    "mean_treatment": peptide_row.mean_treatment,
                }
                peptide_order.append(peptide_id)

            peptide_info[peptide_id]["GeneNames"].update(
                self._split_delimited_values(getattr(peptide_row, gene_name_col, ""))
            )
            peptide_info[peptide_id]["DirectUniprotAccessions"].update(
                self._split_delimited_values(
                    getattr(peptide_row, direct_uniprot_col, "")
                )
            )
            peptide_info[peptide_id]["candidate_sites"].update(
                self._extract_candidate_sites_from_peptide(
                    peptide_id,
                    getattr(peptide_row, "Sequence", ""),
                )
            )

        all_mapping_rows = []
        kinase_to_peptides = defaultdict(list)

        for peptide_id in peptide_order:
            matched_proteins = peptide_to_proteins.get(peptide_id)
            if not matched_proteins:
                continue

            pep_gene_name = ";".join(sorted(peptide_info[peptide_id]["GeneNames"]))
            pep_direct_uniprots = peptide_info[peptide_id]["DirectUniprotAccessions"]
            pep_uniprot = ";".join(sorted(pep_direct_uniprots))
            candidate_sites = peptide_info[peptide_id]["candidate_sites"]
            candidate_sites_str = (
                ";".join(sorted(candidate_sites)) if candidate_sites else ""
            )

            kinase_valid_matches = defaultdict(set)
            for substrate in matched_proteins:
                for entry in substrate_to_kinase_entries.get(substrate, ()):
                    if not self._substrate_matches_peptide(
                        candidate_sites=candidate_sites,
                        direct_uniprots=pep_direct_uniprots,
                        substrate=substrate,
                        positioned_ptm_sites=entry["positioned_ptm_sites"],
                        has_generic_ptm_site=entry["has_generic_ptm_site"],
                    ):
                        continue

                    kinase_valid_matches[entry["kinase"]].add(substrate)
                    if include_mapping_rows:
                        matched_ptm_sites = (
                            ";".join(sorted(entry["ptm_sites"]))
                            if entry["ptm_sites"]
                            else ""
                        )
                        for db in entry["databases"]:
                            all_mapping_rows.append(
                                {
                                    "Peptide_ID": peptide_id,
                                    "Peptide_UniprotName": pep_gene_name,
                                    "Peptide_UniprotID": pep_uniprot,
                                    "Peptide_CandidateSites": candidate_sites_str,
                                    "Substrate_BLAST": substrate,
                                    "Matched_PTM_Sites": matched_ptm_sites,
                                    "Kinase_UniprotID": entry["kinase"],
                                    "Kinase_UniprotName": "",
                                    "Source_Database": db,
                                }
                            )

            if not kinase_valid_matches:
                continue

            mean_control = quantitative_rows[peptide_id]["mean_control"]
            mean_treatment = quantitative_rows[peptide_id]["mean_treatment"]
            for kinase, valid_matches in kinase_valid_matches.items():
                kinase_to_peptides[kinase].append(
                    {
                        "peptide_id": peptide_id,
                        "mean_control": mean_control,
                        "mean_treatment": mean_treatment,
                        "matched_proteins": sorted(valid_matches),
                    }
                )

        if include_mapping_rows:
            df_full_mapping = pd.DataFrame(all_mapping_rows, columns=mapping_columns)
        else:
            df_full_mapping = pd.DataFrame(columns=mapping_columns)

        return kinase_to_peptides, df_full_mapping, peptide_order

    def _calculate_KPEA(
        self,
        df_pooled,
        df_ptm=None,
        df_BLAST=None,
        substrate_cutoff=None,
        n_permutations=None,
        lfc_cutoffs=None,
        kinase_to_peptides=None,
        peptide_order=None,
    ):
        """Calculate KPEA.
        
        Args:
            df_pooled: Input pandas DataFrame containing pooled.
            df_ptm: Input pandas DataFrame containing PTM.
            df_BLAST: Input pandas DataFrame containing BLAST.
            substrate_cutoff: Substrate cutoff processed by this function.
            n_permutations: Number of permutations used by this function.
            lfc_cutoffs: Lfc cutoffs processed by this function.
            kinase_to_peptides: Kinase to peptides processed by this function.
            peptide_order: Peptide order processed by this function.
        
        Returns:
            object: Calculated KPEA.
        """
        if substrate_cutoff is None:
            substrate_cutoff = self.kpea_substrate_cutoff
        if n_permutations is None:
            n_permutations = self.n_permutations
        if lfc_cutoffs is None:
            lfc_cutoffs = self.kpea_lfc_cutoffs
        scoring_cutoffs, selected_cutoff = self._resolve_kpea_scoring_cutoffs(lfc_cutoffs)

        if kinase_to_peptides is None or peptide_order is None:
            df_pooled = self._collapse_duplicate_peptides_for_uka(df_pooled)
            kinase_to_peptides, _, peptide_order = self._map_kinases_to_peptides(
                df_pooled=df_pooled,
                df_ptm=df_ptm,
                df_BLAST=df_BLAST,
                include_mapping_rows=False,
            )

        empty_cols = [
            "Kinase",
            "Kinase_Name",
            "NumSubstrates",
            "MeanSubstrate",
            "MeanPeptideStatistic",
            "MedianPeptideStatistic",
            "KinaseStatistic",
            "KinaseChange",
            "Direction",
            "Direction_PeptideMean",
            "KRSA_MeanZ",
            "KRSA_AbsMeanZ",
            "Z_Score",
            "p_value",
            "FDR",
            "NegLog10EmpiricalP",
            "KPEA_AbsDominantZ",
            "KPEA_CutoffMode",
            "KPEA_SelectedCutoff",
            "Significant_ZScore",
            "Significant_EmpiricalPValue",
            "Significant_FDR",
            "Significant_SelectedMethod",
            "SelectedForReport",
            "Significant",
        ]
        if not kinase_to_peptides:
            return pd.DataFrame(columns=empty_cols)

        peptide_change_lookup = dict(zip(df_pooled["ID"], df_pooled["peptide_change"]))
        peptide_stat_lookup = dict(zip(df_pooled["ID"], df_pooled["peptide_statistic"]))

        mapped_background_ids = {
            row["peptide_id"]
            for peptide_rows in kinase_to_peptides.values()
            for row in peptide_rows
        }
        peptide_ids = [pid for pid in peptide_order if pid in mapped_background_ids]
        if not peptide_ids:
            return pd.DataFrame(columns=empty_cols)

        peptide_changes = np.array(
            [float(peptide_change_lookup.get(pid, 0.0)) for pid in peptide_ids],
            dtype=float,
        )

        kinase_ids = sorted(
            kinase
            for kinase, peptides in kinase_to_peptides.items()
            if len({row["peptide_id"] for row in peptides}) >= substrate_cutoff
        )
        if not kinase_ids:
            return pd.DataFrame(columns=empty_cols)

        peptide_idx = {pid: idx for idx, pid in enumerate(peptide_ids)}
        n_peptides = len(peptide_ids)
        n_kinases = len(kinase_ids)

        substrate_masks = np.zeros((n_kinases, n_peptides), dtype=bool)
        substrate_id_map = {}
        for kinase_idx, kinase in enumerate(kinase_ids):
            substrate_ids = sorted({row["peptide_id"] for row in kinase_to_peptides[kinase]})
            substrate_id_map[kinase] = substrate_ids
            for substrate_id in substrate_ids:
                pep_idx = peptide_idx.get(substrate_id)
                if pep_idx is not None:
                    substrate_masks[kinase_idx, pep_idx] = True

        substrate_mask_matrix = substrate_masks.T.astype(np.int16, copy=False)
        rng = np.random.default_rng()
        hit_source = np.abs(peptide_changes)
        peptide_stats = np.array(
            [float(peptide_stat_lookup.get(pid, np.nan)) for pid in peptide_ids],
            dtype=np.float64,
        )

        observed_z_sum = np.zeros(n_kinases, dtype=np.float64)
        permuted_z_sum = np.zeros((n_permutations, n_kinases), dtype=np.float64)
        n_scored_cutoffs = 0
        for cutoff in scoring_cutoffs:
            hit_indices = np.flatnonzero(hit_source >= cutoff)
            if hit_indices.size == 0:
                self._dprint(f"     Cutoff {cutoff}: skipped (0 hits)")
                continue

            observed_counts, null_counts, _ = self._sample_null_counts(
                hit_indices=hit_indices,
                n_background_peptides=n_peptides,
                n_permutations=n_permutations,
                substrate_mask_matrix=substrate_mask_matrix,
                rng=rng,
            )

            mean_null = null_counts.mean(axis=0).astype(np.float64, copy=False)
            std_null = null_counts.std(axis=0, ddof=1).astype(np.float64, copy=False)
            observed_z_sum += _counts_to_z_numba_1d(
                observed_counts.astype(np.float64, copy=False),
                mean_null,
                std_null,
                float(self.kpea_z_cap),
            )
            _accumulate_counts_to_z_sum_numba_2d(
                null_counts,
                mean_null,
                std_null,
                float(self.kpea_z_cap),
                permuted_z_sum,
            )
            n_scored_cutoffs += 1

        if n_scored_cutoffs == 0:
            return pd.DataFrame(columns=empty_cols)

        mean_z = observed_z_sum / float(n_scored_cutoffs)
        perm_mean_z = permuted_z_sum / float(n_scored_cutoffs)
        empirical_p = (
            np.sum(np.abs(perm_mean_z) >= np.abs(mean_z[np.newaxis, :]), axis=0) + 1.0
        ) / (n_permutations + 1.0)
        _, fdr, _, _ = multipletests(empirical_p, method="fdr_bh")
        neg_log10_empirical_p = -np.log10(
            np.maximum(empirical_p, 1.0 / (n_permutations + 1.0))
        )
        abs_mean_z = np.abs(mean_z)
        representation = np.where(
            mean_z > 0,
            "overrepresented",
            np.where(mean_z < 0, "underrepresented", "none"),
        )
        num_substrates, mean_stats, median_stats, mean_changes = _summarize_kinase_arrays_numba(
            substrate_masks,
            peptide_stats,
            peptide_changes.astype(np.float64, copy=False),
        )

        rows = []
        for kinase_idx, kinase in enumerate(kinase_ids):
            mean_peptide_statistic = float(mean_stats[kinase_idx])
            median_peptide_statistic = float(median_stats[kinase_idx])
            kinase_statistic = mean_peptide_statistic
            kinase_change = float(mean_changes[kinase_idx])
            if pd.isna(kinase_change):
                direction_peptide_mean = "none"
            elif kinase_change > 0:
                direction_peptide_mean = "up"
            elif kinase_change < 0:
                direction_peptide_mean = "down"
            else:
                direction_peptide_mean = "none"

            rows.append(
                {
                    "Kinase": kinase,
                    "Kinase_Name": "",
                    "NumSubstrates": int(num_substrates[kinase_idx]),
                    "MeanSubstrate": mean_peptide_statistic,
                    "MeanPeptideStatistic": mean_peptide_statistic,
                    "MedianPeptideStatistic": median_peptide_statistic,
                    "KinaseStatistic": kinase_statistic,
                    "KinaseChange": kinase_change,
                    "Direction": str(representation[kinase_idx]),
                    "Direction_PeptideMean": direction_peptide_mean,
                    "KRSA_MeanZ": float(mean_z[kinase_idx]),
                    "KRSA_AbsMeanZ": float(abs_mean_z[kinase_idx]),
                    "Z_Score": float(mean_z[kinase_idx]),
                    "p_value": float(empirical_p[kinase_idx]),
                    "FDR": float(fdr[kinase_idx]),
                    "NegLog10EmpiricalP": float(neg_log10_empirical_p[kinase_idx]),
                    "KPEA_AbsDominantZ": float(abs_mean_z[kinase_idx]),
                    "KPEA_CutoffMode": self.kpea_cutoff_mode,
                    "KPEA_SelectedCutoff": float(selected_cutoff),
                }
            )

        df_kpea = pd.DataFrame(rows)
        if df_kpea.empty:
            return pd.DataFrame(columns=empty_cols)

        kinase_name_map = self._resolve_kinase_names_uniprot(df_kpea["Kinase"].tolist())
        df_kpea["Kinase_Name"] = df_kpea["Kinase"].map(kinase_name_map)
        df_kpea = self._annotate_significance_columns(df_kpea)
        df_kpea = df_kpea.sort_values(
            [
                "KPEA_AbsDominantZ",
                "NegLog10EmpiricalP",
                "NumSubstrates",
                "KinaseStatistic",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        for col in empty_cols:
            if col not in df_kpea.columns:
                df_kpea[col] = np.nan
        return df_kpea[empty_cols]

    def _rank_kinase_results(self, df_uka_combined):
        """Rank kinase results.
        
        Args:
            df_uka_combined: Input pandas DataFrame containing UKA combined.
        
        Returns:
            object: Ranked kinase results.
        """
        if df_uka_combined is None or df_uka_combined.empty:
            return df_uka_combined

        rank_score_col = (
            "KRSA_AbsMeanZ" if "KRSA_AbsMeanZ" in df_uka_combined.columns else "KPEA_AbsDominantZ"
        )
        activity_col = (
            "KinaseStatistic"
            if "KinaseStatistic" in df_uka_combined.columns
            else "MeanPeptideStatistic"
            if "MeanPeptideStatistic" in df_uka_combined.columns
            else "MedianPeptideStatistic"
        )
        ranked = (
            df_uka_combined.assign(abs_stat=lambda df: df[activity_col].abs())
            .sort_values(
                [
                    "Significant",
                    rank_score_col,
                    "NegLog10EmpiricalP",
                    "NumSubstrates",
                    "abs_stat",
                ],
                ascending=[False, False, False, False, False],
            )
            .drop_duplicates(subset="Kinase", keep="first")
            .drop(columns="abs_stat")
            .reset_index(drop=True)
        )
        return ranked

    def _plot_volcano(self, df_uka, output_path=None, control=None, condition=None):
        """Plot volcano.
        
        Args:
            df_uka: UKA result DataFrame processed by this helper.
            output_path: Path to the output.
            control: Control processed by this function.
            condition: Condition processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if df_uka is None or df_uka.empty:
            self._dprint("     No kinase data found for volcano plot.")
            return None

        save_path = (
            str(Path(output_path) / f"UKA_volcano_plot_{control}_{condition}.png")
            if output_path is not None
            else None
        )
        if self.volcano_plot_GUI:
            if tk is None:
                raise RuntimeError(
                    "Tkinter GUI plotting is not available in this environment. "
                    "Set volcano_plot_GUI=False for headless execution."
                )
            root = tk.Tk()
            VolcanoPlot(
                root=root,
                debugging_print=self.debugging_print,
                data=df_uka,
                y_axis="z_score",
                z_threshold=self.kpea_zscore_threshold,
                x_axis_col=self.uka_visualization_column,
                x_label=self.uka_visualization_label,
                save_path=save_path,
                gui=True,
            )
            root.mainloop()
        else:
            VolcanoPlot(
                debugging_print=self.debugging_print,
                data=df_uka,
                y_axis="z_score",
                z_threshold=self.kpea_zscore_threshold,
                x_axis_col=self.uka_visualization_column,
                x_label=self.uka_visualization_label,
                save_path=save_path,
                gui=False,
            )
        return None

    def run_kinase_analysis(self, peptide_statistics, control=None, condition=None):
        """Run kinase analysis.
        
        Args:
            peptide_statistics: Peptide statistics processed by this function.
            control: Control processed by this function.
            condition: Condition processed by this function.
        
        Returns:
            object: Run result for kinase analysis.
        """
        print("=====================================================================================")
        print("=======================   Starting Kinase Analysis Stage   ==========================")
        print("=====================================================================================\n")

        if isinstance(peptide_statistics, dict):
            df_peptides = peptide_statistics.get("peptide_statistics")
            control = peptide_statistics.get("control_condition", control)
            condition = peptide_statistics.get("condition", condition)
        else:
            df_peptides = peptide_statistics

        if df_peptides is None or df_peptides.empty:
            raise ValueError("Kinase analysis requires peptide statistics as input.")
        if "Type" not in df_peptides.columns:
            raise ValueError("Peptide statistics must contain a 'Type' column.")

        print("[1]  Loading peptide enrichment and BLAST data...")
        df_BLAST = self._get_blast_data()
        peptide_to_proteins = self._get_peptide_to_proteins(df_BLAST)
        print("     BLAST data loaded successfully.")
        print("\n=====================================================================================\n")

        print("[2]  Loading PTM data...")
        df_ptm_stk, df_ptm_ptk = self._get_ptm_data()
        ptk_lookup = self._get_substrate_to_kinase_lookup(df_ptm_ptk, cache_key="PTK")
        stk_lookup = self._get_substrate_to_kinase_lookup(df_ptm_stk, cache_key="STK")
        print("     PTM data loaded successfully.")
        print("\n=====================================================================================\n")

        df_ptk_pooled = df_peptides[df_peptides["Type"] == "PTK"].copy()
        df_stk_pooled = df_peptides[df_peptides["Type"] == "STK"].copy()

        if df_ptk_pooled.empty and df_stk_pooled.empty:
            raise ValueError("No PTK or STK peptide statistics found for kinase analysis.")

        print("[3]  Calculating KPEA-based kinase scores...")
        df_mapping_frames = []

        def _analyze_branch(array_type, df_branch_pooled, df_ptm_branch, substrate_lookup):
            """Return analyze branch.
            
            Args:
                array_type: Array type processed by this function.
                df_branch_pooled: Input pandas DataFrame containing branch pooled.
                df_ptm_branch: Input pandas DataFrame containing PTM branch.
                substrate_lookup: Substrate lookup processed by this function.
            
            Returns:
                tuple: Analyze branch.
            """
            df_branch_pooled_uka = self._collapse_duplicate_peptides_for_uka(df_branch_pooled)
            if df_branch_pooled_uka.empty:
                return array_type, pd.DataFrame(), pd.DataFrame()

            kinase_to_peptides, df_mapping, peptide_order = self._map_kinases_to_peptides(
                df_pooled=df_branch_pooled_uka,
                df_ptm=df_ptm_branch,
                df_BLAST=df_BLAST,
                include_mapping_rows=self.debugging_print,
                substrate_to_kinase_entries=substrate_lookup,
                peptide_to_proteins=peptide_to_proteins,
            )
            df_uka = self._calculate_KPEA(
                df_pooled=df_branch_pooled_uka,
                df_ptm=df_ptm_branch,
                df_BLAST=df_BLAST,
                kinase_to_peptides=kinase_to_peptides,
                peptide_order=peptide_order,
            )
            if not df_uka.empty:
                df_uka["Type"] = array_type
            if self.debugging_print and not df_mapping.empty:
                df_mapping["Type"] = array_type
            return array_type, df_uka, df_mapping

        branch_specs = [
            ("PTK", df_ptk_pooled, df_ptm_ptk, ptk_lookup),
            ("STK", df_stk_pooled, df_ptm_stk, stk_lookup),
        ]
        branch_results = []
        active_specs = [spec for spec in branch_specs if not spec[1].empty]

        if len(active_specs) > 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(_analyze_branch, *spec)
                    for spec in active_specs
                ]
                for future in futures:
                    branch_results.append(future.result())
        else:
            for spec in active_specs:
                branch_results.append(_analyze_branch(*spec))

        branch_result_map = {array_type: (df_uka, df_mapping) for array_type, df_uka, df_mapping in branch_results}
        df_uka_ptk, df_mapping_ptk = branch_result_map.get(
            "PTK",
            (pd.DataFrame(), pd.DataFrame()),
        )
        df_uka_stk, df_mapping_stk = branch_result_map.get(
            "STK",
            (pd.DataFrame(), pd.DataFrame()),
        )
        for df_mapping_branch in (df_mapping_ptk, df_mapping_stk):
            if self.debugging_print and not df_mapping_branch.empty:
                df_mapping_frames.append(df_mapping_branch)

        df_uka_combined_raw = pd.concat(
            [df for df in (df_uka_ptk, df_uka_stk) if not df.empty],
            ignore_index=True,
        )
        df_uka_combined_raw = self._annotate_significance_columns(df_uka_combined_raw)

        if self.debugging_print and df_mapping_frames:
            df_mapping_combined = pd.concat(df_mapping_frames, ignore_index=True)
            self._write_debug_csv(
                df_mapping_combined,
                f"kinase_peptide_mapping_{control}_{condition}",
            )
            kinase_peptide_groups = (
                df_mapping_combined.drop_duplicates(
                    subset=["Kinase_UniprotID", "Peptide_ID"]
                )
                .groupby("Kinase_UniprotID")["Peptide_ID"]
                .apply(lambda x: ",".join(sorted(x.unique())))
                .reset_index()
                .rename(
                    columns={
                        "Kinase_UniprotID": "Kinase",
                        "Peptide_ID": "Peptides",
                    }
                )
            )
            self._write_debug_csv(
                kinase_peptide_groups,
                f"kinase_peptide_list_{control}_{condition}",
            )

        df_all_kinases = self._rank_kinase_results(df_uka_combined_raw)
        df_significant_kinases = df_all_kinases[
            df_all_kinases["Significant"].fillna(False)
        ].copy()

        if self.volcano_plot:
            print("[4]  Plotting kinase volcano plot...")
            self._plot_volcano(
                df_uka=df_all_kinases,
                output_path=self.volcano_plot_output,
                control=control,
                condition=condition,
            )

        self.all_kinases = df_all_kinases
        self.significant_kinases = df_significant_kinases
        self.all_kinases_raw = df_uka_combined_raw

        print("\n=====================================================================================")
        print("====================   Kinase Analysis Stage Completed   ============================")
        print("=====================================================================================\n")

        return {
            "all_kinases": df_all_kinases,
            "significant_kinases": df_significant_kinases,
            "all_kinases_raw": df_uka_combined_raw,
            "all_kinase_ids": df_all_kinases["Kinase"].tolist()
            if not df_all_kinases.empty
            else [],
            "significant_kinase_ids": df_significant_kinases["Kinase"].tolist()
            if not df_significant_kinases.empty
            else [],
        }
