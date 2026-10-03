"""Upstream kinase activity analysis (UKA/KPEA).

KinaseActivityAnalysis maps the PamChip peptides to proteins and candidate
kinases (enrichment table, BLAST results, PTM and kinase-substrate resources),
applies the evidence filters, scores every kinase with the hypergeometric
Z-score of the KPEA against the chosen null population and exports the kinase
tables and the sets used for the overlap diagrams. Its output feeds
kx_pathway_enrichment_analysis and the CSV export in kx_pipeline_tools.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import threading
import warnings

import numpy as np
import pandas as pd
import requests
from numba import njit

from config.analysis_modules import UPSTREAM_KINASE_ANALYSIS_DEFAULTS
from kx_peptide_ids import normalize_peptide_id_column, normalize_peptide_ids
from kx_plot_results import VennDiagramPlot


# Column schema of every kinase result frame `_calculate_KPEA` produces -- including
# its five empty returns. Module level so the degenerate case in
# `run_kinase_analysis` can carry the SAME schema without duplicating the list.
KPEA_RESULT_COLUMNS = [
    "Kinase",
    "Kinase_Name",
    "Kinase_Name_Resolved",
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
    "KPEA_AbsDominantZ",
    "KPEA_CutoffMode",
    "KPEA_SelectedCutoff",
    "KPEA_BackgroundUniverse",
    "KPEA_ChipNumber",
    "KPEA_PopulationSize",
    "KPEA_BackgroundSubstrates",
    "KPEA_ScoredCutoffs",
    "KPEA_NumHits",
    "KPEA_ObservedHits",
    "Significant_ZScore",
    "Significant_SelectedMethod",
    "SelectedForReport",
    "Significant",
]


@njit(cache=False, nogil=True)
def _counts_to_z_numba_1d(counts, mean_null, std_null, z_cap):
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
def _summarize_kinase_arrays_numba(substrate_masks, peptide_stats, peptide_changes):
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


def _kpea_null_moments(substrate_counts, n_hits, n_peptides):
    """Return the EXACT hypergeometric mean and sd of the null substrate count.

    Drawing ``n_hits`` of ``n_peptides`` peptides without replacement makes a
    kinase's substrate count hypergeometric, so its first two moments are known in
    closed form. Using them instead of Monte-Carlo estimates from the drawn null
    makes the observed z-score -- and therefore ``Z_Score`` and
    ``Significant_ZScore`` -- fully deterministic: there is no sampling anywhere in
    the KPEA scoring path, so no seed and no permutation count exist to configure.

    Degenerate cases are returned as sd = 0 and handled by the callers' existing
    ``std_null <= 1e-12`` branch: a kinase with no substrates or with every peptide
    as a substrate has no null spread, and neither does a cutoff at which every
    peptide is a hit.

    Args:
        substrate_counts: Per-kinase number of substrate peptides (population K).
        n_hits: Number of drawn peptides (sample size N).
        n_peptides: Background population size (M).

    Returns:
        tuple: (mean_null, std_null), both float64 arrays over kinases.
    """
    proportion = np.asarray(substrate_counts, dtype=np.float64) / float(n_peptides)
    mean_null = float(n_hits) * proportion
    if n_peptides <= 1:
        return mean_null, np.zeros_like(mean_null)
    variance = (
        float(n_hits)
        * proportion
        * (1.0 - proportion)
        * (float(n_peptides) - float(n_hits))
        / (float(n_peptides) - 1.0)
    )
    return mean_null, np.sqrt(np.maximum(variance, 0.0))


def _accumulate_kpea_observed_z(
    substrate_mask_matrix,
    hit_index_sets,
    z_cap,
    background_counts=None,
):
    """Accumulate the observed z-scores across the KPEA scoring cutoffs.

    Fully deterministic: the null is described by its exact hypergeometric moments
    (see :func:`_kpea_null_moments`), so no sampling is involved anywhere. There is
    no seed and no permutation count to configure, and repeated runs are
    bit-identical.

    Args:
        substrate_mask_matrix: (n_peptides, n_kinases) 0/1 substrate membership over
            the MEASURED peptides. Observed counts always come from here.
        hit_index_sets: One hit-index array per SCORED cutoff (empty cutoffs are
            dropped by the caller).
        z_cap: Symmetric clamp on individual z-scores.
        background_counts: Optional ``(n_background, substrate_counts)`` describing
            the population the null draws from. Defaults to the measured peptides
            themselves; pass chip-wide counts to reproduce KRSA, which samples from
            its whole mapping table regardless of whether a peptide was measured.

    Returns:
        np.ndarray: ``observed_z_sum`` over the cutoffs. Divide by
            ``len(hit_index_sets)`` for the averaged score.
    """
    n_peptides, n_kinases = substrate_mask_matrix.shape
    if background_counts is None:
        n_background = n_peptides
        substrate_counts = substrate_mask_matrix.sum(axis=0).astype(np.int64, copy=False)
    else:
        n_background, substrate_counts = background_counts
        n_background = int(n_background)
        substrate_counts = np.asarray(substrate_counts, dtype=np.int64)

    observed_z_sum = np.zeros(n_kinases, dtype=np.float64)
    for hit_indices in hit_index_sets:
        n_hits = int(len(hit_indices))
        mean_null, std_null = _kpea_null_moments(substrate_counts, n_hits, n_background)
        observed_counts = (
            substrate_mask_matrix[hit_indices].sum(axis=0).astype(np.float64)
        )
        observed_z_sum += _counts_to_z_numba_1d(
            observed_counts,
            np.ascontiguousarray(mean_null, dtype=np.float64),
            np.ascontiguousarray(std_null, dtype=np.float64),
            float(z_cap),
        )
    return observed_z_sum


class KinaseActivityAnalysis:
    """Stage 2 of the UKA/KPEA workflow: kinase mapping and scoring."""

    DEFAULT_VERIFIED_EVIDENCE_LEVELS = tuple(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["default_verified_evidence_levels"]
    )
    DEFAULT_BLAST_RESULTS_RELATIVE_PATH = Path(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["default_blast_results_relative_path"]
    )
    ALLOWED_KPEA_CUTOFF_MODES = tuple(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["allowed_kpea_cutoff_modes"]
    )
    ALLOWED_KPEA_BACKGROUND_UNIVERSES = tuple(
        UPSTREAM_KINASE_ANALYSIS_DEFAULTS["allowed_kpea_background_universes"]
    )
    # Column(s) used to de-duplicate the ranked kinase table. PTK and STK are
    # independent arrays with disjoint peptide sets, so the same UniProt accession
    # scored on both is TWO independent measurements, not a duplicate: the key is
    # (kinase id, array). Measured with the committed defaults (BLAST_threshold 80,
    # require_known_ptm_site true) 16 accessions are scoreable on both arrays
    # (O14733, P35790, P36507, P42345, P45985, P46734, P49760, Q02750, Q05655,
    # Q07912, Q13163, Q13627, Q14680, Q96KB5, Q9NR20, Q9Y463); on the committed
    # benchmarking run 6 of them carry a result in each contrast.
    RANK_DEDUP_SUBSET = ["Kinase", "Type"]

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
        path_output_peptide_statistic=None,
        debugging_print=True,
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
        kpea_background_universe=UPSTREAM_KINASE_ANALYSIS_DEFAULTS[
            "default_kpea_background_universe"
        ],
        input_stk_ptm_path=UPSTREAM_KINASE_ANALYSIS_DEFAULTS["input_stk_ptm_path"],
        input_ptk_ptm_path=UPSTREAM_KINASE_ANALYSIS_DEFAULTS["input_ptk_ptm_path"],
        kpea_chip_numbers=None,
    ):
        """Store the mapping resources, evidence filters and KPEA settings.
        
        Args:
            debugging_print: Whether to print additional debug information.
            kpea_background_universe: Population the null draws from. ``"chip"``
                (default) uses the full mapping reference -- KRSA's convention --
                while ``"measured"`` restricts it to measured, QC-passed peptides.
            kpea_chip_numbers: Optional ``{"PTK": "86412", "STK": "87102"}`` pinning
                the PamChip VERSION whose peptides form the chip-wide null
                population. Only relevant for
                ``kpea_background_universe="chip"``. ``None`` (default) infers it
                from the measured peptides; pass
                ``chip_number_from_layout_path(<array layout file>)`` to take it
                from the authoritative source instead.
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
        self._kinase_name_cache_lock = threading.Lock()
        self._kinase_name_unresolved = set()
        self._chip_background_cache = {}
        # Provenance of the chip-wide null population, written by
        # `_resolve_chip_number` and read by `_calculate_KPEA` for the export
        # columns. ONE entry per array type: the chip version is resolved once
        # and never re-inferred per comparison.
        self._chip_resolution = {}

        resolved_background_universe = str(kpea_background_universe).lower()
        if resolved_background_universe not in self.ALLOWED_KPEA_BACKGROUND_UNIVERSES:
            raise ValueError(
                f"Unknown kpea_background_universe '{kpea_background_universe}'. "
                "Use 'chip' or 'measured'."
            )
        resolved_cutoff_mode = str(kpea_cutoff_mode).lower()
        if resolved_cutoff_mode not in self.ALLOWED_KPEA_CUTOFF_MODES:
            raise ValueError(
                f"Unknown kpea_cutoff_mode '{kpea_cutoff_mode}'. "
                "Use 'average' or 'primary'."
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
        self.kpea_background_universe = resolved_background_universe
        self.kpea_chip_numbers = {
            str(key).strip().upper(): str(value).strip()
            for key, value in (kpea_chip_numbers or {}).items()
            if value not in (None, "")
        }


    def _dprint(self, *args, **kwargs):
        """Print a message only when debug logging is enabled."""
        if self.debugging_print:
            print(*args, **kwargs)

    @staticmethod
    def _candidate_data_dirs():
        script_root_data_dir = Path(__file__).resolve().parent.parent / "data"
        configured_paths = [
            Path(path_value).expanduser()
            for path_value in UPSTREAM_KINASE_ANALYSIS_DEFAULTS["candidate_data_paths"]
        ]
        return [script_root_data_dir, *configured_paths]

    @classmethod
    def _resolve_data_path(cls, relative_path):
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
        token = "NA" if value is None else str(value).strip()
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token).strip("_")
        return token or "NA"

    @staticmethod
    def _comparison_label(condition, payload):
        control = None
        condition_label = condition
        if isinstance(payload, dict):
            control = payload.get("control_condition")
            # `condition` may be the comparison key ("mock_vs_pSHDAg"); the
            # payload carries the plain test label to put in the legend.
            condition_label = payload.get("condition", condition)
        if control is None:
            return str(condition_label)
        return f"{control} vs {condition_label}"

    @staticmethod
    def _extract_kinase_items(payload):
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
        """Plot overlaps of significant kinases across condition comparisons."""
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
        if self.path_output_peptide_statistic is None:
            return
        output_path = Path(f"{self.path_output_peptide_statistic}_{suffix}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)


    def _zscore_significance_mask(self, df_kpea):
        """Return z-score significance mask."""
        return df_kpea["Z_Score"].abs().fillna(0.0) >= self.kpea_zscore_threshold

    def _kpea_score_stat_label(self):
        if self.kpea_cutoff_mode == "primary":
            return f"KRSA |Z| at cutoff {self.kpea_primary_lfc_cutoff:g}"
        return "KRSA |mean Z|"

    def _kpea_score_basis_label(self):
        if self.kpea_cutoff_mode == "primary":
            return f"primary cutoff {self.kpea_primary_lfc_cutoff:g}"
        return f"mean across cutoffs {list(self.kpea_lfc_cutoffs)}"

    def _resolve_kpea_scoring_cutoffs(self, lfc_cutoffs):
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
        return f"{self._kpea_score_stat_label()} >= {self.kpea_zscore_threshold}"

    def _annotate_significance_columns(self, df_kpea):
        if df_kpea is None or df_kpea.empty:
            return df_kpea

        df_kpea = df_kpea.copy()
        zscore_sig = self._zscore_significance_mask(df_kpea)

        df_kpea["Significant_ZScore"] = zscore_sig
        significant_selected_method = zscore_sig
        df_kpea["Significant_SelectedMethod"] = significant_selected_method
        df_kpea["SelectedForReport"] = significant_selected_method
        df_kpea["Significant"] = df_kpea["Significant_SelectedMethod"]
        return df_kpea

    @staticmethod
    def _split_delimited_values(raw_value):
        if pd.isna(raw_value):
            return set()
        return {
            str(token).strip()
            for token in re.split(r"[;,]", str(raw_value))
            if str(token).strip()
        }

    @staticmethod
    def _is_known_ptm_site(site):
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
        normalized = str(site).strip().upper()
        return normalized in {"Y", "S/T"}

    @classmethod
    def _extract_informative_ptm_site_tokens(cls, raw_site):
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
        return bool(cls._extract_informative_ptm_site_tokens(raw_site))

    def _site_matches_array_type(self, raw_site, array_type):
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
        if pd.isna(level):
            return ""
        return str(level).strip().lower()

    def _verified_interaction_mask(self, df_ptm):
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
        if pd.isna(peptide_id) or pd.isna(sequence):
            return set()

        # The position pair does not have to end the id: 20 chip peptides carry
        # a mutation suffix (``AKT1_309_321_C310S``) and one carries a space
        # (``H2B1B_ 27_40``). With a ``$`` anchor they yielded an empty candidate
        # set, and under the default ``require_known_ptm_site: true`` every
        # positioned kinase assignment of these peptides was silently discarded
        # (PTK M 180->182, STK M 151->152, K shifts for 22 PTK and 23 STK kinases).
        # The leftmost match is the peptide window; a later pair inside a
        # mutation token (``PDCD1_221_229_FS219_220KK``) must not win.
        match = re.search(r"_\s*(\d+)_(\d+)(?:_|$)", str(peptide_id))
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
        # Peptide ids, not protein accessions, despite the column name -- and
        # they carry the same stray separators as the layouts and the enrichment
        # table (H2B1B_ 27_40 here, VGFR1_1320_1332_C1320K/C1321K in the PTK
        # layouts). Normalise before the merge below, which keys on them.
        # See kx_peptide_ids.
        df_BLAST["source_peptide_id"] = normalize_peptide_ids(
            df_BLAST["source_peptide_id"]
        )

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
        return normalize_peptide_id_column(
            pd.read_csv(self.path_file_enrichment_peptides)
        )

    def _get_peptide_enrichment(self):
        if self._peptide_enrichment_cache is None:
            self._peptide_enrichment_cache = self._load_peptide_enrichment()
        return self._peptide_enrichment_cache

    def _get_blast_data(self):
        if self._blast_cache is None:
            self._blast_cache = self._load_BLAST(
                df_peptide_enrichment=self._get_peptide_enrichment()
            )
        return self._blast_cache

    def _get_peptide_to_proteins(self, df_BLAST=None):
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
        if self._ptm_cache is None:
            self._ptm_cache = self._load_and_merge_ptm_data()
        return self._ptm_cache

    def _kinase_name_state(self):
        """Return the shared ``(lock, unresolved-set)`` pair for name lookups.

        Both are created in ``__init__`` and shared by reference across the
        per-comparison workers (``kx_pipeline_tools._prime_kinase_worker``).
        They are (re)installed lazily here so that instances built without
        ``__init__`` -- e.g. in a reproduction script -- still work.

        Returns:
            tuple: ``(threading.Lock, set)``.
        """
        lock = self.__dict__.get("_kinase_name_cache_lock")
        if lock is None:
            lock = threading.Lock()
            self._kinase_name_cache_lock = lock
        unresolved = self.__dict__.get("_kinase_name_unresolved")
        if unresolved is None:
            unresolved = set()
            self._kinase_name_unresolved = unresolved
        return lock, unresolved

    def _resolve_kinase_names_uniprot(self, uniprot_ids, batch_size=100):
        """Resolve kinase names UniProt.

        The cache and its lock are shared by every worker of a run, so the
        lookup runs exactly once per accession instead of once per comparison,
        and every comparison sees the same name. Accessions whose lookup did
        not succeed still map to the accession itself (the display column stays
        populated) but are recorded in ``self._kinase_name_unresolved`` and
        reported on stdout, so an outage is distinguishable from a UniProt
        entry that carries no gene symbol.

        Returns:
            dict: ``{accession: display name}`` for every requested accession.
        """
        id_to_name = {}
        unique_ids = list(dict.fromkeys(str(uid).strip() for uid in uniprot_ids if str(uid).strip()))
        if not unique_ids:
            return id_to_name

        lock, unresolved = self._kinase_name_state()
        with lock:
            # Recomputed INSIDE the lock: a worker that waited here while
            # another one performed the lookup must not repeat it.
            uncached_ids = [
                uid for uid in unique_ids if uid not in self._kinase_name_cache
            ]
            if not uncached_ids:
                return {uid: self._kinase_name_cache[uid] for uid in unique_ids}

            self._dprint(
                f"     Resolving {len(uncached_ids)} kinase names via UniProt API..."
            )
            failed_ids = []
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
                    # Unconditional print: with the shipped default
                    # debugging_print=false a _dprint here makes the outage
                    # invisible, and the accession alias below is then
                    # indistinguishable from a real UniProt name.
                    print(
                        "     WARNING: UniProt name lookup FAILED for batch "
                        f"{i // batch_size + 1} ({len(batch)} accessions): {exc}"
                    )
                    failed_ids.extend(batch)

            # Accessions the API answered for but did not return a row for
            # (obsolete or unknown entries) are unresolved as well.
            missing_ids = [
                uid
                for uid in uncached_ids
                if uid not in self._kinase_name_cache and uid not in failed_ids
            ]
            if missing_ids:
                shown = ", ".join(missing_ids[:10])
                print(
                    "     WARNING: UniProt returned no entry for "
                    f"{len(missing_ids)} accession(s): {shown}"
                    + (" ..." if len(missing_ids) > 10 else "")
                )

            for uid in failed_ids + missing_ids:
                self._kinase_name_cache[uid] = uid
                unresolved.add(uid)

            return {uid: self._kinase_name_cache[uid] for uid in unique_ids}

    def _build_substrate_to_kinase_lookup(self, df_ptm):
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

        unparsed_ids = [
            peptide_id
            for peptide_id in peptide_order
            if re.search(r"_\s*\d+_\d+(?:_|$)", str(peptide_id)) is None
        ]
        if unparsed_ids:
            self._dprint(
                f"     UKA: {len(unparsed_ids)} peptide id(s) carry no "
                "start_end position pair, so their candidate sites stay empty "
                f"(e.g. {', '.join(sorted(map(str, unparsed_ids))[:3])})."
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

    @staticmethod
    def _kpea_array_type(df_pooled):
        """Return "PTK"/"STK" for one branch's peptide table, or None."""
        if "Type" in df_pooled.columns:
            for value in df_pooled["Type"].dropna().astype(str):
                token = value.strip().upper()
                if token in ("PTK", "STK"):
                    return token
        return None

    # PamChip layout files are named "<barcodes> <chip number> Array Layout.txt";
    # the chip number is the only place the array VERSION is recorded.
    CHIP_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{5})(?!\d)")

    # Minimum share of the measured peptides a chip_num token must cover, and the
    # minimum margin over the runner-up, before the token is accepted as the array
    # version. The rule is strict because guessing the wrong version is worse
    # than falling back to the unfiltered reference, the old behaviour.
    CHIP_NUMBER_MIN_COVERAGE = 0.90
    CHIP_NUMBER_MIN_MARGIN = 0.01

    @classmethod
    def chip_number_from_layout_path(cls, layout_path):
        """Parse the PamChip version out of an array-layout file name.

        ``".../641102408_641102409 86412 Array Layout.txt"`` -> ``"86412"``. The
        barcodes are 9 digits and the chip number 5, so a bounded 5-digit match
        picks the chip number without matching a barcode.

        Args:
            layout_path: Path or string pointing at the array-layout file.

        Returns:
            str | None: The chip number, or ``None`` when the name carries none.
        """
        if layout_path is None:
            return None
        candidates = cls.CHIP_NUMBER_PATTERN.findall(Path(layout_path).name)
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _chip_num_tokens(value):
        """Split an ``enrichment_peptides`` ``chip_num`` cell into chip versions."""
        return {
            token
            for token in str(value).strip().split("_")
            if token and token.lower() != "nan"
        }

    def _chip_resolution_state(self):
        """Return the per-array record of how the chip version was resolved.

        Created in ``__init__``; recreated lazily so an instance built without it
        (a reproduction script, a subclass) still works.

        Returns:
            dict: ``{array_type: {"chip_number": str | None, "source": str}}``.
        """
        state = self.__dict__.get("_chip_resolution")
        if state is None:
            state = {}
            self._chip_resolution = state
        return state

    def _warn_null_population(self, message):
        """Report a fallback that changes the null population, debug flag or not.

        ``_dprint`` is gated on ``debugging_print``, which the pipeline default
        leaves off, so every fallback below used to be invisible in a production
        run although it moves M, K and therefore every z-score. Printed for the
        run log AND raised as a ``RuntimeWarning`` so a test can catch it.

        Args:
            message: The text to report.
        """
        print(f"     WARNING: {message}")
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _resolve_chip_number(self, array_type, measured_ids, chip_reference):
        """Decide WHICH PamChip version the chip-wide null population describes.

        ``enrichment_peptides.csv`` is the pooled PamGene catalogue across several
        chip generations (``chip_num`` says which peptide sits on which). Filtering
        it on ``family`` alone therefore yields the union over all generations, not
        the peptides printed on THIS array -- for the benchmarking STK run that is
        181 instead of 144 peptides, so a quarter of the null population is
        physically absent from the chip.

        Resolution order:

        1. ``kpea_chip_numbers`` if the caller set one (the pipeline can fill it
           from the array-layout file name via :meth:`chip_number_from_layout_path`,
           which is the authoritative source);
        2. otherwise infer it from the measured peptides: the chip version whose
           peptide set covers them. Validated on the benchmarking data, where it
           recovers 86412 (PTK, 99.5% vs 97.4% for the runner-up) and 87102 (STK,
           100% vs 74.3%);
        3. otherwise ``None`` -- the reference stays UNFILTERED, i.e. the null
           population becomes the union over every PamChip generation in
           ``enrichment_peptides.csv``. That is the old behaviour; it is now
           reported by ``_warn_null_population`` (stdout + ``RuntimeWarning``)
           and written into the ``KPEA_BackgroundUniverse`` /
           ``KPEA_ChipNumber`` export columns, so a run that took it is
           distinguishable from a clean one.

        The answer is resolved ONCE per array type and reused afterwards (see
        ``_chip_resolution_state``). Re-inferring it per comparison let the QC
        losses of a single comparison decide its own null population; the
        pipeline additionally pins it for the whole run
        (``kx_pipeline_tools._resolve_run_chip_numbers``).

        Args:
            array_type: ``"PTK"`` or ``"STK"``.
            measured_ids: Peptide ids measured on this array.
            chip_reference: ``enrichment_peptides`` rows for ``array_type``.

        Returns:
            str | None: Chip version to filter on, or ``None`` to keep every row.
        """
        state = self._chip_resolution_state()

        pinned = self.kpea_chip_numbers.get(array_type)
        if pinned:
            state[array_type] = {"chip_number": str(pinned), "source": "pinned"}
            return str(pinned)

        # One resolution per array type. Without this every comparison re-infers
        # the chip version from its own QC-passed peptides, so two comparisons of
        # the same run can be scored against different M and K.
        if array_type in state:
            return state[array_type]["chip_number"]

        measured_ids = {str(pid) for pid in measured_ids}
        if not measured_ids or "chip_num" not in chip_reference.columns:
            state[array_type] = {"chip_number": None, "source": "no_reference"}
            return None

        tokens = set()
        for value in chip_reference["chip_num"]:
            tokens |= self._chip_num_tokens(value)
        if not tokens:
            state[array_type] = {"chip_number": None, "source": "no_reference"}
            return None

        coverage = []
        for token in sorted(tokens):
            on_chip = {
                str(peptide_id)
                for peptide_id, value in zip(
                    chip_reference["ID"], chip_reference["chip_num"]
                )
                if token in self._chip_num_tokens(value)
            }
            coverage.append((len(measured_ids & on_chip) / len(measured_ids), token))
        coverage.sort(reverse=True)

        best_share, best_token = coverage[0]
        runner_up = coverage[1][0] if len(coverage) > 1 else 0.0
        if (
            best_share < self.CHIP_NUMBER_MIN_COVERAGE
            or best_share - runner_up < self.CHIP_NUMBER_MIN_MARGIN
        ):
            self._warn_null_population(
                f"KPEA: could not identify the {array_type} chip version "
                f"(best '{best_token}' covers {best_share:.1%} of the measured "
                f"peptides, runner-up {runner_up:.1%}); the null population "
                "becomes the UNION over all PamChip generations in "
                "enrichment_peptides.csv, which changes M, K and every z-score. "
                "Pin the version with kpea_chip_numbers to avoid this."
            )
            state[array_type] = {"chip_number": None, "source": "unresolved"}
            return None
        self._dprint(
            f"     KPEA: {array_type} chip version resolved to '{best_token}' "
            f"({best_share:.1%} of the measured peptides)."
        )
        state[array_type] = {"chip_number": best_token, "source": "inferred"}
        return best_token

    def _build_chip_background(self, array_type, measured_ids=(), chip_number=None):
        """Map EVERY peptide printed on THIS array, ignoring measurement and QC.

        Args:
            array_type: ``"PTK"`` or ``"STK"``.
            measured_ids: Peptide ids measured on this array; used only to infer
                the chip version when it was not pinned explicitly.

        Returns:
            tuple | None: ``(n_background, {kinase: n_substrates})``, or ``None``
                when the chip-wide reference cannot be built.
        """
        enrichment = self._get_peptide_enrichment()
        if enrichment is None or "family" not in enrichment.columns:
            return None
        chip = enrichment[
            enrichment["family"].astype(str).str.strip().str.upper() == array_type
        ]
        if chip.empty:
            return None

        # Restrict the catalogue to the peptides printed on this chip VERSION.
        # Without this the null population is the union over all PamChip
        # generations in the reference file, which inflates M (and, unevenly
        # across kinases, K) with peptides that could never have been measured.
        if chip_number is None:
            chip_number = self._resolve_chip_number(array_type, measured_ids, chip)
        if chip_number is not None:
            on_chip = chip["chip_num"].apply(
                lambda value: chip_number in self._chip_num_tokens(value)
            )
            chip_filtered = chip[on_chip]
            if chip_filtered.empty:
                self._warn_null_population(
                    f"KPEA: no {array_type} peptides carry chip number "
                    f"'{chip_number}'; the null population becomes the union over "
                    "all PamChip generations."
                )
                self._chip_resolution_state()[array_type] = {
                    "chip_number": None,
                    "source": "empty_after_filter",
                }
            else:
                chip = chip_filtered

        # A df_pooled-shaped frame so the real mapping rules apply unchanged. The
        # quantitative columns are placeholders: only the peptide -> kinase links
        # matter here, never the values.
        df_chip = pd.DataFrame(
            {
                "ID": chip["ID"].astype(str),
                "Sequence": chip.get("Sequence", ""),
                "GeneName": chip.get("PepProtein_UniprotName", ""),
                "UniprotAccession": chip.get("PepProtein_UniprotID", ""),
                "mean_control": 0.0,
                "mean_treatment": 0.0,
                "Type": array_type,
            }
        )
        df_ptm_stk, df_ptm_ptk = self._get_ptm_data()
        df_ptm = df_ptm_ptk if array_type == "PTK" else df_ptm_stk
        kinase_to_peptides, _, _ = self._map_kinases_to_peptides(
            df_pooled=df_chip,
            df_ptm=df_ptm,
            df_BLAST=self._get_blast_data(),
            include_mapping_rows=False,
            substrate_to_kinase_entries=self._get_substrate_to_kinase_lookup(
                df_ptm, cache_key=array_type
            ),
            peptide_to_proteins=self._get_peptide_to_proteins(self._get_blast_data()),
        )
        if not kinase_to_peptides:
            return None
        per_kinase = {
            kinase: len({row["peptide_id"] for row in rows})
            for kinase, rows in kinase_to_peptides.items()
        }
        mapped = {
            row["peptide_id"] for rows in kinase_to_peptides.values() for row in rows
        }
        return len(mapped), per_kinase

    def mappable_kinases(self, array_type):
        """Return the kinases the PTM data maps to enough peptides of this chip.

        Counted over EVERY peptide printed on the array's chip version, measured or
        not (the mapping `_build_chip_background` builds for the KPEA null), with
        the floor `_calculate_KPEA` applies to the measured substrates: at least
        ``kpea_substrate_cutoff`` of them. These are the kinases this analysis can
        score on the array; the pathway stage uses them as its
        ``'mappable_kinases'`` background.

        Args:
            array_type: ``"PTK"`` or ``"STK"``.

        Returns:
            dict | None: ``{kinase: substrate peptides on the chip}``, or ``None``
                when the chip-wide reference cannot be built.
        """
        background = self._build_chip_background(
            array_type, chip_number=self.kpea_chip_numbers.get(array_type)
        )
        if background is None:
            return None
        cutoff = max(int(self.kpea_substrate_cutoff), 1)
        return {
            kinase: n_substrates
            for kinase, n_substrates in sorted(background[1].items())
            if n_substrates >= cutoff
        }

    def _chip_background_counts(
        self, kinase_ids, df_pooled, measured_counts, n_measured
    ):
        """Return the chip-wide ``(n_background, substrate_counts)`` for the null.

        This is the sampling population of the hypergeometric null. The observed
        counts always come from the measured, QC-passed peptides; this method
        decides what M and K are counted over.

        KRSA draws its null hit sets with ``sample(map$Substrates, n_hits)`` from
        the whole chip mapping table, including peptides a run never measured or
        that its QC removed. The map is a static, chip-specific package dataset,
        so M and K are fixed per kinase, Z is comparable across runs, and QC
        removing different peptides from run to run does not make K volatile.
        The cost is that the null draws from a larger space than the observed
        count could have come from, so Z is not a strict standardisation of the
        observed count; ``kpea_background_universe = "measured"`` restricts the
        population to what was measured instead. pyKinaXe defaults to ``chip``
        to be comparable with KRSA.

        Returns ``None`` to fall back to the measured population when the
        chip-wide reference is unavailable or inconsistent.
        """
        array_type = self._kpea_array_type(df_pooled)
        if array_type is None:
            return None
        measured_ids = frozenset(df_pooled["ID"].astype(str))
        # Resolve the chip version FIRST and key the cache on it. Keying on the
        # measured set made every comparison build its OWN null population, so two
        # comparisons of one run could be scored against different M and K.
        # `_resolve_chip_number` locks its answer per array type, and the pipeline
        # pins it for the whole run (kx_pipeline_tools._resolve_run_chip_numbers).
        enrichment = self._get_peptide_enrichment()
        if enrichment is not None and "family" in enrichment.columns:
            chip_reference = enrichment[
                enrichment["family"].astype(str).str.strip().str.upper() == array_type
            ]
        else:
            chip_reference = pd.DataFrame(columns=["ID", "chip_num"])
        chip_number = self._resolve_chip_number(
            array_type, measured_ids, chip_reference
        )
        cache_key = (array_type, chip_number)
        if cache_key not in self._chip_background_cache:
            self._chip_background_cache[cache_key] = self._build_chip_background(
                array_type, measured_ids=measured_ids, chip_number=chip_number
            )
        cached = self._chip_background_cache[cache_key]
        if cached is None:
            self._warn_null_population(
                f"KPEA: no chip-wide background for {array_type}; falling back to "
                "the measured peptides -- the z-scores of this comparison are NOT "
                "on the same scale as a chip-wide run."
            )
            return None

        n_background, per_kinase = cached
        counts = np.array(
            [int(per_kinase.get(kinase, 0)) for kinase in kinase_ids], dtype=np.int64
        )
        measured_counts = np.asarray(measured_counts, dtype=np.int64)
        # The chip population must contain the measured one. If it does not, the two
        # references disagree (e.g. peptide ids that do not normalise the same way)
        # and the moments would be nonsense -- fall back rather than guess.
        if n_background < int(n_measured) or np.any(counts < measured_counts):
            self._warn_null_population(
                f"KPEA: chip-wide background for {array_type} does not contain the "
                f"measured population (M_chip={int(n_background)}, "
                f"M_measured={int(n_measured)}); falling back to the measured "
                "peptides."
            )
            return None
        return n_background, counts

    def _calculate_KPEA(
        self,
        df_pooled,
        df_ptm=None,
        df_BLAST=None,
        substrate_cutoff=None,
        lfc_cutoffs=None,
        kinase_to_peptides=None,
        peptide_order=None,
    ):
        if substrate_cutoff is None:
            substrate_cutoff = self.kpea_substrate_cutoff
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

        empty_cols = list(KPEA_RESULT_COLUMNS)
        if not kinase_to_peptides:
            return pd.DataFrame(columns=empty_cols)

        peptide_change_lookup = dict(zip(df_pooled["ID"], df_pooled["peptide_change"]))
        peptide_stat_lookup = dict(zip(df_pooled["ID"], df_pooled["peptide_statistic"]))

        # Background universe for the hypergeometric null. The measured, mapped
        # and QC-passed peptides always define the observed counts; whether they
        # also define the null population depends on kpea_background_universe:
        #   chip     (default) M and K are counted over the full mapping
        #                      reference, as krsa() does (it samples its null
        #                      hit sets from the whole mapping table); see
        #                      _chip_background_counts().
        #   measured           M and K are counted over the measured peptides.
        #                      The internally consistent null, but it moves Z
        #                      away from KRSA kinase by kinase.
        # The agreement with KRSA under both settings is documented in
        # validation_paper/compare_pyKinaXe_results_with_KRSA/NOTE.txt.
        mapped_background_ids = {
            row["peptide_id"]
            for peptide_rows in kinase_to_peptides.values()
            for row in peptide_rows
        }
        # A peptide without a usable peptide_change is EXCLUDED from the population,
        # not carried as a silent non-hit. `hit_source = |peptide_change|` and
        # `NaN >= cutoff` is False, so such a peptide can never enter a hit set --
        # keeping it would enlarge M (and, unevenly across kinases, K) with draws the
        # observed statistic could never have produced. That is the same mismatch
        # `_chip_background_counts` documents for the chip population, except there
        # it is a KRSA-compatible trade-off and here it would be an
        # accident. NaN peptide_change is reachable whenever one group is entirely
        # missing for a peptide (np.nanmean of an all-NaN group), which
        # `log2_slope_mode='krsa_na'` makes common.
        peptide_ids = [
            pid
            for pid in peptide_order
            if pid in mapped_background_ids
            and not pd.isna(peptide_change_lookup.get(pid, np.nan))
        ]
        if not peptide_ids:
            return pd.DataFrame(columns=empty_cols)

        # np.nan, not 0.0: a missing change means UNKNOWN, while 0.0 would claim the
        # peptide was measured and did not move. The filter above makes this default
        # unreachable on the normal path; it stays correct if `peptide_order` is ever
        # supplied by a caller instead of derived from `df_pooled`.
        peptide_changes = np.array(
            [float(peptide_change_lookup.get(pid, np.nan)) for pid in peptide_ids],
            dtype=float,
        )

        peptide_idx = {pid: idx for idx, pid in enumerate(peptide_ids)}
        n_peptides = len(peptide_ids)

        # Build the masks over EVERY candidate kinase first, then apply the substrate
        # cutoff to the mask row sums. Counting on `kinase_to_peptides` instead would
        # count substrates that the filter above has just removed from the
        # population, so a kinase could clear the cutoff on peptides that are not in
        # M -- and its reported NumSubstrates (which comes from the mask) would
        # disagree with the number that admitted it.
        candidate_kinases = sorted(kinase_to_peptides)
        candidate_masks = np.zeros((len(candidate_kinases), n_peptides), dtype=bool)
        for kinase_idx, kinase in enumerate(candidate_kinases):
            for substrate_id in {
                row["peptide_id"] for row in kinase_to_peptides[kinase]
            }:
                pep_idx = peptide_idx.get(substrate_id)
                if pep_idx is not None:
                    candidate_masks[kinase_idx, pep_idx] = True

        # Floor of 1: before the population filter every mapped kinase had at least
        # one substrate in it by construction, so a substrate_cutoff of 0 (the
        # kinase-FAMILY default) could never admit an empty kinase. Keep that
        # invariant -- an all-zero mask row carries no information and would only
        # reach the degenerate std_null == 0 branch.
        effective_cutoff = max(int(substrate_cutoff), 1)
        keep_kinase = candidate_masks.sum(axis=1) >= effective_cutoff
        kinase_ids = [
            kinase for kinase, keep in zip(candidate_kinases, keep_kinase) if keep
        ]
        if not kinase_ids:
            return pd.DataFrame(columns=empty_cols)

        substrate_masks = candidate_masks[keep_kinase]
        substrate_mask_matrix = substrate_masks.T.astype(np.int16, copy=False)
        hit_source = np.abs(peptide_changes)
        peptide_stats = np.array(
            [float(peptide_stat_lookup.get(pid, np.nan)) for pid in peptide_ids],
            dtype=np.float64,
        )

        hit_index_sets = []
        scored_cutoffs = []
        for cutoff in scoring_cutoffs:
            hit_indices = np.flatnonzero(hit_source >= cutoff)
            if hit_indices.size == 0:
                self._dprint(f"     Cutoff {cutoff}: skipped (0 hits)")
                continue
            hit_index_sets.append(hit_indices)
            scored_cutoffs.append(float(cutoff))

        n_scored_cutoffs = len(hit_index_sets)
        if n_scored_cutoffs == 0:
            return pd.DataFrame(columns=empty_cols)

        measured_counts = substrate_mask_matrix.sum(axis=0).astype(np.int64)
        background_counts = None
        if self.kpea_background_universe == "chip":
            background_counts = self._chip_background_counts(
                kinase_ids=kinase_ids,
                df_pooled=df_pooled,
                measured_counts=measured_counts,
                n_measured=n_peptides,
            )

        # Record how the chip version was resolved, so the exported row can be
        # audited. Under `kpea_background_universe = "chip"` the observed count is
        # standardised against (M_chip, K_chip) while `NumSubstrates` reports
        # K_measured, so Z cannot be reconstructed from the exported columns
        # alone; the numbers that produced it are written out as well.
        # `_chip_background_counts` can return None; the effective population is
        # then the measured one, and the label has to say so instead of repeating
        # the configured value. The value range of KPEA_BackgroundUniverse is
        # extended by 'chip_all_generations' (the chip version was not
        # identifiable, so the null is the union over every PamChip generation)
        # and 'measured_fallback' (the chip population was unusable and the null
        # silently became the measured one, which is a different statistic).
        chip_resolution = {}
        if self.kpea_background_universe == "chip":
            chip_resolution = self._chip_resolution_state().get(
                self._kpea_array_type(df_pooled), {}
            )
        chip_number_label = "n/a"
        if background_counts is None:
            effective_background_universe = (
                "measured_fallback"
                if self.kpea_background_universe == "chip"
                else "measured"
            )
            population_size = int(n_peptides)
            background_substrates = measured_counts
        else:
            effective_background_universe = (
                "chip_all_generations"
                if chip_resolution.get("chip_number") is None
                else "chip"
            )
            chip_number_label = str(chip_resolution.get("chip_number") or "unresolved")
            population_size = int(background_counts[0])
            background_substrates = np.asarray(background_counts[1], dtype=np.int64)
        print(
            f"     KPEA null population: universe={effective_background_universe}, "
            f"chip={chip_number_label}, M={population_size}, N={int(n_peptides)}"
        )

        # N per cutoff and the observed counts per kinase.
        # `_accumulate_kpea_observed_z` forms the same sums internally but only
        # returns the accumulated z; recomputing them here leaves that helper's
        # signature unchanged.
        observed_per_cutoff = [
            substrate_mask_matrix[hit_indices].sum(axis=0).astype(np.int64)
            for hit_indices in hit_index_sets
        ]
        scored_cutoffs_label = "|".join(f"{cutoff:g}" for cutoff in scored_cutoffs)
        num_hits_label = "|".join(
            str(int(len(hit_indices))) for hit_indices in hit_index_sets
        )

        observed_z_sum = _accumulate_kpea_observed_z(
            substrate_mask_matrix=substrate_mask_matrix,
            hit_index_sets=hit_index_sets,
            z_cap=float(self.kpea_z_cap),
            background_counts=background_counts,
        )

        mean_z = observed_z_sum / float(n_scored_cutoffs)
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
                    "Kinase_Name_Resolved": True,
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
                    "KPEA_AbsDominantZ": float(abs_mean_z[kinase_idx]),
                    "KPEA_CutoffMode": self.kpea_cutoff_mode,
                    "KPEA_SelectedCutoff": float(selected_cutoff),
                    "KPEA_BackgroundUniverse": effective_background_universe,
                    "KPEA_ChipNumber": chip_number_label,
                    "KPEA_PopulationSize": population_size,
                    "KPEA_BackgroundSubstrates": int(
                        background_substrates[kinase_idx]
                    ),
                    "KPEA_ScoredCutoffs": scored_cutoffs_label,
                    "KPEA_NumHits": num_hits_label,
                    "KPEA_ObservedHits": "|".join(
                        str(int(observed[kinase_idx]))
                        for observed in observed_per_cutoff
                    ),
                }
            )

        df_kpea = pd.DataFrame(rows)
        if df_kpea.empty:
            return pd.DataFrame(columns=empty_cols)

        kinase_name_map = self._resolve_kinase_names_uniprot(df_kpea["Kinase"].tolist())
        df_kpea["Kinase_Name"] = df_kpea["Kinase"].map(kinase_name_map)
        _, unresolved_names = self._kinase_name_state()
        df_kpea["Kinase_Name_Resolved"] = ~(
            df_kpea["Kinase"].astype(str).str.strip().isin(unresolved_names)
        )
        df_kpea = self._annotate_significance_columns(df_kpea)
        df_kpea = df_kpea.sort_values(
            [
                "KPEA_AbsDominantZ",
                "NumSubstrates",
                "KinaseStatistic",
            ],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        for col in empty_cols:
            if col not in df_kpea.columns:
                df_kpea[col] = np.nan
        return df_kpea[empty_cols]

    def _rank_kinase_results(self, df_uka_combined):
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
                    "NumSubstrates",
                    "abs_stat",
                ],
                ascending=[False, False, False, False],
            )
            .drop_duplicates(subset=self.RANK_DEDUP_SUBSET, keep="first")
            .drop(columns="abs_stat")
            .reset_index(drop=True)
        )
        return ranked

    def run_kinase_analysis(self, peptide_statistics, control=None, condition=None):
        print("\n" + "=" * 80)
        print("Starting Kinase Analysis Stage...")
        print("=" * 80)

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

        # Both branches can legitimately return a 0-row frame (e.g. no peptide reaches
        # the lowest `kpea_lfc_cutoffs` entry): `_calculate_KPEA` then returns
        # `pd.DataFrame(columns=KPEA_RESULT_COLUMNS)`, which the filter below drops
        # because a 0-row frame is `.empty`. `pd.concat([])` would raise
        # "No objects to concatenate" and abort the whole run, while
        # `_annotate_significance_columns` and `_rank_kinase_results` both accept an
        # empty frame -- so carry the schema explicitly and report 0 kinases.
        uka_frames = [df for df in (df_uka_ptk, df_uka_stk) if not df.empty]
        if uka_frames:
            df_uka_combined_raw = pd.concat(uka_frames, ignore_index=True)
        else:
            arrays_label = ", ".join(spec[0] for spec in active_specs) or "none"
            print(
                f"     KPEA: no kinase could be scored for {control} vs {condition} "
                f"on {arrays_label} -- returning an empty kinase table "
                f"(0 rows, full schema). Check kpea_lfc_cutoffs "
                f"{tuple(self.kpea_lfc_cutoffs)} and kpea_substrate_cutoff "
                f"{self.kpea_substrate_cutoff} if this is unexpected."
            )
            df_uka_combined_raw = pd.DataFrame(columns=[*KPEA_RESULT_COLUMNS, "Type"])
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

        self.all_kinases = df_all_kinases
        self.significant_kinases = df_significant_kinases
        self.all_kinases_raw = df_uka_combined_raw

        print("\n" + "=" * 80)
        print("Kinase Analysis Stage Completed")
        print("=" * 80 + "\n")

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
