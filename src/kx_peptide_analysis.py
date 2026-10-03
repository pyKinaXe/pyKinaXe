"""Peptide-level statistics of the PTK and STK chip outputs.

PeptideStatistics rearranges the image-processing output into per-condition
matrices, estimates the exposure slope of every spot, applies the peptide QC
(qc_mode), optionally normalises every array (array_normalization), computes
the log2 fold change per peptide (peptide_change, with the per-chip batch
correction) and writes the peptide tables, waterfall plots and heatmaps. Its
output is the input of kx_upstream_kinase_analysis.
"""

from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor

# Kept for the limma path, which currently feeds reported-only columns -- see the
# module docstring. Retained for a future version that acts on peptide-level
# significance again.
from inmoose.limma import eBayes, lmFit, squeezeVar
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

from config.analysis_modules import PEPTIDE_ANALYSIS_DEFAULTS
from kx_peptide_ids import normalize_peptide_id_column
from kx_plot_results import HeatmapPlot_Peptides, WaterfallPlot


class DegenerateLimmaModerationError(RuntimeError):
    """Raised when limma moderation collapses to a single shared variance."""

    def __init__(self, *, df_prior, var_prior):
        """Record the collapsed prior and build the error message."""
        self.df_prior = df_prior
        self.var_prior = var_prior
        super().__init__(
            "Limma moderation collapsed to a common posterior variance "
            f"(df_prior={df_prior}, var_prior={var_prior})."
        )


class PeptideStatistics:
    """Stage 1 of the UKA/KPEA workflow: peptide preprocessing and statistics."""

    ALLOWED_LOG2_SLOPE_MODES = tuple(
        PEPTIDE_ANALYSIS_DEFAULTS["allowed_log2_slope_modes"]
    )
    ALLOWED_BATCH_CORRECTION_METHODS = tuple(
        PEPTIDE_ANALYSIS_DEFAULTS["allowed_batch_correction_methods"]
    )
    ALLOWED_ARRAY_NORMALIZATION_METHODS = tuple(
        PEPTIDE_ANALYSIS_DEFAULTS["allowed_array_normalization_methods"]
    )
    ARRAY_NORMALIZATION_MIN_PEPTIDES = int(
        PEPTIDE_ANALYSIS_DEFAULTS["array_normalization_min_peptides"]
    )
    ALLOWED_QC_MODES = tuple(PEPTIDE_ANALYSIS_DEFAULTS["allowed_qc_modes"])
    QC_KRSA = "QC_KRSA"
    ALLOWED_CONTRAST_MODES = tuple(
        PEPTIDE_ANALYSIS_DEFAULTS["allowed_contrast_modes"]
    )
    CONTRAST_MODE_ALL_PAIRS = "all_pairs"
    CONTRAST_MODE_SINGLE_CONTROL = "single_control"
    # Written by the enricher from the c/t prefix; see
    # kx_data_enricher._assign_logical_chip_conditions.
    CONDITION_ROLE_COLUMN = "Condition Role"
    CONDITION_ROLE_CONTROL = "control"
    # Fallback only, for enrichment tables built without a Condition Role
    # column: keywords that mark a Test Condition label as a control. Matched
    # as WHOLE TOKENS of the lower-cased label (see
    # ``_matches_control_keyword``), never as substrings -- 'DMSOtolerant_clone'
    # and 'Vehiclepretreated_DrugA' are treatment arms, not controls.
    CONTROL_CONDITION_KEYWORDS = (
        "control",
        "ctrl",
        "untreated",
        "baseline",
        "mock",
        "dmso",
        "vehicle",
    )
    # Token separator for that match: every run of non-alphanumeric characters,
    # which covers '_', ' ', '-', '.' and anything else a label may use.
    CONTROL_KEYWORD_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

    def __init__(
        self,
        file_enrichment=None,
        df_ptk=None,
        df_stk=None,
        path_file_enrichment_peptides=PEPTIDE_ANALYSIS_DEFAULTS[
            "path_file_enrichment_peptides"
        ],
        waterfall_plot=False,
        waterfall_plot_output=None,
        heatmap_plot=False,
        heatmap_plot_output=None,
        path_output_peptide_statistic=None,
        log_output=None,
        use_limma=True,
        debugging_print=True,
        log2_slope_mode=PEPTIDE_ANALYSIS_DEFAULTS["default_log2_slope_mode"],
        batch_correction=PEPTIDE_ANALYSIS_DEFAULTS["default_batch_correction"],
        batch_correction_method=PEPTIDE_ANALYSIS_DEFAULTS[
            "default_batch_correction_method"
        ],
        batch_column=PEPTIDE_ANALYSIS_DEFAULTS["batch_column"],
        array_normalization=PEPTIDE_ANALYSIS_DEFAULTS["default_array_normalization"],
        array_normalization_method=PEPTIDE_ANALYSIS_DEFAULTS[
            "default_array_normalization_method"
        ],
        qc_mode=PEPTIDE_ANALYSIS_DEFAULTS["default_qc_mode"],
        qc_krsa_signal_threshold=PEPTIDE_ANALYSIS_DEFAULTS[
            "default_qc_krsa_signal_threshold"
        ],
        qc_krsa_r2_threshold=PEPTIDE_ANALYSIS_DEFAULTS[
            "default_qc_krsa_r2_threshold"
        ],
        contrast_mode=PEPTIDE_ANALYSIS_DEFAULTS["default_contrast_mode"],
        contrasts=None,
        waterfall_lfc_cutoffs=PEPTIDE_ANALYSIS_DEFAULTS[
            "default_waterfall_lfc_cutoffs"
        ],
        waterfall_cutoff_mode=PEPTIDE_ANALYSIS_DEFAULTS[
            "default_waterfall_cutoff_mode"
        ],
        waterfall_primary_lfc_cutoff=PEPTIDE_ANALYSIS_DEFAULTS[
            "default_waterfall_primary_lfc_cutoff"
        ],
    ):
        """Store the peptide-stage inputs, thresholds and output paths.
        
        Args:
            waterfall_plot: Whether to draw the ranked peptide log2 fold
                change (waterfall) plot.
            waterfall_plot_output: Directory the waterfall plots are written to.
            log_output: Directory that collects the run logs (the limma
                fallback reports). When None the logs are written next to
                ``path_output_peptide_statistic`` instead, as before.
            use_limma: Whether to run the limma moderated t-test. Its outputs
                (``p_value``, ``t_statistic``, ``s2_moderated``, ``df_moderated``
                and the ``*_zscore`` columns) are currently exported but read by
                nothing downstream; the flag is kept for a future version that
                uses peptide-level significance again. Turning it off falls back
                to row-wise t-tests and should leave every kinase, family and
                pathway result unchanged.
            debugging_print: Whether to print additional debug information.
            batch_correction: Enable per-chip batch-effect removal from the
                peptide log2 fold change (``True``/``False``, or a string mode
                name). ``True`` uses ``batch_correction_method``.
            batch_correction_method: How the batch effect is removed:
                ``'limma_block'`` (regress batch out of ``peptide_change`` and add
                it to the limma design) or ``'center'`` (adjust ``peptide_change``
                only). Ignored when ``batch_correction`` is disabled.
            batch_column: Enrichment column identifying the batch/chip
                (default ``'Barcode'``).
            array_normalization: Enable the per-array normalisation of
                ``slope_log2`` before the batch correction and the
                treatment-vs-control difference (``True``/``False``, or a
                method name, which both enables it and selects the method).
                ``True`` uses ``array_normalization_method``; ``False``,
                ``None`` and ``'none'``/``'off'`` leave the values as they are.
            array_normalization_method: How the arrays are normalised:
                ``'median'`` shifts every array so that its median over the
                analysed peptides measured on all arrays of its type equals the
                median of those array medians (see
                :meth:`_estimate_array_normalization`). Ignored when
                ``array_normalization`` is disabled.
            qc_mode: Which peptide QC to apply.
                ``'QC_KRSA'`` (default) reproduces KRSA's ``krsa_quick_filter``:
                signal at the longest exposure, R^2 of the linear
                signal-vs-exposure fit, reference-spot removal, all at the
                end-point cycle -- so the peptide set is comparable to KRSA's.
                ``'QC_BioNavigator'`` is the pair of array-specific filters
                pyKinaXe used before that mode existed (STK: pre-phosphorylated
                internal standard; PTK: slope-vs-cycle regression). The name
                records which mode it is; nothing in the code derives those two
                filters from BioNavigator's own QC.
            qc_krsa_signal_threshold: Minimum signal at the longest exposure
                (``QC_KRSA`` only).
            qc_krsa_r2_threshold: Minimum R^2 of the signal-vs-exposure fit
                (``QC_KRSA`` only).
            contrast_mode: Which condition pairs are compared.
                ``'all_pairs'`` (default) pairs EVERY control condition with
                EVERY test condition, so k_c controls and k_t tests give
                k_c * k_t comparisons. Control-vs-control and test-vs-test are
                never produced: the reference side of a comparison is always a
                control. ``'single_control'`` keeps the pre-2026-08-25
                behaviour -- the first control label found is the only
                reference and every other condition, a second control
                included, is treated as a test. With one control label both
                modes produce the same comparisons.
            contrasts: Optional EXPLICIT list of ``(reference, test)`` Test
                Condition pairs. When given it REPLACES the pairs
                ``contrast_mode`` would generate and is the only way to express
                a pairing / blocking criterion (tissue, subject, batch), which
                neither mode can represent -- ``all_pairs`` is a blind cross
                product of the control and test labels. Every name must occur
                in the annotation and every reference must be a control
                condition, otherwise the run aborts. ``None`` (default) keeps
                ``contrast_mode`` and therefore today's behaviour.
            waterfall_lfc_cutoffs: Log2 fold change cutoffs drawn in the
                waterfall plot; these are the KPEA cutoffs of the run, so the
                plot marks the peptides the enrichment step counts.
            waterfall_cutoff_mode: ``'average'`` draws every cutoff and treats
                the smallest one as the significance boundary, ``'primary'``
                draws and uses ``waterfall_primary_lfc_cutoff`` only.
            waterfall_primary_lfc_cutoff: The single cutoff used in
                ``'primary'`` mode.
        """
        self.file_enrichment = file_enrichment
        # Normalise the ids as they enter, not at the merge: they are compared
        # against the peptide-enrichment table, the BLAST table and the KRSA
        # mapping, and a stray space in an array-layout id (JAK1_ 1027_1039,
        # MK07_ 212_224 in the CDRL rat PTK layout) makes every one of those
        # lookups miss. See kx_peptide_ids for what that silently costs.
        self.df_ptk_input = normalize_peptide_id_column(df_ptk)
        self.df_stk_input = normalize_peptide_id_column(df_stk)

        self.path_file_enrichment_peptides = Path(path_file_enrichment_peptides)
        self.waterfall_plot = waterfall_plot
        self.waterfall_plot_output = (
            Path(waterfall_plot_output) if waterfall_plot_output is not None else None
        )
        self.heatmap_plot = heatmap_plot
        self.heatmap_plot_output = Path(heatmap_plot_output) if heatmap_plot_output is not None else None
        self.waterfall_lfc_cutoffs = tuple(waterfall_lfc_cutoffs)
        self.waterfall_cutoff_mode = waterfall_cutoff_mode
        self.waterfall_primary_lfc_cutoff = waterfall_primary_lfc_cutoff
        self.path_output_peptide_statistic = (
            Path(path_output_peptide_statistic)
            if path_output_peptide_statistic is not None
            else None
        )
        self.log_output = Path(log_output) if log_output is not None else None
        self.use_limma = use_limma
        self.debugging_print = debugging_print
        self.log2_slope_mode = str(log2_slope_mode).lower()

        if self.log2_slope_mode not in self.ALLOWED_LOG2_SLOPE_MODES:
            raise ValueError(
                f"Unknown log2_slope_mode '{log2_slope_mode}'. "
                "Use 'epsilon_floor', 'pamgene_zero', or 'krsa_na'."
            )

        # A bare True/False toggles correction with the default method; a string
        # ("limma_block"/"center") both enables it and selects the method, while
        # False/"none"/"off" disable it.
        batch_token = str(batch_correction).strip().lower()
        self.batch_correction = batch_token not in ("false", "none", "off", "0", "")
        if batch_token in self.ALLOWED_BATCH_CORRECTION_METHODS:
            resolved_batch_method = batch_token
        else:
            resolved_batch_method = str(batch_correction_method).strip().lower()
        if self.batch_correction and (
            resolved_batch_method not in self.ALLOWED_BATCH_CORRECTION_METHODS
        ):
            raise ValueError(
                f"Unknown batch_correction_method '{resolved_batch_method}'. "
                f"Use one of {list(self.ALLOWED_BATCH_CORRECTION_METHODS)}."
            )
        self.batch_correction_method = resolved_batch_method
        self.batch_column = str(batch_column).strip()
        # Like batch_correction, a bare True/False toggles the normalisation and a
        # method name both enables it and selects the method. Unlike there, an
        # unknown token is an error instead of silently meaning "on".
        normalization_token = str(array_normalization).strip().lower()
        resolved_normalization_method = (
            str(array_normalization_method).strip().lower()
        )
        if normalization_token in self.ALLOWED_ARRAY_NORMALIZATION_METHODS:
            self.array_normalization = True
            resolved_normalization_method = normalization_token
        elif normalization_token in ("true", "on", "yes", "1"):
            self.array_normalization = True
        elif normalization_token in ("false", "off", "no", "0", "none", ""):
            self.array_normalization = False
        else:
            raise ValueError(
                f"Unknown array_normalization '{array_normalization}'. Use "
                f"True/False or one of {list(self.ALLOWED_ARRAY_NORMALIZATION_METHODS)}."
            )
        if self.array_normalization and (
            resolved_normalization_method not in self.ALLOWED_ARRAY_NORMALIZATION_METHODS
        ):
            raise ValueError(
                f"Unknown array_normalization_method '{array_normalization_method}'. "
                f"Use one of {list(self.ALLOWED_ARRAY_NORMALIZATION_METHODS)}."
            )
        self.array_normalization_method = resolved_normalization_method
        # Filled by run_peptide_statistics: one row per array with its offset.
        self.array_normalization_offsets = None
        resolved_qc_mode = str(qc_mode).strip()
        matched = [mode for mode in self.ALLOWED_QC_MODES
                   if mode.lower() == resolved_qc_mode.lower()]
        if not matched:
            raise ValueError(
                f"Unknown qc_mode '{qc_mode}'. "
                f"Use one of {list(self.ALLOWED_QC_MODES)}."
            )
        self.qc_mode = matched[0]
        self.qc_krsa_signal_threshold = float(qc_krsa_signal_threshold)
        self.qc_krsa_r2_threshold = float(qc_krsa_r2_threshold)
        resolved_contrast_mode = str(contrast_mode).strip().lower()
        if resolved_contrast_mode not in self.ALLOWED_CONTRAST_MODES:
            raise ValueError(
                f"Unknown contrast_mode '{contrast_mode}'. "
                f"Use one of {list(self.ALLOWED_CONTRAST_MODES)}."
            )
        self.contrast_mode = resolved_contrast_mode
        self.explicit_contrasts = self._normalise_explicit_contrasts(contrasts)

        self.peptide_row_name = PEPTIDE_ANALYSIS_DEFAULTS["peptide_row_name"]

    @staticmethod
    def _normalise_explicit_contrasts(contrasts):
        """Validate and normalise an explicit contrast list.

        Args:
            contrasts: ``None``, or an iterable of ``(reference, test)`` Test
                Condition pairs (a YAML ``[[ctrl, test], ...]`` arrives as a
                list of lists).

        Returns:
            list: ``[(reference, test), ...]`` with both members stripped
            strings and duplicates dropped, or ``None`` when no explicit list
            was given (an empty list counts as none, so an empty config value
            falls back to ``contrast_mode`` instead of aborting the run).
        """
        if contrasts is None:
            return None
        if isinstance(contrasts, str):
            raise ValueError(
                "contrasts must be an iterable of (reference, test) pairs, "
                f"not a string: {contrasts!r}."
            )
        normalised = []
        for entry in contrasts:
            if isinstance(entry, str):
                raise ValueError(
                    "Every entry of contrasts must be a (reference, test) "
                    f"pair, not a string: {entry!r}."
                )
            try:
                pair = tuple(entry)
            except TypeError:
                raise ValueError(
                    "Every entry of contrasts must be a (reference, test) "
                    f"pair; got {entry!r}."
                ) from None
            if len(pair) != 2:
                raise ValueError(
                    "Every entry of contrasts must have exactly 2 members "
                    f"(reference, test); got {entry!r}."
                )
            reference, test = str(pair[0]).strip(), str(pair[1]).strip()
            if not reference or not test:
                raise ValueError(
                    f"contrasts contains an empty condition name: {entry!r}."
                )
            if reference == test:
                raise ValueError(
                    f"contrasts pairs a condition with itself: {entry!r}."
                )
            normalised.append((reference, test))
        if not normalised:
            return None
        return list(dict.fromkeys(normalised))

    def _dprint(self, *args, **kwargs):
        """Print a message only when debug logging is enabled."""
        if self.debugging_print:
            print(*args, **kwargs)

    @staticmethod
    def _extract_scalar(value):
        array_value = np.asarray(value)
        if array_value.ndim == 0:
            return float(array_value)
        return float(array_value.flat[0])

    @staticmethod
    def _normalize_replicate_value(value):
        if pd.isna(value):
            return None

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            text_value = str(value).strip()
            return text_value or None

        if not np.isfinite(numeric):
            return None
        if numeric.is_integer():
            return int(numeric)
        return numeric

    @classmethod
    def _build_sample_key(
        cls,
        sample_name,
        biological_replicate=None,
        technical_replicate=None,
    ):
        base_name = str(sample_name).strip()
        suffix_parts = []

        bio_rep = cls._normalize_replicate_value(biological_replicate)
        tech_rep = cls._normalize_replicate_value(technical_replicate)

        if bio_rep is not None:
            suffix_parts.append(f"BR{bio_rep}")
        if tech_rep is not None:
            suffix_parts.append(f"TR{tech_rep}")

        if not suffix_parts:
            return base_name
        return f"{base_name}__{'__'.join(suffix_parts)}"

    @staticmethod
    def _build_requested_group_matrix(df_rearanged, requested_samples):
        n_rows = len(df_rearanged)
        if not requested_samples:
            return np.empty((n_rows, 0), dtype=float)

        matrix_columns = []
        for sample_name in requested_samples:
            if sample_name in df_rearanged.columns:
                values = pd.to_numeric(
                    df_rearanged[sample_name],
                    errors="coerce",
                ).to_numpy(dtype=float)
            else:
                values = np.full(n_rows, np.nan, dtype=float)
            matrix_columns.append(values)

        return np.column_stack(matrix_columns)

    def _format_group_sample_label(self, sample_key, group_prefix):
        sample_info = self.sample_to_biorep.get(sample_key, {})
        label_parts = []

        bio_rep = self._normalize_replicate_value(sample_info.get("bio_rep"))
        tech_rep = self._normalize_replicate_value(sample_info.get("tech_rep"))
        condition_label = sample_info.get("condition") or group_prefix

        if bio_rep is not None:
            label_parts.append(f"BR #{bio_rep}")
        if tech_rep is not None:
            label_parts.append(f"TR #{tech_rep}")

        if label_parts:
            return f"{condition_label} [{' / '.join(label_parts)}]"

        sample_name = sample_info.get("sample_name")
        if sample_name:
            return f"{condition_label} [{sample_name}]"
        return str(condition_label)

    def _write_debug_csv(self, df, suffix):
        if self.path_output_peptide_statistic is None:
            return

        output_path = Path(f"{self.path_output_peptide_statistic}_{suffix}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def _log_file_path(self, name, extension):
        """Resolve where one log file is written.

        Logs land in ``log_output`` when the pipeline provides that directory.
        Standalone use without it keeps the historical location: next to the
        peptide-statistics output, using it as a filename prefix.
        
        Args:
            name: Base name of the log file, without extension.
            extension: File extension including the leading dot.
        
        Returns:
            Path | None: Target path, or None when no output location is configured.
        """
        if self.log_output is not None:
            return self.log_output / f"{name}{extension}"
        if self.path_output_peptide_statistic is not None:
            return Path(f"{self.path_output_peptide_statistic}_{name}{extension}")
        return None

    def _write_log_csv(self, df, name):
        """Write one log table into the run's log directory.
        
        Args:
            name: Base name of the log file.
        """
        output_path = self._log_file_path(name, ".csv")
        if output_path is None:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def _write_log_text(self, lines, name):
        """Write one log summary into the run's log directory.
        
        Args:
            name: Base name of the log file.
        """
        output_path = self._log_file_path(name, ".txt")
        if output_path is None:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(str(line) for line in lines).rstrip()
        output_path.write_text(f"{text}\n" if text else "", encoding="utf-8")

    def _write_debug_text(self, lines, suffix):
        if self.path_output_peptide_statistic is None:
            return

        output_path = Path(f"{self.path_output_peptide_statistic}_{suffix}.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(str(line) for line in lines).rstrip()
        output_path.write_text(f"{text}\n" if text else "", encoding="utf-8")

    @staticmethod
    def _calculate_rowwise_sd(matrix):
        matrix = np.asarray(matrix, dtype=float)
        row_sd = np.full(matrix.shape[0], np.nan, dtype=float)
        valid_rows = np.sum(~np.isnan(matrix), axis=1) >= 2
        if np.any(valid_rows):
            row_sd[valid_rows] = np.nanstd(matrix[valid_rows], axis=1, ddof=1)
        return row_sd

    @staticmethod
    def _run_rowwise_t_tests(control_values, treatment_values, eligible_mask):
        n_rows = control_values.shape[0]
        t_stats = np.full(n_rows, np.nan, dtype=float)
        p_values = np.full(n_rows, np.nan, dtype=float)

        if not np.any(eligible_mask):
            return t_stats, p_values

        ttest_result = ttest_ind(
            treatment_values[eligible_mask],
            control_values[eligible_mask],
            axis=1,
            equal_var=True,
            nan_policy="omit",
        )
        t_stats[eligible_mask] = np.asarray(ttest_result.statistic, dtype=float)
        p_values[eligible_mask] = np.asarray(ttest_result.pvalue, dtype=float)
        return t_stats, p_values

    def _report_limma_exclusions(
        self,
        *,
        df_rearanged,
        chip_label,
        complete_case_mask,
        n_control_values,
        n_treatment_values,
    ):
        excluded_mask = ~complete_case_mask
        if not np.any(excluded_mask):
            return

        total_rows = len(df_rearanged)
        complete_rows = int(np.sum(complete_case_mask))
        excluded_rows = int(np.sum(excluded_mask))

        diagnostics_df = df_rearanged.loc[excluded_mask, ["ID"]].copy()
        diagnostics_df["n_control_values"] = n_control_values[excluded_mask]
        diagnostics_df["n_treatment_values"] = n_treatment_values[excluded_mask]
        diagnostics_df["n_total_values"] = (
            diagnostics_df["n_control_values"] + diagnostics_df["n_treatment_values"]
        )
        diagnostics_df["limma_exclusion_reason"] = "post_qc_missing_values"
        diagnostics_df = diagnostics_df.sort_values(
            by=["n_total_values", "n_treatment_values", "n_control_values", "ID"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

        example_rows = diagnostics_df.head(5)
        example_text = ", ".join(
            f"{row.ID} (C={int(row.n_control_values)}, T={int(row.n_treatment_values)})"
            for row in example_rows.itertuples(index=False)
        )

        summary_lines = [
            (
                f"{chip_label}: Limma will use {complete_rows}/{total_rows} peptides with "
                f"complete sample coverage and exclude {excluded_rows} peptide(s) with "
                "post-QC missing values."
            ),
            (
                "These values are usually not absent in the raw export-image table. "
                "They are introduced later by QC and slope estimation."
            ),
            (
                "Typical reason: saturation filtering removes one or more exposure points; "
                "if fewer than 2 exposure measurements remain for a Barcode/Row/Cycle/peptide, "
                "the slope becomes NaN."
            ),
        ]
        if example_text:
            summary_lines.append(f"Examples: {example_text}")

        for line in summary_lines:
            self._dprint(f"     {line}")

        condition_label = getattr(self, "current_condition", "condition")
        suffix_base = (
            f"limma_missing_{chip_label}_{self.control_condition}_{condition_label}"
        )
        self._write_log_csv(diagnostics_df, suffix_base)
        self._write_log_text(summary_lines, f"{suffix_base}_summary")

    def _report_degenerate_limma(
        self,
        *,
        chip_label,
        df_prior,
        var_prior,
        complete_rows,
        total_rows,
    ):
        summary_lines = [
            (
                f"{chip_label}: Limma moderation collapsed to a shared posterior variance "
                f"(df_prior={df_prior}, var_prior={var_prior})."
            ),
            (
                f"Only {complete_rows}/{total_rows} peptide rows were eligible for limma on this chip."
            ),
            (
                "Instead of using degenerate moderated p-values, the pipeline will use "
                "ordinary row-wise t-tests for complete peptide rows and continue to "
                "filter peptides with insufficient values."
            ),
        ]

        for line in summary_lines:
            self._dprint(f"     {line}")

        condition_label = getattr(self, "current_condition", "condition")
        suffix_base = (
            f"limma_degenerate_{chip_label}_{self.control_condition}_{condition_label}"
        )
        self._write_log_text(summary_lines, f"{suffix_base}_summary")

    def _check_conditions_present_on_both_arrays(self, df_enrichment):
        """Fail when a Test Condition is measured on only one of the arrays.

        Samples are matched BY NAME, so a name that appears on PTK but not on
        STK (or the other way round) is a naming mistake in the annotation --
        typically the same sample written differently in the two files
        (``HPC_CTL`` vs ``CTL_HPC``), or a typo (``CTR_STR`` for ``STR_CTL``).
        Left alone it produces a comparison whose peptide statistics come from a
        single array, which looks like a normal result and is not.

        A run always carries both arrays (``run_peptide_statistics`` requires
        both inputs), so an unpaired condition is never a legitimate design.

        Args:
            df_enrichment: The design table, needs a ``Type`` column to check
                anything; without it the check is skipped.
        """
        if "Type" not in df_enrichment.columns:
            return

        present = (
            df_enrichment[["Test Condition", "Type"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
        )
        all_arrays = set(present["Type"])
        if len(all_arrays) < 2:
            return

        by_condition = present.groupby("Test Condition")["Type"].apply(set)
        unpaired = {
            condition: sorted(all_arrays - arrays)
            for condition, arrays in by_condition.items()
            if arrays != all_arrays
        }
        if not unpaired:
            return

        details = "; ".join(
            f"'{condition}' missing on {', '.join(missing)}"
            for condition, missing in sorted(unpaired.items())
        )
        raise ValueError(
            "Every Test Condition must be measured on both arrays, because "
            "samples are matched by their sample name. Unpaired: "
            f"{details}. Fix the 'Sample name' column so the same sample "
            "carries the same name in both annotation files."
        )

    def _split_control_conditions(self, df_enrichment, unique_conditions):
        """Split the Test Condition labels into controls and tests.

        Three sources of truth, in order of preference:

        1. The ``Condition Role`` column, written by the enricher from the c/t
           prefix of the sample name (``c2_STR_CTL_2_2`` -> condition
           ``STR_CTL``, role ``control``). Authoritative: the annotation file
           says which side a sample is on, nothing is inferred.
        2. Failing that -- an enrichment table built elsewhere, without the
           column -- a keyword match on the label
           (``CONTROL_CONDITION_KEYWORDS``), against WHOLE TOKENS of the label
           only. More than one hit is kept, but warned about: every extra
           control multiplies the comparisons.
        3. Failing that too, the alphabetically first condition, with a warning.

        Args:
            df_enrichment: The design table, read for ``Condition Role``.
            unique_conditions: The distinct Test Condition labels, in
                annotation-file order.

        Returns:
            tuple: ``(control_conditions, non_control_conditions)``, both in
            annotation-file order, controls guaranteed non-empty.
        """
        controls = []
        roles = None
        if self.CONDITION_ROLE_COLUMN in df_enrichment.columns:
            roles = (
                df_enrichment[[
                    "Test Condition",
                    self.CONDITION_ROLE_COLUMN,
                ]]
                .astype(str)
                .drop_duplicates()
            )
            # Rows the enricher could not label (an empty sample name) carry an
            # empty role. Drop them rather than letting '' count as a third
            # role, which would look like a contradiction below.
            roles = roles[roles[self.CONDITION_ROLE_COLUMN].str.strip() != ""]
            if roles.empty:
                roles = None

        if roles is not None:
            # A label carrying BOTH roles means the annotation contradicts
            # itself -- e.g. the same sample name prefixed 'c' on one array and
            # 't' on the other. Never guess which one was meant.
            conflicting = sorted(
                roles.groupby("Test Condition")[self.CONDITION_ROLE_COLUMN]
                .nunique()
                .pipe(lambda counts: counts[counts > 1])
                .index
            )
            if conflicting:
                raise ValueError(
                    "These Test Conditions are marked as BOTH control and test "
                    f"in the annotation: {conflicting}. Check the c/t prefix of "
                    "the affected sample names -- the same sample name must "
                    "carry the same prefix everywhere."
                )
            control_labels = set(
                roles.loc[
                    roles[self.CONDITION_ROLE_COLUMN] == self.CONDITION_ROLE_CONTROL,
                    "Test Condition",
                ]
            )
            controls = [
                condition
                for condition in unique_conditions
                if str(condition) in control_labels
            ]
            # The annotation spoke, and it named no control. Promoting one of the
            # test conditions here would silently produce a test-vs-test
            # comparison, so refuse instead of guessing.
            if not controls:
                raise ValueError(
                    "The annotation marks every condition as a test, so there "
                    "is no control to compare against. Prefix the control "
                    f"wells' sample names with 'c' (found: "
                    f"{sorted(str(c) for c in unique_conditions)})."
                )
            return controls, self._non_control_conditions(unique_conditions, controls)

        # No role information at all -- an enrichment table built outside the
        # enricher. Fall back to a keyword match, then to alphabetical order.
        # The keyword must be a WHOLE TOKEN of the label: a substring match
        # turns treatment arms such as 'DMSOtolerant_clone' or
        # 'Vehiclepretreated_DrugA' into controls, and the control is the
        # DENOMINATOR of every fold change.
        controls = [
            condition
            for condition in unique_conditions
            if self._matches_control_keyword(condition)
        ]

        if len(controls) > 1:
            # Guessed, not annotated. Every extra control is a further
            # reference: in 'all_pairs' each one produces its own set of
            # comparisons, exported next to the real ones under the same names
            # and with the same 'Significant' semantics. Unconditional, not
            # `_dprint` -- debugging_print is false by default
            # (config/pipeline_defaults.yaml:110).
            print(
                "     WARNING: More than one Test Condition carries a control "
                f"keyword: {[str(c) for c in controls]}. All of them are used "
                "as references, so this run produces one set of comparisons "
                "per control. Add a "
                f"'{self.CONDITION_ROLE_COLUMN}' column (or prefix the control "
                "wells' sample names with 'c' and the test wells' with 't') to "
                "say which condition is the control."
            )

        if not controls:
            # Labels that merely contain a keyword are not controls; say so,
            # otherwise the alphabetical fallback below looks arbitrary.
            near_misses = [
                str(condition)
                for condition in unique_conditions
                if any(
                    keyword in str(condition).lower()
                    for keyword in self.CONTROL_CONDITION_KEYWORDS
                )
            ]
            if near_misses:
                print(
                    "     WARNING: These Test Condition labels contain a "
                    f"control keyword only as part of a longer word: "
                    f"{near_misses}. They are treated as TEST conditions. "
                    "Rename the label (e.g. 'vehicle_pretreated' instead of "
                    "'Vehiclepretreated') if one of them really is the control."
                )

        if not controls:
            fallback = sorted(unique_conditions)[0]
            # Unconditional: this is the run's most consequential choice -- the
            # elected control is the DENOMINATOR of every fold change, so getting
            # it wrong flips the sign of every peptide_change and every kinase Z.
            # It must not be hidden behind `debugging_print`, which is false by
            # default.
            print(
                "     WARNING: No control condition could be determined from the "
                "annotation. The design table has no usable "
                f"'{self.CONDITION_ROLE_COLUMN}' column and no Test Condition "
                "label contains one of the fallback keywords "
                f"{list(self.CONTROL_CONDITION_KEYWORDS)}. Falling back to the "
                f"ALPHABETICALLY FIRST label: '{fallback}' is used as the CONTROL "
                "(the denominator of every fold change). Conditions found: "
                f"{sorted(str(c) for c in unique_conditions)}. If that is the "
                "wrong arm, prefix the control wells' sample names with 'c' and "
                "the test wells' with 't', or add a "
                f"'{self.CONDITION_ROLE_COLUMN}' column."
            )
            controls = [fallback]

        return controls, self._non_control_conditions(unique_conditions, controls)

    @classmethod
    def _matches_control_keyword(cls, label):
        """Say whether a label carries a control keyword as a whole token.

        Args:
            label: A Test Condition label; converted to ``str`` and lower-cased
                before the match.

        Returns:
            bool: True when any token of the label -- split on every run of
            non-alphanumeric characters -- is in
            ``CONTROL_CONDITION_KEYWORDS``. 'untreated_ctrl' matches,
            'DMSOtolerant_clone' does not.
        """
        tokens = {
            token
            for token in cls.CONTROL_KEYWORD_TOKEN_SPLIT.split(str(label).lower())
            if token
        }
        return bool(tokens & set(cls.CONTROL_CONDITION_KEYWORDS))

    @staticmethod
    def _non_control_conditions(unique_conditions, controls):
        """Return the conditions that are not controls, order preserved.

        Args:
            unique_conditions: All distinct Test Condition labels.
            controls: The labels classified as controls.

        Returns:
            list: The test conditions, guaranteed non-empty.
        """
        control_set = set(controls)
        non_controls = [
            condition
            for condition in unique_conditions
            if condition not in control_set
        ]
        if not non_controls:
            raise ValueError(
                "No test conditions found. At least one test condition is "
                f"required, but every condition is a control: {controls}."
            )
        return non_controls

    def _check_experimental_design_and_layout(self, df_enrichment):
        """Validate the design table and derive the comparisons to run.

        Returns:
            tuple: ``(contrasts, condition_metadata)`` where ``contrasts`` is a
            list of ``(reference, test)`` Test Condition pairs -- one entry per
            comparison, built according to ``contrast_mode`` -- and
            ``condition_metadata`` maps every condition to its construct,
            barcodes, rows and samples.
        """
        required_columns = [
            "Test Condition",
            "Construct",
            "Barcode",
            "Row",
            "Sample name",
            "Biological Replicate",
            "Technical Replicate",
        ]
        missing_cols = [col for col in required_columns if col not in df_enrichment.columns]
        if missing_cols:
            raise ValueError(
                f"The enrichment file is missing required columns: {missing_cols}"
            )

        unique_conditions = df_enrichment["Test Condition"].unique()
        if len(unique_conditions) < 2:
            raise ValueError(
                "Expected at least 2 test conditions (1 control and 1+ treatment), "
                f"but found {len(unique_conditions)}: {unique_conditions}"
            )

        self._check_conditions_present_on_both_arrays(df_enrichment)

        control_conditions, non_control_conditions = self._split_control_conditions(
            df_enrichment, unique_conditions
        )
        # The first control in annotation-file order is the nominal reference:
        # 'single_control' uses it as its only one, and it is what the legacy
        # self.control_condition attribute reports.
        control_condition = control_conditions[0]

        explicit_contrasts = getattr(self, "explicit_contrasts", None)
        if explicit_contrasts:
            # An explicit pair list WINS over contrast_mode: neither mode can
            # express a pairing / blocking criterion (tissue, subject, batch),
            # so listing the wanted comparisons is the only way to state one.
            known_conditions = set(unique_conditions)
            unknown_pairs = [
                pair
                for pair in explicit_contrasts
                if pair[0] not in known_conditions or pair[1] not in known_conditions
            ]
            if unknown_pairs:
                raise ValueError(
                    "contrasts names Test Conditions that are absent from the "
                    f"annotation: {unknown_pairs}. "
                    f"Present conditions: {sorted(known_conditions)}."
                )
            control_set = set(control_conditions)
            bad_reference_pairs = [
                pair for pair in explicit_contrasts if pair[0] not in control_set
            ]
            if bad_reference_pairs:
                raise ValueError(
                    "The reference (first) side of every contrast must be a "
                    f"control condition. Offending pairs: {bad_reference_pairs}. "
                    f"Control conditions: {control_conditions}."
                )
            contrasts = list(explicit_contrasts)
            contrast_source = f"explicit contrasts ({len(contrasts)} listed)"
        elif self.contrast_mode == self.CONTRAST_MODE_ALL_PAIRS:
            # Every control against every test, controls in the outer loop. Both
            # sides keep ANNOTATION-FILE order (the order Test Condition labels
            # first appear), which is what the single-control path always used --
            # so a one-control run produces the same comparisons in the same
            # order as before this mode existed. Control-vs-control and
            # test-vs-test are not produced: the c/t prefix in the sample
            # name already says which side of a comparison a well belongs on.
            # NOTE this is a BLIND cross product: no pairing, blocking or
            # covariate information enters it -- see the note printed below and
            # use the 'contrasts' parameter when a block criterion exists.
            contrasts = [
                (reference, condition)
                for reference in control_conditions
                for condition in non_control_conditions
            ]
            contrast_source = f"contrast_mode '{self.contrast_mode}'"
        else:
            # Single-control semantics: every other condition, including a second
            # control, is treated as a test. This mode reproduces earlier runs.
            contrasts = [
                (control_condition, condition)
                for condition in unique_conditions
                if condition != control_condition
            ]
            contrast_source = f"contrast_mode '{self.contrast_mode}'"

        if not contrasts:
            raise ValueError(
                "No comparisons to run. At least one control condition and one "
                "test condition are required."
            )

        # Test side actually used, order preserved, duplicates dropped.
        test_conditions_array = list(dict.fromkeys(test for _, test in contrasts))

        condition_metadata = {}
        for condition in unique_conditions:
            condition_data = df_enrichment[df_enrichment["Test Condition"] == condition]
            constructs = condition_data["Construct"].unique()

            if len(constructs) > 1:
                self._dprint(
                    f"     WARNING: Multiple constructs found for condition '{condition}': "
                    f"{constructs}. Using first construct."
                )

            barcode_row_combinations = list(
                condition_data[["Barcode", "Row"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )

            condition_metadata[condition] = {
                "construct": constructs[0] if len(constructs) > 0 else None,
                "barcodes": condition_data["Barcode"].unique().tolist(),
                "rows": condition_data["Row"].unique().tolist(),
                "barcode_row_pairs": barcode_row_combinations,
                "samples": condition_data["Sample name"].unique().tolist(),
                "n_samples": len(condition_data["Sample name"].unique()),
            }

        # Unconditional, for the same reason as the fallback warning above: the
        # run log must state which arm is the reference and which comparisons are
        # run. `_dprint` would suppress all of it under the default
        # `debugging_print: false`.
        print("     Experimental design detected:")
        for control in control_conditions:
            print(f"       - Control condition (reference/denominator): '{control}'")
        for test_condition in test_conditions_array:
            print(f"       - Test condition: '{test_condition}'")
        print(f"       - {contrast_source} -> {len(contrasts)} comparison(s):")
        for reference, condition in contrasts:
            print(
                f"           {reference} vs {condition}   "
                f"(log2 fold change = {condition} - {reference})"
            )

        if explicit_contrasts is None and len(contrasts) > 1:
            print(
                "       - NOTE: this list is the plain cross product of the "
                "control and test labels. No pairing, blocking or covariate "
                "criterion is applied, so a pair that crosses a block (tissue, "
                "subject, batch) is produced and reported like any other. List "
                "the wanted comparisons explicitly via the 'contrasts' "
                "parameter (config key "
                "default_peptide_statistics_params.contrasts) to suppress them."
            )

        self.contrast_source = contrast_source
        self.control_conditions = control_conditions
        self.control_condition = control_condition
        self.test_conditions_array = test_conditions_array
        self.contrasts = contrasts
        self.condition_metadata = condition_metadata
        return contrasts, condition_metadata

    def _load_and_merge_peptide_data(self, df_enrichment):
        df_ptk = self.df_ptk_input.merge(
            df_enrichment,
            left_on=["Barcode", "Row"],
            right_on=["Barcode", "Row"],
            how="left",
        )
        df_stk = self.df_stk_input.merge(
            df_enrichment,
            left_on=["Barcode", "Row"],
            right_on=["Barcode", "Row"],
            how="left",
        )
        return df_ptk, df_stk

    def _filter_high_saturation(
        self,
        df_ptk_merged,
        df_stk_merged,
        threshold_saturation=0.05,
    ):
        df_ptk_filtered = df_ptk_merged[df_ptk_merged["ID"] != "#REF"].copy()
        df_stk_filtered = df_stk_merged[df_stk_merged["ID"] != "#REF"].copy()

        df_ptk_qc_1 = df_ptk_filtered[
            df_ptk_filtered["Signal_Saturation"] < threshold_saturation
        ].copy()
        df_stk_qc_1 = df_stk_filtered[
            df_stk_filtered["Signal_Saturation"] < threshold_saturation
        ].copy()

        self._dprint(
            f"     PTK: {len(df_ptk_filtered) - len(df_ptk_qc_1)} peptide entries filtered out, "
            f"{len(df_ptk_qc_1)} peptide entries remaining"
        )
        self._dprint(
            f"     STK: {len(df_stk_filtered) - len(df_stk_qc_1)} peptide entries filtered out, "
            f"{len(df_stk_qc_1)} peptide entries remaining"
        )

        return df_ptk_qc_1, df_stk_qc_1

    def _calculate_slope(self, row, exposure_cols, exposure_times):
        signal_values = pd.to_numeric(row[exposure_cols], errors="coerce").values
        mask = ~np.isnan(signal_values)
        if mask.sum() < 2:
            return np.nan

        x = np.concatenate([[0], exposure_times[mask]])
        y = np.concatenate([[0], signal_values[mask]])
        slope = np.sum(x * y) / np.sum(x * x)
        return slope * 100

    @staticmethod
    def _rename_exposure_columns(df_pivoted):
        df_pivoted = df_pivoted.copy()
        df_pivoted.columns.name = None

        rename_dict = {}
        for col in df_pivoted.columns:
            try:
                rename_dict[col] = f"Exposure_{int(col)}ms"
            except (TypeError, ValueError):
                continue
        return df_pivoted.rename(columns=rename_dict)

    @staticmethod
    def _extract_exposure_columns_and_times(df_pivoted):
        exposure_pairs = []
        for col in df_pivoted.columns:
            if not isinstance(col, str) or not col.startswith("Exposure_"):
                continue
            try:
                exposure_time = float(col.split("_", 1)[1].replace("ms", ""))
            except (IndexError, ValueError):
                continue
            exposure_pairs.append((exposure_time, col))

        exposure_pairs.sort(key=lambda pair: pair[0])
        exposure_cols = [col for _, col in exposure_pairs]
        exposure_times = np.array([time for time, _ in exposure_pairs], dtype=float)
        return exposure_cols, exposure_times

    @staticmethod
    def _calculate_slopes_from_pivoted(df_pivoted, exposure_cols, exposure_times):
        df_pivoted = df_pivoted.copy()
        if not exposure_cols:
            df_pivoted["slope"] = np.nan
            return df_pivoted

        signal_matrix = df_pivoted[exposure_cols].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=float)
        valid_mask = ~np.isnan(signal_matrix)
        valid_counts = valid_mask.sum(axis=1)

        exposure_weights = exposure_times.reshape(1, -1)
        numerator = np.nansum(signal_matrix * exposure_weights, axis=1)
        denominator = np.where(
            valid_mask,
            exposure_weights**2,
            0.0,
        ).sum(axis=1)

        slopes = np.full(signal_matrix.shape[0], np.nan, dtype=float)
        usable = (valid_counts >= 2) & (denominator > 0)
        slopes[usable] = (numerator[usable] / denominator[usable]) * 100.0
        df_pivoted["slope"] = slopes
        return df_pivoted

    def _calculate_change_in_peptides(self, df_ptk_filtered, df_stk_filtered):
        df_stk_pivoted = df_stk_filtered.pivot_table(
            index=["Barcode", "Cycle", "Row", "ID", "Test Condition"],
            columns="Exposure Time",
            values="I_median",
            aggfunc="first",
        ).reset_index()

        df_ptk_pivoted = df_ptk_filtered.pivot_table(
            index=["Barcode", "Cycle", "Row", "ID", "Test Condition"],
            columns="Exposure Time",
            values="I_median",
            aggfunc="first",
        ).reset_index()

        df_stk_pivoted = self._rename_exposure_columns(df_stk_pivoted)
        df_ptk_pivoted = self._rename_exposure_columns(df_ptk_pivoted)

        exposure_cols_stk, exposure_times_stk = self._extract_exposure_columns_and_times(
            df_stk_pivoted
        )
        exposure_cols_ptk, exposure_times_ptk = self._extract_exposure_columns_and_times(
            df_ptk_pivoted
        )

        df_stk_pivoted = self._calculate_slopes_from_pivoted(
            df_stk_pivoted,
            exposure_cols_stk,
            exposure_times_stk,
        )
        df_ptk_pivoted = self._calculate_slopes_from_pivoted(
            df_ptk_pivoted,
            exposure_cols_ptk,
            exposure_times_ptk,
        )

        return df_ptk_pivoted, df_stk_pivoted

    # Reference / artificial peptides. KRSA drops these in krsa_filter_ref_pep:
    # ``ART_*`` are synthetic control spots and a leading lowercase ``p`` marks a
    # pre-phosphorylated control (pTY3H_64_78, pVASP_150_164 on this STK layout).
    # They are calibration spots, not biology, and must never enter the null.
    KRSA_REFERENCE_PEPTIDE_PATTERN = re.compile(r"^(?:ART_|p[A-Z])")

    @classmethod
    def _is_krsa_reference_peptide(cls, peptide_id):
        """Return True for KRSA's reference / artificial control spots."""
        return bool(cls.KRSA_REFERENCE_PEPTIDE_PATTERN.match(str(peptide_id)))

    def _peptide_quality_control_krsa(self, df_slope, chip_label="chip"):
        """KRSA-style peptide QC (``krsa_quick_filter``), used by ``qc_mode=QC_KRSA``.

        Reproduces the three criteria KRSA applies (run_krsa.R: signal_threshold,
        r2_threshold, krsa_filter_ref_pep):

        1. signal at the LONGEST exposure >= ``qc_krsa_signal_threshold``;
        2. R^2 of the signal-vs-exposure fit >= ``qc_krsa_r2_threshold``, taken
           as the WORST value over the samples so one bad sample disqualifies the
           peptide, which is what makes this stricter than the pyKinaXe default.
           The fit is through the origin and the R^2 uncentered, matching both
           KRSA (``lm(Signal ~ ExposureTime + 0)``) and the slope this pipeline
           actually consumes;
        3. reference / artificial control spots removed.

        All three are evaluated at the END POINT cycle (the largest cycle present),
        matching KRSA's ``krsa_extractEndPoint`` / ``krsa_extractEndPointMaxExp``.

        Measured against KRSA's own ``global_qc_passed_peptides.txt`` (166 PTK /
        112 STK peptides) on the shared benchmarking export, summed over the six
        contrasts: 9 of KRSA's peptides are missing here and 22 are kept that KRSA
        drops. The centered R^2 this filter used previously missed 20. The residual
        difference is KRSA fitting its R^2 on the scaled model from
        ``krsa_scaleModel`` and applying ``krsa_qc_steps`` (negatives clamped to 1,
        saturated points dropped) first; this filter works on the same pivoted
        signals the rest of pyKinaXe uses.

        Args:
            df_slope: Pivoted per-sample frame with ``Exposure_*ms`` columns.
            chip_label: Label used in the diagnostic line only.

        Returns:
            pd.DataFrame: ``df_slope`` restricted to the passing peptides, at the
                end-point cycle (mirrors the PTK default path, which also returns
                only cycle 94).
        """
        if df_slope.empty:
            return df_slope

        exposure_cols, exposure_times = self._extract_exposure_columns_and_times(
            df_slope
        )
        if len(exposure_cols) < 2:
            self._dprint(
                f"     {chip_label}: KRSA QC needs >= 2 exposure columns; kept as-is."
            )
            return df_slope

        end_cycle = df_slope["Cycle"].max()
        df_end = df_slope[df_slope["Cycle"] == end_cycle].copy()
        if df_end.empty:
            return df_end

        signals = df_end[exposure_cols].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )

        # Criterion 1: signal at the longest exposure (columns are sorted ascending).
        max_exposure_signal = signals[:, -1]

        # Criterion 2: R^2 of the THROUGH-ORIGIN fit signal ~ exposure, per row --
        # the same model whose slope the pipeline actually consumes, and the one
        # KRSA scores (``broom::glance`` on ``lm(Signal ~ ExposureTime + 0)``
        # reports the uncentered ``1 - RSS/sum(y^2)``). No intercept is estimated
        # anywhere in this path. Because the model has a single free parameter,
        # two points already leave a residual and support an R^2; only a single
        # point fits trivially, so the gate below is at two, not three.
        with np.errstate(invalid="ignore", divide="ignore"):
            valid = ~np.isnan(signals)
            n_valid = valid.sum(axis=1)
            x = np.where(valid, exposure_times.reshape(1, -1), np.nan)
            y = np.where(valid, signals, np.nan)
            cross = np.nansum(x * y, axis=1)
            sum_xx = np.nansum(x * x, axis=1)
            sum_yy = np.nansum(y * y, axis=1)
            denominator = sum_xx * sum_yy
            r_squared = np.where(denominator > 0, (cross**2) / denominator, np.nan)
        r_squared = np.where(n_valid >= 2, r_squared, np.nan)

        df_end["_krsa_max_exposure_signal"] = max_exposure_signal
        df_end["_krsa_r2"] = r_squared

        signal_ok = (
            df_end.groupby("ID", sort=False)["_krsa_max_exposure_signal"].min()
            >= self.qc_krsa_signal_threshold
        )
        # NaN R^2 propagates to a failure because min() skips NaN by default, so
        # count them explicitly instead of letting them disappear.
        grouped_r2 = df_end.groupby("ID", sort=False)["_krsa_r2"]
        r2_ok = (grouped_r2.min() >= self.qc_krsa_r2_threshold) & (
            grouped_r2.apply(lambda values: not values.isna().any())
        )

        passing = set(signal_ok[signal_ok].index) & set(r2_ok[r2_ok].index)
        reference = {
            pid for pid in df_end["ID"].unique() if self._is_krsa_reference_peptide(pid)
        }
        passing -= reference

        df_qc = df_end[df_end["ID"].isin(passing)].drop(
            columns=["_krsa_max_exposure_signal", "_krsa_r2"]
        )
        self._dprint(
            f"     {chip_label}: KRSA QC kept {df_qc['ID'].nunique()} of "
            f"{df_end['ID'].nunique()} peptides at cycle {end_cycle} "
            f"(signal >= {self.qc_krsa_signal_threshold}, "
            f"R2 >= {self.qc_krsa_r2_threshold}, "
            f"{len(reference)} reference spot(s) removed)."
        )
        return df_qc

    def _peptide_quality_control_stk(self, df_stk_slope):
        cutoff_threshold = 2
        calibration_number = 0.0026

        is_prephosphorylated_stk = df_stk_slope["ID"].str.startswith("p", na=False)
        df_prephospho_stk = df_stk_slope[is_prephosphorylated_stk]
        prephosph_peptides = ", ".join(df_prephospho_stk["ID"].unique())
        self._dprint(
            "     Following pre-phosphorylated peptides are used for STK quality control: "
            f"{prephosph_peptides}."
        )

        mean_prephospho_slope_stk = df_prephospho_stk["slope"].mean()
        threshold = mean_prephospho_slope_stk * calibration_number

        peptide_counts = (
            df_stk_slope.assign(_above_threshold=df_stk_slope["slope"] > threshold)
            .groupby("ID", sort=False)["_above_threshold"]
            .sum()
            .reset_index(name="count_above_threshold")
        )

        peptides_above = peptide_counts[
            peptide_counts["count_above_threshold"] >= cutoff_threshold
        ]["ID"]

        df_stk_slope_filtered = df_stk_slope[df_stk_slope["ID"].isin(peptides_above)]
        prephospho_ids = df_prephospho_stk["ID"].unique()
        df_stk_qc = df_stk_slope_filtered[
            ~df_stk_slope_filtered["ID"].isin(prephospho_ids)
        ]

        self._dprint(
            f"     STK QC results: {df_stk_qc['ID'].nunique()} peptides passed, "
            f"{df_stk_slope['ID'].nunique() - df_stk_qc['ID'].nunique()} peptides did not pass."
        )
        return df_stk_qc

    def _peptide_quality_control_ptk(self, df_ptk_slope):
        threshold_cutoff = 2
        df_ptk_slope_32_92 = df_ptk_slope[df_ptk_slope["Cycle"] < 94].copy()
        df_ptk_valid = df_ptk_slope_32_92[df_ptk_slope_32_92["slope"].notna()].copy()

        if df_ptk_valid.empty:
            return df_ptk_slope.iloc[0:0].copy()

        df_ptk_valid["cycle_float"] = df_ptk_valid["Cycle"].astype(float)
        df_ptk_valid["slope_float"] = df_ptk_valid["slope"].astype(float)
        df_ptk_valid["cycle_sq"] = df_ptk_valid["cycle_float"] ** 2
        df_ptk_valid["slope_sq"] = df_ptk_valid["slope_float"] ** 2
        df_ptk_valid["cycle_slope"] = (
            df_ptk_valid["cycle_float"] * df_ptk_valid["slope_float"]
        )

        df_regression = (
            df_ptk_valid.groupby(["Barcode", "Row", "ID"], sort=False)
            .agg(
                n=("Cycle", "size"),
                sum_x=("cycle_float", "sum"),
                sum_y=("slope_float", "sum"),
                sum_xx=("cycle_sq", "sum"),
                sum_yy=("slope_sq", "sum"),
                sum_xy=("cycle_slope", "sum"),
            )
            .reset_index()
        )
        df_regression = df_regression[df_regression["n"] >= 2].copy()

        if df_regression.empty:
            return df_ptk_slope.iloc[0:0].copy()

        n = df_regression["n"].to_numpy(dtype=float)
        sum_x = df_regression["sum_x"].to_numpy(dtype=float)
        sum_y = df_regression["sum_y"].to_numpy(dtype=float)
        sum_xx = df_regression["sum_xx"].to_numpy(dtype=float)
        sum_yy = df_regression["sum_yy"].to_numpy(dtype=float)
        sum_xy = df_regression["sum_xy"].to_numpy(dtype=float)

        cov_num = n * sum_xy - sum_x * sum_y
        x_var_num = n * sum_xx - sum_x**2
        y_var_num = n * sum_yy - sum_y**2

        slope = np.full(len(df_regression), np.nan, dtype=float)
        valid_slope = np.abs(x_var_num) > 1e-12
        slope[valid_slope] = cov_num[valid_slope] / x_var_num[valid_slope]

        intercept = np.full(len(df_regression), np.nan, dtype=float)
        nonzero_n = n > 0
        intercept[nonzero_n] = (
            sum_y[nonzero_n] - slope[nonzero_n] * sum_x[nonzero_n]
        ) / n[nonzero_n]

        corr_den = np.sqrt(np.maximum(x_var_num * y_var_num, 0.0))
        correlation = np.full(len(df_regression), np.nan, dtype=float)
        valid_corr = corr_den > 1e-12
        correlation[valid_corr] = cov_num[valid_corr] / corr_den[valid_corr]
        correlation = np.clip(correlation, -1.0, 1.0)

        r2 = np.where(y_var_num > 1e-12, correlation**2, 0.0)

        p_value = np.full(len(df_regression), np.nan, dtype=float)
        n_int = df_regression["n"].to_numpy(dtype=int)
        two_point_mask = valid_corr & (n_int == 2)
        p_value[two_point_mask] = 1.0

        finite_corr_mask = valid_corr & (n_int > 2)
        perfect_corr_mask = finite_corr_mask & (np.abs(correlation) >= 1.0)
        p_value[perfect_corr_mask] = 0.0

        nonperfect_corr_mask = finite_corr_mask & (np.abs(correlation) < 1.0)
        if np.any(nonperfect_corr_mask):
            t_stat = correlation[nonperfect_corr_mask] * np.sqrt(
                (n_int[nonperfect_corr_mask] - 2)
                / (1.0 - correlation[nonperfect_corr_mask] ** 2)
            )
            p_value[nonperfect_corr_mask] = 2.0 * stats.t.sf(
                np.abs(t_stat),
                df=n_int[nonperfect_corr_mask] - 2,
            )

        presence = np.full(len(df_regression), np.nan, dtype=float)
        positive_p = p_value > 0
        presence[positive_p] = (
            -np.log10(p_value[positive_p]) * np.sign(slope[positive_p])
        )
        zero_p = p_value == 0
        presence[zero_p] = np.inf * np.sign(slope[zero_p])

        df_regression["slope"] = slope
        df_regression["intercept"] = intercept
        df_regression["r2"] = r2
        df_regression["p_value"] = p_value
        df_regression["presence"] = presence

        peptide_fraction = (
            df_regression.assign(_present=df_regression["presence"] > threshold_cutoff)
            .groupby("ID", sort=False)["_present"]
            .mean()
            .reset_index(name="fractionPresent")
        )

        df_regression = df_regression.merge(peptide_fraction, on="ID", how="left")
        df_regression_filtered = df_regression[
            (df_regression["fractionPresent"] > 0.249)
            & (df_regression["ID"] != "ART_003_EAI(pY)AAPFAKKKXC")
        ]

        df_ptk_slope_94 = df_ptk_slope[
            (df_ptk_slope["Cycle"] == 94)
            & (df_ptk_slope["ID"].isin(df_regression_filtered["ID"]))
        ]

        self._dprint(
            f"     PTK QC results: {df_ptk_slope_94['ID'].nunique()} peptides passed, "
            f"{df_ptk_slope[df_ptk_slope['Cycle'] == 94]['ID'].nunique() - df_ptk_slope_94['ID'].nunique()} peptides did not pass."
        )
        return df_ptk_slope_94

    def _log2_transform_slope(self, df_ptk_slope, df_stk_slope):
        df_ptk_slope = df_ptk_slope.copy()
        df_stk_slope = df_stk_slope.copy()

        log_threshold = 1.0
        epsilon = 1e-2

        if self.log2_slope_mode == "epsilon_floor":
            df_ptk_slope["slope_log2"] = np.log2(
                np.maximum(df_ptk_slope["slope"].values, epsilon)
            )
            df_stk_slope["slope_log2"] = np.log2(
                np.maximum(df_stk_slope["slope"].values, epsilon)
            )
        elif self.log2_slope_mode == "krsa_na":
            # KRSA convention: keep log2(slope) only where it is >= 0 (i.e.
            # slope >= 1); a sub-baseline / undetectable slope is marked missing
            # (NaN) so it DROPS OUT of the group means and limma complete-case
            # rows (as KRSA drops that chip's contribution) instead of being
            # floored to a finite value.
            for _slope_df in (df_ptk_slope, df_stk_slope):
                _slope = _slope_df["slope"].to_numpy(dtype=float)
                _slope_df["slope_log2"] = np.where(
                    _slope >= log_threshold,
                    np.log2(np.maximum(_slope, log_threshold)),
                    np.nan,
                )
        else:
            # PamGene convention (log2_slope_mode='pamgene_zero'): a MEASURED
            # sub-baseline slope (slope <= 1) is clipped to log2(1) = 0. A
            # MISSING slope (NaN, e.g. a spot with too few valid exposures) is
            # not a measurement and must stay NaN -- otherwise it would enter
            # mean_control/mean_treatment (np.nanmean) as a real 0.0 and
            # inflate n_control/n_treatment, bypassing the replicate gates.
            # The other two modes already propagate NaN.
            for _slope_df in (df_ptk_slope, df_stk_slope):
                _slope = _slope_df["slope"].to_numpy(dtype=float)
                _slope_df["slope_log2"] = np.where(
                    np.isnan(_slope),
                    np.nan,
                    np.where(
                        _slope > log_threshold,
                        np.log2(np.maximum(_slope, log_threshold)),
                        0.0,
                    ),
                )

        return df_ptk_slope, df_stk_slope

    ARRAY_NORMALIZATION_LOG_COLUMNS = (
        "Type",
        "Barcode",
        "Row",
        "Test Condition",
        "n_peptides",
        "array_median_log2",
        "reference_median_log2",
        "offset_log2",
        "slope_factor",
        "status",
    )

    @staticmethod
    def _array_key(barcode, row):
        """Return the ``(Barcode, Row)`` key that identifies one array (well)."""
        return (str(barcode).strip(), int(row))

    def _estimate_array_normalization(
        self, df_ptk_qc_2_all, df_stk_qc_2_all, annotated_ids=None
    ):
        """Estimate the per-array offsets of ``array_normalization``, once per run.

        Runs on the run-wide QC result for the same reason the QC runs once: an
        offset estimated per comparison would give a control array shared by
        several comparisons a different correction in each of them. The log2
        values come from the same ``_log2_transform_slope`` the comparisons use.

        Args:
            df_ptk_qc_2_all: PTK rows of the whole run after the second QC.
            df_stk_qc_2_all: STK rows of the whole run after the second QC.
            annotated_ids: Peptide ids with a complete row (sequence, UniProt id
                and name) in the peptide-enrichment table. Any other peptide
                drops out of every downstream stage (the pivot in
                ``_rearange_peptide_data``), so it is left out here as well and
                the estimate uses exactly the peptides the analysis reports.
                None uses every peptide that passed QC.

        Returns:
            dict: ``{"PTK": {(Barcode, Row): offset}, "STK": {...}}``, empty when
                the normalisation is disabled. The per-array table is kept in
                ``self.array_normalization_offsets`` and written to the run log
                as ``array_normalization_offsets.csv``.
        """
        if not self.array_normalization:
            return {}

        estimate_offsets = {"median": self._median_array_offsets}[
            self.array_normalization_method
        ]
        df_ptk_log2, df_stk_log2 = self._log2_transform_slope(
            df_ptk_slope=df_ptk_qc_2_all,
            df_stk_slope=df_stk_qc_2_all,
        )
        offsets = {}
        tables = []
        for chip_label, df_log2 in (("PTK", df_ptk_log2), ("STK", df_stk_log2)):
            if annotated_ids is not None and not df_log2.empty:
                df_log2 = df_log2[df_log2["ID"].isin(annotated_ids)]
            offsets[chip_label], table = estimate_offsets(df_log2, chip_label)
            if not table.empty:
                tables.append(table)
            normalised = table[table["status"] == "normalised"]
            if normalised.empty:
                continue
            lowest = normalised.loc[normalised["offset_log2"].idxmin()]
            highest = normalised.loc[normalised["offset_log2"].idxmax()]
            print(
                f"     Array normalisation ({self.array_normalization_method}), "
                f"{chip_label}: {int(normalised['n_peptides'].iloc[0])} peptides "
                f"on {len(normalised)} arrays; offsets from "
                f"{lowest['offset_log2']:+.3f} ({lowest['Test Condition']}, "
                f"{lowest['Barcode']}/{lowest['Row']}) to "
                f"{highest['offset_log2']:+.3f} ({highest['Test Condition']}, "
                f"{highest['Barcode']}/{highest['Row']}) log2."
            )

        df_offsets = (
            pd.concat(tables, ignore_index=True)
            if tables
            else pd.DataFrame(columns=list(self.ARRAY_NORMALIZATION_LOG_COLUMNS))
        )
        self.array_normalization_offsets = df_offsets
        self._write_log_csv(df_offsets, "array_normalization_offsets")
        return offsets

    def _median_array_offsets(self, df_log2, chip_label):
        """Estimate the ``'median'`` normalisation offsets of one array type.

        The offset of an array is its median ``slope_log2`` minus the median of
        all array medians. The medians are taken over the SAME peptides on every
        array: those with a value on all arrays of the type, at the end-point
        cycle, reference / artificial control spots excluded. With
        ``log2_slope_mode='krsa_na'`` a dim array loses more weak peptides to
        NaN, so a median over each array's own peptides would sit too high on
        exactly the arrays that need the largest correction. Each peptide counts
        once: this frame holds one row per peptide and array, the per-protein
        fan-out only happens later in ``_rearange_peptide_data``.

        An array without any value (e.g. a failed well) is left out of the
        estimate and gets no offset. With fewer than
        ``ARRAY_NORMALIZATION_MIN_PEPTIDES`` common peptides the whole array type
        stays unnormalised.

        Args:
            df_log2: Long frame of the run-wide QC with ``slope_log2``.
            chip_label: ``"PTK"`` or ``"STK"``.

        Returns:
            tuple[dict, pd.DataFrame]: ``{(Barcode, Row): offset}`` (empty when
                the type stays unnormalised) and one log row per array.
        """
        columns = list(self.ARRAY_NORMALIZATION_LOG_COLUMNS)
        if df_log2.empty:
            return {}, pd.DataFrame(columns=columns)

        df_end = self._reduce_to_endpoint_cycle(df_log2, chip_label)
        df_end = df_end[
            ~df_end["ID"].map(self._is_krsa_reference_peptide).astype(bool)
        ]
        values = df_end.set_index(["ID", "Barcode", "Row"])["slope_log2"]
        if not values.index.is_unique:
            raise ValueError(
                f"{chip_label}: more than one end-point value per peptide and "
                "array; the array normalisation needs exactly one."
            )
        matrix = values.unstack(["Barcode", "Row"])
        # An array without a single value carries no information about its level;
        # it is logged as 'no values' but kept out of the estimate.
        empty_arrays = [key for key in matrix.columns if matrix[key].isna().all()]
        matrix = matrix.drop(columns=empty_arrays)
        common = matrix.dropna(axis=0, how="any")
        n_common = len(common)
        conditions = df_end.groupby(["Barcode", "Row"])["Test Condition"].first()

        if n_common >= self.ARRAY_NORMALIZATION_MIN_PEPTIDES:
            array_medians = common.median(axis=0)
            reference = float(np.median(array_medians.to_numpy(dtype=float)))
            offsets = array_medians - reference
            status = "normalised"
        else:
            array_medians = (
                common.median(axis=0)
                if n_common
                else pd.Series(np.nan, index=matrix.columns)
            )
            reference = np.nan
            offsets = pd.Series(0.0, index=matrix.columns)
            status = (
                f"skipped: {n_common} peptide(s) measured on every array, "
                f"minimum {self.ARRAY_NORMALIZATION_MIN_PEPTIDES}"
            )
            print(
                f"     WARNING: {chip_label}: array normalisation skipped -- only "
                f"{n_common} peptide(s) have a value on all {matrix.shape[1]} "
                f"arrays (minimum {self.ARRAY_NORMALIZATION_MIN_PEPTIDES}); this "
                "array type stays UNNORMALISED."
            )

        keys = list(offsets.index)
        offset_values = offsets.to_numpy(dtype=float)
        median_values = array_medians.reindex(offsets.index).to_numpy(dtype=float)
        rows = [
            {
                "Type": chip_label,
                "Barcode": key[0],
                "Row": key[1],
                "Test Condition": conditions.get(key),
                "n_peptides": n_common,
                "array_median_log2": median,
                "reference_median_log2": reference,
                "offset_log2": offset,
                # What the normalisation does to the raw slope of the array.
                "slope_factor": float(np.exp2(-offset)),
                "status": status,
            }
            for key, median, offset in zip(keys, median_values, offset_values)
        ]
        rows += [
            {
                "Type": chip_label,
                "Barcode": key[0],
                "Row": key[1],
                "Test Condition": conditions.get(key),
                "n_peptides": 0,
                "array_median_log2": np.nan,
                "reference_median_log2": reference,
                "offset_log2": np.nan,
                "slope_factor": np.nan,
                "status": "no values",
            }
            for key in empty_arrays
        ]
        table = pd.DataFrame(rows, columns=columns)
        offset_map = (
            {
                self._array_key(*key): float(value)
                for key, value in zip(keys, offset_values)
            }
            if status == "normalised"
            else {}
        )
        return offset_map, table

    def _apply_array_normalization(self, df_log2, offsets):
        """Subtract each array's normalisation offset from ``slope_log2``.

        Args:
            df_log2: Long frame with ``Barcode``, ``Row`` and ``slope_log2``.
            offsets: ``{(Barcode, Row): offset}`` of this array type, from
                :meth:`_estimate_array_normalization`.

        Returns:
            pd.DataFrame: A copy with shifted ``slope_log2``. NaN stays NaN, and
                an array without an offset is left unchanged.
        """
        if not offsets or df_log2.empty:
            return df_log2
        df_log2 = df_log2.copy()
        shift = np.fromiter(
            (
                offsets.get(self._array_key(barcode, row), 0.0)
                for barcode, row in zip(df_log2["Barcode"], df_log2["Row"])
            ),
            dtype=float,
            count=len(df_log2),
        )
        df_log2["slope_log2"] = df_log2["slope_log2"].to_numpy(dtype=float) - shift
        return df_log2

    def _reduce_to_endpoint_cycle(self, df_enriched, chip_label="chip"):
        """Keep one row per (peptide, sample): the end-point cycle.

        WHY THIS CANNOT BE AN ``aggfunc``
        ---------------------------------
        ``pivot_table``'s ``"first"``, ``"last"`` and ``"mean"`` all SKIP NaN. With
        ``log2_slope_mode='krsa_na'`` a sub-baseline slope is NaN by definition, so
        letting the pivot pick the value would silently substitute an earlier
        cycle's finite reading for the missing end-point one -- reinstating exactly
        the measurement that mode marks as absent. Reducing here preserves the NaN.
        It also removes an order dependency: with several cycles left, ``"first"``
        picked whichever row happened to come first upstream.

        ``Cycle == max(Cycle)`` over the whole array matches KRSA's
        ``krsa_extractEndPoint`` (``dplyr::filter(Cycle == max(Cycle))``, a GLOBAL
        max, not a per-peptide one) and the end-point convention already used by
        ``_peptide_quality_control_krsa`` and by the PTK filter, which hard-codes
        cycle 94. A sample that never reached the last cycle therefore drops out
        rather than contributing an earlier, non-comparable cycle -- the same
        trade-off KRSA makes.

        No-op on the default path (``qc_mode='QC_KRSA'``), where QC has already reduced
        both arrays to the end-point cycle. It bites in the legacy STK path, which
        passes every cycle through.

        Args:
            df_enriched: Long frame carrying a ``Cycle`` column.
            chip_label: Label used in the diagnostic line only.

        Returns:
            pd.DataFrame: ``df_enriched`` restricted to the end-point cycle.
        """
        if "Cycle" not in df_enriched.columns or df_enriched.empty:
            return df_enriched
        cycles = pd.to_numeric(df_enriched["Cycle"], errors="coerce")
        if cycles.isna().all():
            return df_enriched
        end_cycle = cycles.max()
        df_endpoint = df_enriched[cycles == end_cycle]
        dropped = len(df_enriched) - len(df_endpoint)
        if dropped:
            self._dprint(
                f"     {chip_label}: reduced to end-point cycle {end_cycle} "
                f"({dropped} row(s) from earlier cycles dropped before pivoting)."
            )
        return df_endpoint

    def _report_unannotated_peptides(self, df_enriched, array_type):
        """Report peptide ids that found no row in the peptide-enrichment table.

        Such a row carries a NaN ``PepProtein_UniprotID``, and the
        ``pivot_table`` in :meth:`_rearange_peptide_data` drops index rows with a
        NaN level -- so the peptide silently leaves EVERY downstream stage.
        Print it instead: an id-spelling drift between the array layout and the
        reference tables is otherwise invisible, and shows up only as a peptide
        count that is quietly too low.

        Args:
            df_enriched: Frame after the merge against the enrichment table.
            array_type: ``"PTK"`` or ``"STK"``.
        """
        if "PepProtein_UniprotID" not in df_enriched.columns:
            return
        missing = sorted(
            set(
                df_enriched.loc[
                    df_enriched["PepProtein_UniprotID"].isna(), "ID"
                ].astype(str)
            )
        )
        if not missing:
            return
        # `_rearange_peptide_data` runs once per comparison, and the answer is a
        # property of the run, not of the contrast -- report each distinct set
        # once instead of repeating an identical warning per comparison.
        reported = self.__dict__.setdefault("_unannotated_reported", set())
        key = (array_type, tuple(missing))
        if key in reported:
            return
        reported.add(key)
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        print(
            f"     WARNING: {array_type}: {len(missing)} peptide id(s) have no row "
            f"in {self.path_file_enrichment_peptides.name} and are therefore "
            f"dropped from every downstream stage ({shown})."
        )

    def _rearange_peptide_data(
        self,
        df_log2_ptk,
        df_log2_stk,
        df_enrichment,
        df_peptide_enrichment,
    ):
        """Rearrange peptide data.
        
        Returns:
            tuple: Rearranged peptide data.
        """
        df_log2_ptk = df_log2_ptk.copy()
        df_log2_stk = df_log2_stk.copy()
        df_enrichment = df_enrichment.copy()
        sample_key_col = "SampleKey"
        df_enrichment[sample_key_col] = df_enrichment.apply(
            lambda row: self._build_sample_key(
                row["Sample name"],
                row.get("Biological Replicate"),
                row.get("Technical Replicate"),
            ),
            axis=1,
        )

        sample_condition_map = (
            df_enrichment[
                [
                    sample_key_col,
                    "Sample name",
                    "Test Condition",
                    "Biological Replicate",
                    "Technical Replicate",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                ["Test Condition", "Sample name", "Biological Replicate", "Technical Replicate"],
                na_position="last",
            )
            .reset_index(drop=True)
        )
        unique_conditions = sample_condition_map["Test Condition"].unique()
        if len(unique_conditions) != 2:
            raise ValueError(
                "Expected exactly 2 test conditions (control and treatment), "
                f"but found {len(unique_conditions)}: {unique_conditions}"
            )

        control_condition = self.control_condition
        treatment_condition = [c for c in unique_conditions if c != control_condition][0]

        control_samples = sample_condition_map[
            sample_condition_map["Test Condition"] == control_condition
        ][sample_key_col].tolist()
        treatment_samples = sample_condition_map[
            sample_condition_map["Test Condition"] == treatment_condition
        ][sample_key_col].tolist()

        self.control_group = control_samples
        self.treatment_group = treatment_samples

        # Map each SampleKey to the chip (Barcode) it was measured on, so the
        # statistics stage can block/regress out per-chip batch effects.
        # df_enrichment holds BOTH arrays, and PTK and STK are physically
        # separate runs: their Barcodes differ and their sample-to-chip
        # partition need not agree. The map is therefore keyed on
        # (SampleKey, Type) and consumed per array by
        # _batch_labels_for(sample_keys, chip_label). A sample that sits on
        # several barcodes WITHIN one array is ambiguous and gets None, so the
        # correction is skipped rather than run on an arbitrary chip.
        barcode_by_type_map = {}
        if self.batch_column in df_enrichment.columns:
            has_type = "Type" in df_enrichment.columns
            df_barcodes = df_enrichment.dropna(subset=[sample_key_col])
            group_cols = [sample_key_col, "Type"] if has_type else [sample_key_col]
            for group_key, group in df_barcodes.groupby(group_cols, dropna=False):
                if has_type:
                    sample_key, array_type = group_key
                    if array_type is None or (
                        isinstance(array_type, float) and np.isnan(array_type)
                    ):
                        continue
                    type_key = str(array_type).strip().upper()
                    if not type_key:
                        continue
                else:
                    # No Type column: fall back to a single array-agnostic entry.
                    sample_key = (
                        group_key[0] if isinstance(group_key, tuple) else group_key
                    )
                    type_key = None
                barcodes = {
                    str(value).strip()
                    for value in group[self.batch_column].dropna().tolist()
                    if str(value).strip()
                }
                barcode_by_type_map.setdefault(sample_key, {})[type_key] = (
                    barcodes.pop() if len(barcodes) == 1 else None
                )

        sample_to_biorep = {}
        for _, sample_info in sample_condition_map.iterrows():
            sample_key = sample_info[sample_key_col]
            barcode_by_type = barcode_by_type_map.get(sample_key, {})
            distinct_barcodes = set(barcode_by_type.values())
            sample_to_biorep[sample_key] = {
                "sample_name": sample_info["Sample name"],
                "bio_rep": sample_info["Biological Replicate"],
                "tech_rep": sample_info["Technical Replicate"],
                "condition": sample_info["Test Condition"],
                "barcode_by_type": barcode_by_type,
                # Diagnostics only: the barcode when every array agrees on it,
                # None otherwise. Never used to drive the batch correction.
                "barcode": (
                    distinct_barcodes.pop() if len(distinct_barcodes) == 1 else None
                ),
            }
        self.sample_to_biorep = sample_to_biorep

        df_ptk_samples = df_log2_ptk.merge(
            df_enrichment[
                [
                    "Barcode",
                    "Row",
                    "Sample name",
                    "Construct",
                    "Type",
                    "Technical Replicate",
                    "Biological Replicate",
                    "Test Condition",
                    sample_key_col,
                ]
            ],
            left_on=["Barcode", "Row"],
            right_on=["Barcode", "Row"],
            how="left",
        )
        df_stk_samples = df_log2_stk.merge(
            df_enrichment[
                [
                    "Barcode",
                    "Row",
                    "Sample name",
                    "Construct",
                    "Type",
                    "Technical Replicate",
                    "Biological Replicate",
                    "Test Condition",
                    sample_key_col,
                ]
            ],
            left_on=["Barcode", "Row"],
            right_on=["Barcode", "Row"],
            how="left",
        )

        df_ptk_enriched = df_ptk_samples.merge(
            df_peptide_enrichment[
                ["ID", "Sequence", "PepProtein_UniprotID", "PepProtein_UniprotName"]
            ],
            on="ID",
            how="left",
        )
        df_stk_enriched = df_stk_samples.merge(
            df_peptide_enrichment[
                ["ID", "Sequence", "PepProtein_UniprotID", "PepProtein_UniprotName"]
            ],
            on="ID",
            how="left",
        )

        self._report_unannotated_peptides(df_ptk_enriched, "PTK")
        self._report_unannotated_peptides(df_stk_enriched, "STK")

        df_ptk_enriched = self._reduce_to_endpoint_cycle(df_ptk_enriched, "PTK")
        df_stk_enriched = self._reduce_to_endpoint_cycle(df_stk_enriched, "STK")

        # aggfunc is an identity here, never a reduction: _reduce_to_endpoint_cycle
        # has already left one row per (peptide, sample). See that method for why
        # the reduction must NOT be expressed as an aggfunc.
        #
        # dropna stays at its default True. Do NOT set dropna=False here to keep
        # all-NaN sample columns: with a MULTI-LEVEL index pandas then materialises
        # the full cartesian product of the level categories, so these four levels
        # blow ~200 peptides up to ~200^4 rows and the process is OOM-killed.
        # A sample whose every value is NaN therefore loses its column here, which
        # is safe: _calculate_peptide_statistics reindexes the requested sample
        # list back onto this frame (see `present_cols` /
        # _build_requested_group_matrix) and treats an absent column as all-NaN.
        df_wide_ptk = df_ptk_enriched.pivot_table(
            index=["ID", "Sequence", "PepProtein_UniprotID", "PepProtein_UniprotName"],
            columns=sample_key_col,
            values="slope_log2",
            aggfunc="first",
        ).reset_index()
        df_wide_stk = df_stk_enriched.pivot_table(
            index=["ID", "Sequence", "PepProtein_UniprotID", "PepProtein_UniprotName"],
            columns=sample_key_col,
            values="slope_log2",
            aggfunc="first",
        ).reset_index()

        for df_wide in (df_wide_ptk, df_wide_stk):
            df_wide.columns.name = None
            df_wide.rename(
                columns={
                    "PepProtein_UniprotID": "UniprotAccession",
                    "PepProtein_UniprotName": "Gene name",
                },
                inplace=True,
            )

        meta_cols = ["ID", "UniprotAccession", "Gene name", "Sequence"]
        sample_cols_ptk = sorted([c for c in df_wide_ptk.columns if c not in meta_cols])
        sample_cols_stk = sorted([c for c in df_wide_stk.columns if c not in meta_cols])

        df_ptk_rearanged = df_wide_ptk[meta_cols + sample_cols_ptk]
        df_stk_rearanged = df_wide_stk[meta_cols + sample_cols_stk]
        return df_ptk_rearanged, df_stk_rearanged

    def _calculate_peptide_statistics_sub(self, pControl, pTreat):
        mean_Treat = np.nanmean(pTreat)
        mean_Control = np.nanmean(pControl)

        std_pControl = np.nanstd(pControl, ddof=1)
        std_pTreat = np.nanstd(pTreat, ddof=1)

        delta = mean_Treat - mean_Control
        n_control = np.sum(~np.isnan(pControl))
        n_treat = np.sum(~np.isnan(pTreat))
        denominator = np.sqrt(
            std_pControl**2 / max(n_control, 1)
            + std_pTreat**2 / max(n_treat, 1)
        )
        peptide_statistic = 0.0 if denominator == 0 else delta / denominator

        return (
            mean_Control,
            mean_Treat,
            std_pControl,
            std_pTreat,
            peptide_statistic,
            delta,
        )

    def _limma_t_test(self, df_rearanged, pControl=None, pTreat=None, batch_design=None):
        """Return limma t test.

        No downstream stage consumes what this returns: the caller discards the
        returned ``logFC`` and the moderated p-values gate no plot or score. Kept
        for a future version with peptide-level significance (see the module
        docstring).

        Args:
            batch_design: Optional ``(n_samples, k-1)`` batch design matrix
                (samples ordered as ``pControl`` then ``pTreat``). When provided,
                it is appended to the design so the moderated statistics are
                estimated with the batch/chip effect blocked out. The condition
                coefficient remains at column index 1.
        """
        self._dprint("     Running limma moderated t-test...")

        sample_cols = list(pControl) + list(pTreat)
        Y = df_rearanged[sample_cols].values.astype(float)

        n_ctrl = len(pControl)
        n_treat = len(pTreat)
        n_samples = n_ctrl + n_treat

        X = np.zeros((n_samples, 2))
        X[:, 0] = 1
        X[n_ctrl:, 1] = 1
        if batch_design is not None:
            batch_design = np.asarray(batch_design, dtype=float)
            if batch_design.shape[0] != n_samples:
                raise ValueError(
                    "batch_design rows must match the number of limma samples "
                    f"({batch_design.shape[0]} != {n_samples})."
                )
            X = np.column_stack([X, batch_design])

        fit = lmFit(Y, X)
        sigma = fit.sigma.values if isinstance(fit.sigma, pd.Series) else np.asarray(fit.sigma)
        df_residual = (
            fit.df_residual.values
            if isinstance(fit.df_residual, pd.Series)
            else np.asarray(fit.df_residual)
        )
        sv_preview = squeezeVar(sigma**2, df_residual)
        s0_sq_preview = self._extract_scalar(sv_preview["var_prior"])
        d0_preview = self._extract_scalar(sv_preview["df_prior"])
        if not np.isfinite(d0_preview) or d0_preview > 1e6:
            raise DegenerateLimmaModerationError(
                df_prior=d0_preview,
                var_prior=s0_sq_preview,
            )

        try:
            fit = eBayes(fit, robust=False)
            coef_col = fit.coefficients.columns[1]
            logFC = fit.coefficients[coef_col].values
            t_stats = fit.t[coef_col].values
            p_values = fit.p_value[coef_col].values
            s2_post = (
                fit.s2_post.values
                if isinstance(fit.s2_post, pd.Series)
                else np.asarray(fit.s2_post)
            )
            df_total_scalar = float(np.median(np.asarray(fit.df_total)))
            s0_sq = self._extract_scalar(fit.s2_prior)
            d0 = self._extract_scalar(fit.df_prior)
        except (KeyError, IndexError, TypeError) as exc:
            self._dprint(
                f"     eBayes failed ({type(exc).__name__}: {exc}), "
                "falling back to manual squeezeVar."
            )
            s2_post = np.asarray(sv_preview["var_post"])
            s0_sq = s0_sq_preview
            d0 = d0_preview

            coef_col = fit.coefficients.columns[1]
            logFC = fit.coefficients[coef_col].values
            stdev_unscaled = fit.stdev_unscaled[coef_col].values
            t_stats = logFC / (stdev_unscaled * np.sqrt(s2_post))

            df_total = df_residual + d0
            df_pooled = np.nansum(df_residual)
            df_total = np.minimum(df_total, df_pooled)
            df_total_scalar = float(np.median(df_total))
            p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=df_total)

        return t_stats, p_values, s2_post, df_total_scalar, logFC, s0_sq, d0

    def _get_raw_sample_columns(self, df_pooled):
        control_cols = sorted(
            [
                col
                for col in df_pooled.columns
                if col.startswith("control_sample_") and not col.endswith("_zscore")
            ],
            key=lambda col: int(col.split("_")[2]),
        )
        treatment_cols = sorted(
            [
                col
                for col in df_pooled.columns
                if col.startswith("treatment_sample_") and not col.endswith("_zscore")
            ],
            key=lambda col: int(col.split("_")[2]),
        )
        return control_cols, treatment_cols

    def _batch_labels_for(self, sample_keys, chip_label=None):
        """Return the batch (chip) label for each sample key, or None if unknown.

        The barcode is resolved PER ARRAY: PTK and STK are separate physical
        runs, so the same sample carries a different Barcode on each and the
        two runs may distribute the samples over the chips differently. Passing
        the array (``chip_label``) is therefore required to get the right
        labels; without it a label is only returned when every array agrees.

        Args:
            sample_keys: Iterable of SampleKey strings (column identifiers).
            chip_label: Array the labels are requested for (``'PTK'``/``'STK'``,
                matching the enrichment ``Type`` column). ``None`` falls back to
                the array-agnostic barcode.

        Returns:
            list: Batch label per sample key (``None`` when no unambiguous
            barcode is known for that array; ``_resolve_batch_design`` then
            skips the correction).
        """
        mapping = getattr(self, "sample_to_biorep", {}) or {}
        array_key = None
        if chip_label is not None:
            array_key = str(chip_label).strip().upper() or None
        labels = []
        for sample_key in sample_keys:
            info = mapping.get(sample_key) or {}
            barcode_by_type = info.get("barcode_by_type") or {}
            if array_key is not None and array_key in barcode_by_type:
                barcode = barcode_by_type[array_key]
            elif None in barcode_by_type:
                # Enrichment table without a Type column: one entry for all arrays.
                barcode = barcode_by_type[None]
            elif array_key is None:
                distinct = set(barcode_by_type.values())
                barcode = distinct.pop() if len(distinct) == 1 else None
            else:
                # This array carries no barcode for the sample -> not estimable.
                barcode = None
            if barcode is None or (isinstance(barcode, float) and np.isnan(barcode)):
                labels.append(None)
            else:
                labels.append(str(barcode).strip() or None)
        return labels

    def _resolve_batch_design(self, condition_indicator, batch_labels):
        """Build the batch design matrix and decide whether correction is safe.

        The batch factor is one-hot encoded with the first level dropped as the
        reference (matching ``limma::removeBatchEffect``). Correction is only
        applied when the batch factor is estimable: every sample must carry a
        barcode, there must be >= 2 batch levels, the batch columns must not be
        collinear with the intercept + condition (i.e. batch not fully confounded
        with the condition), and there must be at least one residual degree of
        freedom.

        Args:
            condition_indicator: 0/1 array marking treatment samples (length n).
            batch_labels: Batch label per sample (length n; None if unknown).

        Returns:
            tuple: ``(batch_dummies, ok, reason)``; ``batch_dummies`` is an
            ``(n, k-1)`` float matrix (or ``None`` when not applicable).
        """
        labels = list(batch_labels)
        n_samples = len(labels)
        if n_samples == 0:
            return None, False, "no samples"
        if any(label is None for label in labels):
            return None, False, "some samples have no barcode"

        unique_labels = list(dict.fromkeys(labels))
        if len(unique_labels) < 2:
            return None, False, "only one batch level (nothing to correct)"

        dummies = np.zeros((n_samples, len(unique_labels)), dtype=float)
        for col_idx, label in enumerate(unique_labels):
            dummies[[i for i, value in enumerate(labels) if value == label], col_idx] = 1.0
        batch_dummies = dummies[:, 1:]  # drop reference level

        design = np.column_stack(
            [
                np.ones(n_samples, dtype=float),
                np.asarray(condition_indicator, dtype=float),
                batch_dummies,
            ]
        )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            return None, False, "batch confounded with condition"
        if design.shape[0] <= design.shape[1]:
            return None, False, "not enough samples to estimate the batch model"
        return batch_dummies, True, ""

    def _remove_batch_effect(
        self, expr, condition_indicator, batch_dummies, return_mask=False
    ):
        """Regress the batch effect out of an expression matrix, row by row.

        Mirrors ``limma::removeBatchEffect``: for each peptide a least-squares
        fit of ``value ~ intercept + condition + batch`` is estimated on the
        non-missing samples, and only the batch component is subtracted, so the
        treatment-vs-control (condition) effect is preserved. Rows with too few
        observations to identify the batch effect are left unchanged.

        Args:
            expr: ``(n_rows, n_samples)`` float matrix (may contain NaN).
            condition_indicator: 0/1 array marking treatment samples (length n).
            batch_dummies: ``(n_samples, k-1)`` batch design from
                :meth:`_resolve_batch_design`.

        Returns:
            np.ndarray: Batch-adjusted copy of ``expr`` (same shape; NaNs kept).
        """
        expr = np.asarray(expr, dtype=float)
        n_rows, n_samples = expr.shape
        n_batch_cols = batch_dummies.shape[1]
        design = np.column_stack(
            [
                np.ones(n_samples, dtype=float),
                np.asarray(condition_indicator, dtype=float),
                batch_dummies,
            ]
        )
        n_params = design.shape[1]

        adjusted = expr.copy()
        corrected = np.zeros(n_rows, dtype=bool)
        for row_idx in range(n_rows):
            values = expr[row_idx]
            observed = ~np.isnan(values)
            # A row is corrected as soon as the batch coefficients are
            # IDENTIFIABLE on its observed samples -- the exactly determined
            # case (n_observed == n_params, zero residual degrees of freedom)
            # included, which is what lmFit does as well. Requiring a spare
            # observation instead skipped exactly the incomplete rows in which
            # the chip offset does NOT cancel out of
            # mean(treatment) - mean(control).
            if int(observed.sum()) < n_params:
                continue
            design_obs = design[observed]
            if np.linalg.matrix_rank(design_obs) < n_params:
                continue
            beta, _residuals, _rank, _sv = np.linalg.lstsq(
                design_obs, values[observed], rcond=None
            )
            beta_batch = beta[-n_batch_cols:]
            # Subtracting from the full row keeps NaN positions as NaN.
            adjusted[row_idx] = values - batch_dummies @ beta_batch
            corrected[row_idx] = True
        if return_mask:
            return adjusted, corrected
        return adjusted

    def _calculate_peptide_statistics(self, df_rearanged, chip_label="chip"):
        # The enrichment-derived control/treatment groups may include samples
        # that have no surviving rows on this chip (e.g. all rows dropped by
        # chip-specific QC, or no Barcode/Row match). Restrict to columns that
        # are actually present in this chip's rearranged frame to avoid
        # KeyError on missing samples.
        requested_control_group = list(self.control_group)
        requested_treatment_group = list(self.treatment_group)
        present_cols = set(df_rearanged.columns)
        control_group = [s for s in requested_control_group if s in present_cols]
        treatment_group = [s for s in requested_treatment_group if s in present_cols]

        missing = [
            s for s in (requested_control_group + requested_treatment_group)
            if s not in present_cols
        ]
        if missing:
            self._dprint(
                f"     Skipping samples missing on this chip: {missing}"
            )
        if not control_group or not treatment_group:
            raise ValueError(
                "No surviving samples on this chip for at least one group "
                f"(control={control_group}, treatment={treatment_group})."
            )

        control_vals = df_rearanged[control_group].to_numpy(dtype=float)
        treat_vals = df_rearanged[treatment_group].to_numpy(dtype=float)
        control_vals_full = self._build_requested_group_matrix(
            df_rearanged=df_rearanged,
            requested_samples=requested_control_group,
        )
        treat_vals_full = self._build_requested_group_matrix(
            df_rearanged=df_rearanged,
            requested_samples=requested_treatment_group,
        )
        n_rows = len(df_rearanged)

        n_control = np.sum(~np.isnan(control_vals_full), axis=1)
        n_treatment = np.sum(~np.isnan(treat_vals_full), axis=1)
        n_total = n_control + n_treatment

        row_has_both_groups = (n_control >= 1) & (n_treatment >= 1)
        row_ttest_eligible = (n_control >= 2) & (n_treatment >= 2)

        # ------------------------------------------------------------------
        # Batch-effect correction: regress the per-chip (Barcode) offset out of
        # the log2 slopes before the treatment-vs-control difference is formed,
        # so peptide_change / peptide_statistic (and, for 'limma_block', the
        # moderated p-values) are batch-adjusted. Falls back silently to the
        # uncorrected difference when the batch factor is not estimable.
        row_batch_corrected = np.zeros(n_rows, dtype=bool)
        batch_design_for_limma = None
        if self.batch_correction:
            present_sample_keys = list(control_group) + list(treatment_group)
            present_condition = np.array(
                [0] * len(control_group) + [1] * len(treatment_group),
                dtype=float,
            )
            present_batch_labels = self._batch_labels_for(
                present_sample_keys, chip_label
            )
            batch_dummies_present, batch_ok, batch_reason = self._resolve_batch_design(
                present_condition, present_batch_labels
            )
            if batch_ok:
                expr_present = np.concatenate([control_vals, treat_vals], axis=1)
                expr_present_adj, row_batch_corrected = self._remove_batch_effect(
                    expr_present,
                    present_condition,
                    batch_dummies_present,
                    return_mask=True,
                )
                control_adj = expr_present_adj[:, : len(control_group)]
                treat_adj = expr_present_adj[:, len(control_group):]

                # Inject the adjusted present columns back into the requested-order
                # full matrices (absent samples stay NaN); the rebinding below then
                # routes all downstream means/SD/z-scores through adjusted values.
                control_vals_full = control_vals_full.copy()
                for present_idx, sample_key in enumerate(control_group):
                    control_vals_full[:, requested_control_group.index(sample_key)] = (
                        control_adj[:, present_idx]
                    )
                treat_vals_full = treat_vals_full.copy()
                for present_idx, sample_key in enumerate(treatment_group):
                    treat_vals_full[:, requested_treatment_group.index(sample_key)] = (
                        treat_adj[:, present_idx]
                    )

                if self.batch_correction_method == "limma_block":
                    batch_design_for_limma = batch_dummies_present
                self._dprint(
                    f"     {chip_label}: batch correction applied "
                    f"(method={self.batch_correction_method}, "
                    f"{batch_dummies_present.shape[1] + 1} chips, "
                    f"column='{self.batch_column}')."
                )
            else:
                self._dprint(
                    f"     {chip_label}: batch correction skipped "
                    f"({batch_reason}); using the uncorrected difference."
                )

        t_stats = np.full(n_rows, np.nan, dtype=float)
        p_values = np.full(n_rows, np.nan, dtype=float)
        s2_moderated = np.full(n_rows, np.nan, dtype=float)
        df_moderated = np.full(n_rows, np.nan, dtype=float)
        statistics_method = np.full(n_rows, "insufficient_values", dtype=object)
        limma_included = np.zeros(n_rows, dtype=bool)
        limma_exclusion_reason = np.full(n_rows, "", dtype=object)
        fallback_method_name = "t_test_fallback"

        # limma's lmFit collapses to a singular design when either group has
        # fewer than 2 samples (no within-group variance to estimate) or when
        # the combined matrix is rank-deficient. In those cases fall back to a
        # plain Welch/Student t-test instead of crashing the whole pipeline.
        use_limma_here = (
            self.use_limma and len(control_group) >= 2 and len(treatment_group) >= 2
        )
        if self.use_limma and not use_limma_here:
            self._dprint(
                f"     {chip_label}: Limma needs >=2 samples per group on each chip; "
                f"falling back to row-wise t-test where possible "
                f"(n_ctrl={len(control_group)}, n_treat={len(treatment_group)})."
            )

        if use_limma_here:
            expr_limma = np.concatenate([control_vals, treat_vals], axis=1)
            complete_case_mask = ~np.isnan(expr_limma).any(axis=1)
            limma_exclusion_reason[~complete_case_mask] = "post_qc_missing_values"

            if not np.any(complete_case_mask):
                self._dprint(
                    f"     {chip_label}: Limma has no complete peptide rows on this chip; "
                    "falling back to row-wise t-test where possible."
                )
                use_limma_here = False
            else:
                if not np.all(complete_case_mask):
                    self._report_limma_exclusions(
                        df_rearanged=df_rearanged,
                        chip_label=chip_label,
                        complete_case_mask=complete_case_mask,
                        n_control_values=n_control,
                        n_treatment_values=n_treatment,
                    )
            try:
                if use_limma_here:
                    # One row per peptide ID for the fit: _rearange_peptide_data
                    # repeats a peptide once per UniProt accession with
                    # byte-identical measurements (up to 16x for
                    # 'H2B1B_ 27_40'), and squeezeVar/eBayes estimate the prior
                    # over exactly the row distribution they are handed. Fitting
                    # the fanned-out frame would weight the prior by annotation
                    # count (pseudoreplication); the per-peptide result is
                    # broadcast back onto every copy afterwards.
                    limma_sample_cols = list(control_group) + list(treatment_group)
                    complete_positions = np.flatnonzero(complete_case_mask)
                    limma_frame = df_rearanged.loc[complete_case_mask].copy()
                    # factorize assigns codes in order of first appearance, so
                    # np.unique(..., return_index=True)[1][c] is the first row of
                    # code c and fit_frame is in code order by construction.
                    dedup_codes, _dedup_ids = pd.factorize(
                        limma_frame["ID"], use_na_sentinel=False
                    )
                    first_row_of_code = np.unique(dedup_codes, return_index=True)[1]
                    fit_frame = limma_frame.iloc[first_row_of_code].copy()
                    if len(fit_frame) < len(limma_frame):
                        max_distinct = int(
                            limma_frame.groupby("ID", sort=False)[limma_sample_cols]
                            .nunique()
                            .to_numpy()
                            .max()
                        )
                        if max_distinct > 1:
                            self._dprint(
                                f"     {chip_label}: WARNING - duplicate peptide IDs "
                                "carry different values; the first annotation row "
                                "is used for the limma fit."
                            )
                        self._dprint(
                            f"     {chip_label}: limma fit over {len(fit_frame)} "
                            f"unique peptide ID(s) instead of {len(limma_frame)} "
                            "annotation rows (protein fanout excluded from the prior)."
                        )
                    (
                        t_stats_unique,
                        p_values_unique,
                        s2_post_unique,
                        df_total,
                        _logFC,
                        _s2_prior,
                        _df_prior,
                    ) = self._limma_t_test(
                        df_rearanged=fit_frame,
                        pControl=control_group,
                        pTreat=treatment_group,
                        batch_design=batch_design_for_limma,
                    )
                    t_stats[complete_positions] = np.asarray(
                        t_stats_unique,
                        dtype=float,
                    )[dedup_codes]
                    p_values[complete_positions] = np.asarray(
                        p_values_unique,
                        dtype=float,
                    )[dedup_codes]
                    s2_moderated[complete_positions] = np.asarray(
                        s2_post_unique,
                        dtype=float,
                    )[dedup_codes]
                    df_moderated[complete_case_mask] = float(df_total)
                    statistics_method[complete_case_mask] = "limma"
                    limma_included[complete_case_mask] = True
            except DegenerateLimmaModerationError as exc:
                self._report_degenerate_limma(
                    chip_label=chip_label,
                    df_prior=exc.df_prior,
                    var_prior=exc.var_prior,
                    complete_rows=int(np.sum(complete_case_mask)),
                    total_rows=len(df_rearanged),
                )
                fallback_method_name = "t_test_degenerate_limma"
                use_limma_here = False
            except (np.linalg.LinAlgError, ValueError) as exc:
                self._dprint(
                    f"     {chip_label}: Limma fit failed ({type(exc).__name__}: {exc}); "
                    "falling back to row-wise t-test where possible."
                )
                use_limma_here = False

        fallback_mask = np.isnan(p_values) & row_ttest_eligible
        if np.any(fallback_mask):
            fallback_t_stats, fallback_p_values = self._run_rowwise_t_tests(
                control_vals_full,
                treat_vals_full,
                fallback_mask,
            )
            t_stats[fallback_mask] = fallback_t_stats[fallback_mask]
            p_values[fallback_mask] = fallback_p_values[fallback_mask]
            statistics_method[fallback_mask] = fallback_method_name
            self._dprint(
                f"     {chip_label}: Using row-wise t-test fallback for "
                f"{int(np.sum(fallback_mask))} peptide(s) without a limma result."
            )

        mean_control = np.nanmean(control_vals_full, axis=1)
        mean_treatment = np.nanmean(treat_vals_full, axis=1)
        std_control = self._calculate_rowwise_sd(control_vals_full)
        std_treatment = self._calculate_rowwise_sd(treat_vals_full)
        delta = mean_treatment - mean_control

        denominator = np.sqrt(
            std_control**2 / np.maximum(n_control, 1)
            + std_treatment**2 / np.maximum(n_treatment, 1)
        )
        peptide_statistic = np.where(denominator == 0, 0.0, delta / denominator)
        average_expression = (mean_control + mean_treatment) / 2.0
        logp_value = np.full(n_rows, np.nan, dtype=float)
        positive_p_mask = p_values > 0
        logp_value[positive_p_mask] = -np.log10(p_values[positive_p_mask])
        zero_p_mask = p_values == 0
        logp_value[zero_p_mask] = np.inf

        statistics_method[np.isnan(p_values) & row_has_both_groups] = "insufficient_replicates"
        statistics_method[np.isnan(p_values) & ~row_has_both_groups] = "missing_group_values"

        df_pooled = df_rearanged[
            ["ID", "UniprotAccession", "Gene name", "Sequence"]
        ].copy()
        df_pooled.rename(columns={"Gene name": "GeneName"}, inplace=True)

        sample_cols = []
        for idx, sample_name in enumerate(requested_control_group, start=1):
            col = f"control_sample_{idx}"
            df_pooled[col] = control_vals_full[:, idx - 1]
            df_pooled[f"control_label_{idx}"] = self._format_group_sample_label(
                sample_name,
                "Control",
            )
            sample_cols.append(col)
        for idx, sample_name in enumerate(requested_treatment_group, start=1):
            col = f"treatment_sample_{idx}"
            df_pooled[col] = treat_vals_full[:, idx - 1]
            df_pooled[f"treatment_label_{idx}"] = self._format_group_sample_label(
                sample_name,
                "Treatment",
            )
            sample_cols.append(col)

        # t_statistic / p_value / logp_value / s2_moderated / df_moderated come
        # from the limma path and are currently REPORTED ONLY -- kept for a future
        # version that scores peptide-level significance again (module docstring).
        df_pooled["average_expression"] = average_expression
        df_pooled["t_statistic"] = t_stats
        df_pooled["p_value"] = p_values
        df_pooled["logp_value"] = logp_value
        df_pooled["mean_control"] = mean_control
        df_pooled["mean_treatment"] = mean_treatment
        df_pooled["SD_control"] = std_control
        df_pooled["SD_treatment"] = std_treatment
        df_pooled["peptide_statistic"] = peptide_statistic
        df_pooled["peptide_change"] = delta
        df_pooled["s2_moderated"] = s2_moderated
        df_pooled["df_moderated"] = df_moderated
        df_pooled["n_control_values"] = n_control
        df_pooled["n_treatment_values"] = n_treatment
        df_pooled["n_total_values"] = n_total
        df_pooled["statistics_method"] = statistics_method
        df_pooled["limma_included"] = limma_included
        df_pooled["limma_exclusion_reason"] = np.where(
            limma_exclusion_reason != "",
            limma_exclusion_reason,
            np.nan,
        )
        # True only where the per-chip (Barcode) offset was actually regressed
        # out of this row; False when batch correction is off, the batch factor
        # was not estimable for the run, or the row had too few observations.
        df_pooled["batch_corrected"] = row_batch_corrected

        expr = np.concatenate([control_vals_full, treat_vals_full], axis=1)
        mean_g = np.nanmean(expr, axis=1, keepdims=True)
        raw_scale = self._calculate_rowwise_sd(expr)
        moderated_scale = np.full(n_rows, np.nan, dtype=float)
        valid_moderated_mask = np.isfinite(s2_moderated) & (s2_moderated > 0)
        moderated_scale[valid_moderated_mask] = np.sqrt(
            s2_moderated[valid_moderated_mask]
        )

        zscore_scale = raw_scale.copy()
        zscore_scale_source = np.full(n_rows, "unscaled", dtype=object)
        valid_raw_scale_mask = np.isfinite(raw_scale) & (raw_scale > 0)
        zscore_scale_source[valid_raw_scale_mask] = "observed_row_sd"
        zscore_scale[valid_moderated_mask] = moderated_scale[valid_moderated_mask]
        zscore_scale_source[valid_moderated_mask] = "moderated_sd"

        with np.errstate(divide="ignore", invalid="ignore"):
            scaled_values = (expr - mean_g) / zscore_scale.reshape(-1, 1)
        scaled_values[~np.isfinite(scaled_values)] = np.nan

        # The *_zscore columns use limma's moderated SD where available. Since the
        # peptide heatmap switched to log2(S100) they have no reader either --
        # exported for inspection and for a future significance-aware plot.
        df_pooled["zscore_scale_source"] = zscore_scale_source
        for i, col in enumerate(sample_cols):
            df_pooled[f"{col}_zscore"] = scaled_values[:, i]

        # Only rows whose peptide_change is NOT computable are dropped. That is
        # exactly "missing_group_values" (one group entirely NaN -> np.nanmean over
        # an all-NaN group -> NaN delta). Rows labelled "insufficient_replicates"
        # have data in BOTH groups and therefore a FINITE peptide_change; only the
        # test statistic could not be formed, and t_statistic/p_value are REPORTED
        # ONLY (see above). Dropping them removed the peptide from the hit set AND
        # from the measured universe -- peptide_change is the one column the kinase
        # stage reads, and kx_upstream_kinase_analysis.py already excludes rows
        # with a NaN peptide_change on its own.
        insufficient_mask = df_pooled["statistics_method"].eq("missing_group_values")
        kept_without_test = df_pooled["statistics_method"].eq("insufficient_replicates")
        if np.any(kept_without_test):
            self._dprint(
                f"     {chip_label}: Keeping {int(np.sum(kept_without_test))} peptide(s) "
                "with a finite peptide_change but no test statistic "
                "(statistics_method='insufficient_replicates')."
            )
        if np.any(insufficient_mask):
            insufficient_rows = df_pooled.loc[insufficient_mask].copy()
            self._dprint(
                f"     {chip_label}: Filtering out {len(insufficient_rows)} peptide(s) "
                "with an entirely missing group before downstream analysis."
            )
            condition_label = getattr(self, "current_condition", "condition")
            self._write_debug_csv(
                insufficient_rows,
                f"insufficient_peptides_{chip_label}_{self.control_condition}_{condition_label}",
            )
            df_pooled = (
                df_pooled.loc[~insufficient_mask]
                .reset_index(drop=True)
            )

        # Neither warning is behind debugging_print: an emptied or replicate-less
        # chip table is otherwise SILENT whenever only ONE array is affected --
        # the PTK/STK concat in _compute_condition_result stays non-empty and the
        # run continues on half the data.
        if n_rows > 0 and df_pooled.empty:
            print(
                f"     WARNING: {chip_label}: ALL {n_rows} peptide(s) were removed "
                "because at least one of the two groups is entirely missing "
                f"(control={requested_control_group}, "
                f"treatment={requested_treatment_group}). "
                "This chip contributes NOTHING to the kinase analysis."
            )
        elif not df_pooled.empty and not np.isfinite(
            df_pooled["p_value"].to_numpy(dtype=float)
        ).any():
            print(
                f"     WARNING: {chip_label}: no peptide has a usable test statistic "
                "(fewer than 2 usable replicates in at least one group; "
                f"control={requested_control_group}, "
                f"treatment={requested_treatment_group}). "
                "peptide_change is still exported, so the kinase analysis runs "
                "WITHOUT any replicate support."
            )

        return df_pooled

    def _plot_waterfall(self, df_peptides, output_path=None, control=None, condition=None):
        """Plot the ranked peptide log2 fold changes as a waterfall."""
        if df_peptides is None or df_peptides.empty:
            self._dprint("     No peptide data found for waterfall plot.")
            return None

        save_path = (
            str(Path(output_path) / f"peptides_waterfall_plot_{control}_{condition}.png")
            if output_path is not None
            else None
        )

        try:
            WaterfallPlot(
                debugging_print=self.debugging_print,
                data=df_peptides,
                value_col="peptide_change",
                lfc_cutoffs=self.waterfall_lfc_cutoffs,
                cutoff_mode=self.waterfall_cutoff_mode,
                primary_lfc_cutoff=self.waterfall_primary_lfc_cutoff,
                x_label="Log2 Fold Change",
                title=f"Peptide log2 fold change: {condition} vs {control}",
                save_path=save_path,
            )
        except ValueError as exc:
            self._dprint(f"     Peptide waterfall skipped: {exc}")
        return None

    def _plot_heatmap(self, df_peptides, output_path=None, control=None, condition=None):
        """Plot one peptide signal heatmap per array (PTK and STK separately).

        The two arrays carry different peptides and sit on different signal
        scales, so a shared panel would compress both. Every peptide that passed
        QC is drawn -- there is no p-value filter -- and the colour is the
        per-sample log2(S100), the quantity ``peptide_change`` is built from.
        """
        if df_peptides is None or df_peptides.empty:
            self._dprint("     No peptide data available for heatmap plot.")
            return None

        if "Type" in df_peptides.columns:
            # Dedup per array: one peptide can be printed on BOTH arrays, and each
            # copy belongs in its own panel.
            panels = [
                (str(array), group.drop_duplicates(subset="ID", keep="first"))
                for array, group in df_peptides.groupby("Type", sort=True)
            ]
        else:
            self._dprint(
                "     No 'Type' column found; drawing one combined peptide heatmap."
            )
            panels = [(None, df_peptides.drop_duplicates(subset="ID", keep="first"))]

        for array, df_panel in panels:
            suffix = f"{array}_" if array else ""
            save_path = (
                str(
                    Path(output_path)
                    / f"peptides_heatmap_{suffix}{control}_{condition}.png"
                )
                if output_path is not None
                else None
            )
            title_array = f"{array} " if array else ""
            try:
                HeatmapPlot_Peptides(
                    debugging_print=self.debugging_print,
                    data=df_panel,
                    save_path=save_path,
                    title=(
                        f"{title_array}Peptide signal: {condition} vs {control}"
                    ),
                )
            except ValueError as exc:
                self._dprint(
                    f"     Peptide heatmap skipped for {array or 'all peptides'}: {exc}"
                )
        return None

    def _build_condition_worker(self):
        worker = PeptideStatistics(
            path_file_enrichment_peptides=self.path_file_enrichment_peptides,
            waterfall_plot=False,
            waterfall_plot_output=self.waterfall_plot_output,
            waterfall_lfc_cutoffs=self.waterfall_lfc_cutoffs,
            waterfall_cutoff_mode=self.waterfall_cutoff_mode,
            waterfall_primary_lfc_cutoff=self.waterfall_primary_lfc_cutoff,
            heatmap_plot=False,
            heatmap_plot_output=self.heatmap_plot_output,
            path_output_peptide_statistic=self.path_output_peptide_statistic,
            log_output=self.log_output,
            use_limma=self.use_limma,
            debugging_print=self.debugging_print,
            log2_slope_mode=self.log2_slope_mode,
            batch_correction=self.batch_correction,
            batch_correction_method=self.batch_correction_method,
            batch_column=self.batch_column,
            array_normalization=self.array_normalization,
            array_normalization_method=self.array_normalization_method,
            # The QC settings have to be passed through; otherwise the
            # per-condition worker would use the configured defaults and
            # qc_mode and the two thresholds would have no effect.
            qc_mode=self.qc_mode,
            qc_krsa_signal_threshold=self.qc_krsa_signal_threshold,
            qc_krsa_r2_threshold=self.qc_krsa_r2_threshold,
        )
        # One worker is built PER COMPARISON, so a set kept on the worker itself
        # would never see a second comparison and `_report_unannotated_peptides`
        # would repeat its identical warning once per contrast. Hand the children
        # the parent's set so the run reports each distinct case exactly once.
        worker._unannotated_reported = self.__dict__.setdefault(
            "_unannotated_reported", set()
        )
        return worker

    def _compute_condition_result(
        self,
        *,
        condition,
        control_condition,
        construct,
        df_enrichment_condition,
        df_peptide_enrichment,
        df_ptk_qc_1,
        df_stk_qc_1,
        df_ptk_slope,
        df_stk_slope,
        df_ptk_qc_2,
        df_stk_qc_2,
        array_offsets=None,
    ):
        """Compute condition result.
        
        Args:
            df_ptk_qc_2: PTK rows of this comparison AFTER the run-wide second
                peptide QC (see ``run_peptide_statistics``). The QC is NOT run
                here: all three QC variants aggregate across the samples they
                see, so running them per comparison gives every comparison its
                own peptide population.
            df_stk_qc_2: STK rows of this comparison after the run-wide QC.
            array_offsets: Per-array normalisation offsets of the whole run,
                ``{"PTK": {(Barcode, Row): offset}, "STK": {...}}``; estimated
                once in ``run_peptide_statistics``, never per comparison. Empty
                or None leaves ``slope_log2`` unchanged.
        """
        worker = self._build_condition_worker()
        worker.control_condition = control_condition
        worker.current_condition = condition

        # The second-pass peptide QC does NOT run here. All three QC variants
        # aggregate ACROSS the samples of the frame they are given -- the KRSA
        # filter takes the worst signal / R^2 over the samples present, the two
        # legacy filters count how many samples clear a threshold -- so running
        # them on the two arms of ONE comparison gave every comparison of a run
        # its own peptide population, its own hypergeometric M and its own
        # kinase panel, with no exported column recording which. The QC now runs
        # ONCE per run in `run_peptide_statistics` and this function receives the
        # rows of this comparison already filtered; only the log2 fold change
        # stays comparison-specific.

        df_ptk_log2, df_stk_log2 = worker._log2_transform_slope(
            df_ptk_slope=df_ptk_qc_2,
            df_stk_slope=df_stk_qc_2,
        )
        if array_offsets:
            df_ptk_log2 = worker._apply_array_normalization(
                df_ptk_log2, array_offsets.get("PTK")
            )
            df_stk_log2 = worker._apply_array_normalization(
                df_stk_log2, array_offsets.get("STK")
            )
        df_ptk_rearanged, df_stk_rearanged = worker._rearange_peptide_data(
            df_log2_ptk=df_ptk_log2,
            df_log2_stk=df_stk_log2,
            df_enrichment=df_enrichment_condition,
            df_peptide_enrichment=df_peptide_enrichment,
        )
        df_ptk_pooled = worker._calculate_peptide_statistics(
            df_rearanged=df_ptk_rearanged,
            chip_label="PTK",
        )
        df_stk_pooled = worker._calculate_peptide_statistics(
            df_rearanged=df_stk_rearanged,
            chip_label="STK",
        )

        df_ptk_pooled["Type"] = "PTK"
        df_stk_pooled["Type"] = "STK"
        df_peptides = pd.concat([df_ptk_pooled, df_stk_pooled], ignore_index=True)
        df_peptides["ControlCondition"] = control_condition
        df_peptides["Condition"] = condition
        df_peptides["Construct"] = construct

        # Dedup on (Type, ID), not ID alone: the enrichment table carries one row
        # per (peptide, protein) so a peptide repeats with identical statistics,
        # but a peptide printed on BOTH arrays has two legitimate measurements.
        # The waterfall collapses to one row per label itself; the heatmaps split
        # by Type and need both.
        df_peptides_plot = (
            df_peptides.sort_values("p_value", ascending=True)
            .drop_duplicates(subset=["Type", "ID"], keep="first")
            .reset_index(drop=True)
        )

        return {
            "peptide_statistics": df_peptides,
            "control_condition": control_condition,
            "condition": condition,
            "construct": construct,
            "df_ptk_qc_1": df_ptk_qc_1,
            "df_stk_qc_1": df_stk_qc_1,
            "df_ptk_slope": df_ptk_slope,
            "df_stk_slope": df_stk_slope,
            "df_peptides_plot": df_peptides_plot,
        }

    def run_peptide_statistics(self):
        """Run peptide statistics.
        
        Returns:
            dict: One entry per comparison, keyed ``"<control>_vs_<test>"``.
            Each value carries ``peptide_statistics``, ``control_condition``,
            ``condition`` and ``construct``. The KEY identifies the comparison
            (a test condition repeats across keys once a run has more than one
            control); the ``control_condition``/``condition`` pair is what
            downstream stages put into file names and plot titles.
        """
        print("\n" + "=" * 80)
        print("Starting Peptide Statistics Stage...")
        print("=" * 80)

        if self.file_enrichment is None:
            raise ValueError("The enrichment file is missing.")
        if self.df_ptk_input is None or self.df_stk_input is None:
            raise ValueError("Please provide PTK and STK input data.")
        if self.path_file_enrichment_peptides is None:
            raise ValueError("Please provide a peptide enrichment file.")

        print("[1]  Loading enrichment data...")
        df_enrichment = self.file_enrichment
        print("     Enrichment data loaded successfully.\n")

        print("     Checking experimental design and layout...")
        (
            contrasts,
            condition_metadata,
        ) = self._check_experimental_design_and_layout(df_enrichment=df_enrichment)
        print("     Experimental design and layout check completed successfully.\n")

        print("[2]  Loading peptide enrichment data...")
        df_peptide_enrichment = pd.read_csv(self.path_file_enrichment_peptides)
        # BOTH sides, in the same commit: H2B1B_ 27_40 carries its space in the
        # layout AND here, so those two currently agree. Normalising only the
        # export side would break that pair while fixing the other one.
        df_peptide_enrichment = normalize_peptide_id_column(df_peptide_enrichment)
        print("     Peptide enrichment data loaded successfully.")
        print("\n=====================================================================================\n")

        print("[3]  Loading and processing export-image data for PTK and STK...")
        df_ptk_merged, df_stk_merged = self._load_and_merge_peptide_data(
            df_enrichment=df_enrichment
        )

        if self.debugging_print:
            df_merged_peptides = pd.concat([df_ptk_merged, df_stk_merged], ignore_index=True)
            df_merged_peptides_filtered = df_merged_peptides[
                df_merged_peptides["Cycle"] == 94
            ]
            self._write_debug_csv(df_merged_peptides_filtered, "peptide_merged")

        print("     Export-image data for PTK and STK loaded and processed successfully.")
        print("\n=====================================================================================\n")

        print("     Precomputing shared first-pass QC and slope values...")
        df_ptk_qc_1_all, df_stk_qc_1_all = self._filter_high_saturation(
            df_ptk_merged=df_ptk_merged,
            df_stk_merged=df_stk_merged,
            threshold_saturation=0.05,
        )
        df_ptk_slope_all, df_stk_slope_all = self._calculate_change_in_peptides(
            df_ptk_filtered=df_ptk_qc_1_all,
            df_stk_filtered=df_stk_qc_1_all,
        )
        # Second-pass peptide QC, ONCE for the whole run. It must not run per
        # comparison: every variant aggregates across the samples of the frame
        # (KRSA takes the worst signal / R^2 over them, the legacy filters count
        # samples above a threshold), so a per-comparison QC hands each
        # comparison a different peptide population and therefore a different
        # hypergeometric M and kinase panel. Run-wide is also what KRSA does
        # (`global_qc_passed_peptides.txt`).
        if self.qc_mode == self.QC_KRSA:
            df_ptk_qc_2_all = self._peptide_quality_control_krsa(
                df_ptk_slope_all, "PTK"
            )
            df_stk_qc_2_all = self._peptide_quality_control_krsa(
                df_stk_slope_all, "STK"
            )
        else:
            df_ptk_qc_2_all = self._peptide_quality_control_ptk(
                df_ptk_slope=df_ptk_slope_all
            )
            df_stk_qc_2_all = self._peptide_quality_control_stk(
                df_stk_slope=df_stk_slope_all
            )
        n_ptk_qc_2 = (
            int(df_ptk_qc_2_all["ID"].nunique())
            if "ID" in df_ptk_qc_2_all.columns
            else 0
        )
        n_stk_qc_2 = (
            int(df_stk_qc_2_all["ID"].nunique())
            if "ID" in df_stk_qc_2_all.columns
            else 0
        )
        print("     Shared slope precomputation completed successfully.")
        print(
            f"     Run-wide peptide QC ({self.qc_mode}): PTK {n_ptk_qc_2} / "
            f"STK {n_stk_qc_2} peptides passed -- the SAME population is used "
            "by every comparison of this run."
        )
        # Per-array normalisation, likewise ONCE for the whole run (see
        # _estimate_array_normalization); applied inside every comparison.
        array_offsets = self._estimate_array_normalization(
            df_ptk_qc_2_all=df_ptk_qc_2_all,
            df_stk_qc_2_all=df_stk_qc_2_all,
            # The peptides _rearange_peptide_data keeps: its pivot drops every
            # row with a missing annotation field.
            annotated_ids=set(
                df_peptide_enrichment.dropna(
                    subset=["Sequence", "PepProtein_UniprotID", "PepProtein_UniprotName"]
                )["ID"]
            ),
        )
        print("\n=====================================================================================\n")

        all_results = {}
        condition_payloads = []
        print("     Preparing per-comparison peptide statistics jobs...")
        for control_condition, condition in contrasts:
            construct = condition_metadata[condition]["construct"]
            pair = [condition, control_condition]

            df_enrichment_condition = df_enrichment[
                df_enrichment["Test Condition"].isin(pair)
            ].copy()
            df_ptk_qc_1 = df_ptk_qc_1_all[
                df_ptk_qc_1_all["Test Condition"].isin(pair)
            ].copy()
            df_stk_qc_1 = df_stk_qc_1_all[
                df_stk_qc_1_all["Test Condition"].isin(pair)
            ].copy()
            df_ptk_slope = df_ptk_slope_all[
                df_ptk_slope_all["Test Condition"].isin(pair)
            ].copy()
            df_stk_slope = df_stk_slope_all[
                df_stk_slope_all["Test Condition"].isin(pair)
            ].copy()
            # Row subset of the RUN-WIDE QC result. Because every QC variant
            # returns a row subset of its input, subsetting after the QC is
            # identical to subsetting before it -- except that the peptide set
            # is now the run's, not this comparison's.
            df_ptk_qc_2 = df_ptk_qc_2_all[
                df_ptk_qc_2_all["Test Condition"].isin(pair)
            ].copy()
            df_stk_qc_2 = df_stk_qc_2_all[
                df_stk_qc_2_all["Test Condition"].isin(pair)
            ].copy()
            condition_payloads.append(
                {
                    # comparison_key identifies the job; with several controls a
                    # test condition appears in more than one comparison, so the
                    # test label alone is no longer unique. Every user-facing
                    # label and file name still comes from the control/condition
                    # pair, not from this key.
                    "comparison_key": f"{control_condition}_vs_{condition}",
                    "control_condition": control_condition,
                    "condition": condition,
                    "construct": construct,
                    "df_enrichment_condition": df_enrichment_condition,
                    "df_ptk_qc_1": df_ptk_qc_1,
                    "df_stk_qc_1": df_stk_qc_1,
                    "df_ptk_slope": df_ptk_slope,
                    "df_stk_slope": df_stk_slope,
                    "df_ptk_qc_2": df_ptk_qc_2,
                    "df_stk_qc_2": df_stk_qc_2,
                }
            )

        print("     Running per-comparison Stage 1 computations in parallel...")
        condition_results_map = {}
        max_workers = min(len(condition_payloads), 4) if condition_payloads else 1
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._compute_condition_result,
                        condition=payload["condition"],
                        control_condition=payload["control_condition"],
                        construct=payload["construct"],
                        df_enrichment_condition=payload["df_enrichment_condition"],
                        df_peptide_enrichment=df_peptide_enrichment,
                        df_ptk_qc_1=payload["df_ptk_qc_1"],
                        df_stk_qc_1=payload["df_stk_qc_1"],
                        df_ptk_slope=payload["df_ptk_slope"],
                        df_stk_slope=payload["df_stk_slope"],
                        df_ptk_qc_2=payload["df_ptk_qc_2"],
                        df_stk_qc_2=payload["df_stk_qc_2"],
                        array_offsets=array_offsets,
                    )
                    for payload in condition_payloads
                ]
                # Keyed from the payload, not from the result: the test label
                # alone repeats across comparisons once there is more than one
                # control.
                for payload, future in zip(condition_payloads, futures):
                    condition_results_map[payload["comparison_key"]] = future.result()
        else:
            for payload in condition_payloads:
                result = self._compute_condition_result(
                    condition=payload["condition"],
                    control_condition=payload["control_condition"],
                    construct=payload["construct"],
                    df_enrichment_condition=payload["df_enrichment_condition"],
                    df_peptide_enrichment=df_peptide_enrichment,
                    df_ptk_qc_1=payload["df_ptk_qc_1"],
                    df_stk_qc_1=payload["df_stk_qc_1"],
                    df_ptk_slope=payload["df_ptk_slope"],
                    df_stk_slope=payload["df_stk_slope"],
                    df_ptk_qc_2=payload["df_ptk_qc_2"],
                    df_stk_qc_2=payload["df_stk_qc_2"],
                    array_offsets=array_offsets,
                )
                condition_results_map[payload["comparison_key"]] = result

        print("     Iterating through comparisons:")
        for payload in condition_payloads:
            condition = payload["condition"]
            control_condition = payload["control_condition"]
            construct = payload["construct"]
            result = condition_results_map[payload["comparison_key"]]

            print("\n=====================================================================================\n")
            print(
                f"Comparing condition: {control_condition} vs. {condition} "
                f"(Construct: {construct})"
            )
            print("\n=====================================================================================\n")

            print("[4-8] Condition-specific QC, slope reuse, transformation, and peptide statistics completed.")
            if self.debugging_print:
                df_qc1_peptides = pd.concat(
                    [result["df_ptk_qc_1"], result["df_stk_qc_1"]],
                    ignore_index=True,
                )
                df_qc1_peptides_filtered = df_qc1_peptides[df_qc1_peptides["Cycle"] == 94]
                self._write_debug_csv(
                    df_qc1_peptides_filtered,
                    f"peptide_qc1_{control_condition}_{condition}",
                )

                df_slope_peptides = pd.concat(
                    [result["df_ptk_slope"], result["df_stk_slope"]],
                    ignore_index=True,
                )
                self._write_debug_csv(
                    df_slope_peptides,
                    f"peptide_slope_{control_condition}_{condition}",
                )

                df_peptides_sorted = result["peptide_statistics"].sort_values(
                    by="p_value",
                    ascending=True,
                )
                self._write_debug_csv(
                    df_peptides_sorted,
                    f"peptide_statistics_{control_condition}_{condition}",
                )
            print("     Calculation of peptide statistics completed successfully.")
            print("\n=====================================================================================\n")

            if self.waterfall_plot:
                print("[9]  Plotting peptide waterfall plot...")
                self._plot_waterfall(
                    df_peptides=result["df_peptides_plot"],
                    output_path=self.waterfall_plot_output,
                    control=control_condition,
                    condition=condition,
                )
            if self.heatmap_plot:
                print("[10] Plotting peptide heatmap...")
                self._plot_heatmap(
                    df_peptides=result["df_peptides_plot"],
                    output_path=self.heatmap_plot_output,
                    control=control_condition,
                    condition=condition,
                )

            all_results[payload["comparison_key"]] = {
                "peptide_statistics": result["peptide_statistics"],
                "control_condition": result["control_condition"],
                "condition": result["condition"],
                "construct": result["construct"],
            }

        self.peptide_statistics_output = all_results

        print("\n" + "=" * 80)
        print("Peptide Statistics Stage Completed")
        print("=" * 80 + "\n")

        return all_results
