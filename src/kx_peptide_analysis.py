"""Peptide-level statistical analysis for the pyKinaXe downstream workflow.

This module implements stage 1 of the UKA/KPEA analysis stack. It takes the
processed PTK/STK chip outputs produced by ``kx_image_processor.py`` together
with the enriched experimental-design table from ``kx_data_enricher.py`` and
turns them into condition-wise peptide statistics.

The main responsibilities here are:

- reorganizing chip-level PTK/STK outputs into analysis-ready matrices
- estimating peptide-level response metrics across cycles / exposures
- running limma-style moderated statistics via ``inmoose`` when enabled
- preparing peptide-level output tables for the later kinase-analysis stage
- generating peptide heatmaps and volcano plots when requested

The core public class is ``PeptideStatistics``. Its output is consumed by
``kx_upstream_kinase_analysis.py``.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from inmoose.limma import eBayes, lmFit, squeezeVar
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind

from config.analysis_modules import PEPTIDE_ANALYSIS_DEFAULTS
from kx_plot_results import HeatmapPlot_Peptides, VolcanoPlot

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - optional GUI dependency
    tk = None


class DegenerateLimmaModerationError(RuntimeError):
    """Raised when limma moderation collapses to a single shared variance."""

    def __init__(self, *, df_prior, var_prior):
        """Initialize the DegenerateLimmaModerationError instance.
        
        Args:
            df_prior: Input pandas DataFrame containing prior.
            var_prior: Var prior processed by this function.
        
        Returns:
            None: Constructors initialize object state in place.
        """
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

    def __init__(
        self,
        file_enrichment=None,
        df_ptk=None,
        df_stk=None,
        path_file_enrichment_peptides=PEPTIDE_ANALYSIS_DEFAULTS[
            "path_file_enrichment_peptides"
        ],
        significance_level_peptides=0.05,
        volcano_plot=False,
        volcano_plot_GUI=False,
        volcano_plot_output=None,
        heatmap_plot=False,
        heatmap_plot_GUI=False,
        heatmap_plot_output=None,
        path_output_peptide_statistic=None,
        use_limma=True,
        debugging_print=True,
        log2_slope_mode=PEPTIDE_ANALYSIS_DEFAULTS["default_log2_slope_mode"],
    ):
        """Initialize the PeptideStatistics instance.
        
        Args:
            file_enrichment: File enrichment processed by this function.
            df_ptk: Input pandas DataFrame containing PTK.
            df_stk: Input pandas DataFrame containing STK.
            path_file_enrichment_peptides: Path to the file enrichment peptides.
            significance_level_peptides: Significance level peptides processed by this function.
            volcano_plot: Volcano plot processed by this function.
            volcano_plot_GUI: Volcano plot GUI processed by this function.
            volcano_plot_output: Volcano plot output processed by this function.
            heatmap_plot: Heatmap plot processed by this function.
            heatmap_plot_GUI: Heatmap plot GUI processed by this function.
            heatmap_plot_output: Heatmap plot output processed by this function.
            path_output_peptide_statistic: Path to the output peptide statistic.
            use_limma: Boolean flag controlling whether to use limma.
            debugging_print: Whether to print additional debug information.
            log2_slope_mode: Log2 slope mode processed by this function.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        self.file_enrichment = file_enrichment
        self.df_ptk_input = df_ptk
        self.df_stk_input = df_stk

        self.path_file_enrichment_peptides = Path(path_file_enrichment_peptides)
        self.significance_level_peptides = significance_level_peptides
        self.volcano_plot = volcano_plot
        self.volcano_plot_GUI = volcano_plot_GUI
        self.volcano_plot_output = Path(volcano_plot_output) if volcano_plot_output is not None else None
        self.heatmap_plot = heatmap_plot
        self.heatmap_plot_GUI = heatmap_plot_GUI
        self.heatmap_plot_output = Path(heatmap_plot_output) if heatmap_plot_output is not None else None
        self.path_output_peptide_statistic = (
            Path(path_output_peptide_statistic)
            if path_output_peptide_statistic is not None
            else None
        )
        self.use_limma = use_limma
        self.debugging_print = debugging_print
        self.log2_slope_mode = str(log2_slope_mode).lower()

        if self.log2_slope_mode not in self.ALLOWED_LOG2_SLOPE_MODES:
            raise ValueError(
                f"Unknown log2_slope_mode '{log2_slope_mode}'. "
                "Use 'pamgene_zero' or 'epsilon_floor'."
            )

        self.peptide_row_name = PEPTIDE_ANALYSIS_DEFAULTS["peptide_row_name"]
        self.y_axis_peptides = PEPTIDE_ANALYSIS_DEFAULTS["y_axis_peptides"]

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
    def _extract_scalar(value):
        """Extract scalar.
        
        Args:
            value: Input value processed by this helper.
        
        Returns:
            object: Extracted scalar.
        """
        array_value = np.asarray(value)
        if array_value.ndim == 0:
            return float(array_value)
        return float(array_value.flat[0])

    @staticmethod
    def _normalize_replicate_value(value):
        """Normalize replicate value.
        
        Args:
            value: Input value processed by this helper.
        
        Returns:
            object: Normalized replicate value.
        """
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
        """Build sample key.
        
        Args:
            sample_name: Sample name processed by this function.
            biological_replicate: Biological replicate processed by this function.
            technical_replicate: Technical replicate processed by this function.
        
        Returns:
            object: Constructed sample key.
        """
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
        """Build requested group matrix.
        
        Args:
            df_rearanged: Input pandas DataFrame containing rearanged.
            requested_samples: Requested samples processed by this function.
        
        Returns:
            object: Constructed requested group matrix.
        """
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
        """Format group sample label.
        
        Args:
            sample_key: Sample key processed by this function.
            group_prefix: Group prefix processed by this function.
        
        Returns:
            object: Formatted group sample label.
        """
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

    def _write_debug_text(self, lines, suffix):
        """Write debug text.
        
        Args:
            lines: Lines processed by this function.
            suffix: Suffix processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self.path_output_peptide_statistic is None:
            return

        output_path = Path(f"{self.path_output_peptide_statistic}_{suffix}.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(str(line) for line in lines).rstrip()
        output_path.write_text(f"{text}\n" if text else "", encoding="utf-8")

    @staticmethod
    def _calculate_rowwise_sd(matrix):
        """Calculate rowwise sd.
        
        Args:
            matrix: Matrix processed by this function.
        
        Returns:
            object: Calculated rowwise sd.
        """
        matrix = np.asarray(matrix, dtype=float)
        row_sd = np.full(matrix.shape[0], np.nan, dtype=float)
        valid_rows = np.sum(~np.isnan(matrix), axis=1) >= 2
        if np.any(valid_rows):
            row_sd[valid_rows] = np.nanstd(matrix[valid_rows], axis=1, ddof=1)
        return row_sd

    @staticmethod
    def _run_rowwise_t_tests(control_values, treatment_values, eligible_mask):
        """Run rowwise t tests.
        
        Args:
            control_values: Control values processed by this function.
            treatment_values: Treatment values processed by this function.
            eligible_mask: Eligible mask processed by this function.
        
        Returns:
            tuple: Run result for rowwise t tests.
        """
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
        """Report limma exclusions.
        
        Args:
            df_rearanged: Input pandas DataFrame containing rearanged.
            chip_label: Chip label processed by this function.
            complete_case_mask: Complete case mask processed by this function.
            n_control_values: Number of control values used by this function.
            n_treatment_values: Number of treatment values used by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
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
        self._write_debug_csv(diagnostics_df, suffix_base)
        self._write_debug_text(summary_lines, f"{suffix_base}_summary")

    def _report_degenerate_limma(
        self,
        *,
        chip_label,
        df_prior,
        var_prior,
        complete_rows,
        total_rows,
    ):
        """Report degenerate limma.
        
        Args:
            chip_label: Chip label processed by this function.
            df_prior: Input pandas DataFrame containing prior.
            var_prior: Var prior processed by this function.
            complete_rows: Complete rows processed by this function.
            total_rows: Total rows processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
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
        self._write_debug_text(summary_lines, f"{suffix_base}_summary")

    def _check_experimental_design_and_layout(self, df_enrichment):
        """Check experimental design and layout.
        
        Args:
            df_enrichment: Input pandas DataFrame containing enrichment.
        
        Returns:
            tuple: Check result for experimental design and layout.
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

        control_keywords = [
            "control",
            "ctrl",
            "untreated",
            "baseline",
            "mock",
            "dmso",
            "vehicle",
        ]
        control_condition = None
        for condition in unique_conditions:
            condition_lower = str(condition).lower()
            if any(keyword in condition_lower for keyword in control_keywords):
                control_condition = condition
                break

        if control_condition is None:
            control_condition = sorted(unique_conditions)[0]
            self._dprint(
                "     WARNING: No explicit control condition found. "
                f"Using '{control_condition}' as control condition."
            )

        test_conditions_array = [
            cond for cond in unique_conditions if cond != control_condition
        ]
        if not test_conditions_array:
            raise ValueError(
                "No test conditions found. At least one test condition is required."
            )

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

        self._dprint("     Experimental design detected:")
        self._dprint(f"       - Control condition: '{control_condition}'")
        for test_condition in test_conditions_array:
            self._dprint(f"       - Test condition: '{test_condition}'")

        self.control_condition = control_condition
        self.test_conditions_array = test_conditions_array
        self.condition_metadata = condition_metadata
        return control_condition, test_conditions_array, condition_metadata

    def _load_and_merge_peptide_data(self, df_enrichment):
        """Load and merge peptide data.
        
        Args:
            df_enrichment: Input pandas DataFrame containing enrichment.
        
        Returns:
            tuple: Loaded and merge peptide data.
        """
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
        """Filter high saturation.
        
        Args:
            df_ptk_merged: Input pandas DataFrame containing PTK merged.
            df_stk_merged: Input pandas DataFrame containing STK merged.
            threshold_saturation: Threshold value used to filter, classify, or flag results.
        
        Returns:
            tuple: Filtered high saturation.
        """
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
        """Calculate slope.
        
        Args:
            row: Row processed by this function.
            exposure_cols: Exposure cols processed by this function.
            exposure_times: Exposure times processed by this function.
        
        Returns:
            object: Calculated slope.
        """
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
        """Rename exposure columns.
        
        Args:
            df_pivoted: Input pandas DataFrame containing pivoted.
        
        Returns:
            object: Renamed exposure columns.
        """
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
        """Extract exposure columns and times.
        
        Args:
            df_pivoted: Input pandas DataFrame containing pivoted.
        
        Returns:
            tuple: Extracted exposure columns and times.
        """
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
        """Calculate slopes from pivoted.
        
        Args:
            df_pivoted: Input pandas DataFrame containing pivoted.
            exposure_cols: Exposure cols processed by this function.
            exposure_times: Exposure times processed by this function.
        
        Returns:
            object: Calculated slopes from pivoted.
        """
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
        """Calculate change in peptides.
        
        Args:
            df_ptk_filtered: Input pandas DataFrame containing PTK filtered.
            df_stk_filtered: Input pandas DataFrame containing STK filtered.
        
        Returns:
            tuple: Calculated change in peptides.
        """
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

    def _peptide_quality_control_stk(self, df_stk_slope):
        """Return peptide quality control STK.
        
        Args:
            df_stk_slope: Input pandas DataFrame containing STK slope.
        
        Returns:
            object: Peptide quality control STK.
        """
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
        """Return peptide quality control PTK.
        
        Args:
            df_ptk_slope: Input pandas DataFrame containing PTK slope.
        
        Returns:
            object: Peptide quality control PTK.
        """
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
        """Return log2 transform slope.
        
        Args:
            df_ptk_slope: Input pandas DataFrame containing PTK slope.
            df_stk_slope: Input pandas DataFrame containing STK slope.
        
        Returns:
            tuple: Log2 transform slope.
        """
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
        else:
            df_ptk_slope["slope_log2"] = np.where(
                df_ptk_slope["slope"] > log_threshold,
                np.log2(df_ptk_slope["slope"].clip(lower=log_threshold)),
                0.0,
            )
            df_stk_slope["slope_log2"] = np.where(
                df_stk_slope["slope"] > log_threshold,
                np.log2(df_stk_slope["slope"].clip(lower=log_threshold)),
                0.0,
            )

        return df_ptk_slope, df_stk_slope

    def _rearange_peptide_data(
        self,
        df_log2_ptk,
        df_log2_stk,
        df_enrichment,
        df_peptide_enrichment,
    ):
        """Rearrange peptide data.
        
        Args:
            df_log2_ptk: Input pandas DataFrame containing log2 PTK.
            df_log2_stk: Input pandas DataFrame containing log2 STK.
            df_enrichment: Input pandas DataFrame containing enrichment.
            df_peptide_enrichment: Input pandas DataFrame containing peptide enrichment.
        
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

        sample_to_biorep = {}
        for _, sample_info in sample_condition_map.iterrows():
            sample_key = sample_info[sample_key_col]
            sample_to_biorep[sample_key] = {
                "sample_name": sample_info["Sample name"],
                "bio_rep": sample_info["Biological Replicate"],
                "tech_rep": sample_info["Technical Replicate"],
                "condition": sample_info["Test Condition"],
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
        """Calculate peptide statistics sub.
        
        Args:
            pControl: PControl processed by this function.
            pTreat: PTreat processed by this function.
        
        Returns:
            tuple: Calculated peptide statistics sub.
        """
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

    def _limma_t_test(self, df_rearanged, pControl=None, pTreat=None):
        """Return limma t test.
        
        Args:
            df_rearanged: Input pandas DataFrame containing rearanged.
            pControl: PControl processed by this function.
            pTreat: PTreat processed by this function.
        
        Returns:
            tuple: Limma t test.
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
        """Get raw sample columns.
        
        Args:
            df_pooled: Input pandas DataFrame containing pooled.
        
        Returns:
            tuple: Requested raw sample columns.
        """
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

    def _calculate_peptide_statistics(self, df_rearanged, chip_label="chip"):
        # The enrichment-derived control/treatment groups may include samples
        # that have no surviving rows on this chip (e.g. all rows dropped by
        # chip-specific QC, or no Barcode/Row match). Restrict to columns that
        # are actually present in this chip's rearranged frame to avoid
        # KeyError on missing samples.
        """Calculate peptide statistics.
        
        Args:
            df_rearanged: Input pandas DataFrame containing rearanged.
            chip_label: Chip label processed by this function.
        
        Returns:
            object: Calculated peptide statistics.
        """
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
                    (
                        t_stats_complete,
                        p_values_complete,
                        s2_post,
                        df_total,
                        _logFC,
                        _s2_prior,
                        _df_prior,
                    ) = self._limma_t_test(
                        df_rearanged=df_rearanged.loc[complete_case_mask].copy(),
                        pControl=control_group,
                        pTreat=treatment_group,
                    )
                    t_stats[complete_case_mask] = np.asarray(
                        t_stats_complete,
                        dtype=float,
                    )
                    p_values[complete_case_mask] = np.asarray(
                        p_values_complete,
                        dtype=float,
                    )
                    s2_moderated[complete_case_mask] = np.asarray(s2_post, dtype=float)
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

        df_pooled["zscore_scale_source"] = zscore_scale_source
        for i, col in enumerate(sample_cols):
            df_pooled[f"{col}_zscore"] = scaled_values[:, i]

        insufficient_mask = df_pooled["statistics_method"].isin(
            ["insufficient_replicates", "missing_group_values"]
        )
        if np.any(insufficient_mask):
            insufficient_rows = df_pooled.loc[insufficient_mask].copy()
            self._dprint(
                f"     {chip_label}: Filtering out {len(insufficient_rows)} peptide(s) "
                "with insufficient values before downstream analysis."
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

        return df_pooled

    def _plot_volcano(self, df_peptides, output_path=None, control=None, condition=None):
        """Plot volcano.
        
        Args:
            df_peptides: Input pandas DataFrame containing peptides.
            output_path: Path to the output.
            control: Control processed by this function.
            condition: Condition processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if df_peptides is None or df_peptides.empty:
            self._dprint("     No peptide data found for volcano plot.")
            return None

        df_peptides = df_peptides.rename(columns={"peptide_change": "delta"})
        save_path = (
            str(Path(output_path) / f"peptides_volcano_plot_{control}_{condition}.png")
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
                data=df_peptides,
                y_axis=self.y_axis_peptides,
                x_label="Peptide Change (log2)",
                save_path=save_path,
                gui=True,
            )
            root.mainloop()
        else:
            VolcanoPlot(
                debugging_print=self.debugging_print,
                data=df_peptides,
                y_axis=self.y_axis_peptides,
                x_label="Peptide Change (log2)",
                save_path=save_path,
                gui=False,
            )
        return None

    def _plot_heatmap(self, df_peptides, output_path=None, control=None, condition=None):
        """Plot heatmap.
        
        Args:
            df_peptides: Input pandas DataFrame containing peptides.
            output_path: Path to the output.
            control: Control processed by this function.
            condition: Condition processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if df_peptides is None or df_peptides.empty:
            self._dprint("     No peptide data available for heatmap plot.")
            return None

        save_path = (
            str(Path(output_path) / f"peptides_heatmap_{control}_{condition}.png")
            if output_path is not None
            else None
        )

        try:
            if self.heatmap_plot_GUI:
                if tk is None:
                    raise RuntimeError(
                        "Tkinter GUI plotting is not available in this environment. "
                        "Set heatmap_plot_GUI=False for headless execution."
                    )
                root = tk.Tk()
                HeatmapPlot_Peptides(
                    root=root,
                    debugging_print=self.debugging_print,
                    data=df_peptides,
                    significance_threshold=self.significance_level_peptides,
                    cmap="RdYlBu_r",
                    save_path=save_path,
                    title="Peptide Expression",
                    gui=True,
                )
                root.mainloop()
            else:
                HeatmapPlot_Peptides(
                    debugging_print=self.debugging_print,
                    data=df_peptides,
                    significance_threshold=self.significance_level_peptides,
                    cmap="RdYlBu_r",
                    save_path=save_path,
                    title="Peptide Expression",
                    gui=False,
                )
        except ValueError as exc:
            self._dprint(f"     Peptide heatmap skipped: {exc}")
        return None

    def _build_condition_worker(self):
        """Build condition worker.
        
        Args:
            None.
        
        Returns:
            object: Constructed condition worker.
        """
        return PeptideStatistics(
            path_file_enrichment_peptides=self.path_file_enrichment_peptides,
            significance_level_peptides=self.significance_level_peptides,
            volcano_plot=False,
            volcano_plot_GUI=False,
            volcano_plot_output=self.volcano_plot_output,
            heatmap_plot=False,
            heatmap_plot_GUI=False,
            heatmap_plot_output=self.heatmap_plot_output,
            path_output_peptide_statistic=self.path_output_peptide_statistic,
            use_limma=self.use_limma,
            debugging_print=self.debugging_print,
            log2_slope_mode=self.log2_slope_mode,
        )

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
    ):
        """Compute condition result.
        
        Args:
            condition: Condition processed by this function.
            control_condition: Control condition processed by this function.
            construct: Construct processed by this function.
            df_enrichment_condition: Input pandas DataFrame containing enrichment condition.
            df_peptide_enrichment: Input pandas DataFrame containing peptide enrichment.
            df_ptk_qc_1: Input pandas DataFrame containing PTK qc 1.
            df_stk_qc_1: Input pandas DataFrame containing STK qc 1.
            df_ptk_slope: Input pandas DataFrame containing PTK slope.
            df_stk_slope: Input pandas DataFrame containing STK slope.
        
        Returns:
            dict: Computed condition result.
        """
        worker = self._build_condition_worker()
        worker.control_condition = control_condition
        worker.current_condition = condition

        df_ptk_qc_2 = worker._peptide_quality_control_ptk(df_ptk_slope=df_ptk_slope)
        df_stk_qc_2 = worker._peptide_quality_control_stk(df_stk_slope=df_stk_slope)

        df_ptk_log2, df_stk_log2 = worker._log2_transform_slope(
            df_ptk_slope=df_ptk_qc_2,
            df_stk_slope=df_stk_qc_2,
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

        df_peptides_plot = (
            df_peptides.sort_values("p_value", ascending=True)
            .drop_duplicates(subset="ID", keep="first")
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
        
        Args:
            None.
        
        Returns:
            object: Run result for peptide statistics.
        """
        print("=====================================================================================")
        print("=======================   Starting Peptide Statistics Stage   =======================")
        print("=====================================================================================\n")

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
            control_condition,
            test_conditions_array,
            condition_metadata,
        ) = self._check_experimental_design_and_layout(df_enrichment=df_enrichment)
        print("     Experimental design and layout check completed successfully.\n")

        print("[2]  Loading peptide enrichment data...")
        df_peptide_enrichment = pd.read_csv(self.path_file_enrichment_peptides)
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
        print("     Shared slope precomputation completed successfully.")
        print("\n=====================================================================================\n")

        all_results = {}
        condition_payloads = []
        print("     Preparing per-condition peptide statistics jobs...")
        for condition in test_conditions_array:
            construct = condition_metadata[condition]["construct"]

            df_enrichment_condition = df_enrichment[
                df_enrichment["Test Condition"].isin([condition, control_condition])
            ].copy()
            df_ptk_qc_1 = df_ptk_qc_1_all[
                df_ptk_qc_1_all["Test Condition"].isin([condition, control_condition])
            ].copy()
            df_stk_qc_1 = df_stk_qc_1_all[
                df_stk_qc_1_all["Test Condition"].isin([condition, control_condition])
            ].copy()
            df_ptk_slope = df_ptk_slope_all[
                df_ptk_slope_all["Test Condition"].isin([condition, control_condition])
            ].copy()
            df_stk_slope = df_stk_slope_all[
                df_stk_slope_all["Test Condition"].isin([condition, control_condition])
            ].copy()
            condition_payloads.append(
                {
                    "condition": condition,
                    "construct": construct,
                    "df_enrichment_condition": df_enrichment_condition,
                    "df_ptk_qc_1": df_ptk_qc_1,
                    "df_stk_qc_1": df_stk_qc_1,
                    "df_ptk_slope": df_ptk_slope,
                    "df_stk_slope": df_stk_slope,
                }
            )

        print("     Running per-condition Stage 1 computations in parallel...")
        condition_results_map = {}
        max_workers = min(len(condition_payloads), 4) if condition_payloads else 1
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._compute_condition_result,
                        condition=payload["condition"],
                        control_condition=control_condition,
                        construct=payload["construct"],
                        df_enrichment_condition=payload["df_enrichment_condition"],
                        df_peptide_enrichment=df_peptide_enrichment,
                        df_ptk_qc_1=payload["df_ptk_qc_1"],
                        df_stk_qc_1=payload["df_stk_qc_1"],
                        df_ptk_slope=payload["df_ptk_slope"],
                        df_stk_slope=payload["df_stk_slope"],
                    )
                    for payload in condition_payloads
                ]
                for future in futures:
                    result = future.result()
                    condition_results_map[result["condition"]] = result
        else:
            for payload in condition_payloads:
                result = self._compute_condition_result(
                    condition=payload["condition"],
                    control_condition=control_condition,
                    construct=payload["construct"],
                    df_enrichment_condition=payload["df_enrichment_condition"],
                    df_peptide_enrichment=df_peptide_enrichment,
                    df_ptk_qc_1=payload["df_ptk_qc_1"],
                    df_stk_qc_1=payload["df_stk_qc_1"],
                    df_ptk_slope=payload["df_ptk_slope"],
                    df_stk_slope=payload["df_stk_slope"],
                )
                condition_results_map[result["condition"]] = result

        print("     Iterating through test conditions:")
        for payload in condition_payloads:
            condition = payload["condition"]
            construct = payload["construct"]
            result = condition_results_map[condition]

            print("\n=====================================================================================\n")
            print(
                f"Comparing condition: Control vs. {condition} (Construct: {construct})"
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

            if self.volcano_plot:
                print("[9]  Plotting peptide volcano plot...")
                self._plot_volcano(
                    df_peptides=result["df_peptides_plot"],
                    output_path=self.volcano_plot_output,
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

            all_results[condition] = {
                "peptide_statistics": result["peptide_statistics"],
                "control_condition": result["control_condition"],
                "condition": result["condition"],
                "construct": result["construct"],
            }

        self.peptide_statistics_output = all_results

        print("\n=====================================================================================")
        print("====================   Peptide Statistics Stage Completed   =========================")
        print("=====================================================================================\n")

        return all_results
