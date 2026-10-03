"""Compare kinase and pathway results against a reference result set.

Benchmarking_Heatmap computes the overlap of significant kinases between two
result sets, compares their pathway enrichment (KEGG, Reactome, WikiPathways)
and draws a side-by-side heatmap of the shared kinases.
"""

import ast
from collections import defaultdict
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from pathlib import Path

import yaml

from kx_pathway_enrichment_analysis import profile_kinases
from kx_plot_results import (
    INDIVIDUAL_COLOR_SCALE,
    VennDiagramPlot,
    _resolve_color_scale,
    _data_limits,
)

DEFAULT_CONFIG_PATH = "config/heatmap_plot_config.yaml"


def _resolve_config_path(path):
    path = Path(path)
    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [repo_root / path, Path.cwd() / path, path]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


class Benchmarking_Heatmap:
    """
    Benchmarking comparison of kinase activity profiles from two sources.

    Parameters
    ----------
    input_results : pd.DataFrame
        DataFrame with columns ['Kinase', 'Activity'] from pyKinaXe.
    input_benchmarking : pd.DataFrame
        DataFrame with columns ['Kinase', 'Activity'] from the reference study.
    label_results : str
        Display label for the results dataset (y-axis of heatmap).
    label_benchmarking : str
        Display label for the benchmarking dataset (y-axis of heatmap).
    save_dir : str or Path, optional
        Directory to save outputs. If None, nothing is saved.
    heatmap : bool
        Whether to generate the comparison heatmap.
    dpi : int
        Resolution for saved figures.
    image_format : str
        File format for saved figures ('png', 'pdf', 'svg').
    significance_level_pathways : float
        g:SCS threshold for pathway enrichment.
    pathway_sources : list of str
        g:Profiler source identifiers to query (default: KEGG, REAC, WP).
    venn_plot : bool
        Whether to generate pyKinaXe-vs-reference overlap diagrams.
    venn_plot_output : str or Path, optional
        Directory to save overlap diagrams. Defaults to save_dir/benchmarking_venn_plots.
    venn_plot_tables : bool
        Whether to save Venn/overlap membership tables next to the plots.
    debugging_print : bool
        If True, print verbose diagnostics.
    input_results_all : pd.DataFrame, optional
        Full kinase-analysis table from the results workflow (every scored
        kinase, not only the called ones). If provided, benchmarking reports
        whether reference kinases appear anywhere in the analysis. If omitted,
        that question CANNOT be answered: the reported/significant table is
        used as a stand-in, a UserWarning is emitted, and the kinase trace
        omits the 'in_<label>_analysis' / 'significant_in_<label>_analysis'
        columns instead of asserting a panel coverage it cannot know.
    activity_label : str
        Label for the heatmap colour scale.
    """

    # Class-level defaults so the flag is readable even on partially
    # constructed instances (and never silently missing).
    pathway_query_failed = False
    pathway_query_errors = None

    def __init__(
        self,
        input_results: pd.DataFrame,
        input_benchmarking: pd.DataFrame,
        label_results: str = "pyKinaXe",
        label_benchmarking: str = "Reference",
        save_dir: str = None,
        heatmap: bool = True,
        dpi: int = 300,
        image_format: str = "png",
        significance_level_pathways: float = 0.05,
        pathway_sources: list = None,
        venn_plot: bool = True,
        venn_plot_output: str = None,
        venn_plot_tables: bool = True,
        debugging_print: bool = False,
        input_results_all: pd.DataFrame = None,
        activity_label: str = "Activity change",
    ):
        # config
        """Set up the comparison of a kinase result table against a reference table.
        
        Args:
            debugging_print (bool): Whether to print additional debug information.
        """
        cfg = self._load_config()

        cell = cfg.get("cell", {})
        self.CELL_HEIGHT = cell.get("height", 0.6)
        self.CELL_WIDTH = cell.get("width", 0.45)
        self.CELL_FONTSIZE = cell.get("fontsize", 9)
        self.CELL_LINEWIDTH = cell.get("linewidth", 0.5)
        self.CELL_LINECOLOR = cell.get("linecolor", "black")

        margins = cfg.get("margins", {})
        self.MARGIN_LEFT = margins.get("left", 2.0)
        self.MARGIN_RIGHT = margins.get("right", 2.5)
        self.MARGIN_TOP = margins.get("top", 2.0)
        self.MARGIN_BOTTOM = margins.get("bottom", 1.0)

        plot = cfg.get("plot", {})
        self.CMAP = plot.get("cmap", "RdBu_r")
        self.COLOR_SCALE = _resolve_color_scale(
            plot.get("color_scale", INDIVIDUAL_COLOR_SCALE)
        )
        self.FIXED_VALUE_STEPS = tuple(
            sorted(
                float(step)
                for step in cfg.get("fixed_scale", {}).get(
                    "benchmarking_value_steps", (1.0, 2.0)
                )
            )
        )
        self.TITLE_PAD = plot.get("title_pad", 20)

        cbar = cfg.get("colorbar", {})
        self.CBAR_LABEL_FONTSIZE = cbar.get("label_fontsize", 12)
        self.CBAR_WIDTH_FACTOR = cbar.get("width_factor", 2)
        self.CBAR_OFFSET_X = cbar.get("offset_x", 0.05)

        axes = cfg.get("axes", {})
        self.AXES_LABEL_FONTSIZE = axes.get("label_fontsize", 12)
        self.TITLE_FONTSIZE = axes.get("title_fontsize", 14)
        self.X_TICK_ROTATION = axes.get("x_tick_rotation", 45)
        self.X_TICK_FONTSIZE = axes.get("x_tick_fontsize", 8)
        self.Y_TICK_FONTSIZE = axes.get("y_tick_fontsize", 10)

        # parameters
        self.label_results = label_results
        self.label_benchmarking = label_benchmarking
        self.save_dir = Path(save_dir) if save_dir else None
        self.do_heatmap = heatmap
        self.dpi = dpi
        self.image_format = image_format.lower()
        self.significance_level_pathways = significance_level_pathways
        self.pathway_sources = (
            pathway_sources if pathway_sources is not None else ["KEGG", "REAC", "WP"]
        )
        self.do_venn_plot = venn_plot
        if venn_plot_output is not None:
            self.venn_plot_output = Path(venn_plot_output)
        elif self.save_dir is not None:
            self.venn_plot_output = self.save_dir / "benchmarking_venn_plots"
        else:
            self.venn_plot_output = None
        self.venn_plot_tables = venn_plot_tables
        self.debugging_print = debugging_print
        self.activity_label = activity_label

        # pathway query status
        self.pathway_query_failed = False
        self.pathway_query_errors = {}

        # validate & store inputs
        # Both sides must annotate the array, or the key degrades to the name.
        self.key_on_array = self._decide_array_keying(
            input_results, input_benchmarking,
            self.label_results, self.label_benchmarking,
        )
        self.df_results, self.dropped_results = self._validate_input(
            input_results, "input_results", use_array=self.key_on_array
        )
        self.df_bench, self.dropped_bench = self._validate_input(
            input_benchmarking, "input_benchmarking", use_array=self.key_on_array
        )

        # sets
        self.set_results = set(self.df_results[self.KEY_COL])
        self.set_bench = set(self.df_bench[self.KEY_COL])

        self._have_full_results = input_results_all is not None
        if not self._have_full_results:
            warnings.warn(
                "Benchmarking_Heatmap was constructed without 'input_results_all'. "
                "The reported (significant) table is used as a stand-in, so panel "
                "coverage cannot be distinguished from a missed call threshold; the "
                "kinase trace therefore omits the "
                f"'in_{label_results}_analysis' / "
                f"'significant_in_{label_results}_analysis' columns.",
                UserWarning,
                stacklevel=2,
            )
        self.df_results_all = self._validate_full_results_input(input_results_all)

        self.set_results_all = set(self.df_results_all[self.KEY_COL])

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # public API
    # ================================================================

    def run(self) -> dict:
        """Execute the full benchmarking pipeline.
        
        Returns
        -------
        dict with keys:
            'kinase_overlap'    : overlap statistics dict
            'pathway_enrichment': dict  source -> overlap statistics dict
            'heatmap_fig'       : matplotlib Figure or None
        """
        results = {}

        # 1. kinase overlap
        kinase_stats = self._compare_kinases()
        results["kinase_overlap"] = kinase_stats

        kinase_trace = self._trace_benchmarking_kinases()
        results["benchmarking_kinase_trace"] = kinase_trace

        # 2. pathway enrichment per source
        pathway_stats = {}
        for src in self.pathway_sources:
            stats = self._compare_pathway_enrichment(src)
            pathway_stats[src] = stats
        results["pathway_enrichment"] = pathway_stats
        results["pathway_query_failed"] = self.pathway_query_failed
        results["pathway_query_errors"] = dict(self.pathway_query_errors or {})

        if self.pathway_query_failed:
            failed_sources = ", ".join(sorted(self.pathway_query_errors))
            print(
                "WARNING: g:Profiler query FAILED for source(s): "
                f"{failed_sources}. The pathway enrichment results are "
                "INCOMPLETE: this is a query failure, not a negative "
                "biological result."
            )

        # 3. Venn / overlap diagrams
        if self.do_venn_plot:
            results["venn_diagrams"] = self._plot_benchmarking_venn_diagrams(
                pathway_stats=pathway_stats,
            )
        else:
            results["venn_diagrams"] = {}

        # 4. heatmap
        if self.do_heatmap:
            fig = self._draw_heatmap()
            results["heatmap_fig"] = fig
        else:
            results["heatmap_fig"] = None

        # 5. save summary CSV
        if self.save_dir:
            self._save_summary(kinase_stats, pathway_stats, kinase_trace)

        return results

    # ================================================================
    # internal helpers
    # ================================================================

    @staticmethod
    def _load_config(path=None):
        """Load the module configuration from its YAML companion file."""
        if path is None:
            path = DEFAULT_CONFIG_PATH
        resolved_path = _resolve_config_path(path)
        try:
            with open(resolved_path) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def _dprint(self, *args, **kwargs):
        """Print a message only when debug logging is enabled."""
        if self.debugging_print:
            print(*args, **kwargs)

    # -- (array, kinase) keying ------------------------------------------
    # PTK and STK are independent arrays with disjoint peptide sets, so a
    # kinase called on PTK by one tool and on STK by the other is NOT an
    # agreement.  Every comparison below is therefore keyed on the pair
    # (Type, Kinase), never on the bare name.

    KEY_COL = "_ArrayKinaseKey"

    @staticmethod
    def _array_type(value) -> str:
        """Normalise an array label ('Type') into a comparable string."""
        try:
            if value is None or pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip().upper()

    @classmethod
    def _add_kinase_key(cls, df: pd.DataFrame, use_array: bool = True) -> pd.DataFrame:
        """Attach the (Type, Kinase) key column used by every comparison.

        Frames without a 'Type' column -- and every frame when ``use_array``
        is False -- get the empty array label, so single-array input behaves
        exactly as before.
        """
        df = df.copy()
        if use_array and "Type" in df.columns:
            types = df["Type"].map(cls._array_type)
        else:
            types = pd.Series([""] * len(df), index=df.index, dtype=object)
        df[cls.KEY_COL] = list(zip(types, df["Kinase"].astype(str)))
        return df

    @classmethod
    def _ensure_kinase_key(
        cls, df: pd.DataFrame, use_array: bool = True
    ) -> pd.DataFrame:
        """Return a frame that carries the key column, adding it if missing."""
        if cls.KEY_COL in df.columns:
            return df
        return cls._add_kinase_key(df, use_array=use_array)

    @classmethod
    def _has_array_column(cls, df: pd.DataFrame) -> bool:
        """True when the frame carries a 'Type' column with a usable label."""
        if df is None or "Type" not in df.columns:
            return False
        return bool(df["Type"].map(cls._array_type).astype(bool).any())

    @classmethod
    def _decide_array_keying(cls, df_results, df_bench, label_results,
                             label_benchmarking) -> bool:
        """Decide whether the key may carry the array half.

        BOTH sides must annotate the array.  If only one does, ('', 'ABL1')
        could never match ('PTK', 'ABL1') and every overlap would silently
        collapse to zero -- which is exactly what the shipped reference tables
        under data/benchmarking_data_set/ (header 'Kinase,Activity', no 'Type')
        would produce against a pyKinaXe result table that does carry 'Type'.
        In that case the array half is dropped on BOTH sides and the comparison
        degrades to the pre-fix, name-only behaviour -- loudly, because the
        (array, kinase) rule then cannot be enforced here.
        """
        have_results = cls._has_array_column(df_results)
        have_bench = cls._has_array_column(df_bench)
        if have_results == have_bench:
            return have_results
        annotated = label_results if have_results else label_benchmarking
        bare = label_benchmarking if have_results else label_results
        message = (
            f"WARNING: array keying DISABLED - '{annotated}' carries a usable "
            f"'Type' column, '{bare}' does not. Both sides fall back to the bare "
            "kinase name, i.e. to the behaviour before the (array, kinase) fix. "
            "A kinase matched across PTK and STK is NOT an agreement, but "
            "the array cannot be checked with "
            f"this input. Annotate '{bare}' with a 'Type' column to get "
            "(array, kinase) keying."
        )
        print(message)
        warnings.warn(message, stacklevel=3)
        return False

    @staticmethod
    def _key_label(key) -> str:
        """Human readable form of a (Type, Kinase) key."""
        array_type, kinase = key
        return f"{kinase} ({array_type})" if array_type else str(kinase)

    @classmethod
    def _validate_input(
        cls, df: pd.DataFrame, name: str, use_array: bool = True
    ) -> tuple:
        """Validate input and report every row whose 'Activity' is unusable.

        Rows whose 'Activity' cannot be read as a number are excluded from all
        comparisons, but they are never dropped silently: the affected kinase
        keys are printed and returned so the caller can report the pre-drop
        size and carry them through the trace table.

        Args:
            use_array (bool): Carry the array label in the key. False when only
                one of the two inputs annotates the array.

        Returns:
            tuple: ``(validated_frame, dropped_keys)`` where the second item is
            the ordered list of unique (Type, Kinase) keys that were removed.
        """
        df = df.copy()
        if "Kinase" not in df.columns or "Activity" not in df.columns:
            raise ValueError(
                f"{name} must contain columns 'Kinase' and 'Activity'. "
                f"Found: {list(df.columns)}"
            )
        df["Activity"] = pd.to_numeric(df["Activity"], errors="coerce")
        # Key BEFORE dropping, so the discarded rows carry a key too.
        df = cls._add_kinase_key(df, use_array=use_array)
        unreadable = df["Activity"].isna()
        dropped = list(dict.fromkeys(df.loc[unreadable, cls.KEY_COL].tolist()))
        if dropped:
            print(
                f"WARNING: {name}: {int(unreadable.sum())} of {len(df)} rows carry a "
                f"non-numeric 'Activity' and are excluded from every comparison; "
                f"{len(dropped)} kinase(s) affected: "
                f"{', '.join(cls._key_label(k) for k in dropped)}"
            )
        df = df.loc[~unreadable].copy()
        return df, dropped

    def _validate_full_results_input(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Validate the full kinase-analysis table.

        If `df` is None there is no full table: the significant/reported table
        is returned as a stand-in so the class stays runnable. Callers MUST
        consult `self._have_full_results` before reporting anything as
        "present in the analysis" -- in the stand-in case that is unknowable.
        """
        if df is None:
            df = self._ensure_kinase_key(self.df_results.copy())
            if "Significant" not in df.columns:
                df["Significant"] = True
            return df

        df = df.copy()
        if "Kinase" not in df.columns:
            raise ValueError(
                "input_results_all must contain a 'Kinase' column. "
                f"Found: {list(df.columns)}"
            )
        df = df[df["Kinase"].notna()].copy()
        df["Kinase"] = df["Kinase"].astype(str)

        numeric_cols = [
            "Activity",
            "NumSubstrates",
            "MeanSubstrate",
            "MeanPeptideStatistic",
            "MedianPeptideStatistic",
            "KinaseStatistic",
            "KinaseChange",
            "KRSA_MeanZ",
            "KRSA_AbsMeanZ",
            "Z_Score",
            "KPEA_AbsDominantZ",
            "KPEA_SelectedCutoff",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = self._add_kinase_key(df, use_array=self.key_on_array)
        if "Significant" in df.columns:
            df["Significant"] = df["Significant"].fillna(False).astype(bool)
        else:
            df["Significant"] = df[self.KEY_COL].isin(self.set_results)
        return df

    # kinase overlap

    def _compare_kinases(self) -> dict:
        overlap = self.set_results & self.set_bench
        only_results = self.set_results - self.set_bench
        only_bench = self.set_bench - self.set_results

        dropped_bench = sorted(set(self.dropped_bench) - self.set_bench)
        dropped_results = sorted(set(self.dropped_results) - self.set_results)
        handed_in_bench = len(self.set_bench) + len(dropped_bench)
        handed_in_results = len(self.set_results) + len(dropped_results)

        coverage_of_bench = (
            len(overlap) / len(self.set_bench) * 100 if self.set_bench else 0.0
        )
        coverage_of_results = (
            len(overlap) / len(self.set_results) * 100 if self.set_results else 0.0
        )
        coverage_of_bench_handed_in = (
            len(overlap) / handed_in_bench * 100 if handed_in_bench else 0.0
        )
        coverage_of_results_handed_in = (
            len(overlap) / handed_in_results * 100 if handed_in_results else 0.0
        )

        print("\n" + "=" * 80)
        print("KINASE OVERLAP ANALYSIS")
        print("=" * 80)
        print(
            f"Total kinases in {self.label_results}: {len(self.set_results)}"
            f" (of {handed_in_results} handed in)"
        )
        print(
            f"Total kinases in {self.label_benchmarking}: {len(self.set_bench)}"
            f" (of {handed_in_bench} handed in)"
        )
        if dropped_results:
            print(
                f"Dropped from {self.label_results}, Activity not numeric "
                f"({len(dropped_results)}): "
                f"{', '.join(self._key_label(k) for k in dropped_results)}"
            )
        if dropped_bench:
            print(
                f"Dropped from {self.label_benchmarking}, Activity not numeric "
                f"({len(dropped_bench)}): "
                f"{', '.join(self._key_label(k) for k in dropped_bench)}"
            )
        print(f"\nOverlapping kinases: {len(overlap)}")
        print(f"Only in {self.label_results}: {len(only_results)}")
        print(f"Only in {self.label_benchmarking}: {len(only_bench)}")
        print(f"\nCoverage of {self.label_benchmarking}: {coverage_of_bench:.1f}%")
        print(f"Coverage of {self.label_results}: {coverage_of_results:.1f}%")
        if dropped_bench or dropped_results:
            print(
                f"Coverage against everything handed in -- "
                f"{self.label_benchmarking}: {coverage_of_bench_handed_in:.1f}%, "
                f"{self.label_results}: {coverage_of_results_handed_in:.1f}%"
            )
        if overlap:
            labels = ", ".join(self._key_label(k) for k in sorted(overlap))
            print(f"\nShared kinases: {labels}")
        print("=" * 80)

        return {
            "overlap": sorted(overlap),
            "only_in_results": sorted(only_results),
            "only_in_benchmarking": sorted(only_bench),
            "overlap_labels": [self._key_label(k) for k in sorted(overlap)],
            "only_in_results_labels": [
                self._key_label(k) for k in sorted(only_results)
            ],
            "only_in_benchmarking_labels": [
                self._key_label(k) for k in sorted(only_bench)
            ],
            "coverage_of_bench_pct": coverage_of_bench,
            "coverage_of_results_pct": coverage_of_results,
            "dropped_non_numeric_benchmarking": dropped_bench,
            "dropped_non_numeric_results": dropped_results,
            "n_bench_kinases_handed_in": handed_in_bench,
            "n_results_kinases_handed_in": handed_in_results,
            "coverage_of_bench_handed_in_pct": coverage_of_bench_handed_in,
            "coverage_of_results_handed_in_pct": coverage_of_results_handed_in,
        }

    def _deduplicate_full_results_table(self) -> pd.DataFrame:
        """Keep one row per kinase from the full results table for tracing."""
        df = self._ensure_kinase_key(self.df_results_all.copy())
        if df.empty:
            return df

        activity_col = (
            "KinaseStatistic"
            if "KinaseStatistic" in df.columns
            else "MeanPeptideStatistic"
            if "MeanPeptideStatistic" in df.columns
            else "Activity"
            if "Activity" in df.columns
            else "MedianPeptideStatistic"
            if "MedianPeptideStatistic" in df.columns
            else None
        )
        rank_score = None
        for col in ("KRSA_AbsMeanZ", "KPEA_AbsDominantZ", "Z_Score"):
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").abs()
            rank_score = (
                values if rank_score is None else rank_score.combine_first(values)
            )

        df["_trace_significant"] = df["Significant"].fillna(False).astype(bool)
        df["_trace_rank_score"] = (
            rank_score.fillna(-np.inf) if rank_score is not None else 0.0
        )
        df["_trace_activity_abs"] = (
            pd.to_numeric(df[activity_col], errors="coerce").abs()
            if activity_col is not None
            else 0.0
        )

        # Effektstaerke zuerst, Promiskuitaet nur noch als letzter Tiebreak.
        sort_cols = ["_trace_significant", "_trace_rank_score", "_trace_activity_abs"]
        ascending = [False, False, False]
        if "NumSubstrates" in df.columns:
            sort_cols.append("NumSubstrates")
            ascending.append(False)

        dedup = (
            df.sort_values(sort_cols, ascending=ascending)
            .drop_duplicates(subset=self.KEY_COL, keep="first")
            .drop(columns=["_trace_rank_score", "_trace_activity_abs"])
            .reset_index(drop=True)
        )
        dedup[f"{self.label_results}_analysis_rank"] = np.arange(1, len(dedup) + 1)
        return dedup

    def _trace_benchmarking_kinases(self) -> pd.DataFrame:
        """Trace every reference kinase against all available result kinases."""
        df_full = self._deduplicate_full_results_table()
        full_lookup = {
            key: row
            for key, row in df_full.set_index(self.KEY_COL, drop=False).iterrows()
        }
        df_sig = self._deduplicate_activity_table(self.df_results)
        df_bench = self._deduplicate_activity_table(self.df_bench).reset_index()

        result_metric_cols = [
            "Kinase_Name",
            "Type",
            "NumSubstrates",
            "Activity",
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
            f"{self.label_results}_analysis_rank",
        ]
        # Only report metrics the table actually carries.  Without
        # 'input_results_all' the stand-in table has none of them, and emitting
        # all-NaN columns would suggest the kinase was scored and came out empty.
        available_metric_cols = [
            col for col in result_metric_cols if col in df_full.columns
        ]

        # Reference kinases whose 'Activity' was unreadable are carried through
        # the trace with Activity = NaN instead of vanishing.
        bench_entries = [
            (
                bench_row[self.KEY_COL],
                bench_row["Kinase"],
                bench_row["Activity"],
                False,
            )
            for _, bench_row in df_bench.iterrows()
        ]
        traced = {key for key, _, _, _ in bench_entries}
        bench_entries += [
            (key, key[1], np.nan, True)
            for key in dict.fromkeys(self.dropped_bench)
            if key not in traced
        ]

        rows = []
        for key, kinase, bench_activity, activity_unreadable in bench_entries:
            result_row = full_lookup.get(key)
            in_analysis = result_row is not None
            in_significant_results = key in self.set_results
            significant_in_analysis = (
                bool(result_row.get("Significant", False))
                if result_row is not None
                else False
            )

            row = {
                "Kinase": kinase,
                "Array": key[0],
                f"{self.label_benchmarking}_Activity": bench_activity,
                f"{self.label_benchmarking}_Activity_unreadable": activity_unreadable,
            }
            if self._have_full_results:
                row[f"in_{self.label_results}_analysis"] = in_analysis
                row[f"significant_in_{self.label_results}_analysis"] = (
                    significant_in_analysis
                )
            row[f"in_{self.label_results}_benchmark_input"] = in_significant_results
            row[f"{self.label_results}_Activity"] = (
                df_sig["Activity"].get(key, np.nan)
                if key in df_sig.index
                else np.nan
            )
            for col in available_metric_cols:
                if col == "Activity":
                    continue
                row[f"{self.label_results}_{col}"] = (
                    result_row.get(col, np.nan)
                    if result_row is not None and col in result_row.index
                    else np.nan
                )
            rows.append(row)

        trace = pd.DataFrame(rows)

        total = len(trace)
        print("\n" + "=" * 80)
        print(f"{self.label_benchmarking.upper()} KINASE TRACE IN {self.label_results.upper()}")
        print("=" * 80)
        n_unreadable = int(trace[f"{self.label_benchmarking}_Activity_unreadable"].sum())
        if n_unreadable:
            print(
                f"{self.label_benchmarking} kinases without a readable Activity, "
                f"carried with Activity = NaN: {n_unreadable}"
            )
        if self._have_full_results:
            found_any = int(trace[f"in_{self.label_results}_analysis"].sum())
            found_sig = int(
                trace[f"significant_in_{self.label_results}_analysis"].sum()
            )
            print(
                f"{self.label_benchmarking} kinases found anywhere in "
                f"{self.label_results} analysis: {found_any}/{total}"
            )
            print(
                f"{self.label_benchmarking} kinases significant in "
                f"{self.label_results} analysis: {found_sig}/{total}"
            )
            if total and found_any < total:
                missing = [
                    self._key_label(key)
                    for key in trace.loc[
                        ~trace[f"in_{self.label_results}_analysis"], ["Array", "Kinase"]
                    ].itertuples(index=False, name=None)
                ]
                print(
                    f"Missing from {self.label_results} analysis: {', '.join(missing)}"
                )
        else:
            found_input = int(
                trace[f"in_{self.label_results}_benchmark_input"].sum()
            )
            print(
                f"{self.label_benchmarking} kinases among the reported "
                f"{self.label_results} results: {found_input}/{total}"
            )
            print(
                "Full analysis table ('input_results_all') not supplied - panel "
                "coverage is UNKNOWN; the remaining kinases may have been scored "
                "but not called."
            )
            if total and found_input < total:
                not_reported = [
                    self._key_label(key)
                    for key in trace.loc[
                        ~trace[f"in_{self.label_results}_benchmark_input"],
                        ["Array", "Kinase"],
                    ].itertuples(index=False, name=None)
                ]
                print(
                    f"Not among the reported {self.label_results} results: "
                    f"{', '.join(not_reported)}"
                )
        print("=" * 80)

        return trace

    # pathway enrichment

    def _run_gprofiler(self, kinase_list: list, source: str) -> pd.DataFrame:
        """Run g:Profiler against all annotated genes.

        The kinases are sent as gene symbols (UniProt accessions are converted),
        as in the pipeline's pathway stage; ``intersections`` lists them as given.

        Returns:
            pd.DataFrame: Run result for g:Profiler.
        """
        try:
            out = profile_kinases(
                kinase_list,
                sources=[source],
                user_threshold=self.significance_level_pathways,
            )
        except (
            requests.exceptions.RequestException,
            ValueError,
            KeyError,
        ) as e:
            self._record_pathway_query_failure(source, e)
            print(
                f"ERROR: g:Profiler query FAILED for source={source} "
                f"({type(e).__name__}: {e}). No pathway result can be "
                "reported for this source; this is NOT a negative "
                "biological result."
            )
            out = pd.DataFrame()
        return out

    def _record_pathway_query_failure(self, source: str, exc: Exception) -> None:
        """Record a failed g:Profiler query so the run can report it.

        Args:
            source (str): Pathway source whose query failed.
            exc (Exception): The exception that ended the query.
        """
        self.pathway_query_failed = True
        if getattr(self, "pathway_query_errors", None) is None:
            self.pathway_query_errors = {}
        self.pathway_query_errors.setdefault(source, []).append(
            f"{type(exc).__name__}: {exc}"
        )

    def _compare_pathway_enrichment(self, source: str) -> dict:
        print(f"\n{'=' * 80}")
        print(f"PATHWAY ENRICHMENT COMPARISON: {source}")
        print("=" * 80)

        results_list = self.df_results["Kinase"].tolist()
        bench_list = self.df_bench["Kinase"].tolist()

        df_res = self._run_gprofiler(results_list, source)
        df_ben = self._run_gprofiler(bench_list, source)

        if (self.pathway_query_errors or {}).get(source):
            print(
                f"PATHWAY ENRICHMENT UNAVAILABLE for {source}, query failed: "
                + "; ".join(self.pathway_query_errors[source])
            )
            return None

        if df_res.empty and df_ben.empty:
            print("No enrichment results for either input.")
            return None

        native_res = set(df_res["native"]) if not df_res.empty else set()
        native_ben = set(df_ben["native"]) if not df_ben.empty else set()

        overlap = native_res & native_ben
        only_res = native_res - native_ben
        only_ben = native_ben - native_res

        coverage = (
            len(overlap) / len(native_ben) * 100 if native_ben else 0.0
        )

        print(f"Pathways in {self.label_results}: {len(native_res)}")
        print(f"Pathways in {self.label_benchmarking}: {len(native_ben)}")
        print(f"\nOverlapping pathways: {len(overlap)}")
        print(f"Only in {self.label_results}: {len(only_res)}")
        print(f"Only in {self.label_benchmarking}: {len(only_ben)}")
        print(f"\nCoverage of {self.label_benchmarking}: {coverage:.1f}%")

        if overlap:
            # resolve names if available
            name_map = {}
            for df_src in (df_res, df_ben):
                if not df_src.empty and "name" in df_src.columns:
                    name_map.update(
                        dict(zip(df_src["native"], df_src["name"]))
                    )
            #print(f"\nShared pathways:")
            #for pw in sorted(overlap):
            #    name = name_map.get(pw, "")
            #    print(f"  {pw}  {name}")

        fraction_table = self.build_pathway_kinase_fraction_table(
            df_results_enrichment=df_res,
            df_benchmarking_enrichment=df_ben,
            source=source,
        )
        self._dprint(
            "\nTop pathway kinase fractions:\n"
            + fraction_table[
                [
                    "Pathway",
                    f"{self.label_results} k/n",
                    f"{self.label_benchmarking} k/n",
                    "Status",
                ]
            ]
            .head(10)
            .to_string(index=False)
            if not fraction_table.empty
            else "No pathway kinase fraction table generated."
        )

        print("=" * 80)

        return {
            "overlap": sorted(overlap),
            "only_in_results": sorted(only_res),
            "only_in_benchmarking": sorted(only_ben),
            "coverage_of_bench_pct": coverage,
            "df_results_enrichment": df_res,
            "df_benchmarking_enrichment": df_ben,
            "pathway_kinase_fraction_table": fraction_table,
        }

    @staticmethod
    def _to_intersection_list(value) -> list:
        """Normalize g:Profiler intersection values from lists or CSV strings.
        
        Returns:
            list: Converted intersection list.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() == "nan":
                return []
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
            return [part.strip() for part in stripped.split(";") if part.strip()]
        return [str(value)]

    @staticmethod
    def _safe_int(value):
        if value is None:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except TypeError:
            pass
        return int(value)

    @staticmethod
    def _safe_float(value):
        if value is None:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except TypeError:
            pass
        return float(value)

    @staticmethod
    def _fraction_label(row) -> str:
        if row is None:
            return "n.s."
        k = Benchmarking_Heatmap._safe_int(row.get("intersection_size"))
        n = Benchmarking_Heatmap._safe_int(row.get("query_size"))
        if pd.isna(k) or pd.isna(n):
            return ""
        return f"{k}/{n}"

    @staticmethod
    def _row_value(row, column, default=np.nan):
        if row is None or column not in row:
            return default
        return row[column]

    @staticmethod
    def _pathway_lookup(df: pd.DataFrame) -> dict:
        if df is None or df.empty or "native" not in df.columns:
            return {}

        sort_cols = ["p_value"] if "p_value" in df.columns else None
        df_lookup = df.copy()
        if sort_cols:
            df_lookup = df_lookup.sort_values(sort_cols, ascending=True)

        return {
            str(row["native"]): row
            for _, row in df_lookup.drop_duplicates(subset="native", keep="first").iterrows()
        }

    def build_pathway_kinase_fraction_table(
        self,
        df_results_enrichment: pd.DataFrame,
        df_benchmarking_enrichment: pd.DataFrame,
        source: str = None,
    ) -> pd.DataFrame:
        """Build a side-by-side pathway table with per-dataset kinase fractions.
        
        The displayed k/n value is intersection_size/query_size from g:Profiler,
        e.g. 12/22 means 12 of the 22 query kinases annotated for that source
        intersect the pathway. If a pathway is absent from one significant
        result set, that side is shown as n.s. (not significant/returned), not
        as 0/n.
        """
        result_lookup = self._pathway_lookup(df_results_enrichment)
        bench_lookup = self._pathway_lookup(df_benchmarking_enrichment)
        pathway_ids = sorted(set(result_lookup) | set(bench_lookup))

        if not pathway_ids:
            return pd.DataFrame()

        result_k_col = f"{self.label_results}_intersection_size"
        result_n_col = f"{self.label_results}_query_size"
        result_p_col = f"{self.label_results}_p_value"
        bench_k_col = f"{self.label_benchmarking}_intersection_size"
        bench_n_col = f"{self.label_benchmarking}_query_size"
        bench_p_col = f"{self.label_benchmarking}_p_value"

        rows = []
        for pathway_id in pathway_ids:
            result_row = result_lookup.get(pathway_id)
            bench_row = bench_lookup.get(pathway_id)

            result_intersections = self._to_intersection_list(
                self._row_value(result_row, "intersections", [])
            )
            bench_intersections = self._to_intersection_list(
                self._row_value(bench_row, "intersections", [])
            )

            if result_row is not None and bench_row is not None:
                status = "shared"
            elif result_row is not None:
                status = f"only in {self.label_results}"
            else:
                status = f"only in {self.label_benchmarking}"

            pathway_name = self._row_value(result_row, "name", None)
            if pathway_name is None or pd.isna(pathway_name):
                pathway_name = self._row_value(bench_row, "name", pathway_id)

            rows.append(
                {
                    "Source": source,
                    "Pathway_ID": pathway_id,
                    "Pathway": pathway_name,
                    f"{self.label_results} k/n": self._fraction_label(result_row),
                    f"{self.label_benchmarking} k/n": self._fraction_label(bench_row),
                    result_k_col: self._safe_int(
                        self._row_value(result_row, "intersection_size", np.nan)
                    ),
                    result_n_col: self._safe_int(
                        self._row_value(result_row, "query_size", np.nan)
                    ),
                    result_p_col: self._safe_float(
                        self._row_value(result_row, "p_value", np.nan)
                    ),
                    f"{self.label_results}_precision": self._safe_float(
                        self._row_value(result_row, "precision", np.nan)
                    ),
                    bench_k_col: self._safe_int(
                        self._row_value(bench_row, "intersection_size", np.nan)
                    ),
                    bench_n_col: self._safe_int(
                        self._row_value(bench_row, "query_size", np.nan)
                    ),
                    bench_p_col: self._safe_float(
                        self._row_value(bench_row, "p_value", np.nan)
                    ),
                    f"{self.label_benchmarking}_precision": self._safe_float(
                        self._row_value(bench_row, "precision", np.nan)
                    ),
                    "Status": status,
                    f"{self.label_results}_intersections": ", ".join(
                        sorted(result_intersections)
                    ),
                    f"{self.label_benchmarking}_intersections": ", ".join(
                        sorted(bench_intersections)
                    ),
                    "Shared_intersections": ", ".join(
                        sorted(set(result_intersections) & set(bench_intersections))
                    ),
                }
            )

        table = pd.DataFrame(rows)
        table["_result_sort"] = table[result_p_col].fillna(np.inf)
        table["_bench_sort"] = table[bench_p_col].fillna(np.inf)
        table.sort_values(
            ["_result_sort", "_bench_sort", "Pathway_ID"],
            ascending=[True, True, True],
            inplace=True,
        )
        table.drop(columns=["_result_sort", "_bench_sort"], inplace=True)
        table.reset_index(drop=True, inplace=True)
        return table

    @staticmethod
    def _safe_filename_token(value):
        token = "NA" if value is None else str(value).strip()
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token).strip("_")
        return token or "NA"

    @staticmethod
    def _extract_pathway_items(df_pathways):
        if df_pathways is None or df_pathways.empty:
            return []

        for column in ("native", "term_id", "id", "name"):
            if column in df_pathways.columns:
                return df_pathways[column].dropna().astype(str).tolist()
        return []

    def _plot_benchmarking_venn_diagrams(self, pathway_stats: dict) -> dict:
        if self.venn_plot_output is None:
            self._dprint("No Venn output directory configured; skipping Venn plots.")
            return {}

        output_dir = Path(self.venn_plot_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}
        print("\n" + "=" * 80)
        print("BENCHMARKING OVERLAP DIAGRAMS")
        print("=" * 80)

        kinase_groups = {
            self.label_results: [
                self._key_label(k) for k in sorted(self.set_results)
            ],
            self.label_benchmarking: [
                self._key_label(k) for k in sorted(self.set_bench)
            ],
        }
        if any(kinase_groups.values()):
            save_path = output_dir / f"benchmarking_kinases_overlap.{self.image_format}"
            table_dir = (
                output_dir / "benchmarking_kinases_overlap_tables"
                if self.venn_plot_tables
                else None
            )
            plotter = VennDiagramPlot(
                groups=kinase_groups,
                title="Benchmarking kinase overlap",
                item_label="kinases",
                save_path=save_path,
                save_tables_dir=table_dir,
                dpi=self.dpi,
                image_format=self.image_format,
                debugging_print=self.debugging_print,
            )
            fig = plotter.plot()
            plt.close(fig)
            outputs["kinases"] = {
                "plot": save_path,
                "tables": table_dir,
                "group_sizes": {
                    group_name: len(group_values)
                    for group_name, group_values in plotter.group_sets.items()
                },
            }
            print(f"Kinase overlap diagram saved: {save_path}")

        pathway_outputs = {}
        for source, stats in pathway_stats.items():
            if stats is None:
                continue

            pathway_groups = {
                self.label_results: self._extract_pathway_items(
                    stats.get("df_results_enrichment")
                ),
                self.label_benchmarking: self._extract_pathway_items(
                    stats.get("df_benchmarking_enrichment")
                ),
            }
            if not any(pathway_groups.values()):
                self._dprint(f"No {source} pathways available for Venn plotting.")
                continue

            safe_source = self._safe_filename_token(source)
            save_path = (
                output_dir
                / f"benchmarking_pathways_{safe_source}_overlap.{self.image_format}"
            )
            table_dir = (
                output_dir / f"benchmarking_pathways_{safe_source}_overlap_tables"
                if self.venn_plot_tables
                else None
            )
            plotter = VennDiagramPlot(
                groups=pathway_groups,
                title=f"Benchmarking {source} pathway overlap",
                item_label="pathways",
                save_path=save_path,
                save_tables_dir=table_dir,
                dpi=self.dpi,
                image_format=self.image_format,
                debugging_print=self.debugging_print,
            )
            fig = plotter.plot()
            plt.close(fig)
            pathway_outputs[source] = {
                "plot": save_path,
                "tables": table_dir,
                "group_sizes": {
                    group_name: len(group_values)
                    for group_name, group_values in plotter.group_sets.items()
                },
            }
            print(f"{source} pathway overlap diagram saved: {save_path}")

        if pathway_outputs:
            outputs["pathways"] = pathway_outputs

        if outputs:
            print(f"Benchmarking overlap diagrams saved to: {output_dir}")
        else:
            print("No benchmarking overlap diagrams were generated.")
        print("=" * 80)

        return outputs

    @classmethod
    def _deduplicate_activity_table(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Keep one activity value per (array, kinase), preferring the strongest absolute signal.

        Returns:
            pd.DataFrame: Deduplicate activity table, indexed by the (Type, Kinase) key.
        """
        return (
            cls._ensure_kinase_key(df)
            .assign(_abs=lambda d: d["Activity"].abs())
            .sort_values("_abs", ascending=False)
            .drop_duplicates(subset=cls.KEY_COL, keep="first")
            .drop(columns="_abs")
            .set_index(cls.KEY_COL)
        )

    @staticmethod
    def _build_duplicate_heatmap_labels(df: pd.DataFrame) -> list:
        labels = []
        kinase_counts = df["Kinase"].astype(str).value_counts()

        for _, row in df.iterrows():
            kinase = str(row["Kinase"])
            if kinase_counts.get(kinase, 0) <= 1:
                labels.append(kinase)
                continue

            type_label = ""
            if "Type" in row.index and pd.notna(row["Type"]):
                type_label = str(row["Type"])

            labels.append(f"{kinase} ({type_label})" if type_label else kinase)

        label_counts = pd.Series(labels).value_counts()
        seen = defaultdict(int)
        unique_labels = []
        for label in labels:
            seen[label] += 1
            if label_counts[label] > 1:
                unique_labels.append(f"{label} #{seen[label]}")
            else:
                unique_labels.append(label)
        return unique_labels

    # heatmap

    def _draw_heatmap(self):
        df_b = self._deduplicate_activity_table(self.df_bench)

        paired = self.df_results[self.KEY_COL].isin(self.set_bench)
        unpaired = sorted(set(self.df_results.loc[~paired, self.KEY_COL]))
        if unpaired:
            print(
                f"WARNING: no {self.label_benchmarking} value keyed by "
                f"(Type, Kinase) for {len(unpaired)} {self.label_results} "
                f"row(s); they are omitted from the heatmap: "
                f"{', '.join(self._key_label(k) for k in unpaired)}"
            )
        df_r_shared = self.df_results[paired].copy()
        if df_r_shared.empty:
            self._dprint("No shared kinases; skipping heatmap.")
            return None

        df_r_shared["_original_order"] = np.arange(len(df_r_shared))
        sort_cols = ["Kinase", "_original_order"]
        if "Type" in df_r_shared.columns:
            sort_cols.insert(1, "Type")
        df_r_shared = (
            df_r_shared.sort_values(sort_cols, kind="mergesort")
            .drop(columns="_original_order")
            .reset_index(drop=True)
        )

        heatmap_labels = self._build_duplicate_heatmap_labels(df_r_shared)
        shared_keys = df_r_shared[self.KEY_COL].tolist()
        shared_kinases = df_r_shared["Kinase"].astype(str).tolist()
        act_results = df_r_shared["Activity"].values.astype(float)
        act_bench = df_b.loc[shared_keys, "Activity"].values.astype(float)

        matrix = np.array([act_results, act_bench])
        heatmap_df = pd.DataFrame(
            matrix,
            index=[self.label_results, self.label_benchmarking],
            columns=heatmap_labels,
        )

        heatmap_values = pd.DataFrame(
            {
                "Display_Label": heatmap_labels,
                "Kinase": shared_kinases,
                f"Activity_{self.label_results}": act_results,
                f"Activity_{self.label_benchmarking}": act_bench,
            }
        )
        for col in ["Type", "ActivityMetric", "NumSubstrates", "KPEA_SelectedCutoff"]:
            if col in df_r_shared.columns:
                heatmap_values[f"{self.label_results}_{col}"] = df_r_shared[col].values

        # figure size
        n_cols = len(heatmap_labels)
        n_rows = 2
        max_label_len = max(len(str(k)) for k in heatmap_labels)
        margin_bottom = max(self.MARGIN_BOTTOM, max_label_len * 0.08)

        fig_width = self.MARGIN_LEFT + n_cols * self.CELL_WIDTH + self.MARGIN_RIGHT
        fig_height = self.MARGIN_TOP + n_rows * self.CELL_HEIGHT + margin_bottom

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # colour limits
        # 'individual' (the default) stretches the scale to this comparison's own
        # range; 'fixed' keeps the historical stepped limits so heatmaps of
        # different benchmarking runs stay comparable.
        if self.COLOR_SCALE == INDIVIDUAL_COLOR_SCALE:
            vmin, vmax = _data_limits(matrix)
        else:
            abs_max = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
            step = next(
                (step for step in self.FIXED_VALUE_STEPS if abs_max < step),
                None,
            )
            vmin, vmax = (-step, step) if step is not None else (-abs_max, abs_max)

        # draw
        hm = sns.heatmap(
            heatmap_df,
            ax=ax,
            cmap=self.CMAP,
            center=0,
            vmin=vmin,
            vmax=vmax,
            linewidths=self.CELL_LINEWIDTH,
            linecolor=self.CELL_LINECOLOR,
            annot=False,
            cbar=False,
            xticklabels=True,
            yticklabels=True,
        )

        # axes formatting
        ax.set_xlabel("Kinases", fontsize=self.AXES_LABEL_FONTSIZE)
        ax.set_ylabel("", fontsize=self.AXES_LABEL_FONTSIZE)
        ax.set_title(
            "Kinase Activity Comparison",
            fontsize=self.TITLE_FONTSIZE,
            pad=self.TITLE_PAD,
        )

        ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.xaxis.set_label_position("top")

        for label in ax.get_xticklabels():
            label.set_rotation(self.X_TICK_ROTATION)
            label.set_ha("left")
            label.set_fontsize(self.X_TICK_FONTSIZE)

        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_fontsize(self.Y_TICK_FONTSIZE)

        # layout (must come before colorbar so get_position is correct)
        fig.subplots_adjust(
            left=self.MARGIN_LEFT / fig_width,
            right=1.0 - self.MARGIN_RIGHT / fig_width,
            top=1.0 - self.MARGIN_TOP / fig_height,
            bottom=margin_bottom / fig_height,
        )

        # colorbar (anchored to exact axes bounds)
        cbar_width_fig = (self.CBAR_WIDTH_FACTOR * self.CELL_WIDTH) / fig_width

        ax_pos = ax.get_position()
        cbar_x = ax_pos.x1 + self.CBAR_OFFSET_X
        cbar_y = ax_pos.y0
        cbar_h = ax_pos.height

        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width_fig, cbar_h])
        cbar = fig.colorbar(hm.collections[0], cax=cbar_ax)
        cbar.set_label(self.activity_label, fontsize=self.Y_TICK_FONTSIZE)
        cbar.ax.tick_params(labelsize=self.X_TICK_FONTSIZE)

        # save
        if self.save_dir:
            path = self.save_dir / f"benchmarking_heatmap.{self.image_format}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight", format=self.image_format)
            self._dprint(f"Heatmap saved: {path.absolute()}")
            heatmap_values_path = self.save_dir / "benchmarking_heatmap_values.csv"
            heatmap_values.to_csv(heatmap_values_path, index=False)
            self._dprint(f"Heatmap values saved: {heatmap_values_path.absolute()}")

        plt.close(fig)
        return fig

    # save summary

    def _save_summary(
        self,
        kinase_stats: dict,
        pathway_stats: dict,
        kinase_trace: pd.DataFrame = None,
    ):
        # kinase overlap CSV
        df_r = self._deduplicate_activity_table(self.df_results)
        df_b = self._deduplicate_activity_table(self.df_bench)
        rows = []
        shared_keys = self.set_results & self.set_bench
        for key in sorted(self.set_results | self.set_bench):
            act_r = df_r["Activity"].get(key, np.nan)
            act_b = df_b["Activity"].get(key, np.nan)
            rows.append(
                {
                    "Kinase": key[1],
                    "Type": key[0],
                    f"Activity_{self.label_results}": act_r,
                    f"Activity_{self.label_benchmarking}": act_b,
                    "shared": key in shared_keys,
                }
            )
        df_summary = pd.DataFrame(rows)
        path = self.save_dir / "benchmarking_kinase_summary.csv"
        df_summary.to_csv(path, index=False)
        self._dprint(f"Kinase summary saved: {path.absolute()}")

        if kinase_trace is not None and not kinase_trace.empty:
            path = self.save_dir / "benchmarking_kinase_trace.csv"
            kinase_trace.to_csv(path, index=False)
            self._dprint(f"Benchmarking kinase trace saved: {path.absolute()}")

        # pathway overlap CSVs
        for src, stats in pathway_stats.items():
            if stats is None:
                continue
            for key, label in [
                ("df_results_enrichment", self.label_results),
                ("df_benchmarking_enrichment", self.label_benchmarking),
            ]:
                df_pw = stats.get(key)
                if df_pw is not None and not df_pw.empty:
                    fname = f"benchmarking_pathways_{src}_{label}.csv"
                    df_pw.to_csv(self.save_dir / fname, index=False)

            fraction_table = stats.get("pathway_kinase_fraction_table")
            if fraction_table is not None and not fraction_table.empty:
                fname = f"benchmarking_pathways_{src}_kinase_fraction_comparison.csv"
                fraction_table.to_csv(self.save_dir / fname, index=False)

        # pathway query status: a failed query must stay distinguishable
        # from a genuine "no enrichment" result in the written output.
        errors = self.pathway_query_errors or {}
        status_rows = []
        for src in self.pathway_sources:
            src_errors = errors.get(src, [])
            if src_errors:
                status = "FAILED"
            elif pathway_stats.get(src) is None:
                status = "NO_ENRICHMENT"
            else:
                status = "OK"
            status_rows.append(
                {
                    "source": src,
                    "status": status,
                    "error": " | ".join(src_errors),
                }
            )
        status_path = self.save_dir / "benchmarking_pathway_status.csv"
        pd.DataFrame(status_rows).to_csv(status_path, index=False)
        self._dprint(f"Pathway query status saved: {status_path.absolute()}")


# ====================================================================
# Example / standalone usage
# ====================================================================
if __name__ == "__main__":
    # Demo data
    kinases_shared = [f"KINASE_{i}" for i in range(1, 16)]
    kinases_only_a = [f"ONLY_A_{i}" for i in range(1, 6)]
    kinases_only_b = [f"ONLY_B_{i}" for i in range(1, 4)]

    df_a = pd.DataFrame(
        {
            "Kinase": kinases_shared + kinases_only_a,
            "Activity": np.round(np.random.uniform(-2, 2, 20), 2),
        }
    )
    df_b = pd.DataFrame(
        {
            "Kinase": kinases_shared + kinases_only_b,
            "Activity": np.round(np.random.uniform(-2, 2, 18), 2),
        }
    )

    bench = Benchmarking_Heatmap(
        input_results=df_a,
        input_benchmarking=df_b,
        label_results="pyKinaXe",
        label_benchmarking="BioNavigator",
        save_dir="results/benchmarking_demo",
        heatmap=True,
        dpi=150,
        debugging_print=True,
    )

    results = bench.run()
    print("\nDone.")
