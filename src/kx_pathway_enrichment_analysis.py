"""Pathway enrichment utilities for the pyKinaXe downstream workflow.

This module implements stage 3 of the downstream analysis stack. It consumes
significant kinase results from ``kx_upstream_kinase_analysis.py`` and queries
g:Profiler to summarize pathway-level signals across KEGG, Reactome, and
WikiPathways.

Its main tasks are:

- normalize kinase result payloads into a common DataFrame shape
- query g:Profiler for one or more pathway sources
- save per-condition pathway tables
- build pathway heatmaps and venn-ready pathway sets

The public entry point is ``PathwayEnrichmentAnalysis``.
"""

from pathlib import Path
import re

from gprofiler import GProfiler
import pandas as pd

from config.analysis_modules import PATHWAY_ENRICHMENT_DEFAULTS
from kx_plot_results import HeatmapPlot_UKA, VennDiagramPlot

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - optional GUI dependency
    tk = None


class PathwayEnrichmentAnalysis:
    """Stage 3 of the UKA/KPEA workflow: pathway enrichment and heatmaps."""

    UKA_METRIC_OPTIONS = dict(PATHWAY_ENRICHMENT_DEFAULTS["uka_metric_options"])

    def __init__(
        self,
        significance_level_pathways=PATHWAY_ENRICHMENT_DEFAULTS[
            "default_significance_level_pathways"
        ],
        heatmap_plot=False,
        heatmap_plot_GUI=False,
        heatmap_plot_output=None,
        uka_visualization_metric=PATHWAY_ENRICHMENT_DEFAULTS[
            "default_uka_visualization_metric"
        ],
        debugging_print=True,
    ):
        """Initialize the PathwayEnrichmentAnalysis instance.
        
        Args:
            significance_level_pathways: Significance level pathways processed by this function.
            heatmap_plot: Heatmap plot processed by this function.
            heatmap_plot_GUI: Heatmap plot GUI processed by this function.
            heatmap_plot_output: Heatmap plot output processed by this function.
            uka_visualization_metric: UKA visualization metric processed by this function.
            debugging_print: Whether to print additional debug information.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        self.significance_level_pathways = significance_level_pathways
        self.heatmap_plot = heatmap_plot
        self.heatmap_plot_GUI = heatmap_plot_GUI
        self.heatmap_plot_output = (
            Path(heatmap_plot_output) if heatmap_plot_output is not None else None
        )
        self.debugging_print = debugging_print
        self._gprofiler = GProfiler(return_dataframe=True)

        resolved_uka_metric = str(uka_visualization_metric).lower()
        if resolved_uka_metric not in self.UKA_METRIC_OPTIONS:
            raise ValueError(
                f"Unknown uka_visualization_metric '{resolved_uka_metric}'. "
                "Use 'kinase_change' or 'kinase_statistic'."
            )
        self.uka_visualization_metric = resolved_uka_metric
        metric_config = self.UKA_METRIC_OPTIONS[resolved_uka_metric]
        self.uka_visualization_column = metric_config["column"]
        self.uka_visualization_label = metric_config["label"]

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
    def _extract_pathway_items(df_pathways):
        """Extract pathway items.
        
        Args:
            df_pathways: Input pandas DataFrame containing pathways.
        
        Returns:
            object: Extracted pathway items.
        """
        if df_pathways is None or df_pathways.empty:
            return []

        for column in ("native", "term_id", "id", "name"):
            if column in df_pathways.columns:
                return df_pathways[column].dropna().astype(str).tolist()
        return []

    def _normalize_significant_kinases(self, significant_kinases):
        """Normalize significant kinases.
        
        Args:
            significant_kinases: Significant kinases processed by this function.
        
        Returns:
            object: Normalized significant kinases.
        """
        if isinstance(significant_kinases, dict):
            significant_kinases = significant_kinases.get("significant_kinases")

        if significant_kinases is None:
            return pd.DataFrame()

        if isinstance(significant_kinases, pd.DataFrame):
            return significant_kinases.copy()

        kinase_list = list(significant_kinases)
        if not kinase_list:
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "Kinase": kinase_list,
                "MeanPeptideStatistic": 0.0,
                "MedianPeptideStatistic": 0.0,
                self.uka_visualization_column: 0.0,
                "NumSubstrates": 0,
                "Significant": True,
            }
        )

    def _extract_ranked_kinase_list(self, significant_kinases):
        """Extract ranked kinase list.
        
        Args:
            significant_kinases: Significant kinases processed by this function.
        
        Returns:
            object: Extracted ranked kinase list.
        """
        if significant_kinases.empty:
            return []

        rank_score_col = (
            "KRSA_AbsMeanZ"
            if "KRSA_AbsMeanZ" in significant_kinases.columns
            else "KPEA_AbsDominantZ"
            if "KPEA_AbsDominantZ" in significant_kinases.columns
            else self.uka_visualization_column
        )
        activity_col = (
            "KinaseStatistic"
            if "KinaseStatistic" in significant_kinases.columns
            else "MeanPeptideStatistic"
            if "MeanPeptideStatistic" in significant_kinases.columns
            else "MedianPeptideStatistic"
            if "MedianPeptideStatistic" in significant_kinases.columns
            else self.uka_visualization_column
        )

        return (
            significant_kinases.assign(
                abs_delta=lambda d: d[activity_col].abs()
                if activity_col in d.columns
                else 0
            )
            .sort_values(
                [rank_score_col, "NumSubstrates", "abs_delta"],
                ascending=[False, False, False],
            )
            .drop_duplicates(subset="Kinase", keep="first")["Kinase"]
            .tolist()
        )

    def _profile_source(self, kinase_list, source):
        """Profile source.
        
        Args:
            kinase_list: Kinase list processed by this function.
            source: Source processed by this function.
        
        Returns:
            object: Profiled source.
        """
        if not kinase_list:
            return pd.DataFrame()

        try:
            return self._gprofiler.profile(
                organism="hsapiens",
                query=kinase_list,
                all_results=False,
                combined=False,
                ordered=False,
                no_iea=False,
                sources=[source],
                numeric_namespace="ENTREZGENE_ACC",
                domain_scope="annotated",
                measure_underrepresentation=False,
                significance_threshold_method="g_SCS",
                user_threshold=self.significance_level_pathways,
                no_evidences=False,
                background=None,
            )
        except Exception as exc:
            self._dprint(
                f"     WARNING: Pathway enrichment failed for source={source}. Reason: {exc}"
            )
            return pd.DataFrame()

    def _profile_sources(self, kinase_list, sources):
        """Profile sources.
        
        Args:
            kinase_list: Kinase list processed by this function.
            sources: Sources processed by this function.
        
        Returns:
            object: Profiled sources.
        """
        sources = tuple(sources)
        if not kinase_list:
            return {source: pd.DataFrame() for source in sources}

        try:
            profiled = self._gprofiler.profile(
                organism="hsapiens",
                query=kinase_list,
                all_results=False,
                combined=False,
                ordered=False,
                no_iea=False,
                sources=list(sources),
                numeric_namespace="ENTREZGENE_ACC",
                domain_scope="annotated",
                measure_underrepresentation=False,
                significance_threshold_method="g_SCS",
                user_threshold=self.significance_level_pathways,
                no_evidences=False,
                background=None,
            )
        except Exception as exc:
            self._dprint(
                "     WARNING: Combined pathway enrichment request failed; "
                f"falling back to per-source requests. Reason: {exc}"
            )
            return {
                source: self._profile_source(kinase_list, source)
                for source in sources
            }

        if profiled is None:
            return {source: pd.DataFrame() for source in sources}

        if not isinstance(profiled, pd.DataFrame):
            profiled = pd.DataFrame(profiled)

        if profiled.empty:
            return {source: pd.DataFrame(columns=profiled.columns) for source in sources}

        if "source" not in profiled.columns:
            if len(sources) == 1:
                return {sources[0]: profiled}
            self._dprint(
                "     Combined pathway enrichment response did not include a "
                "'source' column; falling back to per-source requests."
            )
            return {
                source: self._profile_source(kinase_list, source)
                for source in sources
            }

        return {
            source: profiled[profiled["source"] == source].copy()
            for source in sources
        }

    def _plot_heatmap(self, significant_kinases, pathways, output_path=None, control=None, condition=None):
        """Plot heatmap.
        
        Args:
            significant_kinases: Significant kinases processed by this function.
            pathways: Pathways processed by this function.
            output_path: Path to the output.
            control: Control processed by this function.
            condition: Condition processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        source_labels = {
            "KEGG": "KEGG",
            "WP": "WP",
            "REAC": "REAC",
        }

        for source, df_pathways in pathways.items():
            if df_pathways is None or df_pathways.empty:
                self._dprint(
                    f"     No significant pathways found in {source} enrichment analysis."
                )
                continue

            if "name" not in df_pathways.columns or "intersections" not in df_pathways.columns:
                self._dprint(
                    f"     Skipping {source} heatmap because required columns are missing."
                )
                continue

            save_path = (
                str(Path(output_path) / f"{source}_UKA_heatmap_{control}_{condition}.png")
                if output_path is not None
                else None
            )
            enrichment_data = df_pathways[["name", "intersections"]].copy()

            try:
                if self.heatmap_plot_GUI:
                    if tk is None:
                        raise RuntimeError(
                            "Tkinter GUI plotting is not available in this environment. "
                            "Set heatmap_plot_GUI=False for headless execution."
                        )
                    root = tk.Tk()
                    HeatmapPlot_UKA(
                        root=root,
                        debugging_print=self.debugging_print,
                        enrichment_data=enrichment_data,
                        results_data=significant_kinases,
                        y_axis="metric",
                        value_col=self.uka_visualization_column,
                        value_label=self.uka_visualization_label,
                        gui=True,
                        save_path=save_path,
                        data_source=source_labels[source],
                    )
                    root.mainloop()
                else:
                    HeatmapPlot_UKA(
                        debugging_print=self.debugging_print,
                        enrichment_data=enrichment_data,
                        results_data=significant_kinases,
                        y_axis="metric",
                        value_col=self.uka_visualization_column,
                        value_label=self.uka_visualization_label,
                        gui=False,
                        save_path=save_path,
                        data_source=source_labels[source],
                    )
            except ValueError as exc:
                self._dprint(f"     {source} heatmap skipped: {exc}")

    def plot_pathway_overlap_venn(
        self,
        pathway_results_by_condition,
        output_path,
        save_tables=True,
        sources=("KEGG", "WP", "REAC"),
    ):
        """Plot overlaps of enriched pathway terms across condition comparisons.
        
        Args:
            pathway_results_by_condition: Pathway results by condition processed by this function.
            output_path: Path to the output.
            save_tables: Save tables processed by this function.
            sources: Sources processed by this function.
        
        Returns:
            object: Plot output for pathway overlap venn.
        """
        if not pathway_results_by_condition:
            return {}

        output_path = Path(output_path)
        outputs = {}

        for source in sources:
            groups = {}
            for condition, payload in pathway_results_by_condition.items():
                label = self._comparison_label(condition, payload)
                if isinstance(payload, dict):
                    df_pathways = payload.get(f"pathways_{source}")
                else:
                    df_pathways = None
                groups[label] = self._extract_pathway_items(df_pathways)

            if not any(groups.values()):
                self._dprint(f"     No {source} pathways found for overlap plotting.")
                continue

            safe_source = self._safe_filename_token(source)
            save_path = output_path / f"pathways_{safe_source}_overlap.png"
            table_dir = (
                output_path / f"pathways_{safe_source}_overlap_tables"
                if save_tables
                else None
            )

            plotter = VennDiagramPlot(
                groups=groups,
                title=f"{source} pathway overlap",
                item_label="pathways",
                save_path=save_path,
                save_tables_dir=table_dir,
                debugging_print=self.debugging_print,
            )
            fig = plotter.plot()
            import matplotlib.pyplot as plt

            plt.close(fig)

            outputs[source] = {
                "plot": save_path,
                "tables": table_dir,
                "group_sizes": {
                    group_name: len(group_values)
                    for group_name, group_values in plotter.group_sets.items()
                },
            }
            print(f"     {source} pathway overlap diagram saved: {save_path}")

        return outputs

    def run_pathway_enrichment(self, significant_kinases, control=None, condition=None):
        """Run pathway enrichment.
        
        Args:
            significant_kinases: Significant kinases processed by this function.
            control: Control processed by this function.
            condition: Condition processed by this function.
        
        Returns:
            dict: Run result for pathway enrichment.
        """
        print("=====================================================================================")
        print("====================   Starting Pathway Enrichment Stage   ==========================")
        print("=====================================================================================\n")

        df_significant_kinases = self._normalize_significant_kinases(significant_kinases)
        if df_significant_kinases.empty:
            self._dprint("     No significant kinases found for pathway enrichment.")
            pathways = {"KEGG": pd.DataFrame(), "WP": pd.DataFrame(), "REAC": pd.DataFrame()}
        else:
            kinase_list = self._extract_ranked_kinase_list(df_significant_kinases)
            print("[1]  Performing pathway enrichment analysis...")
            pathways = self._profile_sources(kinase_list, ("KEGG", "WP", "REAC"))
            print("     Pathway enrichment analysis completed successfully.")

        if self.heatmap_plot and not df_significant_kinases.empty:
            print("[2]  Plotting pathway heatmaps...")
            self._plot_heatmap(
                significant_kinases=df_significant_kinases,
                pathways=pathways,
                output_path=self.heatmap_plot_output,
                control=control,
                condition=condition,
            )

        self.significant_kinases = df_significant_kinases
        self.pathways = pathways

        print("\n=====================================================================================")
        print("=================   Pathway Enrichment Stage Completed   ============================")
        print("=====================================================================================\n")

        return {
            "significant_kinases": df_significant_kinases,
            "pathways_KEGG": pathways["KEGG"],
            "pathways_WP": pathways["WP"],
            "pathways_REAC": pathways["REAC"],
        }
