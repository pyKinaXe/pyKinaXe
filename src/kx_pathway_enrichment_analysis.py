"""Pathway enrichment of the significant kinases with g:Profiler.

PathwayEnrichmentAnalysis takes the kinase results of kx_upstream_kinase_analysis,
queries g:Profiler for KEGG, Reactome and WikiPathways, writes one pathway table
per condition and source, and produces the pathway heatmaps and the sets used
for the overlap diagrams.

The kinases are sent as gene symbols, not as UniProt accessions: g:Profiler
excludes every query id that Ensembl links to more than one gene, without an
error in the result, and many kinase accessions are linked to readthrough,
antisense or unrelated genes as well (P28482, MAPK1, also to GPSM2).
"""

from functools import lru_cache
from pathlib import Path
import re

import pandas as pd
import requests

from config.analysis_modules import PATHWAY_ENRICHMENT_DEFAULTS
from kx_plot_results import (
    HeatmapPlot_UKA,
    HeatmapPlot_UKA_Comparisons,
    UKAComparison,
    VennDiagramPlot,
)


# Columns of a g:GOSt result table, in the order the gprofiler client
# (gprofiler-official 1.0) returned them, so the pathway tables keep their layout.
GOST_RESULT_COLUMNS = (
    "source",
    "native",
    "name",
    "p_value",
    "significant",
    "description",
    "term_size",
    "query_size",
    "intersection_size",
    "effective_domain_size",
    "precision",
    "recall",
    "query",
    "parents",
    "intersections",
    "evidences",
)


@lru_cache(maxsize=None)
def load_uniprot_gene_symbols(path=PATHWAY_ENRICHMENT_DEFAULTS["gene_symbols_path"]):
    """Return ``{accession: primary gene symbol}`` from a UniProt gene-name table.

    The table has the columns ``Entry`` and ``Gene Names (primary)``. An entry
    encoded by several genes lists them as ``'HBA1; HBA2'``; the first is used.

    Args:
        path: TSV file; by default the release the PTM tables were built from.

    Returns:
        dict: Accession -> gene symbol. Empty, with a warning, when the file is
            missing; the kinases are then sent as accessions.
    """
    path = Path(path)
    if not path.is_file():
        print(
            f"     WARNING: gene symbol table {path} not found; the pathway query "
            "falls back to UniProt accessions, which g:Profiler drops whenever "
            "Ensembl links them to more than one gene."
        )
        return {}
    table = pd.read_csv(path, sep="\t", dtype=str).dropna(
        subset=["Entry", "Gene Names (primary)"]
    )
    return {
        accession.strip(): names.split(";")[0].strip()
        for accession, names in zip(table["Entry"], table["Gene Names (primary)"])
        if names.strip()
    }


def gprofiler_gene_symbols(kinase_ids):
    """Return the gene symbol g:Profiler is queried with for each kinase id.

    A UniProt accession, with or without isoform suffix, becomes its primary gene
    symbol. Any other id (already a gene symbol, or an accession without a symbol
    in the table, such as a deleted entry) is sent unchanged.

    Args:
        kinase_ids: Kinase ids as they appear in the kinase tables.

    Returns:
        dict: ``{kinase id: query gene}`` in the order of ``kinase_ids``.
    """
    symbols = load_uniprot_gene_symbols()
    genes = {}
    for kinase_id in kinase_ids:
        kinase_id = str(kinase_id).strip()
        if kinase_id:
            genes[kinase_id] = (
                symbols.get(kinase_id)
                or symbols.get(kinase_id.split("-")[0])
                or kinase_id
            )
    return genes


def query_gprofiler(
    query,
    sources,
    user_threshold,
    background=None,
    url=PATHWAY_ENRICHMENT_DEFAULTS["gprofiler_profile_url"],
    timeout=120,
):
    """Run one g:GOSt enrichment query.

    g:SCS correction, IEA annotations included, unordered query, numeric ids read
    as ENTREZGENE_ACC. The statistical domain is every gene annotated in the
    source (``domain_scope`` 'annotated') or, with a ``background``, the
    background genes annotated in it ('custom_annotated'). The gprofiler client
    always sends 'custom' with a background, which also counts background genes
    without any annotation, so the REST API is called directly.

    Args:
        query: Gene ids of the query.
        sources: g:Profiler source ids, e.g. ``["KEGG", "WP", "REAC"]``.
        user_threshold: Significance threshold of the g:SCS correction.
        background: Gene ids of a custom background, or None.
        url: g:GOSt endpoint.
        timeout: Seconds to wait for the response.

    Returns:
        tuple[pd.DataFrame, dict]: The significant terms in the gprofiler
            client's layout (``intersections`` lists the query ids annotated to
            each term) and the response metadata (``version``; ``genes_metadata``
            with the ``failed`` and ``ambiguous`` query ids).

    Raises:
        requests.RequestException: The request failed or g:Profiler answered
            with an error.
    """
    response = requests.post(
        url,
        json={
            "organism": "hsapiens",
            "query": list(query),
            "sources": list(sources),
            "user_threshold": user_threshold,
            "all_results": False,
            "ordered": False,
            "no_evidences": False,
            "combined": False,
            "measure_underrepresentation": False,
            "no_iea": False,
            "numeric_ns": "ENTREZGENE_ACC",
            "domain_scope": "annotated" if background is None else "custom_annotated",
            "significance_threshold_method": "g_SCS",
            "background": "" if background is None else list(background),
        },
        headers={"User-Agent": "pyKinaXe"},
        timeout=timeout,
    )
    if response.status_code != 200:
        try:
            message = response.json()["message"]
        except (ValueError, KeyError, TypeError):
            message = f"HTTP {response.status_code}"
        raise requests.HTTPError(
            f"g:Profiler query failed: {message}", response=response
        )
    payload = response.json()
    meta = payload["meta"]

    # A term's 'intersections' holds one evidence list per entry of the query's
    # 'ensgs', empty when that gene is not annotated to the term. The query id
    # behind an Ensembl gene comes from the one-to-one mappings (ambiguous ids
    # never reach 'ensgs'), exactly as the gprofiler client reconstructs it.
    query_meta = meta["genes_metadata"]["query"]["query_1"]
    query_id_of = {
        ensgs[0]: query_id
        for query_id, ensgs in query_meta["mapping"].items()
        if len(ensgs) == 1
    }
    genes = [query_id_of.get(ensg, ensg) for ensg in query_meta["ensgs"]]
    rows = []
    for term in payload["result"]:
        evidence = term["intersections"]
        rows.append(
            {
                **term,
                "evidences": [codes for codes in evidence if codes],
                "intersections": [
                    gene for codes, gene in zip(evidence, genes) if codes
                ],
            }
        )
    return pd.DataFrame(rows, columns=list(GOST_RESULT_COLUMNS)), meta


def profile_kinases(
    kinase_ids, sources, user_threshold, background_ids=None, label=None
):
    """Enrich kinases with g:Profiler, sending them as gene symbols.

    ``intersections`` of the returned table lists the kinase ids as given, so the
    result joins back onto the kinase tables. Query genes g:Profiler could not
    use (not found, or linked to several Ensembl genes) are printed, since the
    result table itself does not show them.

    Args:
        kinase_ids: Kinase ids of the query (UniProt accessions or gene symbols).
        sources: g:Profiler source ids.
        user_threshold: Significance threshold of the g:SCS correction.
        background_ids: Kinase ids of a custom background, or None for every
            annotated gene. The query kinases are added to it, so the query
            always lies inside the domain.
        label: Name of the query in the printed lines, e.g. the comparison.

    Returns:
        pd.DataFrame: The significant terms, see :func:`query_gprofiler`.
    """
    gene_of = gprofiler_gene_symbols(kinase_ids)
    if not gene_of:
        return pd.DataFrame(columns=list(GOST_RESULT_COLUMNS))
    kinase_ids_of_gene = {}
    for kinase_id, gene in gene_of.items():
        kinase_ids_of_gene.setdefault(gene, []).append(kinase_id)
    background = None
    if background_ids is not None:
        background = list(
            dict.fromkeys(gprofiler_gene_symbols([*background_ids, *gene_of]).values())
        )

    df, meta = query_gprofiler(
        list(kinase_ids_of_gene), sources, user_threshold, background=background
    )

    tag = f" ({label})" if label else ""
    domain = (
        "all annotated genes"
        if background is None
        else f"the {len(background)} background kinases annotated in each source"
    )
    print(
        f"     g:Profiler {meta.get('version', 'unknown version')}{tag}: "
        f"{len(gene_of)} kinases as {len(kinase_ids_of_gene)} gene symbols, "
        f"domain: {domain}"
    )
    genes_meta = meta.get("genes_metadata", {})
    failed = list(genes_meta.get("failed") or [])
    ambiguous = dict(genes_meta.get("ambiguous") or {})
    if failed or ambiguous:

        def _describe(gene, reason):
            """Return ``'GENE = ACCESSION (reason)'`` for one unused query gene."""
            kinases = [k for k in kinase_ids_of_gene.get(gene, []) if k != gene]
            return f"{gene}{' = ' + '/'.join(kinases) if kinases else ''} ({reason})"

        left_out = [_describe(gene, "not found") for gene in failed] + [
            _describe(gene, f"{len(candidates)} Ensembl genes")
            for gene, candidates in ambiguous.items()
        ]
        print(
            f"     WARNING: g:Profiler left out {len(left_out)} of "
            f"{len(kinase_ids_of_gene)} query genes{tag}: " + "; ".join(left_out)
        )

    df["intersections"] = [
        [kinase for gene in genes for kinase in kinase_ids_of_gene.get(gene, [gene])]
        for genes in df["intersections"]
    ]
    return df


class PathwayEnrichmentAnalysis:
    """Stage 3 of the UKA/KPEA workflow: pathway enrichment and heatmaps."""

    UKA_METRIC_OPTIONS = dict(PATHWAY_ENRICHMENT_DEFAULTS["uka_metric_options"])
    ALLOWED_PATHWAY_BACKGROUNDS = tuple(
        PATHWAY_ENRICHMENT_DEFAULTS["allowed_pathway_backgrounds"]
    )

    def __init__(
        self,
        significance_level_pathways=PATHWAY_ENRICHMENT_DEFAULTS[
            "default_significance_level_pathways"
        ],
        heatmap_plot=False,
        heatmap_plot_output=None,
        uka_visualization_metric=PATHWAY_ENRICHMENT_DEFAULTS[
            "default_uka_visualization_metric"
        ],
        pathway_background=PATHWAY_ENRICHMENT_DEFAULTS["default_pathway_background"],
        background_kinases=None,
        debugging_print=True,
    ):
        """Store the enrichment settings.

        Args:
            pathway_background: ``'annotated'`` (every gene annotated in the
                source) or ``'mappable_kinases'`` (only ``background_kinases``).
            background_kinases: Kinase ids of the ``'mappable_kinases'``
                background; ignored for ``'annotated'``.
            debugging_print: Whether to print additional debug information.
        """
        self.significance_level_pathways = significance_level_pathways
        self.heatmap_plot = heatmap_plot
        self.heatmap_plot_output = (
            Path(heatmap_plot_output) if heatmap_plot_output is not None else None
        )
        self.debugging_print = debugging_print

        resolved_background = str(pathway_background).lower()
        if resolved_background not in self.ALLOWED_PATHWAY_BACKGROUNDS:
            raise ValueError(
                f"Unknown pathway_background '{pathway_background}'. "
                "Use 'annotated' or 'mappable_kinases'."
            )
        if resolved_background == "mappable_kinases" and not background_kinases:
            raise ValueError(
                "pathway_background 'mappable_kinases' needs the mappable kinases "
                "as background_kinases."
            )
        self.pathway_background = resolved_background
        self.background_kinases = (
            list(background_kinases)
            if resolved_background == "mappable_kinases"
            else None
        )

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
        """Print a message only when debug logging is enabled."""
        if self.debugging_print:
            print(*args, **kwargs)

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
    def _extract_pathway_items(df_pathways):
        if df_pathways is None or df_pathways.empty:
            return []

        for column in ("native", "term_id", "id", "name"):
            if column in df_pathways.columns:
                return df_pathways[column].dropna().astype(str).tolist()
        return []

    def _normalize_significant_kinases(self, significant_kinases):
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

    def _profile_source(self, kinase_list, source, label=None):
        if not kinase_list:
            return pd.DataFrame()

        try:
            return profile_kinases(
                kinase_list,
                sources=[source],
                user_threshold=self.significance_level_pathways,
                background_ids=self.background_kinases,
                label=label,
            )
        except Exception as exc:
            # Unconditional print: the empty table written for this source would
            # otherwise be indistinguishable from "no enriched pathway".
            print(
                f"     WARNING: Pathway enrichment failed for source={source}"
                f"{f' ({label})' if label else ''}; its table stays EMPTY, which "
                f"is a query failure, not a negative result. Reason: {exc}"
            )
            return pd.DataFrame()

    def _profile_sources(self, kinase_list, sources, label=None):
        sources = tuple(sources)
        if not kinase_list:
            return {source: pd.DataFrame() for source in sources}

        try:
            profiled = profile_kinases(
                kinase_list,
                sources=list(sources),
                user_threshold=self.significance_level_pathways,
                background_ids=self.background_kinases,
                label=label,
            )
        except Exception as exc:
            self._dprint(
                "     WARNING: Combined pathway enrichment request failed; "
                f"falling back to per-source requests. Reason: {exc}"
            )
            return {
                source: self._profile_source(kinase_list, source, label=label)
                for source in sources
            }

        if profiled.empty:
            return {source: pd.DataFrame(columns=profiled.columns) for source in sources}

        return {
            source: profiled[profiled["source"] == source].copy()
            for source in sources
        }

    def _plot_heatmap(self, significant_kinases, pathways, output_path=None, control=None, condition=None):
        """Plot the kinase-pathway heatmap of ONE comparison, one figure per source."""
        source_labels = {
            "KEGG": "KEGG",
            "WP": "WP",
            "REAC": "REAC",
        }
        title = self._comparison_label(
            condition, {"control_condition": control, "condition": condition}
        )

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
                HeatmapPlot_UKA(
                    debugging_print=self.debugging_print,
                    enrichment_data=enrichment_data,
                    results_data=significant_kinases,
                    y_axis="metric",
                    value_col=self.uka_visualization_column,
                    value_label=self.uka_visualization_label,
                    save_path=save_path,
                    data_source=source_labels[source],
                    title=title,
                )
            except ValueError as exc:
                self._dprint(f"     {source} heatmap skipped: {exc}")

    def _plot_comparison_heatmaps(self, all_results, output_path=None, layouts=None):
        """Draw the cross-comparison pathway heatmaps: one figure per source and layout.

        Every control/test comparison of the run becomes one slice (``split``
        layout) or one panel (``panels`` layout) of the same heatmap, drawn on
        the union of the enriched pathways and significant kinases with one
        shared colorbar. Needs at least two comparisons.

        Args:
            all_results: Mapping comparison key -> condition results as built by
                ``run_uka_analysis`` (``control_condition``, ``condition``,
                ``significant_kinases``, ``pathways_<SOURCE>``).
            output_path: Folder for the PNG files; None only draws.
            layouts: Layout names; the heatmap config's
                ``comparison_heatmap.layouts`` when None.

        Returns:
            dict: ``{source: {layout: save_path}}`` of the figures drawn.
        """
        if layouts is None:
            layouts = HeatmapPlot_UKA_Comparisons.configured_layouts()
        if not layouts or len(all_results) < 2:
            return {}

        sources = ("KEGG", "WP", "REAC")
        comparisons = {source: [] for source in sources}
        for key, results in all_results.items():
            control = results.get("control_condition")
            condition = results.get("condition", key)
            label = self._comparison_label(key, results)
            significant = self._normalize_significant_kinases(
                results.get("significant_kinases")
            )
            for source in sources:
                df_pathways = results.get(f"pathways_{source}")
                if (
                    df_pathways is None
                    or df_pathways.empty
                    or "name" not in df_pathways.columns
                    or "intersections" not in df_pathways.columns
                ):
                    enrichment = pd.DataFrame(columns=["name", "intersections"])
                else:
                    columns = ["name", "intersections"]
                    if "p_value" in df_pathways.columns:
                        columns.append("p_value")
                    enrichment = df_pathways[columns].copy()
                comparisons[source].append(
                    UKAComparison(
                        label=label,
                        results_data=significant,
                        enrichment_data=enrichment,
                        condition=condition,
                        control=control,
                    )
                )

        # The orientation goes into the file name, so a run with transpose
        # switched on does not overwrite the pathway-row version.
        suffix = "_kinase_rows" if HeatmapPlot_UKA_Comparisons.configured_transpose() else ""
        outputs = {}
        for source in sources:
            if all(item.enrichment_data.empty for item in comparisons[source]):
                self._dprint(
                    f"     No significant {source} pathways in any comparison; "
                    "comparison heatmap skipped."
                )
                continue
            for layout in layouts:
                save_path = (
                    str(
                        Path(output_path)
                        / f"{source}_UKA_heatmap_comparisons_{layout}{suffix}.png"
                    )
                    if output_path is not None
                    else None
                )
                try:
                    HeatmapPlot_UKA_Comparisons(
                        comparisons=comparisons[source],
                        value_col=self.uka_visualization_column,
                        value_label=self.uka_visualization_label,
                        layout=layout,
                        save_path=save_path,
                        data_source=source,
                        debugging_print=self.debugging_print,
                    )
                except ValueError as exc:
                    self._dprint(f"     {source} comparison heatmap ({layout}) skipped: {exc}")
                    continue
                outputs.setdefault(source, {})[layout] = save_path
                if save_path is not None:
                    print(f"     {source} comparison heatmap ({layout}) saved: {save_path}")
        return outputs

    def plot_pathway_overlap_venn(
        self,
        pathway_results_by_condition,
        output_path,
        save_tables=True,
        sources=("KEGG", "WP", "REAC"),
    ):
        """Plot overlaps of enriched pathway terms across condition comparisons."""
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
        print("\n" + "=" * 80)
        print("Starting Pathway Enrichment Stage...")
        print("=" * 80)

        df_significant_kinases = self._normalize_significant_kinases(significant_kinases)
        if df_significant_kinases.empty:
            self._dprint("     No significant kinases found for pathway enrichment.")
            pathways = {"KEGG": pd.DataFrame(), "WP": pd.DataFrame(), "REAC": pd.DataFrame()}
        else:
            kinase_list = self._extract_ranked_kinase_list(df_significant_kinases)
            print("[1]  Performing pathway enrichment analysis...")
            pathways = self._profile_sources(
                kinase_list,
                ("KEGG", "WP", "REAC"),
                label=self._comparison_label(
                    condition, {"control_condition": control, "condition": condition}
                ),
            )
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

        print("\n" + "=" * 80)
        print("Pathway Enrichment Stage Completed")
        print("=" * 80 + "\n")

        return {
            "significant_kinases": df_significant_kinases,
            "pathways_KEGG": pathways["KEGG"],
            "pathways_WP": pathways["WP"],
            "pathways_REAC": pathways["REAC"],
        }
