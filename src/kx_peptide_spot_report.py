"""Per-spot peptide report.

Builds a single flat table (one row per raw spot x exposure-time measurement)
that ties together, for every peptide:

- ``peptide_change`` (per Control-vs-Test comparison; the table is sorted by it),
- the fitted ``slope`` (per physical spot; exposures collapsed, reused from
  :class:`PeptideStatistics`),
- ``SigmBg``: pyKinaXe has no BioNavigator ``Median_SigmBg``, so the per-spot
  median intensity ``I_median`` is used in its place,
- ``Signal_Saturation`` (per spot x exposure),
- ``Kinases``: all kinases (UniProt ids) that map onto the peptide, from the UKA
  peptide->kinase mapping.

``SigmBg`` and ``Signal_Saturation`` come from the same ``final_output_bn`` rows,
so they are aligned spot for spot. ``slope`` is attached per physical spot (join
key ``[Array, Barcode, Cycle, Row, ID]``); ``peptide_change`` and ``Kinases`` are
peptide-level and repeated across a peptide's spot rows.

Output: one Excel file, one sheet per Control-vs-Test comparison, each sorted by
``peptide_change`` (descending). No figures, no pathways. The script runs the
image and peptide stages itself, then assembles the report::

    python src/kx_peptide_spot_report.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

import pandas as pd


matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_dir in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

from kx_peptide_analysis import PeptideStatistics  # noqa: E402
from kx_upstream_kinase_analysis import KinaseActivityAnalysis  # noqa: E402


# final_output_bn columns we rely on (per spot x exposure-time).
SLOPE_KEYS = ["Array", "Barcode", "Cycle", "Row", "ID"]
SIGNAL_COLUMN = "I_median"  # pyKinaXe's per-spot signal; stands in for "SigmBg".


def _combine_processor_outputs(processor_PTK, processor_STK) -> pd.DataFrame:
    """Concatenate the PTK and STK ``final_output_bn`` tables with an ``Array`` tag.

    Returns:
        pd.DataFrame: One row per raw spot x exposure-time measurement.
    """
    frames = []
    for array_type, processor in (("PTK", processor_PTK), ("STK", processor_STK)):
        bn = getattr(processor, "final_output_bn", None)
        if bn is None or bn.empty:
            continue
        frame = bn.copy()
        frame.insert(0, "Array", array_type)
        frames.append(frame)
    if not frames:
        raise ValueError(
            "No final_output_bn spot data found on the processors. "
            "Run image processing before building the peptide-spot report."
        )
    return pd.concat(frames, ignore_index=True)


def _attach_spot_slopes(spot_df: pd.DataFrame) -> pd.DataFrame:
    """Fit one slope per physical spot and merge it onto the raw rows.

    Reuses :class:`PeptideStatistics`' slope math so the definition matches the
    rest of the pipeline. The slope is fit over exposure times per
    ``[Array, Barcode, Cycle, Row, ID]`` and repeated across that spot's rows.

    Args:
        spot_df (pd.DataFrame): Combined spot x exposure table.

    Returns:
        pd.DataFrame: ``spot_df`` with an added ``slope`` column.
    """
    if "Exposure Time" not in spot_df.columns or SIGNAL_COLUMN not in spot_df.columns:
        spot_df = spot_df.copy()
        spot_df["slope"] = pd.NA
        return spot_df

    pivoted = (
        spot_df.pivot_table(
            index=SLOPE_KEYS,
            columns="Exposure Time",
            values=SIGNAL_COLUMN,
            aggfunc="mean",
        )
        .reset_index()
    )
    pivoted = PeptideStatistics._rename_exposure_columns(pivoted)
    exposure_cols, exposure_times = PeptideStatistics._extract_exposure_columns_and_times(
        pivoted
    )
    pivoted = PeptideStatistics._calculate_slopes_from_pivoted(
        pivoted, exposure_cols, exposure_times
    )
    slope_lookup = pivoted[SLOPE_KEYS + ["slope"]]
    return spot_df.merge(slope_lookup, on=SLOPE_KEYS, how="left")


def build_peptide_to_kinases(df_peptides: pd.DataFrame) -> dict[str, str]:
    """Map every peptide id to the kinases (UniProt ids) that phosphorylate it.

    Reuses the UKA mapping machinery per array branch (PTK/STK). The mapping is
    comparison-independent, so any comparison's peptide table can seed it.

    Args:
        df_peptides (pd.DataFrame): Peptide statistics table (needs ``ID`` and
            ``Type`` columns).

    Returns:
        dict[str, str]: ``{peptide_id: "KIN1,KIN2,..."}`` (comma-joined UniProt ids).
    """
    uka = KinaseActivityAnalysis()
    df_blast = uka._get_blast_data()
    peptide_to_proteins = uka._get_peptide_to_proteins(df_blast)
    df_ptm_stk, df_ptm_ptk = uka._get_ptm_data()

    branches = (
        ("PTK", df_ptm_ptk, uka._get_substrate_to_kinase_lookup(df_ptm_ptk, cache_key="PTK")),
        ("STK", df_ptm_stk, uka._get_substrate_to_kinase_lookup(df_ptm_stk, cache_key="STK")),
    )

    peptide_to_kinases = defaultdict(set)
    for array_type, df_ptm, lookup in branches:
        df_branch = df_peptides[df_peptides["Type"] == array_type].copy()
        if df_branch.empty:
            continue
        df_branch = uka._collapse_duplicate_peptides_for_uka(df_branch)
        kinase_to_peptides, _, _ = uka._map_kinases_to_peptides(
            df_pooled=df_branch,
            df_ptm=df_ptm,
            df_BLAST=df_blast,
            include_mapping_rows=False,
            substrate_to_kinase_entries=lookup,
            peptide_to_proteins=peptide_to_proteins,
        )
        for kinase, peptide_rows in kinase_to_peptides.items():
            for row in peptide_rows:
                peptide_to_kinases[row["peptide_id"]].add(str(kinase))

    return {
        peptide_id: ",".join(sorted(kinases))
        for peptide_id, kinases in peptide_to_kinases.items()
    }


# Column order in the exported sheets (peptide_change first, then the raw
# per-spot values, then the mapped kinases).
_OUTPUT_COLUMNS = [
    "Comparison", "ID", "peptide_change",
    "Array", "Barcode", "Row", "spotRow", "spotCol", "Exposure Time", "Cycle",
    "Sequence", "slope", "SigmBg", "Signal_Saturation", "Kinases",
]


def build_report_frame(spot_df, peptide_change_by_id, peptide_to_kinases, control, condition):
    """Assemble and sort one comparison's per-spot report table.

    Args:
        spot_df (pd.DataFrame): Combined spot table with ``slope`` already attached.
        peptide_change_by_id (dict): ``{peptide_id: peptide_change}`` for this comparison.
        peptide_to_kinases (dict): ``{peptide_id: "kin1,kin2"}``.
        control (str): Control condition label.
        condition (str): Test condition label.

    Returns:
        pd.DataFrame: Sorted report (descending ``peptide_change``).
    """
    out = spot_df.copy()
    out["Comparison"] = f"{control}_vs_{condition}"
    out["peptide_change"] = out["ID"].map(peptide_change_by_id)
    out["Kinases"] = out["ID"].map(peptide_to_kinases).fillna("")
    out = out.rename(columns={SIGNAL_COLUMN: "SigmBg"})

    for col in _OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[_OUTPUT_COLUMNS]

    # Peptides ordered by peptide_change (desc); spots of a peptide stay grouped.
    return out.sort_values(
        ["peptide_change", "ID", "Array", "Barcode", "Row", "Exposure Time"],
        ascending=[False, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_peptide_spot_report(processor_PTK, processor_STK, peptide_results, output_path):
    """Build the per-spot peptide report and write one Excel (a sheet per comparison).

    Args:
        processor_PTK: PTK image processor (with ``final_output_bn``).
        processor_STK: STK image processor (with ``final_output_bn``).
        peptide_results: ``{condition: payload}`` from the peptide-statistics stage.
        output_path: Path of the ``.xlsx`` to write.

    Returns:
        dict: ``{comparison_label: report_dataframe}``.
    """
    spot_df = _combine_processor_outputs(processor_PTK, processor_STK)
    spot_df = _attach_spot_slopes(spot_df)

    # The peptide->kinase mapping is comparison-independent; seed it once.
    seed_peptides = next(iter(peptide_results.values()))["peptide_statistics"]
    peptide_to_kinases = build_peptide_to_kinases(seed_peptides)

    reports = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for comparison_key, payload in peptide_results.items():
            control = payload["control_condition"]
            # The dict key identifies the comparison; the sheet name is built
            # from the control/test labels of the pair.
            condition = payload["condition"]
            stats = payload["peptide_statistics"]
            change_by_id = dict(zip(stats["ID"], stats["peptide_change"]))
            report = build_report_frame(
                spot_df, change_by_id, peptide_to_kinases, control, condition
            )
            label = f"{control}_vs_{condition}"
            reports[label] = report
            # Excel sheet names are capped at 31 chars.
            report.to_excel(writer, sheet_name=label[:31], index=False)
    print(
        f"Wrote peptide-spot report: {len(reports)} comparison sheet(s), "
        f"{len(spot_df)} spot rows each -> {output_path}"
    )
    return reports


def main():
    """Standalone entry point: run image + peptide stages, then build the report."""
    from kx_pipeline_tools import (
        build_results_dir,
        run_image_analysis_pipeline,
        run_peptide_statistics_analysis,
    )

    (
        _loader_ptk,
        _loader_stk,
        enricher,
        processor_ptk,
        processor_stk,
        analysis_timestamp,
        _pipeline_duration,
        experiment_name,
        results_parent_relpath,
    ) = run_image_analysis_pipeline()

    peptide_results, _ = run_peptide_statistics_analysis(
        enricher=enricher,
        processor_PTK=processor_ptk,
        processor_STK=processor_stk,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
    )

    output_path = (
        build_results_dir(
            analysis_timestamp,
            experiment_name,
            results_parent_relpath=results_parent_relpath,
        )
        / f"{analysis_timestamp}_downstream_analysis"
        / f"{analysis_timestamp}_peptide_spot_report.xlsx"
    )
    build_peptide_spot_report(processor_ptk, processor_stk, peptide_results, output_path)


if __name__ == "__main__":
    main()
