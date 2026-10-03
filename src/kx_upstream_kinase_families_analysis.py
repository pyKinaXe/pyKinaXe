"""Upstream kinase-family activity analysis.

KinaseFamiliesActivityAnalysis runs the same hypergeometric enrichment as the
single-kinase UKA (KinaseActivityAnalysis), but scores kinase families (the
``ptm_enzyme_family`` column of the PTM interaction files) instead of
individual kinases (``ptm_enzyme``). Families are the unit KRSA scores, which
makes the results comparable to KRSA's.

Compared with the single-kinase UKA: the substrate lookup is keyed by
``ptm_enzyme_family``; no UniProt name resolution is needed, so
``_resolve_kinase_names_uniprot`` is the identity; the ranked table
de-duplicates on ``["Kinase", "Type"]`` (``RANK_DEDUP_SUBSET``), because broad
families occur in both the STK and the PTK interaction file and are scored once
per array; the null population is the whole KRSA mapping table
(``default_kpea_background_universe: chip``); and the significance threshold is
|Z| >= 2, KRSA's hit convention. The scoring itself (hit definition,
hypergeometric null, Z-score, significance flag) is inherited unchanged and is
deterministic, since the null is described by its exact moments.

Defaults come from ``config/upstream_kinase_families_analysis.yaml``. The family
threshold (2.0) is higher than the single-kinase threshold (1.3) because a
family carries more substrates and its Z has more resolution.

The module writes the family results to its own CSV tables and produces no
figures or pathway outputs. The terminal pipeline calls
``run_kinase_families_analysis`` inside the timed analysis, after the
single-kinase and pathway stages; the results are not fed into any downstream
stage. Standalone use, which runs the image and
peptide stages first::

    python src/kx_upstream_kinase_families_analysis.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

import numpy as np
import pandas as pd


# Force a non-interactive backend so the script runs headlessly.
matplotlib.use("Agg")

# Allow running this file directly without installing the package or setting
# PYTHONPATH: put the repository root (for the ``config`` package) and ``src``
# (for sibling modules) on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_dir in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

from config.upstream_kinase_families_analysis import (  # noqa: E402
    UPSTREAM_KINASE_FAMILIES_ANALYSIS_DEFAULTS as FAMILY_DEFAULTS,
)
from kx_peptide_ids import normalize_peptide_id  # noqa: E402
from kx_upstream_kinase_analysis import KinaseActivityAnalysis  # noqa: E402


FAMILY_COLUMN = "ptm_enzyme_family"


class KinaseFamiliesActivityAnalysis(KinaseActivityAnalysis):
    """Family-level variant of :class:`KinaseActivityAnalysis`.

    The scoring machinery is inherited; only the grouping unit changes from the
    individual kinase (``ptm_enzyme``) to its family (``ptm_enzyme_family``).
    """

    # Keep a family's STK and PTK results as separate rows in the ranked output.
    # Many broad families (e.g. ``Tyr protein kinase``, ``PKC``, ``MAP kinase
    # kinase``) are annotated on enzymes present in both interaction files, so a
    # single family is legitimately scored once per array. De-duplicating on the
    # family label alone (the single-kinase default) would silently drop one branch
    # and, because rows rank by |Z|, could keep the worse-supported branch.
    RANK_DEDUP_SUBSET = ["Kinase", "Type"]

    # Columns of the optional debug mapping table (mirrors the parent's layout).
    MAPPING_COLUMNS = [
        "Peptide_ID", "Peptide_UniprotName", "Peptide_UniprotID",
        "Peptide_CandidateSites", "Substrate_BLAST", "Matched_PTM_Sites",
        "Kinase_UniprotID", "Kinase_UniprotName", "Source_Database",
    ]

    def __init__(
        self,
        *args,
        family_mapping_source=FAMILY_DEFAULTS["family_mapping_source"],
        krsa_ptk_mapping_path=FAMILY_DEFAULTS["krsa_ptk_mapping_path"],
        krsa_stk_mapping_path=FAMILY_DEFAULTS["krsa_stk_mapping_path"],
        **kwargs,
    ):
        """Initialize the family analysis.

        Args:
            family_mapping_source: 'krsa' (default) uses KRSA's curated
                peptide->family tables; 'uniprot' derives families from
                ``ptm_enzyme_family`` (BLAST + PTM + UniProt).
            krsa_ptk_mapping_path: KRSA PTK peptide->family CSV (Substrates, Kinases).
            krsa_stk_mapping_path: KRSA STK peptide->family CSV.
            *args, **kwargs: Forwarded to :class:`KinaseActivityAnalysis`.
        """
        super().__init__(*args, **kwargs)
        source = str(family_mapping_source).strip().lower()
        allowed = {s.lower() for s in FAMILY_DEFAULTS["allowed_family_mapping_sources"]}
        if source not in allowed:
            raise ValueError(
                f"Unknown family_mapping_source '{family_mapping_source}'. "
                f"Allowed: {sorted(allowed)}."
            )
        self.family_mapping_source = source
        self.krsa_ptk_mapping_path = Path(krsa_ptk_mapping_path)
        self.krsa_stk_mapping_path = Path(krsa_stk_mapping_path)
        self._krsa_family_cache = {}

    @staticmethod
    def _normalize_krsa_peptide_id(peptide_id):
        """Normalize a peptide id so KRSA and pyKinaXe forms match.

        KRSA writes mutant peptides as ``..._C1320K/C1321K`` and some array
        layouts carry a stray space (``JAK1_ 1027_1039``); :mod:`kx_peptide_ids`
        removes both, on every side of every id comparison, so the two spellings
        meet.
        """
        return normalize_peptide_id(peptide_id)

    def _get_krsa_family_map(self, array_type):
        """Load and cache KRSA's ``{peptide_id: (families,)}`` map for one array.

        Args:
            array_type (str): "PTK" or "STK".

        Returns:
            dict[str, tuple[str, ...]]: normalized peptide id -> family labels.
        """
        array_type = str(array_type).upper()
        if array_type in self._krsa_family_cache:
            return self._krsa_family_cache[array_type]
        path = (
            self.krsa_ptk_mapping_path
            if array_type == "PTK"
            else self.krsa_stk_mapping_path
        )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"KRSA {array_type} mapping file not found: {path}. "
                "Provide it or set family_mapping_source='uniprot'."
            )
        df = pd.read_csv(path)
        mapping = {}
        for row in df.itertuples(index=False):
            peptide_id = self._normalize_krsa_peptide_id(row.Substrates)
            families = tuple(
                fam.strip() for fam in str(row.Kinases).split() if fam.strip()
            )
            if peptide_id:
                mapping[peptide_id] = families
        self._krsa_family_cache[array_type] = mapping
        return mapping

    @staticmethod
    def _infer_array_type(df_pooled):
        """Determine 'PTK'/'STK' from a branch's peptide table 'Type' column."""
        if "Type" in df_pooled.columns:
            for value in df_pooled["Type"].dropna().astype(str):
                token = value.strip().upper()
                if token in ("PTK", "STK"):
                    return token
        raise ValueError(
            "Cannot determine array type (PTK/STK) for the KRSA mapping: "
            "peptide table has no usable 'Type' column."
        )

    def _map_kinases_to_peptides(
        self,
        df_pooled,
        df_ptm=None,
        df_BLAST=None,
        include_mapping_rows=True,
        substrate_to_kinase_entries=None,
        peptide_to_proteins=None,
    ):
        """Map peptides to families for scoring.

        In KRSA mode the family assignment is read directly from KRSA's curated
        peptide->family table (BLAST/PTM are bypassed entirely). In UniProt mode
        this delegates to the parent (BLAST + PTM + ``ptm_enzyme_family``).

        Returns:
            tuple: ``(kinase_to_peptides, mapping_rows_df, peptide_order)`` with the
            same shape the parent produces, so :meth:`_calculate_KPEA` is unchanged.
        """
        if self.family_mapping_source != "krsa":
            return super()._map_kinases_to_peptides(
                df_pooled,
                df_ptm,
                df_BLAST,
                include_mapping_rows,
                substrate_to_kinase_entries,
                peptide_to_proteins,
            )

        krsa_map = self._get_krsa_family_map(self._infer_array_type(df_pooled))
        kinase_to_peptides = defaultdict(list)
        peptide_order = []
        mapping_rows = []
        seen = set()
        for peptide_id in df_pooled["ID"].astype(str):
            if peptide_id in seen:
                continue
            seen.add(peptide_id)
            peptide_order.append(peptide_id)
            families = krsa_map.get(self._normalize_krsa_peptide_id(peptide_id), ())
            for family in families:
                kinase_to_peptides[family].append({"peptide_id": peptide_id})
                if include_mapping_rows:
                    mapping_rows.append(
                        {
                            "Peptide_ID": peptide_id,
                            "Kinase_UniprotID": family,
                            "Kinase_UniprotName": family,
                            "Source_Database": "KRSA",
                        }
                    )
        df_mapping = (
            pd.DataFrame(mapping_rows, columns=self.MAPPING_COLUMNS)
            if include_mapping_rows
            else pd.DataFrame()
        )
        return dict(kinase_to_peptides), df_mapping, peptide_order

    def _build_substrate_to_kinase_lookup(self, df_ptm):
        """Build a substrate -> family lookup (same shape as the parent's).

        Identical to :meth:`KinaseActivityAnalysis._build_substrate_to_kinase_lookup`
        except the ``"kinase"`` key holds the enzyme's ``ptm_enzyme_family`` label
        rather than its UniProt id. Rows whose family is blank/unknown are skipped
        (those enzymes cannot be grouped into a family).

        Args:
            df_ptm: PTM interaction table (must contain ``ptm_enzyme_family``).

        Returns:
            dict: ``{substrate: [ {kinase(=family), databases, ptm_sites, ...} ]}``.
        """
        # In KRSA mode the peptide->family mapping is read from KRSA's curated
        # table (see _map_kinases_to_peptides), so this PTM-derived lookup is
        # unused: skip the expensive build and avoid needing ptm_enzyme_family.
        if getattr(self, "family_mapping_source", "uniprot") == "krsa":
            return {}

        if FAMILY_COLUMN not in df_ptm.columns:
            raise KeyError(
                f"PTM interaction data is missing the '{FAMILY_COLUMN}' column. "
                "Regenerate the interaction files with enzyme annotation enabled "
                "(`python src/kx_data_enricher.py omnipath`)."
            )

        substrate_lookup = defaultdict(
            lambda: defaultdict(lambda: {"databases": set(), "ptm_sites": set()})
        )

        has_source_database = "source_database" in df_ptm.columns
        has_source = "source" in df_ptm.columns
        has_site = "site" in df_ptm.columns

        for row in df_ptm.itertuples(index=False):
            substrate = row.uniprot_id
            family = getattr(row, FAMILY_COLUMN, "")
            # Skip enzymes without a resolved family; they have no grouping key.
            if family is None or (isinstance(family, float) and pd.isna(family)):
                continue
            family = str(family).strip()
            if not family:
                continue

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
            entry = substrate_lookup[substrate][family]
            entry["databases"].add(database)
            entry["ptm_sites"].update(
                site_token.upper()
                for site_token in self._extract_informative_ptm_site_tokens(site_value)
            )

        normalized_lookup = {}
        for substrate, family_map in substrate_lookup.items():
            entries = []
            for family, payload in family_map.items():
                ptm_sites = frozenset(payload["ptm_sites"])
                positioned_ptm_sites = frozenset(
                    site for site in ptm_sites if self._is_known_ptm_site(site)
                )
                entries.append(
                    {
                        "kinase": family,
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

    def _chip_background_counts(
        self, kinase_ids, df_pooled, measured_counts, n_measured
    ):
        """Chip-wide null population taken straight from KRSA's mapping table.

        This is the exact population ``krsa()`` samples from: every peptide listed in
        ``KRSA_Mapping_<chip>.csv`` (M = 193 PTK / 141 STK), with K counted over that
        same table -- not over the peptides this run measured. It is what makes the
        family z-scores directly comparable to KRSA's ``AvgZ``.

        Falls back to the parent (which maps the peptide-enrichment reference) when
        the family mapping is not the KRSA one.

        Returns:
            tuple | None: ``(n_background, substrate_counts)`` aligned to
                ``kinase_ids``, or ``None`` to keep the measured population.
        """
        if self.family_mapping_source != "krsa":
            return super()._chip_background_counts(
                kinase_ids, df_pooled, measured_counts, n_measured
            )

        array_type = self._kpea_array_type(df_pooled)
        if array_type is None:
            return None
        krsa_map = self._get_krsa_family_map(array_type)
        if not krsa_map:
            return None

        per_family = defaultdict(int)
        for families in krsa_map.values():
            for family in set(families):
                per_family[family] += 1
        counts = np.array(
            [int(per_family.get(kinase, 0)) for kinase in kinase_ids], dtype=np.int64
        )
        measured_counts = np.asarray(measured_counts, dtype=np.int64)
        n_background = len(krsa_map)
        if n_background < int(n_measured) or np.any(counts < measured_counts):
            self._dprint(
                f"     KPEA: KRSA {array_type} mapping is smaller than the measured "
                "population; keeping the measured peptides as the null population."
            )
            return None
        return n_background, counts

    def _resolve_kinase_names_uniprot(self, uniprot_ids, batch_size=100):
        """Identity map: the scoring unit is already a family label, not a UniProt id.

        Args:
            uniprot_ids: The family labels used as the scoring unit.
            batch_size: Unused; kept for signature compatibility.

        Returns:
            dict: ``{family_label: family_label}``.
        """
        return {
            str(value).strip(): str(value).strip()
            for value in uniprot_ids
            if str(value).strip()
        }


def build_family_analysis(debugging_print: bool = False) -> KinaseFamiliesActivityAnalysis:
    """Construct the family analysis worker from the family config defaults.

    Shared runtime parameters (permutations, BLAST threshold, mapping-strictness
    flags, ...) use the class defaults; only the family-config-controlled
    parameters (chiefly the higher Z-score threshold) are overridden here.
    Figures are disabled.

    Args:
        debugging_print (bool): Print verbose per-stage diagnostics.

    Returns:
        KinaseFamiliesActivityAnalysis: The configured worker.
    """
    fam = FAMILY_DEFAULTS
    return KinaseFamiliesActivityAnalysis(
        path_file_enrichment_peptides=fam["path_file_enrichment_peptides"],
        input_stk_ptm_path=fam["input_stk_ptm_path"],
        input_ptk_ptm_path=fam["input_ptk_ptm_path"],
        verified_evidence_levels=fam["default_verified_evidence_levels"],
        kpea_lfc_cutoffs=fam["default_kpea_lfc_cutoffs"],
        kpea_cutoff_mode=fam["default_kpea_cutoff_mode"],
        kpea_substrate_cutoff=fam["default_kpea_substrate_cutoff"],
        kpea_zscore_threshold=fam["default_kpea_zscore_threshold"],
        kpea_z_cap=fam["default_kpea_z_cap"],
        kpea_background_universe=fam["default_kpea_background_universe"],
        family_mapping_source=fam["family_mapping_source"],
        krsa_ptk_mapping_path=fam["krsa_ptk_mapping_path"],
        krsa_stk_mapping_path=fam["krsa_stk_mapping_path"],
        debugging_print=debugging_print,
    )


def _prepare_family_frame(df, control, condition):
    """Tidy a per-condition family result for the CSV export.

    Args:
        df: The ``all_kinases`` / ``significant_kinases`` DataFrame (Kinase = family).
        control: Control condition label.
        condition: Test condition label.

    Returns:
        pd.DataFrame: A copy with a Comparison column and family-named columns.
    """
    out = df.copy()
    rename = {}
    if "Kinase" in out.columns:
        rename["Kinase"] = "Kinase_Family"
    out = out.rename(columns=rename)
    drop_cols = [c for c in ("Kinase_Name", "Kinase_Name_Resolved") if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out.insert(0, "Comparison", f"{control}_vs_{condition}")
    return out


def run_kinase_families_analysis(
    peptide_results,
    output_dir,
    family_analysis=None,
    debugging_print: bool = False,
    output_mode: str = "user",
):
    """Run the family-level analysis for every condition and save its CSV tables.

    One CSV per comparison, ordered by |Z_Score| so the hits come first. Every
    family is in it -- there is no separate significance file, the ``Significant``
    column marks them.

    Args:
        peptide_results: ``{condition: payload}`` from the peptide-statistics stage;
            each payload carries a ``control_condition`` key.
        output_dir: Directory receiving one ``families_<control>_<condition>.csv``
            per comparison.
        family_analysis: Optional pre-built worker (else one is created).
        debugging_print (bool): Verbose diagnostics.
        output_mode (str): ``'user'`` for the short column set, ``'developer'``
            for every column. Applied by
            ``kx_pipeline_tools.project_output_columns``.

    Returns:
        dict: ``{condition: result_dict}`` (result_dict as returned by
        :meth:`KinaseActivityAnalysis.run_kinase_analysis`).
    """
    worker = family_analysis or build_family_analysis(debugging_print=debugging_print)

    # Imported here, not at module scope: kx_pipeline_tools imports this module's
    # runner, so a top-level import would close the cycle.
    from kx_pipeline_tools import (
        project_output_columns,
        rank_by_absolute_z,
        resolve_output_mode,
    )

    resolved_mode = resolve_output_mode(output_mode)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    written = []
    total_rows = 0
    total_significant = 0
    for comparison_key, payload in peptide_results.items():
        control = payload["control_condition"]
        # The dict key identifies the comparison; the file name uses the
        # control/test labels of the pair.
        condition = payload["condition"]
        result = worker.run_kinase_analysis(
            peptide_statistics=payload,
            control=control,
            condition=condition,
        )
        results[comparison_key] = result

        # One file per comparison, holding every family. No separate significance
        # file: the Significant column carries that and the rows are ordered by
        # |Z|, so the hits are at the top. Unlike the single-kinase tables these
        # are NOT split by array -- no family occurs on both arrays, so there is
        # nothing a split would disentangle.
        frame = rank_by_absolute_z(
            _prepare_family_frame(result["all_kinases"], control, condition)
        )
        exported = project_output_columns(frame, "families", resolved_mode)
        output_path = output_dir / f"families_{control}_{condition}.csv"
        exported.to_csv(output_path, index=False)
        written.append(output_path)

        total_rows += len(frame)
        if "Significant" in frame.columns:
            total_significant += int(frame["Significant"].fillna(False).sum())

    print(
        f"Wrote kinase-family results: {total_rows} rows "
        f"({total_significant} significant) in {len(written)} file(s) -> {output_dir}"
    )
    return results


def main():
    """Standalone debug entry point.

    Runs the image + peptide-statistics stages (via the shared pipeline helpers),
    then the family-level analysis, and writes its CSV tables into the same
    ``results_kinase_families/`` folder the terminal pipeline uses. Requires the
    same input data as the normal terminal pipeline.
    """
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

    peptide_results, _peptide_analysis = run_peptide_statistics_analysis(
        enricher=enricher,
        processor_PTK=processor_ptk,
        processor_STK=processor_stk,
        analysis_timestamp=analysis_timestamp,
        experiment_name=experiment_name,
        results_parent_relpath=results_parent_relpath,
    )

    output_dir = (
        build_results_dir(
            analysis_timestamp,
            experiment_name,
            results_parent_relpath=results_parent_relpath,
        )
        / f"{analysis_timestamp}_downstream_analysis"
        / "results_kinase_families"
    )
    run_kinase_families_analysis(peptide_results, output_dir)


if __name__ == "__main__":
    main()
