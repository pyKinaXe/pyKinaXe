# Codebase Guide

This document describes the modules of pyKinaXe, how data flows through them
and where the results end up. Installation is covered in `installation.md`.

## 1. Pipeline

pyKinaXe processes a pair of PamGene runs, one PTK and one STK run, in seven
stages:

1. data import (the PTK run folder first, then the STK run folder)
2. sample-annotation enrichment
3. image processing
4. peptide statistics
5. upstream kinase activity analysis
6. pathway enrichment
7. plotting and result export

## 2. Modules

### `scripts/kx_kinase_extraction_pipeline.py`

Terminal entry point. It sets up the import paths, selects the headless
matplotlib backend and calls the stages implemented in `kx_pipeline_tools`.

### `src/kx_pipeline_tools.py`

Orchestration shared by the terminal and the web workflow: discovery of valid
PTK/STK runs, construction of the `DataLoader` instances, the results-directory
layout, the default downstream parameters, the staged image pipeline, the
peptide, kinase, family and pathway stages, and the optional figures.

### `src/kx_data_importer.py`

Input layer: discovers data folders, parses the TIFF file names, loads the
sample annotation and the array layout (`ArrayLayoutLoader`) and reads the TIFF
images into an `xarray`.

### `src/kx_data_enricher.py`

Experimental-design layer: combines the PTK and STK sample annotations, parses
the sample names into construct and replicate structure, assigns the
control/test roles and writes the enriched design table. The module also
contains the collectors for external resources (UniProt/BLAST, OmniPath PTM
data, the liver-kinase list).

### `src/kx_image_processor.py`

Turns the raw PamChip images into spot-level data: aperture detection and
masking, background subtraction, detection of the T- and J-shaped reference
spot patterns, placement and refinement of the peptide grid, per-spot
intensities, QC plots and the processed tables.

### `src/kx_peptide_analysis.py`

Stage 1 of the downstream analysis: rearranges the PTK/STK outputs into
comparison-ready matrices, computes the peptide statistics (with `inmoose.limma`
when configured) and produces the peptide waterfall plots and per-array peptide
heatmaps. Output: one peptide table per comparison.

`array_normalization: true` (default `false`) normalises
every array (one well, one sample) before `peptide_change` is formed. With
`array_normalization_method: median` each array is shifted so that its median
`log2(slope)` over the analysed peptides measured on all arrays of its type
equals the median of the array medians. The offsets are estimated once per run,
after the run-wide QC, and logged per array. This removes array-wide intensity
differences within a chip, which the per-chip batch correction cannot see.
KRSA does not normalise, and the published results were computed without it;
the default `false` reproduces either.

### `src/kx_upstream_kinase_analysis.py`

Stage 2: maps peptides to proteins and kinase-substrate evidence (BLAST,
enrichment table, PTM resources), scores the kinases with the KPEA Z-score and
writes the kinase tables and the sets for the overlap diagrams. Output: one
kinase table per array and comparison.

### `src/kx_upstream_kinase_families_analysis.py`

The same scoring at the level of kinase families, with KRSA's family
vocabulary. Output: one family table per comparison.

### `src/kx_pathway_enrichment_analysis.py`

Stage 3: takes the significant kinases, queries g:Profiler with their gene
symbols and writes the pathway tables for KEGG, Reactome and WikiPathways, plus
heatmaps and the sets for the overlap diagrams. The statistical domain is every
gene annotated in the source, or with `pathway_background: mappable_kinases`
only the kinases the PTM data maps onto the chip.

### `src/kx_plot_results.py`

Plotting classes: peptide waterfall plots, peptide and pathway heatmaps, venn
diagrams. All figures are rendered on the Agg backend and written to
`save_path`.

### `src/kx_benchmarking.py`

Comparison of pyKinaXe kinase and pathway results against an external reference
result set: overlap summaries, pathway enrichment comparison, heatmaps and venn
diagrams.

### `webapp/pykinaxe_webapp.py`

Flask API and runtime manager of the web app: uploads, the persistent FIFO
queue, job metadata and logs under the runtime root, polling and download
endpoints. Works with a local runtime folder or a mounted Hugging Face bucket.

### `webapp/backend/kx_web_kinase_extraction_pipeline.py`

Non-interactive pipeline runner for the web app: builds the `DataLoader`,
`DataEnricher` and `ImageProcessor` instances for an uploaded job, redirects
all outputs into the job folder, runs the same workflow as the terminal
pipeline and builds the downloadable archive.

## 3. Sample names

pyKinaXe derives the experimental design from the `Sample name` column of the
sample-annotation files: the construct, the biological and technical replicate,
and the control/test role. A biological replicate is a sample prepared again in
a new sample-preparation batch with the same protocol; a technical replicate
comes from the same preparation batch.

### Parsing

`DataEnricher._parse_sample_name()` expects the name to end with two replicate
identifiers, the biological replicate first:

```text
<construct_name><separator><biological_replicate><separator><technical_replicate>
```

Accepted separators: `_`, `-`, `.`, `,`, space, `/`, `\`, `|`, `:`. Accepted
replicate formats: Arabic numerals (`1`, `2`, `3`) and Roman numerals (`I`,
`II`, `III`). Examples that parse:

- `mock_1_1`
- `HBx.2.1`
- `puc18-3-2`
- `sample A/I/1`
- `constructX II 3`

Internally the names are standardised to
`<construct_name>_<biological_replicate>_<technical_replicate>`, so `mock.1.2`
becomes `mock_1_2` and `HBx-II-1` becomes `HBx_2_1`. The parser takes the last
two numeric or Roman-numeral tokens as the replicate numbers and everything
before them as the construct name, so the final two tokens must be the
replicate numbers (`HBx_mutant_2_1` is fine, `HBx_2024_mutant_2_1_extra` is
not).

If the replicate numbers cannot be parsed from the name, the explicit columns
`Biological replicate` and `Technical replicate` of the annotation file are
used. The sample annotation file is created during the experiment from the
information entered in the instrument software.

### Test conditions and roles

The `Test Condition` of a sample is its name without the leading `c`/`t`
prefix and without the replicate numbers:

| raw sample name | Test Condition | role |
|---|---|---|
| `c2_STR_CTL_2_2` | `STR_CTL` | control |
| `t1_pSHDAg` | `pSHDAg` | test |
| `t2_Exer_HPC_1_2` | `Exer_HPC` | test |
| `mock_2_1` (no prefix) | `mock` | control (inferred) |

Samples are matched by name across wells, chips and both arrays. The well order
in the two annotation files does not matter, but the same sample must carry the
same name in both files; the analysis stops when a condition appears on only
one array. The role is stored in the `Condition Role` column (`control`/`test`)
and is established in one of two ways.

Preferred: the `c`/`t` prefix. When every sample name of a chip type starts
with `c` or `t`, an optional chip number and `_` (for example `c1_STR_CTL_2_1`,
`t1_Exer_STR_2_1`), the prefix gives the role and the number is the chip index,
so `t1_pSHDAg` and `t2_pSHDAg` are the same condition on two chips. With several
controls the analysis compares every control with every test
(`contrast_mode: all_pairs` in `config/pipeline_defaults.yaml`); two controls
and two tests give four comparisons, named after the samples.
Control-vs-control and test-vs-test comparisons are not produced.

Fallback: alphabetical inference. When a chip type does not use the prefix
consistently, the PTK and STK tables are processed separately, the rows are
grouped by technical and then biological replicate, and within each group the
construct that sorts first alphabetically becomes the control; all others become
tests. Keywords such as `control`, `ctrl` or `mock` are not used for this
decision, so a group containing `HBx_1_1`, `Mock_1_1` and `TreatmentA_1_1`
makes `HBx` the control. To make `Mock` the control without a prefix, name it so
that it sorts first, for example `A_Mock_1_1`.

To see how a pair of annotation files is parsed without running the analysis:

```bash
python scripts/kx_data_annotations.py --list
python scripts/kx_data_annotations.py --ptk 1,1 --stk 1,2
```

This prints the raw files, the enriched table with `Test Condition`,
`Condition Role`, `Construct` and `Type`, and the comparisons the analysis would
run.

## 4. Configuration

Defaults live in the YAML files under `config/`: data folder discovery patterns,
image-processing thresholds and geometry, downstream statistical defaults,
pipeline flags and plot styling. The `config/*.py` modules load them and
normalise a few values (compiled regexes, repository-relative `Path` objects,
tuples instead of lists).

## 5. Choosing run folders

The pipeline expects one complete PTK run folder and one complete STK run
folder, PTK first. A run folder is the folder that contains `ImageResults/`, the
`* Sample Annotation.txt` and the `* Array Layout.txt`; its name carries the
barcodes, the `-on` segment, a PamChip type token such as `1200PTKlysv04` or
`1300STKlysv09`, `run` and a 12-digit timestamp:

```text
Experimental_data/
  October_2022/
    640208616_640208517_640208518-on 1200PTKlysv04-run 211117152830/
      ImageResults/
      640208616_640208517_640208518 86402 Sample Annotation.txt
      640208616_640208517_640208518 86402 Array Layout.txt
      640208616_640208517_640208518-on 1200PTKlysv04-run 211117152830.PS12Protocol
    710300320_710300321_710300322-on 1300STKlysv09-run 211117095001/
      ImageResults/
      710300320_710300321_710300322 87102 Sample Annotation.txt
      710300320_710300321_710300322 87102 Array Layout.txt
      710300320_710300321_710300322-on 1300STKlysv09-run 211117095001.PS12Protocol
```

Select the run folders themselves, not `ImageResults/` and not the parent
folder. The importer then finds the TIFF images under `ImageResults/`, the
sample annotation matching `*Sample Annotation*.txt` and the array layout ending
with `Array Layout.txt`.

## 6. Results

Results are written under `results/` unless overridden. PTK and STK outputs
share a parent folder derived from their common source folder; the web app
writes under its runtime root and builds the download archive from the job
folder. A run produces the source-data path note, the enriched annotation
table, the processed chip intensity tables and one
`<timestamp>_downstream_analysis/` folder:

```
<timestamp>_downstream_analysis/
├── run_config.txt   z-score thresholds, LFC hit cutoffs, null and pathway
│                    backgrounds, BLAST and pathway thresholds of the run
├── results_peptides/
│   ├── peptide_statistics_<control>_<condition>.csv
│   └── plots/   peptides_waterfall_plot_*.png, peptides_heatmap_{PTK,STK}_*.png
├── results_individual_kinases/
│   ├── kinases_PTK_<control>_<condition>.csv
│   ├── kinases_STK_<control>_<condition>.csv
│   └── plots/   kinases_significant_overlap.png + _tables/
├── results_kinase_families/
│   └── families_<control>_<condition>.csv
├── results_pathways/
│   ├── pathways_{KEGG,WikiPathways,Reactome}_<control>_<condition>.csv
│   └── plots/   {KEGG,WP,REAC}_UKA_heatmap_<control>_<condition>.png,
│                {KEGG,WP,REAC}_UKA_heatmap_comparisons_panels.png
│                  (+ _kinase_rows when comparison_heatmap.transpose is on),
│                pathways_*_overlap.png + _tables/
└── logs/        limma fallback reports
```

Every table is a CSV. There are no separate significance files: every kinase
and family is in its table, ordered by `|Z_Score|`, and the `Significant`
column (`|Z_Score| >= threshold`) marks the hits.

The kinase tables are split by array. A kinase with substrates on both arrays
keeps both measurements with their own substrate counts; within one array each
kinase appears once. Family tables are not split, because no family occurs on
both arrays. The ranked, de-duplicated `all_kinases` / `significant_kinases`
frames exist in memory and feed the pathway enrichment and the venn diagrams,
which work on the combined set; the combined significant set is the union of
the per-array calls.

The peptide heatmap is drawn once per array and shows every peptide that passed
QC. Its colour is the per-sample `log2(S100)`, the log2 of the
exposure-normalised reaction slope, the quantity `peptide_change` is the
treatment-minus-control difference of. Because it is a signal rather than a
contrast, the colour ramp runs from low to high (`plot.peptide_signal_cmap`, a
matplotlib colormap name or a list of colour stops) and is not centred on zero.

The pathway heatmaps come in two forms. `<source>_UKA_heatmap_<control>_<condition>.png`
shows one comparison: its enriched pathways as rows (plus an `All Kinases` row),
its significant kinases that fall into at least one of them as columns, and the
kinase metric (`uka_visualization_metric`) as colour; a white cell means the
kinase is not a member of that pathway. The title names the comparison
(`mock vs pSHDAg`). `<source>_UKA_heatmap_comparisons_<layout>.png` puts every
comparison of the run on one set of axes, the union of the pathways and kinases
with one shared colorbar: in the `split` layout each cell is divided into one
slice per comparison (a key next to the colorbar gives the slice order), in the
`panels` layout the comparisons sit side by side. Grey marks a cell the single
heatmap of that comparison does not have, i.e. the kinase is not significant or
the pathway is not enriched there. Pathways are ordered by the number of
comparisons they are enriched in, kinases by the mean metric.

That figure is sized for a page rather than for the screen. `target_width`
(7.0 inches by default) is a width budget: the cells are squeezed into
whatever is left after the labels, and the type is never scaled. The margins
are measured from the glyphs of the labels that are actually drawn, the tick
labels turn upright once the columns get too narrow for 45 degrees, the
colorbar and key move underneath the plot when the right-hand column no longer
fits, and the file is written whole instead of cropped to its content, so the
PNG really has that width. The budget yields in one case only: a column is
never squeezed past the point where the upright tick labels would touch
(`min_cell_width: auto`, measured on the labels). When that happens the figure
comes out wider than asked for and the log says by how much, because the
alternative would be shrinking the type. Shortening the labels with
`label_max_chars` is what gets a figure back under its budget.
`target_height` works the same way vertically, with the line height of the row
labels as its floor.

`transpose: true` swaps the axes, putting the kinases on the y axis for a
kinase-by-kinase comparison and the pathway names under the x axis; those
files carry a `_kinase_rows` suffix.

The panels of the `panels` layout stack downwards by default
(`panel_direction`), sharing one column axis that only the last panel labels,
which is what keeps the figure inside a text column. Each panel shows only the
rows that mean something for its comparison (`panel_rows: present`): a row
that is grey all the way across says nothing about that comparison, because
the pathway is not enriched in it, so it is dropped from that panel. On the
benchmarking run that turns three panels of forty rows into panels of 32, 5
and 19 rows, and the figure from 34 into 18 inches; `panel_rows: shared` keeps
the full union everywhere. The columns stay shared either way, so the panels
still line up. `panel_direction: horizontal` puts them side
by side instead, and then every panel repeats the whole column axis, so three
transposed panels of pathway names need about 17 inches before their labels
clear each other, which is what the figure then takes. Everything else,
layouts, slice direction, kinase order, wrapping and colours, is set under
`comparison_heatmap` in `config/heatmap_plot_config.yaml`. The figures need at
least two comparisons.

### Output modes

`output_mode` in `default_uka_kpea_params` selects the column set of the CSV
tables:

- `user` (default): the short, publication-facing columns.
  Kinases: `Kinase, Kinase_Name, NumSubstrates, KinaseChange,
  MeanPeptideStatistic, MedianPeptideStatistic, Z_Score, Significant, Type`.
  Families: `Comparison, Kinase_Family` and the same statistics columns.
  Pathways: `source, native, name, p_value, significant, description,
  precision, recall, parents, intersections`. Peptides: identity, the
  per-sample value/label pairs, group means and SDs, `average_expression`,
  `t_statistic`, `p_value`, `logp_value`, `peptide_change` and the run context.
- `developer`: every column the analysis produces.

Both modes drop columns that are exact copies of another column
(`KINASE_DUPLICATE_COLUMNS` in `kx_pipeline_tools.py`): `MeanSubstrate` and
`KinaseStatistic` equal `MeanPeptideStatistic`, `KRSA_MeanZ` equals `Z_Score`,
`KRSA_AbsMeanZ` and `KPEA_AbsDominantZ` equal `|Z_Score|`, and
`Significant_ZScore`, `Significant_SelectedMethod` and `SelectedForReport` equal
`Significant`. The in-memory frames keep every column, because
`_rank_kinase_results` ranks on `KRSA_AbsMeanZ` and `KinaseStatistic`.

The folders are configurable via `peptide_results_output`,
`kinase_results_output`, `family_results_output`, `pathway_results_output` and
`log_output` in `default_uka_kpea_params`. Empty tables are not written, so a
missing `pathways_WikiPathways_*.csv` means that source returned no enriched
terms.

### Run logs

`<timestamp>_downstream_analysis/logs/` (configured by `log_output`) collects
the diagnostics of the peptide-statistics stage and is created only when there
is something to report:

- `limma_missing_<CHIP>_<control>_<condition>_summary.txt` and a matching
  `.csv`: peptides dropped from the limma fit because QC or slope estimation
  left NaNs in at least one sample (limma is complete-case), with the affected
  peptide IDs and the number of surviving control/treatment values.
- `limma_degenerate_<CHIP>_<control>_<condition>_summary.txt`: the empirical
  Bayes prior collapsed (`df_prior` infinite). The stage falls back to row-wise
  t-tests for that chip and marks those peptides
  `statistics_method = t_test_degenerate_limma`.
- `array_normalization_offsets.csv` (only with `array_normalization: true`):
  one row per array with `Type`, `Barcode`, `Row`,
  `Test Condition`, the number of common peptides, the array median, the
  reference median, the subtracted `offset_log2`, the equivalent
  `slope_factor` applied to the raw slope, and `status` (`normalised`, or
  `skipped: ...` when fewer than `array_normalization_min_peptides` peptides
  were measured on every array of the type).

Both are written regardless of `debugging_print`; only the console output is
gated on that flag.

limma still runs when `use_limma` is set, but no downstream stage reads its
output: `peptide_change` is computed independently of it, the kinase and family
scoring use the log2 fold change and the Z-score, and the peptide heatmap
colours by `log2(S100)`. `p_value`, `t_statistic`, `logp_value`, `s2_moderated`,
`df_moderated` and the `*_zscore` columns are therefore exported but unused, and the `batch_correction_method` settings
`limma_block` and `center` give identical results, since they differ only in
whether the batch factor enters the limma design.

## 7. Web app runtime

The web app works on a runtime root that is a plain directory:
`webapp/runtime/` locally or a mounted Hugging Face Storage Bucket in Spaces. It
contains `jobs/<job_id>/...` (uploaded inputs, result folders,
`job_state.json`), `server_audit.log` and the downloadable archives. The queue
is FIFO and survives process restarts as long as the runtime root persists.

## 8. Entry points

Terminal:

```bash
python scripts/kx_kinase_extraction_pipeline.py
```

Web backend:

```bash
python webapp/pykinaxe_webapp.py
```

The scripts under `validation_paper/` compare pyKinaXe with BioNavigator, KRSA
and pamgeneAnalyzeR; each folder has a `NOTE.txt` describing its inputs and
outputs.
