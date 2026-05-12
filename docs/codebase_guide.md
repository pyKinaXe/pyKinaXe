# Codebase Guide

This document is the architecture-oriented companion to the root `README.md`.
It explains the essential moving parts of pyKinaXe and how they fit together.

## 1. Conceptual Pipeline

pyKinaXe processes paired PamGene phosphopeptide runs:

- one `PTK` run (see example folders below)
- one `STK` run (see example folders below)

The workflow:

1. data import (choose first PTK and then STK run folders)
2. sample-annotation enrichment
3. image processing
4. peptide statistics
5. upstream kinase activity analysis
6. pathway enrichment
7. plotting and result export

## 2. High-Level Module Responsibilities

### `scripts/kx_kinase_extraction_pipeline.py`

The main human-facing terminal entry point.

Responsibilities:

- prepare import paths
- force a non-GUI matplotlib backend for headless runs
- expose the top-level analysis stages clearly in `main()`
- preserve a few historical imports used by tests and local helper scripts

### `src/kx_pipeline_tools.py`

Shared orchestration helpers used by the terminal and web workflows.

Responsibilities:

- discover valid PTK/STK runs
- build `DataLoader` instances from user selections
- compute consistent results-directory layouts
- resolve default downstream-analysis parameters
- run the staged PTK/STK image pipeline
- run peptide, kinase, and pathway analysis stages
- render optional publication and processing-stage figures

Think of this file as the main “glue” layer between scientific modules.

### `src/kx_data_importer.py`

Input-loading layer for PamGene runs.

Responsibilities:

- discover valid data folders
- parse TIFF file naming metadata
- load sample annotation tables
- load array layout metadata
- load TIFF images into `xarray`
- expose imported data in forms the downstream modules expect

The `ArrayLayoutLoader` functionality is now defined in this same file, so all
input-loading concerns live in one place.

### `src/kx_data_enricher.py`

Experimental-design enrichment layer.

Responsibilities:

- combine PTK and STK sample annotation tables
- parse sample names into condition / biological / technical replicate structure
- produce the enriched design table used downstream
- provide several utility CLI/data-collection helpers for external resources

In addition to `DataEnricher`, this file contains utility classes for:

- UniProt/BLAST-based data collection
- OmniPath PTM extraction
- liver-kinase list extraction

### Sample Name Conventions In The Sample Annotation Files

pyKinaXe derives a substantial part of the experimental design from the
`Sample name` column in the sample-annotation files. Because of that,
correct sample naming is not just a cosmetic issue: it directly affects how
test and control conditions are inferred, as well as the information on biological and technical replicates.

Biological replicate is a sample that was prepared again (as a new/different sample-preparation batch) using the same sample preparation protocol. Technical replicate is a sample that was derived from the same sample-preparation batch.

The parser in `DataEnricher._parse_sample_name()` expects the sample name to
end with **two replicate identifiers**:

- first trailing number: **Biological replicate**
- second trailing number: **Technical replicate**

The core pattern is therefore:

```text
<construct_name><separator><biological_replicate><separator><technical_replicate>
```

Accepted separators include:

- `_`
- `-`
- `.`
- `,`
- space
- `/`
- `\`
- `|`
- `:`

Accepted replicate formats:

- Arabic numerals, e.g. `1`, `2`, `3`
- Roman numerals, e.g. `I`, `II`, `III`

Examples that pyKinaXe can parse:

- `mock_1_1`
- `HBx.2.1`
- `puc18-3-2`
- `sample A/I/1`
- `constructX II 3`

After parsing, pyKinaXe standardizes names internally to:

```text
<construct_name>_<biological_replicate>_<technical_replicate>
```

So, for example:

- `mock.1.2` becomes `mock_1_2`
- `HBx-II-1` becomes `HBx_2_1`

Important parsing rule:

- pyKinaXe captures the **last two numeric/roman-numeral tokens** in the sample
  name
- everything before those two trailing replicate tokens is treated as the
  construct name

That means names should be designed so the final two tokens truly are the
biological and technical replicate numbers.

For example:

- good: `HBx_mutant_2_1`
- risky/ambiguous: `HBx_2024_mutant_2_1_extra`

### Fallback To Explicit Replicate Columns

If replicate numbers are not parsed successfully from `Sample name`, pyKinaXe
tries to use the explicit columns:

- `Biological replicate`
- `Technical replicate`

However, this fallback is only reliable when those columns are correctly filled
in and available in the input annotations. The sample annotation file is created during experiment after the user fills in the information requested by the experiment control software.

### How Test Conditions Are Inferred

pyKinaXe does not simply preserve the raw sample names. It also infers
`Test Condition` labels from the parsed construct names.

The logic in `DataEnricher._determine_test_conditions()` works as follows:

- PTK and STK annotation tables are processed separately
- within each source table, rows are grouped by technical replicate
- within each technical-replicate group, rows are further grouped by biological replicate
- within each such subgroup, pyKinaXe strips off the final replicate numbers and looks only at the construct names
- the **alphabetically smallest** construct name is labeled `Control`
- remaining construct names are labeled `Test1`, `Test2`, `Test3`, ... in alphabetical order

This is an essential point:

- the control condition is inferred from **alphabetical ordering**
- not from keywords like `control`, `ctrl`, or `mock` alone

So if you want a particular construct to become `Control`, make sure its
construct name sorts first alphabetically within the relevant replicate group.

For example, if the subgroup contains:

- `HBx_1_1`
- `Mock_1_1`
- `TreatmentA_1_1`

then alphabetically the labels are assigned based on the construct names
`HBx`, `Mock`, `TreatmentA`, so `HBx` would become `Control`, which may be
biologically wrong if you intended `Mock` to be the control.

A safer naming scheme would make the intended control sort first, for example:

- `A_Mock_1_1`
- `HBx_1_1`
- `TreatmentA_1_1`

### Why Correct Naming Matters

Incorrect or inconsistent sample naming can lead to:

- wrong biological/technical replicate assignments
- incorrect standardization of sample names in the enriched table
- wrong grouping of samples before peptide statistics
- incorrect control/test-condition assignment
- PTK/STK mismatches in downstream comparison logic

In short, sample naming is part of the experimental design, not just metadata.

### Recommended Practical Convention

For the least ambiguity, a good convention is:

```text
<construct_name>_<biological_replicate>_<technical_replicate>
```

Examples (one biological replicate and two technical replicates):

- `A_Mock_1_1`
- `A_Mock_1_2`
- `HBx_1_1`
- `HBx_1_2`

Recommendations:

- use the same naming convention in both PTK and STK runs
- always place biological replicate before technical replicate
- keep the final two tokens reserved for replicate identifiers only
- avoid adding extra suffixes after replicate numbers
- make the intended control construct alphabetically first

### `src/kx_image_processor.py`

The largest and most image-centric part of the repository.

Responsibilities:

- detect optical aperture in each image
- apply circular and square masks
- subtract spatially heterogeneous background
- detect reference spot "TeeWee" and "Blue Ricky" patterns
- infer and refine the peptide grid
- calculate per-spot intensities
- export QC plots and processed tables

This module is where the raw PamChip images become quantitative spot-level data.

### `src/kx_peptide_analysis.py`

Stage 1 of the downstream UKA/KPEA workflow.

Responsibilities:

- reorganize PTK/STK outputs into comparison-ready matrices
- compute peptide-level statistics
- use `inmoose.limma` when configured
- produce peptide volcano plots and peptide heatmaps

Primary output:

- per-condition peptide statistics tables

### `src/kx_upstream_kinase_analysis.py`

Stage 2 of the downstream workflow.

Responsibilities:

- map peptides to proteins and kinase-substrate evidence
- combine BLAST/enrichment/PTM resources
- compute upstream kinase activity scores
- perform permutation-based scoring and significance estimation
- produce kinase-level volcano outputs and Venn tables

Primary output:

- per-condition kinase activity tables

### `src/kx_pathway_enrichment_analysis.py`

Stage 3 of the downstream workflow.

Responsibilities:

- take significant kinase results from the kinase stage
- call g:Profiler enrichment
- summarize pathway hits
- produce pathway heatmaps and Venn pathway sets

Primary output:

- per-condition pathway tables for KEGG, Reactome, and WikiPathways

### `src/kx_plot_results.py`

Reusable plotting toolkit.

Responsibilities:

- peptide and kinase volcano plots
- peptide and pathway heatmaps
- venn diagrams
- optional Tkinter-backed GUI display modes

This is a plotting utility layer rather than a workflow orchestrator.

### `src/kx_benchmarking.py`

Optional benchmarking/comparison utilities.

Responsibilities:

- compare pyKinaXe kinase outputs to an external reference result set
- compute overlap summaries
- compare pathway enrichment sets
- draw heatmaps and Venn diagrams for benchmarking studies

### `webapp/pykinaxe_webapp.py`

The Flask API and runtime manager for the web app.

Responsibilities:

- receive uploads or local-path job requests
- manage the persistent FIFO queue
- keep job metadata, logs, and queue state under the runtime root
- serve polling endpoints, downloads, and generated files
- support both local filesystem runtime and mounted Hugging Face bucket runtime

### `webapp/backend/kx_web_kinase_extraction_pipeline.py`

Non-interactive web-facing pipeline runner.

Responsibilities:

- create `DataLoader`, `DataEnricher`, and `ImageProcessor` instances for uploaded data
- override outputs into web-job-specific folders
- run the same scientific workflow as the terminal backend
- summarize outputs for the frontend
- create downloadable job archives

## 3. Configuration System

The codebase separates code from defaults through YAML-backed config loaders.

### `config/*.yaml`

Stores user-adjustable defaults such as:

- data folder discovery patterns
- image-processing thresholds and geometry constants
- downstream statistical defaults
- pipeline runtime flags
- plot styling

### `config/*.py`

Loads those YAML files and performs small bits of normalization, for example:

- compiling regexes
- converting relative paths into repository-relative `Path` objects
- converting YAML lists into tuples where immutability is useful

## 4. Data Flow In Practice

### Choosing PTK/STK Run Folders

The pipeline expects one complete `PTK` run folder and one complete `STK` run folder (order is important, PTK first, STK second). In practical terms, that usually means selecting the folder whose name
contains:

- the chip or barcode identifiers
- several chip/barcode identifiers joined by underscores
- the `-on` segment
- a PamChip-type token such as `1200PTKlysv04` or `1300STKlysv09`
- the substring `run`
- either `PTK` or `STK` inside that PamChip-type token
- a trailing 12-digit timestamp suffix

Typical examples look like:

(PTK)

- `640208616_640208517-on 1200PTKlysv04-run 211117152830`
- `640108916_640108917_640108921-on 1200PTKlysv04-run 211118154511`

(STK)

- `710300320_710300321-on 1300STKlysv09-run 211117095001`
- `710303214_710303215_710303216-on 1300STKlysv09-run 211118095903`

What the user should choose:

- choose the run folder itself
- not only its `ImageResults/` subfolder
- not the parent folder containing many runs

In other words, if the filesystem looks like:

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

then the correct selections are:

- `640208616_640208517_640208518-on 1200PTKlysv04-run 211117152830`
- `710300320_710300321_710300322-on 1300STKlysv09-run 211117095001`

The importer then finds:

- TIFF images under `ImageResults/`
- sample annotation metadata in the run folder, typically matching `*Sample Annotation*.txt`
- array layout metadata in the run folder, typically ending with `Array Layout.txt`


### Stage A: Import

`DataLoader`:

- validates folder structure
- parses TIFF file names into metadata
- loads sample annotation
- loads array layout
- loads TIFF images

### Stage B: Enrichment

`DataEnricher`:

- combines PTK and STK sample annotations
- infers test-condition structure
- saves an enriched experiment-design table

### Stage C: Image Processing

`ImageProcessor`:

- finds optical apertures
- masks and background-corrects images
- locates reference spots
- locates and refines the peptide grid
- computes per-spot intensities
- exports quality-control plots and final processed tables

### Stage D: Peptide Statistics

`PeptideStatistics`:

- builds per-condition PTK/STK matrices
- estimates slopes / peptide responses
- runs statistical testing
- saves peptide-level tables and plots

### Stage E: Upstream Kinase Analysis

`KinaseActivityAnalysis`:

- maps peptide evidence to kinases
- scores kinases
- estimates significance
- exports kinase tables and volcano/venn outputs

### Stage F: Pathway Enrichment

`PathwayEnrichmentAnalysis`:

- uses significant kinase sets
- queries g:Profiler
- saves per-pathway outputs and heatmaps

## 5. Result Layout

The repository writes results under `results/` by default unless overridden.

Important behavior:

- PTK and STK outputs share a common parent structure derived from their common source-data parent
- terminal and web flows both try to preserve meaningful experiment grouping
- the web app places job-specific work under its runtime root and then builds downloadable archives from those job folders

Typical content includes:

- source-data path notes
- enriched annotation tables
- processed chip intensity tables
- peptide heatmaps and volcano plots
- kinase outputs
- pathway outputs
- venn tables and images

## 6. Web App Runtime Model

The web app is intentionally designed around a runtime root that behaves like a
regular filesystem path.

That is important because the same codepath can work with:

- `webapp/runtime/` locally
- a mounted Hugging Face Storage Bucket in Spaces

Inside the runtime root, the web app keeps:

- `jobs/<job_id>/...` uploaded inputs, result folders, and persisted `job_state.json`
- `server_audit.log` runtime audit messages
- downloadable archives for completed jobs

The queue is FIFO and persistent across process restarts as long as the runtime
root itself persists.

## 7. Entry Points And Typical Use Cases

### Terminal users

Use:

```bash
python scripts/kx_kinase_extraction_pipeline.py
```

Best for:

- local desktop use
- batch scientific runs
- exploratory QC and figure generation

### Web users

Use:

```bash
python webapp/pykinaxe_webapp.py
```

Best for:

- browser-based uploads
- queueing multiple jobs
- exposing the workflow through a simple frontend

### Utility / developer use

Files in `tests/` and `notebooks/` are primarily for validation, comparisons,
and exploratory analysis rather than end-user execution.
