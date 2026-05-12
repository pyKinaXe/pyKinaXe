# Current UKA/KPEA Method

This document summarizes the current downstream analysis method used by
pyKinaXe after image processing has produced PTK/STK spot intensities.

## Short Summary

The current UKA/KPEA workflow is a staged downstream analysis with three main parts:

1. peptide-level statistics
2. upstream kinase activity analysis
3. pathway enrichment

The kinase-analysis core is a **KRSA-like, permutation-based enrichment test**
operating at the level of individual kinases.

What the core tests:

- whether substrate peptides assigned to a kinase appear among the strongly changed peptides
  more often or less often than expected by chance

What it does **not** test directly:

- it does not directly prove kinase activation or inhibition
- the primary enrichment direction is first an overrepresentation / underrepresentation signal

## 1. Overall Workflow

At a high level, the downstream analysis does the following:

1. load the enriched experimental design table
2. build peptide-level PTK/STK matrices from processed image outputs
3. compute peptide statistics per condition
4. load BLAST-derived peptide-to-protein mappings
5. load PTM / kinase-substrate interaction resources
6. build kinase-to-peptide mappings
7. run the KRSA-like enrichment test across one or more cutoffs
8. assign significance using z-score, empirical p-value, or FDR
9. optionally run pathway enrichment and generate plots

In simple terms:

- first the code turns processed chip signals into peptide-level condition effects
- then it asks which kinases are unusually enriched among changed peptides
- finally it summarizes those kinase results at pathway level

## 2. Stage 1: Peptide Statistics

Stage 1 is implemented mainly in `PeptideStatistics` in
`src/kx_peptide_analysis.py`.

### 2.1 Inputs

This stage uses:

- the enriched PTK/STK design table from `DataEnricher`
- the processed PTK outputs from `ImageProcessor`
- the processed STK outputs from `ImageProcessor`

### 2.2 Experiment Design

The peptide statistics stage relies on the enriched table to know:

- which samples belong to control
- which samples belong to each test condition
- which PTK/STK rows belong to the same condition comparison

This is one reason why correct sample naming and replicate annotation matter so much.

### 2.3 Slope And Log2 Transformation

The image-processing pipeline produces per-spot values that are then converted
into analysis-ready peptide measurements. A slope is estimated across exposure
times, and that slope is then log2-transformed.

The parameter `log2_slope_mode` controls how small values are handled:

- `pamgene_zero`: slopes `<= 1` are effectively floored to `0`
- `epsilon_floor`: uses `log2(max(slope, epsilon))`

### 2.4 Main Peptide-Level Quantities

The peptide statistics stage produces several important columns:

| Column | Meaning |
| --- | --- |
| `peptide_change` | mean treatment minus mean control |
| `peptide_statistic` | peptide effect normalized by estimated standard error |
| `t_statistic` | moderated limma t-statistic or classical t-statistic |
| `p_value` | peptide p-value |
| `pamgene_snr` | PamGene-style signal-to-noise quantity |

### 2.5 Core Formulas

For peptide `j`, with control values `C_j` and treatment values `T_j`:

```text
mean_control_j   = mean(C_j)
mean_treatment_j = mean(T_j)
SD_control_j     = sample_sd(C_j)
SD_treatment_j   = sample_sd(T_j)
```

The main effect size is:

```text
peptide_change_j = mean_treatment_j - mean_control_j
```

The repository also computes:

```text
denom_j = sqrt((SD_control_j^2 / n_control_j) + (SD_treatment_j^2 / n_treatment_j))
peptide_statistic_j = peptide_change_j / denom_j
```

If the denominator is zero, the implementation avoids infinities by setting
`peptide_statistic` to `0`.

### 2.6 LIMMA / inmoose

If `use_limma=True`, pyKinaXe uses `inmoose.limma` to fit a moderated model
across the peptide-by-sample matrix. In that case:

- `t_statistic` and `p_value` come from the moderated limma fit
- moderated variance-related information is also retained

If LIMMA is disabled, a classical two-sample t-test is used instead.

Important practical point:

- the downstream KRSA-like kinase core does **not** use the limma t-statistic as
  its primary hit-definition score
- it primarily uses `peptide_change`

### 2.7 PamGene SNR

An additional PamGene-style signal-to-noise quantity is also computed:

```text
pamgene_snr_j =
  (mean_treatment_j - mean_control_j) /
  sqrt(SD_treatment_j^2 + SD_control_j^2)
```

This is conceptually different from `peptide_statistic`, because it does not
also divide by group size in the same way.

## 3. Stage 2: Upstream Kinase Analysis

Stage 2 is implemented mainly in `KinaseActivityAnalysis` in
`src/kx_upstream_kinase_analysis.py`.

### 3.1 Inputs To The Kinase Stage

The kinase stage combines:

- peptide statistics from stage 1
- peptide-enrichment annotation resources
- BLAST-derived peptide-to-protein mappings
- PTM / kinase-substrate interaction databases

Optional strictness controls include:

- `use_verified_interactions_only=True`
- `require_known_ptm_site=True`

These make the mapping stricter, often reducing the number of kinase-peptide links.

### 3.2 What The KRSA-Like Core Uses

The active enrichment core is the KPEA / KRSA-like routine inside
`src/kx_upstream_kinase_analysis.py`.

Its primary inputs are:

- the pooled peptide dataset for one comparison
- the kinase-to-peptide mapping
- the vector of `peptide_change` values
- the vector of `peptide_statistic` values

Important current behavior:

- **hit definition is driven primarily by `peptide_change`**
- `peptide_statistic` contributes descriptive peptide summaries, but the main
  KRSA-style hit lists are thresholded from peptide changes

## 4. Hit Definition And Cutoffs

The algorithm can evaluate one or more absolute fold-change-like cutoffs via:

- `kpea_lfc_cutoffs`

Example:

```text
[0.2, 0.3, 0.4]
```

For each cutoff:

- peptides with sufficiently large absolute `peptide_change` become hits
- the algorithm counts how many hit peptides map to each kinase

The cutoff mode is controlled by `kpea_cutoff_mode`:

- `average`: combine information across several cutoffs
- `primary`: use one selected cutoff only

The primary cutoff can be set via:

- `kpea_primary_lfc_cutoff`

## 5. KRSA-Like Enrichment Logic

For each kinase and each cutoff, pyKinaXe asks:

- among the peptides mapped to this kinase, how many are in the hit list?
- is that count larger or smaller than expected by chance?

### 5.1 Observed Count

For each kinase:

- determine the substrate peptides assigned to that kinase
- count how many of those peptides are currently in the hit list

### 5.2 Null Distribution

The null distribution is generated through permutations:

- the hit list size is preserved
- hit labels are redistributed many times
- the default analysis may use values such as `n_permutations = 5000`

This builds an expected count distribution for each kinase at each cutoff.

### 5.3 Z-Score

The observed kinase hit count is compared to the permutation null:

```text
Z = (observed_count - mean_null_count) / sd_null_count
```

Interpretation:

- `Z > 0`: overrepresentation
- `Z < 0`: underrepresentation
- larger `|Z|`: stronger deviation from the null

### 5.4 Combining Across Cutoffs

If `kpea_cutoff_mode="average"`:

- the z-scores across cutoffs are averaged into a signed summary score

If `kpea_cutoff_mode="primary"`:

- the final score comes from one selected cutoff only

Important output columns include:

| Column | Meaning |
| --- | --- |
| `KRSA_MeanZ` | signed mean z-score across cutoffs |
| `KRSA_AbsMeanZ` | absolute value of the mean z-score |
| `KPEA_AbsDominantZ` | compatibility / alias-style absolute summary score |

## 6. Significance

The final significance framework can use one of:

- `z_score`
- `p_value`
- `fdr`

This is controlled by:

- `kpea_significance_method`

Additional thresholds include:

- `kpea_zscore_threshold`
- `kpea_empirical_p_threshold`
- `kpea_fdr_threshold`

There is also a substrate-support filter:

- `kpea_substrate_cutoff`
- `support_filtered_min_num_substrates`

This means the workflow distinguishes between:

- raw statistical significance
- reportable significance after minimum-support filtering

## 7. Important Kinase Output Columns

The kinase stage writes tables with columns such as:

| Column | Meaning |
| --- | --- |
| `Kinase` | kinase identifier |
| `NumSubstrates` | number of mapped substrate peptides |
| `MedianPeptideStatistic` | median peptide statistic over substrates |
| `MeanPeptideStatistic` | mean peptide statistic over substrates |
| `KinaseChange` | mean substrate peptide change |
| `Direction` | overrepresented / underrepresented / none |
| `Direction_PeptideMean` | descriptive direction from mean substrate change |
| `KRSA_MeanZ` | signed enrichment z-score summary |
| `KRSA_AbsMeanZ` | absolute enrichment z-score summary |
| `p_value` | empirical permutation p-value |
| `FDR` | BH-corrected p-value |
| `NegLog10EmpiricalP` | `-log10(p_value)` |
| `KPEA_CutoffMode` | whether `average` or `primary` scoring was used |
| `KPEA_SelectedCutoff` | selected primary cutoff, if applicable |
| `Significant` | raw significance flag |
| `SelectedForReport` | significance plus support filtering |

## 8. Stage 3: Pathway Enrichment

Stage 3 is implemented mainly in `PathwayEnrichmentAnalysis` in
`src/kx_pathway_enrichment_analysis.py`.

This stage:

- takes significant kinase results
- submits ranked/significant kinase lists to g:Profiler
- summarizes pathways across sources such as KEGG, Reactome, and WikiPathways
- generates pathway heatmaps and venn-ready result sets

This stage is conceptually downstream of the kinase-enrichment stage and does
not redefine the kinase-scoring method itself.

## 9. Practical Interpretation Notes

Several interpretation points are important:

1. `Direction = overrepresented` does **not** automatically mean direct kinase activation.
2. `Direction = underrepresented` does **not** automatically mean direct kinase inhibition.
3. The most biologically intuitive directional signal often lives in:
   - `KinaseChange`
   - `Direction_PeptideMean`
4. Mapping strictness has a major effect on which kinases survive:
   - verified-only filtering is stricter
   - known-site filtering is stricter
5. The choice of cutoff grid and permutation count can materially affect which kinases are emphasized.

## 10. Parameter Sensitivity

The following parameters are especially influential:

| Parameter | Effect |
| --- | --- |
| `kpea_lfc_cutoffs` | lower values create larger hit lists; higher values are stricter |
| `kpea_cutoff_mode` | `average` smooths across cutoffs; `primary` uses a single cutoff |
| `kpea_primary_lfc_cutoff` | chooses the single cutoff used in primary mode |
| `kpea_zscore_threshold` | larger threshold is stricter for z-score significance |
| `kpea_empirical_p_threshold` | larger threshold is less strict |
| `kpea_fdr_threshold` | larger threshold is less strict |
| `kpea_substrate_cutoff` | filters kinases with too few substrates early |
| `support_filtered_min_num_substrates` | controls reporting strictness |
| `n_permutations` | more permutations improve stability but increase runtime |
| `use_verified_interactions_only` | stricter kinase-substrate mapping |
| `require_known_ptm_site` | stricter site-level evidence filtering |

## 11. Bottom Line

The current pyKinaXe downstream method is:

- a complete staged PTK/STK downstream workflow
- with peptide statistics followed by kinase enrichment and pathway enrichment
- using a KRSA-like, permutation-based kinase-enrichment core
- operating on individual kinase mappings
- driven primarily by peptide-change hit lists
- with optional strict mapping filters and support-based reporting filters

In short:

- stage 1 asks which peptides changed
- stage 2 asks which kinases are unusually represented among those changed peptides
- stage 3 asks which pathways are enriched among the significant kinase results
