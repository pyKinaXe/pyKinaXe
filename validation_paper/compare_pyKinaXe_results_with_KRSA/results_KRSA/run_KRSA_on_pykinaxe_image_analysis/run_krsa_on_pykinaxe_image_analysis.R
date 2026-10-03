#!/usr/bin/env Rscript
# run_krsa_on_pykinaxe_image_analysis.R
#
# Runs the KRSA package (https://github.com/CogDisResLab/KRSA) on pyKinaXe's
# own image analysis, so that the kinase-family comparison in the parent folder
# has a KRSA side that starts from exactly the spot values pyKinaXe produced.
# The input is pyKinaXe's *_Export_image_analysis_<ARRAY>_bn.csv (pyKinaXe's
# spot quantification in BioNavigator-compatible column layout; "_bn" refers
# to the format, not to BioNavigator as the source). No TIFF is read here and
# no pyKinaXe code runs: krsa_qc_steps(), krsa_scaleModel(),
# krsa_quick_filter(), krsa_group_diff() and krsa() are the package's own.
#
# Input (ships in this folder):
#   exports_pykinaxe/<dataset folder>/<chip run folder>/
#       <timestamp>_Export_image_analysis_<PTK|STK>_bn.csv   spot values
#       <timestamp>_data_enrichment.txt                      sample annotation
#   Two datasets ship here and both are processed by default:
#       exports_image_analysis_pyKinaXe_benchmarking_data_set/          -> benchmarking
#       exports_image_analysis_pykinaxe_CDRL_vwr-rats-kinome_data_set/  -> CDRL_vwr-rats-kinome
#
# Output:
#   <output-dir>/<dataset>/<PTK|STK>/tables/   per comparison:
#       <test>_vs_<control>_LFC.txt                   peptide log2 fold changes
#       <test>_vs_<control>_KRSA_Zscores_primary.txt  krsa() at the first cutoff
#       <test>_vs_<control>_KRSA_acrossChip.txt       Z per cutoff and AvgZ (read by
#                                                     the comparison)
#       <test>_vs_<control>_KRSA_withinChip.txt       Z per barcode and cutoff
#       <test>_vs_<control>_top_kinases.txt           |AvgZ| >= z-threshold
#       global_qc_passed_peptides.txt                 KRSA's global QC list
#   <output-dir>/<dataset>/<PTK|STK>/figures/   KRSA report figures (PDF)
#   <output-dir>/<dataset>/run_manifest.txt     settings, thresholds, counts
#   The <dataset> level is dropped when --input-dir points at one dataset
#   folder, which is how run_KRSA_on_pykinaxe_input.py drives it.
#
# Usage:
#   Rscript run_krsa_on_pykinaxe_image_analysis.R                 # both datasets
#   Rscript run_krsa_on_pykinaxe_image_analysis.R --datasets benchmarking --no-figures
#   Rscript run_krsa_on_pykinaxe_image_analysis.R --help
# Paths resolve relative to this script. KRSA is installed from GitHub if
# missing (remotes::install_github) unless --no-install is given; the version
# used is written to the manifest.
#
# The control/test role is read from the "Condition Role" column of the sample
# annotation (with a fallback to Test Condition == "Control"), and every
# test x control pair is run: 3 comparisons on the benchmarking dataset, 4 on
# the rat dataset with its two controls.

KRSA_SOURCE <- "https://github.com/CogDisResLab/KRSA"

# ======================= SCRIPT LOCATION ===================================

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 1L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(normalizePath(ofile)))
  normalizePath(getwd())
}
SCRIPT_DIR <- get_script_dir()
REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", "..", "..", ".."), mustWork = FALSE)

# Paths in the run manifest are written relative to the repository root.
rel_to_repo <- function(path) {
  p    <- normalizePath(path, mustWork = FALSE)
  root <- paste0(normalizePath(REPO_ROOT, mustWork = FALSE), "/")
  ifelse(startsWith(p, root), substring(p, nchar(root) + 1), p)
}

# ======================= CONFIGURATION =====================================
# Every value here is overridable on the command line; see parse_args().

config <- list(
  input_dir  = file.path(SCRIPT_DIR, "exports_pykinaxe"),
  output_dir = file.path(SCRIPT_DIR, "output"),

  # Which dataset folders to run. NULL = all that are found. Matching is a
  # case-insensitive substring test against both the folder name and the short
  # label ("benchmarking", "CDRL_vwr-rats-kinome"), so --datasets rat works.
  datasets = NULL,

  # QC / hit-calling thresholds. These are the KRSA report template's defaults
  # and they are also what pyKinaXe's QC_KRSA mode reproduces, so do not change
  # one side only -- the comparison assumes both sides used these.
  signal_threshold = 5,        # min signal at max exposure of the end-point cycle
  r2_threshold     = 0.9,      # min R^2 of the (signal ~ exposure) fit
  lfc_cutoffs      = c(0.2, 0.3, 0.4),
  z_hit_threshold  = 2,        # |Z| for calling a kinase family a hit

  # Random sampling. krsa() draws `iterations` peptide sets to build its null;
  # `seed` makes that draw reproducible, so two runs of this script agree
  # exactly. (pyKinaXe's own Z is analytic and needs no seed.)
  iterations = 2000,
  seed       = 123,

  # Replicate spots (a peptide printed on several spots of the array) are
  # collapsed to one value per peptide/sample/exposure/cycle. mean() is what
  # KRSA's crosstab reader effectively receives from BioNavigator.
  spot_aggregate = function(x) mean(x, na.rm = TRUE),

  make_figures   = TRUE,
  install_missing = TRUE
)

# Chip-specific kinase-substrate mapping and coverage reference data, shipped
# inside the KRSA package as lazy-loaded datasets. These tables -- not the data --
# decide which peptides count as a family's substrates.
chip_ref_names <- list(
  PTK = list(cov = "KRSA_coverage_PTK_PamChip_86402_v1",
             map = "KRSA_Mapping_PTK_PamChip_86402_v1"),
  STK = list(cov = "KRSA_coverage_STK_PamChip_87102_v2",
             map = "KRSA_Mapping_STK_PamChip_87102_v1")
)

# ======================= COMMAND LINE ======================================

USAGE <- "
Usage: Rscript run_krsa_on_pykinaxe_image_analysis.R [options]

  --input-dir PATH        folder holding the dataset folders, or ONE dataset
                          folder (default: <script dir>/exports_pykinaxe)
  --output-dir PATH       where results are written (default: <script dir>/output)
  --datasets A,B          only these datasets (substring match; default: all)
  --iterations N          krsa() sampling iterations (default: 2000)
  --seed N                sampling seed (default: 123)
  --signal-threshold X    min end-point signal for peptide QC (default: 5)
  --r2-threshold X        min R^2 for peptide QC (default: 0.9)
  --lfc-cutoffs A,B,C     hit-peptide log2 FC cutoffs (default: 0.2,0.3,0.4)
  --z-threshold X         |Z| for a kinase-family hit (default: 2)
  --no-figures            write tables only (much faster)
  --no-install            never install KRSA; fail if it is missing
  --list                  list the datasets/arrays that would run, then exit
  --help                  this text
"

parse_args <- function(argv) {
  take_value <- function(i, flag) {
    if (i + 1L > length(argv)) stop("Missing value for ", flag, call. = FALSE)
    argv[i + 1L]
  }
  as_num_vec <- function(txt, flag) {
    v <- suppressWarnings(as.numeric(strsplit(txt, ",")[[1]]))
    if (any(is.na(v))) stop("Not numeric: ", flag, " ", txt, call. = FALSE)
    v
  }
  only_list <- FALSE
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]
    skip <- 2L
    switch(a,
      "--input-dir"        = config$input_dir  <<- take_value(i, a),
      "--output-dir"       = config$output_dir <<- take_value(i, a),
      "--datasets"         = config$datasets   <<- trimws(strsplit(take_value(i, a), ",")[[1]]),
      "--iterations"       = config$iterations <<- as_num_vec(take_value(i, a), a)[1],
      "--seed"             = config$seed       <<- as_num_vec(take_value(i, a), a)[1],
      "--signal-threshold" = config$signal_threshold <<- as_num_vec(take_value(i, a), a)[1],
      "--r2-threshold"     = config$r2_threshold     <<- as_num_vec(take_value(i, a), a)[1],
      "--lfc-cutoffs"      = config$lfc_cutoffs      <<- as_num_vec(take_value(i, a), a),
      "--z-threshold"      = config$z_hit_threshold  <<- as_num_vec(take_value(i, a), a)[1],
      "--no-figures"       = { config$make_figures   <<- FALSE; skip <- 1L },
      "--no-install"       = { config$install_missing <<- FALSE; skip <- 1L },
      "--list"             = { only_list <- TRUE; skip <- 1L },
      "--help"             = { cat(USAGE); quit(save = "no", status = 0) },
      stop("Unknown argument: ", a, "\n", USAGE, call. = FALSE)
    )
    i <- i + skip
  }
  only_list
}

ONLY_LIST <- parse_args(commandArgs(trailingOnly = TRUE))

# ======================= PACKAGES ==========================================

ensure_packages <- function() {
  cran_pkgs <- c("dplyr", "tidyr", "readr", "stringr", "purrr", "tibble",
                 "ggplot2", "pheatmap")
  missing_cran <- cran_pkgs[!vapply(cran_pkgs, requireNamespace,
                                    logical(1), quietly = TRUE)]
  if (length(missing_cran)) {
    if (!config$install_missing) {
      stop("Missing packages: ", paste(missing_cran, collapse = ", "),
           "\n  install.packages(c(",
           paste0('"', missing_cran, '"', collapse = ", "), "))", call. = FALSE)
    }
    message("Installing missing CRAN packages: ",
            paste(missing_cran, collapse = ", "))
    utils::install.packages(missing_cran, repos = "https://cloud.r-project.org")
  }

  if (!requireNamespace("KRSA", quietly = TRUE)) {
    if (!config$install_missing) {
      stop("KRSA is not installed. Install it from ", KRSA_SOURCE, ":\n",
           '  remotes::install_github("CogDisResLab/KRSA")', call. = FALSE)
    }
    if (!requireNamespace("remotes", quietly = TRUE)) {
      utils::install.packages("remotes", repos = "https://cloud.r-project.org")
    }
    message("KRSA not found -- installing from ", KRSA_SOURCE, " ...")
    remotes::install_github("CogDisResLab/KRSA", upgrade = "never")
    if (!requireNamespace("KRSA", quietly = TRUE)) {
      stop("KRSA installation failed; see the messages above.", call. = FALSE)
    }
  }
}
ensure_packages()

suppressPackageStartupMessages({
  library(KRSA)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(purrr)
  library(tibble)   # KRSA plotting calls column_to_rownames() unqualified
  library(ggplot2)
})

KRSA_VERSION <- as.character(utils::packageVersion("KRSA"))

chip_refs <- lapply(chip_ref_names, function(nm) {
  list(cov = get(nm$cov, envir = asNamespace("KRSA")),
       map = get(nm$map, envir = asNamespace("KRSA")),
       cov_name = nm$cov, map_name = nm$map)
})

# ======================= HELPERS ===========================================

#' Short, readable name for a dataset folder:
#'   exports_image_analysis_pyKinaXe_benchmarking_data_set -> benchmarking
#'   exports_image_analysis_pykinaxe_CDRL_vwr-rats-kinome_data_set
#'                                            -> CDRL_vwr-rats-kinome
dataset_label <- function(folder_name) {
  lbl <- sub("(?i)^exports?_image_analysis_py_?kinaxe_?", "", folder_name,
             perl = TRUE)
  lbl <- sub("(?i)^exports?_", "", lbl, perl = TRUE)
  lbl <- sub("(?i)_data_?set$", "", lbl, perl = TRUE)
  if (!nzchar(lbl)) folder_name else lbl
}

list_export_files <- function(dir) {
  list.files(dir, pattern = "_Export_image_analysis_(PTK|STK)_bn\\.csv$",
             recursive = TRUE, full.names = TRUE)
}

#' Datasets under `input_dir`. If the exports sit DIRECTLY in it (any depth
#' below it but not inside a sibling dataset folder), the folder itself is the
#' single dataset and gets no extra output level -- see the OUTPUT block above.
discover_datasets <- function(input_dir) {
  if (!dir.exists(input_dir)) {
    stop("Input folder does not exist: ", input_dir, call. = FALSE)
  }
  subdirs <- list.dirs(input_dir, recursive = FALSE, full.names = TRUE)
  with_exports <- subdirs[vapply(subdirs,
                                 function(d) length(list_export_files(d)) > 0,
                                 logical(1))]
  # A dataset folder is a subfolder that itself holds chip-run folders with
  # exports. If the exports are one level down only (input_dir IS a dataset),
  # every "with_exports" entry is a chip-run folder instead -- distinguish the
  # two by whether the export sits directly in the subfolder.
  direct_child_export <- vapply(with_exports, function(d) {
    any(grepl("_Export_image_analysis_(PTK|STK)_bn\\.csv$",
              list.files(d, full.names = FALSE)))
  }, logical(1))

  if (length(with_exports) && any(direct_child_export)) {
    return(list(list(label = dataset_label(basename(input_dir)),
                     dir = input_dir, nested = FALSE)))
  }
  if (!length(with_exports)) {
    if (length(list_export_files(input_dir))) {
      return(list(list(label = dataset_label(basename(input_dir)),
                       dir = input_dir, nested = FALSE)))
    }
    stop("No '*_Export_image_analysis_<PTK|STK>_bn.csv' found under ",
         input_dir, call. = FALSE)
  }
  lapply(with_exports, function(d) list(label = dataset_label(basename(d)),
                                        dir = d, nested = TRUE))
}

select_datasets <- function(datasets, wanted) {
  if (is.null(wanted) || !length(wanted)) return(datasets)
  keep <- vapply(datasets, function(ds) {
    any(vapply(wanted, function(w) {
      grepl(w, ds$label, ignore.case = TRUE, fixed = FALSE) ||
        grepl(w, basename(ds$dir), ignore.case = TRUE, fixed = FALSE)
    }, logical(1)))
  }, logical(1))
  if (!any(keep)) {
    stop("No dataset matched --datasets ", paste(wanted, collapse = ","),
         "\n  available: ",
         paste(vapply(datasets, `[[`, character(1), "label"), collapse = ", "),
         call. = FALSE)
  }
  datasets[keep]
}

#' Read a pyKinaXe "Export_image_analysis" *_bn.csv and reshape it into the tidy
#' schema the KRSA functions expect (SampleName/Peptide/ExposureTime/Signal/
#' SignalSaturation/Barcode/Group/Cycle).
read_image_analysis <- function(bn_file, meta_file, chip_type) {
  raw <- readr::read_csv(bn_file, show_col_types = FALSE,
                         progress = FALSE)

  required <- c("Barcode", "Row", "Exposure Time", "Cycle", "ID",
                "I_median", "Signal_Saturation")
  missing <- setdiff(required, colnames(raw))
  if (length(missing)) {
    stop("Export file is missing expected columns: ",
         paste(missing, collapse = ", "), call. = FALSE)
  }

  meta <- read_sample_annotation(meta_file, chip_type)

  tidy <- raw %>%
    dplyr::rename(
      Peptide          = "ID",
      Signal           = "I_median",
      SignalSaturation = "Signal_Saturation",
      ExposureTime     = "Exposure Time"
    ) %>%
    dplyr::group_by(.data$Barcode, .data$Row, .data$Peptide,
                    .data$ExposureTime, .data$Cycle) %>%
    dplyr::summarise(
      Signal           = config$spot_aggregate(.data$Signal),
      SignalSaturation = config$spot_aggregate(.data$SignalSaturation),
      .groups = "drop"
    ) %>%
    dplyr::inner_join(meta, by = c("Barcode", "Row"))

  if (!nrow(tidy)) {
    stop("No row of ", basename(bn_file), " matched the ", chip_type,
         " rows of ", basename(meta_file),
         " on (Barcode, Row) -- wrong annotation file?", call. = FALSE)
  }

  n_unannotated <- raw %>%
    dplyr::distinct(.data$Barcode, .data$Row) %>%
    dplyr::anti_join(meta, by = c("Barcode", "Row")) %>%
    nrow()
  if (n_unannotated) {
    warning(sprintf("%d (Barcode, Row) well(s) of %s carry no %s annotation ",
                    n_unannotated, basename(bn_file), chip_type),
            "and were dropped.", call. = FALSE)
  }

  attr(tidy, "sample_meta") <- meta
  tidy
}

#' Read the sample annotation and resolve which construct is a control and which
#' a test.
#'
#' Two schemas are accepted:
#'   current (>= 2026-08-25): a `Condition Role` column holding control/test,
#'       while `Test Condition` is the sample name and carries no role;
#'   legacy: no role column, `Test Condition == "Control"` marks the control.
read_sample_annotation <- function(meta_file, chip_type) {
  meta_raw <- readr::read_tsv(meta_file, show_col_types = FALSE,
                              progress = FALSE)

  needed <- c("Barcode", "Row", "Sample name", "Construct", "Type")
  missing <- setdiff(needed, colnames(meta_raw))
  if (length(missing)) {
    stop(basename(meta_file), " is missing column(s): ",
         paste(missing, collapse = ", "), call. = FALSE)
  }

  meta <- meta_raw %>% dplyr::filter(.data$Type == chip_type)
  if (!nrow(meta)) {
    stop(basename(meta_file), " has no rows with Type == '", chip_type, "'.",
         call. = FALSE)
  }

  if ("Condition Role" %in% colnames(meta)) {
    role <- tolower(trimws(as.character(meta[["Condition Role"]])))
    schema <- "Condition Role"
  } else if ("Test Condition" %in% colnames(meta)) {
    role <- ifelse(tolower(trimws(as.character(meta[["Test Condition"]]))) ==
                     "control", "control", "test")
    schema <- "legacy Test Condition == 'Control'"
  } else {
    stop(basename(meta_file),
         " has neither a 'Condition Role' nor a 'Test Condition' column, ",
         "so control and test samples cannot be told apart.", call. = FALSE)
  }
  unknown <- setdiff(unique(role), c("control", "test"))
  if (length(unknown)) {
    stop("Unexpected condition role(s) in ", basename(meta_file), ": ",
         paste(unknown, collapse = ", "), " (expected control/test).",
         call. = FALSE)
  }

  out <- meta %>%
    dplyr::transmute(
      Barcode    = .data$Barcode,
      Row        = .data$Row,
      SampleName = .data$`Sample name`,
      Group      = .data$Construct,
      Role       = role
    )

  # A construct is either a control or a test, never both -- otherwise the
  # comparison list below is ambiguous.
  mixed <- out %>%
    dplyr::distinct(.data$Group, .data$Role) %>%
    dplyr::count(.data$Group) %>%
    dplyr::filter(.data$n > 1)
  if (nrow(mixed)) {
    stop("Construct(s) annotated as BOTH control and test in ",
         basename(meta_file), ": ",
         paste(mixed$Group, collapse = ", "), call. = FALSE)
  }

  attr(out, "role_schema") <- schema
  out
}

#' Every test x control pair, in a stable order. The benchmarking data has one
#' control (mock) and three tests; the rat data two controls and two tests, and
#' all four pairs are what pyKinaXe's family output carries.
build_comparisons <- function(meta) {
  roles    <- meta %>% dplyr::distinct(.data$Group, .data$Role)
  controls <- sort(roles$Group[roles$Role == "control"])
  tests    <- sort(roles$Group[roles$Role == "test"])
  if (!length(controls) || !length(tests)) {
    stop("Need at least one control and one test construct; found controls={",
         paste(controls, collapse = ","), "} tests={",
         paste(tests, collapse = ","), "}", call. = FALSE)
  }
  out <- list()
  for (ctl in controls) for (tst in tests) {
    out[[length(out) + 1L]] <- list(name = paste0(tst, "_vs_", ctl),
                                    test = tst, control = ctl)
  }
  out
}

#' Save one figure. pheatmap draws itself to the open device, ggplot objects are
#' printed. Failures are caught so one bad plot never aborts the pipeline.
save_fig <- function(expr, path, width = 8, height = 8) {
  if (!isTRUE(config$make_figures)) return(invisible())
  tryCatch({
    grDevices::pdf(path, width = width, height = height)
    on.exit(grDevices::dev.off(), add = TRUE)
    obj <- force(expr)
    if (inherits(obj, "ggplot")) print(obj)
  }, error = function(e) {
    message("    [skip figure] ", basename(path), ": ", conditionMessage(e))
  })
}

#' Run krsa() over several peptide sets and stack the Z-score tables, tagging
#' each row with the method (peptide set) it came from.
run_multi_krsa <- function(peptide_sets, map_file, cov_file) {
  peptide_sets <- peptide_sets[vapply(peptide_sets, length, integer(1)) > 0]
  if (!length(peptide_sets)) return(NULL)
  purrr::imap(peptide_sets, function(peps, nm) {
    krsa(peps, itr = config$iterations, seed = config$seed,
         map_file = map_file, cov_file = cov_file) %>%
      dplyr::mutate(method = nm)
  }) %>% dplyr::bind_rows()
}

# ======================= PER-COMPARISON ANALYSIS ===========================

analyse_comparison <- function(comp, data_modeled, data_pw_200,
                               refs, chip_type, dirs) {
  groups <- c(comp$test, comp$control)   # krsa_group_diff: (case, control)
  message(sprintf("  -> %s  (LFC = %s - %s)", comp$name, groups[1], groups[2]))

  ## 1. QC filter peptides for this pair of groups
  pep_qc <- krsa_quick_filter(
    data = data_pw_200, data2 = data_modeled$scaled,
    signal_threshold = config$signal_threshold,
    r2_threshold = config$r2_threshold, groups = groups
  )
  message(sprintf("     %d peptides passed QC", length(pep_qc)))
  if (length(pep_qc) < 3) {
    message("     Too few peptides passed QC; skipping comparison.")
    return(invisible(list(comparison = comp$name, qc_peptides = length(pep_qc),
                          hits = 0L, kinase_hits = character(0),
                          status = "skipped: QC")))
  }

  ## 2. Log2 fold changes (paired within chip)
  diff_df <- krsa_group_diff(data_modeled$scaled, groups, pep_qc, byChip = TRUE)
  readr::write_delim(diff_df,
                     file.path(dirs$tables, paste0(comp$name, "_LFC.txt")),
                     delim = "\t")

  ## 3. Hit peptides by LFC cutoff (across-chip mean and per chip)
  sig_across <- krsa_get_diff(diff_df, totalMeanLFC, config$lfc_cutoffs)
  sig_within <- krsa_get_diff_byChip(diff_df, LFC, config$lfc_cutoffs)

  primary_thr <- as.character(config$lfc_cutoffs[1])
  hits <- sig_across[[primary_thr]]
  message(sprintf("     %d hit peptides at LFC >= %s", length(hits), primary_thr))
  if (length(hits) < 3) {
    message("     Too few hits for random sampling; skipping KRSA step.")
    return(invisible(list(comparison = comp$name, qc_peptides = length(pep_qc),
                          hits = length(hits), kinase_hits = character(0),
                          status = "skipped: too few hit peptides")))
  }

  ## 4. KRSA random sampling
  # 4a. Single representative run (primary cutoff); keeps the count matrix for
  #     the histogram figure.
  fin <- krsa(hits, return_count = TRUE, seed = config$seed,
              itr = config$iterations,
              map_file = refs$map, cov_file = refs$cov)
  readr::write_delim(
    fin$KRSA_Table,
    file.path(dirs$tables, paste0(comp$name, "_KRSA_Zscores_primary.txt")),
    delim = "\t"
  )

  # 4b. Across-chip peptide sets (every LFC cutoff) -> AvgZ. THIS is the table
  #     the family comparison reads: AvgZ is the mean over the cutoffs, which is
  #     the structural counterpart of pyKinaXe's cutoff-averaged Z_Score.
  across_sets <- setNames(sig_across, paste0("meanLFC.", names(sig_across)))
  z_across <- run_multi_krsa(across_sets, refs$map, refs$cov)
  avg_across <- z_across %>%
    dplyr::group_by(Kinase) %>%
    dplyr::mutate(AvgZ = mean(Z)) %>%
    dplyr::ungroup()
  readr::write_delim(
    avg_across,
    file.path(dirs$tables, paste0(comp$name, "_KRSA_acrossChip.txt")),
    delim = "\t"
  )

  # 4c. Within-chip peptide sets (per barcode x cutoff) -> AvgZ
  within_sets <- purrr::imap(sig_within, function(sets, bc) {
    setNames(sets, paste0(bc, ".", names(sets)))
  }) %>% purrr::flatten()
  z_within <- run_multi_krsa(within_sets, refs$map, refs$cov)
  avg_within <- z_within %>%
    dplyr::group_by(Kinase) %>%
    dplyr::mutate(AvgZ = mean(Z)) %>%
    dplyr::ungroup()
  readr::write_delim(
    avg_within,
    file.path(dirs$tables, paste0(comp$name, "_KRSA_withinChip.txt")),
    delim = "\t"
  )

  ## 5. Top kinase hits (|Z| >= threshold, within-chip averaged -- KRSA's own
  ##    convention in its report template).
  kinase_hits <- krsa_top_hits(avg_within, config$z_hit_threshold)
  readr::write_lines(
    kinase_hits,
    file.path(dirs$tables, paste0(comp$name, "_top_kinases.txt"))
  )
  message(sprintf("     %d kinase hits at |Z| >= %s",
                  length(kinase_hits), config$z_hit_threshold))

  ## 6. Figures
  fig <- function(name) {
    file.path(dirs$figures, paste0(comp$name, "_", name, ".pdf"))
  }
  save_fig(krsa_heatmap(data_modeled$normalized, hits,
                        groups = groups, scale = "row"), fig("heatmap"))
  save_fig(krsa_violin_plot(data_modeled$scaled, hits, "Barcode",
                            groups = groups), fig("violin"))
  save_fig(krsa_waterfall(diff_df, lfc_thr = config$lfc_cutoffs[1],
                          byChip = TRUE), fig("waterfall"))
  save_fig(krsa_zscores_plot(avg_across), fig("zscores_acrossChip"), height = 10)
  save_fig(krsa_zscores_plot(avg_within), fig("zscores_withinChip"), height = 10)

  bothways <- c(utils::head(fin$KRSA_Table, 10)$Kinase,
                utils::tail(fin$KRSA_Table, 10)$Kinase)
  save_fig(krsa_histogram_plot(fin$KRSA_Table, fin$count_mtx, bothways),
           fig("histogram"))
  save_fig(krsa_coverage_plot(refs$cov, avg_within, chip_type),
           fig("coverage"), height = 10)

  if (length(kinase_hits)) {
    save_fig(krsa_reverse_krsa_plot(refs$cov, diff_df, kinase_hits,
                                    config$lfc_cutoffs[1], byChip = FALSE),
             fig("reverseKRSA"))
    save_fig(krsa_ball_model(kinase_hits, avg_within, 10, 2.5, 4.8),
             fig("ballmodel"), height = 10)
  }

  invisible(list(comparison = comp$name, qc_peptides = length(pep_qc),
                 hits = length(hits), kinase_hits = kinase_hits,
                 status = "ok"))
}

# ======================= PER-ARRAY ANALYSIS ================================

analyse_array <- function(bn_file, meta_file, chip_type, out_root) {
  message("\n-----------------------------------------------------")
  message(sprintf("Array: %s", chip_type))
  message(sprintf("  data: %s", bn_file))
  message(sprintf("  meta: %s", meta_file))

  refs <- chip_refs[[chip_type]]
  if (is.null(refs)) stop("Unknown chip type: ", chip_type, call. = FALSE)

  dirs <- list(tables  = file.path(out_root, chip_type, "tables"),
               figures = file.path(out_root, chip_type, "figures"))
  invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

  ## Read + reshape pyKinaXe's spot export
  data <- read_image_analysis(bn_file, meta_file, chip_type)
  meta <- attr(data, "sample_meta")

  ## KRSA's QC pre-processing (signal < 1 -> 1, drop saturated points)
  data <- krsa_qc_steps(data)

  ## End-point cycle (last cycle), all exposures and the maximum exposure only
  data_pw_200 <- krsa_extractEndPointMaxExp(data, chip_type)
  data_pw     <- krsa_extractEndPoint(data, chip_type)

  ## Linear model fit / scaling / normalisation
  data_modeled <- krsa_scaleModel(data_pw, unique(data_pw$Peptide))

  ## Global QC peptide list (KRSA's own; the number to compare against
  ## pyKinaXe's peptide count)
  ppPassAll <- krsa_filter_lowPeps(data_pw_200, config$signal_threshold)
  ppPassR2  <- krsa_filter_nonLinear(
    dplyr::filter(data_modeled$scaled, Peptide %in% ppPassAll),
    config$r2_threshold
  )
  new_pep <- krsa_filter_ref_pep(ppPassR2)
  readr::write_lines(new_pep,
                     file.path(dirs$tables, "global_qc_passed_peptides.txt"))
  n_pep_total <- dplyr::n_distinct(data_pw$Peptide)
  message(sprintf("  %d of %d peptides passed global QC",
                  length(new_pep), n_pep_total))

  ## Global figures
  save_fig(krsa_cv_plot(data_modeled$scaled, new_pep),
           file.path(dirs$figures, "global_cv.pdf"))
  save_fig(krsa_violin_plot(data_modeled$scaled, new_pep, "Group"),
           file.path(dirs$figures, "global_violin_byGroup.pdf"))
  save_fig(krsa_heatmap(data_modeled$scaled, new_pep, scale = "row"),
           file.path(dirs$figures, "global_heatmap.pdf"))
  save_fig(krsa_heatmap_grouped(data_modeled$grouped, new_pep, scale = "row"),
           file.path(dirs$figures, "global_heatmap_grouped.pdf"))

  ## Comparisons: every test x control pair
  comparisons <- build_comparisons(meta)
  message(sprintf("  Roles from '%s': controls={%s} tests={%s}",
                  attr(meta, "role_schema"),
                  paste(sort(unique(meta$Group[meta$Role == "control"])),
                        collapse = ", "),
                  paste(sort(unique(meta$Group[meta$Role == "test"])),
                        collapse = ", ")))
  message(sprintf("  %d comparison(s): %s", length(comparisons),
                  paste(vapply(comparisons, `[[`, character(1), "name"),
                        collapse = ", ")))

  results <- lapply(comparisons, function(comp) {
    tryCatch(
      analyse_comparison(comp, data_modeled, data_pw_200,
                         refs, chip_type, dirs),
      error = function(e) {
        message("     [ERROR] ", comp$name, ": ", conditionMessage(e))
        list(comparison = comp$name, qc_peptides = NA_integer_, hits = NA_integer_,
             kinase_hits = character(0),
             status = paste("failed:", conditionMessage(e)))
      }
    )
  })

  list(array = chip_type, bn_file = bn_file, meta_file = meta_file,
       peptides_total = n_pep_total, peptides_qc = length(new_pep),
       map = refs$map_name, cov = refs$cov_name,
       comparisons = results)
}

# ======================= PER-DATASET ANALYSIS ==============================

analyse_dataset <- function(ds, out_root) {
  message("\n=====================================================")
  message(sprintf("Dataset: %s", ds$label))
  message(sprintf("  input: %s", ds$dir))
  message(sprintf("  output: %s", out_root))
  message("=====================================================")
  dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

  bn_files <- sort(list_export_files(ds$dir))
  array_results <- list()
  for (bn in bn_files) {
    chip_type <- stringr::str_match(
      basename(bn), "_Export_image_analysis_(PTK|STK)_bn\\.csv$")[, 2]
    if (is.na(chip_type)) {
      message("Skipping (cannot determine array): ", bn)
      next
    }
    meta_file <- list.files(dirname(bn), pattern = "_data_enrichment\\.txt$",
                            full.names = TRUE)
    if (!length(meta_file)) {
      message("Skipping (no *_data_enrichment.txt next to it): ", dirname(bn))
      next
    }
    res <- tryCatch(
      analyse_array(bn, meta_file[1], chip_type, out_root),
      error = function(e) {
        message("  [ERROR] array failed: ", conditionMessage(e))
        list(array = chip_type, bn_file = bn, meta_file = meta_file[1],
             peptides_total = NA_integer_, peptides_qc = NA_integer_,
             map = NA_character_, cov = NA_character_,
             comparisons = list(), error = conditionMessage(e))
      }
    )
    array_results[[length(array_results) + 1L]] <- res
  }
  if (!length(array_results)) {
    stop("Nothing to analyse in ", ds$dir, call. = FALSE)
  }
  write_manifest(ds, out_root, array_results)
  list(label = ds$label, dir = ds$dir, out = out_root, arrays = array_results)
}

write_manifest <- function(ds, out_root, array_results) {
  lines <- c(
    "KRSA on pyKinaXe image analysis -- run manifest",
    "==============================================",
    sprintf("written           : %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    sprintf("script            : %s",
            rel_to_repo(file.path(SCRIPT_DIR, "run_krsa_on_pykinaxe_image_analysis.R"))),
    sprintf("dataset           : %s", ds$label),
    sprintf("input folder      : %s", rel_to_repo(ds$dir)),
    sprintf("output folder     : %s", rel_to_repo(out_root)),
    "",
    sprintf("KRSA version      : %s  (%s)", KRSA_VERSION, KRSA_SOURCE),
    sprintf("R version         : %s", R.version.string),
    "",
    "thresholds and settings (change one side only at your peril -- the family",
    "comparison assumes pyKinaXe used the same):",
    sprintf("  signal_threshold: %s", config$signal_threshold),
    sprintf("  r2_threshold    : %s", config$r2_threshold),
    sprintf("  lfc_cutoffs     : %s", paste(config$lfc_cutoffs, collapse = ", ")),
    sprintf("  z_hit_threshold : %s", config$z_hit_threshold),
    sprintf("  iterations      : %s", config$iterations),
    sprintf("  seed            : %s", config$seed),
    sprintf("  figures         : %s", if (config$make_figures) "yes" else "no"),
    ""
  )
  for (ar in array_results) {
    lines <- c(lines,
      sprintf("[%s]", ar$array),
      sprintf("  spot export     : %s", basename(ar$bn_file)),
      sprintf("  sample annotation: %s", basename(ar$meta_file)),
      sprintf("  mapping table   : %s", ar$map),
      sprintf("  coverage table  : %s", ar$cov),
      sprintf("  peptides passing global QC: %s of %s",
              ar$peptides_qc, ar$peptides_total))
    if (!is.null(ar$error)) lines <- c(lines, sprintf("  ERROR: %s", ar$error))
    for (cr in ar$comparisons) {
      lines <- c(lines, sprintf(
        "  %-28s QC peptides %-4s hit peptides %-4s kinase hits %-3s  %s",
        cr$comparison, cr$qc_peptides, cr$hits, length(cr$kinase_hits),
        if (length(cr$kinase_hits))
          paste0("(", paste(cr$kinase_hits, collapse = ", "), ")")
        else cr$status))
    }
    lines <- c(lines, "")
  }
  writeLines(lines, file.path(out_root, "run_manifest.txt"))
}

# ======================= DRIVER ============================================

main <- function() {
  input_dir  <- normalizePath(config$input_dir, mustWork = FALSE)
  datasets   <- select_datasets(discover_datasets(input_dir), config$datasets)

  if (ONLY_LIST) {
    cat("Datasets under ", input_dir, ":\n", sep = "")
    for (ds in datasets) {
      cat(sprintf("  %-26s %s\n", ds$label, ds$dir))
      for (bn in sort(list_export_files(ds$dir))) {
        cat(sprintf("      %s\n", basename(bn)))
      }
    }
    return(invisible(NULL))
  }

  message(sprintf("KRSA %s on pyKinaXe image analysis -- %d dataset(s): %s",
                  KRSA_VERSION, length(datasets),
                  paste(vapply(datasets, `[[`, character(1), "label"),
                        collapse = ", ")))

  run <- list()
  for (ds in datasets) {
    out_root <- if (isTRUE(ds$nested)) {
      file.path(config$output_dir, ds$label)
    } else {
      config$output_dir
    }
    run[[length(run) + 1L]] <- tryCatch(
      analyse_dataset(ds, out_root),
      error = function(e) {
        message("[ERROR] dataset ", ds$label, " failed: ", conditionMessage(e))
        list(label = ds$label, dir = ds$dir, out = out_root, arrays = list(),
             error = conditionMessage(e))
      }
    )
  }

  ## Final summary -- the same numbers the manifests carry, in one place.
  message("\n=====================================================")
  message("SUMMARY")
  message("=====================================================")
  failed <- 0L
  for (r in run) {
    if (!is.null(r$error)) {
      message(sprintf("  %-24s FAILED: %s", r$label, r$error))
      failed <- failed + 1L
      next
    }
    for (ar in r$arrays) {
      message(sprintf("  %-24s %s  QC peptides %s/%s",
                      r$label, ar$array, ar$peptides_qc, ar$peptides_total))
      for (cr in ar$comparisons) {
        if (!identical(cr$status, "ok")) failed <- failed + 1L
        message(sprintf("      %-30s hits %-4s kinases %s%s",
                        cr$comparison, cr$hits, length(cr$kinase_hits),
                        if (identical(cr$status, "ok")) ""
                        else paste0("  [", cr$status, "]")))
      }
    }
    message(sprintf("      -> %s", r$out))
  }
  if (failed) {
    message(sprintf("\n%d comparison(s)/dataset(s) did not complete.", failed))
  }
  message("\nDone.")
  invisible(run)
}

main()
