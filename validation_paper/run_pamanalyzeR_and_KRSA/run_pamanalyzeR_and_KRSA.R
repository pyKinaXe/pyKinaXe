#!/usr/bin/env Rscript
# run_pamanalyzeR_and_KRSA.R
#
# Runs the published R stack end to end on raw PamChip TIFFs and times both
# stages: pamgeneAnalyzeR (https://github.com/amelbek/pamgeneAnalyzeR) for the
# spot quantification and KRSA (https://github.com/CogDisResLab/KRSA) for the
# kinase-family scoring. No pyKinaXe code or output is used. NOTE.txt in this
# folder describes the method, the measured properties of pamgeneAnalyzeR and
# the results on record.
#
# Per dataset and array the script records the wall-clock and CPU time of
#   stage 1  pamgeneAnalyzeR: reference selection, grid detection, registration,
#            per-spot signal extraction, merge, Z'-factor, background subtraction
#   stage 2  KRSA: QC, end-point extraction, linear model, log2 fold changes,
#            hit peptides, krsa() over every LFC cutoff, hit kinases
#
# Points to know when reading the output (all measured on this repository's
# data; details in NOTE.txt):
#   * find_centers() returns Row/Col transposed relative to PamGene's
#     "* Array Layout.txt"; the layout is transposed before the merge
#     (transpose_layout()).
#   * find_centers() and substractBackground() are STK-only (12 x 12 grid, 160
#     columns). PTK grids come from locate_grid_from_profiles(), an extension in
#     this script; registration and extraction remain package code on both arrays.
#   * find_centers() fails on R >= 4.0 (class(matrix) has length 2).
#     read_target_tiff() sets a single-element class on the matrix so the
#     package runs unmodified; patch_find_centers() is the fallback.
#   * Signals are scaled by 65535 before KRSA (--signal-scale) and KRSA's
#     saturation filter is off, because pamgeneAnalyzeR produces no saturation
#     measure. All other KRSA settings match run_krsa_on_pykinaxe_image_analysis.R.
#
# Input: --data-dir (default <repo>/data) with the dataset folders
# benchmarking_data_set/ and CDRL_vwr-rats-kinome_data_set/, each holding one
# run folder per array with ImageResults/*.tif, "* Array Layout.txt" and
# "* Sample Annotation.txt". The array type is read from the layout columns.
# Groups and roles come from the sample names (c/t prefix, or --controls regex).
#
# Output (--output-dir, default <script dir>/output): per dataset and array the
# pamgeneAnalyzeR signal files, raw_signal.tsv, normalised_signal.tsv and the
# KRSA tables (same names as run_krsa_on_pykinaxe_image_analysis.R), plus
# run_manifest.txt per dataset, timing.csv and timing_summary.txt.
#
# Usage:
#   Rscript run_pamanalyzeR_and_KRSA.R                       # both datasets
#   Rscript run_pamanalyzeR_and_KRSA.R --datasets benchmarking
#   Rscript run_pamanalyzeR_and_KRSA.R --limit-images 8      # smoke test, not timed
#   Rscript run_pamanalyzeR_and_KRSA.R --help
# Paths resolve relative to this script. Stage 1 is parallel by package default;
# --cores 1 makes it serial. --reuse-signal skips stage 1 and marks the run as
# not timed.

PAMGENEANALYZER_SOURCE <- "https://github.com/amelbek/pamgeneAnalyzeR"
KRSA_SOURCE            <- "https://github.com/CogDisResLab/KRSA"

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
REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."), mustWork = FALSE)

# Paths in the run manifest are written relative to the repository root.
rel_to_repo <- function(path) {
  p    <- normalizePath(path, mustWork = FALSE)
  root <- paste0(normalizePath(REPO_ROOT, mustWork = FALSE), "/")
  ifelse(startsWith(p, root), substring(p, nchar(root) + 1), p)
}

# ======================= CONFIGURATION =====================================

config <- list(
  data_dir   = file.path(REPO_ROOT, "data"),
  output_dir = file.path(SCRIPT_DIR, "output"),

  # NULL = every dataset folder found. Substring match, case-insensitive,
  # against the folder name and the short label, so --datasets rat works.
  datasets = NULL,

  # --- pamgeneAnalyzeR -----------------------------------------------------
  radius = 5,            # spot aperture radius in px; the package default
  # "published": find_centers() on STK, locate_grid_from_profiles() on PTK
  #              (the package cannot do PTK at all; NOTE.txt, property 2)
  # "profile"  : the profile locator on both arrays
  grid_mode = "published",
  # Which arrays to run. "STK" is the package-faithful benchmark: the only
  # array pamgeneAnalyzeR itself supports, on its own published code path,
  # with no extension from this script anywhere in it. See --arrays.
  arrays = c("PTK", "STK"),
  # --unmodified: run pamgeneAnalyzeR EXACTLY as its vignette does, as a pure
  # image-analysis benchmark. Forces STK (the only array it supports), passes
  # the layout untransposed, and takes the vignette's reference image (the
  # first file in the folder). The peptide labels are then wrong (NOTE.txt,
  # property 1), so the KRSA stage is switched off: this mode measures TIME,
  # not results. Pair it with run_krsa_on_bionavigator.R, which times KRSA on
  # its own native input.
  unmodified = FALSE,
  # niftyreg registration parameters; the package defaults
  n_levels = 4, max_iterations = 5, use_block_percentage = 50,
  cores = NULL,          # NULL = detectCores() - 1
  limit_images = NULL,   # smoke test: only this many TIFFs per array

  # --- the bridge ----------------------------------------------------------
  signal_scale = 65535,  # [0,1] float -> 16-bit counts; see "THE BRIDGE"

  # --- KRSA ----------------------------------------------------------------
  # Identical to run_krsa_on_pykinaxe_image_analysis.R. Do not change one side
  # only -- the comparison assumes both used these.
  signal_threshold = 5,
  r2_threshold     = 0.9,
  lfc_cutoffs      = c(0.2, 0.3, 0.4),
  z_hit_threshold  = 2,
  iterations       = 2000,
  seed             = 123,

  # --- annotation ----------------------------------------------------------
  # Which constructs are controls when `Sample name` has no c/t prefix.
  controls = "^mock$",

  install_missing = TRUE,
  reuse_signal    = FALSE,
  run_image_analysis = TRUE,
  run_krsa           = TRUE
)

# KRSA's chip-specific mapping/coverage tables. These -- not the data -- decide
# which peptides count as a kinase family's substrates. Same tables as the
# sibling script, so the two runs are comparable.
chip_ref_names <- list(
  PTK = list(cov = "KRSA_coverage_PTK_PamChip_86402_v1",
             map = "KRSA_Mapping_PTK_PamChip_86402_v1"),
  STK = list(cov = "KRSA_coverage_STK_PamChip_87102_v2",
             map = "KRSA_Mapping_STK_PamChip_87102_v1")
)

# ======================= COMMAND LINE ======================================

USAGE <- "
Usage: Rscript run_pamanalyzeR_and_KRSA.R [options]

  input / output
    --data-dir PATH       folder holding the *_data_set folders
                          (default: <repo>/data)
    --output-dir PATH     where results go (default: <script dir>/output)
    --datasets A,B        only these datasets (substring match; default: all)
    --list                list what would run, then exit

  pamgeneAnalyzeR (stage 1)
    --radius N            spot aperture radius in px (default: 5)
    --grid MODE           published | profile (default: published)
                          published = find_centers() on STK, profile locator on
                          PTK (the package has no PTK support at all)
    --unmodified          run pamgeneAnalyzeR EXACTLY as its vignette does:
                          STK only, layout NOT transposed, reference image =
                          first file in the folder. A pure IMAGE-ANALYSIS
                          TIMING -- peptide labels are mislabelled in this mode,
                          so KRSA is switched off. Use
                          run_krsa_on_bionavigator.R for the KRSA timing.
    --arrays A,B          PTK,STK (default: both). --arrays STK is the
                          PACKAGE-FAITHFUL benchmark: the only array
                          pamgeneAnalyzeR supports, entirely on its own
                          published code path, no extension from this script involved.
    --cores N             workers for registration (default: detectCores() - 1)
    --limit-images N      only N images per array -- SMOKE TEST, not a timing
    --signal-scale X      [0,1] float -> counts factor (default: 65535)

  KRSA (stage 2)
    --signal-threshold X  min end-point signal (default: 5)
    --r2-threshold X      min R^2 of the signal ~ exposure fit (default: 0.9)
    --lfc-cutoffs A,B,C   hit-peptide log2 FC cutoffs (default: 0.2,0.3,0.4)
    --z-threshold X       |Z| for a kinase-family hit (default: 2)
    --iterations N        krsa() sampling iterations (default: 2000)
    --seed N              sampling seed (default: 123)

  annotation
    --controls REGEX      constructs that are controls when the sample name
                          carries no c/t prefix (default: '^mock$')

  staging
    --no-image-analysis   skip stage 1 (needs --reuse-signal data present)
    --no-krsa             skip stage 2 (time the image analysis alone)
    --reuse-signal        reuse per-image signal files where they exist.
                          MARKS THE RUN AS NOT TIMED. For iteration only.
    --no-install          never install a missing package; fail instead
    --help                this text
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
      "--data-dir"         = config$data_dir   <<- take_value(i, a),
      "--output-dir"       = config$output_dir <<- take_value(i, a),
      "--datasets"         = config$datasets   <<- trimws(strsplit(take_value(i, a), ",")[[1]]),
      "--radius"           = config$radius     <<- as_num_vec(take_value(i, a), a)[1],
      "--grid"             = config$grid_mode  <<- take_value(i, a),
      "--arrays"           = config$arrays     <<- toupper(trimws(strsplit(take_value(i, a), ",")[[1]])),
      "--unmodified"       = { config$unmodified <<- TRUE; skip <- 1L },
      "--cores"            = config$cores      <<- as_num_vec(take_value(i, a), a)[1],
      "--limit-images"     = config$limit_images <<- as_num_vec(take_value(i, a), a)[1],
      "--signal-scale"     = config$signal_scale <<- as_num_vec(take_value(i, a), a)[1],
      "--signal-threshold" = config$signal_threshold <<- as_num_vec(take_value(i, a), a)[1],
      "--r2-threshold"     = config$r2_threshold     <<- as_num_vec(take_value(i, a), a)[1],
      "--lfc-cutoffs"      = config$lfc_cutoffs      <<- as_num_vec(take_value(i, a), a),
      "--z-threshold"      = config$z_hit_threshold  <<- as_num_vec(take_value(i, a), a)[1],
      "--iterations"       = config$iterations <<- as_num_vec(take_value(i, a), a)[1],
      "--seed"             = config$seed       <<- as_num_vec(take_value(i, a), a)[1],
      "--controls"         = config$controls   <<- take_value(i, a),
      "--no-image-analysis" = { config$run_image_analysis <<- FALSE; skip <- 1L },
      "--no-krsa"          = { config$run_krsa        <<- FALSE; skip <- 1L },
      "--reuse-signal"     = { config$reuse_signal    <<- TRUE;  skip <- 1L },
      "--no-install"       = { config$install_missing <<- FALSE; skip <- 1L },
      "--list"             = { only_list <- TRUE; skip <- 1L },
      "--help"             = { cat(USAGE); quit(save = "no", status = 0) },
      stop("Unknown argument: ", a, "\n", USAGE, call. = FALSE)
    )
    i <- i + skip
  }
  if (isTRUE(config$unmodified)) {
    config$arrays   <<- "STK"    # the only array the package can address
    config$grid_mode <<- "published"
    config$run_krsa  <<- FALSE   # labels are wrong without the transpose
    message("[--unmodified] pamgeneAnalyzeR exactly as its vignette runs it: ",
            "STK only, layout NOT transposed, reference image = first file. ",
            "This is an IMAGE-ANALYSIS TIMING; KRSA is off because the peptide ",
            "labels are mislabelled without the transpose (see FINDING 1). ",
            "Time KRSA with run_krsa_on_bionavigator.R instead.")
  }
  bad <- setdiff(config$arrays, c("PTK", "STK"))
  if (length(bad)) {
    stop("--arrays takes PTK and/or STK, got: ", paste(bad, collapse = ", "),
         call. = FALSE)
  }
  if (!config$grid_mode %in% c("published", "profile")) {
    stop("--grid must be 'published' or 'profile', got: ", config$grid_mode,
         call. = FALSE)
  }
  only_list
}

ONLY_LIST <- parse_args(commandArgs(trailingOnly = TRUE))

# ======================= PACKAGES ==========================================

ensure_packages <- function() {
  # gtools is used by mergeExperiments() and doParallel by registerAllImages(),
  # but neither is declared in pamgeneAnalyzeR's DESCRIPTION, so they have to
  # be asked for explicitly here.
  cran_pkgs <- c("dplyr", "tidyr", "readr", "stringr", "purrr", "tibble",
                 "ggplot2", "tiff", "imager", "RNiftyReg", "mmand",
                 "foreach", "doParallel", "gtools")
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

  github_pkg <- function(pkg, repo, source) {
    if (requireNamespace(pkg, quietly = TRUE)) return(invisible())
    if (!config$install_missing) {
      stop(pkg, " is not installed. Install it from ", source, ":\n",
           '  remotes::install_github("', repo, '")', call. = FALSE)
    }
    if (!requireNamespace("remotes", quietly = TRUE)) {
      utils::install.packages("remotes", repos = "https://cloud.r-project.org")
    }
    message(pkg, " not found -- installing from ", source, " ...")
    remotes::install_github(repo, upgrade = "never")
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(pkg, " installation failed; see the messages above.", call. = FALSE)
    }
  }
  github_pkg("pamgeneAnalyzeR", "amelbek/pamgeneAnalyzeR", PAMGENEANALYZER_SOURCE)
  github_pkg("KRSA", "CogDisResLab/KRSA", KRSA_SOURCE)
}
ensure_packages()

suppressPackageStartupMessages({
  library(pamgeneAnalyzeR)
  library(KRSA)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(purrr)
  library(tibble)
  library(ggplot2)
  library(tiff)
})

PAMGENEANALYZER_VERSION <- as.character(utils::packageVersion("pamgeneAnalyzeR"))
KRSA_VERSION            <- as.character(utils::packageVersion("KRSA"))

chip_refs <- lapply(chip_ref_names, function(nm) {
  list(cov = get(nm$cov, envir = asNamespace("KRSA")),
       map = get(nm$map, envir = asNamespace("KRSA")),
       cov_name = nm$cov, map_name = nm$map)
})

# registerAllImages() picks its OWN worker count internally --
#   cores = parallel::detectCores(); cl <- parallel::makeCluster(cores - 1)
# and ignores anything passed to it. So --cores cannot set a worker count; it
# can only choose between the package's parallel default and serial execution
# (parallel.registration = TRUE / FALSE). Report what the package will actually
# use, and say so plainly rather than printing a number the script does not control.
PKG_CORES <- max(1L, parallel::detectCores() - 1L)
N_CORES <- if (is.null(config$cores)) PKG_CORES else as.integer(config$cores)
if (N_CORES > 1L && N_CORES != PKG_CORES) {
  message(sprintf(paste("[note] --cores %d ignored for registration:",
                        "registerAllImages() sets its own worker count",
                        "(detectCores() - 1 = %d). Use --cores 1 for serial."),
                  N_CORES, PKG_CORES))
  N_CORES <- PKG_CORES
}
PARALLEL_REG <- N_CORES > 1L
CORES_NOTE <- if (PARALLEL_REG) {
  sprintf("%d (registerAllImages()'s own default, detectCores() - 1)", PKG_CORES)
} else {
  "1 -- serial (parallel.registration = FALSE; NOT the package default)"
}

# ======================= R >= 4.0 COMPATIBILITY ============================

#' Read a TIFF so that pamgeneAnalyzeR::find_centers() accepts it.
#'
#' find_centers() guards its input with `if (class(image) != "matrix")`, which
#' errors on R >= 4.0 because class(a_matrix) is c("matrix", "array") and `!=`
#' therefore returns a length-2 vector. Setting an explicit single-element
#' class on the matrix satisfies the guard without touching the package, so the
#' package runs as shipped.
read_target_tiff <- function(path) {
  structure(unclass(tiff::readTIFF(source = path)), class = "matrix")
}

#' Fallback for the same problem: rewrite that one guard to is.matrix() and
#' leave every other line of find_centers() untouched. Only used if the class
#' stamp above turns out not to satisfy the guard (a future package version
#' checking the class differently).
patch_find_centers <- function() {
  ns  <- asNamespace("pamgeneAnalyzeR")
  src <- deparse(get("find_centers", envir = ns))
  hit <- grepl('class(image) != "matrix"', src, fixed = TRUE)
  if (!any(hit)) {
    # A future release fixed it; use the package function unchanged.
    return(get("find_centers", envir = ns))
  }
  src[hit] <- sub('class(image) != "matrix"', '!is.matrix(image)',
                  src[hit], fixed = TRUE)
  eval(parse(text = paste(src, collapse = "\n")), envir = ns)
}
find_centers_compat <- patch_find_centers()

# ======================= THE GRID EXTENSION ================================

#' Locate an n x n spot grid from the image alone.
#'
#' Smooth; clip the top `clip_q` of pixels so that the very bright control
#' spots cannot dominate a projection; project onto each axis; then match a
#' comb of n teeth over pitch and phase, scoring (mean at the nodes) minus
#' (mean at the half-pitch interstices). The interstitial term is what makes
#' the score peak at the true phase instead of anywhere bright.
#'
#' Nothing outside the image is consulted. See the header for the validation.
#'
#' Returns pitch and origin for each axis, in the image's own (dim 1, dim 2)
#' coordinates -- `x` indexes dim 1, `y` indexes dim 2, matching the package's
#' own convention in extract_signal().
locate_grid_from_profiles <- function(img, n, pitch_range = c(21.0, 22.0),
                                      margin = 30, clip_q = 0.98,
                                      pitch_step = 0.05, phase_step = 0.5) {
  sm  <- t(as.matrix(imager::isoblur(imager::as.cimg(t(img)), 2, gaussian = TRUE)))
  cap <- stats::quantile(sm, clip_q, na.rm = TRUE)
  sm[sm > cap] <- cap

  profile <- function(v) { v <- v - stats::median(v); v[v < 0] <- 0; v }

  fit_axis <- function(p_prof, len) {
    best <- list(score = -Inf, pitch = NA_real_, origin = NA_real_)
    for (p in seq(pitch_range[1], pitch_range[2], by = pitch_step)) {
      hi <- len - p * (n - 1) - margin
      if (hi <= margin) next
      for (o in seq(margin, hi, by = phase_step)) {
        nodes <- round(seq(o,         by = p, length.out = n))
        inter <- round(seq(o + p / 2, by = p, length.out = n - 1))
        s <- mean(p_prof[nodes]) - mean(p_prof[inter])
        if (s > best$score) best <- list(score = s, pitch = p, origin = o)
      }
    }
    if (!is.finite(best$score)) {
      stop("Could not locate a ", n, "x", n, " grid in this image.", call. = FALSE)
    }
    best
  }

  fx <- fit_axis(profile(rowSums(sm)), nrow(sm))
  fy <- fit_axis(profile(colSums(sm)), ncol(sm))
  list(pitch_x = fx$pitch, x0 = fx$origin,
       pitch_y = fy$pitch, y0 = fy$origin)
}

#' Build a centers.coords table in exactly the shape find_centers() returns
#' (names / x / y / Row / Col), from a located grid.
#'
#' Row and Col follow the PACKAGE's convention, verified against it on STK:
#' Col indexes the x axis (dim 1) and Row indexes the y axis (dim 2). Keeping
#' that convention means the same transpose_layout() fix applies to both code
#' paths (NOTE.txt, property 1).
#'
#' The 8 background points are placed `bg_gap` pitches outside the grid on each
#' side, mirroring what find_centers() does for STK, and carry the same names
#' so that substractBackground()'s arithmetic applies unchanged.
centers_from_grid <- function(grid, n, img, bg_gap = 1.6) {
  xs <- seq(grid$x0, by = grid$pitch_x, length.out = n)   # dim 1
  ys <- seq(grid$y0, by = grid$pitch_y, length.out = n)   # dim 2
  g  <- expand.grid(Col = seq_len(n), Row = seq_len(n))   # Col -> x, Row -> y
  peptides <- data.frame(
    names = paste0("Peptide", seq_len(nrow(g))),
    x = xs[g$Col], y = ys[g$Row], Row = g$Row, Col = g$Col,
    stringsAsFactors = FALSE
  )

  xmid <- mean(range(xs)); ymid <- mean(range(ys))
  dx   <- grid$pitch_x * bg_gap; dy <- grid$pitch_y * bg_gap
  clampx <- function(v) pmin(pmax(v, 8), nrow(img) - 8)
  clampy <- function(v) pmin(pmax(v, 8), ncol(img) - 8)
  bg <- data.frame(
    names = c("top_background1", "top_background2",
              "bottom_background1", "bottom_background2",
              "left_background1", "left_background2",
              "right_background1", "right_background2"),
    x = clampx(c(min(xs) - dx, min(xs) - dx, max(xs) + dx, max(xs) + dx,
                 xmid - dx,    xmid + dx,    xmid - dx,    xmid + dx)),
    y = clampy(c(ymid - dy,    ymid + dy,    ymid - dy,    ymid + dy,
                 min(ys) - dy, min(ys) - dy, max(ys) + dy, max(ys) + dy)),
    Row = c(-1, -1, -2, -2, -3, -3, -4, -4),
    Col = c(-1, -1, -2, -2, -3, -3, -4, -4),
    stringsAsFactors = FALSE
  )
  rbind(peptides, bg)
}

# ======================= HELPERS ===========================================

fmt_secs <- function(s) {
  if (!is.finite(s)) return("     n/a")
  if (s < 60) return(sprintf("%7.1fs", s))
  sprintf("%7.1fs (%s)", s,
          sprintf("%d:%02d", floor(s / 60), round(s %% 60)))
}

dataset_label <- function(folder_name) {
  lbl <- sub("(?i)_data_?set$", "", folder_name, perl = TRUE)
  if (!nzchar(lbl)) folder_name else lbl
}

#' PamGene's layout/annotation files carry a trailing tab, so read.table()
#' chokes on them. read_tsv handles it.
read_pam_table <- function(path) {
  as.data.frame(suppressWarnings(
    readr::read_tsv(path, show_col_types = FALSE, progress = FALSE)))
}

#' The package indexes the grid transposed relative to the layout
#' file. Swap Row and Col on the LAYOUT so the merge lands correctly, and drop
#' the off-grid #REF rows: those carry negative Row/Col that collide with the
#' background points' keys, and dropping them is also what leaves STK with
#' exactly the 160 columns substractBackground() insists on.
transpose_layout <- function(layout) {
  keep <- layout$Row > 0 & layout$Col > 0
  data.frame(Row = layout$Col[keep], Col = layout$Row[keep],
             ID  = as.character(layout$ID[keep]), stringsAsFactors = FALSE)
}

#' PTK if the layout has a `Tyr` column, STK if it has `Ser` and `Thr`.
#' The folder name is not trusted.
array_type_from_layout <- function(layout) {
  if ("Tyr" %in% names(layout)) return("PTK")
  if (all(c("Ser", "Thr") %in% names(layout))) return("STK")
  stop("Cannot tell PTK from STK: the Array Layout has neither a 'Tyr' nor a ",
       "'Ser'+'Thr' column.", call. = FALSE)
}

#' Sample annotation -> Barcode / Well / SampleName / Group / Role, in pure R.
#' See the header for the two naming conventions handled.
read_sample_annotation <- function(path, controls_regex) {
  ann <- read_pam_table(path)
  needed <- c("Barcode", "Row", "Sample name")
  missing <- setdiff(needed, names(ann))
  if (length(missing)) {
    stop(basename(path), " is missing column(s): ",
         paste(missing, collapse = ", "), call. = FALSE)
  }
  nm <- trimws(as.character(ann[["Sample name"]]))
  prefixed <- grepl("^[ct][0-9]+_", nm)

  group <- character(length(nm))
  role  <- character(length(nm))
  # "c1_HPC_CTL_1_1" -> HPC_CTL / control ; "t1_STR_Exer_2_1" -> STR_Exer / test
  group[prefixed] <- sub("^[ct][0-9]+_(.*)_[0-9]+_[0-9]+$", "\\1", nm[prefixed])
  role[prefixed]  <- ifelse(substr(nm[prefixed], 1, 1) == "c", "control", "test")
  # "mock 2.1" -> mock ; "pSHDAg 2.1" -> pSHDAg
  bare <- trimws(sub("[ ._-]*[0-9]+([._][0-9]+)*$", "", nm[!prefixed]))
  group[!prefixed] <- bare
  role[!prefixed]  <- ifelse(grepl(controls_regex, bare, perl = TRUE),
                             "control", "test")

  out <- data.frame(
    Barcode    = as.character(ann$Barcode),
    Well       = paste0("W", as.integer(ann$Row)),
    SampleName = gsub("[ ]+", "_", nm),
    Group      = group,
    Role       = role,
    stringsAsFactors = FALSE
  )
  out <- out[nzchar(out$Group), , drop = FALSE]

  mixed <- out %>% distinct(Group, Role) %>% count(Group) %>% filter(n > 1)
  if (nrow(mixed)) {
    stop("Construct(s) annotated as BOTH control and test in ", basename(path),
         ": ", paste(mixed$Group, collapse = ", "), call. = FALSE)
  }
  if (!any(out$Role == "control") || !any(out$Role == "test")) {
    stop("Need at least one control and one test construct in ", basename(path),
         "; got roles: ", paste(unique(out$Role), collapse = ", "),
         ". Adjust --controls.", call. = FALSE)
  }
  out
}

#' Every test x control pair, in a stable order.
build_comparisons <- function(meta) {
  roles    <- meta %>% distinct(Group, Role)
  controls <- sort(roles$Group[roles$Role == "control"])
  tests    <- sort(roles$Group[roles$Role == "test"])
  out <- list()
  for (ctl in controls) for (tst in tests) {
    out[[length(out) + 1L]] <- list(name = paste0(tst, "_vs_", ctl),
                                    test = tst, control = ctl)
  }
  out
}

#' The registration target and the image the grid is detected on.
#' Deterministic: the last cycle at the longest exposure, on the lowest
#' barcode's first well -- i.e. the brightest, most fully developed image of
#' the run, which is what a registration target should be. The vignette simply
#' takes the first file in the folder; that is arbitrary and can pick a 10 ms
#' image with almost no signal.
pick_reference_image <- function(tif_files) {
  a <- parse_image_names(tif_files)
  ord <- order(a$Barcode, a$Well, -a$Cycle, -a$ExposureTime)
  best_bc <- a$Barcode[ord][1]; best_w <- a$Well[ord][1]
  sel <- a$Barcode == best_bc & a$Well == best_w
  a2 <- a[sel, , drop = FALSE]
  a2$file[order(-a2$Cycle, -a2$ExposureTime)][1]
}

#' <chipID>_<well>_F1_T<exposure>_P<cycle>_I<n>_A<n>.tif
parse_image_names <- function(files) {
  b <- basename(files)
  data.frame(
    file         = files,
    Barcode      = sub("^([^_]+)_.*$", "\\1", b),
    Well         = sub("^[^_]+_(W[0-9]+)_.*$", "\\1", b),
    ExposureTime = as.numeric(sub("^.*_T([0-9]+)_.*$", "\\1", b)),
    Cycle        = as.numeric(sub("^.*_P([0-9]+)_.*$", "\\1", b)),
    stringsAsFactors = FALSE
  )
}

#' substractBackground()'s arithmetic, without its hard 160-column check.
#' For STK (160 columns) the package function is called unchanged; for PTK it
#' cannot be, so the identical operation is done here.
subtract_background <- function(raw) {
  pheno <- c("chipID", "well", "exposureTime", "cycle", "ZpFactor")
  n_controls <- c("right_background1", "right_background2",
                  "left_background1", "left_background2",
                  "top_background1", "top_background2",
                  "bottom_background1", "bottom_background2")
  sig <- raw[, -which(names(raw) %in% pheno), drop = FALSE]
  missing <- setdiff(n_controls, names(sig))
  if (length(missing)) {
    stop("Background points missing from the signal table: ",
         paste(missing, collapse = ", "), call. = FALSE)
  }
  if (ncol(sig) == 160) {
    # the published path, STK
    return(pamgeneAnalyzeR::substractBackground(rawData = raw))
  }
  n <- apply(sig, 1, function(x) x - mean(as.numeric(x[n_controls])))
  data.frame(raw[, which(names(raw) %in% pheno), drop = FALSE], t(n),
             check.names = FALSE)
}

# ======================= STAGE 1: pamgeneAnalyzeR ==========================

#' Quantify every image of one array with pamgeneAnalyzeR.
#'
#' Returns the background-subtracted wide signal table plus everything the
#' manifest needs to say what was actually done.
run_image_analysis <- function(array_dir, chip_type, layout, out_dir) {
  tif_dir <- file.path(array_dir, "ImageResults")
  tifs <- sort(list.files(tif_dir, pattern = "\\.tif$", full.names = TRUE))
  if (!length(tifs)) {
    stop("No TIFFs under ", tif_dir, call. = FALSE)
  }
  if (!is.null(config$limit_images)) {
    tifs <- head(tifs, as.integer(config$limit_images))
  }
  sig_dir <- file.path(out_dir, "signal")
  dir.create(sig_dir, recursive = TRUE, showWarnings = FALSE)

  n_grid <- if (identical(chip_type, "PTK")) 14L else 12L
  ref_file <- if (isTRUE(config$unmodified)) tifs[1] else pick_reference_image(tifs)
  message(sprintf("  reference image : %s", basename(ref_file)))
  target <- read_target_tiff(ref_file)

  ## --- spot centres ------------------------------------------------------
  use_published <- identical(config$grid_mode, "published") &&
                   identical(chip_type, "STK")
  if (use_published) {
    # the package's own function, unmodified, on a class-stamped matrix; the
    # source-rewriting shim is only a fallback
    centers <- tryCatch(pamgeneAnalyzeR::find_centers(target),
                        error = function(e) find_centers_compat(target))
    grid_note <- "pamgeneAnalyzeR::find_centers() (published, 12x12)"
    grid_detail <- NA_character_
  } else {
    g <- locate_grid_from_profiles(target, n_grid)
    centers <- centers_from_grid(g, n_grid, target)
    grid_note <- sprintf(
      "locate_grid_from_profiles() -- EXTENSION, not pamgeneAnalyzeR (%dx%d)",
      n_grid, n_grid)
    grid_detail <- sprintf("pitch (%.2f, %.2f) px, origin (%.1f, %.1f) px",
                           g$pitch_x, g$pitch_y, g$x0, g$y0)
  }
  message(sprintf("  spot grid       : %s", grid_note))
  if (!is.na(grid_detail)) message(sprintf("                    %s", grid_detail))

  ## --- register + extract ------------------------------------------------
  layout_t <- if (isTRUE(config$unmodified)) {
    # exactly what the vignette passes: the layout as read, untransposed
    layout[layout$Row > 0 & layout$Col > 0, c("Row", "Col", "ID"), drop = FALSE]
  } else {
    transpose_layout(layout)              # NOTE.txt, property 1
  }
  n_done <- 0L
  if (config$reuse_signal) {
    want <- sub("\\.tif$", ".txt", basename(tifs))
    have <- file.exists(file.path(sig_dir, want))
    n_done <- sum(have)
    tifs <- tifs[!have]
  }
  if (length(tifs)) {
    stage_dir <- file.path(out_dir, ".stage_tifs")
    unlink(stage_dir, recursive = TRUE)
    dir.create(stage_dir, recursive = TRUE, showWarnings = FALSE)
    # registerAllImages() takes a DIRECTORY, so when a subset is to be run the
    # subset is symlinked into a staging folder rather than copied.
    ok <- file.symlink(normalizePath(tifs), file.path(stage_dir, basename(tifs)))
    if (!all(ok)) file.copy(tifs, file.path(stage_dir, basename(tifs)))

    pamgeneAnalyzeR::registerAllImages(
      inDirectory  = stage_dir,
      outDirectory = sig_dir,
      target       = target,
      nLevels      = config$n_levels,
      maxIterations = config$max_iterations,
      useBlockPercentage = config$use_block_percentage,
      centers.coords = centers,
      radius         = config$radius,
      parallel.sig.extract  = FALSE,
      pamgene.layout.file   = layout_t,
      parallel.registration = PARALLEL_REG
    )
    unlink(stage_dir, recursive = TRUE)
  }
  n_images <- length(list.files(sig_dir, pattern = "\\.txt$"))
  message(sprintf("  quantified      : %d image(s)%s", n_images,
                  if (n_done) sprintf(" (%d reused)", n_done) else ""))

  ## --- merge, Z'-factor, background subtraction --------------------------
  raw <- pamgeneAnalyzeR::mergeExperiments(path = sig_dir, value = "median")

  has_controls <- all(paste0("control_", c("left", "right"), rep(1:4, each = 2))
                      %in% names(raw)) ||
                  all(c("control_left1", "control_right1") %in% names(raw))
  zp_note <- NA_character_
  if (has_controls) {
    raw <- pamgeneAnalyzeR::ZpFactor_forAll(rawData = raw)
    zp_note <- sprintf("median Z'-factor %.3f (range %.3f .. %.3f)",
                       stats::median(raw$ZpFactor, na.rm = TRUE),
                       min(raw$ZpFactor, na.rm = TRUE),
                       max(raw$ZpFactor, na.rm = TRUE))
  } else {
    zp_note <- paste("not computed: pamgeneAnalyzeR places its 8 positive",
                     "control points by the STK control-spot geometry, which",
                     "does not exist on this array")
  }
  message(sprintf("  Z'-factor       : %s", zp_note))

  norm <- subtract_background(raw)

  readr::write_tsv(raw,  file.path(out_dir, "raw_signal.tsv"))
  readr::write_tsv(norm, file.path(out_dir, "normalised_signal.tsv"))

  list(signal = norm, n_images = n_images, n_reused = n_done,
       reference = basename(ref_file), grid_note = grid_note,
       grid_detail = grid_detail, zp_note = zp_note,
       n_spots = sum(grepl("^Peptide", centers$names)))
}

# ======================= THE BRIDGE ========================================

#' pamgeneAnalyzeR's wide table -> the tidy frame KRSA's functions expect.
#'
#' Signals are multiplied by --signal-scale here; see "THE BRIDGE INTO KRSA"
#' in the header for why that is not optional.
to_krsa_frame <- function(signal, meta) {
  pheno <- c("chipID", "well", "exposureTime", "cycle", "ZpFactor")
  pep_cols <- setdiff(names(signal), pheno)
  # the control and background points are not peptides
  pep_cols <- pep_cols[!grepl("^(control_|left_background|right_background|top_background|bottom_background|Peptide[0-9]+$|NA$)",
                              pep_cols)]
  pep_cols <- pep_cols[!grepl("^#REF", pep_cols)]

  long <- signal %>%
    select(all_of(c("chipID", "well", "exposureTime", "cycle")), all_of(pep_cols)) %>%
    tidyr::pivot_longer(all_of(pep_cols), names_to = "Peptide", values_to = "Signal") %>%
    mutate(
      Barcode      = as.character(.data$chipID),
      Well         = as.character(.data$well),
      ExposureTime = as.numeric(sub("^T", "", as.character(.data$exposureTime))),
      Cycle        = as.numeric(sub("^P", "", as.character(.data$cycle))),
      Signal       = .data$Signal * config$signal_scale
    ) %>%
    inner_join(meta, by = c("Barcode", "Well"))

  if (!nrow(long)) {
    stop("No (Barcode, Well) of the images matched the sample annotation.",
         call. = FALSE)
  }
  # a peptide printed on several spots is averaged, as BioNavigator's crosstab
  # effectively hands KRSA
  long %>%
    group_by(SampleName, Peptide, ExposureTime, Cycle, Barcode, Group) %>%
    summarise(Signal = mean(.data$Signal, na.rm = TRUE), .groups = "drop")
}

# ======================= STAGE 2: KRSA =====================================

analyse_comparison <- function(comp, data_modeled, data_pw_200, refs, tables_dir) {
  groups <- c(comp$test, comp$control)      # krsa_group_diff: (case, control)
  message(sprintf("  -> %s  (LFC = %s - %s)", comp$name, groups[1], groups[2]))

  pep_qc <- krsa_quick_filter(
    data = data_pw_200, data2 = data_modeled$scaled,
    signal_threshold = config$signal_threshold,
    r2_threshold = config$r2_threshold, groups = groups)
  message(sprintf("     %d peptides passed QC", length(pep_qc)))
  if (length(pep_qc) < 3) {
    return(list(comparison = comp$name, qc_peptides = length(pep_qc),
                hits = 0L, kinase_hits = character(0), status = "skipped: QC"))
  }

  diff_df <- krsa_group_diff(data_modeled$scaled, groups, pep_qc, byChip = TRUE)
  readr::write_delim(diff_df, file.path(tables_dir, paste0(comp$name, "_LFC.txt")),
                     delim = "\t")

  sig_across <- krsa_get_diff(diff_df, totalMeanLFC, config$lfc_cutoffs)
  sig_within <- krsa_get_diff_byChip(diff_df, LFC, config$lfc_cutoffs)

  primary <- as.character(config$lfc_cutoffs[1])
  hits <- sig_across[[primary]]
  message(sprintf("     %d hit peptides at LFC >= %s", length(hits), primary))
  if (length(hits) < 3) {
    return(list(comparison = comp$name, qc_peptides = length(pep_qc),
                hits = length(hits), kinase_hits = character(0),
                status = "skipped: too few hit peptides"))
  }

  fin <- krsa(hits, return_count = TRUE, seed = config$seed,
              itr = config$iterations, map_file = refs$map, cov_file = refs$cov)
  readr::write_delim(fin$KRSA_Table,
    file.path(tables_dir, paste0(comp$name, "_KRSA_Zscores_primary.txt")),
    delim = "\t")

  run_multi_krsa <- function(sets) {
    sets <- sets[vapply(sets, length, integer(1)) > 0]
    if (!length(sets)) return(NULL)
    purrr::imap(sets, function(peps, nm) {
      krsa(peps, itr = config$iterations, seed = config$seed,
           map_file = refs$map, cov_file = refs$cov) %>% mutate(method = nm)
    }) %>% bind_rows()
  }

  # across-chip peptide sets over every LFC cutoff -> AvgZ. This is the table
  # the family comparisons read.
  z_across <- run_multi_krsa(setNames(sig_across, paste0("meanLFC.", names(sig_across))))
  avg_across <- z_across %>% group_by(Kinase) %>% mutate(AvgZ = mean(Z)) %>% ungroup()
  readr::write_delim(avg_across,
    file.path(tables_dir, paste0(comp$name, "_KRSA_acrossChip.txt")), delim = "\t")

  within_sets <- purrr::imap(sig_within, function(sets, bc) {
    setNames(sets, paste0(bc, ".", names(sets)))
  }) %>% purrr::flatten()
  z_within <- run_multi_krsa(within_sets)
  avg_within <- z_within %>% group_by(Kinase) %>% mutate(AvgZ = mean(Z)) %>% ungroup()
  readr::write_delim(avg_within,
    file.path(tables_dir, paste0(comp$name, "_KRSA_withinChip.txt")), delim = "\t")

  kinase_hits <- krsa_top_hits(avg_within, config$z_hit_threshold)
  readr::write_lines(kinase_hits,
                     file.path(tables_dir, paste0(comp$name, "_top_kinases.txt")))
  message(sprintf("     %d kinase hits at |Z| >= %s", length(kinase_hits),
                  config$z_hit_threshold))

  list(comparison = comp$name, qc_peptides = length(pep_qc), hits = length(hits),
       kinase_hits = kinase_hits, status = "ok")
}

run_krsa_stage <- function(krsa_df, chip_type, meta, out_dir) {
  refs <- chip_refs[[chip_type]]
  tables_dir <- file.path(out_dir, "tables")
  dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)

  # sat_qc = FALSE: pamgeneAnalyzeR produces no saturation measure, so there is
  # nothing to filter on. See "THE BRIDGE INTO KRSA".
  data <- krsa_qc_steps(krsa_df, sat_qc = FALSE)

  data_pw_200  <- krsa_extractEndPointMaxExp(data, chip_type)
  data_pw      <- krsa_extractEndPoint(data, chip_type)
  data_modeled <- krsa_scaleModel(data_pw, unique(data_pw$Peptide))

  ppPassAll <- krsa_filter_lowPeps(data_pw_200, config$signal_threshold)
  ppPassR2  <- krsa_filter_nonLinear(
    filter(data_modeled$scaled, Peptide %in% ppPassAll), config$r2_threshold)
  new_pep <- krsa_filter_ref_pep(ppPassR2)
  readr::write_lines(new_pep, file.path(tables_dir, "global_qc_passed_peptides.txt"))
  n_total <- n_distinct(data_pw$Peptide)
  message(sprintf("  %d of %d peptides passed global QC", length(new_pep), n_total))

  comparisons <- build_comparisons(meta)
  message(sprintf("  %d comparison(s): %s", length(comparisons),
                  paste(vapply(comparisons, `[[`, character(1), "name"), collapse = ", ")))

  results <- lapply(comparisons, function(comp) {
    tryCatch(analyse_comparison(comp, data_modeled, data_pw_200, refs, tables_dir),
      error = function(e) {
        message("     [ERROR] ", comp$name, ": ", conditionMessage(e))
        list(comparison = comp$name, qc_peptides = NA_integer_, hits = NA_integer_,
             kinase_hits = character(0),
             status = paste("failed:", conditionMessage(e)))
      })
  })

  list(peptides_total = n_total, peptides_qc = length(new_pep),
       map = refs$map_name, cov = refs$cov_name, comparisons = results)
}

# ======================= TIMING ============================================

#' Run `expr`, returning its value together with wall-clock and CPU seconds.
#' CPU is the R process's own user+system time and therefore does NOT include
#' the parallel workers' CPU -- with --cores > 1 the CPU figure for stage 1 is
#' a lower bound. The wall clock is the number to quote.
timed <- function(expr) {
  p0 <- proc.time(); w0 <- Sys.time()
  value <- force(expr)
  w1 <- Sys.time(); p1 <- proc.time()
  list(value = value,
       wall = as.numeric(difftime(w1, w0, units = "secs")),
       cpu  = as.numeric((p1 - p0)[["user.self"]] + (p1 - p0)[["sys.self"]]))
}

TIMINGS <- list()
record_timing <- function(dataset, array, stage, wall, cpu, n_images = NA_integer_,
                          note = "") {
  TIMINGS[[length(TIMINGS) + 1L]] <<- data.frame(
    dataset = dataset, array = array, stage = stage,
    wall_seconds = round(wall, 3), cpu_seconds = round(cpu, 3),
    n_images = n_images, note = note, stringsAsFactors = FALSE)
}

# ======================= DISCOVERY =========================================

#' An array folder is one that holds an ImageResults folder, an Array Layout
#' and a Sample Annotation.
discover_arrays <- function(dataset_dir) {
  subdirs <- list.dirs(dataset_dir, recursive = FALSE, full.names = TRUE)
  out <- list()
  for (d in subdirs) {
    if (!dir.exists(file.path(d, "ImageResults"))) next
    lay <- list.files(d, pattern = "Array Layout\\.txt$", full.names = TRUE)
    ann <- list.files(d, pattern = "Sample Annotation\\.txt$", full.names = TRUE)
    if (!length(lay) || !length(ann)) {
      message("  [skip] ", basename(d),
              ": no Array Layout / Sample Annotation next to ImageResults")
      next
    }
    out[[length(out) + 1L]] <- list(dir = d, layout = lay[1], annotation = ann[1])
  }
  out
}

discover_datasets <- function(data_dir) {
  if (!dir.exists(data_dir)) {
    stop("Data folder does not exist: ", data_dir, call. = FALSE)
  }
  subdirs <- list.dirs(data_dir, recursive = FALSE, full.names = TRUE)
  ds <- list()
  for (d in subdirs) {
    arrays <- discover_arrays(d)
    if (!length(arrays)) next
    ds[[length(ds) + 1L]] <- list(label = dataset_label(basename(d)),
                                  dir = d, arrays = arrays)
  }
  if (!length(ds)) {
    stop("No dataset with raw images found under ", data_dir, call. = FALSE)
  }
  ds
}

select_datasets <- function(datasets, wanted) {
  if (is.null(wanted) || !length(wanted)) return(datasets)
  keep <- vapply(datasets, function(ds) {
    any(vapply(wanted, function(w) {
      grepl(w, ds$label, ignore.case = TRUE) ||
        grepl(w, basename(ds$dir), ignore.case = TRUE)
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

# ======================= PER-ARRAY DRIVER ==================================

analyse_array <- function(arr, dataset_label_, out_root) {
  layout <- read_pam_table(arr$layout)
  chip_type <- array_type_from_layout(layout)
  if (!chip_type %in% config$arrays) {
    message(sprintf("\n[skip] %s (%s): not in --arrays %s", chip_type,
                    basename(arr$dir), paste(config$arrays, collapse = ",")))
    return(NULL)
  }
  out_dir <- file.path(out_root, chip_type)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  message("\n-----------------------------------------------------")
  message(sprintf("Array: %s   (%s)", chip_type, basename(arr$dir)))
  message(sprintf("  layout     : %s", basename(arr$layout)))
  message(sprintf("  annotation : %s", basename(arr$annotation)))

  meta <- read_sample_annotation(arr$annotation, config$controls)
  message(sprintf("  controls={%s}  tests={%s}",
                  paste(sort(unique(meta$Group[meta$Role == "control"])), collapse = ", "),
                  paste(sort(unique(meta$Group[meta$Role == "test"])), collapse = ", ")))

  res <- list(array = chip_type, dir = arr$dir, chip_type = chip_type)

  ## --- stage 1 -----------------------------------------------------------
  if (config$run_image_analysis) {
    message("  [stage 1] pamgeneAnalyzeR ...")
    t1 <- timed(run_image_analysis(arr$dir, chip_type, layout, out_dir))
    ia <- t1$value
    res <- c(res, ia[c("n_images", "n_reused", "reference", "grid_note",
                       "grid_detail", "zp_note", "n_spots")])
    res$signal <- ia$signal
    res$stage1_wall <- t1$wall; res$stage1_cpu <- t1$cpu
    record_timing(dataset_label_, chip_type, "pamgeneAnalyzeR", t1$wall, t1$cpu,
                  ia$n_images,
                  if (ia$n_reused > 0) sprintf("NOT TIMED: %d image(s) reused", ia$n_reused) else "")
    message(sprintf("  [stage 1] done in %s wall / %s cpu",
                    fmt_secs(t1$wall), fmt_secs(t1$cpu)))
  } else {
    f <- file.path(out_dir, "normalised_signal.tsv")
    if (!file.exists(f)) {
      stop("--no-image-analysis, but ", f, " does not exist.", call. = FALSE)
    }
    res$signal <- as.data.frame(readr::read_tsv(f, show_col_types = FALSE,
                                                progress = FALSE))
    res$stage1_wall <- NA_real_; res$stage1_cpu <- NA_real_
    message("  [stage 1] skipped (--no-image-analysis); signal read from disk")
  }

  ## --- stage 2 -----------------------------------------------------------
  if (config$run_krsa) {
    message("  [stage 2] KRSA ...")
    t2 <- timed({
      krsa_df <- to_krsa_frame(res$signal, meta)
      run_krsa_stage(krsa_df, chip_type, meta, out_dir)
    })
    res <- c(res, t2$value)
    res$stage2_wall <- t2$wall; res$stage2_cpu <- t2$cpu
    record_timing(dataset_label_, chip_type, "KRSA", t2$wall, t2$cpu)
    message(sprintf("  [stage 2] done in %s wall / %s cpu",
                    fmt_secs(t2$wall), fmt_secs(t2$cpu)))
  } else {
    res$stage2_wall <- NA_real_; res$stage2_cpu <- NA_real_
    res$comparisons <- list()
    message("  [stage 2] skipped (--no-krsa)")
  }

  res$signal <- NULL          # do not keep whole tables alive across arrays
  res
}

# ======================= PER-DATASET DRIVER ================================

analyse_dataset <- function(ds, out_root) {
  message("\n=====================================================")
  message(sprintf("Dataset: %s", ds$label))
  message(sprintf("  input : %s", ds$dir))
  message(sprintf("  output: %s", out_root))
  message(sprintf("  arrays: %d", length(ds$arrays)))
  message("=====================================================")
  dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

  w0 <- Sys.time()
  arrays <- lapply(ds$arrays, function(arr) {
    tryCatch(analyse_array(arr, ds$label, out_root),
      error = function(e) {
        message("  [ERROR] array ", basename(arr$dir), " failed: ",
                conditionMessage(e))
        list(array = NA_character_, dir = arr$dir, error = conditionMessage(e),
             stage1_wall = NA_real_, stage2_wall = NA_real_,
             stage1_cpu = NA_real_, stage2_cpu = NA_real_, comparisons = list())
      })
  })
  arrays <- Filter(Negate(is.null), arrays)
  if (!length(arrays)) {
    stop("No array of ", ds$label, " matched --arrays ",
         paste(config$arrays, collapse = ","), call. = FALSE)
  }
  wall_total <- as.numeric(difftime(Sys.time(), w0, units = "secs"))

  s1w <- sum(vapply(arrays, function(a) a$stage1_wall %||% NA_real_, numeric(1)), na.rm = TRUE)
  s2w <- sum(vapply(arrays, function(a) a$stage2_wall %||% NA_real_, numeric(1)), na.rm = TRUE)
  s1c <- sum(vapply(arrays, function(a) a$stage1_cpu  %||% NA_real_, numeric(1)), na.rm = TRUE)
  s2c <- sum(vapply(arrays, function(a) a$stage2_cpu  %||% NA_real_, numeric(1)), na.rm = TRUE)
  n_img <- sum(vapply(arrays, function(a) as.numeric(a$n_images %||% NA_real_), numeric(1)), na.rm = TRUE)

  record_timing(ds$label, "ALL", "pamgeneAnalyzeR (dataset)", s1w, s1c, n_img)
  record_timing(ds$label, "ALL", "KRSA (dataset)",            s2w, s2c)
  record_timing(ds$label, "ALL", "TOTAL (dataset)", wall_total, s1c + s2c, n_img)

  write_manifest(ds, out_root, arrays, wall_total, s1w, s2w, s1c, s2c)
  list(label = ds$label, dir = ds$dir, out = out_root, arrays = arrays,
       wall_total = wall_total, stage1_wall = s1w, stage2_wall = s2w,
       stage1_cpu = s1c, stage2_cpu = s2c, n_images = n_img)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# ======================= MANIFEST ==========================================

write_manifest <- function(ds, out_root, arrays, wall_total, s1w, s2w, s1c, s2c) {
  lines <- c(
    "pamgeneAnalyzeR + KRSA, end to end on raw TIFFs -- run manifest",
    "===============================================================",
    sprintf("written           : %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    sprintf("script            : %s",
            rel_to_repo(file.path(SCRIPT_DIR, "run_pamanalyzeR_and_KRSA.R"))),
    sprintf("dataset           : %s", ds$label),
    sprintf("input folder      : %s", rel_to_repo(ds$dir)),
    sprintf("output folder     : %s", rel_to_repo(out_root)),
    "",
    sprintf("pamgeneAnalyzeR   : %s  (%s)", PAMGENEANALYZER_VERSION, PAMGENEANALYZER_SOURCE),
    sprintf("KRSA              : %s  (%s)", KRSA_VERSION, KRSA_SOURCE),
    sprintf("R                 : %s", R.version.string),
    sprintf("platform          : %s", R.version$platform),
    sprintf("registration cores: %s", CORES_NOTE),
    "",
    "TIMING (wall clock is the number to quote; CPU excludes parallel workers)",
    sprintf("  stage 1  pamgeneAnalyzeR : %s wall  / %s cpu", fmt_secs(s1w), fmt_secs(s1c)),
    sprintf("  stage 2  KRSA            : %s wall  / %s cpu", fmt_secs(s2w), fmt_secs(s2c)),
    sprintf("  TOTAL                    : %s wall", fmt_secs(wall_total)),
    ""
  )
  if (isTRUE(config$reuse_signal)) {
    lines <- c(lines,
      "  *** --reuse-signal WAS USED. Stage 1 is NOT A VALID TIMING. ***", "")
  }
  if (!is.null(config$limit_images)) {
    lines <- c(lines, sprintf(
      "  *** --limit-images %s WAS USED. This is a smoke test, not a timing. ***",
      config$limit_images), "")
  }

  lines <- c(lines,
    "settings (identical to run_krsa_on_pykinaxe_image_analysis.R, so the two",
    "runs are comparable -- do not change one side only):",
    sprintf("  spot radius      : %s px", config$radius),
    sprintf("  arrays           : %s", paste(config$arrays, collapse = ", ")),
    sprintf("  layout           : %s",
            if (isTRUE(config$unmodified)) "AS READ, not transposed (--unmodified)"
            else "transposed before the merge (see FINDING 1)"),
    sprintf("  reference image  : %s",
            if (isTRUE(config$unmodified)) "first file in the folder (vignette's choice)"
            else "last cycle at the longest exposure"),
    sprintf("  grid mode        : %s", config$grid_mode),
    sprintf("  signal scale     : %s  ([0,1] float -> counts)", config$signal_scale),
    sprintf("  saturation filter: OFF (pamgeneAnalyzeR produces no saturation measure)"),
    sprintf("  signal_threshold : %s", config$signal_threshold),
    sprintf("  r2_threshold     : %s", config$r2_threshold),
    sprintf("  lfc_cutoffs      : %s", paste(config$lfc_cutoffs, collapse = ", ")),
    sprintf("  z_hit_threshold  : %s", config$z_hit_threshold),
    sprintf("  iterations       : %s", config$iterations),
    sprintf("  seed             : %s", config$seed),
    sprintf("  control regex    : %s", config$controls),
    "",
    "known pamgeneAnalyzeR issues worked around (see the script header):",
    "  [1] find_centers() returns Row/Col transposed vs the PamChip Array",
    "      Layout; the layout is transposed before merging (r 0.43 -> 0.9997)",
    "  [2] find_centers() (12x12) and substractBackground() (160 columns) are",
    "      STK-only; every other function is grid-agnostic. PTK gets its grid",
    "      from the extension, but registration/extraction stay the package's",
    "  [3] find_centers()'s class() guard errors on R >= 4.0; shimmed",
    "")

  faithful <- identical(sort(config$arrays), "STK") &&
              identical(config$grid_mode, "published")
  lines <- c(lines, if (isTRUE(config$unmodified)) c(
    "  >> UNMODIFIED pamgeneAnalyzeR RUN, exactly as its own vignette runs it:",
    "     STK only, layout as read, reference image = first file in the folder.",
    "     This is an IMAGE-ANALYSIS TIMING ONLY. The peptide labels it produces",
    "     are wrong (FINDING 1), and KRSA was not run on them. The KRSA timing",
    "     comes from run_krsa_on_bionavigator.R, on KRSA's own native input.",
    "") else if (faithful) c(
    "  >> PACKAGE-FAITHFUL RUN: STK only, on pamgeneAnalyzeR's own published",
    "     code path (find_centers(), substractBackground(), ZpFactor_forAll()).",
    "     No grid extension from this script is involved in these numbers.",
    "") else c(
    "  >> NOT a package-faithful run: it includes PTK, which pamgeneAnalyzeR",
    "     does not support -- those arrays use the grid extension of this script. Use",
    "     --arrays STK for the figure that rests on the package alone.",
    ""))

  for (ar in arrays) {
    if (!is.null(ar$error)) {
      lines <- c(lines, sprintf("[%s]  ERROR: %s", basename(ar$dir), ar$error), "")
      next
    }
    lines <- c(lines,
      sprintf("[%s]", ar$array),
      sprintf("  folder          : %s", basename(ar$dir)),
      sprintf("  images          : %s", ar$n_images %||% NA),
      sprintf("  reference image : %s", ar$reference %||% NA),
      sprintf("  spot grid       : %s", ar$grid_note %||% NA),
      if (!is.null(ar$grid_detail) && !is.na(ar$grid_detail))
        sprintf("                    %s", ar$grid_detail) else NULL,
      sprintf("  spots addressed : %s", ar$n_spots %||% NA),
      sprintf("  Z'-factor       : %s", ar$zp_note %||% NA),
      sprintf("  mapping table   : %s", ar$map %||% NA),
      sprintf("  coverage table  : %s", ar$cov %||% NA),
      sprintf("  peptides passing global QC: %s of %s",
              ar$peptides_qc %||% NA, ar$peptides_total %||% NA),
      sprintf("  stage 1 / stage 2: %s / %s wall",
              fmt_secs(ar$stage1_wall), fmt_secs(ar$stage2_wall)))
    for (cr in ar$comparisons) {
      lines <- c(lines, sprintf(
        "  %-28s QC peptides %-4s hit peptides %-4s kinase hits %-3s  %s",
        cr$comparison, cr$qc_peptides, cr$hits, length(cr$kinase_hits),
        if (length(cr$kinase_hits))
          paste0("(", paste(cr$kinase_hits, collapse = ", "), ")") else cr$status))
    }
    lines <- c(lines, "")
  }
  writeLines(lines, file.path(out_root, "run_manifest.txt"))
}

write_timing_report <- function(runs) {
  if (!length(TIMINGS)) return(invisible())
  tbl <- bind_rows(TIMINGS)
  readr::write_csv(tbl, file.path(config$output_dir, "timing.csv"))

  invalid <- isTRUE(config$reuse_signal) || !is.null(config$limit_images)
  lines <- c(
    "pamgeneAnalyzeR + KRSA on raw PamChip TIFFs -- runtime",
    "=====================================================",
    sprintf("written : %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    sprintf("machine : %s, R %s.%s", R.version$platform,
            R.version$major, R.version$minor),
    sprintf("registration: %s", CORES_NOTE),
    sprintf("packages: pamgeneAnalyzeR %s, KRSA %s",
            PAMGENEANALYZER_VERSION, KRSA_VERSION),
    "")
  if (invalid) {
    lines <- c(lines,
      "*** THIS RUN IS NOT A VALID TIMING ***",
      if (isTRUE(config$reuse_signal)) "    --reuse-signal was used" else NULL,
      if (!is.null(config$limit_images))
        sprintf("    --limit-images %s was used", config$limit_images) else NULL,
      "")
  }
  lines <- c(lines,
    "Wall clock is what you waited for. Stage 1 is parallel over the cores",
    "above; stage 2 (krsa() sampling) is single-threaded. The CPU column is",
    "this R process only and so understates stage 1 whenever cores > 1.",
    "",
    sprintf("%-24s %-6s %-28s %12s %12s %8s", "dataset", "array", "stage",
            "wall (s)", "cpu (s)", "images"),
    strrep("-", 94))
  for (i in seq_len(nrow(tbl))) {
    r <- tbl[i, ]
    if (r$stage == "pamgeneAnalyzeR (dataset)") lines <- c(lines, strrep("-", 94))
    lines <- c(lines, sprintf("%-24s %-6s %-28s %12.1f %12.1f %8s",
      r$dataset, r$array, r$stage, r$wall_seconds, r$cpu_seconds,
      ifelse(is.na(r$n_images), "-", as.character(r$n_images))))
    if (nzchar(r$note)) lines <- c(lines, sprintf("%76s%s", "", r$note))
  }

  lines <- c(lines, "", "PER-DATASET TOTALS", strrep("-", 94))
  grand <- 0
  for (r in runs) {
    if (is.null(r$wall_total)) next
    grand <- grand + r$wall_total
    lines <- c(lines, sprintf(
      "  %-22s  stage 1 %s   stage 2 %s   TOTAL %s   (%s images)",
      r$label, fmt_secs(r$stage1_wall), fmt_secs(r$stage2_wall),
      fmt_secs(r$wall_total), r$n_images))
  }
  lines <- c(lines, sprintf("  %-22s  %s", "ALL DATASETS", fmt_secs(grand)), "")
  writeLines(lines, file.path(config$output_dir, "timing_summary.txt"))
  cat("\n", paste(lines, collapse = "\n"), "\n", sep = "")
}

# ======================= DRIVER ============================================

main <- function() {
  data_dir <- normalizePath(config$data_dir, mustWork = FALSE)
  datasets <- select_datasets(discover_datasets(data_dir), config$datasets)
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)

  if (ONLY_LIST) {
    cat("Datasets under ", data_dir, ":\n", sep = "")
    for (ds in datasets) {
      cat(sprintf("  %-24s %s\n", ds$label, ds$dir))
      for (arr in ds$arrays) {
        n <- length(list.files(file.path(arr$dir, "ImageResults"),
                               pattern = "\\.tif$"))
        ct <- tryCatch(array_type_from_layout(read_pam_table(arr$layout)),
                       error = function(e) "??")
        cat(sprintf("      %-4s %5d TIFFs   %s\n", ct, n, basename(arr$dir)))
      }
    }
    return(invisible(NULL))
  }

  message(sprintf(
    "pamgeneAnalyzeR %s + KRSA %s on raw TIFFs -- %d dataset(s): %s",
    PAMGENEANALYZER_VERSION, KRSA_VERSION, length(datasets),
    paste(vapply(datasets, `[[`, character(1), "label"), collapse = ", ")))
  message(sprintf("registration: %s", CORES_NOTE))

  runs <- list()
  for (ds in datasets) {
    out_root <- file.path(config$output_dir, ds$label)
    runs[[length(runs) + 1L]] <- tryCatch(
      analyse_dataset(ds, out_root),
      error = function(e) {
        message("[ERROR] dataset ", ds$label, " failed: ", conditionMessage(e))
        list(label = ds$label, dir = ds$dir, out = out_root, arrays = list(),
             error = conditionMessage(e))
      })
  }

  message("\n=====================================================")
  message("SUMMARY")
  message("=====================================================")
  failed <- 0L
  for (r in runs) {
    if (!is.null(r$error)) {
      message(sprintf("  %-24s FAILED: %s", r$label, r$error)); failed <- failed + 1L; next
    }
    for (ar in r$arrays) {
      if (!is.null(ar$error)) { failed <- failed + 1L; next }
      message(sprintf("  %-24s %-4s  %s images  QC peptides %s/%s",
                      r$label, ar$array, ar$n_images %||% NA,
                      ar$peptides_qc %||% NA, ar$peptides_total %||% NA))
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
  if (failed) message(sprintf("\n%d comparison(s)/array(s) did not complete.", failed))

  write_timing_report(runs)
  message("\nDone.")
  invisible(runs)
}

main()
