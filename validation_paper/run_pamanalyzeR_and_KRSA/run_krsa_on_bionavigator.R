#!/usr/bin/env Rscript
# run_krsa_on_bionavigator.R
#
# Runs KRSA on its native input, a BioNavigator image-analysis export, and
# times it. run_pamanalyzeR_and_KRSA.R has to rescale [0,1] TIFF floats and
# switch KRSA's saturation filter off because pamgeneAnalyzeR produces no
# saturation measure; a BioNavigator export has Median_SigmBg in counts and a
# Signal_Saturation column, so KRSA runs here with krsa_qc_steps() at its
# default sat_qc = TRUE. Together with run_pamanalyzeR_and_KRSA.R --unmodified
# this times the image analysis and the kinase analysis separately, each on the
# input it was designed for.
#
# Input (--bn-dir, default <script dir>/exports_BN, searched recursively):
#   Export-image_analysis_PTK_afterwash_template-Export.csv   PTK, cycle 94
#   Export-image_analysis_STK_template-Export.csv             STK, cycle 124
# The data-set prefix of BioNavigator's columns is stripped (ds3.Median_SigmBg
# -> Median_SigmBg). KRSA uses the end-point cycle only, so the PTK prewash
# export (kinetic cycles 32-92) is not loaded; --include-prewash loads it
# anyway. The sample annotation is PamGene's "* Sample Annotation.txt" next to
# the exports (--annotation-dir).
#
# Output (--output-dir, default <script dir>/output_krsa_on_bn):
#   <PTK|STK>/tables/   same table names as run_pamanalyzeR_and_KRSA.R
#   run_manifest.txt    settings, per-array detail, timings
#   timing.csv, timing_summary.txt
#
# Settings match run_pamanalyzeR_and_KRSA.R (signal >= 5, R^2 >= 0.9, LFC
# cutoffs 0.2/0.3/0.4, |Z| >= 2, 2000 iterations, seed 123, the same mapping and
# coverage tables) except for the saturation filter, which is on here.
# --krsa-scope decides how much of KRSA runs; one krsa() call costs about 3 s at
# 2000 iterations and the runtime is linear in the call count:
#   across   krsa() over the 3 LFC cutoffs, across-chip AvgZ (3 calls per
#            comparison); the counterpart of pyKinaXe's cutoff-averaged Z
#   full     across plus within-chip per barcode plus the primary run, i.e.
#            KRSA's report template (1 + 3 + 3 x barcodes calls); default
#   primary  one krsa() at the first cutoff (1 call)
#
# Usage:
#   Rscript run_krsa_on_bionavigator.R
#   Rscript run_krsa_on_bionavigator.R --krsa-scope across
#   Rscript run_krsa_on_bionavigator.R --help

KRSA_SOURCE <- "https://github.com/CogDisResLab/KRSA"

get_script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", a, value = TRUE)
  if (length(fa) == 1L) return(dirname(normalizePath(sub("^--file=", "", fa))))
  normalizePath(getwd())
}
SCRIPT_DIR <- get_script_dir()
REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."), mustWork = FALSE)

config <- list(
  # The BioNavigator export and the sample annotation ship IN THIS FOLDER, so
  # the KRSA benchmark is self-contained and needs nothing from data/.
  bn_dir = file.path(SCRIPT_DIR, "exports_BN"),
  annotation_dir = file.path(SCRIPT_DIR, "exports_BN"),
  output_dir = file.path(SCRIPT_DIR, "results", "krsa_on_bionavigator"),
  dataset_label = "benchmarking",
  arrays = c("PTK", "STK"),
  include_prewash = FALSE,
  sat_qc = TRUE,               # KRSA's own default; this input supports it
  signal_threshold = 5,
  r2_threshold     = 0.9,
  lfc_cutoffs      = c(0.2, 0.3, 0.4),
  z_hit_threshold  = 2,
  iterations       = 2000,
  seed             = 123,
  krsa_scope       = "full",
  controls         = "^mock$",
  install_missing  = TRUE
)

chip_ref_names <- list(
  PTK = list(cov = "KRSA_coverage_PTK_PamChip_86402_v1",
             map = "KRSA_Mapping_PTK_PamChip_86402_v1"),
  STK = list(cov = "KRSA_coverage_STK_PamChip_87102_v2",
             map = "KRSA_Mapping_STK_PamChip_87102_v1")
)

USAGE <- "
Usage: Rscript run_krsa_on_bionavigator.R [options]

  --bn-dir PATH          folder with the BioNavigator Export csv files
  --annotation-dir PATH  folder holding the run folders with Sample Annotation
  --output-dir PATH      where results go
  --arrays A,B           PTK,STK (default: both)
  --krsa-scope S         across | full | primary (default: full)
                         across  = 3 krsa() calls per comparison (AvgZ; the
                                   counterpart of pyKinaXe's averaged Z)
                         full    = KRSA's own report template (+ within-chip)
                         primary = a single krsa() call per comparison
  --include-prewash      also load the PTK prewash export (KRSA discards it)
  --no-sat-qc            switch KRSA's saturation filter off (not the default)
  --signal-threshold X   default 5
  --r2-threshold X       default 0.9
  --lfc-cutoffs A,B,C    default 0.2,0.3,0.4
  --z-threshold X        default 2
  --iterations N         default 2000
  --seed N               default 123
  --controls REGEX       control constructs (default '^mock$')
  --no-install           fail instead of installing a missing package
  --help                 this text
"

parse_args <- function(argv) {
  tv <- function(i, f) { if (i + 1L > length(argv)) stop("Missing value for ", f, call. = FALSE); argv[i + 1L] }
  nv <- function(t, f) { v <- suppressWarnings(as.numeric(strsplit(t, ",")[[1]]))
                         if (any(is.na(v))) stop("Not numeric: ", f, " ", t, call. = FALSE); v }
  i <- 1L
  while (i <= length(argv)) {
    a <- argv[[i]]; skip <- 2L
    switch(a,
      "--bn-dir"           = config$bn_dir <<- tv(i, a),
      "--annotation-dir"   = config$annotation_dir <<- tv(i, a),
      "--output-dir"       = config$output_dir <<- tv(i, a),
      "--arrays"           = config$arrays <<- toupper(trimws(strsplit(tv(i, a), ",")[[1]])),
      "--krsa-scope"       = config$krsa_scope <<- tolower(tv(i, a)),
      "--signal-threshold" = config$signal_threshold <<- nv(tv(i, a), a)[1],
      "--r2-threshold"     = config$r2_threshold <<- nv(tv(i, a), a)[1],
      "--lfc-cutoffs"      = config$lfc_cutoffs <<- nv(tv(i, a), a),
      "--z-threshold"      = config$z_hit_threshold <<- nv(tv(i, a), a)[1],
      "--iterations"       = config$iterations <<- nv(tv(i, a), a)[1],
      "--seed"             = config$seed <<- nv(tv(i, a), a)[1],
      "--controls"         = config$controls <<- tv(i, a),
      "--include-prewash"  = { config$include_prewash <<- TRUE; skip <- 1L },
      "--no-sat-qc"        = { config$sat_qc <<- FALSE; skip <- 1L },
      "--no-install"       = { config$install_missing <<- FALSE; skip <- 1L },
      "--help"             = { cat(USAGE); quit(save = "no", status = 0) },
      stop("Unknown argument: ", a, "\n", USAGE, call. = FALSE))
    i <- i + skip
  }
  if (!config$krsa_scope %in% c("across", "full", "primary")) {
    stop("--krsa-scope must be across, full or primary", call. = FALSE)
  }
  bad <- setdiff(config$arrays, c("PTK", "STK"))
  if (length(bad)) stop("--arrays takes PTK and/or STK, got: ", paste(bad, collapse = ","), call. = FALSE)
}
parse_args(commandArgs(trailingOnly = TRUE))

ensure_packages <- function() {
  need <- c("dplyr", "tidyr", "readr", "stringr", "purrr", "tibble", "ggplot2")
  miss <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
  if (length(miss)) {
    if (!config$install_missing) stop("Missing: ", paste(miss, collapse = ", "), call. = FALSE)
    utils::install.packages(miss, repos = "https://cloud.r-project.org")
  }
  if (!requireNamespace("KRSA", quietly = TRUE)) {
    if (!config$install_missing) stop("KRSA missing; install from ", KRSA_SOURCE, call. = FALSE)
    if (!requireNamespace("remotes", quietly = TRUE)) {
      utils::install.packages("remotes", repos = "https://cloud.r-project.org")
    }
    remotes::install_github("CogDisResLab/KRSA", upgrade = "never")
  }
}
ensure_packages()
suppressPackageStartupMessages({
  library(KRSA); library(dplyr); library(tidyr); library(readr)
  library(stringr); library(purrr); library(tibble); library(ggplot2)
})
KRSA_VERSION <- as.character(utils::packageVersion("KRSA"))
chip_refs <- lapply(chip_ref_names, function(nm)
  list(cov = get(nm$cov, envir = asNamespace("KRSA")),
       map = get(nm$map, envir = asNamespace("KRSA")),
       cov_name = nm$cov, map_name = nm$map))

fmt_secs <- function(s) if (!is.finite(s)) "     n/a" else if (s < 60) sprintf("%7.1fs", s) else
  sprintf("%7.1fs (%d:%02d)", s, floor(s / 60), round(s %% 60))

timed <- function(expr) {
  p0 <- proc.time(); w0 <- Sys.time(); v <- force(expr)
  w1 <- Sys.time(); p1 <- proc.time()
  list(value = v, wall = as.numeric(difftime(w1, w0, units = "secs")),
       cpu = as.numeric((p1 - p0)[["user.self"]] + (p1 - p0)[["sys.self"]]))
}

#' Strip BioNavigator's data-set prefix: ds0.Barcode -> Barcode,
#' ds3.Median_SigmBg -> Median_SigmBg, ds2..spotCol -> spotCol.
strip_bn_prefix <- function(x) sub("^[a-z]+[0-9]*\\.+", "", x)

read_bn_export <- function(files) {
  purrr::map_dfr(files, function(f) {
    d <- readr::read_csv(f, show_col_types = FALSE, progress = FALSE,
                         name_repair = "minimal")
    names(d) <- strip_bn_prefix(names(d))
    need <- c("Barcode", "Row", "Exposure Time", "Cycle", "ID",
              "Median_SigmBg", "Signal_Saturation")
    missing <- setdiff(need, names(d))
    if (length(missing)) {
      stop(basename(f), " is missing column(s): ", paste(missing, collapse = ", "),
           call. = FALSE)
    }
    d %>% transmute(
      Barcode          = as.character(.data$Barcode),
      Well             = paste0("W", as.integer(.data$Row)),
      ExposureTime     = as.numeric(.data$`Exposure Time`),
      Cycle            = as.numeric(.data$Cycle),
      Peptide          = as.character(.data$ID),
      Signal           = as.numeric(.data$Median_SigmBg),
      SignalSaturation = as.numeric(.data$Signal_Saturation))
  })
}

read_sample_annotation <- function(path, controls_regex) {
  ann <- as.data.frame(suppressWarnings(
    readr::read_tsv(path, show_col_types = FALSE, progress = FALSE)))
  need <- c("Barcode", "Row", "Sample name")
  if (length(setdiff(need, names(ann)))) {
    stop(basename(path), " is missing ", paste(setdiff(need, names(ann)), collapse = ", "),
         call. = FALSE)
  }
  nm <- trimws(as.character(ann[["Sample name"]]))
  pre <- grepl("^[ct][0-9]+_", nm)
  group <- character(length(nm)); role <- character(length(nm))
  group[pre] <- sub("^[ct][0-9]+_(.*)_[0-9]+_[0-9]+$", "\\1", nm[pre])
  role[pre]  <- ifelse(substr(nm[pre], 1, 1) == "c", "control", "test")
  bare <- trimws(sub("[ ._-]*[0-9]+([._][0-9]+)*$", "", nm[!pre]))
  group[!pre] <- bare
  role[!pre]  <- ifelse(grepl(controls_regex, bare, perl = TRUE), "control", "test")
  out <- data.frame(Barcode = as.character(ann$Barcode),
                    Well = paste0("W", as.integer(ann$Row)),
                    SampleName = gsub("[ ]+", "_", nm),
                    Group = group, Role = role, stringsAsFactors = FALSE)
  out[nzchar(out$Group), , drop = FALSE]
}

build_comparisons <- function(meta) {
  r <- meta %>% distinct(Group, Role)
  out <- list()
  for (ctl in sort(r$Group[r$Role == "control"]))
    for (tst in sort(r$Group[r$Role == "test"]))
      out[[length(out) + 1L]] <- list(name = paste0(tst, "_vs_", ctl), test = tst, control = ctl)
  out
}

run_multi_krsa <- function(sets, refs) {
  sets <- sets[vapply(sets, length, integer(1)) > 0]
  if (!length(sets)) return(NULL)
  purrr::imap(sets, function(peps, nm)
    krsa(peps, itr = config$iterations, seed = config$seed,
         map_file = refs$map, cov_file = refs$cov) %>% mutate(method = nm)) %>%
    bind_rows()
}

analyse_comparison <- function(comp, modeled, pw200, refs, tdir) {
  groups <- c(comp$test, comp$control)
  message(sprintf("  -> %s  (LFC = %s - %s)", comp$name, groups[1], groups[2]))
  pep_qc <- krsa_quick_filter(data = pw200, data2 = modeled$scaled,
                              signal_threshold = config$signal_threshold,
                              r2_threshold = config$r2_threshold, groups = groups)
  message(sprintf("     %d peptides passed QC", length(pep_qc)))
  if (length(pep_qc) < 3) {
    return(list(comparison = comp$name, qc_peptides = length(pep_qc), hits = 0L,
                kinase_hits = character(0), krsa_calls = 0L, status = "skipped: QC"))
  }
  diff_df <- krsa_group_diff(modeled$scaled, groups, pep_qc, byChip = TRUE)
  readr::write_delim(diff_df, file.path(tdir, paste0(comp$name, "_LFC.txt")), delim = "\t")

  sig_across <- krsa_get_diff(diff_df, totalMeanLFC, config$lfc_cutoffs)
  primary <- as.character(config$lfc_cutoffs[1])
  hits <- sig_across[[primary]]
  message(sprintf("     %d hit peptides at LFC >= %s", length(hits), primary))
  if (length(hits) < 3) {
    return(list(comparison = comp$name, qc_peptides = length(pep_qc), hits = length(hits),
                kinase_hits = character(0), krsa_calls = 0L,
                status = "skipped: too few hit peptides"))
  }

  calls <- 0L; kinase_hits <- character(0)
  if (config$krsa_scope == "primary") {
    fin <- krsa(hits, return_count = TRUE, seed = config$seed, itr = config$iterations,
                map_file = refs$map, cov_file = refs$cov); calls <- 1L
    readr::write_delim(fin$KRSA_Table,
      file.path(tdir, paste0(comp$name, "_KRSA_Zscores_primary.txt")), delim = "\t")
    kinase_hits <- krsa_top_hits(mutate(fin$KRSA_Table, AvgZ = Z), config$z_hit_threshold)
  } else {
    if (config$krsa_scope == "full") {
      fin <- krsa(hits, return_count = TRUE, seed = config$seed, itr = config$iterations,
                  map_file = refs$map, cov_file = refs$cov); calls <- calls + 1L
      readr::write_delim(fin$KRSA_Table,
        file.path(tdir, paste0(comp$name, "_KRSA_Zscores_primary.txt")), delim = "\t")
    }
    z_across <- run_multi_krsa(setNames(sig_across, paste0("meanLFC.", names(sig_across))), refs)
    calls <- calls + length(sig_across)
    avg_across <- z_across %>% group_by(Kinase) %>% mutate(AvgZ = mean(Z)) %>% ungroup()
    readr::write_delim(avg_across,
      file.path(tdir, paste0(comp$name, "_KRSA_acrossChip.txt")), delim = "\t")
    kinase_hits <- krsa_top_hits(avg_across, config$z_hit_threshold)

    if (config$krsa_scope == "full") {
      sig_within <- krsa_get_diff_byChip(diff_df, LFC, config$lfc_cutoffs)
      within_sets <- purrr::imap(sig_within, function(s, bc)
        setNames(s, paste0(bc, ".", names(s)))) %>% purrr::flatten()
      z_within <- run_multi_krsa(within_sets, refs)
      calls <- calls + length(within_sets)
      avg_within <- z_within %>% group_by(Kinase) %>% mutate(AvgZ = mean(Z)) %>% ungroup()
      readr::write_delim(avg_within,
        file.path(tdir, paste0(comp$name, "_KRSA_withinChip.txt")), delim = "\t")
      kinase_hits <- krsa_top_hits(avg_within, config$z_hit_threshold)
    }
  }
  readr::write_lines(kinase_hits, file.path(tdir, paste0(comp$name, "_top_kinases.txt")))
  message(sprintf("     %d kinase hits at |Z| >= %s  (%d krsa() call(s))",
                  length(kinase_hits), config$z_hit_threshold, calls))
  list(comparison = comp$name, qc_peptides = length(pep_qc), hits = length(hits),
       kinase_hits = kinase_hits, krsa_calls = calls, status = "ok")
}

analyse_array <- function(chip_type) {
  message("\n-----------------------------------------------------")
  message(sprintf("Array: %s", chip_type))
  pat <- if (chip_type == "PTK") {
    if (config$include_prewash) "PTK_(afterwash|prewash)" else "PTK_afterwash"
  } else "STK"
  files <- list.files(config$bn_dir, pattern = paste0(pat, ".*\\.csv$"),
                      full.names = TRUE, recursive = TRUE)
  if (!length(files)) stop("No ", chip_type, " export under ", config$bn_dir, call. = FALSE)
  for (f in files) message(sprintf("  input : %s", basename(f)))

  raw <- read_bn_export(files)
  bc  <- unique(raw$Barcode)

  # Pick the annotation by BARCODE OVERLAP, not by file name: the PamGene files
  # are named after the article number (86412 = PTK, 87102 = STK), not after the
  # array, so matching on "PTK"/"STK" in the path only works while they still
  # sit under their original run folders.
  ann_files <- list.files(config$annotation_dir, pattern = "Sample Annotation\\.txt$",
                          recursive = TRUE, full.names = TRUE)
  if (!length(ann_files)) {
    stop("No '* Sample Annotation.txt' under ", config$annotation_dir, call. = FALSE)
  }
  metas <- lapply(ann_files, function(f)
    tryCatch(read_sample_annotation(f, config$controls), error = function(e) NULL))
  overlap <- vapply(metas, function(m)
    if (is.null(m)) 0L else sum(unique(m$Barcode) %in% bc), integer(1))
  if (max(overlap) == 0L) {
    stop("No sample annotation matches the barcodes of the ", chip_type,
         " export (", paste(bc, collapse = ", "), "). Looked in: ",
         paste(basename(ann_files), collapse = ", "), call. = FALSE)
  }
  pick <- which.max(overlap)
  message(sprintf("  annot : %s  (%d of %d barcodes matched)",
                  basename(ann_files[pick]), overlap[pick], length(bc)))
  meta <- metas[[pick]]
  meta <- meta[meta$Barcode %in% bc, , drop = FALSE]
  message(sprintf("  controls={%s}  tests={%s}",
                  paste(sort(unique(meta$Group[meta$Role == "control"])), collapse = ", "),
                  paste(sort(unique(meta$Group[meta$Role == "test"])), collapse = ", ")))

  df <- raw %>%
    inner_join(meta, by = c("Barcode", "Well")) %>%
    group_by(SampleName, Peptide, ExposureTime, Cycle, Barcode, Group) %>%
    summarise(Signal = mean(.data$Signal, na.rm = TRUE),
              SignalSaturation = max(.data$SignalSaturation, na.rm = TRUE),
              .groups = "drop")
  message(sprintf("  %d rows, %d peptides, %d samples, cycles %s, exposures %s",
                  nrow(df), n_distinct(df$Peptide), n_distinct(df$SampleName),
                  paste(sort(unique(df$Cycle)), collapse = "/"),
                  paste(sort(unique(df$ExposureTime)), collapse = "/")))

  refs <- chip_refs[[chip_type]]
  tdir <- file.path(config$output_dir, chip_type, "tables")
  dir.create(tdir, recursive = TRUE, showWarnings = FALSE)

  data  <- krsa_qc_steps(df, sat_qc = config$sat_qc)
  pw200 <- krsa_extractEndPointMaxExp(data, chip_type)
  pw    <- krsa_extractEndPoint(data, chip_type)
  modeled <- krsa_scaleModel(pw, unique(pw$Peptide))

  ppAll <- krsa_filter_lowPeps(pw200, config$signal_threshold)
  ppR2  <- krsa_filter_nonLinear(filter(modeled$scaled, Peptide %in% ppAll), config$r2_threshold)
  new_pep <- krsa_filter_ref_pep(ppR2)
  readr::write_lines(new_pep, file.path(tdir, "global_qc_passed_peptides.txt"))
  n_total <- n_distinct(pw$Peptide)
  message(sprintf("  %d of %d peptides passed global QC", length(new_pep), n_total))

  comps <- build_comparisons(meta)
  message(sprintf("  %d comparison(s): %s", length(comps),
                  paste(vapply(comps, `[[`, character(1), "name"), collapse = ", ")))
  res <- lapply(comps, function(cp) tryCatch(
    analyse_comparison(cp, modeled, pw200, refs, tdir),
    error = function(e) { message("     [ERROR] ", cp$name, ": ", conditionMessage(e))
      list(comparison = cp$name, qc_peptides = NA_integer_, hits = NA_integer_,
           kinase_hits = character(0), krsa_calls = 0L,
           status = paste("failed:", conditionMessage(e))) }))

  list(array = chip_type, files = basename(files), peptides_total = n_total,
       peptides_qc = length(new_pep), map = refs$map_name, cov = refs$cov_name,
       comparisons = res,
       krsa_calls = sum(vapply(res, function(r) as.integer(r$krsa_calls), integer(1))))
}

main <- function() {
  dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("KRSA %s on BioNavigator input -- dataset %s, arrays %s, scope %s",
                  KRSA_VERSION, config$dataset_label,
                  paste(config$arrays, collapse = "+"), config$krsa_scope))

  rows <- list(); arrays <- list(); w0 <- Sys.time()
  for (ct in config$arrays) {
    t <- timed(analyse_array(ct))
    arrays[[length(arrays) + 1L]] <- c(t$value, list(wall = t$wall, cpu = t$cpu))
    rows[[length(rows) + 1L]] <- data.frame(
      dataset = config$dataset_label, array = ct, stage = "KRSA (BioNavigator input)",
      scope = config$krsa_scope, krsa_calls = t$value$krsa_calls,
      wall_seconds = round(t$wall, 3), cpu_seconds = round(t$cpu, 3))
    message(sprintf("  [KRSA] %s done in %s wall / %s cpu",
                    ct, fmt_secs(t$wall), fmt_secs(t$cpu)))
  }
  total <- as.numeric(difftime(Sys.time(), w0, units = "secs"))
  calls <- sum(vapply(arrays, function(a) as.integer(a$krsa_calls), integer(1)))
  rows[[length(rows) + 1L]] <- data.frame(
    dataset = config$dataset_label, array = "ALL", stage = "KRSA (dataset)",
    scope = config$krsa_scope, krsa_calls = calls,
    wall_seconds = round(total, 3),
    cpu_seconds = round(sum(vapply(arrays, `[[`, numeric(1), "cpu")), 3))
  tbl <- bind_rows(rows)
  readr::write_csv(tbl, file.path(config$output_dir, "timing.csv"))

  lines <- c(
    "KRSA on BioNavigator image analysis -- runtime",
    "=============================================",
    sprintf("written : %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    sprintf("machine : %s, R %s.%s", R.version$platform, R.version$major, R.version$minor),
    sprintf("KRSA    : %s  (%s)", KRSA_VERSION, KRSA_SOURCE),
    sprintf("dataset : %s", config$dataset_label),
    sprintf("input   : BioNavigator export -- KRSA's native input, so the",
            ""),
    "          saturation filter runs at KRSA's own default and no signal",
    "          rescaling is needed.",
    "",
    sprintf("krsa scope        : %s", config$krsa_scope),
    sprintf("saturation filter : %s", if (config$sat_qc) "ON (KRSA default)" else "OFF"),
    sprintf("signal_threshold  : %s", config$signal_threshold),
    sprintf("r2_threshold      : %s", config$r2_threshold),
    sprintf("lfc_cutoffs       : %s", paste(config$lfc_cutoffs, collapse = ", ")),
    sprintf("z_hit_threshold   : %s", config$z_hit_threshold),
    sprintf("iterations        : %s", config$iterations),
    sprintf("seed              : %s", config$seed),
    "",
    "KRSA is single-threaded: krsa() samples in a serial loop, and the runtime",
    "is (number of krsa() calls) x (cost of one call). Quote the scope with the",
    "number.",
    "",
    sprintf("%-6s %-28s %8s %12s %12s", "array", "stage", "krsa()", "wall (s)", "cpu (s)"),
    strrep("-", 72))
  for (i in seq_len(nrow(tbl))) {
    r <- tbl[i, ]
    lines <- c(lines, sprintf("%-6s %-28s %8d %12.1f %12.1f",
      r$array, r$stage, r$krsa_calls, r$wall_seconds, r$cpu_seconds))
  }
  lines <- c(lines, "", sprintf("TOTAL  %s  over %d krsa() call(s)  -> %.2f s per call",
                                fmt_secs(total), calls, total / max(1, calls)), "")
  for (a in arrays) {
    lines <- c(lines, sprintf("[%s]  input %s", a$array, paste(a$files, collapse = ", ")),
      sprintf("   mapping %s / coverage %s", a$map, a$cov),
      sprintf("   peptides passing global QC: %d of %d", a$peptides_qc, a$peptides_total))
    for (cr in a$comparisons) {
      lines <- c(lines, sprintf("   %-28s QC %-4s hits %-4s kinases %-3s %s",
        cr$comparison, cr$qc_peptides, cr$hits, length(cr$kinase_hits),
        if (length(cr$kinase_hits)) paste0("(", paste(cr$kinase_hits, collapse = ", "), ")")
        else cr$status))
    }
    lines <- c(lines, "")
  }
  writeLines(lines, file.path(config$output_dir, "timing_summary.txt"))
  writeLines(lines, file.path(config$output_dir, "run_manifest.txt"))
  cat("\n", paste(lines, collapse = "\n"), "\n", sep = "")
  message("Done.")
}
main()
