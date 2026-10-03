"""Run the KRSA package on pyKinaXe's own image analysis for both datasets.

The script drives run_KRSA_on_pykinaxe_image_analysis/run_krsa_on_pykinaxe_image_analysis.R
once per dataset and writes each dataset's KRSA results into the folder the
family comparison reads:

    run_KRSA_on_pykinaxe_image_analysis/exports_pykinaxe/     input
        exports_image_analysis_pyKinaXe_benchmarking_data_set/
        exports_image_analysis_pykinaxe_CDRL_vwr-rats-kinome_data_set/
    results_benchmarking_data_set/                            output, next to this script
    results_CDRL_vwr-rats-kinome_data_set/
        PTK/tables/   <test>_vs_<control>_KRSA_acrossChip.txt   read by ../compare_pykinaxe_krsa.py
        PTK/figures/  STK/tables/  STK/figures/
        run_manifest.txt   thresholds, KRSA version, peptide and hit counts
        run_krsa.log       everything the R run printed

The scoring is KRSA's own R code; this file only decides which dataset goes
where, keeps a log and reports what came out. The R script can also be run by
hand (cd run_KRSA_on_pykinaxe_image_analysis; Rscript run_krsa_on_pykinaxe_image_analysis.R),
in which case it writes into its own output/<dataset>/.

The input files <timestamp>_Export_image_analysis_<PTK|STK>_bn.csv are
pyKinaXe's spot values in BioNavigator-compatible column layout ("_bn" refers
to the format, not to BioNavigator as the source).

Requirements: R with the KRSA package; the R script installs KRSA from GitHub
if it is missing (--no-install forbids that). Only the Python standard library
is used.

Run:
    python validation_paper/compare_pyKinaXe_results_with_KRSA/results_KRSA/run_KRSA_on_pykinaxe_input.py
        --datasets rat          one dataset
        --no-figures            tables only, much faster
        --list                  show what would run
        --dry-run               print the Rscript commands only

Both datasets take about 2 to 6 minutes each with figures. Re-running
overwrites the tables and figures of the datasets it runs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
KRSA_DIR = Path(__file__).resolve().parent                 # results_KRSA/
RUNNER_DIR = KRSA_DIR / "run_KRSA_on_pykinaxe_image_analysis"
R_SCRIPT = RUNNER_DIR / "run_krsa_on_pykinaxe_image_analysis.R"
EXPORTS_ROOT = RUNNER_DIR / "exports_pykinaxe"

# The datasets, in the order they are run. `exports` is the folder of pyKinaXe
# image-analysis exports (input), `results` the folder the KRSA tables are
# written to (output, next to this script). Both are names, not paths, so this
# block stays readable; they are resolved against EXPORTS_ROOT and KRSA_DIR.
DATASETS: dict[str, dict[str, str]] = {
    "benchmarking": {
        "title": "benchmarking (HDV constructs, 2 chips per array)",
        "exports": "exports_image_analysis_pyKinaXe_benchmarking_data_set",
        "results": "results_benchmarking_data_set",
    },
    "rat": {
        "title": "CDRL vwr-rats-kinome (3 chips per array, two controls)",
        "exports": "exports_image_analysis_pykinaxe_CDRL_vwr-rats-kinome_data_set",
        "results": "results_CDRL_vwr-rats-kinome_data_set",
    },
}

LOG_NAME = "run_krsa.log"
MANIFEST_NAME = "run_manifest.txt"
# ============================================================================


def resolve_rscript(explicit: str | None) -> str:
    """Locate the Rscript executable, or explain how to get one."""
    candidate = explicit or shutil.which("Rscript")
    if candidate and (Path(candidate).exists() or shutil.which(candidate)):
        return candidate
    raise SystemExit(
        "Rscript not found on PATH.\n"
        "  Install R (e.g. `brew install r`) or pass --rscript /path/to/Rscript.\n"
        "  KRSA itself is installed by the R script from "
        "https://github.com/CogDisResLab/KRSA."
    )


def r_script_args(args: argparse.Namespace) -> list[str]:
    """Translate the options shared with the R script into its CLI flags.

    Only the options actually given on the command line are passed on, so the R
    script's own documented defaults stay the single source of truth for the
    thresholds -- they are not duplicated here.
    """
    passthrough: list[str] = []
    for flag, value in (
        ("--iterations", args.iterations),
        ("--seed", args.seed),
        ("--signal-threshold", args.signal_threshold),
        ("--r2-threshold", args.r2_threshold),
        ("--z-threshold", args.z_threshold),
        ("--lfc-cutoffs", args.lfc_cutoffs),
    ):
        if value is not None:
            passthrough += [flag, str(value)]
    if args.no_figures:
        passthrough.append("--no-figures")
    if args.no_install:
        passthrough.append("--no-install")
    return passthrough


def run_dataset(
    key: str,
    spec: dict[str, str],
    rscript: str,
    extra_args: list[str],
    dry_run: bool,
) -> tuple[bool, Path]:
    """Run the R script for one dataset. Returns (success, output folder)."""
    input_dir = EXPORTS_ROOT / spec["exports"]
    output_dir = KRSA_DIR / spec["results"]

    if not input_dir.is_dir():
        print(f"  [ERROR] input folder missing: {input_dir}")
        return False, output_dir

    command = [
        rscript,
        str(R_SCRIPT),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]

    print(f"\n{'=' * 78}")
    print(f"DATASET  {key}  --  {spec['title']}")
    print(f"  input  {input_dir}")
    print(f"  output {output_dir}")
    print(f"  $ {' '.join(command)}")
    print(f"{'=' * 78}", flush=True)

    if dry_run:
        return True, output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / LOG_NAME

    # Stream the R output to the terminal AND into the dataset's log, so a run
    # can be watched live and read again afterwards.
    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            f"# {' '.join(command)}\n"
            f"# started {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=RUNNER_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        returncode = process.wait()
        log.write(f"\n# exit status {returncode}\n")

    if returncode != 0:
        print(f"  [ERROR] {key}: Rscript exited with status {returncode} "
              f"(log: {log_path})")
        return False, output_dir
    return True, output_dir


def print_manifest(output_dir: Path) -> None:
    """Echo the per-array blocks of the manifest the R script just wrote."""
    manifest = output_dir / MANIFEST_NAME
    if not manifest.is_file():
        print(f"  (no {MANIFEST_NAME} in {output_dir})")
        return
    keep = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.startswith("["):          # [PTK] / [STK]
            keep = True
        if keep and line.strip():
            print(f"  {line}")


def list_datasets(selected: list[str]) -> None:
    print(f"Input root : {EXPORTS_ROOT}")
    print(f"Output root: {KRSA_DIR}")
    print(f"R script   : {R_SCRIPT}")
    for key in selected:
        spec = DATASETS[key]
        input_dir = EXPORTS_ROOT / spec["exports"]
        output_dir = KRSA_DIR / spec["results"]
        state = "ok" if input_dir.is_dir() else "MISSING"
        print(f"\n  {key}  ({spec['title']})")
        print(f"    exports [{state}] {input_dir}")
        if input_dir.is_dir():
            for export in sorted(input_dir.glob("*/*_Export_image_analysis_*_bn.csv")):
                size_mb = export.stat().st_size / 1e6
                print(f"        {export.parent.name}/{export.name}  ({size_mb:.0f} MB)")
        existing = sorted(p.name for p in output_dir.glob("*")) if output_dir.is_dir() else []
        print(f"    results {output_dir}"
              f"{'  (holds: ' + ', '.join(existing) + ')' if existing else '  (empty)'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Defaults for every threshold live in the R script; this driver "
               "only forwards the ones given here.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), metavar="NAME",
        default=sorted(DATASETS),
        help=f"which datasets to run (default: all -- {', '.join(sorted(DATASETS))})",
    )
    parser.add_argument("--rscript", default=None,
                        help="path to the Rscript executable (default: from PATH)")
    parser.add_argument("--no-figures", action="store_true",
                        help="tables only; skips every KRSA figure (much faster)")
    parser.add_argument("--no-install", action="store_true",
                        help="fail instead of installing a missing KRSA from GitHub")
    parser.add_argument("--iterations", type=int, default=None,
                        help="krsa() sampling iterations (R default: 2000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling seed (R default: 123)")
    parser.add_argument("--signal-threshold", type=float, default=None,
                        help="peptide QC signal threshold (R default: 5)")
    parser.add_argument("--r2-threshold", type=float, default=None,
                        help="peptide QC R^2 threshold (R default: 0.9)")
    parser.add_argument("--z-threshold", type=float, default=None,
                        help="|Z| for a kinase-family hit (R default: 2)")
    parser.add_argument("--lfc-cutoffs", default=None,
                        help="hit-peptide log2 FC cutoffs, comma separated "
                             "(R default: 0.2,0.3,0.4)")
    parser.add_argument("--list", action="store_true",
                        help="show the datasets and their folders, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the Rscript commands without running them")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue with the next dataset after a failure")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected: list[str] = list(args.datasets)

    if args.list:
        list_datasets(selected)
        return 0

    if not R_SCRIPT.is_file():
        raise SystemExit(f"R script not found: {R_SCRIPT}")

    rscript = resolve_rscript(args.rscript)
    extra_args = r_script_args(args)

    print(f"KRSA on pyKinaXe image analysis -- {len(selected)} dataset(s): "
          f"{', '.join(selected)}")
    print(f"  Rscript: {rscript}")

    outcomes: list[tuple[str, bool, Path]] = []
    for key in selected:
        ok, output_dir = run_dataset(key, DATASETS[key], rscript, extra_args,
                                     args.dry_run)
        outcomes.append((key, ok, output_dir))
        if not ok and not args.keep_going:
            print("\nStopping after the first failure (--keep-going overrides).")
            break

    if args.dry_run:
        print("\nDry run -- nothing was executed.")
        return 0

    print(f"\n{'=' * 78}")
    print("ALL DATASETS")
    print(f"{'=' * 78}")
    for key, ok, output_dir in outcomes:
        print(f"\n{key}: {'ok' if ok else 'FAILED'}  -> {output_dir}")
        if ok:
            print_manifest(output_dir)

    failed = [key for key, ok, _ in outcomes if not ok]
    skipped = [key for key in selected if key not in {k for k, _, _ in outcomes}]
    if failed or skipped:
        print("\nFailed: " + (", ".join(failed) or "-")
              + " | not run: " + (", ".join(skipped) or "-"))
        return 1
    print("\nNext: ../compare_pykinaxe_krsa.py puts these tables next to "
          "pyKinaXe's family results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
