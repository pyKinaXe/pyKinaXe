"""Shift one frame across a range of offsets and record what the detector finds.

The image is only translated: angle 0 and a whole number of pixels, so the
transformation is a plain array copy and any disagreement belongs to the
detector. ../rotation/ is the corresponding rotation test on the same shared
code (../robustness_common.py). NOTE.txt in this folder holds the result.

    python validation_paper/validate_robustnes_of_image_analysis/translation/translation_sweep.py
    ... --axes x --from -100 --to 100 --step 2
    ... --image 640091210_W1_F1_T10_P32_I1_A30.tif --no-panels

Output, under output/sweep_<stem>/: legend.png, <stem>_original.png and one
marked figure per offset (<stem>_x+040.png, <stem>_y-030.png), all at the same
print width of 94.5 mm.

The two axes are swept separately (dx with dy = 0, dy with dx = 0). The
terminal output lists per axis how many offsets were detected, over which
range, the range the frame allows, and two residuals: the T fiducial, which is
printed on the chip and must move by exactly the applied shift, and the
aperture centre, which does not follow the shift exactly because the disc is
clipped by the frame and a shift changes which part of the rim is visible.

Steps that would push the printed content out of the frame are reported as
not testable and excluded, so the limit found here is the frame's, unlike the
rotation test, where it is the detector's. A detector failure appears only at
short exposures: a large shift towards the clipped edge together with the
black fill makes the aperture detection lose the bright blob, and the skewed
mask cuts off the reference spots. frames.mode: exposures sweeps the same
array at 10, 50 and 200 ms to show it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from robustness_common import (  # noqa: E402
    admissible_shift_range,
    cleanup,
    collect_candidates,
    contiguous_window,
    load_settings,
    pick_frames,
    resolve_path,
    run_sweep,
)

DEFAULTS: Dict[str, Any] = {
    "axes": ["x", "y"],
    "from_px": -160,
    "to_px": 160,
    "step_px": 5,
}


def build_steps(cfg: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    """The offsets to apply, in order, with no rotation at any of them.

    Whole pixels only. A whole-pixel shift with no rotation is an exact array
    copy, and giving that up for sub-pixel resolution would trade the one
    property that makes this half of the test clean.

    Args:
        cfg (Dict[str, Any]): The effective settings.
        stem (str): The frame's file stem, for the panel file names.

    Returns:
        List[Dict[str, Any]]: One step per (axis, offset).
    """
    start, stop, step = int(cfg["from_px"]), int(cfg["to_px"]), int(cfg["step_px"])
    offsets = list(range(start, stop + 1, step))

    steps: List[Dict[str, Any]] = []
    for axis in cfg["axes"]:
        for offset in offsets:
            shift = (float(offset), 0.0) if axis == "x" else (0.0, float(offset))
            steps.append({
                "value": float(offset),
                "series": axis,
                "angle_deg": 0.0,             # TRANSLATION ONLY
                "shift": shift,
                "applied": f"translated ({shift[0]:+.0f}, {shift[1]:+.0f}) px",
                "stem": f"{stem}_{axis}{offset:+05d}",
            })
    return steps


def report_axis(subset: List[Dict[str, Any]], axis: str, allowed: Tuple[float, float]) -> None:
    """Print what one axis of the sweep found.

    Args:
        subset (List[Dict[str, Any]]): That axis's rows.
        axis (str): "x" or "y".
        allowed (Tuple[float, float]): The range the frame allows.
    """
    hits = [r for r in subset if r["detected"]]
    print(f"  {axis} axis: {len(hits)} of {len(subset)} offsets detected"
          f"   frame allows [{allowed[0]:+.0f}, {allowed[1]:+.0f}] px")
    measured = contiguous_window(subset)
    if measured:
        low, high, inside = measured
        print(f"    detected over [{low:+.0f}, {high:+.0f}] px"
              + (f"   with {inside} failure(s) INSIDE it" if inside else "   (contiguous)"))
    if not hits:
        return

    component = "t_dx" if axis == "x" else "t_dy"
    cross = "t_dy" if axis == "x" else "t_dx"
    residual = np.array([abs(r[component] - r["value"]) for r in hits], dtype=float)
    drift = np.array([abs(r[cross]) for r in hits], dtype=float)
    ap_res = np.array([
        abs((r["ap_dx"] if axis == "x" else r["ap_dy"]) - r["value"]) for r in hits
    ], dtype=float)
    delta = np.array([r["median_abs_delta"] for r in hits], dtype=float)
    print(f"    T fiducial residual   median {np.nanmedian(residual):.2f} px, "
          f"worst {np.nanmax(residual):.2f} px   (cross-axis drift worst "
          f"{np.nanmax(drift):.2f} px)")
    print(f"    aperture residual     median {np.nanmedian(ap_res):.1f} px, "
          f"worst {np.nanmax(ap_res):.1f} px   (expected to be large: clipped disc)")
    print(f"    spot values           median |dI| {np.nanmedian(delta):.3f} counts, "
          f"worst {np.nanmax(delta):.3f}")


def main() -> int:
    """Sweep the offset on one frame and write one figure per offset.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(HERE / "config_translation.yaml"))
    parser.add_argument("--source", help="Folder to take the frame from.")
    parser.add_argument("--image", help="Use exactly this frame, by file name.")
    parser.add_argument("--seed", type=int, help="Seed for the default frame choice.")
    parser.add_argument("--out", help="Where to write the figures.")
    parser.add_argument("--axes", help="Which axes to sweep: x, y or x,y.")
    parser.add_argument("--from", dest="from_px", type=int, metavar="PX")
    parser.add_argument("--to", dest="to_px", type=int, metavar="PX")
    parser.add_argument("--step", dest="step_px", type=int, metavar="PX")
    parser.add_argument("--exposures-mode", action="store_true",
                        help="Sweep every configured exposure of the chosen array, "
                             "not only its brightest frame.")
    parser.add_argument("--no-panels", action="store_true",
                        help="Report to the terminal only, write no figures.")
    parser.add_argument("--no-panel-legend", action="store_true",
                        help="Leave the legend off the panels; legend.png is "
                             "written either way.")
    args = parser.parse_args()

    try:
        cfg = load_settings(Path(args.config), "translation_sweep", DEFAULTS)
        for key, value in (("source", args.source), ("seed", args.seed),
                           ("output_dir", args.out), ("from_px", args.from_px),
                           ("to_px", args.to_px), ("step_px", args.step_px)):
            if value is not None:
                cfg[key] = value
        if args.axes:
            cfg["axes"] = [a.strip() for a in args.axes.split(",") if a.strip()]
        if args.image:
            cfg["frames"]["image"] = args.image
        if args.exposures_mode:
            cfg["frames"]["mode"] = "exposures"
        if args.no_panels:
            cfg["panels"] = False
        if args.no_panel_legend:
            cfg["panel_legend"] = False

        unknown = [a for a in cfg["axes"] if a not in ("x", "y")]
        if unknown:
            print(f"ERROR: axes must be x and/or y, got {unknown}.")
            return 1

        source = resolve_path(str(cfg["source"]))
        if not source.is_dir():
            print(f"ERROR: source folder not found: {source}")
            return 1
        out_dir = Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = HERE / out_dir

        candidates = collect_candidates(source, cfg.get("cycles"), cfg.get("exposures"))
        if not candidates:
            print(f"ERROR: no TIFF in {source} matched the cycle/exposure filter.")
            return 1
        frames = pick_frames(cfg, candidates, source)

        print(f"source:   {source}")
        print(f"offsets:  {cfg['from_px']:+d} to {cfg['to_px']:+d} px in steps of "
              f"{cfg['step_px']:d} on {' and '.join(cfg['axes'])}"
              f"   TRANSLATION ONLY, no rotation")
        print(f"output:   {out_dir}\n")

        worst = 0
        for path, meta in frames:
            print(f"frame:    {path.name}")
            steps = build_steps(cfg, path.stem)
            result = run_sweep(path, meta, steps, cfg, out_dir / f"sweep_{path.stem}")
            if result is None or not result["rows"]:
                print("  nothing could be tested on this frame.\n")
                worst = 1
                continue

            if cfg.get("panels", True):
                print(f"\n  panels:  sweep_{path.stem}/  "
                      f"({len(result['rows']) + 1} figures, "
                      "incl. the original, plus legend.png)")
            if result["skipped"]:
                print(f"  {result['skipped']} offset(s) not testable "
                      "(content would leave the frame)")
            for axis in cfg["axes"]:
                subset = [r for r in result["rows"] if r["series"] == axis]
                if subset:
                    report_axis(subset, axis, admissible_shift_range(
                        result["box"], result["shape"], float(cfg["clearance_px"]), axis))
            print()
        return worst
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
