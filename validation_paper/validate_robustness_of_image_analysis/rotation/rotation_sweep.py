"""Rotate one frame across a range of angles and record what the detector finds.

The image is only rotated: pivot at the frame centre, no shift, so the angle is
the only difference between two steps. ../translation/ is the corresponding
translation test on the same shared code (../robustness_common.py). NOTE.txt
in this folder holds the result and the derivation of the tolerated window.

    python validation_paper/validate_robustnes_of_image_analysis/rotation/rotation_sweep.py
    ... --from -3 --to 3 --step 0.05
    ... --image 640091210_W1_F1_T200_P94_I473_A30.tif --no-panels

Output, under output/sweep_<stem>/: legend.png (the four markings, once),
<stem>_original.png (the untransformed frame) and one marked figure per angle
(<stem>_rot+00.30.png), all at the same print width of 94.5 mm. The numbers go
to the terminal: per angle whether the grid was found and how the spot values
changed, and at the end the measured window next to the one predicted from
the frame's own reference spots.

The window is set by _validate_t_j_geometry() in src/kx_image_processor.py:
the J middle must sit 19 spot spacings to the right of the T middle and 1
spacing (21.5 px) below it, each within spacing_tolerance = 10 px. Nineteen
spacings are a 408.5 px lever arm, so the tolerated rotation is
2 * asin(10 / 408.5) = 2.80 deg wide, and because the chips are printed with a
vertical offset of 19 to 21 px rather than 21.5 px the window is centred at
theta* = +0.07 to +0.35 deg. predicted_window() computes this from the frame's
reference spots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from robustness_common import (  # noqa: E402
    cleanup,
    collect_candidates,
    contiguous_window,
    detected,
    load_settings,
    pick_frames,
    resolve_path,
    run_sweep,
)

DEFAULTS: Dict[str, Any] = {
    "from_deg": -2.0,
    "to_deg": 2.0,
    "step_deg": 0.1,
}


def predicted_window(
    baseline: Dict[str, Any], spot_spacing: float = 21.5, tolerance: float = 10.0
) -> Optional[Tuple[float, float]]:
    """The rotation window _validate_t_j_geometry allows for THIS frame.

    Derived, not fitted, and derived from the frame's OWN reference spots rather
    than from the nominal layout -- which is the point, because the two disagree
    by about 2 px and that is what pushes the window off centre.

    Under a rotation by theta the T->J vector (dx, dy) has vertical component
    dx*sin(theta) + dy*cos(theta) and horizontal dx*cos(theta) - dy*sin(theta).
    The admissible thetas are those for which both stay within ``tolerance`` of
    their nominal values.

    Args:
        baseline (Dict[str, Any]): A detection on the UNROTATED frame.
        spot_spacing (float): The pitch the check compares against, in px.
        tolerance (float): reference_spot_detection.spacing_tolerance, in px.

    Returns:
        Optional[Tuple[float, float]]: (low, high) in degrees, or None if the
        unrotated frame carries no reference spots to derive it from.
    """
    if not detected(baseline):
        return None
    dx = float(baseline["j_shape"][1][0] - baseline["t_shape"][1][0])
    dy = float(baseline["j_shape"][1][1] - baseline["t_shape"][1][1])

    thetas = np.deg2rad(np.linspace(-10.0, 10.0, 40001))
    vertical = dx * np.sin(thetas) + dy * np.cos(thetas)
    horizontal = dx * np.cos(thetas) - dy * np.sin(thetas)
    ok = (np.abs(vertical - spot_spacing) < tolerance) & (
        np.abs(horizontal - 19 * spot_spacing) < tolerance
    )
    if not ok.any():
        return None
    return float(np.rad2deg(thetas[ok].min())), float(np.rad2deg(thetas[ok].max()))


def build_steps(cfg: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    """The angles to apply, in order, with no translation at any of them.

    Args:
        cfg (Dict[str, Any]): The effective settings.
        stem (str): The frame's file stem, for the panel file names.

    Returns:
        List[Dict[str, Any]]: One step per angle.
    """
    start, stop, step = (
        float(cfg["from_deg"]), float(cfg["to_deg"]), float(cfg["step_deg"])
    )
    n_steps = int(round((stop - start) / step)) + 1
    angles = np.round(start + step * np.arange(n_steps), 6)
    return [
        {
            "value": float(a),
            "angle_deg": float(a),
            "shift": (0.0, 0.0),          # ROTATION ONLY
            "applied": f"rotated {a:+.2f}\u00b0",
            "stem": f"{stem}_rot{a:+06.2f}",
        }
        for a in angles
    ]


def main() -> int:
    """Sweep the angle on one frame and write one figure per angle.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(HERE / "config_rotation.yaml"))
    parser.add_argument("--source", help="Folder to take the frame from.")
    parser.add_argument("--image", help="Use exactly this frame, by file name.")
    parser.add_argument("--seed", type=int, help="Seed for the default frame choice.")
    parser.add_argument("--out", help="Where to write the figures.")
    parser.add_argument("--from", dest="from_deg", type=float, metavar="DEG")
    parser.add_argument("--to", dest="to_deg", type=float, metavar="DEG")
    parser.add_argument("--step", dest="step_deg", type=float, metavar="DEG")
    parser.add_argument("--no-panels", action="store_true",
                        help="Report to the terminal only, write no figures.")
    parser.add_argument("--no-panel-legend", action="store_true",
                        help="Leave the legend off the panels; legend.png is "
                             "written either way.")
    args = parser.parse_args()

    try:
        cfg = load_settings(Path(args.config), "rotation_sweep", DEFAULTS)
        for key, value in (("source", args.source), ("seed", args.seed),
                           ("output_dir", args.out), ("from_deg", args.from_deg),
                           ("to_deg", args.to_deg), ("step_deg", args.step_deg)):
            if value is not None:
                cfg[key] = value
        if args.image:
            cfg["frames"]["image"] = args.image
        if args.no_panels:
            cfg["panels"] = False
        if args.no_panel_legend:
            cfg["panel_legend"] = False

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
        print(f"angles:   {cfg['from_deg']:+.2f} to {cfg['to_deg']:+.2f} deg in steps "
              f"of {cfg['step_deg']:.2f}   ROTATION ONLY, no translation")
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

            window = predicted_window(result["baseline"])
            rows = result["rows"]
            hits = sum(1 for r in rows if r["detected"])
            if cfg.get("panels", True):
                print(f"\n  panels:  sweep_{path.stem}/  ({len(rows) + 1} figures, "
                      "incl. the original, plus legend.png)")
            print(f"  {hits} of {len(rows)} angles detected"
                  + (f", {result['skipped']} not testable (content would leave the frame)"
                     if result["skipped"] else ""))
            measured = contiguous_window(rows)
            if measured:
                low, high, inside = measured
                print(f"  measured window:  [{low:+.2f}, {high:+.2f}] deg"
                      + (f"   with {inside} failure(s) INSIDE it" if inside
                         else "   (contiguous)"))
            if window is not None:
                print(f"  predicted window: [{window[0]:+.2f}, {window[1]:+.2f}] deg")
            print()
        return worst
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
