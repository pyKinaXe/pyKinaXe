"""Shared code of the rotation and translation robustness tests.

The two tests in rotation/ and translation/ differ only in the swept parameter
and in how the result is reported, so the transformation (apply_transform(),
one resampling pass, black fill), the admissibility check (box_stays_in_frame()
on the corners of the content box), the yardstick (measure_content_box(),
independent of the detector), the detector call (detect_one(), one frame per
ImageProcessor), the markings (draw_detection_panel(), those of the publication
figure), the per-step figure (render_panel(), one frame per figure sized for
print), the legend (legend_handles(), render_legend()) and the sweep loop
(run_sweep()) live here.

The markings are copied from create_publication_figure() in
src/kx_image_processor.py, drawn on one panel: background-subtracted frame on a
symmetric log scale (RdBu_r), reference spots as yellow circles (r = 6, lw 3),
the peptide-grid mask as a lime rotated rectangle (lw 3.5), the aperture as a
black dashed circle (lw 3.5) and the grid spots as lime circles with
r = integration_radius (lw 2). The grid-spot radius is the aperture the
pipeline integrates over, so a circle that misses its spot means a wrong value
in the export.

Nothing is written to disk except the figures: the transformed frame replaces
the loader's pixel array in memory and is restored afterwards, and
ImageProcessor's results tree is redirected to a temporary directory.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _import_dir in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

# Redirect the pipeline's own results tree before ImageProcessor is imported, so
# a run leaves nothing behind but its figures.
_RESULTS_TMP = Path(tempfile.mkdtemp(prefix="pykinaxe_robustness_"))
os.environ["PYKINAXE_RESULTS_ROOT"] = str(_RESULTS_TMP)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from kx_data_importer import DataLoader  # noqa: E402
from kx_image_processor import ImageProcessor  # noqa: E402


def cleanup() -> None:
    """Remove the temporary results tree ImageProcessor wrote into."""
    shutil.rmtree(_RESULTS_TMP, ignore_errors=True)


# ---------------------------------------------------------------------------
# Print geometry -- the figures are sized for PAPER, not for the screen.
# ---------------------------------------------------------------------------
# ONE panel per figure, and TWO of those are meant to sit side by side on A4
# portrait:
#
#     210 mm - 2 x 8 mm margin - 5 mm gutter = 2 x 94.5 mm wide
#
# Each PNG is built at exactly that width and saved WITHOUT a tight bounding box,
# so the width is real at the run's dpi. Placed at 100 % on the page, every font
# size below IS its point size on paper -- nothing is scaled afterwards, which is
# what makes print text unreadable. The same convention as the waterfalls in
# validation_paper/compare_pyKinaXe_results_with_KRSA/.
PANEL_WIDTH_MM = 94.5
FONT_TITLE = 9.0
FONT_LEGEND = 8.0
# The two strips around the image: the title above it, the legend below it. The
# image itself gets the full panel width and whatever height its aspect ratio
# asks for, so no white gutter is ever left inside the frame.
TITLE_STRIP_MM = 5.5
LEGEND_STRIP_MM = 9.5          # the four entries in two rows of two
# Legend marker sizes, in points, chosen against FONT_LEGEND rather than against
# the markers on the image: a legend entry has to be readable next to its own
# label, not to scale with the frame.
LEGEND_MARKER_REF = 7.0
LEGEND_MARKER_GRID = 5.0

# ---------------------------------------------------------------------------
# The publication figure's marker geometry, verbatim.
# ---------------------------------------------------------------------------
# The RADII are in image pixels, so they scale with the panel on their own and
# are copied unchanged. The LINE WIDTHS and legend marker sizes are in POINTS and
# do NOT: drawn unchanged on a 94.5 mm panel they would be 1.7x heavier against
# the same spots as on the 165 mm panel these numbers were chosen on. MARKER_SCALE
# restores that ratio, so a panel looks like the paper figure, only smaller.
MARKER_REFERENCE_WIDTH_MM = 165.0
MARKER_SCALE = PANEL_WIDTH_MM / MARKER_REFERENCE_WIDTH_MM

REF_SPOT_RADIUS = 6
REF_SPOT_LINEWIDTH = 3 * MARKER_SCALE
REF_SPOT_COLOR = "yellow"
SQUARE_MASK_COLOR = "lime"
SQUARE_MASK_LINEWIDTH = 3.5 * MARKER_SCALE
APERTURE_COLOR = "black"
APERTURE_LINEWIDTH = 3.5 * MARKER_SCALE
GRID_SPOT_COLOR = "lime"
GRID_SPOT_LINEWIDTH = 2 * MARKER_SCALE
GRID_SPOT_ALPHA = 0.8

FILENAME_FIELDS = [
    "PamChip_barcode", "Array", "FoV", "Exposure_time",
    "Pump_cycle", "Image_number", "Temperature",
]
FILENAME_RE = re.compile(
    r"^(\d+)_W(\d+)_F(\d+)_T(\d+)_P(\d+)_I(\d+)_A(\d+)\.tiff?$", re.IGNORECASE
)

# Settings both tests share. Each test adds its own on top.
COMMON_DEFAULTS: Dict[str, Any] = {
    "source": "data/CDRL_vwr-rats-kinome_data_set",
    "seed": 0,
    "output_dir": "output",
    "cycles": None,
    "exposures": [10, 50, 200],
    "clearance_px": 21.5,
    "spline_order": 3,
    "integration_radius": 5.0,
    "dpi": 150,
    "panels": True,
    # The legend is ALWAYS written as its own figure (legend.png). This decides
    # only whether it is ALSO repeated under every panel.
    "panel_legend": True,
    # 300, because the panels are built at a fixed PRINT width (PANEL_WIDTH_MM)
    # and the dpi is therefore the only thing deciding how many pixels that
    # width carries. Below ~300 the markings soften on paper.
    "panel_dpi": 300,
    "frames": {"image": None, "mode": "brightest"},
    "content_box": {
        "smooth_sigma": 5.0,
        "spot_rel_threshold": 0.15,
        "spot_min_area": 12,
    },
}


def load_settings(path: Path, key: str, extra_defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Read one test's YAML block, falling back to the defaults for what is absent.

    Args:
        path (Path): The config file.
        key (str): Top-level key in it, e.g. "rotation_sweep".
        extra_defaults (Dict[str, Any]): The test's own defaults, merged over the
            shared ones.

    Returns:
        Dict[str, Any]: The effective settings.
    """
    settings: Dict[str, Any] = {**COMMON_DEFAULTS, **extra_defaults}
    for nested in ("content_box", "frames"):
        settings[nested] = dict(settings[nested])

    if not path.exists():
        print(f"WARNING: no config at {path}; using built-in defaults.")
        return settings

    with path.open("r", encoding="utf-8") as handle:
        loaded = (yaml.safe_load(handle) or {}).get(key) or {}
    for name, value in loaded.items():
        if name in ("content_box", "frames") and isinstance(value, dict):
            settings[name].update(value)
        else:
            settings[name] = value
    return settings


def resolve_path(value: str) -> Path:
    """Resolve a configured path against the repository root, then this folder.

    Args:
        value (str): Path as written in the config or on the command line.

    Returns:
        Path: The first candidate that exists, else the repository-root one.
    """
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (REPO_ROOT, HERE, Path.cwd()):
        if (base / candidate).exists():
            return base / candidate
    return REPO_ROOT / candidate


def parse_tiff_name(name: str) -> Optional[Dict[str, int]]:
    """Parse a PamGene TIFF file name into its seven metadata fields.

    The same fields, in the same order, as DataLoader.COORDINATE_NAMES -- the
    file name is what the loader parses, so this parser has to agree with it.

    Args:
        name (str): File name, not a path.

    Returns:
        Optional[Dict[str, int]]: The parsed fields, or None if it does not match.
    """
    match = FILENAME_RE.match(name)
    if match is None:
        return None
    return {field: int(value) for field, value in zip(FILENAME_FIELDS, match.groups())}


# ---------------------------------------------------------------------------
# measuring the frame -- independent of the detector under test
# ---------------------------------------------------------------------------
def measure_aperture(smoothed: np.ndarray) -> Optional[Dict[str, Any]]:
    """Locate the illuminated aperture as the largest bright connected region.

    Not ImageProcessor._detect_aperture_global(), so that the yardstick does not
    drift with the detector under test.

    Args:
        smoothed (np.ndarray): The frame after Gaussian smoothing, float.

    Returns:
        Optional[Dict[str, Any]]: The aperture mask, or None if there is none.
    """
    low, high = np.percentile(smoothed, 2), np.percentile(smoothed, 60)
    mask = ndi.binary_fill_holes(
        ndi.binary_opening(smoothed > 0.5 * (low + high), np.ones((9, 9)))
    )
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return None
    sizes = ndi.sum(mask, labels, range(1, n_labels + 1))
    return {"mask": labels == (int(np.argmax(sizes)) + 1)}


def measure_content_box(
    path: Path, cfg: Dict[str, Any]
) -> Optional[Tuple[float, float, float, float]]:
    """Bound everything printed on the chip that is visible in this frame.

    Both the peptide grid AND the reference spots, which sit outside the grid and
    therefore set the horizontal extent. This box, not the grid, is what "the
    grid and the reference spots stay visible" is about.

    Args:
        path (Path): The frame to measure.
        cfg (Dict[str, Any]): The ``content_box`` settings.

    Returns:
        Optional[Tuple[float, float, float, float]]: (x0, y0, x1, y1), or None if
        nothing is above the threshold -- which is what a faint frame looks like.
    """
    image = np.asarray(Image.open(path)).astype(float)
    smoothed = ndi.gaussian_filter(image, cfg["smooth_sigma"])

    aperture = measure_aperture(smoothed)
    if aperture is None:
        return None
    mask = aperture["mask"]

    background = float(np.median(smoothed[mask]))
    peak = float(smoothed[mask].max())
    spots = (smoothed > background + cfg["spot_rel_threshold"] * (peak - background)) & mask

    labels, n_labels = ndi.label(spots)
    if n_labels == 0:
        return None
    sizes = ndi.sum(spots, labels, range(1, n_labels + 1))
    keep = [i + 1 for i, size in enumerate(sizes) if size >= int(cfg["spot_min_area"])]
    if not keep:
        return None

    ys, xs = np.nonzero(np.isin(labels, keep))
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


# ---------------------------------------------------------------------------
# the transformation
# ---------------------------------------------------------------------------
def apply_transform(
    image: np.ndarray,
    angle_deg: float,
    pivot_xy: Tuple[float, float],
    shift_xy: Tuple[float, float],
    order: int,
) -> np.ndarray:
    """Rotate about ``pivot_xy`` and translate, filling the rest with black.

    ONE resampling pass for both. Each test passes zero for the parameter it is
    not sweeping, so the rotation test never translates and the translation test
    never rotates -- but they run through the identical code path, which is what
    makes their results comparable.

    A WHOLE-PIXEL TRANSLATION WITH NO ROTATION IS EXACT. The inverse matrix is
    then the identity and the offset is integral, so affine_transform copies
    pixels rather than interpolating them: every surviving pixel is bit-identical
    to the original and any disagreement belongs to the detector alone.

    The result is clipped back into the input's own value range. A cubic spline
    overshoots at a step edge and this image has a hard one -- the black border
    the transformation itself creates -- so without the clip there would be a
    bright rim just inside it, and the aperture detector would be handed a
    feature that is pure interpolation artefact.

    Args:
        angle_deg (float): Rotation, positive turns clockwise on screen (the y
            axis of an image points down).
        pivot_xy (Tuple[float, float]): Rotation centre as (x, y).
        shift_xy (Tuple[float, float]): Translation, as (dx, dy).
        order (int): Spline order for the resampling.

    Returns:
        np.ndarray: The transformed frame, same shape and dtype.
    """
    radians = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(radians), np.sin(radians)
    # ndimage works in (y, x), so the matrix is written in that order.
    forward = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    inverse = np.linalg.inv(forward)

    pivot = np.array([pivot_xy[1], pivot_xy[0]], dtype=float)
    shift = np.array([shift_xy[1], shift_xy[0]], dtype=float)
    # affine_transform reads the input at ``matrix @ out_coord + offset``.
    offset = pivot - inverse @ (pivot + shift)

    moved = ndi.affine_transform(
        image.astype(np.float32), matrix=inverse, offset=offset,
        order=order, mode="constant", cval=0.0,
    )
    moved = np.clip(moved, float(image.min()), float(image.max()))
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        moved = np.clip(np.rint(moved), info.min, info.max)
    return moved.astype(image.dtype)


def box_stays_in_frame(
    box: Tuple[float, float, float, float],
    angle_deg: float,
    pivot: Tuple[float, float],
    shift: Tuple[float, float],
    shape: Tuple[int, int],
    clearance: float,
) -> bool:
    """Whether the content box survives a candidate transformation.

    Checked on the four corners analytically rather than by transforming the
    frame and re-measuring it: the transformation is rigid, so the corners are
    exact and a rejected step costs nothing.

    Args:
        box (Tuple[float, float, float, float]): (x0, y0, x1, y1).
        angle_deg (float): Candidate rotation.
        pivot (Tuple[float, float]): Rotation centre, (x, y).
        shift (Tuple[float, float]): Candidate translation, (dx, dy).
        shape (Tuple[int, int]): (height, width) of the frame.
        clearance (float): Pixels that must stay free at every edge.

    Returns:
        bool: True if every corner stays inside the frame.
    """
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    radians = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(radians), np.sin(radians)
    dx = corners[:, 0] - pivot[0]
    dy = corners[:, 1] - pivot[1]
    rx = pivot[0] + cos_a * dx - sin_a * dy + shift[0]
    ry = pivot[1] + sin_a * dx + cos_a * dy + shift[1]

    height, width = shape
    return bool(
        rx.min() >= clearance and ry.min() >= clearance
        and rx.max() <= width - 1 - clearance
        and ry.max() <= height - 1 - clearance
    )


# ---------------------------------------------------------------------------
# running the detector and drawing what it found
# ---------------------------------------------------------------------------
def detect_one(loader: DataLoader, index: int) -> Dict[str, Any]:
    """Run the full image pipeline on a SINGLE frame and extract the result.

    One frame per processor: ImageProcessor's per-image aperture
    outlier rejection compares each frame against the running median of the ones
    before it (center_outlier_min_samples = 5), so a stack of differently
    transformed frames would let them correct each other. A stack of one cannot,
    which is the honest setting here: every frame is judged on its own, exactly
    as a single submitted image would be.

    Everything the drawing needs is copied out as plain arrays, so a result
    cannot be invalidated by a later run over the same loader.

    Args:
        loader (DataLoader): A loaded DataLoader.
        index (int): Index of the frame inside loader.images.

    Returns:
        Dict[str, Any]: display image, reference shapes, mask, aperture, grid and
        the per-spot median intensities.
    """
    processor = ImageProcessor(loader, images=[index])
    processor.process(layout_loader=loader.layout_loader, verbose=False)

    original = processor.original_images.isel(image_idx=0).values
    background = processor._background_images.isel(image_idx=0).values
    subtracted = original.astype(np.float32) - background.astype(np.float32)

    t_shape = processor._refined_reference_spots["t_shape"][0]
    j_shape = processor._refined_reference_spots["j_shape"][0]
    params = processor._square_mask_params[0]
    refined = processor._refined_grid_positions[0]
    center_x, center_y = processor._centers[0]
    medians = processor.refined_spot_median_intensities

    return {
        "display": processor._symmetric_log_transform(subtracted),
        "shape": original.shape,
        "t_shape": None if t_shape is None else np.array(t_shape, dtype=float),
        "j_shape": None if j_shape is None else np.array(j_shape, dtype=float),
        "mask": None if params is None else dict(params),
        "center": (float(center_x), float(center_y)),
        "radius": float(processor.radius),
        "grid": None if refined is None else np.array(refined, dtype=float),
        # The per-spot value the export carries (I_median). Kept so a sweep can
        # ask the question the picture cannot: not only "was the grid found" but
        # "did the spots come out the same".
        "median": None if medians is None else np.array(
            medians.isel(image_idx=0).values, dtype=float
        ),
    }


def detected(detection: Dict[str, Any]) -> bool:
    """Whether the detector found both reference shapes on this frame.

    Args:
        detection (Dict[str, Any]): The result of detect_one().

    Returns:
        bool: True if the grid could be placed at all.
    """
    return detection["t_shape"] is not None and detection["j_shape"] is not None


def t_middle(detection: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """The T shape's centre spot -- the array's own fiducial.

    Printed on the chip, so under a translation it must move by exactly the
    applied shift. The APERTURE centre must not: the disc is clipped by the frame
    in the raw data, so moving it changes which part of the rim is visible and
    with it the fitted centre.

    Args:
        detection (Dict[str, Any]): The result of detect_one().

    Returns:
        Optional[Tuple[float, float]]: (x, y), or None if nothing was detected.
    """
    if detection["t_shape"] is None:
        return None
    return float(detection["t_shape"][1][0]), float(detection["t_shape"][1][1])


def legend_handles() -> List[Line2D]:
    """The four markings, as legend entries, in the order they are applied.

    Returns:
        List[Line2D]: One handle per marking.
    """
    return [
        Line2D([0], [0], color=APERTURE_COLOR, linestyle="--",
               linewidth=APERTURE_LINEWIDTH, label="aperture"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=REF_SPOT_COLOR, markersize=LEGEND_MARKER_REF,
               markeredgewidth=REF_SPOT_LINEWIDTH, label="reference spots (T, J)"),
        Line2D([0], [0], color=SQUARE_MASK_COLOR, linewidth=SQUARE_MASK_LINEWIDTH,
               label="peptide-grid mask"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=GRID_SPOT_COLOR, markersize=LEGEND_MARKER_GRID,
               markeredgewidth=GRID_SPOT_LINEWIDTH, label="integrated spots"),
    ]


def draw_rotated_rectangle(ax, center_x, center_y, width, height, angle) -> None:
    """Draw the square mask, with the corner arithmetic the processor uses.

    Args:
        ax: The axes to draw on.
        center_x: Rectangle centre, x.
        center_y: Rectangle centre, y.
        width: Rectangle width.
        height: Rectangle height.
        angle: Rotation in radians.
    """
    half_w, half_h = width / 2, height / 2
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h],
         [-half_w, half_h], [-half_w, -half_h]]
    )
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated = corners @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T
    ax.plot(rotated[:, 0] + center_x, rotated[:, 1] + center_y,
            color=SQUARE_MASK_COLOR, linewidth=SQUARE_MASK_LINEWIDTH)


def draw_detection_panel(ax, detection: Dict[str, Any], integration_radius: float) -> None:
    """Draw one frame with the publication figure's four markings.

    Args:
        ax: The axes to draw on.
        detection (Dict[str, Any]): The result of detect_one().
        integration_radius (float): Radius of the grid-spot circles, i.e. the
            aperture the pipeline actually integrates over.
    """
    display = detection["display"]
    vmax = np.max(np.abs(display[display != 0])) if np.any(display != 0) else 1
    ax.imshow(display, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    for shape in (detection["t_shape"], detection["j_shape"]):
        if shape is None:
            continue
        for x, y in shape:
            ax.add_patch(
                plt.Circle((x, y), REF_SPOT_RADIUS, color=REF_SPOT_COLOR,
                           fill=False, linewidth=REF_SPOT_LINEWIDTH)
            )

    params = detection["mask"]
    if params is not None:
        draw_rotated_rectangle(
            ax, params["center_x"], params["center_y"],
            params["right_x"] - params["left_x"],
            params["bottom_y"] - params["top_y"],
            params["angle_rad"],
        )

    center_x, center_y = detection["center"]
    ax.add_patch(
        plt.Circle((center_x, center_y), detection["radius"], color=APERTURE_COLOR,
                   fill=False, linewidth=APERTURE_LINEWIDTH, linestyle="--")
    )

    refined = detection["grid"]
    if refined is not None:
        n_rows, n_cols = refined.shape[:2]
        for row in range(n_rows):
            for col in range(n_cols):
                x, y = refined[row, col]
                ax.add_patch(
                    plt.Circle((x, y), integration_radius, color=GRID_SPOT_COLOR,
                               fill=False, linewidth=GRID_SPOT_LINEWIDTH,
                               alpha=GRID_SPOT_ALPHA)
                )

    height, width = detection["shape"]
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def render_panel(
    detection: Dict[str, Any],
    title: str,
    integration_radius: float,
    out_dir: Path,
    stem: str,
    dpi: int,
    with_legend: bool = True,
) -> Path:
    """One figure, ONE frame: the markings on it, a bare title, the legend.

    The original and each transformed step get their OWN figure, all built to the
    same width and the same font sizes, so any two of them -- the original and one
    angle, or two angles -- can be placed next to each other on the page and read
    as a pair. Printing the original again beside every step, as this used to do,
    made each figure twice as wide and therefore half as large on paper.

    Args:
        detection (Dict[str, Any]): The result of detect_one() for this frame.
        title (str): The panel title, e.g. "original" or "rotated +0.30 deg".
        integration_radius (float): Radius of the grid-spot circles.
        out_dir (Path): Where to write.
        stem (str): File stem, so the steps of one frame sort next to each other.
        dpi (int): Output resolution. 300 for print; the width in mm is fixed.
        with_legend (bool): Draw the legend under the image. False leaves the
            panel bare and the strip is not paid for at all, for a page that
            sets render_legend()'s standalone column once beside several panels.

    Returns:
        Path: The written file.
    """
    height_px, width_px = detection["shape"]
    width_in = PANEL_WIDTH_MM / 25.4
    image_in = width_in * height_px / width_px
    title_in = TITLE_STRIP_MM / 25.4
    legend_in = LEGEND_STRIP_MM / 25.4 if with_legend else 0.0
    height_in = image_in + title_in + legend_in

    # add_axes, not subplots: the image is given the full panel width and exactly
    # the height its aspect ratio asks for, so the figure carries no white margin
    # around it and the printed width of the PANEL is the printed width of the
    # IMAGE.
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes((0.0, legend_in / height_in, 1.0, image_in / height_in))
    draw_detection_panel(ax, detection, integration_radius)

    # Bare title: "original" or what was applied, nothing else, in one
    # colour. The file name used to sit above it as a figure title and the
    # detected aperture, radius and grid angle used to be printed next to it --
    # all of it goes to the terminal instead, so the figure carries no annotation
    # that is not part of the image itself.
    #
    # Consequence: a panel whose detection failed is not marked as
    # such. It shows no reference spots, no grid mask and no spot circles, and
    # that absence IS the result -- but nothing on the figure says so. Which
    # steps failed is in the run's terminal output.
    ax.set_title(title, fontsize=FONT_TITLE, pad=3)

    if with_legend:
        fig.legend(handles=legend_handles(), loc="lower center", ncol=2,
                   fontsize=FONT_LEGEND, frameon=False,
                   bbox_to_anchor=(0.5, 0.0), borderaxespad=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    # No bbox_inches="tight": it would crop the figure to its content and the
    # printed width would no longer be PANEL_WIDTH_MM.
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def render_legend(out_dir: Path, dpi: int, stem: str = "legend") -> Path:
    """The four markings as their own figure, with no panel attached.

    Written once per sweep folder, at the SAME point sizes as the panels and
    cropped to its own content, so it can be placed ONCE next to several panels
    instead of being repeated inside each of them.

    ONE COLUMN, the four entries stacked. That is a column, not a strip, and it
    is meant to stand BESIDE the panels: two 94.5 mm panels fill 189 mm of the
    190 mm A4 text width, so a horizontal strip would have to go under them and
    cost page height, while a column fits in the margin of a page laid out with
    one panel per row, or beside a pair on landscape. The panel's own legend
    stays in two rows of two -- it has only 94.5 mm to work with, and four
    stacked rows would cost it 19 mm of height.

    Args:
        out_dir (Path): Where to write.
        dpi (int): Output resolution, the panels' own.
        stem (str): File stem.

    Returns:
        Path: The written file.
    """
    fig = plt.figure(figsize=(PANEL_WIDTH_MM / 25.4, PANEL_WIDTH_MM / 25.4))
    fig.legend(handles=legend_handles(), loc="center", ncol=1,
               fontsize=FONT_LEGEND, frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    # bbox_inches="tight" HERE and nowhere else: the column has no fixed width to
    # preserve, and cropping it to its content is what lets it be set against
    # whatever it is placed beside. The figsize above is only an upper bound.
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# picking the frames
# ---------------------------------------------------------------------------
def collect_candidates(source: Path, cycles, exposures) -> List[Tuple[Path, Dict[str, int]]]:
    """List every TIFF under ``source`` a test may use.

    Args:
        source (Path): A run folder, an ImageResults folder, or any folder
            containing run folders.
        cycles: Pump cycles to keep, or None for all.
        exposures: Exposure times to keep, or None for all.

    Returns:
        List[Tuple[Path, Dict[str, int]]]: Candidate files with their metadata.
    """
    if source.name == "ImageResults":
        image_dirs = [source]
    else:
        image_dirs = sorted({p for p in source.rglob("ImageResults") if p.is_dir()})

    candidates: List[Tuple[Path, Dict[str, int]]] = []
    for image_dir in image_dirs:
        for path in sorted(image_dir.glob("*.tif")):
            meta = parse_tiff_name(path.name)
            if meta is None:
                continue
            if cycles and meta["Pump_cycle"] not in set(cycles):
                continue
            if exposures and meta["Exposure_time"] not in set(exposures):
                continue
            candidates.append((path, meta))
    return candidates


def brightest_sibling(image_dir: Path, meta: Dict[str, int]) -> Optional[Path]:
    """The brightest frame of the same array, used to measure the content box.

    A frame at the first kinetic cycle and 10 ms often has no spot above any
    sensible threshold, so its own content box cannot be measured. The chip does
    not move between cycles, so the brightest frame of the SAME (barcode, well)
    has the same geometry and does show it.

    Args:
        image_dir (Path): The ImageResults directory the frame came from.
        meta (Dict[str, int]): The frame's parsed metadata.

    Returns:
        Optional[Path]: The brightest sibling, or None if there is none.
    """
    best, best_key = None, (-1, -1)
    prefix = f"{meta['PamChip_barcode']}_W{meta['Array']}_"
    for path in image_dir.glob(f"{prefix}*.tif"):
        sibling = parse_tiff_name(path.name)
        if sibling is None:
            continue
        key = (sibling["Pump_cycle"], sibling["Exposure_time"])
        if key > best_key:
            best, best_key = path, key
    return best


def pick_frames(
    cfg: Dict[str, Any], candidates: List[Tuple[Path, Dict[str, int]]], source: Path
) -> List[Tuple[Path, Dict[str, int]]]:
    """Choose which frame(s) the sweep runs on.

    Both tests use the same two modes, so a rotation and a translation run can be
    pointed at the same frame and compared directly.

        image: <name>       exactly that file, whatever its exposure
        mode: brightest     one random ARRAY, then its BRIGHTEST frame. Not any
                            random frame: a 10 ms first-kinetic frame would add
                            "the signal was marginal anyway" as a second
                            explanation for every failure.
        mode: exposures     one random array and cycle, then EVERY configured
                            exposure of it. Same chip, same geometry, different
                            signal levels -- which is what separates a detection
                            limit from a signal limit.

    Args:
        cfg (Dict[str, Any]): The effective settings.
        candidates (List[Tuple[Path, Dict[str, int]]]): The filtered pool.
        source (Path): The folder they came from, for the error message.

    Returns:
        List[Tuple[Path, Dict[str, int]]]: The chosen frames, brightest first.
    """
    frames_cfg = cfg["frames"]
    wanted = frames_cfg.get("image")
    if wanted:
        matches = [
            c for c in candidates
            if c[0].name == str(wanted) or str(c[0]).endswith(str(wanted))
        ]
        if not matches:
            raise SystemExit(f"ERROR: {wanted} is not in the pool drawn from {source}.")
        return matches[:1]

    rng = np.random.default_rng(int(cfg["seed"]))
    arrays = sorted({
        (c[0].parent, c[1]["PamChip_barcode"], c[1]["Array"]) for c in candidates
    })
    image_dir, barcode, well = arrays[int(rng.choice(len(arrays)))]
    of_array = [
        c for c in candidates
        if c[0].parent == image_dir
        and c[1]["PamChip_barcode"] == barcode
        and c[1]["Array"] == well
    ]

    mode = str(frames_cfg.get("mode", "brightest"))
    if mode == "brightest":
        return [max(of_array, key=lambda c: (c[1]["Pump_cycle"], c[1]["Exposure_time"]))]
    if mode == "exposures":
        cycle = max(c[1]["Pump_cycle"] for c in of_array)
        of_cycle = [c for c in of_array if c[1]["Pump_cycle"] == cycle]
        return sorted(of_cycle, key=lambda c: -c[1]["Exposure_time"])
    raise SystemExit(f"ERROR: unknown frames.mode {mode!r}; use 'brightest' or 'exposures'.")


def open_loader(path: Path) -> Tuple[DataLoader, int]:
    """Load the run folder a frame belongs to and return it with the frame's index.

    NOT load_data(): that also insists on a Sample Annotation, which these tests
    have no use for -- they stop at the spot geometry and never reach the peptide
    stage where sample names matter -- and which the raw STK run does not keep in
    its run root anyway. Loading only the layout and the images is what lets an
    arbitrary run folder be pointed at without preparing it first.

    Args:
        path (Path): One TIFF inside <run>/ImageResults/.

    Returns:
        Tuple[DataLoader, int]: The loader and the index of that frame in it.
    """
    run_dir = path.parent.parent
    loader = DataLoader(
        data_dir=str(run_dir.parent.parent),
        experiment_name=run_dir.parent.name,
        subfolder=run_dir.name,
    )
    loader.load_array_layout(verbose=False)
    loader.load_images(verbose=False)
    index = [str(p) for p in loader.images.file_path.values].index(str(path))
    return loader, index


# ---------------------------------------------------------------------------
# the sweep both tests drive
# ---------------------------------------------------------------------------
def run_sweep(
    path: Path,
    meta: Dict[str, int],
    steps: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    panel_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Apply each step to one frame, detect, and collect what happened.

    The loop both tests share. A step is a dict carrying the transformation and
    how to label it:

        value          the swept quantity, the thing being varied
        angle_deg      rotation to apply (0 in the translation test)
        shift          (dx, dy) to apply (0, 0 in the rotation test)
        applied        the panel title's first line, e.g. "rotated +0.30 deg"
        stem           the panel's file stem
        series         optional label when one run sweeps several directions

    A step whose content box would leave the frame is reported as NOT TESTED and
    left out of the rows. That distinction matters: an image the transformation
    has pushed the grid out of is not a fair test, and counting it as a detection
    failure would blame the detector for the test's own limit.

    Args:
        path (Path): The frame to sweep.
        meta (Dict[str, int]): Its parsed metadata.
        steps (Sequence[Dict[str, Any]]): The steps, in the order to run them.
        cfg (Dict[str, Any]): The effective settings.
        panel_dir (Path): Where the panels go -- one for the original, one per
            step, plus the standalone legend.

    Returns:
        Optional[Dict[str, Any]]: The baseline detection, the rows, and how many
        steps were skipped -- or None if the frame could not be set up.
    """
    loader, index = open_loader(path)
    pixels = loader._images.values
    pristine = pixels[index].copy()
    shape = pristine.shape
    pivot = ((shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0)

    reference = brightest_sibling(path.parent, meta)
    box = None if reference is None else measure_content_box(reference, cfg["content_box"])
    if box is None:
        print(f"  ERROR: content box of {path.name}'s array is not measurable.")
        return None

    baseline = detect_one(loader, index)
    if not detected(baseline):
        print(f"  WARNING: {path.name} is not detected even UNTRANSFORMED. "
              "Every failure below would be meaningless, so it is skipped.")
        return None
    base_t = t_middle(baseline)

    # The original is drawn ONCE per frame, not again beside every step: every
    # figure this run writes is one panel of the same width, so the original and
    # any step can be placed side by side on the page afterwards. The legend is
    # written once too, as its own stacked column, so a page composed that way
    # does not have to repeat it under each panel.
    panel_legend = bool(cfg.get("panel_legend", True))
    if cfg.get("panels", True):
        render_panel(baseline, "original", float(cfg["integration_radius"]),
                     panel_dir, f"{path.stem}_original",
                     int(cfg.get("panel_dpi", cfg["dpi"])), panel_legend)
        render_legend(panel_dir, int(cfg.get("panel_dpi", cfg["dpi"])))

    rows: List[Dict[str, Any]] = []
    skipped = 0
    try:
        for step in steps:
            angle = float(step.get("angle_deg", 0.0))
            shift = tuple(float(v) for v in step.get("shift", (0.0, 0.0)))
            if not box_stays_in_frame(box, angle, pivot, shift, shape,
                                      float(cfg["clearance_px"])):
                skipped += 1
                print(f"  {step['applied']}: content leaves the frame, not tested")
                continue

            pixels[index] = apply_transform(
                pristine, angle, pivot, shift, int(cfg["spline_order"])
            )
            moved = detect_one(loader, index)
            ok = detected(moved)

            delta = float("nan")
            if ok and moved["median"] is not None and baseline["median"] is not None:
                delta = float(np.nanmedian(np.abs(moved["median"] - baseline["median"])))

            moved_t = t_middle(moved)
            rows.append({
                **step,
                "detected": ok,
                "grid_angle": (
                    float(moved["mask"]["angle_deg"]) if moved["mask"] else float("nan")
                ),
                # How far the array's own FIDUCIAL was found to have moved. Under
                # a translation this must equal the applied shift exactly.
                "t_dx": float("nan") if moved_t is None else moved_t[0] - base_t[0],
                "t_dy": float("nan") if moved_t is None else moved_t[1] - base_t[1],
                # How far the fitted APERTURE centre moved. This must NOT equal
                # the applied shift: the disc is already clipped by the frame in
                # the raw data, so moving it changes which part of the rim is
                # visible and with it the fitted centre. Recorded next to the
                # fiducial so the two can be told apart rather than confused.
                "ap_dx": moved["center"][0] - baseline["center"][0],
                "ap_dy": moved["center"][1] - baseline["center"][1],
                "median_abs_delta": delta,
            })

            if cfg.get("panels", True):
                render_panel(
                    moved, step["applied"], float(cfg["integration_radius"]),
                    panel_dir, step["stem"],
                    int(cfg.get("panel_dpi", cfg["dpi"])), panel_legend,
                )

            print(f"  {step['applied']}: {'detected' if ok else 'DETECTION FAILED'}"
                  + (f"   median |dI| {delta:.2f} counts" if ok else ""))
    finally:
        pixels[index] = pristine
        del loader

    return {"baseline": baseline, "rows": rows, "skipped": skipped, "box": box,
            "shape": shape, "pivot": pivot}


def admissible_shift_range(
    box: Tuple[float, float, float, float],
    shape: Tuple[int, int],
    clearance: float,
    axis: str,
) -> Tuple[float, float]:
    """How far this frame may be shifted along one axis, from its content box.

    The translation analogue of the rotation test's predicted window: derived
    from the geometry, not fitted to the outcome. Note that the boundary it
    describes is the FRAME's, not the detector's -- past it the printed content
    would leave the image, so there is nothing left to detect.

    Args:
        box (Tuple[float, float, float, float]): (x0, y0, x1, y1) of the content.
        shape (Tuple[int, int]): (height, width) of the frame.
        clearance (float): Pixels that must stay free at every edge.
        axis (str): "x" or "y".

    Returns:
        Tuple[float, float]: (most negative, most positive) shift in pixels.
    """
    x0, y0, x1, y1 = box
    height, width = shape
    if axis == "x":
        return clearance - x0, (width - 1 - clearance) - x1
    if axis == "y":
        return clearance - y0, (height - 1 - clearance) - y1
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def contiguous_window(
    rows: List[Dict[str, Any]], key: str = "value"
) -> Optional[Tuple[float, float, int]]:
    """The range of swept values that were detected, and any hole inside it.

    Args:
        rows (List[Dict[str, Any]]): The sweep's rows.
        key (str): Which field carries the swept value.

    Returns:
        Optional[Tuple[float, float, int]]: (low, high, failures inside), or None
        if nothing was detected at all.
    """
    hits = [r[key] for r in rows if r["detected"]]
    if not hits:
        return None
    low, high = min(hits), max(hits)
    inside = sum(1 for r in rows if not r["detected"] and low < r[key] < high)
    return low, high, inside
