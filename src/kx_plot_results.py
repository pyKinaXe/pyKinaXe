"""Plotting classes for pyKinaXe outputs: peptide waterfall plots, venn diagrams
and the heatmaps for pathway and peptide results.

All figures are rendered on the Agg backend and written to ``save_path``; style
defaults come from the YAML files in ``config/``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Rectangle
from matplotlib.textpath import TextPath
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

WATERFALL_CONFIG_PATH = "config/waterfall_plot_config.yaml"
HEATMAP_CONFIG_PATH = "config/heatmap_plot_config.yaml"

INDIVIDUAL_COLOR_SCALE = "individual"
FIXED_COLOR_SCALE = "fixed"
ALLOWED_COLOR_SCALES = (INDIVIDUAL_COLOR_SCALE, FIXED_COLOR_SCALE)


def _resolve_config_path(path):
    path = Path(path)
    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [repo_root / path, Path.cwd() / path, path]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _load_yaml_style_config(path):
    """Load one of the YAML plot-style configuration files.
    
    Returns:
        dict: Loaded config, or an empty mapping when the file is missing.
    """
    resolved_path = _resolve_config_path(path)
    try:
        with open(resolved_path) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"     Config not found at {resolved_path}, using defaults.")
        cfg = {}
    return cfg or {}


def _resolve_color_scale(value):
    """Validate the configured heatmap colour-scale mode.
    
    Returns:
        str: Either ``'individual'`` or ``'fixed'``.
    """
    resolved = str(value).strip().lower()
    if resolved not in ALLOWED_COLOR_SCALES:
        raise ValueError(
            f"Unknown heatmap color_scale '{value}'. "
            f"Use one of {list(ALLOWED_COLOR_SCALES)}."
        )
    return resolved


def _resolve_cmap(value, name="custom"):
    """Turn a configured colormap value into something ``sns.heatmap`` accepts.

    Accepts either a matplotlib colormap name (``"coolwarm"``) or a list of
    colours to interpolate between (``["blue", "purple", "red"]``). The list form
    exists because the standard blue-to-red maps run through a very light middle,
    which on an absolute signal reads like a missing value rather than a mid-range
    one; a hand-built ramp stays saturated all the way across.
    
    Args:
        value: Colormap name or list of colour stops.
        name: Name registered for a list-built colormap.
    
    Returns:
        object: A colormap name or a Colormap instance.
    """
    if isinstance(value, (list, tuple)):
        stops = [str(stop) for stop in value]
        if len(stops) < 2:
            raise ValueError(
                "A colormap given as a list needs at least two colour stops, "
                f"got {stops!r}."
            )
        return LinearSegmentedColormap.from_list(name, stops)
    return value


def _data_limits(matrix):
    """Return the data-driven colour limits of one heatmap matrix.
    
    The limits are the lowest and highest finite value of the matrix, so each
    plot uses its full colour range. For the signed heatmaps zero stays the
    neutral colour through the ``center=0`` argument of ``sns.heatmap``, which is
    why the limits do not have to be symmetric.
    
    Args:
        matrix: Numeric array or DataFrame holding the heatmap values.
    
    Returns:
        tuple: ``(vmin, vmax)`` for ``sns.heatmap``.
    """
    values = np.asarray(pd.DataFrame(matrix).values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        # A constant matrix would collapse the colorbar; widen it slightly so the
        # single value still renders and the colorbar keeps readable ticks.
        pad = abs(vmin) * 0.1 or 1.0
        return vmin - pad, vmax + pad
    return vmin, vmax


PATHWAY_SOURCE_NAMES = {"KEGG": "KEGG", "WP": "WikiPathways", "REAC": "Reactome"}


def _pathway_axis_label(source):
    """Return the y-axis label of a kinase-pathway heatmap for one pathway source.

    Args:
        source: g:Profiler source code (``"KEGG"``, ``"WP"``, ``"REAC"``), a
            display name, or None.

    Returns:
        str: ``"<source> pathway"``, or ``"Pathway"`` when no source is known.
    """
    if source is None or not str(source).strip():
        return "Pathway"
    return f"{PATHWAY_SOURCE_NAMES.get(str(source), str(source))} pathway"


def _parse_intersection(value):
    """Return the accessions of one g:Profiler ``intersections`` entry as a list.

    The column holds a Python list when it comes straight from g:Profiler and
    its string form (``"['P42679', 'Q13164']"``) once it has been through CSV.

    Args:
        value: List of accessions, its string representation, or NaN/None.

    Returns:
        list[str]: The accessions, stripped of whitespace and quotes.
    """
    if isinstance(value, str):
        cleaned = value.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        return [g.strip() for g in cleaned.split(",") if g.strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(g).strip() for g in value if str(g).strip()]


def _apply_uka_heatmap_style(plot, cfg):
    """Copy the kinase-pathway heatmap style of ``cfg`` onto ``plot``.

    Shared by HeatmapPlot_UKA (one comparison) and HeatmapPlot_UKA_Comparisons
    (all comparisons of a run) so both read the same YAML sections and draw
    cells, margins, colorbar and axes identically.

    Args:
        plot: Object that receives the upper-case style attributes.
        cfg: Mapping loaded from ``config/heatmap_plot_config.yaml``.
    """
    cell = cfg.get('cell', {})
    plot.CELL_HEIGHT = cell.get('height', 0.25)
    plot.CELL_WIDTH = cell.get('width', 0.3)
    plot.CELL_FONTSIZE = cell.get('fontsize', 9)
    plot.CELL_LINEWIDTH = cell.get('linewidth', 0.5)
    plot.CELL_LINECOLOR = cell.get('linecolor', 'black')

    margins = cfg.get('margins', {})
    plot.MARGIN_LEFT = margins.get('left', 4.5)
    plot.MARGIN_RIGHT = margins.get('right', 1.5)
    plot.MARGIN_TOP = margins.get('top', 1.0)
    plot.MARGIN_BOTTOM = margins.get('bottom', 2.0)

    style = cfg.get('plot', {})
    plot.CMAP = style.get('cmap', 'RdBu_r')
    plot.DPI = style.get('dpi', 300)
    plot.IMAGE_FORMAT = style.get('image_format', 'png')
    plot.TITLE_PAD = style.get('title_pad', 20)
    plot.COLOR_SCALE = _resolve_color_scale(
        style.get('color_scale', INDIVIDUAL_COLOR_SCALE)
    )

    fixed_scale = cfg.get('fixed_scale', {})
    plot.FIXED_ZSCORE_ABS_MAX = float(
        fixed_scale.get('uka_zscore_abs_max', 3.0)
    )
    plot.FIXED_VALUE_ABS_MAX = float(
        fixed_scale.get('uka_value_abs_max', 0.5)
    )

    cbar = cfg.get('colorbar', {})
    plot.CBAR_LABEL_FONTSIZE = cbar.get('label_fontsize', 12)
    plot.CBAR_WIDTH_FACTOR = cbar.get('width_factor', 2)
    plot.CBAR_HEIGHT_ROWS = cbar.get('height_rows', 20)
    plot.CBAR_OFFSET_X = cbar.get('offset_x', 0.05)

    axes = cfg.get('axes', {})
    plot.AXES_LABEL_FONTSIZE = axes.get('label_fontsize', 12)
    plot.TITLE_FONTSIZE = axes.get('title_fontsize', 14)
    plot.X_TICK_ROTATION = axes.get('x_tick_rotation', 45)
    plot.X_TICK_FONTSIZE = axes.get('x_tick_fontsize', 8)
    plot.Y_TICK_FONTSIZE = axes.get('y_tick_fontsize', 9)
    plot.X_TICK_HA = axes.get('x_tick_ha', 'right')


def _centered_colormap(cmap, vmin, vmax, center=0.0):
    """Cut ``cmap`` so ``center`` keeps its neutral colour on a [vmin, vmax] colorbar.

    This mirrors what ``sns.heatmap(center=0)`` does for the per-comparison
    heatmaps: the colour slope is the same on both sides of ``center`` and only
    the part of the map the data reaches is shown, so the shared colorbar of
    the comparison heatmaps reads exactly like the single-comparison ones.

    Args:
        cmap: Colormap name or instance.
        vmin: Lowest value of the colorbar.
        vmax: Highest value of the colorbar.
        center: Value that maps to the middle colour of ``cmap``.

    Returns:
        tuple: ``(ListedColormap, Normalize)`` to draw and to label the colorbar with.
    """
    base = colormaps.get_cmap(cmap)
    vrange = max(vmax - center, center - vmin)
    if not np.isfinite(vrange) or vrange <= 0:
        vrange = 1.0
    symmetric = Normalize(center - vrange, center + vrange)
    cmin, cmax = symmetric([vmin, vmax])
    cut = ListedColormap(base(np.linspace(cmin, cmax, 256)))
    return cut, Normalize(vmin, vmax)


class WaterfallPlot:
    """Waterfall plot of the ranked peptide log2 fold changes.

    Every peptide becomes one horizontal stem that starts at zero and ends at
    its log2 fold change, and the peptides are ordered by that value. The plot
    therefore reads as a waterfall: the strongest decrease at the bottom, the
    strongest increase at the top, and the peptides that barely move in the
    middle.

    Expects a DataFrame with one row per peptide and the columns:

    - ``ID`` (or ``peptide``): peptide identifier used as the y-axis label
    - ``peptide_change``: the log2 fold change plotted on the x-axis

    Significance is derived from the log2 fold change cutoffs, not from a
    p-value. The cutoffs are the KPEA cutoffs of the run, so the plot marks the
    same peptides the enrichment step counts:

    - ``cutoff_mode='average'``: every cutoff in ``lfc_cutoffs`` is drawn and the
      smallest one is the significance boundary (with the default cutoffs
      ``[0.2, 0.3, 0.4]`` the lines sit at +/-0.2, +/-0.3 and +/-0.4, and
      ``|log2FC| >= 0.2`` counts as significant)
    - ``cutoff_mode='primary'``: only ``primary_lfc_cutoff`` is drawn and used
    """

    ALLOWED_CUTOFF_MODES = ("average", "primary")

    def __init__(
        self,
        data=None,
        csv_path=None,
        lfc_cutoffs=(0.2, 0.3, 0.4),
        cutoff_mode="average",
        primary_lfc_cutoff=None,
        value_col="peptide_change",
        label_col=None,
        x_label="Log2 Fold Change",
        title=None,
        save_path=None,
        dpi=None,
        image_format=None,
        debugging_print=False,
    ):
        """Store the plot inputs and style settings.
        
        Args:
            data: Peptide-level DataFrame to plot.
            csv_path: Path to a CSV holding that DataFrame instead.
            lfc_cutoffs: Log2 fold change cutoffs to draw (KPEA
                ``kpea_lfc_cutoffs``).
            cutoff_mode: ``'average'`` draws every cutoff, ``'primary'`` only
                ``primary_lfc_cutoff`` (KPEA ``kpea_cutoff_mode``).
            primary_lfc_cutoff: The single cutoff used in ``'primary'`` mode;
                defaults to the smallest value in ``lfc_cutoffs``.
            value_col: Column holding the log2 fold change.
            label_col: Column holding the peptide labels; auto-detected from
                ``ID`` / ``peptide`` when omitted.
            x_label: X axis label.
            title: Plot title.
            save_path: Path the figure is written to.
            dpi: Output resolution; defaults to the configured value.
            image_format: Output format; defaults to the configured value.
            debugging_print: Whether to print additional debug information.
        """
        cfg = self.load_config()

        marker = cfg.get("marker", {})
        self.MK_SIZE_SIG = marker.get("size_significant", 18)
        self.MK_SIZE_NS = marker.get("size_not_significant", 12)
        self.MK_ALPHA_SIG = marker.get("alpha_significant", 0.9)
        self.MK_ALPHA_NS = marker.get("alpha_not_significant", 0.6)
        self.MK_COLOR_UP = marker.get("color_up", "red")
        self.MK_COLOR_DOWN = marker.get("color_down", "blue")
        self.MK_COLOR_NS = marker.get("color_not_significant", "black")

        stem = cfg.get("stem", {})
        self.STEM_COLOR = stem.get("color", "gray")
        self.STEM_LINEWIDTH = stem.get("linewidth", 0.8)
        self.STEM_ALPHA_SIG = stem.get("alpha_significant", 0.9)
        self.STEM_ALPHA_NS = stem.get("alpha_not_significant", 0.45)

        tl = cfg.get("threshold_lines", {})
        self.TL_CUTOFF_COLOR = tl.get("cutoff_color", "black")
        self.TL_CUTOFF_LINESTYLE = tl.get("cutoff_linestyle", "--")
        self.TL_CUTOFF_LINEWIDTH = tl.get("cutoff_linewidth", 1.4)
        self.TL_SECONDARY_COLOR = tl.get("secondary_cutoff_color", "black")
        self.TL_SECONDARY_LINESTYLE = tl.get("secondary_cutoff_linestyle", "--")
        self.TL_SECONDARY_LINEWIDTH = tl.get("secondary_cutoff_linewidth", 0.9)
        self.TL_SECONDARY_ALPHA = tl.get("secondary_cutoff_alpha", 0.55)
        self.TL_CENTER_COLOR = tl.get("center_color", "black")
        self.TL_CENTER_LINESTYLE = tl.get("center_linestyle", "-")
        self.TL_CENTER_LINEWIDTH = tl.get("center_linewidth", 0.9)

        ax_cfg = cfg.get("axes", {})
        self.AXES_LABEL_FONTSIZE = ax_cfg.get("label_fontsize", 12)
        self.TITLE_FONTSIZE = ax_cfg.get("title_fontsize", 14)
        self.GRID_ALPHA = ax_cfg.get("grid_alpha", 0.25)
        self.X_TICK_FONTSIZE = ax_cfg.get("x_tick_fontsize", 10)
        self.Y_TICK_FONTSIZE = ax_cfg.get("y_tick_fontsize", 5)

        pl = cfg.get("plot", {})
        self.FIG_WIDTH = pl.get("figsize_width", 7.0)
        self.ROW_HEIGHT = pl.get("row_height", 0.1)
        self.MIN_HEIGHT = pl.get("min_height", 4.0)
        self.MAX_HEIGHT = pl.get("max_height", 45.0)
        self.MARGIN_LEFT = pl.get("margin_left", 1.9)
        self.MARGIN_RIGHT = pl.get("margin_right", 0.6)
        self.MARGIN_TOP = pl.get("margin_top", 0.7)
        self.MARGIN_BOTTOM = pl.get("margin_bottom", 0.7)
        self.DPI = pl.get("dpi", 300)
        self.IMAGE_FORMAT = pl.get("image_format", "png")

        self.debugging_print = debugging_print
        self.value_col = value_col
        self.x_label = x_label
        self.title = title
        self.save_path = save_path
        self.dpi = self.DPI if dpi is None else dpi
        self.image_format = str(
            self.IMAGE_FORMAT if image_format is None else image_format
        ).lower()

        self.cutoffs, self.significance_cutoff = self._resolve_cutoffs(
            lfc_cutoffs=lfc_cutoffs,
            cutoff_mode=cutoff_mode,
            primary_lfc_cutoff=primary_lfc_cutoff,
        )
        self.cutoff_mode = str(cutoff_mode).strip().lower()

        if data is not None:
            df = data.copy()
        elif csv_path is not None:
            df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either data or csv_path must be provided")

        self.label_col = self._resolve_label_col(df, label_col)
        self.data = self._normalize_data(df)

        self._calc_fig_size()
        self.fig, self.ax = plt.subplots(
            figsize=(self._fig_width, self._fig_height)
        )
        self._draw_plot()

        if self.save_path:
            self._save_plot()

        plt.close(self.fig)

    @staticmethod
    def load_config(path=None):
        return _load_yaml_style_config(
            WATERFALL_CONFIG_PATH if path is None else path
        )

    def _dprint(self, *args, **kwargs):
        """Print only if debugging_print is enabled."""
        if self.debugging_print:
            print(*args, **kwargs)

    @classmethod
    def _resolve_cutoffs(cls, lfc_cutoffs, cutoff_mode, primary_lfc_cutoff):
        """Turn the configured KPEA cutoffs into drawn lines plus a boundary.
        
        Args:
            lfc_cutoffs: Configured log2 fold change cutoffs.
            cutoff_mode: ``'average'`` (draw all) or ``'primary'`` (draw one).
            primary_lfc_cutoff: The cutoff used in ``'primary'`` mode.
        
        Returns:
            tuple: ``(cutoffs, significance_cutoff)`` with ascending cutoffs.
        """
        mode = str(cutoff_mode).strip().lower()
        if mode not in cls.ALLOWED_CUTOFF_MODES:
            raise ValueError(
                f"Unknown cutoff_mode '{cutoff_mode}'. "
                f"Use one of {list(cls.ALLOWED_CUTOFF_MODES)}."
            )

        candidates = [float(cutoff) for cutoff in (lfc_cutoffs or ())]
        if any((not np.isfinite(cutoff)) or cutoff < 0 for cutoff in candidates):
            raise ValueError("lfc_cutoffs must contain only finite values >= 0.")
        cutoffs = tuple(sorted(set(candidates)))
        if not cutoffs:
            raise ValueError("lfc_cutoffs must contain at least one cutoff.")

        if mode == "primary":
            primary = (
                cutoffs[0]
                if primary_lfc_cutoff is None
                else float(primary_lfc_cutoff)
            )
            if (not np.isfinite(primary)) or primary < 0:
                raise ValueError("primary_lfc_cutoff must be a finite value >= 0.")
            return (primary,), primary

        # 'average' mode scores the mean across every cutoff, so every cutoff is
        # drawn; the smallest one is where a peptide starts to count.
        return cutoffs, cutoffs[0]

    def _resolve_label_col(self, df, label_col):
        """Resolve the column holding the peptide labels.
        
        Args:
            label_col: Explicitly requested column, or None.
        
        Returns:
            str: Name of the label column.
        """
        if label_col is not None:
            if label_col not in df.columns:
                raise ValueError(
                    f"Requested label column '{label_col}' is not present in the data."
                )
            return label_col

        for candidate in ("ID", "peptide"):
            if candidate in df.columns:
                return candidate

        raise ValueError("Could not find an 'ID' or 'peptide' column")

    def _normalize_data(self, df):
        """Rank the peptides and flag the significant ones.
        
        Returns:
            pd.DataFrame: One row per peptide, ordered by log2 fold change.
        """
        if self.value_col not in df.columns:
            raise ValueError(
                f"Requested value column '{self.value_col}' is not present in the data."
            )

        normalized = pd.DataFrame(
            {
                "label": df[self.label_col].astype(str),
                "value": pd.to_numeric(df[self.value_col], errors="coerce"),
            }
        )
        normalized = normalized.dropna(subset=["value"])
        normalized = normalized.drop_duplicates(subset="label", keep="first")
        if normalized.empty:
            raise ValueError(
                f"No finite '{self.value_col}' values left to plot."
            )

        # Ascending, so the y-axis grows from the strongest decrease at the
        # bottom to the strongest increase at the top.
        normalized = normalized.sort_values("value", ascending=True).reset_index(
            drop=True
        )
        normalized["_significant"] = (
            normalized["value"].abs() >= self.significance_cutoff
        )
        return normalized

    def _calc_fig_size(self):
        """Compute figure size based on the number of peptides."""
        n_rows = len(self.data)

        max_label_len = max(len(label) for label in self.data["label"])
        self._margin_left = max(self.MARGIN_LEFT, max_label_len * 0.075)

        # The configured width wins unless long peptide labels would leave less
        # than two inches for the stems themselves.
        self._fig_width = max(
            float(self.FIG_WIDTH),
            self._margin_left + self.MARGIN_RIGHT + 2.0,
        )
        raw_height = self.MARGIN_TOP + n_rows * self.ROW_HEIGHT + self.MARGIN_BOTTOM
        self._fig_height = float(
            min(max(raw_height, self.MIN_HEIGHT), self.MAX_HEIGHT)
        )

    def _point_styles(self):
        """Return per-peptide marker colors, sizes and alphas.
        
        Returns:
            tuple: Colors, sizes, marker alphas and stem alphas.
        """
        colors = []
        sizes = []
        marker_alphas = []
        stem_alphas = []

        for value, significant in zip(
            self.data["value"], self.data["_significant"]
        ):
            if not significant:
                colors.append(self.MK_COLOR_NS)
                sizes.append(self.MK_SIZE_NS)
                marker_alphas.append(self.MK_ALPHA_NS)
                stem_alphas.append(self.STEM_ALPHA_NS)
            elif value > 0:
                colors.append(self.MK_COLOR_UP)
                sizes.append(self.MK_SIZE_SIG)
                marker_alphas.append(self.MK_ALPHA_SIG)
                stem_alphas.append(self.STEM_ALPHA_SIG)
            else:
                colors.append(self.MK_COLOR_DOWN)
                sizes.append(self.MK_SIZE_SIG)
                marker_alphas.append(self.MK_ALPHA_SIG)
                stem_alphas.append(self.STEM_ALPHA_SIG)

        return colors, sizes, marker_alphas, stem_alphas

    def _draw_threshold_lines(self):
        """Draw the zero line and one dashed line per configured cutoff."""
        self.ax.axvline(
            0,
            color=self.TL_CENTER_COLOR,
            linestyle=self.TL_CENTER_LINESTYLE,
            linewidth=self.TL_CENTER_LINEWIDTH,
            zorder=1,
        )

        for cutoff in self.cutoffs:
            if cutoff <= 0:
                continue
            is_boundary = cutoff == self.significance_cutoff
            style = {
                "color": self.TL_CUTOFF_COLOR if is_boundary else self.TL_SECONDARY_COLOR,
                "linestyle": (
                    self.TL_CUTOFF_LINESTYLE
                    if is_boundary
                    else self.TL_SECONDARY_LINESTYLE
                ),
                "linewidth": (
                    self.TL_CUTOFF_LINEWIDTH
                    if is_boundary
                    else self.TL_SECONDARY_LINEWIDTH
                ),
                "alpha": 1.0 if is_boundary else self.TL_SECONDARY_ALPHA,
                "zorder": 1,
            }
            self.ax.axvline(cutoff, **style)
            self.ax.axvline(-cutoff, **style)

    def _format_axes(self):
        """Format the axes, ticks and title."""
        n_rows = len(self.data)
        self.ax.set_ylim(-0.75, n_rows - 0.25)
        self.ax.set_yticks(range(n_rows))
        self.ax.set_yticklabels(
            self.data["label"].tolist(), fontsize=self.Y_TICK_FONTSIZE
        )
        self.ax.tick_params(axis="y", length=2, pad=1.5)
        self.ax.tick_params(axis="x", labelsize=self.X_TICK_FONTSIZE)

        self.ax.set_xlabel(self.x_label, fontsize=self.AXES_LABEL_FONTSIZE)
        self.ax.set_ylabel("")
        if self.title:
            self.ax.set_title(self.title, fontsize=self.TITLE_FONTSIZE)

        self.ax.grid(axis="x", alpha=self.GRID_ALPHA)
        self.ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            self.ax.spines[side].set_visible(False)

    def _apply_fixed_layout(self):
        """Apply the layout using fixed margins."""
        self.fig.subplots_adjust(
            left=self._margin_left / self._fig_width,
            right=1.0 - self.MARGIN_RIGHT / self._fig_width,
            top=1.0 - self.MARGIN_TOP / self._fig_height,
            bottom=self.MARGIN_BOTTOM / self._fig_height,
        )

    def _draw_plot(self):
        """Draw the waterfall."""
        colors, sizes, marker_alphas, stem_alphas = self._point_styles()
        positions = np.arange(len(self.data))
        values = self.data["value"].to_numpy(dtype=float)

        for y, value, stem_alpha in zip(positions, values, stem_alphas):
            self.ax.hlines(
                y,
                0,
                value,
                color=self.STEM_COLOR,
                linewidth=self.STEM_LINEWIDTH,
                alpha=stem_alpha,
                zorder=2,
            )
        self.ax.scatter(
            values,
            positions,
            c=colors,
            s=sizes,
            alpha=marker_alphas,
            linewidths=0,
            zorder=3,
        )

        self._draw_threshold_lines()
        self._format_axes()
        self._apply_fixed_layout()

        n_sig = int(self.data["_significant"].sum())
        n_up = int((self.data["_significant"] & (self.data["value"] > 0)).sum())
        n_down = int((self.data["_significant"] & (self.data["value"] < 0)).sum())
        self._dprint(
            f"     Waterfall: {len(self.data)} peptides | significant: {n_sig} "
            f"(up: {n_up}, down: {n_down}) | cutoff mode: {self.cutoff_mode} | "
            f"cutoffs: {[f'{c:g}' for c in self.cutoffs]}"
        )

    def _save_plot(self):
        """Save the waterfall figure."""
        try:
            save_path = Path(self.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not save_path.suffix:
                save_path = save_path.with_suffix(f".{self.image_format}")
            self.fig.savefig(
                save_path,
                dpi=self.dpi,
                bbox_inches="tight",
                format=self.image_format,
            )
            self._dprint(f"     Waterfall plot saved to: {save_path.absolute()}")
        except Exception as e:
            print(f"Error saving waterfall plot: {e}")
            raise


class VennDiagramPlot:
    """Plot overlaps between named item lists.

    The class accepts any named groups of items, for example kinase IDs or
    pathway IDs. For one to three groups it draws a circle-based Venn diagram.
    For more than three groups it draws an UpSet-style intersection plot,
    because a true Venn diagram becomes hard to read and ambiguous.
    """

    DEFAULT_COLORS = (
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
    )

    THREE_GROUP_COMPARISON_COLORS = (
        "#1F3A5F",
        "#B13A3A",
        "#4A4A4A",
    )

    def __init__(
        self,
        groups: Mapping[str, Iterable] | Sequence[tuple[str, Iterable]],
        title: str | None = None,
        item_label: str = "items",
        save_path: str | Path | None = None,
        save_tables_dir: str | Path | None = None,
        plot_type: str = "auto",
        case_sensitive: bool = True,
        strip_items: bool = True,
        show_percent: bool = False,
        max_upset_intersections: int = 40,
        figsize: tuple[float, float] | None = None,
        dpi: int = 300,
        image_format: str = "png",
        colors: Sequence[str] | None = None,
        debugging_print: bool = False,
    ):
        """Store the groups and the drawing options.
        
        Args:
            debugging_print (bool): Whether to print additional debug information.
        """
        self.title = title
        self.item_label = item_label
        self.save_path = Path(save_path) if save_path is not None else None
        self.save_tables_dir = (
            Path(save_tables_dir) if save_tables_dir is not None else None
        )
        self.plot_type = str(plot_type).lower()
        self.case_sensitive = case_sensitive
        self.strip_items = strip_items
        self.show_percent = show_percent
        self.max_upset_intersections = int(max_upset_intersections)
        self.figsize = figsize
        self.dpi = int(dpi)
        self.image_format = image_format.lower()
        self.colors = tuple(colors) if colors is not None else None
        self.debugging_print = debugging_print

        self.group_sets = self._normalize_groups(groups)
        self.group_names = list(self.group_sets)
        self.n_groups = len(self.group_names)
        if self.n_groups == 0:
            raise ValueError("At least one group is required.")
        if self.colors is None:
            self.colors = (
                self.THREE_GROUP_COMPARISON_COLORS
                if self.n_groups == 3
                else self.DEFAULT_COLORS
            )

        self.universe = set().union(*self.group_sets.values())
        self.region_sets = self._compute_exact_regions()
        self.fig = None
        self.ax = None

    @classmethod
    def from_dataframes(
        cls,
        dataframes: Mapping[str, pd.DataFrame],
        column: str,
        **kwargs,
    ) -> "VennDiagramPlot":
        """Create a plot from one column in multiple DataFrames."""
        groups = {}
        for name, df in dataframes.items():
            if column not in df.columns:
                raise ValueError(f"Column '{column}' is missing in group '{name}'.")
            groups[name] = df[column].dropna().tolist()
        return cls(groups=groups, **kwargs)

    def _dprint(self, message: str) -> None:
        """Print a message only when debug logging is enabled.
        
        Args:
            message (str): Status or log message to record.
        """
        if self.debugging_print:
            print(message)

    def _normalize_groups(
        self,
        groups: Mapping[str, Iterable] | Sequence[tuple[str, Iterable]],
    ) -> dict[str, set[str]]:
        if isinstance(groups, Mapping):
            raw_groups = list(groups.items())
        else:
            raw_groups = list(groups)

        normalized = {}
        for name, values in raw_groups:
            label = str(name).strip()
            if not label:
                raise ValueError("Group names must not be empty.")
            if label in normalized:
                raise ValueError(f"Duplicate group name: {label}")
            normalized[label] = self._normalize_item_set(values)
        return normalized

    def _normalize_item_set(self, values: Iterable) -> set[str]:
        """Normalize item set.

        Args:
            values (Iterable): Collection of input values processed by this helper.
        """
        if values is None:
            return set()

        items = set()
        for value in values:
            item = self._normalize_item(value)
            if item is not None:
                items.add(item)
        return items

    def _normalize_item(self, value) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        item = str(value)
        if self.strip_items:
            item = item.strip()
        if not item or item.lower() in {"nan", "none"}:
            return None
        if not self.case_sensitive:
            item = item.upper()
        return item

    def _compute_exact_regions(self) -> dict[tuple[str, ...], set[str]]:
        regions = {}
        names = self.group_names
        for size in range(1, len(names) + 1):
            for combo in combinations(names, size):
                selected_sets = [self.group_sets[name] for name in combo]
                selected = set.intersection(*selected_sets) if selected_sets else set()
                excluded_names = [name for name in names if name not in combo]
                excluded = (
                    set().union(*(self.group_sets[name] for name in excluded_names))
                    if excluded_names
                    else set()
                )
                regions[combo] = selected - excluded
        return regions

    def _region_count(self, *names: str) -> int:
        return len(self.region_sets.get(tuple(names), set()))

    def _format_count(self, count: int) -> str:
        if not self.show_percent:
            return str(count)
        total = len(self.universe)
        percent = 0 if total == 0 else 100 * count / total
        return f"{count}\n({percent:.1f}%)"

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for name in self.group_names:
            rows.append(
                {
                    "Group": name,
                    "N": len(self.group_sets[name]),
                    "Fraction_of_union": (
                        0.0 if not self.universe else len(self.group_sets[name]) / len(self.universe)
                    ),
                }
            )
        return pd.DataFrame(rows)

    def pairwise_overlap_table(self) -> pd.DataFrame:
        rows = []
        for left, right in combinations(self.group_names, 2):
            left_set = self.group_sets[left]
            right_set = self.group_sets[right]
            overlap = left_set & right_set
            union = left_set | right_set
            rows.append(
                {
                    "Group_A": left,
                    "Group_B": right,
                    "N_A": len(left_set),
                    "N_B": len(right_set),
                    "Overlap": len(overlap),
                    "Union": len(union),
                    "Jaccard": 0.0 if not union else len(overlap) / len(union),
                    "Overlap_Items": ";".join(sorted(overlap)),
                }
            )
        return pd.DataFrame(rows)

    def membership_table(self) -> pd.DataFrame:
        rows = []
        for item in sorted(self.universe):
            row = {"Item": item}
            for name in self.group_names:
                row[name] = item in self.group_sets[name]
            row["N_Groups"] = sum(bool(row[name]) for name in self.group_names)
            row["Groups"] = ";".join(
                name for name in self.group_names if row[name]
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def region_table(self, include_items: bool = True) -> pd.DataFrame:
        rows = []
        for combo, items in sorted(
            self.region_sets.items(),
            key=lambda pair: (-len(pair[1]), len(pair[0]), pair[0]),
        ):
            row = {
                "Region": " & ".join(combo),
                "Included_Groups": ";".join(combo),
                "Excluded_Groups": ";".join(
                    name for name in self.group_names if name not in combo
                ),
                "N": len(items),
                "Mask": "".join("1" if name in combo else "0" for name in self.group_names),
            }
            if include_items:
                row["Items"] = ";".join(sorted(items))
            rows.append(row)
        return pd.DataFrame(rows)

    def save_tables(self, output_dir: str | Path | None = None) -> None:
        target_dir = output_dir if output_dir is not None else self.save_tables_dir
        if target_dir is None:
            raise ValueError("No output directory provided for overlap tables.")
        output_dir = Path(target_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.summary_table().to_csv(output_dir / "venn_group_summary.csv", index=False)
        self.region_table(include_items=True).to_csv(
            output_dir / "venn_exact_regions.csv",
            index=False,
        )
        self.membership_table().to_csv(output_dir / "venn_membership_table.csv", index=False)
        self.pairwise_overlap_table().to_csv(
            output_dir / "venn_pairwise_overlaps.csv",
            index=False,
        )

    def plot(self):
        """Handle plot."""
        plot_type = self.plot_type
        if plot_type == "auto":
            plot_type = "venn" if self.n_groups <= 3 else "upset"

        if plot_type == "venn":
            if self.n_groups > 3:
                self._dprint(
                    "More than three groups requested; using an UpSet-style plot."
                )
                fig = self._plot_upset()
            else:
                fig = self._plot_venn()
        elif plot_type in {"upset", "intersection"}:
            fig = self._plot_upset()
        else:
            raise ValueError("plot_type must be 'auto', 'venn', or 'upset'.")

        if self.save_tables_dir is not None:
            self.save_tables(self.save_tables_dir)
        if self.save_path is not None:
            self._save_figure(fig)
        return fig

    def _save_figure(self, fig) -> None:
        path = self.save_path
        if path.suffix == "":
            path = path.with_suffix(f".{self.image_format}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", format=path.suffix.lstrip("."))
        self._dprint(f"Saved Venn diagram: {path}")

    def _plot_venn(self):
        if self.n_groups == 1:
            return self._plot_venn_1()
        if self.n_groups == 2:
            return self._plot_venn_2()
        if self.n_groups == 3:
            return self._plot_venn_3()
        raise ValueError("Circle Venn plots support one to three groups.")

    def _new_venn_figure(self, default_size: tuple[float, float]):
        fig, ax = plt.subplots(figsize=self.figsize or default_size)
        ax.set_aspect("equal")
        ax.axis("off")
        if self.title:
            ax.set_title(self.title, fontsize=14, pad=16)
        self.fig = fig
        self.ax = ax
        return fig, ax

    def _wrapped_label(self, name: str) -> str:
        return textwrap.fill(f"{name}\n(n={len(self.group_sets[name])})", width=24)

    def _plot_venn_1(self):
        name = self.group_names[0]
        fig, ax = self._new_venn_figure((5.0, 4.5))
        ax.add_patch(Circle((0, 0), 1.0, color=self.colors[0], alpha=0.42, lw=2))
        ax.text(0, 0, self._format_count(len(self.group_sets[name])), ha="center", va="center", fontsize=16, weight="bold")
        ax.text(0, 1.25, self._wrapped_label(name), ha="center", va="bottom", fontsize=11)
        ax.text(0, -1.35, f"Union: {len(self.universe)} {self.item_label}", ha="center", fontsize=10)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.5, 1.55)
        return fig

    def _plot_venn_2(self):
        a, b = self.group_names
        fig, ax = self._new_venn_figure((6.2, 4.8))
        circles = [
            ((-0.45, 0), self.colors[0], a),
            ((0.45, 0), self.colors[1], b),
        ]
        for center, color, _name in circles:
            ax.add_patch(Circle(center, 0.85, color=color, alpha=0.42, lw=2))

        ax.text(-0.75, 0, self._format_count(self._region_count(a)), ha="center", va="center", fontsize=14, weight="bold")
        ax.text(0.75, 0, self._format_count(self._region_count(b)), ha="center", va="center", fontsize=14, weight="bold")
        ax.text(0, 0, self._format_count(self._region_count(a, b)), ha="center", va="center", fontsize=14, weight="bold")
        ax.text(-0.72, 1.05, self._wrapped_label(a), ha="center", va="bottom", fontsize=11)
        ax.text(0.72, 1.05, self._wrapped_label(b), ha="center", va="bottom", fontsize=11)
        ax.text(0, -1.18, f"Union: {len(self.universe)} {self.item_label}", ha="center", fontsize=10)
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.35, 1.45)
        return fig

    def _plot_venn_3(self):
        a, b, c = self.group_names
        fig, ax = self._new_venn_figure((7.0, 5.8))
        circle_specs = [
            ((-0.45, 0.22), self.colors[0], a),
            ((0.45, 0.22), self.colors[1], b),
            ((0.0, -0.42), self.colors[2], c),
        ]
        for center, color, _name in circle_specs:
            ax.add_patch(Circle(center, 0.82, color=color, alpha=0.42, lw=2))

        count_positions = {
            (a,): (-0.73, 0.35),
            (b,): (0.73, 0.35),
            (c,): (0.0, -0.75),
            (a, b): (0.0, 0.58),
            (a, c): (-0.47, -0.18),
            (b, c): (0.47, -0.18),
            (a, b, c): (0.0, 0.03),
        }
        for combo, position in count_positions.items():
            ax.text(
                *position,
                self._format_count(self._region_count(*combo)),
                ha="center",
                va="center",
                fontsize=13,
                weight="bold",
            )

        ax.text(-0.82, 1.15, self._wrapped_label(a), ha="center", va="bottom", fontsize=11)
        ax.text(0.82, 1.15, self._wrapped_label(b), ha="center", va="bottom", fontsize=11)
        ax.text(0.0, -1.48, self._wrapped_label(c), ha="center", va="top", fontsize=11)
        ax.text(0, -1.72, f"Union: {len(self.universe)} {self.item_label}", ha="center", fontsize=10)
        ax.set_xlim(-1.65, 1.65)
        ax.set_ylim(-1.85, 1.55)
        return fig

    def _plot_upset(self):
        non_empty = [
            (combo, items)
            for combo, items in self.region_sets.items()
            if len(items) > 0
        ]
        non_empty.sort(key=lambda pair: (-len(pair[1]), -len(pair[0]), pair[0]))
        non_empty = non_empty[: self.max_upset_intersections]

        if not non_empty:
            non_empty = [(tuple(), set())]

        n_intersections = len(non_empty)
        width = max(7.5, n_intersections * 0.42)
        height = max(5.0, 2.8 + self.n_groups * 0.35)
        fig = plt.figure(figsize=self.figsize or (width, height))
        grid = fig.add_gridspec(2, 1, height_ratios=[3.0, max(1.3, self.n_groups * 0.35)], hspace=0.05)
        ax_bar = fig.add_subplot(grid[0])
        ax_matrix = fig.add_subplot(grid[1], sharex=ax_bar)

        x_positions = np.arange(n_intersections)
        counts = [len(items) for _, items in non_empty]
        ax_bar.bar(x_positions, counts, color="#4C78A8", alpha=0.85)
        for x, count in zip(x_positions, counts):
            ax_bar.text(x, count, str(count), ha="center", va="bottom", fontsize=8)
        ax_bar.set_ylabel(f"{self.item_label} in exact overlap")
        ax_bar.grid(axis="y", alpha=0.25)
        if self.title:
            ax_bar.set_title(self.title, fontsize=14, pad=12)

        for y_idx, name in enumerate(reversed(self.group_names)):
            y = y_idx
            ax_matrix.scatter(
                x_positions,
                np.full(n_intersections, y),
                s=28,
                color="#DDDDDD",
                zorder=1,
            )
            active_x = [
                x
                for x, (combo, _items) in enumerate(non_empty)
                if name in combo
            ]
            if active_x:
                ax_matrix.scatter(
                    active_x,
                    np.full(len(active_x), y),
                    s=38,
                    color="#333333",
                    zorder=3,
                )

        for x, (combo, _items) in enumerate(non_empty):
            active_y = [
                self.n_groups - 1 - self.group_names.index(name)
                for name in combo
            ]
            if len(active_y) > 1:
                ax_matrix.plot(
                    [x, x],
                    [min(active_y), max(active_y)],
                    color="#333333",
                    lw=1.4,
                    zorder=2,
                )

        y_labels = [
            f"{name} (n={len(self.group_sets[name])})"
            for name in reversed(self.group_names)
        ]
        ax_matrix.set_yticks(range(self.n_groups))
        ax_matrix.set_yticklabels(y_labels)
        ax_matrix.set_xlabel("Exact overlap regions")
        ax_matrix.set_xlim(-0.6, n_intersections - 0.4)
        ax_matrix.set_ylim(-0.6, self.n_groups - 0.4)
        ax_matrix.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_matrix.grid(axis="x", alpha=0.15)

        self.fig = fig
        self.ax = ax_bar
        return fig



class HeatmapPlot_UKA:
    """
    Heatmap visualization for UKA (upstream kinase analysis) pathway enrichment.
    """

    def __init__(self,
                 enrichment_data=None,
                 results_data=None,
                 y_axis='z_score',
                 value_col=None,
                 value_label=None,
                 delta_threshold=0,
                 p_threshold=0.05,
                 z_threshold=1.96,
                 save_path=None,
                 dpi=300,
                 image_format='png',
                 data_source=None,
                 title=None,
                 debugging_print=False):

        """Load the heatmap configuration and store the plot inputs.
        
        Args:
            data_source: Pathway source code (``"KEGG"``, ``"WP"``, ``"REAC"``)
                named in the y-axis label.
            title: Figure title, normally the comparison (``"mock vs pSHDAg"``);
                None or ``""`` draws no title.
            debugging_print: Whether to print additional debug information.
        """
        cfg = self.load_config()
        _apply_uka_heatmap_style(self, cfg)

        self.enrichment_data = enrichment_data
        self.y_axis = y_axis.lower()
        self.value_col = value_col
        self.value_label = value_label
        self.delta_threshold = delta_threshold
        self.p_threshold = p_threshold
        self.z_threshold = z_threshold
        self.save_path = save_path
        self.dpi = dpi
        self.image_format = image_format.lower()
        self.data_source = data_source
        self.title = title
        self.debugging_print = debugging_print

        self.results_data = results_data.copy()
        self.label_col = 'Kinase'
        if self.value_col is None:
            if self.y_axis == 'z_score':
                self.value_col = 'Z_Score'
            elif 'KinaseStatistic' in self.results_data.columns:
                self.value_col = 'KinaseStatistic'
            elif 'MeanPeptideStatistic' in self.results_data.columns:
                self.value_col = 'MeanPeptideStatistic'
            elif 'MedianPeptideStatistic' in self.results_data.columns:
                self.value_col = 'MedianPeptideStatistic'
            else:
                self.value_col = 'Delta'
        if self.value_label is None:
            self.value_label = 'Z-Score' if self.value_col == 'Z_Score' else self.value_col

        self._prepare_data()
        self._calc_fig_size()
        self.fig, self.ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
        self._draw_heatmap()

        if self.save_path:
            self._save_plot()

        plt.close(self.fig)

    @staticmethod
    def load_config(path=None):
        return _load_yaml_style_config(
            HEATMAP_CONFIG_PATH if path is None else path
        )
    
    def _dprint(self, *args, **kwargs):
        """Print only if debugging_print is enabled."""
        if self.debugging_print:
            print(*args, **kwargs)

    def _calc_fig_size(self):
        n_rows = len(self.pathways)
        n_cols = len(self.kinases)

        max_label_len = max(len(p) for p in self.pathways)
        self._margin_left = max(self.MARGIN_LEFT, max_label_len * 0.075)

        self._fig_width = self._margin_left + n_cols * self.CELL_WIDTH + self.MARGIN_RIGHT
        self._fig_height = self.MARGIN_TOP + n_rows * self.CELL_HEIGHT + self.MARGIN_BOTTOM

    def _prepare_data(self):
        # One row per (kinase, array): the same UniProt accession can be scored on
        # BOTH the PTK and the STK array, and those are two independent
        # measurements. Both become their own column; only the column LABEL is
        # disambiguated with the array. Matching against the pathway
        # intersections always uses the bare accession, because that is what
        # g:Profiler was queried with (kx_pathway_enrichment_analysis.
        # _extract_ranked_kinase_list de-duplicates the query on "Kinase").
        has_type = 'Type' in self.results_data.columns
        accession_counts = self.results_data[self.label_col].astype(str).value_counts()

        value_dict = {}
        self._column_accessions = {}
        for _, row in self.results_data.iterrows():
            accession = str(row[self.label_col])
            value = row[self.value_col]
            if has_type and accession_counts.get(accession, 0) > 1:
                column = f"{accession} ({row['Type']})"
            else:
                column = accession
            if column in value_dict:
                previous = value_dict[column]
                new_rank = abs(value) if pd.notna(value) else float('-inf')
                old_rank = abs(previous) if pd.notna(previous) else float('-inf')
                if new_rank <= old_rank:
                    continue
            value_dict[column] = value
            self._column_accessions[column] = accession

        kinases_with_val = sorted(value_dict.items(), key=lambda x: x[1])
        self.kinases = [k for k, _ in kinases_with_val]

        pathway_kinases = {}
        for _, row in self.enrichment_data.iterrows():
            pathway_name = row['name']
            pathway_kinases[pathway_name] = _parse_intersection(row['intersections'])

        self.pathway_names = list(pathway_kinases.keys())

        kinases_in_pathways = set()
        for genes in pathway_kinases.values():
            kinases_in_pathways.update(genes)
        self.kinases = [
            k for k in self.kinases
            if self._column_accessions[k] in kinases_in_pathways
        ]

        all_row = [value_dict.get(k, 0) for k in self.kinases]

        matrix = [all_row]
        for pathway in self.pathway_names:
            row_vals = []
            pw_kinases = pathway_kinases[pathway]
            for kinase in self.kinases:
                if self._column_accessions[kinase] in pw_kinases:
                    row_vals.append(value_dict.get(kinase, 0))
                else:
                    row_vals.append(np.nan)
            matrix.append(row_vals)

        self.pathways = ['All Kinases'] + self.pathway_names
        self.heatmap_data = pd.DataFrame(matrix, index=self.pathways, columns=self.kinases)

        n_pathways = len(self.pathways)
        n_kinases = len(self.kinases)
        n_filled = int((~self.heatmap_data.isna()).sum().sum())

        self._dprint(
            f"Value: {self.value_col} | "
            f"Pathways: {n_pathways} | Kinases: {n_kinases} | "
            f"Filled cells: {n_filled}"
        )

    def _get_heatmap_params(self):
        if self.COLOR_SCALE == FIXED_COLOR_SCALE:
            abs_max = (
                self.FIXED_ZSCORE_ABS_MAX
                if self.value_col == 'Z_Score'
                else self.FIXED_VALUE_ABS_MAX
            )
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = _data_limits(self.heatmap_data)

        return {
            'cmap': self.CMAP,
            'center': 0,
            'cbar_kws': {'label': self.value_label},
            'linewidths': self.CELL_LINEWIDTH,
            'linecolor': self.CELL_LINECOLOR,
            'vmin': vmin,
            'vmax': vmax,
            'cbar': False,
            'mask': False,
        }

    def _apply_fixed_layout(self):
        self.fig.subplots_adjust(
            left=self._margin_left / self._fig_width,
            right=1.0 - self.MARGIN_RIGHT / self._fig_width,
            top=1.0 - self.MARGIN_TOP / self._fig_height,
            bottom=self.MARGIN_BOTTOM / self._fig_height
        )

    def _format_axes(self):
        self.ax.set_xlabel(
            f'Kinase (ordered by {self.value_label}, low to high)',
            fontsize=self.AXES_LABEL_FONTSIZE,
        )
        self.ax.set_ylabel(_pathway_axis_label(self.data_source), fontsize=self.AXES_LABEL_FONTSIZE)
        # The title names the comparison; what is plotted is already said by the
        # colorbar label and the axis labels.
        if self.title:
            self.ax.set_title(self.title, fontsize=self.TITLE_FONTSIZE, pad=self.TITLE_PAD)
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=self.X_TICK_ROTATION, ha=self.X_TICK_HA, fontsize=self.X_TICK_FONTSIZE)
        self.ax.set_yticklabels(self.ax.get_yticklabels(), rotation=0, fontsize=self.Y_TICK_FONTSIZE)
        self._apply_fixed_layout()

    def _add_colorbar(self, heatmap_obj):
        cbar_width_inches = self.CBAR_WIDTH_FACTOR * self.CELL_WIDTH
        n_rows = len(self.pathways)
        cbar_rows = n_rows if n_rows < self.CBAR_HEIGHT_ROWS else self.CBAR_HEIGHT_ROWS
        cbar_height_inches = cbar_rows * self.CELL_HEIGHT

        ax_pos = self.ax.get_position()
        cbar_width_fig = cbar_width_inches / self._fig_width
        cbar_height_fig = cbar_height_inches / self._fig_height

        cbar_x = ax_pos.x1 + self.CBAR_OFFSET_X
        cbar_y = ax_pos.y0 + (ax_pos.height - cbar_height_fig) / 2

        cbar_ax = self.fig.add_axes([cbar_x, cbar_y, cbar_width_fig, cbar_height_fig])

        cbar = self.fig.colorbar(heatmap_obj.collections[0], cax=cbar_ax)
        cbar.set_label(self.value_label, fontsize=self.CBAR_LABEL_FONTSIZE)

    def _draw_heatmap(self):
        self.ax.clear()
        params = self._get_heatmap_params()
        heatmap_obj = sns.heatmap(self.heatmap_data, ax=self.ax, **params)
        self._format_axes()
        self._add_colorbar(heatmap_obj)

    def _save_plot(self):
        try:
            save_path = Path(self.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not save_path.suffix:
                save_path = save_path.with_suffix(f'.{self.image_format}')
            self.fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
            self._dprint(f"Heatmap saved to: {save_path.absolute()}")
        except Exception as e:
            print(f"Error saving heatmap: {e}")


@dataclass
class UKAComparison:
    """Inputs of ONE control/test comparison for HeatmapPlot_UKA_Comparisons.

    Attributes:
        label: Name of the comparison in panel titles and the slice key,
            e.g. ``"mock vs pSHDAg"``.
        results_data: Significant-kinase table with ``Kinase``, the plotted
            value column and, when both arrays were measured, ``Type``.
        enrichment_data: Pathway table with ``name`` and ``intersections``;
            ``p_value`` is used for the row order when present.
        condition: Test condition (sample name). With a control shared by every
            comparison it becomes the short slice label.
        control: Control condition (sample name) or None.
    """

    label: str
    results_data: pd.DataFrame
    enrichment_data: pd.DataFrame
    condition: str | None = None
    control: str | None = None


class HeatmapPlot_UKA_Comparisons:
    """One kinase-pathway heatmap for ALL control/test comparisons of a run.

    One axis carries the union of the enriched pathways, the other the union of
    the significant kinases that fall into at least one of them, so every
    comparison is drawn on the same axes and reads off the same colorbar.
    Two layouts:

    ``split``
        A single panel. Every cell is divided into one slice per comparison,
        side by side (``split_orientation: vertical``) or stacked
        (``horizontal``); the slice order is the comparison order and a key
        next to the colorbar spells it out.
    ``panels``
        One panel per comparison, side by side, sharing the axes, the row and
        column order and one colorbar.

    In both layouts a slice/cell is

    * coloured -- the kinase is significant in that comparison, the pathway is
      enriched in it and the kinase is one of the pathway's hits;
    * white -- kinase and pathway are both present in that comparison, but the
      kinase is not a member of the pathway (as in the single heatmaps);
    * grey -- the kinase is not significant or the pathway is not enriched in
      that comparison, i.e. the cell a per-comparison heatmap would not have.

    The ``All Kinases`` row/column shows every kinase's value in each comparison
    it is significant in, which makes the kinase-level overlap readable on its own.

    ``transpose`` swaps the axes: pathways become the columns and kinases the
    rows, which turns the figure into a kinase-by-kinase comparison.
    ``target_width`` fits the figure into a fixed width in inches (a journal's
    text width): the CELLS are squeezed, never the type. Font sizes stay as
    configured, the margins are measured from the labels that actually appear,
    long row labels are wrapped and the tick rotation and the position of the
    colorbar/key follow from the remaining room.
    """

    SPLIT_LAYOUT = "split"
    PANELS_LAYOUT = "panels"
    LAYOUTS = (SPLIT_LAYOUT, PANELS_LAYOUT)
    ORIENTATIONS = ("vertical", "horizontal")
    KINASE_ORDERS = ("value", "presence")
    LEGEND_POSITIONS = ("auto", "right", "bottom")
    PANEL_DIRECTIONS = ("horizontal", "vertical")
    PANEL_ROW_MODES = ("present", "shared")
    ALL_KINASES_ROW = "All Kinases"

    # Text metrics in inches per point. Label widths are measured on the real
    # glyphs (see _measure); these are the fallback and the line height, and
    # they size the margins, so a figure with a target width really has it.
    _CHAR_WIDTH_PER_PT = 0.0085
    _LINE_HEIGHT_PER_PT = 0.0187
    _PAD = 0.12
    # Share of a target width 'auto' wrapping gives the row labels.
    _ROW_LABEL_FRACTION = 0.34
    # Measured label widths, keyed by (text, fontsize).
    _TEXT_WIDTH_CACHE = {}
    # Gap left between neighbouring upright tick labels, and the extra column
    # width 45-degree labels need over upright ones (1/sin 45).
    _LABEL_CLEARANCE = 0.02
    _DIAGONAL_PITCH_FACTOR = 1.42
    # Geometry of the slice key, in inches.
    _KEY_SLICE_WIDTH = 0.28
    _KEY_CELL_HEIGHT = 0.42
    _KEY_BAND_HEIGHT = 0.2
    _KEY_BAND_WIDTH = 0.6
    _KEY_STEP = 0.2
    _KEY_SWATCH = 0.18
    _KEY_PAD = 0.1
    _KEY_BLOCK_GAP = 0.3
    # Room the colorbar tick labels need next to the bar itself.
    _CBAR_TICK_ROOM = 0.95
    # Thickness of the horizontal colorbar under the plot, and the shortest
    # it may get before the key moves onto its own row.
    _CBAR_BAR_THICKNESS = 0.18
    _CBAR_MIN_LENGTH = 1.4
    # Thinnest rule that still prints, and the thinnest slice worth separating.
    _MIN_RULE_WIDTH = 0.15
    _MIN_SEPARATED_SLICE = 0.05

    def __init__(
        self,
        comparisons,
        value_col,
        value_label=None,
        layout=SPLIT_LAYOUT,
        panel_direction=None,
        save_path=None,
        data_source=None,
        title=None,
        transpose=None,
        target_width=None,
        target_height=None,
        orientation=None,
        debugging_print=False,
    ):
        """Load the heatmap configuration, build the union matrices and draw.

        Args:
            comparisons: Iterable of UKAComparison, in the order the slices /
                panels should appear. At least two.
            value_col: Column of ``results_data`` that colours the cells.
            value_label: Colorbar label; defaults to the column name.
            layout: ``"split"`` or ``"panels"``.
            panel_direction: How the panels of the ``panels`` layout are
                arranged: ``"horizontal"`` side by side, each with its own
                column axis, or ``"vertical"`` stacked downwards sharing one.
                None takes ``comparison_heatmap.panel_direction``.
            save_path: Output file; drawn but not saved when None.
            data_source: Pathway source code for the pathway axis label.
            title: Figure title. None picks a default for the split layout
                (``"<control> vs <cond1> / <cond2> ..."`` when every comparison
                shares one control) and no title for the panels layout; ``""``
                suppresses it.
            transpose: Kinases on the y axis and pathways on the x axis;
                None takes ``comparison_heatmap.transpose`` from the config.
            target_width: Figure width in inches the cells are squeezed into;
                None takes ``comparison_heatmap.target_width`` from the config,
                where null lets the figure grow with the number of columns.
            target_height: The same for the height, squeezing the rows down to
                the line height of their labels.
            orientation: Overrides ``comparison_heatmap.split_orientation``.
            debugging_print: Whether to print additional debug information.
        """
        cfg = self.load_config()
        _apply_uka_heatmap_style(self, cfg)

        section = cfg.get("comparison_heatmap", {}) or {}
        self.SPLIT_ORIENTATION = self._resolve_choice(
            orientation if orientation is not None
            else section.get("split_orientation", "vertical"),
            self.ORIENTATIONS,
            "split_orientation",
        )
        self.SLICE_MIN_SIZE = float(section.get("slice_min_size", 0.12))
        min_cell_width = section.get("min_cell_width", "auto")
        self.MIN_CELL_WIDTH = (
            None
            if min_cell_width is None or str(min_cell_width).strip().lower() == "auto"
            else float(min_cell_width)
        )
        min_cell_height = section.get("min_cell_height")
        self.MIN_CELL_HEIGHT = None if min_cell_height is None else float(min_cell_height)
        self.ABSENT_COLOR = section.get("absent_color", "0.8")
        self.SLICE_LINEWIDTH = float(section.get("slice_linewidth", 0.3))
        self.SLICE_LINECOLOR = section.get("slice_linecolor", "white")
        self.PANEL_GAP = float(section.get("panel_gap", 0.4))
        self.KINASE_ORDER = self._resolve_choice(
            section.get("kinase_order", "value"), self.KINASE_ORDERS, "kinase_order"
        )
        self.KEY_FONTSIZE = section.get("key_fontsize", 9)
        self.PANEL_DIRECTION = self._resolve_choice(
            section.get("panel_direction", "vertical"),
            self.PANEL_DIRECTIONS,
            "panel_direction",
        )
        self.PANEL_ROWS = self._resolve_choice(
            section.get("panel_rows", "present"), self.PANEL_ROW_MODES, "panel_rows"
        )
        self.ROW_LABEL_WRAP = section.get("row_label_wrap")
        self.LABEL_MAX_CHARS = section.get("label_max_chars")
        self.LEGEND_POSITION = self._resolve_choice(
            section.get("legend_position", "auto"), self.LEGEND_POSITIONS, "legend_position"
        )
        self.TICK_ROTATION = section.get("tick_rotation", "auto")
        self.transpose = bool(
            section.get("transpose", False) if transpose is None else transpose
        )
        configured_width = section.get("target_width") if target_width is None else target_width
        self.target_width = None if configured_width is None else float(configured_width)
        configured_height = (
            section.get("target_height") if target_height is None else target_height
        )
        self.target_height = None if configured_height is None else float(configured_height)

        self.comparisons = list(comparisons)
        if len(self.comparisons) < 2:
            raise ValueError(
                "HeatmapPlot_UKA_Comparisons needs at least two comparisons, "
                f"got {len(self.comparisons)}."
            )
        self.layout = self._resolve_choice(layout, self.LAYOUTS, "layout")
        if panel_direction is not None:
            self.PANEL_DIRECTION = self._resolve_choice(
                panel_direction, self.PANEL_DIRECTIONS, "panel_direction"
            )
        self.value_col = value_col
        self.value_label = (
            value_label
            if value_label is not None
            else ("Z-Score" if value_col == "Z_Score" else value_col)
        )
        self.save_path = save_path
        self.data_source = data_source
        self.debugging_print = debugging_print

        self._prepare_data()
        self._resolve_axes()
        self._resolve_panel_rows()
        self._resolve_colors()
        self._resolve_labels(title)
        self._draw()

        if self.save_path:
            self._save_plot()

        plt.close(self.fig)

    # ------------------------------------------------------------------ config
    @staticmethod
    def load_config(path=None):
        return _load_yaml_style_config(
            HEATMAP_CONFIG_PATH if path is None else path
        )

    @classmethod
    def configured_layouts(cls, path=None):
        """Return the layouts ``comparison_heatmap.layouts`` of the config asks for.

        Args:
            path: Config file; the default heatmap config when None.

        Returns:
            tuple[str, ...]: Layout names in config order, duplicates removed.
                Both layouts when the key is missing; empty when it is an
                empty list or None (the comparison heatmaps are then off).
        """
        section = cls.load_config(path).get("comparison_heatmap", {}) or {}
        layouts = section.get("layouts", list(cls.LAYOUTS))
        if layouts is None:
            return ()
        if isinstance(layouts, str):
            layouts = [layouts]
        resolved = []
        for layout in layouts:
            layout = cls._resolve_choice(layout, cls.LAYOUTS, "layouts")
            if layout not in resolved:
                resolved.append(layout)
        return tuple(resolved)

    @classmethod
    def configured_transpose(cls, path=None):
        """Return whether the config puts the kinases on the y axis.

        Args:
            path: Config file; the default heatmap config when None.

        Returns:
            bool: ``comparison_heatmap.transpose``.
        """
        section = cls.load_config(path).get("comparison_heatmap", {}) or {}
        return bool(section.get("transpose", False))

    @staticmethod
    def _resolve_choice(value, allowed, name):
        """Validate a configured keyword.

        Returns:
            str: The lower-cased value.
        """
        resolved = str(value).strip().lower()
        if resolved not in allowed:
            raise ValueError(
                f"Unknown comparison heatmap {name} '{value}'. Use one of {list(allowed)}."
            )
        return resolved

    def _dprint(self, *args, **kwargs):
        """Print only if debugging_print is enabled."""
        if self.debugging_print:
            print(*args, **kwargs)

    # -------------------------------------------------------------------- data
    @staticmethod
    def _outranks(new, old):
        """Return whether ``new`` replaces ``old`` in the per-column dedup.

        Same rule as HeatmapPlot_UKA: the value with the larger magnitude wins,
        NaN loses against everything.
        """
        new_rank = abs(new) if pd.notna(new) else float("-inf")
        old_rank = abs(old) if pd.notna(old) else float("-inf")
        return new_rank > old_rank

    def _prepare_data(self):
        """Build the union axes and one value / absent matrix per comparison."""
        # An accession scored on BOTH arrays is two independent measurements and
        # gets one entry per array, labelled "<accession> (<Type>)". The
        # decision is taken over ALL comparisons, so a label means the same
        # thing in every slice. Matching against the pathway hits uses the bare
        # accession, which is what g:Profiler was queried with.
        types_per_accession = {}
        for comparison in self.comparisons:
            df = comparison.results_data
            if df is None or df.empty or "Type" not in df.columns:
                continue
            for accession, array_type in zip(df["Kinase"].astype(str), df["Type"].astype(str)):
                types_per_accession.setdefault(accession, set()).add(array_type)

        self._column_accessions = {}
        per_comparison = []
        for comparison in self.comparisons:
            values = {}
            df = comparison.results_data
            if df is not None and not df.empty and "Kinase" in df.columns:
                has_type = "Type" in df.columns
                has_value = self.value_col in df.columns
                for _, row in df.iterrows():
                    accession = str(row["Kinase"])
                    column = accession
                    if has_type and len(types_per_accession.get(accession, ())) > 1:
                        column = f"{accession} ({row['Type']})"
                    value = row[self.value_col] if has_value else np.nan
                    if column in values and not self._outranks(value, values[column]):
                        continue
                    values[column] = value
                    self._column_accessions[column] = accession

            hits = {}
            p_values = {}
            pathway_order = []
            enrichment = comparison.enrichment_data
            if (
                enrichment is not None
                and not enrichment.empty
                and {"name", "intersections"} <= set(enrichment.columns)
            ):
                has_p = "p_value" in enrichment.columns
                for _, row in enrichment.iterrows():
                    name = str(row["name"])
                    if name not in hits:
                        pathway_order.append(name)
                        hits[name] = set()
                    hits[name].update(_parse_intersection(row["intersections"]))
                    if has_p and pd.notna(row["p_value"]):
                        p_values[name] = min(p_values.get(name, np.inf), float(row["p_value"]))

            per_comparison.append(
                {
                    "values": values,
                    "hits": hits,
                    "p_values": p_values,
                    "pathway_order": pathway_order,
                }
            )
        self._per_comparison = per_comparison

        # Kinase universe: every significant kinase that is a hit of at least one
        # enriched pathway in at least one comparison -- the union of what the
        # single heatmaps show.
        kinases_in_pathways = set()
        for entry in per_comparison:
            for members in entry["hits"].values():
                kinases_in_pathways.update(members)
        all_columns = {
            column
            for entry in per_comparison
            for column in entry["values"]
            if self._column_accessions[column] in kinases_in_pathways
        }
        if not all_columns:
            raise ValueError(
                "No significant kinase falls into an enriched pathway in any comparison."
            )

        def mean_value(column):
            observed = [
                entry["values"][column]
                for entry in per_comparison
                if column in entry["values"] and pd.notna(entry["values"][column])
            ]
            return float(np.mean(observed)) if observed else 0.0

        def presence(column):
            return sum(column in entry["values"] for entry in per_comparison)

        if self.KINASE_ORDER == "presence":
            self.kinases = sorted(all_columns, key=lambda c: (-presence(c), mean_value(c), c))
        else:
            self.kinases = sorted(all_columns, key=lambda c: (mean_value(c), c))

        # Pathway order: enriched in most comparisons first, then by their best
        # p-value, then by first appearance (g:Profiler lists by p-value).
        first_seen = {}
        for entry in per_comparison:
            for name in entry["pathway_order"]:
                first_seen.setdefault(name, len(first_seen))

        def pathway_key(name):
            enriched_in = sum(name in entry["hits"] for entry in per_comparison)
            best_p = min(
                (entry["p_values"].get(name, np.inf) for entry in per_comparison),
                default=np.inf,
            )
            return (-enriched_in, best_p, first_seen[name])

        self.pathway_names = sorted(first_seen, key=pathway_key)
        self.pathways = [self.ALL_KINASES_ROW] + self.pathway_names

        n_comparisons = len(self.comparisons)
        n_pathways, n_kinases = len(self.pathways), len(self.kinases)
        self.values = np.full((n_comparisons, n_pathways, n_kinases), np.nan)
        self.absent = np.zeros((n_comparisons, n_pathways, n_kinases), dtype=bool)
        for s, entry in enumerate(per_comparison):
            for j, column in enumerate(self.kinases):
                if column not in entry["values"]:
                    # Not significant in this comparison: the whole kinase is absent.
                    self.absent[s, :, j] = True
                    continue
                value = entry["values"][column]
                accession = self._column_accessions[column]
                self.values[s, 0, j] = value
                for r, name in enumerate(self.pathway_names, start=1):
                    if name not in entry["hits"]:
                        self.absent[s, r, j] = True
                    elif accession in entry["hits"][name]:
                        self.values[s, r, j] = value
                    # else: member of neither -> NaN -> white, as in the single heatmaps

        n_filled = int(np.isfinite(self.values).sum())
        self._dprint(
            f"Value: {self.value_col} | Layout: {self.layout} | "
            f"Comparisons: {n_comparisons} | Pathways: {n_pathways} | Kinases: {n_kinases} | "
            f"Filled slices: {n_filled}"
        )

    def _resolve_axes(self):
        """Put pathways and kinases on the axes, swapping them when transposed."""
        if self.transpose:
            self.values = self.values.transpose(0, 2, 1)
            self.absent = self.absent.transpose(0, 2, 1)
            self.row_labels, self.col_labels = list(self.kinases), list(self.pathways)
        else:
            self.row_labels, self.col_labels = list(self.pathways), list(self.kinases)

    def _resolve_panel_rows(self):
        """Choose the rows each panel shows.

        A panel of the ``panels`` layout is one comparison, and a row that is
        absent all the way across says nothing about it: the pathway is not
        enriched there (or, transposed, the kinase is not significant there).
        With ``panel_rows: present`` those rows are dropped per panel, which is
        what keeps a stack of panels down to the rows that carry a result; the
        columns stay shared, so the panels still line up. ``shared`` keeps the
        full union in every panel. The split layout always keeps every row,
        because its comparisons share the cells.
        """
        n_rows = len(self.row_labels)
        every_row = np.arange(n_rows)
        if self.layout == self.SPLIT_LAYOUT or self.PANEL_ROWS == "shared":
            self.panel_rows = [every_row for _ in self.comparisons]
            return

        self.panel_rows = []
        for index in range(len(self.comparisons)):
            keep = every_row[~self.absent[index].all(axis=1)]
            # A comparison with significant kinases but no enriched pathway
            # keeps its summary row rather than collapsing to nothing.
            self.panel_rows.append(keep if keep.size else every_row[:1])
        self._dprint(
            "Rows per panel: "
            + ", ".join(
                f"{c.label}: {len(rows)}/{n_rows}"
                for c, rows in zip(self.comparisons, self.panel_rows)
            )
        )

    def _displayed_rows(self):
        """Return the row indices that appear in at least one panel."""
        used = sorted({int(i) for rows in self.panel_rows for i in rows})
        return used or list(range(len(self.row_labels)))

    def _resolve_colors(self):
        """Pick the one colour scale every slice and panel is drawn with."""
        if self.COLOR_SCALE == FIXED_COLOR_SCALE:
            abs_max = (
                self.FIXED_ZSCORE_ABS_MAX
                if self.value_col == "Z_Score"
                else self.FIXED_VALUE_ABS_MAX
            )
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = _data_limits(self.values.reshape(len(self.comparisons), -1))
        self.cmap, self.norm = _centered_colormap(self.CMAP, vmin, vmax, center=0.0)

    def _resolve_labels(self, title):
        """Derive slice labels, panel titles and the figure title."""
        controls = {comparison.control for comparison in self.comparisons}
        shared_control = controls.pop() if len(controls) == 1 else None
        if shared_control is not None and all(c.condition for c in self.comparisons):
            self.slice_labels = [str(c.condition) for c in self.comparisons]
            default_title = f"{shared_control} vs " + " / ".join(self.slice_labels)
        else:
            self.slice_labels = [str(c.label) for c in self.comparisons]
            default_title = None
        self.panel_titles = [str(c.label) for c in self.comparisons]
        if title is None:
            self.title = default_title if self.layout == self.SPLIT_LAYOUT else None
        else:
            self.title = title or None

    # ------------------------------------------------------------- text metrics
    def _char_width(self, fontsize):
        """Return the width of one character in inches at ``fontsize``."""
        return self._CHAR_WIDTH_PER_PT * float(fontsize)

    def _line_height(self, fontsize):
        """Return the height of one text line in inches at ``fontsize``."""
        return self._LINE_HEIGHT_PER_PT * float(fontsize)

    @classmethod
    def _measure(cls, text, fontsize):
        """Return the rendered width of one line of text in inches.

        The margins have to be right to the millimetre for a target width to
        hold, and counting characters is up to a quarter out on the mixed-case
        pathway names. The glyph outlines are measured once per (text, size).
        """
        key = (text, float(fontsize))
        cached = cls._TEXT_WIDTH_CACHE.get(key)
        if cached is None:
            try:
                path = TextPath((0, 0), text, size=float(fontsize), prop=FontProperties())
                cached = float(path.get_extents().width) / 72.0
            except Exception:
                cached = len(text) * cls._CHAR_WIDTH_PER_PT * float(fontsize)
            cls._TEXT_WIDTH_CACHE[key] = cached
        return cached

    @classmethod
    def _measure_height(cls, text, fontsize):
        """Return the inked height of one line of text in inches."""
        key = ("h", text, float(fontsize))
        cached = cls._TEXT_WIDTH_CACHE.get(key)
        if cached is None:
            try:
                path = TextPath((0, 0), text, size=float(fontsize), prop=FontProperties())
                cached = float(path.get_extents().height) / 72.0
            except Exception:
                cached = float(fontsize) / 72.0
            cls._TEXT_WIDTH_CACHE[key] = cached
        return cached

    def _text_width(self, text, fontsize):
        """Return the width in inches of the longest line of ``text``."""
        return max(self._measure(line, fontsize) for line in str(text).split("\n"))

    def _column_pitch(self):
        """Return the column width upright tick labels need to clear each other.

        Upright labels lie across the columns, so what has to fit is the inked
        height of the tallest one: accessions in capitals and digits need less
        than pathway names with their descenders, which is why this is measured
        rather than taken from the font size.
        """
        heights = [
            self._measure_height(label, self.X_TICK_FONTSIZE) for label in self.col_display
        ]
        tallest = max(heights) if heights else self._line_height(self.X_TICK_FONTSIZE)
        return tallest + self._LABEL_CLEARANCE

    def _tick_extent(self, labels, fontsize, rotation):
        """Return how far rotated tick labels reach away from their axis, in inches."""
        if not labels:
            return 0.0
        longest = max(self._text_width(label, fontsize) for label in labels)
        radians = np.radians(float(rotation) % 180.0)
        return abs(longest * np.sin(radians)) + self._line_height(fontsize) * abs(
            np.cos(radians)
        )

    def _shorten(self, labels):
        """Cut labels longer than ``label_max_chars`` back to it, with an ellipsis."""
        limit = self.LABEL_MAX_CHARS
        if not limit:
            return list(labels)
        limit = int(limit)
        return [
            label if len(label) <= limit else label[: max(1, limit - 1)].rstrip() + "…"
            for label in labels
        ]

    def _resolve_row_display(self):
        """Shorten and optionally wrap the row labels, which set the left margin.

        Wrapping buys width at the price of height: every row grows to the
        tallest label, so it stays off unless it is asked for.
        """
        labels = self._shorten(self.row_labels)
        wrap = self.ROW_LABEL_WRAP
        chars = None
        if isinstance(wrap, str):
            if wrap.strip().lower() != "auto":
                raise ValueError(
                    f"Unknown comparison heatmap row_label_wrap '{wrap}'. "
                    "Use 'auto', null or a number of characters."
                )
            if self.target_width is not None:
                budget = self.target_width * self._ROW_LABEL_FRACTION
                chars = max(14, int(budget / self._char_width(self.Y_TICK_FONTSIZE)))
        elif wrap is not None:
            chars = int(wrap)

        longest = max(len(label) for label in labels)
        if chars is None or chars >= longest:
            self.row_display = labels
        else:
            self.row_display = [textwrap.fill(label, chars) for label in labels]

    def _resolve_col_display(self):
        """Shorten the column labels, which are read along the tick direction."""
        self.col_display = self._shorten(self.col_labels)

    def _fitting_fontsize(self, text, fontsize, fig_width, minimum=8):
        """Return the largest size <= ``fontsize`` at which ``text`` fits the figure."""
        if not text:
            return fontsize
        while fontsize > minimum and self._measure(text, fontsize) > fig_width - 2 * self._PAD:
            fontsize -= 1
        return fontsize

    def _figure_text(self, text, y_fraction, fontsize, center, va="bottom"):
        """Place centred figure text, shifted so it stays inside the figure."""
        fig_w = float(self.fig.get_size_inches()[0])
        fontsize = self._fitting_fontsize(text, fontsize, fig_w)
        half = self._measure(text, fontsize) / 2
        x = min(max(center, half + self._PAD), fig_w - half - self._PAD)
        return self.fig.text(
            x / fig_w, y_fraction, text, ha="center", va=va, fontsize=fontsize
        )

    def _resolve_tick_rotation(self, cell_width, pitch):
        """Return the x tick rotation, upright when the columns get too narrow."""
        setting = self.TICK_ROTATION
        if isinstance(setting, str):
            if setting.strip().lower() != "auto":
                raise ValueError(
                    f"Unknown comparison heatmap tick_rotation '{setting}'. "
                    "Use 'auto' or an angle in degrees."
                )
            needed = self._DIAGONAL_PITCH_FACTOR * pitch
            return float(self.X_TICK_ROTATION) if cell_width >= needed else 90.0
        return float(setting)

    # ------------------------------------------------------------------ geometry
    def _geometry(self):
        """Return every size of the figure, in inches.

        The cells absorb a target width: the margins follow from the labels and
        their configured font sizes, the panels stack sideways or downwards,
        the colorbar and key move below the plot when the right-hand block
        would not fit, and the cell width takes whatever is left over.
        """
        split = self.layout == self.SPLIT_LAYOUT
        n_comparisons = len(self.comparisons)
        n_panels = 1 if split else n_comparisons
        n_slices = n_comparisons if split else 1
        slices_vertical = split and self.SPLIT_ORIENTATION == "vertical"
        n_cols = len(self.col_labels)
        row_counts = [len(rows) for rows in self.panel_rows[:n_panels]]

        self._resolve_row_display()
        self._resolve_col_display()

        # Only the labels that really appear may dictate the row height and
        # the left margin.
        shown = self._displayed_rows()
        row_display = [self.row_display[i] for i in shown]
        row_lines = max(text.count("\n") + 1 for text in row_display)
        cell_h = max(self.CELL_HEIGHT, row_lines * self._line_height(self.Y_TICK_FONTSIZE))
        if split and not slices_vertical:
            cell_h = max(cell_h, n_slices * self.SLICE_MIN_SIZE)
        cell_w = (
            max(self.CELL_WIDTH, n_slices * self.SLICE_MIN_SIZE)
            if slices_vertical
            else self.CELL_WIDTH
        )

        margin_left = max(
            0.35,
            max(self._text_width(text, self.Y_TICK_FONTSIZE) for text in row_display)
            + self._line_height(self.AXES_LABEL_FONTSIZE)
            + 2 * self._PAD
            + 0.06,
        )
        gap = self.PANEL_GAP if n_panels > 1 else 0.0
        key = self._key_geometry(with_slices=split, direction="column")
        cbar_w = self.CBAR_WIDTH_FACTOR * self.CELL_WIDTH
        right_block = (
            self._KEY_BLOCK_GAP
            + max(cbar_w + self._CBAR_TICK_ROOM, key["key_w"])
            + self._PAD
        )
        bottom_right_block = self._PAD + self._line_height(self.X_TICK_FONTSIZE) / 2

        # How far a column may be squeezed: down to where the upright tick
        # labels would touch, unless a width was configured instead.
        floor = self.MIN_CELL_WIDTH if self.MIN_CELL_WIDTH is not None else self._column_pitch()

        direction = "horizontal" if n_panels == 1 else self.PANEL_DIRECTION
        panels_across = n_panels if direction == "horizontal" else 1
        panels_down = 1 if direction == "horizontal" else n_panels
        gap_across = gap if panels_across > 1 else 0.0
        gap_down = gap if panels_down > 1 else 0.0

        legend = self.LEGEND_POSITION
        if legend == "auto":
            if self.target_width is None:
                legend = "right"
            else:
                room = (
                    self.target_width
                    - margin_left
                    - right_block
                    - (panels_across - 1) * gap_across
                )
                legend = (
                    "right"
                    if room >= panels_across * n_cols * max(floor, self.CELL_WIDTH / 2)
                    else "bottom"
                )
        if legend == "bottom":
            key = self._key_geometry(with_slices=split, direction="row")
            right_block = bottom_right_block

        # Past the floor the figure gets wider than asked for, because the
        # alternative is shrinking the type.
        if self.target_width is not None:
            room = (
                self.target_width - margin_left - right_block - (panels_across - 1) * gap_across
            )
            cell_w = min(cell_w, room / (panels_across * n_cols))
            if cell_w < floor:
                cell_w = floor
                self._dprint(
                    f"     {panels_across * n_cols} columns and "
                    f"{margin_left:.1f} in of row labels do not fit "
                    f"{self.target_width} in at {self.X_TICK_FONTSIZE} pt; the figure is "
                    f"{margin_left + panels_across * n_cols * floor + (panels_across - 1) * gap_across + right_block:.1f}"
                    " in wide instead. Shorten the labels (label_max_chars) or "
                    "give the panels their own figure to get under it."
                )
        cell_w = max(cell_w, floor)

        rotation = self._resolve_tick_rotation(cell_w, self._column_pitch())
        heat_w = n_cols * cell_w
        panels_w = panels_across * heat_w + (panels_across - 1) * gap_across
        fig_w = margin_left + panels_w + right_block

        title_fontsize = self._fitting_fontsize(self.title, self.TITLE_FONTSIZE, fig_w)
        panel_title_h = (
            self._line_height(self.TITLE_FONTSIZE - 2) + 0.1 if n_panels > 1 else 0.0
        )
        margin_top = self._PAD
        if self.title:
            margin_top += self._line_height(title_fontsize) + self.TITLE_PAD / 72.0

        margin_bottom = (
            self._tick_extent(self.col_display, self.X_TICK_FONTSIZE, rotation)
            + self._line_height(self.AXES_LABEL_FONTSIZE)
            + 3 * self._PAD
        )
        legend_h, legend_rows, cbar_len = 0.0, 0, 0.0
        cbar_stack = 0.0
        if legend == "bottom":
            cbar_len = max(self._CBAR_MIN_LENGTH, min(2.4, panels_w * 0.3))
            # The bar itself, its tick labels and its name, stacked downwards.
            cbar_stack = (
                self._CBAR_BAR_THICKNESS
                + self._line_height(self.CBAR_LABEL_FONTSIZE - 2)
                + self._line_height(self.CBAR_LABEL_FONTSIZE)
                + 0.06
            )
            # Colorbar and key side by side when the width allows, else stacked.
            available = fig_w - 2 * self._PAD
            # Rather than wrap onto a second row, let the bar get shorter first.
            side_by_side = available - self._CBAR_TICK_ROOM - key["key_w"]
            if side_by_side >= self._CBAR_MIN_LENGTH:
                cbar_len = min(cbar_len, side_by_side)
                legend_rows = 1
            else:
                legend_rows = 2
            legend_h = (
                cbar_stack + 0.12 + key["key_h"]
                if legend_rows == 2
                else max(cbar_stack, key["key_h"])
            )
            margin_bottom += legend_h + 2 * self._PAD

        # Everything around the cells is now fixed, so the height budget can be
        # handed to the rows. Stacked panels queue their rows up, side-by-side
        # ones are as tall as the longest.
        panel_overhead = panels_down * panel_title_h + (panels_down - 1) * gap_down
        rows_down = sum(row_counts) if panels_down > 1 else max(row_counts)
        if self.target_height is not None:
            floor = self.MIN_CELL_HEIGHT or row_lines * self._line_height(
                self.Y_TICK_FONTSIZE
            )
            room = self.target_height - margin_top - margin_bottom - panel_overhead
            cell_h = max(floor, min(cell_h, room / rows_down))
            if cell_h <= floor and rows_down * floor > room:
                self._dprint(
                    f"     Target height {self.target_height} in is below what "
                    f"{rows_down} rows need at the {self.Y_TICK_FONTSIZE} pt label "
                    f"line height ({floor:.3f} in); the figure is taller than asked for."
                )

        panel_heights = [count * cell_h for count in row_counts]
        heat_h = max(panel_heights)
        panels_h = rows_down * cell_h + panel_overhead
        fig_h = margin_top + panels_h + margin_bottom
        if legend == "right":
            side_h = key["key_h"] + 0.5 + min(self.CBAR_HEIGHT_ROWS * cell_h, heat_h)
            fig_h = max(fig_h, margin_top + side_h + margin_bottom)

        return {
            "split": split,
            "vertical": slices_vertical,
            "direction": direction,
            "n_panels": n_panels,
            "n_slices": n_slices,
            "row_counts": row_counts,
            "panel_heights": panel_heights,
            "n_cols": n_cols,
            "cell_w": cell_w,
            "cell_h": cell_h,
            "heat_w": heat_w,
            "heat_h": heat_h,
            "panels_w": panels_w,
            "panels_h": panels_h,
            "panel_title_h": panel_title_h,
            "gap_across": gap_across,
            "gap_down": gap_down,
            "margin_left": margin_left,
            "margin_top": margin_top,
            "margin_bottom": margin_bottom,
            "right_block": right_block,
            "rotation": rotation,
            "title_fontsize": title_fontsize,
            "legend": legend,
            "legend_h": legend_h,
            "legend_rows": legend_rows,
            "cbar_len": cbar_len,
            "cbar_stack": cbar_stack,
            "key": key,
            "fig_w": fig_w,
            "fig_h": fig_h,
        }

    # ----------------------------------------------------------------- drawing
    def _compose_rgba(self, values, absent):
        """Return the RGBA image of one value matrix (white = NaN, grey = absent)."""
        values = np.asarray(values, dtype=float)
        rgba = np.asarray(self.cmap(self.norm(np.ma.masked_invalid(values))), dtype=float)
        rgba[~np.isfinite(values)] = (1.0, 1.0, 1.0, 1.0)
        rgba[np.asarray(absent, dtype=bool)] = to_rgba(self.ABSENT_COLOR)
        return rgba

    def _panel_matrix(self, index, geometry):
        """Return the value / absent matrix drawn in panel ``index``."""
        if not geometry["split"]:
            rows = self.panel_rows[index]
            return self.values[index][rows], self.absent[index][rows]
        # The split layout is one panel that keeps every row.
        n_rows = len(self.row_labels)
        n_cols, n_slices = geometry["n_cols"], geometry["n_slices"]
        if geometry["vertical"]:
            # (slice, row, col) -> (row, col, slice) -> stripes inside each column
            values = np.moveaxis(self.values, 0, -1).reshape(n_rows, n_cols * n_slices)
            absent = np.moveaxis(self.absent, 0, -1).reshape(n_rows, n_cols * n_slices)
        else:
            # (slice, row, col) -> (row, slice, col) -> bands inside each row
            values = np.moveaxis(self.values, 0, 1).reshape(n_rows * n_slices, n_cols)
            absent = np.moveaxis(self.absent, 0, 1).reshape(n_rows * n_slices, n_cols)
        return values, absent

    def _rule_widths(self, geometry, n_slices):
        """Return the cell-border and slice-separator line widths for these cells.

        A squeezed figure has cells of a tenth of an inch, where a 0.5 pt rule
        takes a fifth of the cell and the colour disappears behind the grid, so
        both rules thin out with the cells and a separator thinner than the
        stripe it divides is dropped altogether.
        """
        cell = min(geometry["cell_w"], geometry["cell_h"])
        border = min(self.CELL_LINEWIDTH, max(self._MIN_RULE_WIDTH, cell * 2.5))
        if n_slices < 2:
            return border, 0.0
        size = geometry["cell_w"] if geometry["vertical"] else geometry["cell_h"]
        slice_size = size / n_slices
        if slice_size < self._MIN_SEPARATED_SLICE:
            return border, 0.0
        return border, min(self.SLICE_LINEWIDTH, max(self._MIN_RULE_WIDTH, slice_size * 4))

    def _draw_cells(self, ax, rgba, geometry, n_slices, n_rows):
        """Draw the RGBA cell image plus the cell borders and slice separators."""
        n_cols = geometry["n_cols"]
        border_width, slice_width = self._rule_widths(geometry, n_slices)
        ax.imshow(
            rgba,
            extent=(0, n_cols, n_rows, 0),
            aspect="auto",
            interpolation="nearest",
            zorder=1,
        )
        if n_slices > 1 and slice_width > 0:
            if geometry["vertical"]:
                xs = [j + s / n_slices for j in range(n_cols) for s in range(1, n_slices)]
                ax.vlines(
                    xs, 0, n_rows,
                    colors=self.SLICE_LINECOLOR, linewidth=slice_width, zorder=2,
                )
            else:
                ys = [r + s / n_slices for r in range(n_rows) for s in range(1, n_slices)]
                ax.hlines(
                    ys, 0, n_cols,
                    colors=self.SLICE_LINECOLOR, linewidth=slice_width, zorder=2,
                )
        ax.hlines(
            np.arange(n_rows + 1), 0, n_cols,
            colors=self.CELL_LINECOLOR, linewidth=border_width, zorder=3, clip_on=False,
        )
        ax.vlines(
            np.arange(n_cols + 1), 0, n_rows,
            colors=self.CELL_LINECOLOR, linewidth=border_width, zorder=3, clip_on=False,
        )

    def _format_heat_axes(self, ax, geometry, show_ylabels, rows, show_xlabels=True):
        """Ticks, limits and spines of one heatmap axes."""
        n_rows, n_cols = len(rows), geometry["n_cols"]
        rotation = geometry["rotation"]
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_xticks(np.arange(n_cols) + 0.5)
        if show_xlabels:
            ax.set_xticklabels(
                self.col_display,
                rotation=rotation,
                ha="center" if rotation >= 89.0 else self.X_TICK_HA,
                fontsize=self.X_TICK_FONTSIZE,
            )
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        if show_ylabels:
            ax.set_yticklabels(
                [self.row_display[i] for i in rows],
                rotation=0,
                fontsize=self.Y_TICK_FONTSIZE,
            )
        else:
            ax.tick_params(axis="y", labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _kinase_axis_label(self):
        """Return the axis label describing the kinase order."""
        if self.KINASE_ORDER == "presence":
            return (
                f"Kinase (ordered by number of comparisons, then mean "
                f"{self.value_label}, low to high)"
            )
        return f"Kinase (ordered by mean {self.value_label}, low to high)"

    def _axis_labels(self):
        """Return the (y, x) axis labels for the current orientation."""
        kinase = self._kinase_axis_label()
        pathway = _pathway_axis_label(self.data_source)
        return (kinase, pathway) if self.transpose else (pathway, kinase)

    def _draw(self):
        """Draw the figure: one panel with sliced cells, or one panel per comparison."""
        geometry = self._geometry()
        fig_w, fig_h = geometry["fig_w"], geometry["fig_h"]
        heat_w = geometry["heat_w"]
        margin_left, panels_w = geometry["margin_left"], geometry["panels_w"]
        n_panels, n_slices = geometry["n_panels"], geometry["n_slices"]
        self.geometry = geometry

        self.fig = plt.figure(figsize=(fig_w, fig_h))
        heat_top = fig_h - geometry["margin_top"]
        y_label, x_label = self._axis_labels()

        sideways = geometry["direction"] == "horizontal"
        panel_title_h = geometry["panel_title_h"]
        panel_heights = geometry["panel_heights"]
        self.axes = []
        next_top = heat_top
        for index in range(n_panels):
            panel_h = panel_heights[index]
            if sideways:
                x0 = margin_left + index * (heat_w + geometry["gap_across"])
                y_top = heat_top - panel_title_h
            else:
                # Stacked downwards: every panel carries its own title, and
                # only the last one repeats the column labels. Panels differ
                # in height when they differ in rows, so each one starts where
                # the previous one ended.
                x0 = margin_left
                y_top = next_top - panel_title_h
                next_top = y_top - panel_h - geometry["gap_down"]
            ax = self.fig.add_axes(
                [x0 / fig_w, (y_top - panel_h) / fig_h, heat_w / fig_w, panel_h / fig_h]
            )
            rows = self.panel_rows[index]
            values, absent = self._panel_matrix(index, geometry)
            self._draw_cells(
                ax, self._compose_rgba(values, absent), geometry, n_slices, len(rows)
            )
            self._format_heat_axes(
                ax,
                geometry,
                show_ylabels=True,
                rows=rows,
                show_xlabels=(sideways or index == n_panels - 1),
            )
            if n_panels > 1:
                ax.set_title(
                    self.panel_titles[index], fontsize=self.TITLE_FONTSIZE - 2, pad=4
                )
            self.axes.append(ax)
        self.ax = self.axes[0]

        # One y axis label for the whole block, placed where the left margin
        # was measured for it.
        self.fig.text(
            (self._PAD + self._line_height(self.AXES_LABEL_FONTSIZE) / 2) / fig_w,
            (heat_top - geometry["panels_h"] / 2) / fig_h,
            y_label,
            ha="center", va="center", rotation=90, fontsize=self.AXES_LABEL_FONTSIZE,
        )

        x_label_y = self._PAD
        if geometry["legend"] == "bottom":
            x_label_y += geometry["legend_h"] + self._PAD
        self._figure_text(
            x_label,
            x_label_y / fig_h,
            self.AXES_LABEL_FONTSIZE,
            center=margin_left + panels_w / 2,
        )
        if self.title:
            self._figure_text(
                self.title,
                (fig_h - self._PAD) / fig_h,
                geometry["title_fontsize"],
                center=margin_left + panels_w / 2,
                va="top",
            )

        if geometry["legend"] == "right":
            self._place_legend_right(geometry, heat_top - panel_title_h)
        else:
            self._place_legend_bottom(geometry)

    def _add_colorbar(self, rect, orientation="vertical"):
        """Add the shared colorbar into the figure rectangle ``rect`` (inches)."""
        fig_w, fig_h = self.fig.get_size_inches()
        x, y, width, height = rect
        cbar_ax = self.fig.add_axes([x / fig_w, y / fig_h, width / fig_w, height / fig_h])
        cbar = self.fig.colorbar(
            ScalarMappable(norm=self.norm, cmap=self.cmap),
            cax=cbar_ax,
            orientation=orientation,
        )
        cbar.set_label(self.value_label, fontsize=self.CBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=self.CBAR_LABEL_FONTSIZE - 2)
        return cbar

    def _place_legend_right(self, geometry, heat_top):
        """Colorbar and key in the column to the right of the heatmap."""
        fig_w, fig_h = geometry["fig_w"], geometry["fig_h"]
        key = geometry["key"]
        side_x = geometry["margin_left"] + geometry["panels_w"] + self._KEY_BLOCK_GAP
        block_h = geometry["panels_h"]
        cbar_h = min(self.CBAR_HEIGHT_ROWS * geometry["cell_h"], block_h)
        cbar_h = min(cbar_h, max(1.2, block_h - key["key_h"] - 0.5))
        cbar_w = self.CBAR_WIDTH_FACTOR * self.CELL_WIDTH
        self._add_colorbar((side_x, heat_top - cbar_h, cbar_w, cbar_h))

        key_top = heat_top - cbar_h - 0.5
        key_ax = self.fig.add_axes(
            [
                side_x / fig_w,
                (key_top - key["key_h"]) / fig_h,
                key["key_w"] / fig_w,
                key["key_h"] / fig_h,
            ]
        )
        self._draw_key(key_ax, key)

    def _place_legend_bottom(self, geometry):
        """Colorbar and key in the strip underneath the heatmap."""
        fig_w, fig_h = geometry["fig_w"], geometry["fig_h"]
        key = geometry["key"]
        pad, legend_h = self._PAD, geometry["legend_h"]
        # The bar hangs from the top of the strip; its ticks and label sit below it.
        cbar_y = pad + legend_h - self._CBAR_BAR_THICKNESS
        self._add_colorbar(
            (pad, cbar_y, geometry["cbar_len"], self._CBAR_BAR_THICKNESS),
            orientation="horizontal",
        )

        if geometry["legend_rows"] == 2:
            key_x = pad
            key_y = pad + legend_h - geometry["cbar_stack"] - 0.12 - key["key_h"]
        else:
            key_x = pad + geometry["cbar_len"] + self._CBAR_TICK_ROOM
            key_y = pad + (legend_h - key["key_h"]) / 2
        key_ax = self.fig.add_axes(
            [key_x / fig_w, key_y / fig_h, key["key_w"] / fig_w, key["key_h"] / fig_h]
        )
        self._draw_key(key_ax, key)

    # --------------------------------------------------------------------- key
    def _swatch_entries(self):
        """Return the (colour, text) pairs explaining grey and white cells."""
        return [
            (to_rgba(self.ABSENT_COLOR), "not significant / not enriched\nin that comparison"),
            ("white", "kinase not in pathway"),
        ]

    def _key_geometry(self, with_slices, direction="column"):
        """Return the size of the key in inches and the positions used to draw it.

        Args:
            with_slices: Whether the key explains the slice order (split layout).
            direction: ``"column"`` stacks the slice key above the colour
                swatches (key to the right of the plot), ``"row"`` puts them
                side by side (key underneath the plot).

        Returns:
            dict: Key geometry, including ``key_w`` and ``key_h``.
        """
        n = len(self.comparisons)
        char_w = self._char_width(self.KEY_FONTSIZE)
        line_h = self._line_height(self.KEY_FONTSIZE)
        pad = self._KEY_PAD

        swatches = self._swatch_entries()
        swatch_text_w = max(
            len(line) for _, text in swatches for line in text.split("\n")
        ) * char_w
        swatch_w = self._KEY_SWATCH + 0.12 + swatch_text_w
        swatch_h = sum(
            max(self._KEY_SWATCH, line_h * (text.count("\n") + 1)) + 0.12
            for _, text in swatches
        )

        geometry = {
            "n": n,
            "with_slices": with_slices,
            "direction": direction,
            "pad": pad,
            "swatch_w": swatch_w,
            "swatch_h": swatch_h,
        }
        if not with_slices:
            geometry["key_w"] = pad + swatch_w + pad
            geometry["key_h"] = pad + swatch_h + pad
            geometry["swatch_x"] = pad
            geometry["swatch_top"] = geometry["key_h"] - pad
            return geometry

        label_w = max(len(label) for label in self.slice_labels) * char_w
        if self.SPLIT_ORIENTATION == "vertical":
            cell_w = n * self._KEY_SLICE_WIDTH
            cell_h = self._KEY_CELL_HEIGHT
            slice_w = cell_w + 0.2 + label_w
            slice_h = cell_h + self._KEY_STEP * n
        else:
            cell_w = self._KEY_BAND_WIDTH
            cell_h = n * self._KEY_BAND_HEIGHT
            slice_w = cell_w + 0.15 + label_w
            slice_h = cell_h
        geometry.update(cell_w=cell_w, cell_h=cell_h, slice_w=slice_w, slice_h=slice_h)

        if direction == "row":
            geometry["key_w"] = pad + slice_w + self._KEY_BLOCK_GAP + swatch_w + pad
            geometry["key_h"] = pad + max(slice_h, swatch_h) + pad
            geometry["slice_top"] = geometry["key_h"] - pad
            geometry["swatch_x"] = pad + slice_w + self._KEY_BLOCK_GAP
            geometry["swatch_top"] = geometry["key_h"] - pad
        else:
            geometry["key_w"] = pad + max(slice_w, swatch_w) + pad
            geometry["key_h"] = pad + slice_h + 0.3 + swatch_h + pad
            geometry["slice_top"] = geometry["key_h"] - pad
            geometry["swatch_x"] = pad
            geometry["swatch_top"] = geometry["key_h"] - pad - slice_h - 0.3
        return geometry

    def _draw_key(self, ax, geometry):
        """Draw the slice key and the colour swatches into ``ax`` (inch coordinates)."""
        ax.set_xlim(0, geometry["key_w"])
        ax.set_ylim(0, geometry["key_h"])
        ax.set_aspect("equal")
        ax.axis("off")
        pad = geometry["pad"]
        fontsize = self.KEY_FONTSIZE

        if geometry["with_slices"]:
            n = geometry["n"]
            cell_w, cell_h = geometry["cell_w"], geometry["cell_h"]
            x0 = pad
            y_top = geometry["slice_top"]
            y0 = y_top - cell_h
            if self.SPLIT_ORIENTATION == "vertical":
                slice_w = cell_w / n
                text_x = x0 + cell_w + 0.2
                for s, label in enumerate(self.slice_labels):
                    ax.add_patch(
                        Rectangle(
                            (x0 + s * slice_w, y0), slice_w, cell_h,
                            facecolor="white", edgecolor="0.55", linewidth=0.6,
                        )
                    )
                    x_center = x0 + (s + 0.5) * slice_w
                    y_label = y0 - self._KEY_STEP * (n - s)
                    ax.plot(
                        [x_center, x_center, text_x - 0.06],
                        [y0, y_label, y_label],
                        color="0.3", linewidth=0.6, solid_capstyle="round",
                    )
                    ax.text(text_x, y_label, label, fontsize=fontsize, ha="left", va="center")
            else:
                band_h = cell_h / n
                text_x = x0 + cell_w + 0.15
                for s, label in enumerate(self.slice_labels):
                    y_band = y_top - (s + 1) * band_h
                    ax.add_patch(
                        Rectangle(
                            (x0, y_band), cell_w, band_h,
                            facecolor="white", edgecolor="0.55", linewidth=0.6,
                        )
                    )
                    ax.text(
                        text_x, y_band + band_h / 2, label,
                        fontsize=fontsize, ha="left", va="center",
                    )
            ax.add_patch(
                Rectangle(
                    (x0, y0), cell_w, cell_h,
                    fill=False, edgecolor=self.CELL_LINECOLOR, linewidth=0.9,
                )
            )

        line_h = self._line_height(fontsize)
        x_swatch = geometry["swatch_x"]
        y = geometry["swatch_top"]
        for color, text in self._swatch_entries():
            row_h = max(self._KEY_SWATCH, line_h * (text.count("\n") + 1))
            ax.add_patch(
                Rectangle(
                    (x_swatch, y - row_h / 2 - self._KEY_SWATCH / 2),
                    self._KEY_SWATCH, self._KEY_SWATCH,
                    facecolor=color, edgecolor=self.CELL_LINECOLOR, linewidth=0.6,
                )
            )
            ax.text(
                x_swatch + self._KEY_SWATCH + 0.12, y - row_h / 2, text,
                fontsize=fontsize, ha="left", va="center", linespacing=1.15,
            )
            y -= row_h + 0.12

    def _save_plot(self):
        try:
            save_path = Path(self.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not save_path.suffix:
                save_path = save_path.with_suffix(f'.{self.IMAGE_FORMAT}')
            # A target size is only kept when the figure is saved whole; the
            # tight box would crop it back to the drawn content.
            budgeted = self.target_width is not None or self.target_height is not None
            bbox = None if budgeted else "tight"
            self.fig.savefig(
                save_path, dpi=self.DPI, bbox_inches=bbox, format=save_path.suffix.lstrip('.')
            )
            size = self.fig.get_size_inches()
            self._dprint(
                f"Heatmap saved to: {save_path.absolute()} "
                f"({size[0]:.2f} x {size[1]:.2f} in)"
            )
        except Exception as e:
            print(f"Error saving heatmap: {e}")


class HeatmapPlot_Peptides:
    """
    Heatmap of the per-sample peptide signal.

    The colour encodes ``log2(S100)`` -- the log2 of the exposure-normalised
    reaction slope, which is the very quantity ``peptide_change`` is built from
    (peptide_change = mean(log2 S100) of treatment - mean of control). It is an
    ABSOLUTE signal, not a contrast: the values are positive throughout, so the
    scale is sequential and zero carries no special meaning. That is why this
    heatmap, unlike the kinase-pathway one, is not centred on zero.

    Every peptide handed in is drawn; the caller decides the peptide set (the
    pipeline passes everything that passed QC) and splits PTK from STK.

    Expects a DataFrame with the following columns:
        - ID: peptide identifier
        - control_sample_1 to control_sample_4: log2(S100) of the control samples
        - treatment_sample_1 to treatment_sample_4: log2(S100) of the test samples
        - optional control_label_1 / treatment_label_1 etc. for descriptive
          sample labels (for example bio/tech replicate combinations)
    """

    def __init__(self,
                 data=None,
                 csv_path=None,
                 cmap=None,
                 save_path=None,
                 dpi=300,
                 image_format='png',
                 title=None,
                 value_label='log2(S100)',
                 debugging_print=False):
        
        """Load the heatmap configuration and store the plot inputs.
        
        Args:
            cmap: Colormap override; defaults to the configured sequential map.
            value_label: Colorbar label for the plotted quantity.
            debugging_print: Whether to print additional debug information.
        """
        cfg = self.load_config()

        cell = cfg.get('cell', {})
        self.CELL_HEIGHT = cell.get('height', 0.25)
        self.CELL_WIDTH = cell.get('width', 0.3)
        self.CELL_FONTSIZE = cell.get('fontsize', 9)
        self.CELL_LINEWIDTH = cell.get('linewidth', 0.5)
        self.CELL_LINECOLOR = cell.get('linecolor', 'black')

        margins = cfg.get('margins', {})
        self.MARGIN_LEFT = margins.get('left', 4.5)
        self.MARGIN_RIGHT = margins.get('right', 1.5)
        self.MARGIN_TOP = margins.get('top', 1.0)
        self.MARGIN_BOTTOM = margins.get('bottom', 2.0)

        plot = cfg.get('plot', {})
        # The peptide heatmap shows an absolute signal: low -> high, no neutral
        # midpoint. Configured as a colormap name or a list of colour stops.
        self.CMAP = _resolve_cmap(
            plot.get('peptide_signal_cmap', 'coolwarm'),
            name='peptide_signal',
        )
        self.DPI = plot.get('dpi', 300)
        self.IMAGE_FORMAT = plot.get('image_format', 'png')
        self.TITLE_PAD = plot.get('title_pad', 20)
        self.COLOR_SCALE = _resolve_color_scale(
            plot.get('color_scale', INDIVIDUAL_COLOR_SCALE)
        )

        fixed_range = cfg.get('fixed_scale', {}).get(
            'peptide_log2_signal_range', (0.0, 12.0)
        )
        self.FIXED_VMIN, self.FIXED_VMAX = (float(v) for v in fixed_range)

        axes = cfg.get('axes', {})
        self.AXES_LABEL_FONTSIZE = axes.get('label_fontsize', 12)
        self.TITLE_FONTSIZE = axes.get('title_fontsize', 14)
        self.X_TICK_ROTATION = axes.get('x_tick_rotation', 45)
        self.X_TICK_FONTSIZE = axes.get('x_tick_fontsize', 8)
        self.Y_TICK_FONTSIZE = axes.get('y_tick_fontsize', 9)
        self.X_TICK_HA = axes.get('x_tick_ha', 'left')

        cbar = cfg.get('colorbar', {})
        self.CBAR_LABEL_FONTSIZE = cbar.get('label_fontsize', 12)
        self.CBAR_WIDTH_FACTOR = cbar.get('width_factor', 2)
        self.CBAR_HEIGHT_ROWS = cbar.get('height_rows', 20)
        self.CBAR_OFFSET_X = cbar.get('offset_x', 0.05)

        self.debugging_print = debugging_print
        self.cmap = self.CMAP if cmap is None else cmap
        self.value_label = value_label
        self.save_path = save_path
        self.dpi = dpi
        self.image_format = image_format.lower()
        self.title = title

        if data is not None:
            self.df = data.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either data or csv_path must be provided")

        self._prepare_data()
        self._calc_fig_size()
        self.fig, self.ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
        self._draw_heatmap()

        if self.save_path:
            self._save_plot()

        plt.close(self.fig)

    @staticmethod
    def load_config(path=None):
        return _load_yaml_style_config(
            HEATMAP_CONFIG_PATH if path is None else path
        )
    
    def _dprint(self, *args, **kwargs):
        """Print only if debugging_print is enabled."""
        if self.debugging_print:
            print(*args, **kwargs)

    def _calc_fig_size(self):
        """Compute figure size based on the data."""
        n_rows = len(self.peptides)
        n_cols = len(self.sample_labels)
        
        max_label_len = max(len(str(p)) for p in self.peptides)
        self._margin_left = max(self.MARGIN_LEFT, max_label_len * 0.075)
        
        self._fig_width = self._margin_left + n_cols * self.CELL_WIDTH + self.MARGIN_RIGHT
        self._fig_height = self.MARGIN_TOP + n_rows * self.CELL_HEIGHT + self.MARGIN_BOTTOM

    def _prepare_data(self):
        """Build the log2(S100) matrix from the per-sample columns.
        
        No filtering happens here: every row handed in is drawn. The caller
        decides the peptide set, which for the pipeline is everything that passed
        QC.
        """
        df_rows = self.df
        if len(df_rows) == 0:
            raise ValueError("No peptides to plot.")

        if 'ID' in df_rows.columns:
            self.peptides = df_rows['ID'].tolist()
        elif 'peptide' in df_rows.columns:
            self.peptides = df_rows['peptide'].tolist()
        else:
            raise ValueError("Could not find an 'ID' or 'peptide' column")

        # The raw per-sample columns hold log2(S100); the *_zscore variants are a
        # row-wise standardisation of exactly those values and are not used
        # here: the colour is meant to be the signal itself, on the same scale
        # peptide_change is computed from.
        control_cols = [
            col for col in df_rows.columns
            if 'control_sample' in col.lower()
            and '_zscore' not in col.lower()
            and 'label' not in col.lower()
        ]
        treatment_cols = [
            col for col in df_rows.columns
            if 'treatment_sample' in col.lower()
            and '_zscore' not in col.lower()
            and 'label' not in col.lower()
        ]
        if not control_cols or not treatment_cols:
            raise ValueError("Could not find control_sample_* or treatment_sample_* columns")

        control_cols = sorted(control_cols, key=lambda x: int(x.split('_')[2]))
        treatment_cols = sorted(treatment_cols, key=lambda x: int(x.split('_')[2]))

        control_labels = [
            self._resolve_sample_axis_label(
                df_filtered=df_rows,
                group_prefix='control',
                sample_col=col,
                fallback=f'Control {i+1}',
            )
            for i, col in enumerate(control_cols)
        ]
        test_labels = [
            self._resolve_sample_axis_label(
                df_filtered=df_rows,
                group_prefix='treatment',
                sample_col=col,
                fallback=f'Test {i+1}',
            )
            for i, col in enumerate(treatment_cols)
        ]
        self.sample_labels = test_labels + control_labels

        self.matrix = np.hstack(
            [
                df_rows[treatment_cols].to_numpy(dtype=float),
                df_rows[control_cols].to_numpy(dtype=float),
            ]
        )
        self.heatmap_data = pd.DataFrame(
            self.matrix,
            index=self.peptides,
            columns=self.sample_labels,
        )
        self._dprint(
            f"     Heatmap: {len(self.peptides)} peptides x "
            f"{len(self.sample_labels)} samples, colour = {self.value_label}"
        )

    @staticmethod
    def _resolve_sample_axis_label(df_filtered, group_prefix, sample_col, fallback):
        try:
            sample_idx = int(sample_col.split('_')[2])
        except (IndexError, ValueError):
            return fallback

        label_col = f"{group_prefix}_label_{sample_idx}"
        if label_col not in df_filtered.columns:
            return fallback

        label_values = (
            df_filtered[label_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        if label_values.empty:
            return fallback

        return label_values.iloc[0]

    def _get_heatmap_params(self):
        """Determine the heatmap parameters.
        
        log2(S100) is an absolute signal, so the scale is sequential and there is
        no ``center`` argument -- centring on zero would push every value onto one
        half of the colormap and waste the other.
        """
        if self.COLOR_SCALE == INDIVIDUAL_COLOR_SCALE:
            vmin, vmax = _data_limits(self.matrix)
        else:
            vmin, vmax = self.FIXED_VMIN, self.FIXED_VMAX

        return {
            'cmap': self.cmap,
            'vmin': vmin,
            'vmax': vmax,
            'linewidths': self.CELL_LINEWIDTH,
            'linecolor': self.CELL_LINECOLOR,
            'cbar': False,
            'xticklabels': True,
            'yticklabels': True,
        }

    def _apply_fixed_layout(self):
        """Apply the layout using fixed margins."""
        self.fig.subplots_adjust(
            left=self._margin_left / self._fig_width,
            right=1.0 - self.MARGIN_RIGHT / self._fig_width,
            top=1.0 - self.MARGIN_TOP / self._fig_height,
            bottom=self.MARGIN_BOTTOM / self._fig_height
        )

    def _format_axes(self):
        self.ax.set_xlabel('Samples', fontsize=self.AXES_LABEL_FONTSIZE)
        self.ax.set_ylabel('Peptides', fontsize=self.AXES_LABEL_FONTSIZE)
        
        title = self.title or 'Peptide Signal'
        self.ax.set_title(title, fontsize=self.TITLE_FONTSIZE, pad=self.TITLE_PAD)
        
        # Move labels to the top without changing the tick positions.
        self.ax.tick_params(axis='x', top=True, bottom=False,
                            labeltop=True, labelbottom=False)
        self.ax.xaxis.set_label_position('top')

        # Adjust styling only here; do not reset tick locations.
        # Use left alignment for the top axis labels.
        for label in self.ax.get_xticklabels():
            label.set_rotation(self.X_TICK_ROTATION)
            label.set_ha('left')
            label.set_fontsize(self.X_TICK_FONTSIZE)
        
        for label in self.ax.get_yticklabels():
            label.set_rotation(0)
            label.set_fontsize(self.Y_TICK_FONTSIZE)

    def _add_colorbar(self, heatmap_obj):
        """Add a top-aligned peptide heatmap colorbar with fixed 12-cell height."""
        cbar_width_inches = 0.5 * self.CBAR_WIDTH_FACTOR * self.CELL_WIDTH
        cbar_height_inches = 12 * self.CELL_HEIGHT

        ax_pos = self.ax.get_position()
        cbar_width_fig = cbar_width_inches / self._fig_width
        cbar_height_fig = cbar_height_inches / self._fig_height

        cbar_x = ax_pos.x1 + self.CBAR_OFFSET_X
        cbar_y = ax_pos.y1 - cbar_height_fig

        cbar_ax = self.fig.add_axes([cbar_x, cbar_y, cbar_width_fig, cbar_height_fig])

        cbar = self.fig.colorbar(heatmap_obj.collections[0], cax=cbar_ax)
        cbar.set_label(self.value_label, fontsize=self.CBAR_LABEL_FONTSIZE)

    def _draw_heatmap(self):
        self.ax.clear()
        params = self._get_heatmap_params()
        heatmap_obj = sns.heatmap(self.heatmap_data, ax=self.ax, **params)
        self._format_axes()
        self._apply_fixed_layout()
        self._add_colorbar(heatmap_obj)

    def _save_plot(self):
        """Save the heatmap."""
        try:
            p = Path(self.save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            
            if not p.suffix:
                p = p.with_suffix(f'.{self.image_format}')
            
            self.fig.savefig(p, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
            self._dprint(f"Heatmap saved: {p.absolute()}")
        except Exception as e:
            print(f"Error while saving the heatmap: {e}")
            raise

    def close(self):
        """Close the figure."""
        plt.close(self.fig)
