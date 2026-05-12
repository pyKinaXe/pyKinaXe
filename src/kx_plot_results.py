"""Reusable plotting primitives for pyKinaXe outputs.

The scientific modules in ``src/`` compute numerical results; this module is
responsible for turning those results into interactive or file-backed figures.

It contains plotting classes for:

- peptide and kinase volcano plots
- venn diagrams for control/condition overlap summaries
- heatmaps for pathway and peptide outputs

Most plotting style defaults are loaded from YAML configuration files in
``config/`` so the scientific code can stay separate from presentation details.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
import textwrap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

try:
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:  # pragma: no cover - depends on optional GUI libs
    tk = None
    ttk = None
    FigureCanvasTkAgg = None

VOLCANO_CONFIG_PATH = "config/volcano_plot_config.yaml"
HEATMAP_CONFIG_PATH = "config/heatmap_plot_config.yaml"


def _resolve_config_path(path):
    """Resolve config path.
    
    Args:
        path: Path value processed by this helper.
    
    Returns:
        object: Resolved config path.
    """
    path = Path(path)
    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [repo_root / path, Path.cwd() / path, path]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _require_tk_gui():
    """Return require Tk GUI.
    
    Args:
        None.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    if tk is None or ttk is None or FigureCanvasTkAgg is None:
        raise RuntimeError(
            "Tkinter GUI plotting is not available in this environment. "
            "Use gui=False or install Tk support."
        )


class VolcanoPlot:
    """
    Interactive Volcano Plot Visualizer.
    Supports both peptide data and UKA (upstream kinase analysis) data.
    
    Data type is auto-detected:
      - If 'Kinase' column exists → UKA mode
      - Otherwise → Peptide mode
    
    For UKA data, y_axis controls what is plotted on the y-axis.
    Significance is always derived from p_threshold / z_threshold, never from data columns.
    """
    
    def __init__(self, 
                 root=None, 
                 data=None, 
                 delta_threshold=0,
                 p_threshold=0.05, 
                 z_threshold=1.96,
                 y_axis='p_value',
                 x_axis_col=None,
                 x_label=None,
                 save_path=None, 
                 dpi=300, 
                 image_format='png', 
                 gui=True,
                 debugging_print=False):

        """Initialize the VolcanoPlot instance.
        
        Args:
            root: Root object or container used by this function.
            data: Data processed by this function.
            delta_threshold: Threshold value used to filter, classify, or flag results.
            p_threshold: Threshold value used to filter, classify, or flag results.
            z_threshold: Threshold value used to filter, classify, or flag results.
            y_axis: Y axis processed by this function.
            x_axis_col: X axis col processed by this function.
            x_label: X label processed by this function.
            save_path: Path to the save.
            dpi: Dpi processed by this function.
            image_format: Image format processed by this function.
            gui: Whether to use the interactive GUI code path.
            debugging_print: Whether to print additional debug information.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        cfg = self.load_config()

        # --- Scatter ---
        sc = cfg.get('scatter', {})
        self.SC_SIZE_SIG = sc.get('size_significant', 30)
        self.SC_SIZE_NS = sc.get('size_not_significant', 20)
        self.SC_ALPHA_SIG = sc.get('alpha_significant', 0.7)
        self.SC_ALPHA_NS = sc.get('alpha_not_significant', 0.5)
        self.SC_COLOR_UP = sc.get('color_up', 'red')
        self.SC_COLOR_DOWN = sc.get('color_down', 'blue')
        self.SC_COLOR_NS = sc.get('color_not_significant', 'gray')

        # --- Threshold lines ---
        tl = cfg.get('threshold_lines', {})
        self.TL_P_COLOR = tl.get('p_color', 'green')
        self.TL_P_LINESTYLE = tl.get('p_linestyle', '--')
        self.TL_P_LINEWIDTH = tl.get('p_linewidth', 1)
        self.TL_Z_COLOR = tl.get('z_color', 'green')
        self.TL_Z_LINESTYLE = tl.get('z_linestyle', '--')
        self.TL_Z_LINEWIDTH = tl.get('z_linewidth', 1)
        self.TL_DELTA_COLOR = tl.get('delta_color', 'orange')
        self.TL_DELTA_LINESTYLE = tl.get('delta_linestyle', '--')
        self.TL_DELTA_LINEWIDTH = tl.get('delta_linewidth', 1)
        self.TL_CENTER_COLOR = tl.get('center_color', 'black')
        self.TL_CENTER_LINESTYLE = tl.get('center_linestyle', '-')
        self.TL_CENTER_LINEWIDTH = tl.get('center_linewidth', 0.5)

        # --- Axes ---
        ax_cfg = cfg.get('axes', {})
        self.AXES_LABEL_FONTSIZE = ax_cfg.get('label_fontsize', 12)
        self.TITLE_FONTSIZE = ax_cfg.get('title_fontsize', 14)
        self.GRID_ALPHA = ax_cfg.get('grid_alpha', 0.3)

        # --- Legend ---
        lg = cfg.get('legend', {})
        self.LG_FONTSIZE = lg.get('fontsize', 10)
        self.LG_MARKERSCALE = lg.get('markerscale', 1.0)
        self.LG_LOC = lg.get('loc', 'upper right')
        self.LG_FRAMEALPHA = lg.get('framealpha', 0.9)
        self.LG_EDGECOLOR = lg.get('edgecolor', 'black')
        self.LG_HANDLELENGTH = lg.get('handlelength', 2.0)
        self.LG_HANDLEHEIGHT = lg.get('handleheight', 1.5)
        self.LG_LABELSPACING = lg.get('labelspacing', 0.5)
        self.LG_BORDERPAD = lg.get('borderpad', 0.5)
        self.LG_HANDLETEXTPAD = lg.get('handletextpad', 0.8)
        self.LG_MARKER_SIG = lg.get('marker_size_significant', 10)
        self.LG_MARKER_NS = lg.get('marker_size_not_significant', 8)

        # --- Plot ---
        pl = cfg.get('plot', {})
        self.FIG_WIDTH = pl.get('figsize_width', 12)
        self.FIG_HEIGHT = pl.get('figsize_height', 8)

        # --- Hover ---
        hv = cfg.get('hover', {})
        self.HOVER_THRESHOLD = hv.get('threshold', 0.02)

        # --- Tooltip ---
        tt = cfg.get('tooltip', {})
        self.TT_OFFSET_X = tt.get('offset_x', 10)
        self.TT_OFFSET_Y = tt.get('offset_y', 10)
        self.TT_BACKGROUND = tt.get('background', 'lightyellow')
        self.TT_BORDERWIDTH = tt.get('borderwidth', 1)
        self.TT_PADDING = tt.get('padding', 5)

        # --- GUI ---
        gui_cfg = cfg.get('gui', {})
        self.GUI_WIDTH = gui_cfg.get('window_width', 1400)
        self.GUI_HEIGHT = gui_cfg.get('window_height', 1200)
        self.GUI_FONT_FAMILY = gui_cfg.get('font_family', 'Arial')
        self.GUI_FONT_SIZE = gui_cfg.get('font_size', 12)

        # --- Instance params ---
        self.gui_mode = gui
        self.debugging_print = debugging_print
        self.delta_threshold = delta_threshold
        self.p_threshold = p_threshold
        self.z_threshold = z_threshold
        self.y_axis = y_axis.lower()
        self.x_axis_col = x_axis_col
        self.x_label = x_label if x_label is not None else 'KinaseStatistic'
        self.save_path = save_path
        self.dpi = dpi
        self.image_format = image_format.lower()
        
        if not self.gui_mode:
            matplotlib.use('Agg')
        
        self.data, self.data_type = self._normalize_data(data)
        
        if self.gui_mode:
            _require_tk_gui()
            if root is None:
                raise ValueError("root parameter is required when gui=True")
            self.root = root
            self.root.title("Volcano Plot - Interactive")
            self.root.geometry(f"{self.GUI_WIDTH}x{self.GUI_HEIGHT}")
            self.tooltip = None
            self.root.protocol("WM_DELETE_WINDOW", self._exit_application)
            self._setup_ui()
            self.points_data = []
            self._draw_plot()
        else:
            self.root = None
            self.tooltip = None
            self.points_data = []
            self.fig, self.ax = plt.subplots(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))
            self._draw_plot_standalone()
        
        if self.save_path:
            self._save_plot()
            
        if not self.gui_mode:
            plt.close(self.fig)

    @staticmethod
    def load_config(path=None):
        """Load config.
        
        Args:
            path: Path value processed by this helper.
        
        Returns:
            object: Loaded config.
        """
        if path is None:
            path = VOLCANO_CONFIG_PATH
        resolved_path = _resolve_config_path(path)
        try:
            with open(resolved_path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"     Config not found at {resolved_path}, using defaults.")
            cfg = {}
        return cfg
    
    def _dprint(self, *args, **kwargs):
        """Print only if debugging_print is enabled.
        
        Args:
            *args: Additional positional arguments forwarded by this function.
            **kwargs: Additional keyword arguments forwarded by this function.
        
        Returns:
            None: Debug output is emitted only for its side effects.
        """
        if self.debugging_print:
            print(*args, **kwargs)
    
    def _normalize_data(self, data):
        """Normalize data.
        
        Args:
            data: Data processed by this function.
        
        Returns:
            tuple: Normalized data.
        """
        df = data.copy()
        
        if 'Kinase' in df.columns:
            data_type = 'UKA'
            statistic_col = self.x_axis_col
            if statistic_col is None:
                if 'KinaseStatistic' in df.columns:
                    statistic_col = 'KinaseStatistic'
                elif 'MeanPeptideStatistic' in df.columns:
                    statistic_col = 'MeanPeptideStatistic'
                elif 'MedianPeptideStatistic' in df.columns:
                    statistic_col = 'MedianPeptideStatistic'
                else:
                    statistic_col = 'Delta'
            if statistic_col not in df.columns:
                raise ValueError(
                    f"Requested UKA volcano x-axis column '{statistic_col}' is not present in the data."
                )
            df = df.rename(columns={'Kinase': 'label', statistic_col: 'delta'})
            
            if self.y_axis == 'z_score':
                df['y_value'] = df['Z_Score_abs'] if 'Z_Score_abs' in df.columns else df['Z_Score'].abs()
                df['y_label'] = '|Z-Score|'
                df['_significant'] = (df['Z_Score'].abs() >= self.z_threshold) & \
                                     (np.abs(df['delta']) >= self.delta_threshold)
            else:
                df['y_value'] = -np.log10(df['p_value'])
                df['y_label'] = '-log10(p-value)'
                df['_significant'] = (df['p_value'] <= self.p_threshold) & \
                                     (np.abs(df['delta']) >= self.delta_threshold)
            
            df['_hover_extra'] = df.apply(
                lambda r: f"{self.x_label}: {r['delta']:.3f} | Z: {r['Z_Score']:.2f} | p: {r['p_value']:.2e} | "
                          f"Type: {r.get('Type', 'N/A')} | Substrates: {r.get('NumSubstrates', 'N/A')}",
                axis=1
            )
        else:
            data_type = 'Peptide'
            
            if 'UniprotAccession' in df.columns:
                df['label'] = df['UniprotAccession']
            elif 'Gene name' in df.columns:
                df['label'] = df['Gene name']
            else:
                df['label'] = [f'Peptide_{i}' for i in range(len(df))]
            
            df['y_value'] = -np.log10(df['p_value'])
            df['y_label'] = '-log10(p-value)'
            df['_significant'] = (df['p_value'] <= self.p_threshold) & \
                                 (np.abs(df['delta']) >= self.delta_threshold)
            df['_hover_extra'] = df.apply(
                lambda r: f"Δ: {r['delta']:.3f} | p: {r['p_value']:.2e}", axis=1
            )
        
        return df, data_type
    
    def _get_colors_sizes(self):
        """Get colors sizes.
        
        Args:
            None.
        
        Returns:
            tuple: Requested colors sizes.
        """
        colors = []
        sizes = []
        alphas = []
        
        for _, row in self.data.iterrows():
            if not row['_significant']:
                colors.append(self.SC_COLOR_NS)
                sizes.append(self.SC_SIZE_NS)
                alphas.append(self.SC_ALPHA_NS)
            elif row['delta'] > 0:
                colors.append(self.SC_COLOR_UP)
                sizes.append(self.SC_SIZE_SIG)
                alphas.append(self.SC_ALPHA_SIG)
            else:
                colors.append(self.SC_COLOR_DOWN)
                sizes.append(self.SC_SIZE_SIG)
                alphas.append(self.SC_ALPHA_SIG)
        
        return colors, sizes, alphas
    
    def _setup_ui(self):
        """Return setup ui.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        title = f"Volcano Plot — {self.data_type} data"
        if self.data_type == 'UKA':
            title += f" (y-axis: {self.y_axis})"
        ttk.Label(control_frame, text=title, 
                  font=(self.GUI_FONT_FAMILY, self.GUI_FONT_SIZE, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="EXIT", 
                   command=self._exit_application).pack(side=tk.RIGHT, padx=5)
        
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.info_label = ttk.Label(info_frame, text="", foreground="blue")
        self.info_label.pack(side=tk.LEFT)
    
    def _draw_common(self):
        """Draw common.
        
        Args:
            None.
        
        Returns:
            tuple: Drawn representation of common.
        """
        self.ax.clear()
        
        colors, sizes, alphas = self._get_colors_sizes()
        
        for i, (_, row) in enumerate(self.data.iterrows()):
            self.ax.scatter(row['delta'], row['y_value'], 
                          c=colors[i], s=sizes[i], alpha=alphas[i])
        
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.SC_COLOR_NS, 
                   markersize=self.LG_MARKER_NS, alpha=self.SC_ALPHA_NS, label='not significant'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.SC_COLOR_UP, 
                   markersize=self.LG_MARKER_SIG, alpha=self.SC_ALPHA_SIG, label='upregulated'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.SC_COLOR_DOWN, 
                   markersize=self.LG_MARKER_SIG, alpha=self.SC_ALPHA_SIG, label='downregulated')
        ]
        self.ax.legend(
            handles=legend_elements, 
            fontsize=self.LG_FONTSIZE,
            markerscale=self.LG_MARKERSCALE,
            loc=self.LG_LOC,
            framealpha=self.LG_FRAMEALPHA,
            edgecolor=self.LG_EDGECOLOR,
            handlelength=self.LG_HANDLELENGTH,
            handleheight=self.LG_HANDLEHEIGHT,
            labelspacing=self.LG_LABELSPACING,
            borderpad=self.LG_BORDERPAD,
            handletextpad=self.LG_HANDLETEXTPAD
        )
        
        # Threshold lines
        if self.y_axis == 'p_value':
            self.ax.axhline(-np.log10(self.p_threshold),
                           color=self.TL_P_COLOR, linestyle=self.TL_P_LINESTYLE, linewidth=self.TL_P_LINEWIDTH)
        else:
            self.ax.axhline(self.z_threshold,
                           color=self.TL_Z_COLOR, linestyle=self.TL_Z_LINESTYLE, linewidth=self.TL_Z_LINEWIDTH)
            self.ax.axhline(-self.z_threshold,
                           color=self.TL_Z_COLOR, linestyle=self.TL_Z_LINESTYLE, linewidth=self.TL_Z_LINEWIDTH)
        
        self.ax.axvline(0, color=self.TL_CENTER_COLOR, linestyle=self.TL_CENTER_LINESTYLE,
                       linewidth=self.TL_CENTER_LINEWIDTH)
        
        if self.delta_threshold > 0:
            self.ax.axvline(self.delta_threshold,
                           color=self.TL_DELTA_COLOR, linestyle=self.TL_DELTA_LINESTYLE, linewidth=self.TL_DELTA_LINEWIDTH)
            self.ax.axvline(-self.delta_threshold,
                           color=self.TL_DELTA_COLOR, linestyle=self.TL_DELTA_LINESTYLE, linewidth=self.TL_DELTA_LINEWIDTH)
        
        # Labels
        y_label = self.data['y_label'].iloc[0]
        self.ax.set_xlabel(self.x_label, fontsize=self.AXES_LABEL_FONTSIZE)
        self.ax.set_ylabel(y_label, fontsize=self.AXES_LABEL_FONTSIZE)
        
        plot_title = f'Volcano Plot ({self.data_type})'
        self.ax.set_title(plot_title, fontsize=self.TITLE_FONTSIZE)
        self.ax.grid(True, alpha=self.GRID_ALPHA)
        
        # Stats
        n_sig = self.data['_significant'].sum()
        sig_up = (self.data['_significant'] & (self.data['delta'] > 0)).sum()
        sig_down = (self.data['_significant'] & (self.data['delta'] < 0)).sum()
        
        return n_sig, sig_up, sig_down
    
    def _draw_plot(self):
        """Draw plot.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.points_data = []
        n_sig, sig_up, sig_down = self._draw_common()
        
        for _, row in self.data.iterrows():
            self.points_data.append({
                'x': row['delta'], 'y': row['y_value'], 'data': row
            })
        
        self.info_label.config(
            text=f"Significant: {n_sig} | Up: {sig_up} | Down: {sig_down} | "
                 f"p-thr: {self.p_threshold} | z-thr: {self.z_threshold} | "
                 f"delta-thr: {self.delta_threshold}"
        )
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.draw()
    
    def _draw_plot_standalone(self):
        """Draw plot standalone.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        n_sig, sig_up, sig_down = self._draw_common()
        self._dprint(f"Significant: {n_sig} | Up: {sig_up} | Down: {sig_down}")
    
    def _save_plot(self):
        """Save plot.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        try:
            save_path = Path(self.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not save_path.suffix:
                save_path = save_path.with_suffix(f'.{self.image_format}')
            self.fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
            self._dprint(f" Plot saved to: {save_path.absolute()}")
        except Exception as e:
            print(f" Error saving plot: {e}")
    
    def _on_hover(self, event):
        """Return on hover.
        
        Args:
            event: Event processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if event.inaxes != self.ax:
            self._hide_tooltip()
            return
        
        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None:
            self._hide_tooltip()
            return
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        min_distance = float('inf')
        closest_point = None
        
        for point in self.points_data:
            dx = (point['x'] - mouse_x) / x_range
            dy = (point['y'] - mouse_y) / y_range
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.HOVER_THRESHOLD and distance < min_distance:
                min_distance = distance
                closest_point = point
        
        if closest_point:
            self._show_tooltip(event, closest_point['data'])
        else:
            self._hide_tooltip()
    
    def _show_tooltip(self, event, row_data):
        """Return show tooltip.
        
        Args:
            event: Event processed by this function.
            row_data: Row data processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self._hide_tooltip()
        
        label = row_data.get('label', 'Unknown')
        extra = row_data.get('_hover_extra', '')
        info_text = f"{label}\n{extra}"
        
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        x = self.root.winfo_pointerx() + self.TT_OFFSET_X
        y = self.root.winfo_pointery() + self.TT_OFFSET_Y
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(self.tooltip, relief=tk.SOLID, borderwidth=self.TT_BORDERWIDTH, padding=str(self.TT_PADDING))
        frame.pack()
        ttk.Label(frame, text=info_text, justify=tk.LEFT,
                 background=self.TT_BACKGROUND, relief=tk.FLAT).pack()
    
    def _hide_tooltip(self, event=None):
        """Return hide tooltip.
        
        Args:
            event: Event processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
    
    def _exit_application(self):
        """Return exit application.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        try:
            self._hide_tooltip()
            self.root.destroy()
        except Exception as e:
            print(f"Error during exit: {e}")
            try:
                self.root.destroy()
            except:
                pass


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
        """Initialize the VennDiagramPlot instance.
        
        Args:
            groups (Mapping[str, Iterable] | Sequence[tuple[str, Iterable]]): Groups processed by this function.
            title (str | None): Title processed by this function.
            item_label (str): Item label used by this function.
            save_path (str | Path | None): Path to the save.
            save_tables_dir (str | Path | None): Directory containing or receiving the save tables.
            plot_type (str): Plot type used by this function.
            case_sensitive (bool): Case sensitive used by this function.
            strip_items (bool): Strip items used by this function.
            show_percent (bool): Show percent used by this function.
            max_upset_intersections (int): Max upset intersections used by this function.
            figsize (tuple[float, float] | None): Figsize processed by this function.
            dpi (int): Dpi used by this function.
            image_format (str): Image format used by this function.
            colors (Sequence[str] | None): Colors processed by this function.
            debugging_print (bool): Whether to print additional debug information.
        
        Returns:
            None: Constructors initialize object state in place.
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
        """Create a plot from one column in multiple DataFrames.
        
        Args:
            dataframes (Mapping[str, pd.DataFrame]): Pandas DataFrame containing dataframes.
            column (str): Column used by this function.
            **kwargs: Additional keyword arguments forwarded by this function.
        
        Returns:
            "VennDiagramPlot": From dataframes.
        """
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
        
        Returns:
            None: Debug output is emitted only for its side effects.
        """
        if self.debugging_print:
            print(message)

    def _normalize_groups(
        self,
        groups: Mapping[str, Iterable] | Sequence[tuple[str, Iterable]],
    ) -> dict[str, set[str]]:
        """Normalize groups.
        
        Args:
            groups (Mapping[str, Iterable] | Sequence[tuple[str, Iterable]]): Groups processed by this function.
        
        Returns:
            dict[str, set[str]]: Normalized groups.
        """
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
        
        Returns:
            set[str]: Normalized item set.
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
        """Normalize item.
        
        Args:
            value: Input value processed by this helper.
        
        Returns:
            str | None: Normalized item.
        """
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
        """Compute exact regions.
        
        Args:
            None.
        
        Returns:
            dict[tuple[str, ...], set[str]]: Computed exact regions.
        """
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
        """Return region count.
        
        Args:
            *names (str): Additional positional arguments forwarded by this function.
        
        Returns:
            int: Region count.
        """
        return len(self.region_sets.get(tuple(names), set()))

    def _format_count(self, count: int) -> str:
        """Format count.
        
        Args:
            count (int): Count used by this function.
        
        Returns:
            str: Formatted count.
        """
        if not self.show_percent:
            return str(count)
        total = len(self.universe)
        percent = 0 if total == 0 else 100 * count / total
        return f"{count}\n({percent:.1f}%)"

    def summary_table(self) -> pd.DataFrame:
        """Return summary table.
        
        Args:
            None.
        
        Returns:
            pd.DataFrame: Summary table.
        """
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
        """Return pairwise overlap table.
        
        Args:
            None.
        
        Returns:
            pd.DataFrame: Pairwise overlap table.
        """
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
        """Return membership table.
        
        Args:
            None.
        
        Returns:
            pd.DataFrame: Membership table.
        """
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
        """Return region table.
        
        Args:
            include_items (bool): Boolean flag controlling whether to include items.
        
        Returns:
            pd.DataFrame: Region table.
        """
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
        """Save tables.
        
        Args:
            output_dir (str | Path | None): Directory containing or receiving the output.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
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
        """Handle plot.
        
        Args:
            None.
        
        Returns:
            object: Plot.
        """
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
        """Save figure.
        
        Args:
            fig: Fig processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        path = self.save_path
        if path.suffix == "":
            path = path.with_suffix(f".{self.image_format}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", format=path.suffix.lstrip("."))
        self._dprint(f"Saved Venn diagram: {path}")

    def _plot_venn(self):
        """Plot venn.
        
        Args:
            None.
        
        Returns:
            object: Plot output for venn.
        """
        if self.n_groups == 1:
            return self._plot_venn_1()
        if self.n_groups == 2:
            return self._plot_venn_2()
        if self.n_groups == 3:
            return self._plot_venn_3()
        raise ValueError("Circle Venn plots support one to three groups.")

    def _new_venn_figure(self, default_size: tuple[float, float]):
        """Return new venn figure.
        
        Args:
            default_size (tuple[float, float]): Default size processed by this function.
        
        Returns:
            tuple: New venn figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize or default_size)
        ax.set_aspect("equal")
        ax.axis("off")
        if self.title:
            ax.set_title(self.title, fontsize=14, pad=16)
        self.fig = fig
        self.ax = ax
        return fig, ax

    def _wrapped_label(self, name: str) -> str:
        """Return wrapped label.
        
        Args:
            name (str): Name used by this function.
        
        Returns:
            str: Wrapped label.
        """
        return textwrap.fill(f"{name}\n(n={len(self.group_sets[name])})", width=24)

    def _plot_venn_1(self):
        """Plot venn 1.
        
        Args:
            None.
        
        Returns:
            object: Plot output for venn 1.
        """
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
        """Plot venn 2.
        
        Args:
            None.
        
        Returns:
            object: Plot output for venn 2.
        """
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
        """Plot venn 3.
        
        Args:
            None.
        
        Returns:
            object: Plot output for venn 3.
        """
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
        """Plot upset.
        
        Args:
            None.
        
        Returns:
            object: Plot output for upset.
        """
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
    Interactive Heatmap Visualizer for UKA (upstream kinase analysis) pathway enrichment.
    """

    def __init__(self, 
                 root=None, 
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
                 gui=True, 
                 data_source=None,
                 debugging_print=False):
        
        """Initialize the HeatmapPlot_UKA instance.
        
        Args:
            root: Root object or container used by this function.
            enrichment_data: Enrichment data processed by this function.
            results_data: Results data processed by this function.
            y_axis: Y axis processed by this function.
            value_col: Value col processed by this function.
            value_label: Value label processed by this function.
            delta_threshold: Threshold value used to filter, classify, or flag results.
            p_threshold: Threshold value used to filter, classify, or flag results.
            z_threshold: Threshold value used to filter, classify, or flag results.
            save_path: Path to the save.
            dpi: Dpi processed by this function.
            image_format: Image format processed by this function.
            gui: Whether to use the interactive GUI code path.
            data_source: Data source processed by this function.
            debugging_print: Whether to print additional debug information.
        
        Returns:
            None: Constructors initialize object state in place.
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
        self.CMAP = plot.get('cmap', 'RdBu_r')
        self.DPI = plot.get('dpi', 300)
        self.IMAGE_FORMAT = plot.get('image_format', 'png')
        self.TITLE_PAD = plot.get('title_pad', 20)

        cbar = cfg.get('colorbar', {})
        self.CBAR_LABEL_FONTSIZE = cbar.get('label_fontsize', 12)
        self.CBAR_WIDTH_FACTOR = cbar.get('width_factor', 2)
        self.CBAR_HEIGHT_ROWS = cbar.get('height_rows', 20)
        self.CBAR_OFFSET_X = cbar.get('offset_x', 0.05)

        axes = cfg.get('axes', {})
        self.AXES_LABEL_FONTSIZE = axes.get('label_fontsize', 12)
        self.TITLE_FONTSIZE = axes.get('title_fontsize', 14)
        self.X_TICK_ROTATION = axes.get('x_tick_rotation', 45)
        self.X_TICK_FONTSIZE = axes.get('x_tick_fontsize', 8)
        self.Y_TICK_FONTSIZE = axes.get('y_tick_fontsize', 9)
        self.X_TICK_HA = axes.get('x_tick_ha', 'right')

        tt = cfg.get('tooltip', {})
        self.TT_OFFSET_X = tt.get('offset_x', 10)
        self.TT_OFFSET_Y = tt.get('offset_y', 10)
        self.TT_BACKGROUND = tt.get('background', 'lightyellow')
        self.TT_BORDERWIDTH = tt.get('borderwidth', 1)
        self.TT_PADDING = tt.get('padding', 5)
        self.TT_MAX_LABEL_LEN = tt.get('max_label_length', 50)

        gui_cfg = cfg.get('gui', {})
        self.GUI_WIDTH = gui_cfg.get('window_width', 1400)
        self.GUI_HEIGHT = gui_cfg.get('window_height', 1000)
        self.GUI_FONT_FAMILY = gui_cfg.get('font_family', 'Arial')
        self.GUI_FONT_SIZE = gui_cfg.get('font_size', 12)

        self.enrichment_data = enrichment_data
        self.gui_mode = gui
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
        self.debugging_print = debugging_print
        if not self.gui_mode:
            matplotlib.use('Agg')

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

        if self.gui_mode:
            _require_tk_gui()
            if root is None:
                raise ValueError("root parameter is required when gui=True")
            self.root = root
            self.root.title("Kinase Heatmap - Interactive")
            self.root.geometry(f"{self.GUI_WIDTH}x{self.GUI_HEIGHT}")
            self.tooltip = None
            self.info_label = None
            self.root.protocol("WM_DELETE_WINDOW", self._exit_application)
            self._prepare_data()
            self._calc_fig_size()
            self._setup_ui()
            self._draw_heatmap()
        else:
            self.root = None
            self.tooltip = None
            self.info_label = None
            self._prepare_data()
            self._calc_fig_size()
            self.fig, self.ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
            self._draw_heatmap_standalone()

        if self.save_path:
            self._save_plot()

        if not self.gui_mode:
            plt.close(self.fig)

    @staticmethod
    def load_config(path=None):
        """Load config.
        
        Args:
            path: Path value processed by this helper.
        
        Returns:
            object: Loaded config.
        """
        if path is None:
            path = HEATMAP_CONFIG_PATH
        resolved_path = _resolve_config_path(path)
        try:
            with open(resolved_path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"     Config not found at {resolved_path}, using defaults.")
            cfg = {}
        return cfg
    
    def _dprint(self, *args, **kwargs):
        """Print only if debugging_print is enabled.
        
        Args:
            *args: Additional positional arguments forwarded by this function.
            **kwargs: Additional keyword arguments forwarded by this function.
        
        Returns:
            None: Debug output is emitted only for its side effects.
        """
        if self.debugging_print:
            print(*args, **kwargs)

    def _calc_fig_size(self):
        """Return calc fig size.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        n_rows = len(self.pathways)
        n_cols = len(self.kinases)

        max_label_len = max(len(p) for p in self.pathways)
        self._margin_left = max(self.MARGIN_LEFT, max_label_len * 0.075)

        self._fig_width = self._margin_left + n_cols * self.CELL_WIDTH + self.MARGIN_RIGHT
        self._fig_height = self.MARGIN_TOP + n_rows * self.CELL_HEIGHT + self.MARGIN_BOTTOM

    def _prepare_data(self):
        """Prepare data.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        value_dict = {}
        for _, row in self.results_data.iterrows():
            kinase = row[self.label_col]
            value = row[self.value_col]
            value_dict[kinase] = value

        kinases_with_val = sorted(value_dict.items(), key=lambda x: x[1])
        self.kinases = [k for k, _ in kinases_with_val]

        pathway_kinases = {}
        for _, row in self.enrichment_data.iterrows():
            pathway_name = row['name']
            intersection = row['intersections']
            if isinstance(intersection, str):
                genes = intersection.replace('[', '').replace(']', '').replace("'", "").split(',')
                genes = [g.strip() for g in genes if g.strip()]
            else:
                genes = list(intersection)
            pathway_kinases[pathway_name] = genes

        self.pathway_names = list(pathway_kinases.keys())

        kinases_in_pathways = set()
        for genes in pathway_kinases.values():
            kinases_in_pathways.update(genes)
        self.kinases = [k for k in self.kinases if k in kinases_in_pathways]

        all_row = [value_dict.get(k, 0) for k in self.kinases]

        matrix = [all_row]
        for pathway in self.pathway_names:
            row_vals = []
            pw_kinases = pathway_kinases[pathway]
            for kinase in self.kinases:
                if kinase in pw_kinases:
                    row_vals.append(value_dict.get(kinase, 0))
                else:
                    row_vals.append(np.nan)
            matrix.append(row_vals)

        self.pathways = ['All Kinases'] + self.pathway_names
        self.heatmap_data = pd.DataFrame(matrix, index=self.pathways, columns=self.kinases)

        n_pathways = len(self.pathways)
        n_kinases = len(self.kinases)
        n_filled = int((~self.heatmap_data.isna()).sum().sum())

        info_text = (f"Value: {self.value_col} | "
                     f"Pathways: {n_pathways} | Kinases: {n_kinases} | "
                     f"Filled cells: {n_filled}")

        if self.gui_mode and self.info_label is not None:
            self.info_label.config(text=info_text)
        else:
            self._dprint(info_text)

    def _get_heatmap_params(self):
        """Get heatmap params.
        
        Args:
            None.
        
        Returns:
            dict: Requested heatmap params.
        """
        if self.value_col == 'Z_Score':
            vmin, vmax = -3, 3
        else:
            vmin, vmax = -0.5, 0.5

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
        """Apply fixed layout.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.fig.subplots_adjust(
            left=self._margin_left / self._fig_width,
            right=1.0 - self.MARGIN_RIGHT / self._fig_width,
            top=1.0 - self.MARGIN_TOP / self._fig_height,
            bottom=self.MARGIN_BOTTOM / self._fig_height
        )

    def _format_axes(self):
        """Format axes.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        value_name = self.value_col
        self.ax.set_xlabel(f'Kinase (ordered by {value_name}, low → high)', fontsize=self.AXES_LABEL_FONTSIZE)
        self.ax.set_ylabel('Pathway', fontsize=self.AXES_LABEL_FONTSIZE)
        self.ax.set_title(f'Kinase Heatmap — {value_name}', fontsize=self.TITLE_FONTSIZE, pad=self.TITLE_PAD)
        self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=self.X_TICK_ROTATION, ha=self.X_TICK_HA, fontsize=self.X_TICK_FONTSIZE)
        self.ax.set_yticklabels(self.ax.get_yticklabels(), rotation=0, fontsize=self.Y_TICK_FONTSIZE)
        self._apply_fixed_layout()

    def _add_colorbar(self, heatmap_obj):
        """Return add colorbar.
        
        Args:
            heatmap_obj: Heatmap obj processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
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
        """Draw heatmap.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.ax.clear()
        params = self._get_heatmap_params()
        heatmap_obj = sns.heatmap(self.heatmap_data, ax=self.ax, **params)
        self._format_axes()
        self._add_colorbar(heatmap_obj)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.draw()

    def _draw_heatmap_standalone(self):
        """Draw heatmap standalone.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.ax.clear()
        params = self._get_heatmap_params()
        heatmap_obj = sns.heatmap(self.heatmap_data, ax=self.ax, **params)
        self._format_axes()
        self._add_colorbar(heatmap_obj)

    def _setup_ui(self):
        """Return setup ui.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        title = f"Kinase Heatmap — UKA data ({self.data_source if self.data_source else ''})".strip()
        title += f" (value: {self.value_col})"
        ttk.Label(control_frame, text=title,
                  font=(self.GUI_FONT_FAMILY, self.GUI_FONT_SIZE, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="EXIT",
                   command=self._exit_application).pack(side=tk.RIGHT, padx=5)

        scroll_container = ttk.Frame(self.root)
        scroll_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.scroll_canvas = tk.Canvas(scroll_container)
        scrollbar_y = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        scrollbar_x = ttk.Scrollbar(scroll_container, orient=tk.HORIZONTAL, command=self.scroll_canvas.xview)

        self.scroll_frame = ttk.Frame(self.scroll_canvas)
        self.scroll_frame.bind("<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_canvas.bind_all("<MouseWheel>",
            lambda e: self.scroll_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.fig, self.ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.scroll_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.info_label = ttk.Label(info_frame, text="", foreground="blue")
        self.info_label.pack(side=tk.LEFT)

    def _on_hover(self, event):
        """Return on hover.
        
        Args:
            event: Event processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if event.inaxes != self.ax:
            self._hide_tooltip()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self._hide_tooltip()
            return

        col_idx = int(x)
        row_idx = int(y)

        if 0 <= col_idx < len(self.kinases) and 0 <= row_idx < len(self.pathways):
            kinase = self.kinases[col_idx]
            pathway = self.pathways[row_idx]
            value = self.heatmap_data.iloc[row_idx, col_idx]

            if not np.isnan(value):
                self._show_tooltip(event, kinase, pathway, value, in_pathway=True)
            else:
                self._show_tooltip(event, kinase, pathway, value, in_pathway=False)
        else:
            self._hide_tooltip()

    def _show_tooltip(self, event, kinase, pathway, value, in_pathway=True):
        """Return show tooltip.
        
        Args:
            event: Event processed by this function.
            kinase: Kinase processed by this function.
            pathway: Pathway processed by this function.
            value: Input value processed by this helper.
            in_pathway: In pathway processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self._hide_tooltip()

        pathway_display = pathway if len(pathway) <= self.TT_MAX_LABEL_LEN else pathway[:self.TT_MAX_LABEL_LEN - 3] + "..."

        if in_pathway:
            info_text = f"Kinase: {kinase}\nPathway: {pathway_display}\n{self.value_col}: {value:.3f}"
        else:
            info_text = f"Kinase: {kinase}\nPathway: {pathway_display}\nNot in pathway"

        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        x = self.root.winfo_pointerx() + self.TT_OFFSET_X
        y = self.root.winfo_pointery() + self.TT_OFFSET_Y
        self.tooltip.wm_geometry(f"+{x}+{y}")

        frame = ttk.Frame(self.tooltip, relief=tk.SOLID, borderwidth=self.TT_BORDERWIDTH, padding=str(self.TT_PADDING))
        frame.pack()
        ttk.Label(frame, text=info_text, justify=tk.LEFT,
                 background=self.TT_BACKGROUND, relief=tk.FLAT).pack()

    def _hide_tooltip(self, event=None):
        """Return hide tooltip.
        
        Args:
            event: Event processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def _save_plot(self):
        """Save plot.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        try:
            save_path = Path(self.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not save_path.suffix:
                save_path = save_path.with_suffix(f'.{self.image_format}')
            self.fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=self.image_format)
            self._dprint(f"Heatmap saved to: {save_path.absolute()}")
        except Exception as e:
            print(f"Error saving heatmap: {e}")

    def _exit_application(self):
        """Return exit application.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        try:
            self._hide_tooltip()
            self.root.destroy()
        except Exception as e:
            print(f"Error during exit: {e}")
            try:
                self.root.destroy()
            except:
                pass


class HeatmapPlot_Peptides:
    """
    Heatmap visualization for peptide-level limma statistics.

    Expects a DataFrame with the following columns:
        - ID: peptide identifier
        - control_sample_1 to control_sample_4: values for control samples
        - treatment_sample_1 to treatment_sample_4: values for test samples
        - optional control_label_1 / treatment_label_1 etc. for descriptive
          sample labels (for example bio/tech replicate combinations)
        - p_value: p-value from the limma analysis (optional)
    """

    def __init__(self, 
                 root=None,
                 data=None,
                 csv_path=None,
                 significance_threshold=0.05,
                 cmap='RdYlBu_r',
                 save_path=None,
                 dpi=300,
                 image_format='png',
                 gui=False,
                 title=None,
                 debugging_print=False):
        
        """Initialize the HeatmapPlot_Peptides instance.
        
        Args:
            root: Root object or container used by this function.
            data: Data processed by this function.
            csv_path: Path to the CSV.
            significance_threshold: Threshold value used to filter, classify, or flag results.
            cmap: Cmap processed by this function.
            save_path: Path to the save.
            dpi: Dpi processed by this function.
            image_format: Image format processed by this function.
            gui: Whether to use the interactive GUI code path.
            title: Title processed by this function.
            debugging_print: Whether to print additional debug information.
        
        Returns:
            None: Constructors initialize object state in place.
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
        self.CMAP = plot.get('cmap', 'RdBu_r')
        self.DPI = plot.get('dpi', 300)
        self.IMAGE_FORMAT = plot.get('image_format', 'png')
        self.TITLE_PAD = plot.get('title_pad', 20)
        self.ZSCORE_ABS_MAX = float(plot.get('zscore_abs_max', 5.0))

        cbar = cfg.get('colorbar', {})
        self.CBAR_LABEL_FONTSIZE = cbar.get('label_fontsize', 12)
        self.CBAR_WIDTH_FACTOR = cbar.get('width_factor', 2)
        self.CBAR_HEIGHT_ROWS = cbar.get('height_rows', 20)
        self.CBAR_OFFSET_X = cbar.get('offset_x', 0.05)

        axes = cfg.get('axes', {})
        self.AXES_LABEL_FONTSIZE = axes.get('label_fontsize', 12)
        self.TITLE_FONTSIZE = axes.get('title_fontsize', 14)
        self.X_TICK_ROTATION = axes.get('x_tick_rotation', 45)
        self.X_TICK_FONTSIZE = axes.get('x_tick_fontsize', 8)
        self.Y_TICK_FONTSIZE = axes.get('y_tick_fontsize', 9)
        self.X_TICK_HA = axes.get('x_tick_ha', 'left')

        tt = cfg.get('tooltip', {})
        self.TT_OFFSET_X = tt.get('offset_x', 10)
        self.TT_OFFSET_Y = tt.get('offset_y', 10)
        self.TT_BACKGROUND = tt.get('background', 'lightyellow')
        self.TT_BORDERWIDTH = tt.get('borderwidth', 1)
        self.TT_PADDING = tt.get('padding', 5)
        self.TT_MAX_LABEL_LEN = tt.get('max_label_length', 50)

        gui_cfg = cfg.get('gui', {})
        self.GUI_WIDTH = gui_cfg.get('window_width', 1400)
        self.GUI_HEIGHT = gui_cfg.get('window_height', 1000)
        self.GUI_FONT_FAMILY = gui_cfg.get('font_family', 'Arial')
        self.GUI_FONT_SIZE = gui_cfg.get('font_size', 12)

        self.gui_mode = gui
        self.debugging_print = debugging_print
        self.significance_threshold = significance_threshold
        self.cmap = cmap
        self.save_path = save_path
        self.dpi = dpi
        self.image_format = image_format.lower()
        self.title = title
        self.tooltip = None
        
        if not self.gui_mode:
            matplotlib.use('Agg')
        
        if data is not None:
            self.df = data.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either data or csv_path must be provided")
        
        if self.gui_mode:
            _require_tk_gui()
            if root is None:
                raise ValueError("The root parameter is required when gui=True")
            self.root = root
            self.root.title("Peptide Heatmap - Interactive")
            self.root.geometry(f"{self.GUI_WIDTH}x{self.GUI_HEIGHT}")
            self.root.protocol("WM_DELETE_WINDOW", self._exit_application)
            self._prepare_data()
            self._calc_fig_size()
            self._setup_ui()
            self._draw_heatmap()
        else:
            self.root = None
            self._prepare_data()
            self._calc_fig_size()
            self.fig, self.ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
            self._draw_heatmap_standalone()
        
        if self.save_path:
            self._save_plot()
        
        if not self.gui_mode:
            plt.close(self.fig)

    @staticmethod
    def load_config(path=None):
        """Load config.
        
        Args:
            path: Path value processed by this helper.
        
        Returns:
            object: Loaded config.
        """
        if path is None:
            path = HEATMAP_CONFIG_PATH
        resolved_path = _resolve_config_path(path)
        try:
            with open(resolved_path) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"     Config not found at {resolved_path}, using defaults.")
            cfg = {}
        return cfg
    
    def _dprint(self, *args, **kwargs):
        """Print only if debugging_print is enabled.
        
        Args:
            *args: Additional positional arguments forwarded by this function.
            **kwargs: Additional keyword arguments forwarded by this function.
        
        Returns:
            None: Debug output is emitted only for its side effects.
        """
        if self.debugging_print:
            print(*args, **kwargs)

    def _calc_fig_size(self):
        """Compute figure size based on the data.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        n_rows = len(self.peptides)
        n_cols = len(self.sample_labels)
        
        max_label_len = max(len(str(p)) for p in self.peptides)
        self._margin_left = max(self.MARGIN_LEFT, max_label_len * 0.075)
        
        self._fig_width = self._margin_left + n_cols * self.CELL_WIDTH + self.MARGIN_RIGHT
        self._fig_height = self.MARGIN_TOP + n_rows * self.CELL_HEIGHT + self.MARGIN_BOTTOM

    def _prepare_data(self):
        """Prepare data for the heatmap.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if 'p_value' in self.df.columns:
            df_filtered = self.df[self.df['p_value'] < self.significance_threshold].copy()
            self._dprint(f"     Filtered to {len(df_filtered)} significant peptides (p < {self.significance_threshold}) from {len(self.df)} total")
        elif 'adj_p_value' in self.df.columns:
            df_filtered = self.df[self.df['adj_p_value'] < self.significance_threshold].copy()
            self._dprint(f"     Filtered to {len(df_filtered)} significant peptides (adj_p < {self.significance_threshold}) from {len(self.df)} total")
        else:
            df_filtered = self.df.copy()
            self._dprint(f"     No p-value column found, showing all {len(df_filtered)} peptides")
        
        if len(df_filtered) == 0:
            raise ValueError(f"No significant peptides found with p-value < {self.significance_threshold}")
        
        if 'ID' in df_filtered.columns:
            self.peptides = df_filtered['ID'].tolist()
        elif 'peptide' in df_filtered.columns:
            self.peptides = df_filtered['peptide'].tolist()
        else:
            raise ValueError("Could not find an 'ID' or 'peptide' column")
        
        control_cols_zscore = [col for col in df_filtered.columns if 'control_sample' in col.lower() and '_zscore' in col.lower()]
        treatment_cols_zscore = [col for col in df_filtered.columns if 'treatment_sample' in col.lower() and '_zscore' in col.lower()]
        
        if control_cols_zscore and treatment_cols_zscore:
            candidate_control_cols = sorted(
                control_cols_zscore,
                key=lambda x: int(x.split('_')[2]),
            )
            candidate_treatment_cols = sorted(
                treatment_cols_zscore,
                key=lambda x: int(x.split('_')[2]),
            )
            candidate_matrix = np.hstack(
                [
                    df_filtered[candidate_treatment_cols].values,
                    df_filtered[candidate_control_cols].values,
                ]
            )
            if np.isfinite(candidate_matrix).any():
                self._dprint("     Using z-score values for heatmap")
                control_cols = candidate_control_cols
                treatment_cols = candidate_treatment_cols
                self.use_zscore = True
            else:
                self._dprint(
                    "     Z-score columns contain no finite values; falling back to raw values for heatmap"
                )
                control_cols = [
                    col for col in df_filtered.columns
                    if 'control_sample' in col.lower() and '_zscore' not in col.lower()
                ]
                treatment_cols = [
                    col for col in df_filtered.columns
                    if 'treatment_sample' in col.lower() and '_zscore' not in col.lower()
                ]
                control_cols = sorted(control_cols, key=lambda x: int(x.split('_')[2]))
                treatment_cols = sorted(treatment_cols, key=lambda x: int(x.split('_')[2]))
                self.use_zscore = False
        else:
            control_cols = [col for col in df_filtered.columns if 'control_sample' in col.lower() and '_zscore' not in col.lower()]
            treatment_cols = [col for col in df_filtered.columns if 'treatment_sample' in col.lower() and '_zscore' not in col.lower()]
            
            if not control_cols or not treatment_cols:
                raise ValueError("Could not find control_sample_* or treatment_sample_* columns")
            
            control_cols = sorted(control_cols, key=lambda x: int(x.split('_')[2]))
            treatment_cols = sorted(treatment_cols, key=lambda x: int(x.split('_')[2]))
            self.use_zscore = False
            self._dprint("     Using raw values for heatmap (no z-scores found)")
        
        control_labels = [
            self._resolve_sample_axis_label(
                df_filtered=df_filtered,
                group_prefix='control',
                sample_col=col,
                fallback=f'Control {i+1}',
            )
            for i, col in enumerate(control_cols)
        ]
        test_labels = [
            self._resolve_sample_axis_label(
                df_filtered=df_filtered,
                group_prefix='treatment',
                sample_col=col,
                fallback=f'Test {i+1}',
            )
            for i, col in enumerate(treatment_cols)
        ]
        self.sample_labels = test_labels + control_labels
        
        treatment_data = df_filtered[treatment_cols].values
        control_data = df_filtered[control_cols].values
        
        self.raw_matrix = np.hstack([treatment_data, control_data])
        self.matrix = self.raw_matrix.copy()
        if self.use_zscore:
            self._dprint(
                f"     Clipping heatmap display to +/-{self.ZSCORE_ABS_MAX:g} z-score"
            )
            self.matrix = np.clip(
                self.matrix,
                -self.ZSCORE_ABS_MAX,
                self.ZSCORE_ABS_MAX,
            )

        self.heatmap_data_raw = pd.DataFrame(
            self.raw_matrix,
            index=self.peptides,
            columns=self.sample_labels
        )
        self.heatmap_data = pd.DataFrame(
            self.matrix,
            index=self.peptides,
            columns=self.sample_labels
        )
        
        if 'p_value' in df_filtered.columns:
            self.p_values = df_filtered['p_value'].values
        elif 'adj_p_value' in df_filtered.columns:
            self.p_values = df_filtered['adj_p_value'].values
        else:
            self.p_values = None

    @staticmethod
    def _resolve_sample_axis_label(df_filtered, group_prefix, sample_col, fallback):
        """Resolve sample axis label.
        
        Args:
            df_filtered: Filtered pandas DataFrame used by this helper.
            group_prefix: Group prefix processed by this function.
            sample_col: Sample col processed by this function.
            fallback: Fallback processed by this function.
        
        Returns:
            object: Resolved sample axis label.
        """
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
        
        Args:
            None.
        
        Returns:
            dict: Requested heatmap params.
        """
        if self.use_zscore:
            return {
                'cmap': self.CMAP,
                'center': 0,
                'vmin': -self.ZSCORE_ABS_MAX,
                'vmax': self.ZSCORE_ABS_MAX,
                'linewidths': self.CELL_LINEWIDTH,
                'linecolor': self.CELL_LINECOLOR,
                'cbar': False,
                'xticklabels': True,
                'yticklabels': True,
            }

        data_min = np.nanmin(self.matrix)
        data_max = np.nanmax(self.matrix)
        
        abs_max = max(abs(data_min), abs(data_max))
        
        if abs_max < 3:
            vmin, vmax = -3, 3
        elif abs_max < 5:
            vmin, vmax = -5, 5
        else:
            vmin, vmax = -abs_max, abs_max
        
        return {
            'cmap': self.CMAP,
            'center': 0,
            'vmin': vmin,
            'vmax': vmax,
            'linewidths': self.CELL_LINEWIDTH,
            'linecolor': self.CELL_LINECOLOR,
            'cbar': False,
            'xticklabels': True,
            'yticklabels': True,
        }

    def _apply_fixed_layout(self):
        """Apply the layout using fixed margins.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.fig.subplots_adjust(
            left=self._margin_left / self._fig_width,
            right=1.0 - self.MARGIN_RIGHT / self._fig_width,
            top=1.0 - self.MARGIN_TOP / self._fig_height,
            bottom=self.MARGIN_BOTTOM / self._fig_height
        )

    def _format_axes(self):
        """Format the axes.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.ax.set_xlabel('Samples', fontsize=self.AXES_LABEL_FONTSIZE)
        self.ax.set_ylabel('Peptides', fontsize=self.AXES_LABEL_FONTSIZE)
        
        title = self.title or 'Peptide Expression'
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
        """Add a top-aligned peptide heatmap colorbar with fixed 12-cell height.
        
        Args:
            heatmap_obj: Heatmap obj processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        cbar_width_inches = 0.5 * self.CBAR_WIDTH_FACTOR * self.CELL_WIDTH
        cbar_height_inches = 12 * self.CELL_HEIGHT

        ax_pos = self.ax.get_position()
        cbar_width_fig = cbar_width_inches / self._fig_width
        cbar_height_fig = cbar_height_inches / self._fig_height

        cbar_x = ax_pos.x1 + self.CBAR_OFFSET_X
        cbar_y = ax_pos.y1 - cbar_height_fig

        cbar_ax = self.fig.add_axes([cbar_x, cbar_y, cbar_width_fig, cbar_height_fig])

        cbar_label = 'Z-Score' if self.use_zscore else 'Expression Value'
        cbar = self.fig.colorbar(heatmap_obj.collections[0], cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=self.CBAR_LABEL_FONTSIZE)

    def _draw_heatmap(self):
        """Draw the heatmap in GUI mode.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.ax.clear()
        params = self._get_heatmap_params()
        heatmap_obj = sns.heatmap(self.heatmap_data, ax=self.ax, **params)
        self._format_axes()
        self._apply_fixed_layout()
        self._add_colorbar(heatmap_obj)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.draw()

    def _draw_heatmap_standalone(self):
        """Draw the heatmap in non-GUI mode.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.ax.clear()
        params = self._get_heatmap_params()
        heatmap_obj = sns.heatmap(self.heatmap_data, ax=self.ax, **params)
        self._format_axes()
        self._apply_fixed_layout()
        self._add_colorbar(heatmap_obj)

    def _save_plot(self):
        """Save the heatmap.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
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

    def show(self):
        """Show the heatmap for interactive use.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        plt.show()

    def close(self):
        """Close the figure.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        plt.close(self.fig)

    def _setup_ui(self):
        """Set up the GUI interface.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        title = self.title or "Peptide Heatmap - Limma Results"
        ttk.Label(control_frame, text=title,
                  font=(self.GUI_FONT_FAMILY, self.GUI_FONT_SIZE, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="EXIT",
                   command=self._exit_application).pack(side=tk.RIGHT, padx=5)
        
        scroll_container = ttk.Frame(self.root)
        scroll_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.scroll_canvas = tk.Canvas(scroll_container)
        scrollbar_y = ttk.Scrollbar(scroll_container, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        scrollbar_x = ttk.Scrollbar(scroll_container, orient=tk.HORIZONTAL, command=self.scroll_canvas.xview)
        
        self.scroll_frame = ttk.Frame(self.scroll_canvas)
        self.scroll_frame.bind("<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        
        self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.scroll_canvas.bind_all("<MouseWheel>",
            lambda e: self.scroll_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        self.fig, self.ax = plt.subplots(figsize=(self._fig_width, self._fig_height))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.scroll_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_hover(self, event):
        """Show a tooltip when hovering over cells.
        
        Args:
            event: Event processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if event.inaxes != self.ax:
            self._hide_tooltip()
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self._hide_tooltip()
            return
        
        col_idx = int(x)
        row_idx = int(y)
        
        if 0 <= col_idx < len(self.sample_labels) and 0 <= row_idx < len(self.peptides):
            peptide = self.peptides[row_idx]
            sample = self.sample_labels[col_idx]
            value = self.heatmap_data_raw.iloc[row_idx, col_idx]
            
            self._show_tooltip(event, peptide, sample, value)
        else:
            self._hide_tooltip()

    def _show_tooltip(self, event, peptide, sample, value):
        """Show the tooltip.
        
        Args:
            event: Event processed by this function.
            peptide: Peptide processed by this function.
            sample: Sample processed by this function.
            value: Input value processed by this helper.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self._hide_tooltip()
        
        val_str = f'{value:.3f}' if not np.isnan(value) else 'N/A'
        text = f"Peptide: {peptide}\nSample: {sample}\nValue: {val_str}"
        
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        x = self.root.winfo_pointerx() + self.TT_OFFSET_X
        y = self.root.winfo_pointery() + self.TT_OFFSET_Y
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(self.tooltip, relief=tk.SOLID, borderwidth=self.TT_BORDERWIDTH, padding=str(self.TT_PADDING))
        frame.pack()
        ttk.Label(frame, text=text, justify=tk.LEFT,
                  background=self.TT_BACKGROUND, relief=tk.FLAT).pack()

    def _hide_tooltip(self, event=None):
        """Tooltip verstecken.
        
        Args:
            event: Event processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def _exit_application(self):
        """Anwendung beenden.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        try:
            self._hide_tooltip()
            self.root.destroy()
        except Exception as e:
            print(f"Error during exit: {e}")
            try:
                self.root.destroy()
            except:
                pass
