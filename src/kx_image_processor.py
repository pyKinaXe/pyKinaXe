"""
ImageProcessor - A class for processing PamChip microarray images.

This module provides the following functionality:
1. Applies circular masks to images based on aperture edge detection
   - Canny-based band method with gamma correction and 2D edge detection
2. Subtracts slowly varying background using morphological operations
3. Detects reference spot patterns (T-shape and J-shape)
4. Identifies the location of hte main spot grid
4. Computes intensities for grid spots
5. Associates peptides of the array with those in the human proteins

TO DO:
    add another fallback method (before the last one) to the method for finding reference spots,
    based on already identified reference spots in other images acquired for other exposure times.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage
import cv2
from joblib import Parallel, delayed
from scipy.ndimage import median_filter, maximum_filter
from scipy.optimize import minimize, differential_evolution
from numba import jit

# Local imports
from config.image_processor import IMAGE_PROCESSOR_CONFIG
from kx_data_importer import ArrayLayoutLoader, DataLoader


_REFERENCE_DIMENSIONS = IMAGE_PROCESSOR_CONFIG["reference_dimensions"]
_EDGE_DETECTION_STATS = IMAGE_PROCESSOR_CONFIG["edge_detection_statistics"]
_GRID_CONFIG = IMAGE_PROCESSOR_CONFIG["grid"]
_FILTERING_CONFIG = IMAGE_PROCESSOR_CONFIG["filtering"]
_INIT_DEFAULTS = IMAGE_PROCESSOR_CONFIG["init_defaults"]
_DEFAULT_ENRICHMENT_FILE = IMAGE_PROCESSOR_CONFIG["paths"]["default_enrichment_file"]


class ImageProcessor:
    """
    ImageProcessor - A class for processing PamChip microarray images.

    This class implements:
    1. Detecting and applying circular masks to images using Canny edge detection:
       gamma correction → 2D band preprocessing → 2D Canny edge detection
    2. Subtracting slowly varying background using morphological operations.
    3. Detecting reference spot patterns (T-shape and J-shape) for locating the main grid.
    4. Identifying and refining the location of the main spot grid using maximum-of-integrated-intensity-based optimization.
    5. Computing integrated, maximum, median, and saturation-corrected intensities for all grid spots.
    6. Associating each spot with human proteins.
    7. Exporting results and providing visualization methods for quality control.
    """
    
    # Reference image dimensions (697x520), used if scaling is needed in case of different image sizes
    REFERENCE_WIDTH = _REFERENCE_DIMENSIONS["width"]
    REFERENCE_HEIGHT = _REFERENCE_DIMENSIONS["height"]
    
    # Magic constants for edge detection (for 697x520 images)
    # Horizontal cross sections (6 total: 2 outward from top edge, top edge, bottom edge, 2 outward from bottom)
    H_LEFT_1_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["left_1"]["median"]
    H_LEFT_1_STD = _EDGE_DETECTION_STATS["horizontal"]["left_1"]["std"]
    H_RIGHT_1_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["right_1"]["median"]
    H_RIGHT_1_STD = _EDGE_DETECTION_STATS["horizontal"]["right_1"]["std"]
    H_LEFT_2_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["left_2"]["median"]
    H_LEFT_2_STD = _EDGE_DETECTION_STATS["horizontal"]["left_2"]["std"]
    H_RIGHT_2_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["right_2"]["median"]
    H_RIGHT_2_STD = _EDGE_DETECTION_STATS["horizontal"]["right_2"]["std"]
    H_LEFT_3_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["left_3"]["median"]
    H_LEFT_3_STD = _EDGE_DETECTION_STATS["horizontal"]["left_3"]["std"]
    H_RIGHT_3_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["right_3"]["median"]
    H_RIGHT_3_STD = _EDGE_DETECTION_STATS["horizontal"]["right_3"]["std"]
    H_LEFT_4_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["left_4"]["median"]
    H_LEFT_4_STD = _EDGE_DETECTION_STATS["horizontal"]["left_4"]["std"]
    H_RIGHT_4_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["right_4"]["median"]
    H_RIGHT_4_STD = _EDGE_DETECTION_STATS["horizontal"]["right_4"]["std"]
    H_LEFT_5_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["left_5"]["median"]
    H_LEFT_5_STD = _EDGE_DETECTION_STATS["horizontal"]["left_5"]["std"]
    H_RIGHT_5_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["right_5"]["median"]
    H_RIGHT_5_STD = _EDGE_DETECTION_STATS["horizontal"]["right_5"]["std"]
    H_LEFT_6_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["left_6"]["median"]
    H_LEFT_6_STD = _EDGE_DETECTION_STATS["horizontal"]["left_6"]["std"]
    H_RIGHT_6_MEDIAN = _EDGE_DETECTION_STATS["horizontal"]["right_6"]["median"]
    H_RIGHT_6_STD = _EDGE_DETECTION_STATS["horizontal"]["right_6"]["std"]
    # Vertical cross sections (6 total: 2 outward from left edge, left edge, right edge, 2 outward from right)
    V_TOP_1_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["top_1"]["median"]
    V_TOP_1_STD = _EDGE_DETECTION_STATS["vertical"]["top_1"]["std"]
    V_BOTTOM_1_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["bottom_1"]["median"]
    V_BOTTOM_1_STD = _EDGE_DETECTION_STATS["vertical"]["bottom_1"]["std"]
    V_TOP_2_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["top_2"]["median"]
    V_TOP_2_STD = _EDGE_DETECTION_STATS["vertical"]["top_2"]["std"]
    V_BOTTOM_2_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["bottom_2"]["median"]
    V_BOTTOM_2_STD = _EDGE_DETECTION_STATS["vertical"]["bottom_2"]["std"]
    V_TOP_3_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["top_3"]["median"]
    V_TOP_3_STD = _EDGE_DETECTION_STATS["vertical"]["top_3"]["std"]
    V_BOTTOM_3_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["bottom_3"]["median"]
    V_BOTTOM_3_STD = _EDGE_DETECTION_STATS["vertical"]["bottom_3"]["std"]
    V_TOP_4_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["top_4"]["median"]
    V_TOP_4_STD = _EDGE_DETECTION_STATS["vertical"]["top_4"]["std"]
    V_BOTTOM_4_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["bottom_4"]["median"]
    V_BOTTOM_4_STD = _EDGE_DETECTION_STATS["vertical"]["bottom_4"]["std"]
    V_TOP_5_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["top_5"]["median"]
    V_TOP_5_STD = _EDGE_DETECTION_STATS["vertical"]["top_5"]["std"]
    V_BOTTOM_5_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["bottom_5"]["median"]
    V_BOTTOM_5_STD = _EDGE_DETECTION_STATS["vertical"]["bottom_5"]["std"]
    V_TOP_6_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["top_6"]["median"]
    V_TOP_6_STD = _EDGE_DETECTION_STATS["vertical"]["top_6"]["std"]
    V_BOTTOM_6_MEDIAN = _EDGE_DETECTION_STATS["vertical"]["bottom_6"]["median"]
    V_BOTTOM_6_STD = _EDGE_DETECTION_STATS["vertical"]["bottom_6"]["std"]
    CENTER_X_MEDIAN = _EDGE_DETECTION_STATS["center"]["x"]["median"]
    CENTER_X_STD = _EDGE_DETECTION_STATS["center"]["x"]["std"]
    CENTER_Y_MEDIAN = _EDGE_DETECTION_STATS["center"]["y"]["median"]
    CENTER_Y_STD = _EDGE_DETECTION_STATS["center"]["y"]["std"]
    N_STD = _EDGE_DETECTION_STATS["n_std"]
    
    # Spot grid constants (for 697x520 images)
    SPOT_SPACING = _GRID_CONFIG["spot_spacing"]
    EDGE_OFFSET = _GRID_CONFIG["edge_offset"]
    
    # Grid dimensions for chips with different peptide types
    PTK_GRID_ROWS = _GRID_CONFIG["ptk_grid_rows"]
    PTK_GRID_COLS = _GRID_CONFIG["ptk_grid_cols"]
    STK_GRID_ROWS = _GRID_CONFIG["stk_grid_rows"]
    STK_GRID_COLS = _GRID_CONFIG["stk_grid_cols"]
    
    # Square mask offsets from reference spots (in units of SPOT_SPACING)
    PTK_H_OFFSET_MULTIPLIER = _GRID_CONFIG["ptk_h_offset_multiplier"]
    PTK_V_OFFSET_MULTIPLIER = _GRID_CONFIG["ptk_v_offset_multiplier"]
    STK_H_OFFSET_MULTIPLIER = _GRID_CONFIG["stk_h_offset_multiplier"]
    STK_V_OFFSET_MULTIPLIER = _GRID_CONFIG["stk_v_offset_multiplier"]
    
    # Square mask side lengths (in units of SPOT_SPACING)
    PTK_SQUARE_SIDE_MULTIPLIER = _GRID_CONFIG["ptk_square_side_multiplier"]
    STK_SQUARE_SIDE_MULTIPLIER = _GRID_CONFIG["stk_square_side_multiplier"]
    
    # Reference spot detection spacing
    REF_VERTICAL_SPACING_OFFSET = _GRID_CONFIG["ref_vertical_spacing_offset"]
    
    # Filtering constants for final output
    STK_FILTER_CYCLE = _FILTERING_CONFIG["stk_filter_cycle"]
    PTK_FILTER_CYCLES = list(_FILTERING_CONFIG["ptk_filter_cycles"])
    PTK_FILTER_EXPOSURE_TIMES = list(_FILTERING_CONFIG["ptk_filter_exposure_times"])
    
    # Default enrichment data file
    DEFAULT_ENRICHMENT_FILE = _DEFAULT_ENRICHMENT_FILE

    @staticmethod
    def _resolve_results_root() -> Path:
        """Resolve results root.
        
        Args:
            None.
        
        Returns:
            Path: Resolved results root.
        """
        override = os.environ.get("PYKINAXE_RESULTS_ROOT")
        if override:
            return Path(override).expanduser().resolve()
        return Path(__file__).parent.parent.resolve() / "results"

    def __init__(
        self,
        loader: DataLoader,
        images: Optional[List[int]] = None,
        radius: float = _INIT_DEFAULTS["radius"],
        smooth_sigma: float = _INIT_DEFAULTS["smooth_sigma"],
        horizontal_y1: int = _INIT_DEFAULTS["horizontal_y1"],
        horizontal_y2: int = _INIT_DEFAULTS["horizontal_y2"],
        horizontal_y3: int = _INIT_DEFAULTS["horizontal_y3"],
        horizontal_y4: int = _INIT_DEFAULTS["horizontal_y4"],
        horizontal_y5: int = _INIT_DEFAULTS["horizontal_y5"],
        horizontal_y6: int = _INIT_DEFAULTS["horizontal_y6"],
        vertical_x1: int = _INIT_DEFAULTS["vertical_x1"],
        vertical_x2: int = _INIT_DEFAULTS["vertical_x2"],
        vertical_x3: int = _INIT_DEFAULTS["vertical_x3"],
        vertical_x4: int = _INIT_DEFAULTS["vertical_x4"],
        vertical_x5: int = _INIT_DEFAULTS["vertical_x5"],
        vertical_x6: int = _INIT_DEFAULTS["vertical_x6"],
        edge_band_half_width: int = _INIT_DEFAULTS["edge_band_half_width"],
        edge_padding: int = _INIT_DEFAULTS["edge_padding"],
        use_constrained_search: bool = _INIT_DEFAULTS["use_constrained_search"],
        gamma: float = _INIT_DEFAULTS["gamma"],
        median_kernel_size: int = _INIT_DEFAULTS["median_kernel_size"],
        canny_low_threshold: float = _INIT_DEFAULTS["canny_low_threshold"],
        canny_high_threshold: float = _INIT_DEFAULTS["canny_high_threshold"]
    ):
        """Initialize ImageProcessor with a DataLoader instance.
        
        Args:
            loader (DataLoader): Loader processed by this function.
            images (Optional[List[int]]): Images processed by this function.
            radius (float): Radius used by this function.
            smooth_sigma (float): Smooth sigma used by this function.
            horizontal_y1 (int): Horizontal y1 used by this function.
            horizontal_y2 (int): Horizontal y2 used by this function.
            horizontal_y3 (int): Horizontal y3 used by this function.
            horizontal_y4 (int): Horizontal y4 used by this function.
            horizontal_y5 (int): Horizontal y5 used by this function.
            horizontal_y6 (int): Horizontal y6 used by this function.
            vertical_x1 (int): Vertical x1 used by this function.
            vertical_x2 (int): Vertical x2 used by this function.
            vertical_x3 (int): Vertical x3 used by this function.
            vertical_x4 (int): Vertical x4 used by this function.
            vertical_x5 (int): Vertical x5 used by this function.
            vertical_x6 (int): Vertical x6 used by this function.
            edge_band_half_width (int): Edge band half width used by this function.
            edge_padding (int): Edge padding used by this function.
            use_constrained_search (bool): Boolean flag controlling whether to use constrained search.
            gamma (float): Gamma used by this function.
            median_kernel_size (int): Median kernel size used by this function.
            canny_low_threshold (float): Threshold value used to filter, classify, or flag results.
            canny_high_threshold (float): Threshold value used to filter, classify, or flag results.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        # Extract images from loader
        if images is not None:
            selected_images = loader.images[images]
        else:
            selected_images = loader.images
        
        if not isinstance(selected_images, xr.DataArray):
            raise TypeError("loader.images must be an xarray.DataArray")
        if radius <= 0:
            raise ValueError("radius must be positive")
        
        # Extract attributes from loader
        self.experiment_name = loader.experiment_name
        self.subfolder_name = loader.subfolder_name
        self.original_images = selected_images
        self.peptides_type = loader.peptides_type
        self.timestamp = loader.timestamp
        self.source_data_path = loader.data_dir
        self.results_parent_relpath = Path(
            getattr(loader, "results_parent_relpath", ".")
        )
        self.results_experiment_relpath = Path(
            getattr(loader, "results_experiment_relpath", self.experiment_name)
        )
        
        if self.peptides_type not in ['PTK', 'STK']:
            raise ValueError(f"peptides_type must be 'PTK' or 'STK', got '{self.peptides_type}'")
        
        # Calculate scale factor based on actual image dimensions
        actual_height, actual_width = selected_images.shape[1:]
        width_scale = actual_width / self.REFERENCE_WIDTH
        height_scale = actual_height / self.REFERENCE_HEIGHT
        self.scale_factor = (width_scale + height_scale) / 2.0
        
        print(f"Image dimensions: {actual_width}x{actual_height} "
              f"(reference: {self.REFERENCE_WIDTH}x{self.REFERENCE_HEIGHT})")
        print(f"Scale factor: {self.scale_factor:.4f}")
        
        # Scale all dimensional parameters
        self.radius = float(radius * self.scale_factor)
        self.smooth_sigma = smooth_sigma  # This is in sigma units, not pixels, so don't scale
        self.horizontal_y1 = int(horizontal_y1 * self.scale_factor)
        self.horizontal_y2 = int(horizontal_y2 * self.scale_factor)
        self.horizontal_y3 = int(horizontal_y3 * self.scale_factor)
        self.horizontal_y4 = int(horizontal_y4 * self.scale_factor)
        self.horizontal_y5 = int(horizontal_y5 * self.scale_factor)
        self.horizontal_y6 = int(horizontal_y6 * self.scale_factor)
        self.vertical_x1 = int(vertical_x1 * self.scale_factor)
        self.vertical_x2 = int(vertical_x2 * self.scale_factor)
        self.vertical_x3 = int(vertical_x3 * self.scale_factor)
        self.vertical_x4 = int(vertical_x4 * self.scale_factor)
        self.vertical_x5 = int(vertical_x5 * self.scale_factor)
        self.vertical_x6 = int(vertical_x6 * self.scale_factor)
        self.edge_band_half_width = max(1, int(edge_band_half_width * self.scale_factor))
        self.edge_padding = int(edge_padding * self.scale_factor)
        self.use_constrained_search = use_constrained_search
        
        # Canny edge detection parameters
        self.gamma = gamma
        self.median_kernel_size = median_kernel_size
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        
        # Extract layout_loader from loader if available
        self._layout_loader = loader.layout_loader if hasattr(loader, 'layout_loader') else None
        self._spot_layout = None
        
        # Enrichment data attributes
        self._enrichment_df = None
        self._enrichment_columns = None
        self._enrichment_by_sequence = None  # Dict: sequence -> list of enrichment records
        self._enrichment_arrays = None  # Dict: column_name -> 2D numpy array
        self._enrichment_file = self.DEFAULT_ENRICHMENT_FILE
        
        # Processing results - initialized as None
        self._masked_images = None
        self._centers = None
        self._detection_metadata = None
        self._h_left_1 = None
        self._h_right_1 = None
        self._h_center_1 = None
        self._h_left_2 = None
        self._h_right_2 = None
        self._h_center_2 = None
        self._h_left_3 = None
        self._h_right_3 = None
        self._h_center_3 = None
        self._h_left_4 = None
        self._h_right_4 = None
        self._h_center_4 = None
        self._h_left_5 = None
        self._h_right_5 = None
        self._h_center_5 = None
        self._h_left_6 = None
        self._h_right_6 = None
        self._h_center_6 = None
        self._v_top_1 = None
        self._v_bottom_1 = None
        self._v_center_1 = None
        self._v_top_2 = None
        self._v_bottom_2 = None
        self._v_center_2 = None
        self._v_top_3 = None
        self._v_bottom_3 = None
        self._v_center_3 = None
        self._v_top_4 = None
        self._v_bottom_4 = None
        self._v_center_4 = None
        self._v_top_5 = None
        self._v_bottom_5 = None
        self._v_center_5 = None
        self._v_top_6 = None
        self._v_bottom_6 = None
        self._v_center_6 = None
        self._validation_results = None
        self._background_subtracted_images = None
        self._background_images = None
        self._background_stage2_images = None
        self._background_subtraction_params = None
        self._filtered_images = None
        self._detected_spots = None
        self._spot_counts = None
        self._spot_detection_params = None
        self._reference_spots = None
        self._reference_spot_params = None
        self._spot_intensities = None
        self._spot_grid_positions = None
        self._spot_intensity_params = None
        self._refined_grid_positions = None
        self._stage1_grid_positions = None
        self._stage2_grid_stats = None
        self._refined_grid_params_list = None
        self._grid_refinement_params = None
        self._refined_spot_intensities = None
        
        self._refined_spot_max_intensities = None
        self._spot_max_intensities = None
        
        # Median intensity and saturation tracking
        self._spot_median_intensities = None
        self._spot_saturation_fractions = None
        self._refined_spot_median_intensities = None
        self._refined_spot_saturation_fractions = None
        
        # Final output table (public attribute)
        self.final_output = None
        self.filtered_final_output = None
        self.final_output_bn = None
        self.results_dir = self._get_results_dir()
        
    @property
    def masked_images(self):
        """Return masked images.
        
        Args:
            None.
        
        Returns:
            object: Stored masked images.
        """
        return self._masked_images
    
    @property
    def centers(self):
        """Return centers.
        
        Args:
            None.
        
        Returns:
            object: Stored centers.
        """
        return self._centers
    
    @property
    def background_subtracted_images(self):
        """Return background subtracted images.
        
        Args:
            None.
        
        Returns:
            object: Stored background subtracted images.
        """
        return self._background_subtracted_images
    
    @property
    def filtered_images(self):
        """Filtered images are not pre-computed but created on demand in detect_reference_spots.
        
        Args:
            None.
        
        Returns:
            None: No stored value is currently available for this property.
        """
        return None
     
    @property
    def background_images(self):
        """Return background images.
        
        Args:
            None.
        
        Returns:
            object: Stored background images.
        """
        return self._background_images
    
    @property
    def reference_spots(self):
        """Return reference spots.
        
        Args:
            None.
        
        Returns:
            object: Stored reference spots.
        """
        return self._reference_spots
    
    @property
    def spot_intensities(self):
        """Return spot intensities.
        
        Args:
            None.
        
        Returns:
            object: Stored spot intensities.
        """
        return self._spot_intensities
    
    @property
    def spot_grid_positions(self):
        """Return spot grid positions.
        
        Args:
            None.
        
        Returns:
            object: Stored spot grid positions.
        """
        return self._spot_grid_positions
    
    @property
    def square_masked_images(self):
        """Return square masked images.
        
        Args:
            None.
        
        Returns:
            object: Stored square masked images.
        """
        return getattr(self, '_square_masked_images', None)
    
    @property
    def square_mask_params(self):
        """Return square mask params.
        
        Args:
            None.
        
        Returns:
            object: Stored square mask params.
        """
        return getattr(self, '_square_mask_params', None)
    
    @property
    def refined_reference_spots(self):
        """Return refined reference spots.
        
        Args:
            None.
        
        Returns:
            object: Stored refined reference spots.
        """
        return getattr(self, '_refined_reference_spots', None)
    
    @property
    def reference_angles(self):
        """Return reference angles.
        
        Args:
            None.
        
        Returns:
            object: Stored reference angles.
        """
        return getattr(self, '_reference_angles', None)
    
    @property
    def enrichment_df(self) -> Optional[pd.DataFrame]:
        """Get the enrichment DataFrame.
        
        Args:
            None.
        
        Returns:
            Optional[pd.DataFrame]: Stored enrichment DataFrame.
        """
        return self._enrichment_df
    
    @property
    def enrichment_columns(self) -> Optional[List[str]]:
        """Get list of enrichment column names (read from CSV).
        
        Args:
            None.
        
        Returns:
            Optional[List[str]]: Stored enrichment columns.
        """
        return self._enrichment_columns
    
    @property
    def enrichment_by_sequence(self) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Get enrichment data indexed by sequence.
        
        Args:
            None.
        
        Returns:
            Optional[Dict[str, List[Dict[str, Any]]]]: Stored enrichment by sequence.
        """
        return self._enrichment_by_sequence
    
    @property
    def enrichment_arrays(self) -> Optional[Dict[str, np.ndarray]]:
        """Get enrichment data as 2D arrays (n_rows x n_cols) for each column.
        
        Args:
            None.
        
        Returns:
            Optional[Dict[str, np.ndarray]]: Stored enrichment arrays.
        """
        return self._enrichment_arrays
    
    def _scale(self, value):
        """Scale a dimensional value by the scale factor.
        
        Args:
            value: Input value processed by this helper.
        
        Returns:
            object: Scale.
        """
        if isinstance(value, (int, np.integer)):
            return int(value * self.scale_factor)
        else:
            return value * self.scale_factor
    
    def _get_search_range(self, median, std, array_length, is_left_or_top_edge=True):
        """Calculate search range with symmetric sigma extension.
        
        Both boundaries use 4-sigma extension from the median.
        
        Args:
            median: Median processed by this function.
            std: Std processed by this function.
            array_length: Array length processed by this function.
            is_left_or_top_edge: Is left or top edge processed by this function.
        
        Returns:
            tuple: Requested search range.
        """
        # Scale median and std based on scale_factor
        scaled_median = median * self.scale_factor
        scaled_std = std * self.scale_factor
        
        if is_left_or_top_edge:
            # Left or top edge: both bounds use 4-sigma
            lower = int(max(0, scaled_median - self.N_STD * scaled_std))
            upper = int(min(array_length - 1, scaled_median + self.N_STD * scaled_std))
        else:
            # Right or bottom edge: both bounds use 4-sigma
            lower = int(max(0, scaled_median - self.N_STD * scaled_std))
            upper = int(min(array_length - 1, scaled_median + self.N_STD * scaled_std))
        
        return lower, upper
    
    def _apply_gamma_correction(self, signal):
        """Apply gamma correction for contrast enhancement.
        
        Args:
            signal: Signal processed by this function.
        
        Returns:
            object: Applied result for gamma correction.
        """
        # Normalize to [0, 1]
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        if signal_max - signal_min < 1e-10:
            return signal
        
        normalized = (signal - signal_min) / (signal_max - signal_min)
        # Apply gamma correction
        corrected = np.power(normalized, self.gamma)
        return corrected
    
    @staticmethod
    def _as_uint8_image(image):
        """Convert to uint8 image.
        
        Args:
            image: Image processed by this function.
        
        Returns:
            object: Converted uint8 image.
        """
        image = np.asarray(image, dtype=np.float32)
        if image.size == 0:
            return np.zeros_like(image, dtype=np.uint8)
        image = np.clip(image, 0.0, 1.0)
        return np.round(image * 255.0).astype(np.uint8)

    def _effective_odd_kernel_size(self, requested_size, shape):
        """Return effective odd kernel size.
        
        Args:
            requested_size: Requested size processed by this function.
            shape: Shape processed by this function.
        
        Returns:
            object: Effective odd kernel size.
        """
        min_dim = int(min(shape)) if shape else int(requested_size)
        kernel_size = int(max(1, requested_size))
        kernel_size = min(kernel_size, min_dim)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        return max(1, kernel_size)

    def _extract_horizontal_band(self, image, center_y):
        """Extract horizontal band.
        
        Args:
            image: Image processed by this function.
            center_y: Center y processed by this function.
        
        Returns:
            tuple: Extracted horizontal band.
        """
        height = image.shape[0]
        start = max(0, center_y - self.edge_band_half_width)
        end = min(height, center_y + self.edge_band_half_width + 1)
        return image[start:end, :], (start, end)

    def _extract_vertical_band(self, image, center_x):
        """Extract vertical band.
        
        Args:
            image: Image processed by this function.
            center_x: Center x processed by this function.
        
        Returns:
            tuple: Extracted vertical band.
        """
        width = image.shape[1]
        start = max(0, center_x - self.edge_band_half_width)
        end = min(width, center_x + self.edge_band_half_width + 1)
        return image[:, start:end], (start, end)

    def _preprocess_band_for_canny(self, band):
        """Return preprocess band for canny.
        
        Args:
            band: Band processed by this function.
        
        Returns:
            tuple: Preprocess band for canny.
        """
        gamma_corrected = self._apply_gamma_correction(band)
        gamma_uint8 = self._as_uint8_image(gamma_corrected)

        processed_uint8 = gamma_uint8
        if self.smooth_sigma > 0:
            processed_uint8 = cv2.GaussianBlur(
                processed_uint8,
                (0, 0),
                sigmaX=self.smooth_sigma,
                sigmaY=self.smooth_sigma,
            )

        median_kernel = self._effective_odd_kernel_size(
            self.median_kernel_size,
            processed_uint8.shape,
        )
        if median_kernel > 1:
            processed_uint8 = cv2.medianBlur(processed_uint8, median_kernel)

        return gamma_corrected, processed_uint8.astype(np.float32) / 255.0, processed_uint8

    def _canny_edge_detection_2d(self, band_uint8):
        """Return canny edge detection 2d.
        
        Args:
            band_uint8: Band uint8 processed by this function.
        
        Returns:
            tuple: Canny edge detection 2d.
        """
        low_thresh = int(round(self.canny_low_threshold * 255.0))
        high_thresh = int(round(self.canny_high_threshold * 255.0))
        low_thresh = max(0, min(low_thresh, 255))
        high_thresh = max(low_thresh + 1, min(high_thresh, 255))
        edges = cv2.Canny(
            band_uint8,
            threshold1=low_thresh,
            threshold2=high_thresh,
            L2gradient=True,
        )
        return edges, low_thresh, high_thresh

    def _detect_edges_canny_band_method(
        self,
        band,
        left_median,
        left_std,
        right_median,
        right_std,
        orientation,
        band_bounds,
    ):
        """Detect edges canny band method.
        
        Args:
            band: Band processed by this function.
            left_median: Left median processed by this function.
            left_std: Left std processed by this function.
            right_median: Right median processed by this function.
            right_std: Right std processed by this function.
            orientation: Orientation processed by this function.
            band_bounds: Band bounds processed by this function.
        
        Returns:
            tuple: Detected edges canny band method.
        """
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        gamma_corrected, processed_band, processed_uint8 = self._preprocess_band_for_canny(band)
        edges, low_thresh_uint8, high_thresh_uint8 = self._canny_edge_detection_2d(
            processed_uint8
        )

        if orientation == "horizontal":
            sobel = cv2.Sobel(processed_band.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            abs_sobel = np.abs(sobel)
            projection = abs_sobel.mean(axis=0)
            edge_projection = edges.sum(axis=0).astype(np.float32) / 255.0
            candidate_positions = np.flatnonzero(edge_projection > 0)
            axis_length = band.shape[1]
        else:
            sobel = cv2.Sobel(processed_band.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            abs_sobel = np.abs(sobel)
            projection = abs_sobel.mean(axis=1)
            edge_projection = edges.sum(axis=1).astype(np.float32) / 255.0
            candidate_positions = np.flatnonzero(edge_projection > 0)
            axis_length = band.shape[0]

        if self.use_constrained_search:
            left_start, left_end = self._get_search_range(
                left_median,
                left_std,
                axis_length,
                is_left_or_top_edge=True,
            )
            right_start, right_end = self._get_search_range(
                right_median,
                right_std,
                axis_length,
                is_left_or_top_edge=False,
            )
        else:
            split_idx = axis_length // 2
            pad = self.edge_padding
            left_start, left_end = pad, split_idx
            right_start, right_end = split_idx, axis_length - pad

        left_mask = (candidate_positions >= left_start) & (candidate_positions <= left_end)
        right_mask = (candidate_positions >= right_start) & (candidate_positions <= right_end)
        left_candidates = candidate_positions[left_mask]
        right_candidates = candidate_positions[right_mask]

        if left_candidates.size > 0:
            left_edge = int(round(float(np.mean(left_candidates))))
        else:
            left_region = projection[left_start : left_end + 1]
            left_edge = left_start + int(np.argmax(left_region))

        if right_candidates.size > 0:
            right_edge = int(round(float(np.mean(right_candidates))))
        else:
            right_region = projection[right_start : right_end + 1]
            right_edge = right_start + int(np.argmax(right_region))

        center = (left_edge + right_edge) // 2

        if orientation == "horizontal":
            raw_profile = band.mean(axis=0)
            processed_profile = processed_band.mean(axis=0)
        else:
            raw_profile = band.mean(axis=1)
            processed_profile = processed_band.mean(axis=1)

        edge_rows, edge_cols = np.nonzero(edges)
        if orientation == "horizontal":
            edge_x_image = edge_cols.astype(np.int32, copy=False)
            edge_y_image = (band_bounds[0] + edge_rows).astype(np.int32, copy=False)
        else:
            edge_x_image = (band_bounds[0] + edge_cols).astype(np.int32, copy=False)
            edge_y_image = edge_rows.astype(np.int32, copy=False)

        canny_metadata = {
            'method': 'canny_2d_band',
            'orientation': orientation,
            'band_bounds': band_bounds,
            'band_shape': band.shape,
            'band_raw': band,
            'band_gamma_corrected': gamma_corrected,
            'band_processed': processed_band,
            'band_edges': edges,
            'gamma_corrected': gamma_corrected.mean(axis=0) if orientation == "horizontal" else gamma_corrected.mean(axis=1),
            'median_filtered': processed_profile,
            'edge_indices': candidate_positions,
            'edge_strength': edge_projection,
            'threshold_candidates': candidate_positions,
            'gradient': projection,
            'abs_gradient': projection,
            'split_idx': axis_length // 2,
            'left_max_gradient': float(projection[left_edge]) if left_edge < len(projection) else 0.0,
            'right_max_gradient': float(projection[right_edge]) if right_edge < len(projection) else 0.0,
            'padding': self.edge_padding,
            'constrained_search': self.use_constrained_search,
            'gamma': self.gamma,
            'median_kernel_size': self.median_kernel_size,
            'canny_low_threshold': self.canny_low_threshold,
            'canny_high_threshold': self.canny_high_threshold,
            'canny_low_threshold_uint8': low_thresh_uint8,
            'canny_high_threshold_uint8': high_thresh_uint8,
            'raw_profile': raw_profile,
            'processed_profile': processed_profile,
            'edge_projection': edge_projection,
            'edge_coords_image_x': edge_x_image,
            'edge_coords_image_y': edge_y_image,
            'left_search_range': (left_start, left_end),
            'right_search_range': (right_start, right_end),
        }

        return left_edge, right_edge, center, canny_metadata
    
    def _process_single_image(self, image, image_idx):
        """Process single image.
        
        Args:
            image: Image processed by this function.
            image_idx: Zero-based index selecting the image.
        
        Returns:
            tuple: Processed single image.
        """
        height, width = image.shape
        
        # Calculate inward shift to move cross-sections closer to center (scaled 15 pixels)
        inward_shift = int(15 * self.scale_factor)
        
        # Apply inward shift: horizontal cross-sections move vertically toward center
        # Top cross-sections (y1, y2, y3) move DOWN (+inward_shift)
        # Bottom cross-sections (y4, y5, y6) move UP (-inward_shift)
        hy_top = min(self.horizontal_y1 + inward_shift, height - 1)
        hy_bottom = min(self.horizontal_y4 - inward_shift, height - 1)
        
        # Apply inward shift: vertical cross-sections move horizontally toward center
        # Left cross-sections (x1, x2, x3) move RIGHT (+inward_shift)
        # Right cross-sections (x4, x5, x6) move LEFT (-inward_shift)
        vx_left = min(self.vertical_x1 + inward_shift, width - 1)
        vx_right = min(self.vertical_x4 - inward_shift, width - 1)

        top_band, top_band_bounds = self._extract_horizontal_band(image, hy_top)
        bottom_band, bottom_band_bounds = self._extract_horizontal_band(image, hy_bottom)
        left_band, left_band_bounds = self._extract_vertical_band(image, vx_left)
        right_band, right_band_bounds = self._extract_vertical_band(image, vx_right)

        top_left_edge, top_right_edge, top_center_x, top_grad_meta = self._detect_edges_canny_band_method(
            top_band, self.H_LEFT_1_MEDIAN, self.H_LEFT_1_STD, self.H_RIGHT_1_MEDIAN, self.H_RIGHT_1_STD,
            orientation="horizontal", band_bounds=top_band_bounds)
        bottom_left_edge, bottom_right_edge, bottom_center_x, bottom_grad_meta = self._detect_edges_canny_band_method(
            bottom_band, self.H_LEFT_4_MEDIAN, self.H_LEFT_4_STD, self.H_RIGHT_4_MEDIAN, self.H_RIGHT_4_STD,
            orientation="horizontal", band_bounds=bottom_band_bounds)
        left_top_edge, left_bottom_edge, left_center_y, left_grad_meta = self._detect_edges_canny_band_method(
            left_band, self.V_TOP_1_MEDIAN, self.V_TOP_1_STD, self.V_BOTTOM_1_MEDIAN, self.V_BOTTOM_1_STD,
            orientation="vertical", band_bounds=left_band_bounds)
        right_top_edge, right_bottom_edge, right_center_y, right_grad_meta = self._detect_edges_canny_band_method(
            right_band, self.V_TOP_4_MEDIAN, self.V_TOP_4_STD, self.V_BOTTOM_4_MEDIAN, self.V_BOTTOM_4_STD,
            orientation="vertical", band_bounds=right_band_bounds)

        center_x = (top_center_x + bottom_center_x) // 2
        center_y = (left_center_y + right_center_y) // 2

        metadata = {
            'band_centers': {
                'top': hy_top,
                'bottom': hy_bottom,
                'left': vx_left,
                'right': vx_right,
            },
            'top_left_edge': top_left_edge,
            'top_right_edge': top_right_edge,
            'top_center_x': top_center_x,
            'bottom_left_edge': bottom_left_edge,
            'bottom_right_edge': bottom_right_edge,
            'bottom_center_x': bottom_center_x,
            'left_top_edge': left_top_edge,
            'left_bottom_edge': left_bottom_edge,
            'left_center_y': left_center_y,
            'right_top_edge': right_top_edge,
            'right_bottom_edge': right_bottom_edge,
            'right_center_y': right_center_y,
            'top_grad_meta': top_grad_meta,
            'bottom_grad_meta': bottom_grad_meta,
            'left_grad_meta': left_grad_meta,
            'right_grad_meta': right_grad_meta,
            'center_x': center_x,
            'center_y': center_y,
            'radius': self.radius,
        }
        return center_x, center_y, self.radius, metadata
    
    def _create_circular_mask(self, shape, center, radius):
        """Create circular mask.
        
        Args:
            shape: Shape processed by this function.
            center: Center processed by this function.
            radius: Radius processed by this function.
        
        Returns:
            bool: Boolean result of the evaluated condition.
        """
        height, width = shape
        center_x, center_y = center
        y, x = np.ogrid[:height, :width]
        return np.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius
    
    def _log_transform_for_display(self, image, epsilon=1.0):
        """Return log transform for display.
        
        Args:
            image: Image processed by this function.
            epsilon: Epsilon processed by this function.
        
        Returns:
            object: Log transform for display.
        """
        return np.log(image + epsilon)
    
    def _symmetric_log_transform(self, image, epsilon=1.0):
        """Return symmetric log transform.
        
        Args:
            image: Image processed by this function.
            epsilon: Epsilon processed by this function.
        
        Returns:
            object: Symmetric log transform.
        """
        return np.sign(image) * np.log(epsilon + np.abs(image))
    
    def _get_peptides_type(self):
        """Get peptides type.
        
        Args:
            None.
        
        Returns:
            object: Requested peptides type.
        """
        return self.peptides_type
    
    def _get_results_dir(self) -> Path:
        """Get the results directory path: results/YYYYMMDDHHMMSS_experiment_name/subfolder_name/
        Creates the directory if it doesn't exist.
        Uses timestamp from loader for consistency.
        Saves source data path info to a text file.
        
        Args:
            None.
        
        Returns:
            Path: Requested results dir.
        """
        # Use timestamp from loader for consistent naming
        experiment_folder = (
            self.results_experiment_relpath.parent
            / f"{self.timestamp}_{self.results_experiment_relpath.name}"
        )

        results_dir = (
            self._resolve_results_root()
            / self.results_parent_relpath
            / experiment_folder
            / self.subfolder_name
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir
        
        # Save source data path info
        if self.source_data_path is not None:
            path_info_file = results_dir / f"{self.timestamp}_source_data_path.txt"
            with open(path_info_file, 'w') as f:
                f.write("Source Data Path:\n")
                f.write(f"{self.source_data_path}\n")
                f.write(f"\nExperiment Name: {self.experiment_name}\n")
                f.write(f"Subfolder Name: {self.subfolder_name}\n")
                f.write(f"Peptides Type: {self.peptides_type}\n")
                f.write(f"Analysis Timestamp: {self.timestamp}\n")
        
        return results_dir
    
    def _get_visualization_dir(self, method_name: str) -> Path:
        """Get the visualization directory for a specific method.
        
        Args:
            method_name (str): Method name used by this function.
        
        Returns:
            Path: Requested visualization dir.
        """
        
        image_folder_name = 'images'
        
        viz_dir = self.results_dir / image_folder_name / method_name
        
        if not viz_dir.exists():
            viz_dir.mkdir(parents=True, exist_ok=True)
        
        return viz_dir
    
    def _save_visualization(self, fig, viz_dir: Path, image_idx: int):
        """Save a figure to the visualization directory.
        
        Args:
            fig: Fig processed by this function.
            viz_dir (Path): Directory containing or receiving the viz.
            image_idx (int): Zero-based index selecting the image.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if viz_dir is None:
            return
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S_%f')
        filename = f"{timestamp}_img_idx_{image_idx:03d}.png"
        filepath = viz_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def apply_circular_mask(self, use_estimated_background=False, verbose=True):
        """Detect aperture edges and apply circular masks to all images.
        
        Uses horizontal and vertical edge-aligned bands at multiple positions to
        detect aperture edges via Canny analysis. Centers are computed from the
        detected edges, and a fixed-radius circular mask is applied to each image.
        
        Algorithm:
        1. Extract edge-aligned bands at predefined y and x coordinates
        2. Preprocess those bands and detect edges using the 2D Canny method
        3. Detect left/right edges in horizontal bands, top/bottom in vertical bands
        4. Calculate center coordinates from edge positions
        5. Create circular mask using fixed radius and detected center
        6. Apply mask to each image (pixels outside = 0)
        
        Args:
            use_estimated_background: Boolean flag controlling whether to use estimated background.
            verbose: Whether to emit progress output while running.
        
        Returns:
            object: Applied result for circular mask.
        """
        # Determine which images to use for edge detection
        if use_estimated_background:
            if self._background_stage2_images is None:
                raise RuntimeError("Stage 2 background images not available. Call subtract_background() first.")
            source_images = self._background_stage2_images
            detection_note = " (using stage 2 estimated background)"
        else:
            source_images = self.original_images
            detection_note = " (using original images)"
        
        n_images = len(source_images.image_idx)
        height, width = source_images.shape[1:]
        
        if verbose:
            print(f"Applying circular masks to {n_images} images{detection_note}...")
            print(f"  Fixed radius: {self.radius} pixels")
            print(f"  Edge detection method: Canny")
        
        masked_images_array = np.zeros((n_images, height, width), dtype=self.original_images.dtype)
        centers_list = []
        detection_metadata = {}
        
        # If background-subtracted images exist, also create masked versions
        apply_mask_to_bg_subtracted = self._background_subtracted_images is not None
        if apply_mask_to_bg_subtracted:
            masked_bg_sub_array = np.zeros((n_images, height, width), dtype=np.float32)
            # Convert to float32 once (2-3x faster than per-image conversion)
            if self._background_subtracted_images.dtype != np.float32:
                bg_sub_images_float = self._background_subtracted_images.astype(np.float32)
            else:
                bg_sub_images_float = self._background_subtracted_images
        
        # Helper function for parallel processing
        def process_single_mask(i):
            """Process single mask.
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Processed single mask.
            """
            detection_img = source_images.isel(image_idx=i).values
            center_x, center_y, radius, metadata = self._process_single_image(detection_img, i)
            
            # Apply mask to original images
            original_img = self.original_images.isel(image_idx=i).values
            mask = self._create_circular_mask((height, width), (center_x, center_y), radius)
            
            masked_img = original_img.copy()
            masked_img[~mask] = 0
            
            # Also apply mask to background-subtracted images if they exist
            masked_bg_sub = None
            if apply_mask_to_bg_subtracted:
                bg_sub_img = bg_sub_images_float.isel(image_idx=i).values
                masked_bg_sub = bg_sub_img.copy()
                masked_bg_sub[~mask] = 0
            
            return i, masked_img, masked_bg_sub, center_x, center_y, metadata
        
        # Parallel processing
        if verbose:
            print("  Processing images in parallel...")
        
        results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(process_single_mask)(i) for i in range(n_images)
        )
        
        # Sort results by index and unpack
        results.sort(key=lambda x: x[0])
        
        centers_list = []
        detection_metadata = {}
        top_left_list, top_right_list, top_center_x_list = [], [], []
        bottom_left_list, bottom_right_list, bottom_center_x_list = [], [], []
        left_top_list, left_bottom_list, left_center_y_list = [], [], []
        right_top_list, right_bottom_list, right_center_y_list = [], [], []
        
        for i, masked_img, masked_bg_sub, center_x, center_y, metadata in results:
            masked_images_array[i] = masked_img
            if masked_bg_sub is not None:
                masked_bg_sub_array[i] = masked_bg_sub
            
            centers_list.append((center_x, center_y))
            detection_metadata[i] = metadata
            
            top_left_list.append(metadata['top_left_edge'])
            top_right_list.append(metadata['top_right_edge'])
            top_center_x_list.append(metadata['top_center_x'])
            bottom_left_list.append(metadata['bottom_left_edge'])
            bottom_right_list.append(metadata['bottom_right_edge'])
            bottom_center_x_list.append(metadata['bottom_center_x'])
            left_top_list.append(metadata['left_top_edge'])
            left_bottom_list.append(metadata['left_bottom_edge'])
            left_center_y_list.append(metadata['left_center_y'])
            right_top_list.append(metadata['right_top_edge'])
            right_bottom_list.append(metadata['right_bottom_edge'])
            right_center_y_list.append(metadata['right_center_y'])
        
        self._centers = np.array(centers_list)
        self._detection_metadata = detection_metadata
        self._h_left_1 = np.array(top_left_list)
        self._h_right_1 = np.array(top_right_list)
        self._h_center_1 = np.array(top_center_x_list)
        self._h_left_4 = np.array(bottom_left_list)
        self._h_right_4 = np.array(bottom_right_list)
        self._h_center_4 = np.array(bottom_center_x_list)
        self._v_top_1 = np.array(left_top_list)
        self._v_bottom_1 = np.array(left_bottom_list)
        self._v_center_1 = np.array(left_center_y_list)
        self._v_top_4 = np.array(right_top_list)
        self._v_bottom_4 = np.array(right_bottom_list)
        self._v_center_4 = np.array(right_center_y_list)
        self._h_left_2 = self._h_left_3 = self._h_left_5 = self._h_left_6 = None
        self._h_right_2 = self._h_right_3 = self._h_right_5 = self._h_right_6 = None
        self._h_center_2 = self._h_center_3 = self._h_center_5 = self._h_center_6 = None
        self._v_top_2 = self._v_top_3 = self._v_top_5 = self._v_top_6 = None
        self._v_bottom_2 = self._v_bottom_3 = self._v_bottom_5 = self._v_bottom_6 = None
        self._v_center_2 = self._v_center_3 = self._v_center_5 = self._v_center_6 = None
        
        self._masked_images = xr.DataArray(
            data=masked_images_array, dims=self.original_images.dims,
            coords=self.original_images.coords,
            attrs={**self.original_images.attrs, 'processing': 'circular_mask_applied',
                   'fixed_radius': self.radius, 'constrained_search': self.use_constrained_search,
                   'detection_source': 'estimated_background_stage2' if use_estimated_background else 'original'}
        )
        
        # Update background-subtracted images with circular mask if applicable
        if apply_mask_to_bg_subtracted:
            self._background_subtracted_images = xr.DataArray(
                data=masked_bg_sub_array, 
                dims=self._background_subtracted_images.dims,
                coords=self._background_subtracted_images.coords,
                attrs={**self._background_subtracted_images.attrs, 
                       'processing': 'background_subtracted_and_circular_masked'}
            )
            if verbose:
                print("  Also applied circular mask to background-subtracted images")
        
        if verbose:
            print("Circular mask application complete!")
            print(f"  Center X - Mean: {np.mean(self._centers[:, 0]):.2f}, Std: {np.std(self._centers[:, 0]):.2f}")
            print(f"  Center Y - Mean: {np.mean(self._centers[:, 1]):.2f}, Std: {np.std(self._centers[:, 1]):.2f}")
        
        return self._masked_images

    def subtract_background(self, structuring_element_size=50, pre_smooth_sigma=3.0, method='white_tophat', 
                           use_masked_images=True, verbose=True):
        """Subtract slowly varying background using two-stage morphological operations.
        
        Applies morphological grey opening in two stages to estimate and subtract background:
        Stage 1: Grey opening with original structuring element
        Stage 2: Grey opening of stage 1 result with 4x larger structuring element
        
        This two-stage approach better removes large-scale background variations while
        preserving spot structures.
        
        Algorithm:
        1. Optional: Gaussian smoothing with sigma=pre_smooth_sigma
        2. Stage 1: Morphological grey opening with circular structuring element
        3. Stage 2: Morphological grey opening of stage 1 with 4x larger structuring element
        4. Subtract stage 2 background from source image
        5. If masked images are used, set pixels outside circular mask to 0
        
        Args:
            structuring_element_size: Structuring element size processed by this function.
            pre_smooth_sigma: Pre smooth sigma processed by this function.
            method: Method processed by this function.
            use_masked_images: Boolean flag controlling whether to use masked images.
            verbose: Whether to emit progress output while running.
        
        Returns:
            object: Subtract background.
        """
        if use_masked_images:
            if self._masked_images is None:
                raise RuntimeError("Images not masked. Call apply_circular_mask() first or set use_masked_images=False.")
            source_images = self._masked_images
            source_note = " (on masked images)"
        else:
            source_images = self.original_images
            source_note = " (on original images)"
        
        n_images = len(source_images.image_idx)
        height, width = source_images.shape[1:]
        
        # Convert entire dataset to float32 once (2-3x faster than per-image conversion)
        if source_images.dtype != np.float32:
            source_images = source_images.astype(np.float32)
        
        # Scale the structuring element size
        scaled_se_size = self._scale(structuring_element_size)
        
        if verbose:
            print(f"\nSubtracting background from {n_images} images{source_note}...")
            print(f"  Method: {method}, SE size: {scaled_se_size:.1f}px (scaled from {structuring_element_size})")
        
        # Stage 1: Structuring element (OpenCV-compatible)
        se_radius = int(scaled_se_size) // 2
        y, x = np.ogrid[-se_radius:se_radius+1, -se_radius:se_radius+1]
        structuring_element = (x**2 + y**2 <= se_radius**2).astype(np.uint8)
        
        # Stage 2: Downsampling factor and scaled structuring element
        downsample_factor = 2
        # For stage 2: 4x SE scaled down by downsample_factor
        se_radius_stage2_scaled = int((scaled_se_size * 1.5) / downsample_factor) // 2
        y2, x2 = np.ogrid[-se_radius_stage2_scaled:se_radius_stage2_scaled+1, 
                          -se_radius_stage2_scaled:se_radius_stage2_scaled+1]
        structuring_element_stage2 = (x2**2 + y2**2 <= se_radius_stage2_scaled**2).astype(np.uint8)
        
        if verbose:
            print(f"  Stage 1 SE size: {scaled_se_size:.1f}px (OpenCV + Parallelization)")
            print(f"  Stage 2 SE size: {scaled_se_size*4:.1f}px effective (OpenCV + Downsampling {downsample_factor}x + Parallelization)")
        
        background_subtracted_array = np.zeros((n_images, height, width), dtype=np.float32)
        background_array = np.zeros((n_images, height, width), dtype=np.float32)
        background_stage2_array = np.zeros((n_images, height, width), dtype=np.float32)
        
        # Prepare data for parallel processing
        def prepare_image_data(i):
            """Prepare image data for processing (no type conversion - already float32).
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Prepared image data.
            """
            source_img = source_images.isel(image_idx=i).values
            
            # Get or create mask if needed
            if use_masked_images:
                center_x, center_y = self._centers[i]
                mask = self._create_circular_mask((height, width), (center_x, center_y), self.radius)
            else:
                mask = np.ones((height, width), dtype=bool)
            
            # Apply smoothing if requested
            if pre_smooth_sigma > 0:
                smoothed_img = ndimage.gaussian_filter(source_img, sigma=pre_smooth_sigma)
            else:
                smoothed_img = source_img.copy()
            
            return i, source_img, smoothed_img, mask
        
        def process_stage1(i, smoothed_img, structuring_element, mask, use_masked):
            """Stage 1: OpenCV morphological opening.
            
            Args:
                i: I processed by this function.
                smoothed_img: Smoothed img processed by this function.
                structuring_element: Structuring element processed by this function.
                mask: Mask processed by this function.
                use_masked: Boolean flag controlling whether to use masked.
            
            Returns:
                tuple: Processed stage1.
            """
            background = cv2.morphologyEx(smoothed_img, cv2.MORPH_OPEN, structuring_element)
            if use_masked:
                background[~mask] = 0
            return i, background
        
        def process_stage2(i, background, structuring_element_stage2, mask, use_masked, ds_factor):
            """Stage 2: Downsampled OpenCV morphological opening.
            
            Args:
                i: I processed by this function.
                background: Background processed by this function.
                structuring_element_stage2: Structuring element stage2 processed by this function.
                mask: Mask processed by this function.
                use_masked: Boolean flag controlling whether to use masked.
                ds_factor: Ds factor processed by this function.
            
            Returns:
                tuple: Processed stage2.
            """
            h, w = background.shape
            
            # Downsample
            downsampled = cv2.resize(background, (w // ds_factor, h // ds_factor), 
                                    interpolation=cv2.INTER_LINEAR)
            
            # Apply morphological opening on downsampled image
            background_stage2_down = cv2.morphologyEx(downsampled, cv2.MORPH_OPEN, 
                                                     structuring_element_stage2)
            
            # Upsample back to original size
            background_stage2 = cv2.resize(background_stage2_down, (w, h), 
                                          interpolation=cv2.INTER_LINEAR)
            
            if use_masked:
                background_stage2[~mask] = 0
            return i, background_stage2
        
        # Prepare all image data
        if verbose:
            print("  Preparing image data...")
        image_data = [prepare_image_data(i) for i in range(n_images)]
        
        # Stage 1: Parallel processing with OpenCV
        if verbose:
            print("  Stage 1: Processing...")
        stage1_results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(process_stage1)(i, smoothed, structuring_element, mask, use_masked_images)
            for i, source, smoothed, mask in image_data
        )
        
        # Sort results by index and extract backgrounds
        stage1_results.sort(key=lambda x: x[0])
        backgrounds_stage1 = [bg for _, bg in stage1_results]
        
        # Stage 2: Parallel processing with OpenCV + downsampling
        if verbose:
            print("  Stage 2: Processing...")
        stage2_results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(process_stage2)(i, backgrounds_stage1[i], structuring_element_stage2, 
                                   mask, use_masked_images, downsample_factor)
            for i, source, smoothed, mask in image_data
        )
        
        # Sort and extract
        stage2_results.sort(key=lambda x: x[0])
        backgrounds_stage2 = [bg for _, bg in stage2_results]
        
        # Compute final subtraction and store results
        if verbose:
            print("  Computing final subtraction...")
        for i in range(n_images):
            source_img = image_data[i][1]  # Original source image
            mask = image_data[i][3]
            
            subtracted = source_img - backgrounds_stage2[i]
            
            if use_masked_images:
                subtracted[~mask] = 0
            
            background_subtracted_array[i] = subtracted
            background_array[i] = backgrounds_stage1[i]
            background_stage2_array[i] = backgrounds_stage2[i]
        
        self._background_subtraction_params = {
            'method': method, 'structuring_element_size': structuring_element_size,
            'pre_smooth_sigma': pre_smooth_sigma, 'use_masked_images': use_masked_images
        }
        
        self._background_subtracted_images = xr.DataArray(
            data=background_subtracted_array, dims=source_images.dims,
            coords=source_images.coords,
            attrs={**source_images.attrs, 'processing': 'background_subtracted',
                   'subtraction_source': 'masked' if use_masked_images else 'original'}
        )
        
        self._background_images = xr.DataArray(
            data=background_array, dims=source_images.dims,
            coords=source_images.coords,
            attrs={**source_images.attrs, 'processing': 'estimated_background_stage1',
                   'subtraction_source': 'masked' if use_masked_images else 'original'}
        )
        
        self._background_stage2_images = xr.DataArray(
            data=background_stage2_array, dims=source_images.dims,
            coords=source_images.coords,
            attrs={**source_images.attrs, 'processing': 'estimated_background_stage2',
                   'subtraction_source': 'masked' if use_masked_images else 'original'}
        )
        
        if verbose:
            mask_values = background_subtracted_array[background_subtracted_array != 0]
            print("Background subtraction complete!")
            print(f"  Signal mean: {np.mean(mask_values):.2f}, std: {np.std(mask_values):.2f}")
        
        return self._background_subtracted_images

    def _find_spot_maxima(self, image, center, mask_radius, edge_padding, pre_smooth_sigma, min_peak_distance, intensity_threshold_percentile):
        """Find spot maxima.
        
        Args:
            image: Image processed by this function.
            center: Center processed by this function.
            mask_radius: Mask radius processed by this function.
            edge_padding: Edge padding processed by this function.
            pre_smooth_sigma: Pre smooth sigma processed by this function.
            min_peak_distance: Min peak distance processed by this function.
            intensity_threshold_percentile: Threshold value used to filter, classify, or flag results.
        
        Returns:
            object: Found spot maxima.
        """
        height, width = image.shape
        center_x, center_y = center
        y_grid, x_grid = np.ogrid[:height, :width]
        distance_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        search_radius = mask_radius - edge_padding
        search_mask = distance_from_center <= search_radius
        
        if pre_smooth_sigma > 0:
            smoothed = ndimage.gaussian_filter(image, sigma=pre_smooth_sigma)
        else:
            smoothed = image.copy()
        
        smoothed_masked = np.full_like(smoothed, -np.inf)
        smoothed_masked[search_mask] = smoothed[search_mask]
        
        neighborhood_size = min_peak_distance if min_peak_distance % 2 == 1 else min_peak_distance + 1
        local_max = maximum_filter(smoothed_masked, size=neighborhood_size)
        peaks_mask = (smoothed_masked == local_max) & search_mask & (smoothed_masked > 0)
        
        positive_values = smoothed_masked[search_mask & (smoothed_masked > 0)]
        if len(positive_values) > 0:
            threshold = np.percentile(positive_values, intensity_threshold_percentile)
            peaks_mask = peaks_mask & (smoothed_masked > threshold)
        
        peak_coords_yx = np.array(np.where(peaks_mask)).T
        if len(peak_coords_yx) == 0:
            return np.array([]).reshape(0, 2)
        
        peak_coords_xy = peak_coords_yx[:, ::-1].astype(np.float32)
        valid_peaks = [[x, y] for x, y in peak_coords_xy if np.sqrt((x - center_x)**2 + (y - center_y)**2) <= search_radius]
        
        if len(valid_peaks) == 0:
            return np.array([]).reshape(0, 2)
        
        peak_coords_xy = np.array(valid_peaks, dtype=np.float32)
        intensities = np.array([image[int(y), int(x)] for x, y in peak_coords_xy])
        sort_idx = np.argsort(intensities)[::-1]
        return peak_coords_xy[sort_idx]

    def _find_spot_at_offset(self, spots, ref_x, ref_y, dx, dy, tolerance):
        """Find spot at offset.
        
        Args:
            spots: Spots processed by this function.
            ref_x: Ref x processed by this function.
            ref_y: Ref y processed by this function.
            dx: Dx processed by this function.
            dy: Dy processed by this function.
            tolerance: Tolerance processed by this function.
        
        Returns:
            object: Found spot at offset.
        """
        expected_x, expected_y = ref_x + dx, ref_y + dy
        distances = np.sqrt((spots[:, 0] - expected_x)**2 + (spots[:, 1] - expected_y)**2)
        min_idx = np.argmin(distances)
        return spots[min_idx].copy() if distances[min_idx] <= tolerance else None

    def _find_t_shape(self, spots, x_min, x_max, horizontal_spacing, vertical_spacing, tolerance, y_min=None, y_max=None):
        """Find t shape.
        
        Args:
            spots: Spots processed by this function.
            x_min: X min processed by this function.
            x_max: X max processed by this function.
            horizontal_spacing: Horizontal spacing processed by this function.
            vertical_spacing: Vertical spacing processed by this function.
            tolerance: Tolerance processed by this function.
            y_min: Y min processed by this function.
            y_max: Y max processed by this function.
        
        Returns:
            object: Found t shape.
        """
        if len(spots) < 4:
            return None
        
        # Filter by x bounds
        candidate_spots = spots[(spots[:, 0] >= x_min) & (spots[:, 0] <= x_max)]
        
        # Filter by y bounds if provided
        if y_min is not None:
            candidate_spots = candidate_spots[candidate_spots[:, 1] >= y_min]
        if y_max is not None:
            candidate_spots = candidate_spots[candidate_spots[:, 1] <= y_max]
        
        if len(candidate_spots) < 4:
            return None
        
        best_match, best_score = None, float('inf')
        for center_spot in candidate_spots:
            cx, cy = center_spot
            top = self._find_spot_at_offset(candidate_spots, cx, cy, 0, -vertical_spacing, tolerance)
            bottom = self._find_spot_at_offset(candidate_spots, cx, cy, 0, vertical_spacing, tolerance)
            right = self._find_spot_at_offset(candidate_spots, cx, cy, horizontal_spacing, 0, tolerance)
            
            if top is None or bottom is None or right is None:
                continue
            
            score = (abs(top[0] - cx) + abs(top[1] - (cy - vertical_spacing)) +
                    abs(bottom[0] - cx) + abs(bottom[1] - (cy + vertical_spacing)) +
                    abs(right[0] - (cx + horizontal_spacing)) + abs(right[1] - cy))
            
            if score < best_score:
                best_score = score
                best_match = np.array([top, center_spot, bottom, right])
        return best_match

    def _find_j_shape(self, spots, x_min, x_max, horizontal_spacing, vertical_spacing, tolerance, y_min=None, y_max=None):
        """Find j shape.
        
        Args:
            spots: Spots processed by this function.
            x_min: X min processed by this function.
            x_max: X max processed by this function.
            horizontal_spacing: Horizontal spacing processed by this function.
            vertical_spacing: Vertical spacing processed by this function.
            tolerance: Tolerance processed by this function.
            y_min: Y min processed by this function.
            y_max: Y max processed by this function.
        
        Returns:
            object: Found j shape.
        """
        if len(spots) < 4:
            return None
        
        # Filter by x bounds
        candidate_spots = spots[(spots[:, 0] >= x_min) & (spots[:, 0] <= x_max)]
        
        # Filter by y bounds if provided
        if y_min is not None:
            candidate_spots = candidate_spots[candidate_spots[:, 1] >= y_min]
        if y_max is not None:
            candidate_spots = candidate_spots[candidate_spots[:, 1] <= y_max]
        
        if len(candidate_spots) < 4:
            return None
        
        best_match, best_score = None, float('inf')
        for bottom_right in candidate_spots:
            brx, bry = bottom_right
            bottom_left = self._find_spot_at_offset(candidate_spots, brx, bry, -horizontal_spacing, 0, tolerance)
            middle = self._find_spot_at_offset(candidate_spots, brx, bry, 0, -vertical_spacing, tolerance)
            top = self._find_spot_at_offset(candidate_spots, brx, bry, 0, -2*vertical_spacing, tolerance)
            
            if bottom_left is None or middle is None or top is None:
                continue
            
            score = (abs(bottom_left[0] - (brx - horizontal_spacing)) + abs(bottom_left[1] - bry) +
                    abs(middle[0] - brx) + abs(middle[1] - (bry - vertical_spacing)) +
                    abs(top[0] - brx) + abs(top[1] - (bry - 2*vertical_spacing)))
            
            if score < best_score:
                best_score = score
                best_match = np.array([top, middle, bottom_left, bottom_right])
        return best_match

    def _find_t_shape_simple(self, spots, candidate_center, h_spacing, v_spacing, tolerance):
        """Find a T-shape pattern given a candidate center spot (simplified version for simple strategy).
        
        T-shape pattern (from top to bottom):
        - top: center_x, center_y - v_spacing
        - center: provided candidate spot
        - bottom: center_x, center_y + v_spacing
        - right: center_x + h_spacing, center_y
        
        Args:
            spots: Spots processed by this function.
            candidate_center: Candidate center processed by this function.
            h_spacing: H spacing processed by this function.
            v_spacing: V spacing processed by this function.
            tolerance: Tolerance processed by this function.
        
        Returns:
            object: Found t shape simple.
        """
        center_x, center_y = candidate_center
        
        # Find spots at expected offsets from candidate center
        top = self._find_spot_at_offset(spots, center_x, center_y, 0, -v_spacing, tolerance)
        bottom = self._find_spot_at_offset(spots, center_x, center_y, 0, v_spacing, tolerance)
        right = self._find_spot_at_offset(spots, center_x, center_y, h_spacing, 0, tolerance)
        
        if top is not None and bottom is not None and right is not None:
            return np.array([top, candidate_center, bottom, right])
        return None

    def _find_j_shape_simple(self, spots, candidate_bottom_right, h_spacing, v_spacing, tolerance):
        """Find a J-shape pattern given a candidate bottom-right spot (simplified version for simple strategy).
        
        J-shape pattern (from top to bottom-right):
        - top: bottom_right_x, bottom_right_y - 2*v_spacing
        - middle: bottom_right_x, bottom_right_y - v_spacing
        - bottom_left: bottom_right_x - h_spacing, bottom_right_y
        - bottom_right: provided candidate spot
        
        Args:
            spots: Spots processed by this function.
            candidate_bottom_right: Candidate bottom right processed by this function.
            h_spacing: H spacing processed by this function.
            v_spacing: V spacing processed by this function.
            tolerance: Tolerance processed by this function.
        
        Returns:
            object: Found j shape simple.
        """
        br_x, br_y = candidate_bottom_right
        
        # Find spots at expected offsets from candidate bottom-right
        top = self._find_spot_at_offset(spots, br_x, br_y, 0, -2 * v_spacing, tolerance)
        middle = self._find_spot_at_offset(spots, br_x, br_y, 0, -v_spacing, tolerance)
        bottom_left = self._find_spot_at_offset(spots, br_x, br_y, -h_spacing, 0, tolerance)
        
        if top is not None and middle is not None and bottom_left is not None:
            return np.array([top, middle, bottom_left, candidate_bottom_right])
        return None

    def _validate_t_j_geometry(self, t_shape, j_shape, spot_spacing, tolerance):
        """Validate the geometric relationship between T-shape and J-shape reference spots.
        
        According to the PamGene array layout:
        - J-shape middle should be 19 spot spacings to the right of T-shape middle
        - J-shape middle should be 1 spot spacing below T-shape middle
        
        Args:
            t_shape: T shape processed by this function.
            j_shape: J shape processed by this function.
            spot_spacing: Spot spacing processed by this function.
            tolerance: Tolerance processed by this function.
        
        Returns:
            object: Validation result for t j geometry.
        """
        # Extract middle spots (index 1 for both T and J)
        t_middle = t_shape[1]  # T-shape center spot
        j_middle = j_shape[1]  # J-shape middle spot
        
        # Expected relative position: J-middle should be at (T_x + 19*spacing, T_y + 1*spacing)
        expected_horizontal_offset = 19 * spot_spacing
        expected_vertical_offset = 1 * spot_spacing
        
        # Actual offsets
        actual_horizontal_offset = j_middle[0] - t_middle[0]
        actual_vertical_offset = j_middle[1] - t_middle[1]
        
        # Check if offsets match expectations within tolerance
        horizontal_valid = abs(actual_horizontal_offset - expected_horizontal_offset) < tolerance
        vertical_valid = abs(actual_vertical_offset - expected_vertical_offset) < tolerance
        
        return horizontal_valid and vertical_valid

    def _detect_reference_shapes_with_fallback(self, spots, left_boundary, right_boundary,
                                                edge_search_width, horizontal_spacing, 
                                                vertical_spacing, spacing_tolerance, scaled_spot_spacing):
        """Detect T-shape and J-shape reference spots (with fallback logic).
        
        This method attempts to find both T-shape (left edge) and J-shape (right edge)
        reference spot patterns in the detected spot coordinates. If only one shape is
        found, it uses its position to define a targeted search region for the other shape.
        
        Args:
            spots: Spots processed by this function.
            left_boundary: Left boundary processed by this function.
            right_boundary: Right boundary processed by this function.
            edge_search_width: Edge search width processed by this function.
            horizontal_spacing: Horizontal spacing processed by this function.
            vertical_spacing: Vertical spacing processed by this function.
            spacing_tolerance: Spacing tolerance processed by this function.
            scaled_spot_spacing: Scaled spot spacing processed by this function.
        
        Returns:
            tuple: Detected reference shapes with fallback.
        """
        used_fallback = False
        
        # Initial detection attempt
        t_shape = self._find_t_shape(spots, left_boundary, left_boundary + edge_search_width,
                                    horizontal_spacing, vertical_spacing, spacing_tolerance)
        j_shape = self._find_j_shape(spots, right_boundary - edge_search_width, right_boundary,
                                    horizontal_spacing, vertical_spacing, spacing_tolerance)
        
        # Fallback logic if one shape is detected but not the other
        if t_shape is not None and j_shape is None:
            # T-shape detected, J-shape not detected
            # Search for J-shape based on T-shape position
            t_rightmost_x = np.max(t_shape[:, 0])
            t_top_y = np.min(t_shape[:, 1])
            t_bottom_y = np.max(t_shape[:, 1])
            
            # J-shape search region
            j_x_min = t_rightmost_x + 16 * scaled_spot_spacing
            j_x_max = t_rightmost_x + 19 * scaled_spot_spacing
            j_y_min = t_top_y - 2 * scaled_spot_spacing
            j_y_max = t_bottom_y + 2 * scaled_spot_spacing
            
            j_shape = self._find_j_shape(spots, j_x_min, j_x_max,
                                        horizontal_spacing, vertical_spacing, spacing_tolerance,
                                        y_min=j_y_min, y_max=j_y_max)
            if j_shape is not None:
                used_fallback = True
        
        elif j_shape is not None and t_shape is None:
            # J-shape detected, T-shape not detected
            # Search for T-shape based on J-shape position
            j_leftmost_x = np.min(j_shape[:, 0])
            j_top_y = np.min(j_shape[:, 1])
            j_bottom_y = np.max(j_shape[:, 1])
            
            # T-shape search region
            t_x_min = j_leftmost_x - 19 * scaled_spot_spacing
            t_x_max = j_leftmost_x - 16 * scaled_spot_spacing
            t_y_min = j_top_y - 2 * scaled_spot_spacing
            t_y_max = j_bottom_y + 2 * scaled_spot_spacing
            
            t_shape = self._find_t_shape(spots, t_x_min, t_x_max,
                                        horizontal_spacing, vertical_spacing, spacing_tolerance,
                                        y_min=t_y_min, y_max=t_y_max)
            if t_shape is not None:
                used_fallback = True
        
        return t_shape, j_shape, used_fallback

    def detect_reference_spots(self, pre_smooth_sigma=5.0, min_peak_distance=10, intensity_threshold_percentile=80,
                           spot_edge_padding=5, edge_search_width=115.0, horizontal_spacing=SPOT_SPACING,
                           vertical_spacing=None, spacing_tolerance=10.0, median_filter_size=15, 
                           use_simple_strategy=True, verbose=True):
        """Detect T-shape and J-shape reference spot patterns in all images.
        
        Searches for characteristic reference spot patterns:
        - T-shape: 4 spots arranged as T (3 vertical + 1 horizontal right)
        - J-shape: 4 spots arranged as J (3 vertical + 1 horizontal left)
        
        Uses progressive fallback strategy:
        1. Standard detection on background-subtracted image
        2. Relaxed geometric constraints if initial detection fails
        3. Median-filtered image if still failing
        4. Reduced edge padding to search closer to mask boundary
        5. Expanded search width and increased tolerance as last resort
        6. Hybrid optimization (final pass): Multi-start L-BFGS-B with validation
           - Stage 1: Try L-BFGS-B from 9 initial positions (3×3 grid around median)
           - Stage 2: Validate result (intensity > 10th percentile threshold + geometry)
           - Stage 3: If validation fails, use differential_evolution as fallback
           - Iterates up to 10 times until valid solution found
        
        Algorithm:
        1. Find local maxima (potential spots) using maximum filter
        2. Search for T-shape pattern in left edge region
        3. Search for J-shape pattern in right edge region
        4. Apply geometric validation and fallback strategies
        5. Perform Powell optimization refinement on missing/outlier shapes
        
        Args:
            pre_smooth_sigma: Pre smooth sigma processed by this function.
            min_peak_distance: Min peak distance processed by this function.
            intensity_threshold_percentile: Threshold value used to filter, classify, or flag results.
            spot_edge_padding: Spot edge padding processed by this function.
            edge_search_width: Edge search width processed by this function.
            horizontal_spacing: Horizontal spacing processed by this function.
            vertical_spacing: Vertical spacing processed by this function.
            spacing_tolerance: Spacing tolerance processed by this function.
            median_filter_size: Median filter size processed by this function.
            use_simple_strategy: Boolean flag controlling whether to use simple strategy.
            verbose: Whether to emit progress output while running.
        
        Returns:
            object: Detected reference spots.
        """
        if self._background_subtracted_images is None:
            raise RuntimeError("Background not subtracted. Call subtract_background() first.")
        
        # Set default vertical spacing if not provided
        if vertical_spacing is None:
            vertical_spacing = 2 * self.SPOT_SPACING + self.REF_VERTICAL_SPACING_OFFSET
        
        n_images = len(self._background_subtracted_images.image_idx)
        
        # Scale dimensional parameters
        scaled_min_peak_distance = self._scale(min_peak_distance)
        scaled_spot_edge_padding = self._scale(spot_edge_padding)
        scaled_edge_search_width = self._scale(edge_search_width)
        scaled_horizontal_spacing = self._scale(horizontal_spacing)
        scaled_vertical_spacing = self._scale(vertical_spacing)
        scaled_spacing_tolerance = self._scale(spacing_tolerance)
        scaled_median_filter_size = self._scale(median_filter_size)
        scaled_spot_spacing = self._scale(self.SPOT_SPACING)
        
        if verbose:
            print(f"\nDetecting reference spots in {n_images} images...")
            print(f"  Scale factor: {self.scale_factor:.4f}")
            print(f"  Scaled parameters: peak_dist={scaled_min_peak_distance:.1f}, "
                  f"search_width={scaled_edge_search_width:.1f}, spacing={scaled_horizontal_spacing:.1f}")
        
        self._spot_detection_params = {
            'pre_smooth_sigma': pre_smooth_sigma, 'min_peak_distance': min_peak_distance,
            'intensity_threshold_percentile': intensity_threshold_percentile, 'edge_padding': spot_edge_padding,
            'median_filter_size': median_filter_size, 'scale_factor': self.scale_factor,
            'scaled_min_peak_distance': scaled_min_peak_distance, 'scaled_spot_edge_padding': scaled_spot_edge_padding,
            'scaled_median_filter_size': scaled_median_filter_size
        }
        self._reference_spot_params = {
            'edge_search_width': edge_search_width, 'horizontal_spacing': horizontal_spacing,
            'vertical_spacing': vertical_spacing, 'spacing_tolerance': spacing_tolerance,
            'scaled_edge_search_width': scaled_edge_search_width, 'scaled_horizontal_spacing': scaled_horizontal_spacing,
            'scaled_vertical_spacing': scaled_vertical_spacing, 'scaled_spacing_tolerance': scaled_spacing_tolerance
        }
        
        # Convert original images to float32 for simple strategy spot detection
        if self.original_images.dtype != np.float32:
            original_images_float = self.original_images.astype(np.float32)
        else:
            original_images_float = self.original_images
        
        # Convert background-subtracted images to float32 for progressive strategy
        if self._background_subtracted_images.dtype != np.float32:
            bg_sub_images_float = self._background_subtracted_images.astype(np.float32)
        else:
            bg_sub_images_float = self._background_subtracted_images
        
        # ============================================================================
        # SIMPLE STRATEGY: Global search with geometric constraint validation
        # ============================================================================
        if use_simple_strategy:
            if verbose:
                print("  Using simple strategy: global search with geometric validation")
            
            # Helper function for simple strategy
            def process_reference_spots_simple(i):
                """Process reference spots simple.
                
                Args:
                    i: I processed by this function.
                
                Returns:
                    tuple: Processed reference spots simple.
                """
                img = original_images_float.isel(image_idx=i).values
                center_x, center_y = self._centers[i]
                
                # Find all spots in the entire circular mask (no edge restriction)
                spots = self._find_spot_maxima(img, (center_x, center_y), self.radius, scaled_spot_edge_padding,
                                              pre_smooth_sigma, scaled_min_peak_distance, intensity_threshold_percentile)
                
                if len(spots) < 8:  # Need at least 4 spots for T + 4 for J
                    return i, spots, None, None
                
                # Split spots into left and right halves
                left_spots = spots[spots[:, 0] < center_x]
                right_spots = spots[spots[:, 0] >= center_x]
                
                # Find all T-shape candidates in left half
                t_candidates = []
                for candidate_spot in left_spots:
                    t_shape = self._find_t_shape_simple(left_spots, candidate_spot, 
                                                        scaled_horizontal_spacing, scaled_vertical_spacing, 
                                                        scaled_spacing_tolerance)
                    if t_shape is not None:
                        t_candidates.append(t_shape)
                
                # Find all J-shape candidates in right half
                j_candidates = []
                for candidate_spot in right_spots:
                    j_shape = self._find_j_shape_simple(right_spots, candidate_spot,
                                                        scaled_horizontal_spacing, scaled_vertical_spacing,
                                                        scaled_spacing_tolerance)
                    if j_shape is not None:
                        j_candidates.append(j_shape)
                
                # If no candidates found, return None
                if len(t_candidates) == 0 or len(j_candidates) == 0:
                    return i, spots, None, None
                
                # Validate T-J pairs based on geometric relationship and pick best
                best_t, best_j = None, None
                best_score = -np.inf
                
                for t_shape in t_candidates:
                    for j_shape in j_candidates:
                        # Validate geometric relationship
                        if self._validate_t_j_geometry(t_shape, j_shape, scaled_spot_spacing, scaled_spacing_tolerance):
                            # Compute combined intensity score
                            t_intensity = sum(img[int(round(y)), int(round(x))] for x, y in t_shape)
                            j_intensity = sum(img[int(round(y)), int(round(x))] for x, y in j_shape)
                            combined_score = t_intensity + j_intensity
                            
                            if combined_score > best_score:
                                best_score = combined_score
                                best_t = t_shape
                                best_j = j_shape
                
                return i, spots, best_t, best_j
            
            # Parallel processing
            if verbose:
                print("  Processing images in parallel...")
            
            results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
                delayed(process_reference_spots_simple)(i) for i in range(n_images)
            )
            
            # Sort and unpack results
            results.sort(key=lambda x: x[0])
            
            all_spots, spot_counts, t_shapes, j_shapes = [], [], [], []
            t_found_count, j_found_count = 0, 0
            
            for i, spots, t_shape, j_shape in results:
                all_spots.append(spots)
                spot_counts.append(len(spots))
                t_shapes.append(t_shape)
                j_shapes.append(j_shape)
                
                if t_shape is not None:
                    t_found_count += 1
                if j_shape is not None:
                    j_found_count += 1
            
            self._detected_spots = all_spots
            self._spot_counts = np.array(spot_counts)
            self._reference_spots = {'t_shape': t_shapes, 'j_shape': j_shapes}
            
            if verbose:
                print("Reference spot detection complete!")
                print(f"  T-shapes: {t_found_count}/{n_images}")
                print(f"  J-shapes: {j_found_count}/{n_images}")
            
            return self._reference_spots
        
        # ============================================================================
        # PROGRESSIVE FALLBACK STRATEGY (original implementation)
        # ============================================================================
        if verbose:
            print("  Using progressive fallback strategy with optimization refinement")
        
        # Helper function for parallel processing
        def process_reference_spots(i):
            """Process reference spots.
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Processed reference spots.
            """
            img = bg_sub_images_float.isel(image_idx=i).values
            center_x, center_y = self._centers[i]
            
            spots = self._find_spot_maxima(img, (center_x, center_y), self.radius, scaled_spot_edge_padding,
                                          pre_smooth_sigma, scaled_min_peak_distance, intensity_threshold_percentile)
            
            left_boundary, right_boundary = center_x - self.radius, center_x + self.radius
            
            # Initial detection attempt on background-subtracted image
            t_shape, j_shape, used_fallback = self._detect_reference_shapes_with_fallback(
                spots, left_boundary, right_boundary, scaled_edge_search_width,
                scaled_horizontal_spacing, scaled_vertical_spacing, scaled_spacing_tolerance, scaled_spot_spacing
            )
            
            fallback_flags = {'t_fallback': used_fallback and t_shape is not None,
                            'j_fallback': used_fallback and j_shape is not None,
                            'filtered': False, 'relaxed': False, 'expanded': False}
            
            # Track if filtered image was created for this specific image
            img_filtered = None
            
            # If either reference spot is still not detected, retry with filtered image
            if t_shape is None or j_shape is None:
                fallback_flags['filtered'] = True
                # Apply median filter on-demand
                height, width = img.shape
                mask = self._create_circular_mask((height, width), (center_x, center_y), self.radius)
                img_filtered = median_filter(img, size=int(scaled_median_filter_size))
                img_filtered[~mask] = 0
                
                spots_filtered = self._find_spot_maxima(img_filtered, (center_x, center_y), self.radius, scaled_spot_edge_padding,
                                              pre_smooth_sigma, scaled_min_peak_distance, intensity_threshold_percentile)
                
                # Detection attempt on filtered image
                t_shape, j_shape, _ = self._detect_reference_shapes_with_fallback(
                    spots_filtered, left_boundary, right_boundary, scaled_edge_search_width,
                    scaled_horizontal_spacing, scaled_vertical_spacing, scaled_spacing_tolerance, scaled_spot_spacing
                )
            
            # Progressive relaxation for remaining failures
            if t_shape is None or j_shape is None:
                # Choose the better image source (filtered if we already tried it, otherwise background-subtracted)
                img_source = img_filtered if img_filtered is not None else img
                
                # Case 1: One shape detected - targeted expansion on the failed side only
                if t_shape is not None and j_shape is None:
                    # T detected, J failed - expand search only on J side (right)
                    fallback_flags['expanded'] = True
                    expanded_width = scaled_edge_search_width * 2
                    spots_relaxed = self._find_spot_maxima(img_source, (center_x, center_y), self.radius, 
                                                           scaled_spot_edge_padding, pre_smooth_sigma, 
                                                           scaled_min_peak_distance, intensity_threshold_percentile)
                    j_shape = self._find_j_shape(spots_relaxed, right_boundary - expanded_width, right_boundary,
                                                 scaled_horizontal_spacing, scaled_vertical_spacing, scaled_spacing_tolerance)
                
                elif j_shape is not None and t_shape is None:
                    # J detected, T failed - expand search only on T side (left)
                    fallback_flags['expanded'] = True
                    expanded_width = scaled_edge_search_width * 2
                    spots_relaxed = self._find_spot_maxima(img_source, (center_x, center_y), self.radius, 
                                                           scaled_spot_edge_padding, pre_smooth_sigma, 
                                                           scaled_min_peak_distance, intensity_threshold_percentile)
                    t_shape = self._find_t_shape(spots_relaxed, left_boundary, left_boundary + expanded_width,
                                                 scaled_horizontal_spacing, scaled_vertical_spacing, scaled_spacing_tolerance)
                
                # Case 2: Both shapes failed - progressive relaxation
                elif t_shape is None and j_shape is None:
                    # Strategy 1: Reduce edge padding (safer - searches closer to mask boundary)
                    fallback_flags['relaxed'] = True
                    reduced_padding = max(0, scaled_spot_edge_padding // 2)
                    spots_reduced = self._find_spot_maxima(img_source, (center_x, center_y), self.radius, 
                                                           reduced_padding, pre_smooth_sigma, 
                                                           scaled_min_peak_distance, intensity_threshold_percentile)
                    
                    t_shape, j_shape, _ = self._detect_reference_shapes_with_fallback(
                        spots_reduced, left_boundary, right_boundary, scaled_edge_search_width,
                        scaled_horizontal_spacing, scaled_vertical_spacing, scaled_spacing_tolerance, scaled_spot_spacing
                    )
                    
                    # Strategy 2: If still failed, expand search width and increase tolerance (riskier)
                    if t_shape is None or j_shape is None:
                        fallback_flags['expanded'] = True
                        expanded_width = scaled_edge_search_width * 2
                        increased_tolerance = scaled_spacing_tolerance * 1.5
                        
                        # Try with no edge padding and expanded search
                        spots_final = self._find_spot_maxima(img_source, (center_x, center_y), self.radius, 
                                                             0, pre_smooth_sigma, 
                                                             scaled_min_peak_distance, intensity_threshold_percentile)
                        
                        if t_shape is None:
                            t_shape = self._find_t_shape(spots_final, left_boundary, left_boundary + expanded_width,
                                                         scaled_horizontal_spacing, scaled_vertical_spacing, increased_tolerance)
                        if j_shape is None:
                            j_shape = self._find_j_shape(spots_final, right_boundary - expanded_width, right_boundary,
                                                         scaled_horizontal_spacing, scaled_vertical_spacing, increased_tolerance)
            
            return i, spots, t_shape, j_shape, fallback_flags
        
        # Parallel processing
        if verbose:
            print("  Processing images in parallel...")
        
        results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(process_reference_spots)(i) for i in range(n_images)
        )
        
        # Sort and unpack results
        results.sort(key=lambda x: x[0])
        
        all_spots, spot_counts, t_shapes, j_shapes = [], [], [], []
        t_found_count, j_found_count = 0, 0
        t_fallback_count, j_fallback_count = 0, 0
        filtered_fallback_count = 0
        relaxed_padding_count = 0
        expanded_search_count = 0
        
        for i, spots, t_shape, j_shape, flags in results:
            all_spots.append(spots)
            spot_counts.append(len(spots))
            t_shapes.append(t_shape)
            j_shapes.append(j_shape)
            
            if t_shape is not None: t_found_count += 1
            if j_shape is not None: j_found_count += 1
            if flags['t_fallback']: t_fallback_count += 1
            if flags['j_fallback']: j_fallback_count += 1
            if flags['filtered']: filtered_fallback_count += 1
            if flags['relaxed']: relaxed_padding_count += 1
            if flags['expanded']: expanded_search_count += 1
        
        self._detected_spots = all_spots
        self._spot_counts = np.array(spot_counts)
        self._reference_spots = {'t_shape': t_shapes, 'j_shape': j_shapes}
        
        if verbose:
            print("Reference spot detection complete!")
            print(f"  T-shapes: {t_found_count}/{n_images} (fallback: {t_fallback_count})")
            print(f"  J-shapes: {j_found_count}/{n_images} (fallback: {j_fallback_count})")
            print(f"  Filtered image fallback: {filtered_fallback_count}/{n_images}")
            if relaxed_padding_count > 0 or expanded_search_count > 0:
                print("  Progressive relaxation applied:")
                if relaxed_padding_count > 0:
                    print(f"    - Reduced edge padding: {relaxed_padding_count}/{n_images}")
                if expanded_search_count > 0:
                    print(f"    - Expanded search width: {expanded_search_count}/{n_images}")
        
        # ============================================================================
        # HYBRID OPTIMIZATION REFINEMENT: Final pass to fix missing/outlier reference spots
        # ============================================================================
        # Strategy: Multi-start L-BFGS-B + validation + differential evolution fallback
        # 1. Try L-BFGS-B from 9 starting positions (3×3 grid around median)
        # 2. Validate result against intensity threshold (10th percentile) + geometry
        # 3. If validation fails, use differential_evolution (slower but more robust)
        # 4. Iterate up to 10 times with different seeds until valid solution found
        
        if verbose:
            print("\nApplying hybrid optimization refinement...")
        
        # Extract middle spot (index 1) coordinates from all detected shapes
        t_middle_spots = []
        j_middle_spots = []
        
        for t_shape in t_shapes:
            if t_shape is not None:
                # T-shape: [top, center, bottom, right] → middle is index 1
                t_middle_spots.append(t_shape[1])
        
        for j_shape in j_shapes:
            if j_shape is not None:
                # J-shape: [top, middle, bottom_left, bottom_right] → middle is index 1
                j_middle_spots.append(j_shape[1])
        
        # Compute median of middle spot positions
        if len(t_middle_spots) > 0:
            t_middle_array = np.array(t_middle_spots)
            t_median_middle = np.median(t_middle_array, axis=0)
        else:
            t_median_middle = None
        
        if len(j_middle_spots) > 0:
            j_middle_array = np.array(j_middle_spots)
            j_median_middle = np.median(j_middle_array, axis=0)
        else:
            j_median_middle = None
        
        if verbose and t_median_middle is not None:
            print(f"  T-shape median middle spot: ({t_median_middle[0]:.1f}, {t_median_middle[1]:.1f})")
        if verbose and j_median_middle is not None:
            print(f"  J-shape median middle spot: ({j_median_middle[0]:.1f}, {j_median_middle[1]:.1f})")
        
        # Define geometric validation function
        def validate_shape_geometry(t_shape_candidate, j_shape_candidate, spot_spacing, t_median=None, j_median=None, 
                                   is_t_outlier_correction=False, is_j_outlier_correction=False):
            """Validate geometric relationship between T and J shapes.
            Returns True if shapes satisfy expected spatial constraints.
            
            Constraints:
            - Horizontal spacing: J-shape should be 16-19×SPOT_SPACING right of T-shape
            - Vertical alignment: Within ±2×SPOT_SPACING
            - If one shape is missing, validate candidate position against median of other shape
            - If correcting an outlier, validate against the SAME shape's median (tighter tolerance)
            
            Args:
                t_shape_candidate: T shape candidate processed by this function.
                j_shape_candidate: J shape candidate processed by this function.
                spot_spacing: Spot spacing processed by this function.
                t_median: T median processed by this function.
                j_median: J median processed by this function.
                is_t_outlier_correction: Is t outlier correction processed by this function.
                is_j_outlier_correction: Is j outlier correction processed by this function.
            
            Returns:
                object: Validation result for shape geometry.
            """
            # Special case: Outlier correction - validate against same shape's median
            if is_j_outlier_correction and j_shape_candidate is not None and j_median is not None:
                j_middle = j_shape_candidate[1]
                distance_from_median = np.sqrt(np.sum((j_middle - j_median)**2))
                # For outlier correction, must be within 0.5×SPOT_SPACING of median
                if verbose and abs(j_middle[0] - 467) < 5:
                    print(f"    [Outlier Validation] J candidate: {j_middle}, J median: {j_median}")
                    print(f"    [Outlier Validation] Distance from median: {distance_from_median:.2f} (max: {0.5*spot_spacing:.2f})")
                    print(f"    [Outlier Validation] Will {'REJECT' if distance_from_median > 0.5*spot_spacing else 'ACCEPT'}")
                return distance_from_median <= 0.5 * spot_spacing
            
            if is_t_outlier_correction and t_shape_candidate is not None and t_median is not None:
                t_middle = t_shape_candidate[1]
                distance_from_median = np.sqrt(np.sum((t_middle - t_median)**2))
                # For outlier correction, must be within 0.5×SPOT_SPACING of median
                return distance_from_median <= 0.5 * spot_spacing
            
            # If both shapes present, validate their relationship
            if t_shape_candidate is not None and j_shape_candidate is not None:
                # Get extreme positions
                t_rightmost_x = np.max(t_shape_candidate[:, 0])
                t_top_y = np.min(t_shape_candidate[:, 1])
                t_bottom_y = np.max(t_shape_candidate[:, 1])
                
                j_leftmost_x = np.min(j_shape_candidate[:, 0])
                j_top_y = np.min(j_shape_candidate[:, 1])
                j_bottom_y = np.max(j_shape_candidate[:, 1])
                
                # Check horizontal spacing (16-19×SPOT_SPACING)
                horizontal_gap = j_leftmost_x - t_rightmost_x
                expected_min = 16 * spot_spacing
                expected_max = 19 * spot_spacing
                
                if not (expected_min <= horizontal_gap <= expected_max):
                    return False
                
                # Check vertical alignment (±2×SPOT_SPACING)
                vertical_tolerance = 2 * spot_spacing
                top_diff = abs(t_top_y - j_top_y)
                bottom_diff = abs(t_bottom_y - j_bottom_y)
                
                if top_diff > vertical_tolerance or bottom_diff > vertical_tolerance:
                    return False
                
                return True
            
            # If T-shape is missing but we have J-shape candidate, validate against expected relationship
            if t_shape_candidate is None and j_shape_candidate is not None and t_median is not None:
                # J-shape middle should be ~19×SPOT_SPACING to the right of T median (from actual medians: 456-131=325, 325/17=19)
                j_middle = j_shape_candidate[1]  # middle spot (index 1)
                expected_horizontal_offset = 19.0 * spot_spacing
                expected_j_x = t_median[0] + expected_horizontal_offset
                expected_j_y = t_median[1] + 1.0 * spot_spacing
                horizontal_diff = abs(j_middle[0] - expected_j_x)
                vertical_diff = abs(j_middle[1] - expected_j_y)  # J is ~1×spacing below T
                
                # DEBUG for image 5
                if verbose and abs(j_middle[0] - 467) < 5 and abs(j_middle[1] - 217) < 5:
                    print(f"    [Validation Debug] J candidate: {j_middle}, T median: {t_median}")
                    print(f"    [Validation Debug] Expected J: [{expected_j_x:.1f}, {expected_j_y:.1f}]")
                    print(f"    [Validation Debug] Horizontal diff: {horizontal_diff:.2f} (tolerance: {spot_spacing:.2f})")
                    print(f"    [Validation Debug] Vertical diff: {vertical_diff:.2f} (tolerance: {spot_spacing:.2f})")
                    print(f"    [Validation Debug] Will {'REJECT' if (horizontal_diff > spot_spacing or vertical_diff > spot_spacing) else 'ACCEPT'}")
                
                # Median-based validation: use tighter tolerance (±1×SPOT_SPACING) since median is robust
                if horizontal_diff > 1.0 * spot_spacing or vertical_diff > 1.0 * spot_spacing:
                    return False
                return True
            
            # If J-shape is missing but we have T-shape candidate, validate against expected relationship  
            if j_shape_candidate is None and t_shape_candidate is not None and j_median is not None:
                # T-shape middle should be ~19×SPOT_SPACING to the left of J median (from actual medians: 456-131=325, 325/17=19)
                t_middle = t_shape_candidate[1]  # middle spot (index 1)
                expected_horizontal_offset = 19.0 * spot_spacing
                horizontal_diff = abs(t_middle[0] - (j_median[0] - expected_horizontal_offset))
                vertical_diff = abs(t_middle[1] - (j_median[1] - 1.0 * spot_spacing))  # T is ~1×spacing above J
                
                # Median-based validation: use tighter tolerance (±1×SPOT_SPACING) since median is robust
                if horizontal_diff > 1.0 * spot_spacing or vertical_diff > 1.0 * spot_spacing:
                    return False
                return True
            
            # If no validation possible, return True
            return True
        
        # Define objective function for Powell optimization
        def compute_shape_intensity(middle_pos, img, shape_type, h_spacing, v_spacing):
            """Compute total intensity from all 4 spots of a shape.
            Returns negative intensity (for minimization).
            
            Args:
                middle_pos: Middle pos processed by this function.
                img: Img processed by this function.
                shape_type: Shape type processed by this function.
                h_spacing: H spacing processed by this function.
                v_spacing: V spacing processed by this function.
            
            Returns:
                object: Computed shape intensity.
            """
            mx, my = middle_pos
            height, width = img.shape
            
            # Reconstruct 4-spot positions based on geometric pattern
            if shape_type == 'T':
                # T-shape: [top, center, bottom, right]
                positions = [
                    (mx, my - v_spacing),        # top
                    (mx, my),                    # center (middle)
                    (mx, my + v_spacing),        # bottom
                    (mx + h_spacing, my)         # right
                ]
            else:  # 'J'
                # J-shape: [top, middle, bottom_left, bottom_right]
                positions = [
                    (mx, my - v_spacing),        # top
                    (mx, my),                    # middle
                    (mx - h_spacing, my + v_spacing),  # bottom_left
                    (mx, my + v_spacing)         # bottom_right
                ]
            
            # Sample intensity at each position with bounds checking
            total_intensity = 0.0
            for x, y in positions:
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < width and 0 <= iy < height:
                    total_intensity += img[iy, ix]
                else:
                    # Heavily penalize out-of-bounds positions
                    return 1e10
            
            # Return negative (for minimization)
            return -total_intensity
        
        # Helper function for parallel multi-start optimization
        def optimize_single_start(initial_guess, bounds, img, shape_type, h_spacing, v_spacing):
            """Run L-BFGS-B optimization from a single starting position.
            Returns (result, intensity) tuple.
            
            Args:
                initial_guess: Initial guess processed by this function.
                bounds: Bounds processed by this function.
                img: Img processed by this function.
                shape_type: Shape type processed by this function.
                h_spacing: H spacing processed by this function.
                v_spacing: V spacing processed by this function.
            
            Returns:
                tuple: Optimize single start.
            """
            result = minimize(
                compute_shape_intensity,
                initial_guess,
                args=(img, shape_type, h_spacing, v_spacing),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            intensity = -result.fun if result.success else -np.inf
            return result, intensity
        
        # Compute intensity thresholds from successfully detected shapes for validation
        # We'll use 10th percentile of total intensities as minimum acceptable
        t_shape_intensities = []
        j_shape_intensities = []
        
        for i, t_shape in enumerate(t_shapes):
            if t_shape is not None:
                img = bg_sub_images_float.isel(image_idx=i).values
                total_int = sum(img[int(round(y)), int(round(x))] 
                               for x, y in t_shape 
                               if 0 <= int(round(x)) < img.shape[1] and 0 <= int(round(y)) < img.shape[0])
                t_shape_intensities.append(total_int)
        
        for i, j_shape in enumerate(j_shapes):
            if j_shape is not None:
                img = bg_sub_images_float.isel(image_idx=i).values
                total_int = sum(img[int(round(y)), int(round(x))] 
                               for x, y in j_shape 
                               if 0 <= int(round(x)) < img.shape[1] and 0 <= int(round(y)) < img.shape[0])
                j_shape_intensities.append(total_int)
        
        # Set intensity thresholds (10th percentile of detected shapes)
        t_intensity_threshold = np.percentile(t_shape_intensities, 10) if len(t_shape_intensities) > 0 else 0
        j_intensity_threshold = np.percentile(j_shape_intensities, 10) if len(j_shape_intensities) > 0 else 0
        
        if verbose:
            print(f"  Intensity validation thresholds: T={t_intensity_threshold:.1f}, J={j_intensity_threshold:.1f}")
        
        # Refine missing or outlier shapes
        t_refined_count = 0
        j_refined_count = 0
        t_outlier_fixed = 0
        j_outlier_fixed = 0
        t_diffevo_used = 0
        j_diffevo_used = 0
        
        for i in range(n_images):
            img = bg_sub_images_float.isel(image_idx=i).values
            t_shape = t_shapes[i]
            j_shape = j_shapes[i]
            
            # Debug: Save initial state
            if verbose and i < 20:
                initial_j = j_shapes[i].copy() if j_shapes[i] is not None else None
                initial_t = t_shapes[i].copy() if t_shapes[i] is not None else None
            
            needs_t_refinement = False
            needs_j_refinement = False
            
            # Check if T-shape needs refinement (missing or outlier >2×SPOT_SPACING from median)
            if t_median_middle is not None:
                if t_shape is None:
                    needs_t_refinement = True
                    if verbose and i < 20:
                        print(f"  [Debug] Image {i}: T-shape missing, will optimize")
                else:
                    t_middle = t_shape[1]
                    t_distance = np.sqrt(np.sum((t_middle - t_median_middle)**2))
                    # Check if outlier (>2×SPOT_SPACING from median)
                    if t_distance > 2 * scaled_spot_spacing:
                        needs_t_refinement = True
                        if verbose and i < 20:
                            print(f"  [Debug] Image {i}: T-shape is outlier (dist={t_distance:.1f} > {2*scaled_spot_spacing:.1f}), will optimize")
                    elif verbose and i < 20:
                        print(f"  [Debug] Image {i}: T-shape detected, skipping optimization")
            
            # Check if J-shape needs refinement (missing or outlier >2×SPOT_SPACING from median)
            if j_median_middle is not None:
                if j_shape is None:
                    needs_j_refinement = True
                    if verbose and i < 20:
                        print(f"  [Debug] Image {i}: J-shape missing, will optimize")
                else:
                    j_middle = j_shape[1]
                    j_distance = np.sqrt(np.sum((j_middle - j_median_middle)**2))
                    # Check if outlier (>2×SPOT_SPACING from median)
                    if j_distance > 2 * scaled_spot_spacing:
                        needs_j_refinement = True
                        if verbose and i < 20:
                            print(f"  [Debug] Image {i}: J-shape is outlier (dist={j_distance:.1f} > {2*scaled_spot_spacing:.1f}), will optimize")
                    elif verbose and i < 20:
                        print(f"  [Debug] Image {i}: J-shape detected, skipping optimization")
            
            # Determine optimization order: 
            # If one shape is an outlier and the other is missing, fix the outlier first
            # so we can use its corrected position to find the missing shape
            t_is_outlier = needs_t_refinement and (t_shape is not None)
            j_is_outlier = needs_j_refinement and (j_shape is not None)
            t_is_missing = needs_t_refinement and (t_shape is None)
            j_is_missing = needs_j_refinement and (j_shape is None)
            
            # Reorder: optimize outliers before missing shapes
            optimize_j_first = j_is_outlier and t_is_missing
            optimize_t_first = t_is_outlier and j_is_missing
            
            if verbose and i < 20 and (optimize_j_first or optimize_t_first):
                if optimize_j_first:
                    print(f"  [Debug] Image {i}: Optimizing J-shape (outlier) BEFORE T-shape (missing) to get good reference")
                elif optimize_t_first:
                    print(f"  [Debug] Image {i}: Optimizing T-shape (outlier) BEFORE J-shape (missing) to get good reference")
            
            # Perform hybrid optimization if needed (with iterative validation retry)
            # Optimize T-shape first (unless J-shape outlier should be fixed first)
            if needs_t_refinement and not optimize_j_first:
                # Determine search center
                was_missing = (t_shape is None)
                if was_missing and j_shapes[i] is not None:
                    # T-shape missing: use J-shape position as reference if available
                    # T is ~19×spacing left (from median analysis: 456-131=325, 325/17=19) and ~1×spacing above J
                    j_middle = j_shapes[i][1]
                    t_search_center = np.array([j_middle[0] - 19.0 * scaled_spot_spacing, j_middle[1] - 1.0 * scaled_spot_spacing])
                    if verbose and i < 20:
                        print(f"  [Debug] Image {i}: Optimizing T-shape using J-shape reference at {j_middle}")
                        print(f"           T search center: {t_search_center} (derived from J-shape, -19×spacing horiz, -1×spacing vert)")
                else:
                    # T-shape is outlier or no J-shape: use median
                    t_search_center = t_median_middle
                    if verbose and i < 20:
                        status = "outlier" if not was_missing else "missing"
                        print(f"  [Debug] Image {i}: Optimizing T-shape ({status}) using median center {t_search_center}")
                
                t_accepted = False
                t_iteration = 0
                max_iterations = 10
                
                t_is_outlier = not was_missing  # Track if this is outlier correction
                
                while not t_accepted and t_iteration < max_iterations:
                    t_iteration += 1
                    
                    # STAGE 1: Multi-start Powell optimization (3×3 grid of initial positions)
                    # Create grid of starting positions around median (±2/3×SPOT_SPACING)
                    grid_spacing = (2.0 / 3.0) * scaled_spot_spacing
                    grid_offsets = [
                        (-grid_spacing, -grid_spacing),
                        (-grid_spacing, 0),
                        (-grid_spacing, grid_spacing),
                        (0, -grid_spacing),
                        (0, 0),
                        (0, grid_spacing),
                        (grid_spacing, -grid_spacing),
                        (grid_spacing, 0),
                        (grid_spacing, grid_spacing)
                    ]
                    
                    best_result = None
                    best_intensity = -np.inf
                    
                    # Define bounds: ±1.5×SPOT_SPACING from search center
                    bounds = [
                        (t_search_center[0] - 1.5*scaled_spot_spacing, t_search_center[0] + 1.5*scaled_spot_spacing),
                        (t_search_center[1] - 1.5*scaled_spot_spacing, t_search_center[1] + 1.5*scaled_spot_spacing)
                    ]
                    
                    # Parallelize multi-start optimization: run all 9 starting positions in parallel
                    initial_guesses = [t_search_center + np.array([dx, dy]) for dx, dy in grid_offsets]
                    
                    parallel_results = Parallel(n_jobs=-1, backend='threading')(
                        delayed(optimize_single_start)(guess, bounds, img, 'T', 
                                                       scaled_horizontal_spacing, scaled_vertical_spacing)
                        for guess in initial_guesses
                    )
                    
                    # Select best result from parallel runs
                    for result, intensity in parallel_results:
                        if result.success and intensity > best_intensity:
                            best_result = result
                            best_intensity = intensity
                    
                    # STAGE 2: Validate result (intensity + geometry)
                    if best_result and best_result.success and best_intensity > t_intensity_threshold:
                        # Reconstruct T-shape candidate
                        mx, my = best_result.x
                        t_shape_candidate = np.array([
                            [mx, my - scaled_vertical_spacing],  # top
                            [mx, my],                            # center (middle)
                            [mx, my + scaled_vertical_spacing],  # bottom
                            [mx + scaled_horizontal_spacing, my] # right
                        ], dtype=np.float32)
                        
                        # Validate geometry with existing J-shape (if available)
                        geom_valid = validate_shape_geometry(t_shape_candidate, j_shapes[i], scaled_spot_spacing, 
                                                            t_median=t_median_middle, j_median=j_median_middle,
                                                            is_t_outlier_correction=t_is_outlier)
                        
                        if geom_valid:
                            # Accept L-BFGS-B result - both intensity and geometry valid
                            was_missing = (t_shapes[i] is None)
                            t_shapes[i] = t_shape_candidate
                            if was_missing:
                                t_refined_count += 1
                            else:
                                t_outlier_fixed += 1
                            t_accepted = True
                            if verbose and i < 20:
                                status = "found" if was_missing else "corrected"
                                print(f"  [Debug] Image {i}: T-shape {status} via L-BFGS-B at middle={t_shape_candidate[1]}")
                            continue  # Exit while loop
                    
                    # If L-BFGS-B failed validation or had low intensity, use differential evolution
                    if not t_accepted:
                        # STAGE 3: Fallback to differential evolution (more robust, slower)
                        bounds_de = [
                            (t_search_center[0] - 1.5*scaled_spot_spacing, t_search_center[0] + 1.5*scaled_spot_spacing),
                            (t_search_center[1] - 1.5*scaled_spot_spacing, t_search_center[1] + 1.5*scaled_spot_spacing)
                        ]
                        
                        result_de = differential_evolution(
                            compute_shape_intensity,
                            bounds_de,
                            args=(img, 'T', scaled_horizontal_spacing, scaled_vertical_spacing),
                            maxiter=300,
                            popsize=10,
                            atol=1e-4,
                            tol=1e-4
                        )
                        
                        if result_de.success and -result_de.fun > t_intensity_threshold * 0.5:  # More lenient for DE
                            mx, my = result_de.x
                            t_shape_candidate = np.array([
                                [mx, my - scaled_vertical_spacing],  # top
                                [mx, my],                            # center (middle)
                                [mx, my + scaled_vertical_spacing],  # bottom
                                [mx + scaled_horizontal_spacing, my] # right
                            ], dtype=np.float32)
                            
                            # Validate geometry
                            geom_valid = validate_shape_geometry(t_shape_candidate, j_shapes[i], scaled_spot_spacing,
                                                                t_median=t_median_middle, j_median=j_median_middle,
                                                                is_t_outlier_correction=t_is_outlier)
                            
                            if geom_valid:
                                # Accept DE result - both intensity and geometry valid
                                was_missing = (t_shapes[i] is None)
                                t_shapes[i] = t_shape_candidate
                                if was_missing:
                                    t_refined_count += 1
                                else:
                                    t_outlier_fixed += 1
                                t_diffevo_used += 1
                                t_accepted = True
                                if verbose and i < 20:
                                    status = "found" if was_missing else "corrected"
                                    print(f"  [Debug] Image {i}: T-shape {status} via DE at middle={t_shape_candidate[1]}")
                                continue  # Exit while loop
                    
                    # If still not accepted after max iterations, give up
                    # (reference spots may not exist in this image)
                if verbose and i < 20 and needs_t_refinement:
                    if t_shapes[i] is not None:
                        print(f"  [Debug] Image {i}: T-shape FINAL position after optimization: middle={t_shapes[i][1]}")
                    else:
                        print(f"  [Debug] Image {i}: T-shape optimization FAILED - remains None")
            
            # Debug: Check if J-shape was modified during T-shape optimization
            if verbose and i < 20 and initial_j is not None:
                if j_shapes[i] is None:
                    print(f"  [WARNING] Image {i}: J-shape was detected but is now None!")
                elif not np.array_equal(initial_j, j_shapes[i]):
                    print(f"  [WARNING] Image {i}: J-shape was MODIFIED during T-shape optimization!")
                    print(f"            Initial middle: {initial_j[1]}, Current middle: {j_shapes[i][1]}")
                    print(f"            needs_t_refinement={needs_t_refinement}, needs_j_refinement={needs_j_refinement}")
                elif verbose and i < 20 and needs_t_refinement:
                    print(f"  [Debug] Image {i}: J-shape UNCHANGED during T-optimization (still at {j_shapes[i][1]})")
            
            if needs_j_refinement:
                # Determine search center
                was_missing = (j_shape is None)
                if was_missing and t_shapes[i] is not None:
                    # J-shape missing: use T-shape position as reference if available
                    # J is ~19×spacing right (from median analysis: 456-131=325, 325/17=19) and ~1×spacing below T
                    t_middle = t_shapes[i][1]
                    j_search_center = np.array([t_middle[0] + 19.0 * scaled_spot_spacing, t_middle[1] + 1.0 * scaled_spot_spacing])
                    if verbose and i < 20:
                        print(f"  [Debug] Image {i}: Optimizing J-shape using T-shape reference at {t_middle}")
                        print(f"           J search center: {j_search_center} (derived from T-shape, +19×spacing horiz, +1×spacing vert)")
                else:
                    # J-shape is outlier or no T-shape: use median
                    j_search_center = j_median_middle
                    if verbose and i < 20:
                        status = "outlier" if not was_missing else "missing"
                        print(f"  [Debug] Image {i}: Optimizing J-shape ({status}) using median center {j_search_center}")
                
                j_accepted = False
                j_iteration = 0
                max_iterations = 10
                
                j_is_outlier = not was_missing  # Track if this is outlier correction
                
                while not j_accepted and j_iteration < max_iterations:
                    j_iteration += 1
                    
                    # STAGE 1: Multi-start Powell optimization (3×3 grid of initial positions)
                    # Create grid of starting positions around median (±2/3×SPOT_SPACING)
                    grid_spacing = (2.0 / 3.0) * scaled_spot_spacing
                    grid_offsets = [
                        (-grid_spacing, -grid_spacing),
                        (-grid_spacing, 0),
                        (-grid_spacing, grid_spacing),
                        (0, -grid_spacing),
                        (0, 0),
                        (0, grid_spacing),
                        (grid_spacing, -grid_spacing),
                        (grid_spacing, 0),
                        (grid_spacing, grid_spacing)
                    ]
                    
                    best_result = None
                    best_intensity = -np.inf
                    
                    # Define bounds: ±1.5×SPOT_SPACING from search center
                    bounds = [
                        (j_search_center[0] - 1.5*scaled_spot_spacing, j_search_center[0] + 1.5*scaled_spot_spacing),
                        (j_search_center[1] - 1.5*scaled_spot_spacing, j_search_center[1] + 1.5*scaled_spot_spacing)
                    ]
                    
                    # Parallelize multi-start optimization: run all 9 starting positions in parallel
                    initial_guesses = [j_search_center + np.array([dx, dy]) for dx, dy in grid_offsets]
                    
                    parallel_results = Parallel(n_jobs=-1, backend='threading')(
                        delayed(optimize_single_start)(guess, bounds, img, 'J', 
                                                       scaled_horizontal_spacing, scaled_vertical_spacing)
                        for guess in initial_guesses
                    )
                    
                    # Select best result from parallel runs
                    for result, intensity in parallel_results:
                        if result.success and intensity > best_intensity:
                            best_result = result
                            best_intensity = intensity
                    
                    # STAGE 2: Validate result (intensity + geometry)
                    if best_result and best_result.success and best_intensity > j_intensity_threshold:
                        # Reconstruct J-shape candidate
                        mx, my = best_result.x
                        j_shape_candidate = np.array([
                            [mx, my - scaled_vertical_spacing],                    # top
                            [mx, my],                                              # middle
                            [mx - scaled_horizontal_spacing, my + scaled_vertical_spacing],  # bottom_left
                            [mx, my + scaled_vertical_spacing]                     # bottom_right
                        ], dtype=np.float32)
                        
                        # Validate geometry with existing T-shape (if available)
                        geom_valid = validate_shape_geometry(t_shapes[i], j_shape_candidate, scaled_spot_spacing,
                                                            t_median=t_median_middle, j_median=j_median_middle,
                                                            is_j_outlier_correction=j_is_outlier)
                        
                        if geom_valid:
                            # Accept L-BFGS-B result - both intensity and geometry valid
                            was_missing = (j_shapes[i] is None)
                            j_shapes[i] = j_shape_candidate
                            if was_missing:
                                j_refined_count += 1
                            else:
                                j_outlier_fixed += 1
                            j_accepted = True
                            continue  # Exit while loop
                    
                    # If L-BFGS-B failed validation or had low intensity, use differential evolution
                    if not j_accepted:
                        # STAGE 3: Fallback to differential evolution (more robust, slower)
                        bounds_de = [
                            (j_search_center[0] - 1.5*scaled_spot_spacing, j_search_center[0] + 1.5*scaled_spot_spacing),
                            (j_search_center[1] - 1.5*scaled_spot_spacing, j_search_center[1] + 1.5*scaled_spot_spacing)
                        ]
                        
                        result_de = differential_evolution(
                            compute_shape_intensity,
                            bounds_de,
                            args=(img, 'J', scaled_horizontal_spacing, scaled_vertical_spacing),
                            maxiter=300,
                            popsize=10,
                            atol=1e-4,
                            tol=1e-4
                        )
                        
                        if result_de.success and -result_de.fun > j_intensity_threshold * 0.5:  # More lenient for DE
                            mx, my = result_de.x
                            j_shape_candidate = np.array([
                                [mx, my - scaled_vertical_spacing],                    # top
                                [mx, my],                                              # middle
                                [mx - scaled_horizontal_spacing, my + scaled_vertical_spacing],  # bottom_left
                                [mx, my + scaled_vertical_spacing]                     # bottom_right
                            ], dtype=np.float32)
                            
                            # Validate geometry
                            geom_valid = validate_shape_geometry(t_shapes[i], j_shape_candidate, scaled_spot_spacing,
                                                                t_median=t_median_middle, j_median=j_median_middle,
                                                                is_j_outlier_correction=j_is_outlier)
                            
                            if geom_valid:
                                # Accept DE result - both intensity and geometry valid
                                was_missing = (j_shapes[i] is None)
                                j_shapes[i] = j_shape_candidate
                                if was_missing:
                                    j_refined_count += 1
                                else:
                                    j_outlier_fixed += 1
                                j_diffevo_used += 1
                                j_accepted = True
                                continue  # Exit while loop
                    
                    # If still not accepted after max iterations, give up
                    # (reference spots may not exist in this image)
            
            # Now optimize T-shape if it was deferred (J-shape outlier was fixed first)
            if needs_t_refinement and optimize_j_first:
                # Determine search center (J should be corrected now)
                was_missing = (t_shape is None)
                if was_missing and j_shapes[i] is not None:
                    # T-shape missing: use corrected J-shape position as reference
                    # T is ~19×spacing left (from median analysis: 456-131=325, 325/17=19) and ~1×spacing above J
                    j_middle = j_shapes[i][1]
                    t_search_center = np.array([j_middle[0] - 19.0 * scaled_spot_spacing, j_middle[1] - 1.0 * scaled_spot_spacing])
                    if verbose and i < 20:
                        print(f"  [Debug] Image {i}: Now optimizing T-shape using CORRECTED J-shape at {j_middle}")
                        print(f"           T search center: {t_search_center} (derived from corrected J-shape, -19×spacing horiz, -1×spacing vert)")
                else:
                    # No J-shape available: use median
                    t_search_center = t_median_middle
                    if verbose and i < 20:
                        status = "outlier" if not was_missing else "missing"
                        print(f"  [Debug] Image {i}: Optimizing T-shape ({status}) using median center {t_search_center}")
                
                t_accepted = False
                t_iteration = 0
                max_iterations = 10
                
                while not t_accepted and t_iteration < max_iterations:
                    t_iteration += 1
                    
                    # STAGE 1: Multi-start Powell optimization (3×3 grid of initial positions)
                    # Create grid of starting positions around median (±2/3×SPOT_SPACING)
                    grid_spacing = (2.0 / 3.0) * scaled_spot_spacing
                    grid_offsets = [
                        (-grid_spacing, -grid_spacing),
                        (-grid_spacing, 0),
                        (-grid_spacing, grid_spacing),
                        (0, -grid_spacing),
                        (0, 0),
                        (0, grid_spacing),
                        (grid_spacing, -grid_spacing),
                        (grid_spacing, 0),
                        (grid_spacing, grid_spacing)
                    ]
                    
                    best_result = None
                    best_intensity = -np.inf
                    
                    # Define bounds: ±1.5×SPOT_SPACING from search center
                    bounds = [
                        (t_search_center[0] - 1.5*scaled_spot_spacing, t_search_center[0] + 1.5*scaled_spot_spacing),
                        (t_search_center[1] - 1.5*scaled_spot_spacing, t_search_center[1] + 1.5*scaled_spot_spacing)
                    ]
                    
                    # Parallelize multi-start optimization: run all 9 starting positions in parallel
                    initial_guesses = [t_search_center + np.array([dx, dy]) for dx, dy in grid_offsets]
                    
                    parallel_results = Parallel(n_jobs=-1, backend='threading')(
                        delayed(optimize_single_start)(guess, bounds, img, 'T', 
                                                       scaled_horizontal_spacing, scaled_vertical_spacing)
                        for guess in initial_guesses
                    )
                    
                    # Select best result from parallel runs
                    for result, intensity in parallel_results:
                        if result.success and intensity > best_intensity:
                            best_result = result
                            best_intensity = intensity
                    
                    # STAGE 2: Validate result (intensity + geometry)
                    if best_result and best_result.success and best_intensity > t_intensity_threshold:
                        # Reconstruct T-shape candidate
                        mx, my = best_result.x
                        t_shape_candidate = np.array([
                            [mx, my - scaled_vertical_spacing],  # top
                            [mx, my],                            # center (middle)
                            [mx, my + scaled_vertical_spacing],  # bottom
                            [mx + scaled_horizontal_spacing, my] # right
                        ], dtype=np.float32)
                        
                        # Validate geometry with existing J-shape (if available)
                        geom_valid = validate_shape_geometry(t_shape_candidate, j_shapes[i], scaled_spot_spacing, 
                                                            t_median=t_median_middle, j_median=j_median_middle)
                        
                        if geom_valid:
                            # Accept L-BFGS-B result - both intensity and geometry valid
                            was_missing = (t_shapes[i] is None)
                            t_shapes[i] = t_shape_candidate
                            if was_missing:
                                t_refined_count += 1
                            else:
                                t_outlier_fixed += 1
                            t_accepted = True
                            if verbose and i < 20:
                                status = "found" if was_missing else "corrected"
                                print(f"  [Debug] Image {i}: T-shape {status} via L-BFGS-B at middle={t_shape_candidate[1]}")
                            continue  # Exit while loop
                    
                    # If L-BFGS-B failed validation or had low intensity, use differential evolution
                    if not t_accepted:
                        # STAGE 3: Fallback to differential evolution (more robust, slower)
                        bounds_de = [
                            (t_search_center[0] - 1.5*scaled_spot_spacing, t_search_center[0] + 1.5*scaled_spot_spacing),
                            (t_search_center[1] - 1.5*scaled_spot_spacing, t_search_center[1] + 1.5*scaled_spot_spacing)
                        ]
                        
                        result_de = differential_evolution(
                            compute_shape_intensity,
                            bounds_de,
                            args=(img, 'T', scaled_horizontal_spacing, scaled_vertical_spacing),
                            maxiter=300,
                            popsize=10,
                            atol=1e-4,
                            tol=1e-4
                        )
                        
                        if result_de.success and -result_de.fun > t_intensity_threshold * 0.5:  # More lenient for DE
                            mx, my = result_de.x
                            t_shape_candidate = np.array([
                                [mx, my - scaled_vertical_spacing],  # top
                                [mx, my],                            # center (middle)
                                [mx, my + scaled_vertical_spacing],  # bottom
                                [mx + scaled_horizontal_spacing, my] # right
                            ], dtype=np.float32)
                            
                            # Validate geometry
                            geom_valid = validate_shape_geometry(t_shape_candidate, j_shapes[i], scaled_spot_spacing,
                                                                t_median=t_median_middle, j_median=j_median_middle)
                            
                            if geom_valid:
                                # Accept DE result - both intensity and geometry valid
                                was_missing = (t_shapes[i] is None)
                                t_shapes[i] = t_shape_candidate
                                if was_missing:
                                    t_refined_count += 1
                                else:
                                    t_outlier_fixed += 1
                                t_diffevo_used += 1
                                t_accepted = True
                                if verbose and i < 20:
                                    status = "found" if was_missing else "corrected"
                                    print(f"  [Debug] Image {i}: T-shape {status} via DE at middle={t_shape_candidate[1]}")
                                continue  # Exit while loop
                    
                    # If still not accepted after max iterations, give up
                    # (reference spots may not exist in this image)
                if verbose and i < 20 and needs_t_refinement:
                    if t_shapes[i] is not None:
                        print(f"  [Debug] Image {i}: T-shape FINAL position after deferred optimization: middle={t_shapes[i][1]}")
                    else:
                        print(f"  [Debug] Image {i}: T-shape deferred optimization FAILED - remains None")
            
            # Debug: Check if T-shape was modified during J-shape optimization
            if verbose and i < 20 and initial_t is not None:
                if t_shapes[i] is None:
                    print(f"  [WARNING] Image {i}: T-shape was detected but is now None!")
                elif not np.array_equal(initial_t, t_shapes[i]):
                    print(f"  [WARNING] Image {i}: T-shape was MODIFIED during J-shape optimization!")
                    print(f"            Initial middle: {initial_t[1]}, Current middle: {t_shapes[i][1]}")
        
        # Debug: Final check for unexpected modifications
        if verbose:
            for i in range(min(20, n_images)):
                if i < 10:
                    pass  # Initial state already saved above
        
        # Update stored results
        self._reference_spots = {'t_shape': t_shapes, 'j_shape': j_shapes}
        
        # Update final counts
        final_t_count = sum(1 for t in t_shapes if t is not None)
        final_j_count = sum(1 for j in j_shapes if j is not None)
        
        if verbose:
            print("Hybrid optimization refinement complete!")
            if t_refined_count > 0 or t_outlier_fixed > 0:
                print(f"  T-shape: recovered {t_refined_count} missing, fixed {t_outlier_fixed} outliers")
                if t_diffevo_used > 0:
                    print(f"           (differential evolution used for {t_diffevo_used} cases)")
            if j_refined_count > 0 or j_outlier_fixed > 0:
                print(f"  J-shape: recovered {j_refined_count} missing, fixed {j_outlier_fixed} outliers")
                if j_diffevo_used > 0:
                    print(f"           (differential evolution used for {j_diffevo_used} cases)")
            print(f"  Final counts: T-shapes {final_t_count}/{n_images}, J-shapes {final_j_count}/{n_images}")
        
        return self._reference_spots

    def _find_max_intensity_position(self, image, x, y, wiggle_offsets, aperture_mask, ap_center, height, width):
        """Find max intensity position.
        
        Args:
            image: Image processed by this function.
            x: X processed by this function.
            y: Y processed by this function.
            wiggle_offsets: Wiggle offsets processed by this function.
            aperture_mask: Aperture mask processed by this function.
            ap_center: Ap center processed by this function.
            height: Height processed by this function.
            width: Width processed by this function.
        
        Returns:
            tuple: Found max intensity position.
        """
        x_int, y_int = int(round(x)), int(round(y))
        best_intensity, best_x, best_y = -np.inf, x, y
        half_ap = aperture_mask.shape[0] // 2
        
        for dx, dy in wiggle_offsets:
            test_x, test_y = x_int + dx, y_int + dy
            y_start, y_end = test_y - half_ap, test_y + half_ap + 1
            x_start, x_end = test_x - half_ap, test_x + half_ap + 1
            
            if y_start < 0 or y_end > height or x_start < 0 or x_end > width:
                continue
            
            region = image[y_start:y_end, x_start:x_end]
            if region.shape != aperture_mask.shape:
                continue
            
            integrated_intensity = np.sum(region[aperture_mask])
            if integrated_intensity > best_intensity:
                best_intensity = integrated_intensity
                best_x, best_y = float(test_x), float(test_y)
        
        return best_x, best_y, np.sqrt((best_x - x)**2 + (best_y - y)**2)

    def refine_reference_spot_positions(self, wiggle_radius=4.0, integration_radius=5.0, 
                                        angle_threshold_deg=1.0, verbose=True):
        """Refine reference spot positions using two-step approach.
        
        Step 1: Geometric correction
        - Validates T-shape and J-shape geometry (vertical alignment, spacing)
        - Corrects positions to ideal geometry if deviations exceed threshold
        - Ensures vertical stem is truly vertical and spacing is exact
        
        Step 2: Intensity-based refinement
        - For each spot, searches within wiggle_radius for maximum integrated intensity
        - Uses circular aperture with integration_radius for intensity calculation
        - Finds sub-pixel position that maximizes signal
        
        Args:
            wiggle_radius: Wiggle radius processed by this function.
            integration_radius: Integration radius processed by this function.
            angle_threshold_deg: Threshold value used to filter, classify, or flag results.
            verbose: Whether to emit progress output while running.
        
        Returns:
            object: Refined reference spot positions.
        """
        if self._reference_spots is None:
            raise RuntimeError("Reference spots not detected. Call detect_reference_spots() first.")
        
        n_images = len(self._reference_spots['t_shape'])
        
        # Scale dimensional parameters
        scaled_wiggle_radius = self._scale(wiggle_radius)
        scaled_integration_radius = self._scale(integration_radius)
        
        if verbose:
            print(f"\nRefining reference spot positions in {n_images} images...")
            print(f"  Scaled parameters: wiggle_radius={scaled_wiggle_radius:.2f}, "
                  f"integration_radius={scaled_integration_radius:.2f}")
        
        ap_size = int(np.ceil(scaled_integration_radius)) * 2 + 1
        ap_center = ap_size // 2
        yy, xx = np.ogrid[:ap_size, :ap_size]
        aperture_mask = ((xx - ap_center)**2 + (yy - ap_center)**2) <= scaled_integration_radius**2
        
        wiggle_int = int(np.ceil(scaled_wiggle_radius))
        wiggle_offsets = [(dx, dy) for dy in range(-wiggle_int, wiggle_int+1) 
                         for dx in range(-wiggle_int, wiggle_int+1) if dx**2 + dy**2 <= scaled_wiggle_radius**2]
        
        # Convert to float32 once (2-3x faster than per-image conversion)
        if self._background_subtracted_images.dtype != np.float32:
            bg_sub_images_float = self._background_subtracted_images.astype(np.float32)
        else:
            bg_sub_images_float = self._background_subtracted_images
        
        # Helper function for parallel processing
        def refine_single_image(i):
            """Refine single image.
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Refined single image.
            """
            img = bg_sub_images_float.isel(image_idx=i).values
            height, width = img.shape
            
            # T-shape processing
            t_shape = self._reference_spots['t_shape'][i]
            refined_t = None
            t_was_corrected = False
            if t_shape is not None:
                # Step 1: Geometric correction
                t_shape_corrected, t_was_corrected = self._check_and_correct_t_shape_geometry(
                    t_shape, angle_threshold_deg, self._scale(self.SPOT_SPACING))
                
                # Step 2: Intensity-based refinement
                refined_t = np.zeros_like(t_shape_corrected)
                for j, (x, y) in enumerate(t_shape_corrected):
                    new_x, new_y, _ = self._find_max_intensity_position(
                        img, x, y, wiggle_offsets, aperture_mask, ap_center, height, width)
                    refined_t[j] = [new_x, new_y]
            
            # J-shape processing
            j_shape = self._reference_spots['j_shape'][i]
            refined_j = None
            j_was_corrected = False
            if j_shape is not None:
                # Step 1: Geometric correction
                j_shape_corrected, j_was_corrected = self._check_and_correct_j_shape_geometry(
                    j_shape, angle_threshold_deg, self._scale(self.SPOT_SPACING))
                
                # Step 2: Intensity-based refinement
                refined_j = np.zeros_like(j_shape_corrected)
                for j, (x, y) in enumerate(j_shape_corrected):
                    new_x, new_y, _ = self._find_max_intensity_position(
                        img, x, y, wiggle_offsets, aperture_mask, ap_center, height, width)
                    refined_j[j] = [new_x, new_y]
            
            return i, refined_t, refined_j, t_was_corrected, j_was_corrected
        
        # Parallel processing
        if verbose:
            print("  Processing images in parallel...")
        
        results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(refine_single_image)(i) for i in range(n_images)
        )
        
        # Sort and unpack results
        results.sort(key=lambda x: x[0])
        
        refined_t_shapes, refined_j_shapes = [], []
        t_geom_corrected_count, j_geom_corrected_count = 0, 0
        
        for i, refined_t, refined_j, t_corrected, j_corrected in results:
            refined_t_shapes.append(refined_t)
            refined_j_shapes.append(refined_j)
            if t_corrected: t_geom_corrected_count += 1
            if j_corrected: j_geom_corrected_count += 1
        
        self._refined_reference_spots = {'t_shape': refined_t_shapes, 'j_shape': refined_j_shapes}
        self._refinement_params = {
            'wiggle_radius': wiggle_radius, 
            'integration_radius': integration_radius,
            'angle_threshold_deg': angle_threshold_deg
        }
        
        if verbose:
            print("Refinement complete!")
            print(f"  T-shapes geometry corrected: {t_geom_corrected_count}/{n_images}")
            print(f"  J-shapes geometry corrected: {j_geom_corrected_count}/{n_images}")
        
        return self._refined_reference_spots
    
    def _check_and_correct_t_shape_geometry(self, t_shape, angle_threshold_deg=1.0, scaled_spot_spacing=None):
        """Check and correct T-shape geometry.
        
        T-shape structure: [top(0), center(1), bottom(2), right(3)]
        - Spots 0,1,2 should be vertically aligned within angle_threshold_deg
        - Distance between vertical neighbors: 2*SPOT_SPACING
        - Spot 3 should be 1*SPOT_SPACING to the right of spot 1
        
        Args:
            t_shape: T shape processed by this function.
            angle_threshold_deg: Threshold value used to filter, classify, or flag results.
            scaled_spot_spacing: Scaled spot spacing processed by this function.
        
        Returns:
            tuple: Check result for and correct t shape geometry.
        """
        if t_shape is None:
            return None, False
        
        corrected = t_shape.copy()
        was_corrected = False
        
        # Vertical spots: top (0), center (1), bottom (2)
        vertical_x = corrected[[0, 1, 2], 0]
        vertical_y = corrected[[0, 1, 2], 1]
        
        # Calculate angle from vertical (top to bottom)
        dx = vertical_x[2] - vertical_x[0]  # bottom_x - top_x
        dy = vertical_y[2] - vertical_y[0]  # bottom_y - top_y
        
        if abs(dy) > 1e-6:
            angle_deg = np.abs(np.degrees(np.arctan(dx / dy)))
        else:
            angle_deg = 90.0 if abs(dx) > 1e-6 else 0.0
        
        # If not vertically aligned, correct positions
        if angle_deg > angle_threshold_deg:
            mean_x = np.mean(vertical_x)
            center_y = corrected[1, 1]  # Use center y as reference
            
            corrected[0, 0] = mean_x
            corrected[0, 1] = center_y - 2 * scaled_spot_spacing
            corrected[1, 0] = mean_x
            # corrected[1, 1] stays the same
            corrected[2, 0] = mean_x
            corrected[2, 1] = center_y + 2 * scaled_spot_spacing
            was_corrected = True
        
        # Check 4th spot (right, index 3)
        # Should be 1*SPOT_SPACING to the right of center (index 1)
        expected_x = corrected[1, 0] + scaled_spot_spacing
        expected_y = corrected[1, 1]
        
        deviation = np.sqrt((corrected[3, 0] - expected_x)**2 + (corrected[3, 1] - expected_y)**2)
        
        if deviation > scaled_spot_spacing / 2:
            corrected[3, 0] = expected_x
            corrected[3, 1] = expected_y
            was_corrected = True
        
        return corrected, was_corrected
    
    
    def _check_and_correct_j_shape_geometry(self, j_shape, angle_threshold_deg=1.0, scaled_spot_spacing=None):
        """Check and correct J-shape geometry.
        
        J-shape structure: [top(0), middle(1), bottom_left(2), bottom_right(3)]
        - Spots 0,1,3 should be vertically aligned within angle_threshold_deg
        - Distance between vertical neighbors: 2*SPOT_SPACING
        - Spot 2 should be 1*SPOT_SPACING to the left of spot 3
        
        Args:
            j_shape: J shape processed by this function.
            angle_threshold_deg: Threshold value used to filter, classify, or flag results.
            scaled_spot_spacing: Scaled spot spacing processed by this function.
        
        Returns:
            tuple: Check result for and correct j shape geometry.
        """
        if j_shape is None:
            return None, False
        
        corrected = j_shape.copy()
        was_corrected = False
        
        # Vertical spots: top (0), middle (1), bottom_right (3)
        vertical_x = corrected[[0, 1, 3], 0]
        vertical_y = corrected[[0, 1, 3], 1]
        
        # Calculate angle from vertical (top to bottom)
        dx = vertical_x[2] - vertical_x[0]  # bottom_right_x - top_x
        dy = vertical_y[2] - vertical_y[0]  # bottom_right_y - top_y
        
        if abs(dy) > 1e-6:
            angle_deg = np.abs(np.degrees(np.arctan(dx / dy)))
        else:
            angle_deg = 90.0 if abs(dx) > 1e-6 else 0.0
        
        # If not vertically aligned, correct positions
        if angle_deg > angle_threshold_deg:
            mean_x = np.mean(vertical_x)
            middle_y = corrected[1, 1]  # Use middle y as reference
            
            corrected[0, 0] = mean_x
            corrected[0, 1] = middle_y - 2 * scaled_spot_spacing
            corrected[1, 0] = mean_x
            # corrected[1, 1] stays the same
            corrected[3, 0] = mean_x
            corrected[3, 1] = middle_y + 2 * scaled_spot_spacing
            was_corrected = True
        
        # Check 4th spot (bottom_left, index 2)
        # Should be 1*SPOT_SPACING to the left of bottom_right (index 3)
        expected_x = corrected[3, 0] - scaled_spot_spacing
        expected_y = corrected[3, 1]
        
        deviation = np.sqrt((corrected[2, 0] - expected_x)**2 + (corrected[2, 1] - expected_y)**2)
        
        if deviation > scaled_spot_spacing / 2:
            corrected[2, 0] = expected_x
            corrected[2, 1] = expected_y
            was_corrected = True
        
        return corrected, was_corrected

    def _estimate_vertical_angle(self, points):
        """Estimate vertical angle.
        
        Args:
            points: Points processed by this function.
        
        Returns:
            object: Estimated vertical angle.
        """
        if len(points) < 2:
            return np.nan
        x, y = points[:, 0], points[:, 1]
        y_mean, x_mean = np.mean(y), np.mean(x)
        numerator = np.sum((y - y_mean) * (x - x_mean))
        denominator = np.sum((y - y_mean) ** 2)
        if denominator == 0:
            return 0.0
        return np.arctan(numerator / denominator)

    def estimate_reference_angles(self, verbose=True):
        """Estimate array rotation angles from reference spot alignments.
        
        Computes rotation angle by fitting a line through the vertical stem of
        T-shape and J-shape patterns. The angle represents deviation from perfect
        vertical alignment (0° = perfectly vertical).
        
        Algorithm:
        1. For T-shape: fit line through top, center, bottom spots (indices 0,1,2)
        2. For J-shape: fit line through top, center, bottom spots (indices 0,1,3)
        3. Compute angle of each fitted line from vertical
        4. Average T and J angles if both available, otherwise use single estimate
        
        Args:
            verbose: Whether to emit progress output while running.
        
        Returns:
            object: Estimated reference angles.
        """
        if self._refined_reference_spots is None:
            raise RuntimeError("Reference spots not refined. Call refine_reference_spot_positions() first.")
        
        n_images = len(self._refined_reference_spots['t_shape'])
        if verbose:
            print(f"\nEstimating reference angles from {n_images} images...")
        
        # Helper function for parallel processing
        def estimate_angle_single(i):
            """Estimate angle single.
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Estimated angle single.
            """
            t_shape = self._refined_reference_spots['t_shape'][i]
            j_shape = self._refined_reference_spots['j_shape'][i]
            
            t_angle = self._estimate_vertical_angle(t_shape[[0, 1, 2], :]) if t_shape is not None else np.nan
            j_angle = self._estimate_vertical_angle(j_shape[[0, 1, 3], :]) if j_shape is not None else np.nan
            
            if not np.isnan(t_angle) and not np.isnan(j_angle):
                avg_angle = (t_angle + j_angle) / 2
            elif not np.isnan(t_angle):
                avg_angle = t_angle
            elif not np.isnan(j_angle):
                avg_angle = j_angle
            else:
                avg_angle = np.nan
            
            return i, t_angle, j_angle, avg_angle
        
        # Parallel processing
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(estimate_angle_single)(i) for i in range(n_images)
        )
        
        # Sort and unpack results
        results.sort(key=lambda x: x[0])
        
        t_angles, j_angles, avg_angles = [], [], []
        for i, t_angle, j_angle, avg_angle in results:
            t_angles.append(t_angle)
            j_angles.append(j_angle)
            avg_angles.append(avg_angle)
        
        self._reference_angles = {
            't_shape_angles': np.array(t_angles), 'j_shape_angles': np.array(j_angles),
            'average_angles': np.array(avg_angles),
            't_shape_angles_deg': np.degrees(t_angles), 'j_shape_angles_deg': np.degrees(j_angles),
            'average_angles_deg': np.degrees(avg_angles)
        }
        
        if verbose:
            valid_avg = [a for a in avg_angles if not np.isnan(a)]
            print(f"Angle estimation complete! Mean: {np.degrees(np.mean(valid_avg)):.3f}°")
        return self._reference_angles

    def _create_rotated_square_mask(self, shape, center, width, height, angle):
        """Create rotated square mask.
        
        Args:
            shape: Shape processed by this function.
            center: Center processed by this function.
            width: Width processed by this function.
            height: Height processed by this function.
            angle: Angle processed by this function.
        
        Returns:
            object: Created rotated square mask.
        """
        img_height, img_width = shape
        center_x, center_y = center
        y_grid, x_grid = np.ogrid[:img_height, :img_width]
        x_centered, y_centered = x_grid - center_x, y_grid - center_y
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        x_rotated = x_centered * cos_a - y_centered * sin_a
        y_rotated = x_centered * sin_a + y_centered * cos_a
        return (np.abs(x_rotated) <= width/2) & (np.abs(y_rotated) <= height/2)

    def apply_square_mask(self, verbose=True) -> xr.DataArray:
        """Apply rotated rectangular masks to isolate the spot grid region.
        
        Creates a rotated square mask centered on the spot grid, with dimensions
        and offsets appropriate for PTK or STK arrays. The mask is estimated from
        T-shape and/or J-shape reference spots.
        
        Mask specifications:
        - PTK: 15×15 spot spacings (322.5×322.5 px), offset 1×5 spacings from references
          (PTK_SQUARE_SIDE_MULTIPLIER, PTK_H_OFFSET_MULTIPLIER, PTK_V_OFFSET_MULTIPLIER)
        - STK: 13×13 spot spacings (279.5×279.5 px), offset 2×4 spacings from references
          (STK_SQUARE_SIDE_MULTIPLIER, STK_H_OFFSET_MULTIPLIER, STK_V_OFFSET_MULTIPLIER)
        
        Selection logic when both T and J have been detected:
        - If centers agree (within 1 spacing): average both estimates
        - If centers disagree: use either T or J based on which gives the mask closer to the expected spot grid location
        
        Algorithm:
        1. Estimate mask bounds from T-shape and/or J-shape positions
        2. Calculate mask dimensions based on peptide type
        3. Create rotated rectangular mask using estimated angle
        4. Apply mask to background-subtracted images
        
        Args:
            verbose: Whether to emit progress output while running.
        
        Returns:
            xr.DataArray: Applied result for square mask.
        """
        if self._refined_reference_spots is None:
            raise RuntimeError("Reference spots not refined. Call refine_reference_spot_positions() first.")
        if self._reference_angles is None:
            raise RuntimeError("Reference angles not estimated. Call estimate_reference_angles() first.")
        
        n_images = len(self._background_subtracted_images.image_idx)
        height, width = self._background_subtracted_images.shape[1:]
        
        if verbose:
            print(f"\nApplying square masks to {n_images} images...")
        
        # Base offsets for reference dimensions (in units of SPOT_SPACING)
        H_OFFSET_PTK = self.PTK_H_OFFSET_MULTIPLIER * self.SPOT_SPACING
        H_OFFSET_STK = self.STK_H_OFFSET_MULTIPLIER * self.SPOT_SPACING
        V_OFFSET_PTK = self.PTK_V_OFFSET_MULTIPLIER * self.SPOT_SPACING
        V_OFFSET_STK = self.STK_V_OFFSET_MULTIPLIER * self.SPOT_SPACING
        
        # Scale offsets
        scaled_h_offset_ptk = self._scale(H_OFFSET_PTK)
        scaled_h_offset_stk = self._scale(H_OFFSET_STK)
        scaled_v_offset_ptk = self._scale(V_OFFSET_PTK)
        scaled_v_offset_stk = self._scale(V_OFFSET_STK)
        scaled_spot_spacing = self._scale(self.SPOT_SPACING)
        
        square_masked_array = np.zeros((n_images, height, width), dtype=np.float32)
        mask_params_list = []
        
        peptides_type = self._get_peptides_type()
        h_offset = scaled_h_offset_ptk if peptides_type == 'PTK' else scaled_h_offset_stk
        v_offset = scaled_v_offset_ptk if peptides_type == 'PTK' else scaled_v_offset_stk
        square_side_multiplier = self.PTK_SQUARE_SIDE_MULTIPLIER if peptides_type == 'PTK' else self.STK_SQUARE_SIDE_MULTIPLIER
        square_side_length = square_side_multiplier * scaled_spot_spacing
        
        if verbose:
            print(f"  Scaled offsets: h_offset={h_offset:.1f}, v_offset={v_offset:.1f}, "
                  f"square_side={square_side_length:.1f}")
        
        # Helper function to estimate mask from t-shape
        def estimate_from_t_shape(t):
            """Estimate from t shape.
            
            Args:
                t: T processed by this function.
            
            Returns:
                tuple: Estimated from t shape.
            """
            T1, T2, T3, T4 = t
            top_y = T1[1] - v_offset
            left_x = T4[0] + h_offset
            bottom_y = top_y + square_side_length
            right_x = left_x + square_side_length
            center_x = (left_x + right_x) / 2
            center_y = (top_y + bottom_y) / 2
            return left_x, right_x, top_y, bottom_y, center_x, center_y
        
        # Helper function to estimate mask from j-shape
        def estimate_from_j_shape(j):
            """Estimate from j shape.
            
            Args:
                j: J processed by this function.
            
            Returns:
                tuple: Estimated from j shape.
            """
            J1, J2, J3, J4 = j
            bottom_y = J4[1] + v_offset
            right_x = J3[0] - h_offset
            top_y = bottom_y - square_side_length
            left_x = right_x - square_side_length
            center_x = (left_x + right_x) / 2
            center_y = (top_y + bottom_y) / 2
            return left_x, right_x, top_y, bottom_y, center_x, center_y
        
        # Convert to float32 once (2-3x faster than per-image conversion)
        if self._background_subtracted_images.dtype != np.float32:
            bg_sub_images_float = self._background_subtracted_images.astype(np.float32)
        else:
            bg_sub_images_float = self._background_subtracted_images
        
        # Helper function for parallel processing
        def process_square_mask(i):
            """Process square mask.
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Processed square mask.
            """
            img = bg_sub_images_float.isel(image_idx=i).values
            t_shape = self._refined_reference_spots['t_shape'][i]
            j_shape = self._refined_reference_spots['j_shape'][i]
            avg_angle = self._reference_angles['average_angles'][i]
            
            if (t_shape is None and j_shape is None) or np.isnan(avg_angle):
                return i, np.zeros_like(img), None, None
            
            # Determine which mask to use
            chosen_source = None
            choice_type = None
            
            if t_shape is not None and j_shape is not None:
                # Both shapes available - estimate both masks and choose
                left_x_t, right_x_t, top_y_t, bottom_y_t, center_x_t, center_y_t = estimate_from_t_shape(t_shape)
                left_x_j, right_x_j, top_y_j, bottom_y_j, center_x_j, center_y_j = estimate_from_j_shape(j_shape)
                
                # Calculate horizontal and vertical distances between the two masks
                h_distance = abs(center_x_t - center_x_j)
                v_distance = abs(center_y_t - center_y_j)
                
                # Check if masks are close or far (using scaled SPOT_SPACING as threshold)
                if h_distance <= scaled_spot_spacing and v_distance <= scaled_spot_spacing:
                    # Masks are close - choose t-shape based mask
                    left_x, right_x, top_y, bottom_y = left_x_t, right_x_t, top_y_t, bottom_y_t
                    center_x, center_y = center_x_t, center_y_t
                    chosen_source = 't_shape_close'
                    choice_type = 'both_close'
                else:
                    # Masks are far - choose the one closer to circular mask center
                    circular_center_x, circular_center_y = self._centers[i]
                    dist_t = np.sqrt((center_x_t - circular_center_x)**2 + (center_y_t - circular_center_y)**2)
                    dist_j = np.sqrt((center_x_j - circular_center_x)**2 + (center_y_j - circular_center_y)**2)
                    
                    if dist_t <= dist_j:
                        left_x, right_x, top_y, bottom_y = left_x_t, right_x_t, top_y_t, bottom_y_t
                        center_x, center_y = center_x_t, center_y_t
                        chosen_source = 't_shape_far'
                    else:
                        left_x, right_x, top_y, bottom_y = left_x_j, right_x_j, top_y_j, bottom_y_j
                        center_x, center_y = center_x_j, center_y_j
                        chosen_source = 'j_shape_far'
                    choice_type = 'both_far'
            
            elif t_shape is not None:
                # Only t-shape available
                left_x, right_x, top_y, bottom_y, center_x, center_y = estimate_from_t_shape(t_shape)
                chosen_source = 't_shape_only'
                choice_type = 't_only'
            
            elif j_shape is not None:
                # Only j-shape available
                left_x, right_x, top_y, bottom_y, center_x, center_y = estimate_from_j_shape(j_shape)
                chosen_source = 'j_shape_only'
                choice_type = 'j_only'
            
            mask_params = {
                'left_x': left_x, 'right_x': right_x, 'top_y': top_y, 'bottom_y': bottom_y,
                'center_x': center_x, 'center_y': center_y, 'angle_rad': avg_angle,
                'angle_deg': np.degrees(avg_angle), 'peptides_type': peptides_type,
                'h_offset': h_offset, 'v_offset': v_offset, 'square_side_length': square_side_length,
                'chosen_source': chosen_source
            }
            
            mask = self._create_rotated_square_mask(
                (height, width),
                (center_x, center_y),
                right_x - left_x,
                bottom_y - top_y,
                avg_angle,
            )
            masked_img = img.copy()
            masked_img[~mask] = 0
            
            return i, masked_img, mask_params, choice_type
        
        # Parallel processing
        if verbose:
            print("  Processing images in parallel...")
        
        results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(process_square_mask)(i) for i in range(n_images)
        )
        
        # Sort and unpack results
        results.sort(key=lambda x: x[0])
        
        mask_params_list = []
        t_mask_count = 0
        j_mask_count = 0
        both_close_count = 0
        both_far_count = 0
        
        for i, masked_img, mask_params, choice_type in results:
            square_masked_array[i] = masked_img
            mask_params_list.append(mask_params)
            
            if choice_type == 't_only':
                t_mask_count += 1
            elif choice_type == 'j_only':
                j_mask_count += 1
            elif choice_type == 'both_close':
                both_close_count += 1
            elif choice_type == 'both_far':
                both_far_count += 1
        
        self._square_mask_params = mask_params_list
        self._square_masked_images = xr.DataArray(
            data=square_masked_array, dims=self._background_subtracted_images.dims,
            coords=self._background_subtracted_images.coords,
            attrs={**self._background_subtracted_images.attrs, 'processing': 'square_masked'}
        )
        
        if verbose:
            n_masked = sum(1 for p in mask_params_list if p is not None)
            print(f"Square mask application complete! Successfully masked: {n_masked}/{n_images}")
            if t_mask_count > 0:
                print(f"  T-shape only: {t_mask_count}")
            if j_mask_count > 0:
                print(f"  J-shape only: {j_mask_count}")
            if both_close_count > 0:
                print(f"  Both shapes (close, used T): {both_close_count}")
            if both_far_count > 0:
                print(f"  Both shapes (far, chose closer): {both_far_count}")
        return self._square_masked_images

    def compute_spot_intensities(self, integration_radius=5.0, verbose=True):
        """Compute integrated, maximum, median intensities and saturation fraction for all spots in the grid.
        
        Grid specifications:
        - PTK: 14×14 grid (196 spots) - defined by PTK_GRID_ROWS, PTK_GRID_COLS
        - STK: 12×12 grid (144 spots) - defined by STK_GRID_ROWS, STK_GRID_COLS
        - Spacing: 21.5 pixels between centers (SPOT_SPACING)
        - Edge offset: 21.5 pixels from square mask boundary (EDGE_OFFSET)
        
        Computes:
        - Integrated intensity: sum of pixel values within aperture
        - Max intensity: maximum pixel value within aperture
        - Median intensity: median pixel value within aperture
        - Saturation fraction: fraction of pixels at the maximum value within each spot
          (indicates potential signal clipping if many pixels share the same max)
        
        Args:
            integration_radius: Integration radius processed by this function.
            verbose: Whether to emit progress output while running.
        
        Returns:
            object: Computed spot intensities.
        """
        if self._square_masked_images is None:
            raise RuntimeError("Square masks not applied. Call apply_square_mask() first.")
        
        # Scale dimensional parameters
        scaled_integration_radius = self._scale(integration_radius)
        scaled_edge_offset = self._scale(self.EDGE_OFFSET)
        scaled_spot_spacing = self._scale(self.SPOT_SPACING)
        
        n_images = len(self._square_masked_images.image_idx)
        height, width = self._square_masked_images.shape[1:]
        
        # Convert to float32 once (2-3x faster than per-image conversion)
        if self._square_masked_images.dtype != np.float32:
            square_masked_images_float = self._square_masked_images.astype(np.float32)
        else:
            square_masked_images_float = self._square_masked_images
        
        peptides_type = self._get_peptides_type()
        n_rows = 14 if peptides_type == 'PTK' else 12
        n_cols = 14 if peptides_type == 'PTK' else 12
        
        if verbose:
            print(f"\nComputing spot intensities for {n_images} images...")
            print(f"  Peptides type: {peptides_type}, Grid: {n_rows}×{n_cols}")
            print(f"  Spacing: {self.SPOT_SPACING}px → {scaled_spot_spacing:.2f}px (scaled)")
            print(f"  Edge offset: {self.EDGE_OFFSET}px → {scaled_edge_offset:.2f}px (scaled)")
            print(f"  Integration radius: {integration_radius}px → {scaled_integration_radius:.2f}px (scaled)")
        
        ap_size = int(np.ceil(scaled_integration_radius)) * 2 + 1
        ap_center = ap_size // 2
        yy, xx = np.ogrid[:ap_size, :ap_size]
        aperture_mask = ((xx - ap_center)**2 + (yy - ap_center)**2) <= scaled_integration_radius**2
        half_ap = ap_size // 2
        n_pixels_in_aperture = int(np.sum(aperture_mask))
        
        if verbose:
            print(f"  Aperture: {ap_size}×{ap_size}px, {n_pixels_in_aperture} pixels/spot")
        
        # Helper function for parallel processing
        def compute_intensities_single(i):
            """Compute intensities single.
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Computed intensities single.
            """
            img = square_masked_images_float.isel(image_idx=i).values
            mask_params = self._square_mask_params[i]
            
            if mask_params is None:
                return (i, np.full((n_rows, n_cols), np.nan), np.full((n_rows, n_cols), np.nan),
                        np.full((n_rows, n_cols), np.nan), np.full((n_rows, n_cols), np.nan), None, False)
            
            center_x = mask_params['center_x']
            center_y = mask_params['center_y']
            angle = mask_params['angle_rad']
            mask_width = mask_params['right_x'] - mask_params['left_x']
            mask_height = mask_params['bottom_y'] - mask_params['top_y']
            
            # Calculate grid positions relative to mask center (before rotation)
            col_positions = -mask_width/2 + scaled_edge_offset + np.arange(n_cols) * scaled_spot_spacing
            row_positions = -mask_height/2 + scaled_edge_offset + np.arange(n_rows) * scaled_spot_spacing
            
            col_grid, row_grid = np.meshgrid(col_positions, row_positions)
            x_rel, y_rel = col_grid.flatten(), row_grid.flatten()
            
            # Rotate positions using the same sign convention as the square mask.
            angle = mask_params['angle_rad']
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = x_rel * cos_a - y_rel * sin_a
            y_rot = x_rel * sin_a + y_rel * cos_a
            
            x_img = x_rot + center_x
            y_img = y_rot + center_y
            
            spot_positions = np.stack([x_img.reshape(n_rows, n_cols), 
                                       y_img.reshape(n_rows, n_cols)], axis=-1)
            
            # Use numba-compiled vectorized computation (5-10x faster than nested loops)
            intensities, max_intensities, median_intensities, saturation_fractions = \
                self._compute_spot_metrics_numba(img, x_img, y_img, aperture_mask, 
                                                 half_ap, n_rows, n_cols)
            
            return (i, intensities, max_intensities, median_intensities, 
                    saturation_fractions, spot_positions, True)
        
        # Parallel processing
        if verbose:
            print("  Processing images in parallel...")
        
        results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(compute_intensities_single)(i) for i in range(n_images)
        )
        
        # Sort and unpack results
        results.sort(key=lambda x: x[0])
        
        intensities_array = np.zeros((n_images, n_rows, n_cols), dtype=np.float32)
        max_intensities_array = np.zeros((n_images, n_rows, n_cols), dtype=np.float32)
        median_intensities_array = np.zeros((n_images, n_rows, n_cols), dtype=np.float32)
        saturation_fractions_array = np.zeros((n_images, n_rows, n_cols), dtype=np.float32)
        self._spot_grid_positions = []
        n_valid_images = 0
        
        for (i, intensities, max_ints, median_ints, sat_fracs, positions, is_valid) in results:
            intensities_array[i] = intensities
            max_intensities_array[i] = max_ints
            median_intensities_array[i] = median_ints
            saturation_fractions_array[i] = sat_fracs
            self._spot_grid_positions.append(positions)
            if is_valid:
                n_valid_images += 1
        
        self._spot_intensity_params = {
            'integration_radius': scaled_integration_radius, 'spot_spacing': scaled_spot_spacing,
            'edge_offset': scaled_edge_offset, 'n_rows': n_rows, 'n_cols': n_cols,
            'peptides_type': peptides_type, 'aperture_size': ap_size,
            'pixels_per_aperture': n_pixels_in_aperture
        }
        
        # Build coordinates
        coords_dict = {'spot_row': np.arange(n_rows), 'spot_col': np.arange(n_cols)}
        for coord_name, coord_data in self._square_masked_images.coords.items():
            if coord_name in ['y', 'x']:
                continue
            if 'image_idx' in coord_data.dims:
                coords_dict[coord_name] = coord_data
            elif coord_name == 'image_idx':
                coords_dict['image_idx'] = coord_data
        
        self._spot_intensities = xr.DataArray(
            data=intensities_array,
            dims=['image_idx', 'spot_row', 'spot_col'],
            coords=coords_dict,
            attrs={
                'processing': 'spot_intensities', 'integration_radius': scaled_integration_radius,
                'spot_spacing': scaled_spot_spacing, 'edge_offset': scaled_edge_offset,
                'n_rows': n_rows, 'n_cols': n_cols, 'peptides_type': peptides_type
            }
        )
        
        self._spot_max_intensities = xr.DataArray(
            data=max_intensities_array,
            dims=['image_idx', 'spot_row', 'spot_col'],
            coords=coords_dict,
            attrs={
                'processing': 'spot_max_intensities', 'integration_radius': scaled_integration_radius,
                'spot_spacing': scaled_spot_spacing, 'edge_offset': scaled_edge_offset,
                'n_rows': n_rows, 'n_cols': n_cols, 'peptides_type': peptides_type
            }
        )
        
        self._spot_median_intensities = xr.DataArray(
            data=median_intensities_array,
            dims=['image_idx', 'spot_row', 'spot_col'],
            coords=coords_dict,
            attrs={
                'processing': 'spot_median_intensities', 'integration_radius': scaled_integration_radius,
                'spot_spacing': scaled_spot_spacing, 'edge_offset': scaled_edge_offset,
                'n_rows': n_rows, 'n_cols': n_cols, 'peptides_type': peptides_type
            }
        )
        
        self._spot_saturation_fractions = xr.DataArray(
            data=saturation_fractions_array,
            dims=['image_idx', 'spot_row', 'spot_col'],
            coords=coords_dict,
            attrs={
                'processing': 'spot_saturation_fractions', 'integration_radius': scaled_integration_radius,
                'description': 'Fraction of pixels at spot maximum intensity (indicates potential clipping)',
                'n_rows': n_rows, 'n_cols': n_cols, 'peptides_type': peptides_type
            }
        )
        
        if verbose:
            valid_mask = ~np.isnan(intensities_array)
            n_valid_spots = np.sum(valid_mask)
            print("\nSpot intensity computation complete!")
            print(f"  Valid images: {n_valid_images}/{n_images}")
            print(f"  Valid spots: {n_valid_spots}/{intensities_array.size}")
            if n_valid_spots > 0:
                valid_vals = intensities_array[valid_mask]
                valid_max_vals = max_intensities_array[valid_mask]
                valid_median_vals = median_intensities_array[valid_mask]
                valid_sat_vals = saturation_fractions_array[valid_mask]
                print(f"  Integrated intensity - Mean: {np.mean(valid_vals):.2f}, Std: {np.std(valid_vals):.2f}")
                print(f"  Max intensity - Mean: {np.mean(valid_max_vals):.2f}, Std: {np.std(valid_max_vals):.2f}")
                print(f"  Median intensity - Mean: {np.mean(valid_median_vals):.2f}, Std: {np.std(valid_median_vals):.2f}")
                print(f"  Saturation fraction - Mean: {np.mean(valid_sat_vals):.4f}, Max: {np.max(valid_sat_vals):.4f}")
                # Spots with high saturation fraction (e.g., >0.1 means >10% pixels at max)
                n_potentially_saturated = np.sum(valid_sat_vals > 0.1)
                print(f"  Spots with >10% pixels at max: {n_potentially_saturated}/{n_valid_spots}")
        
        return self._spot_intensities

    def get_spot_positions(self, image_idx=0):
        """Get spot grid positions for a specific image.
        
        Args:
            image_idx: Zero-based index selecting the image.
        
        Returns:
            object: Requested spot positions.
        """
        if self._spot_grid_positions is None:
            return None
        if image_idx >= len(self._spot_grid_positions):
            raise ValueError(f"image_idx {image_idx} out of range")
        return self._spot_grid_positions[image_idx]

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_intensity_numba(img, x_img, y_img, aperture_mask, half_ap, height, width, min_valid):
        """Numba-compiled inner loop for computing grid intensity.
        10-100x faster than pure Python loop.
        
        Args:
            img: Img processed by this function.
            x_img: X img processed by this function.
            y_img: Y img processed by this function.
            aperture_mask: Aperture mask processed by this function.
            half_ap: Half ap processed by this function.
            height: Height processed by this function.
            width: Width processed by this function.
            min_valid: Min valid processed by this function.
        
        Returns:
            object: Computed intensity Numba.
        """
        total_intensity = 0.0
        n_valid = 0
        ap_h, ap_w = aperture_mask.shape
        
        for idx in range(len(x_img)):
            x, y = x_img[idx], y_img[idx]
            x_int = int(round(x))
            y_int = int(round(y))
            
            y_start = y_int - half_ap
            y_end = y_int + half_ap + 1
            x_start = x_int - half_ap
            x_end = x_int + half_ap + 1
            
            if y_start < 0 or y_end > height or x_start < 0 or x_end > width:
                continue
            
            # Check region size
            if (y_end - y_start) != ap_h or (x_end - x_start) != ap_w:
                continue
            
            # Extract region and apply mask
            region = img[y_start:y_end, x_start:x_end]
            for i in range(ap_h):
                for j in range(ap_w):
                    if aperture_mask[i, j]:
                        total_intensity += region[i, j]
            n_valid += 1
        
        if n_valid < min_valid:
            return 1e10
        return -total_intensity

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_intensity_interpolated_numba(
        img,
        x_img,
        y_img,
        aperture_dx,
        aperture_dy,
        height,
        width,
        min_valid,
    ):
        """Compute total integrated intensity using bilinear interpolation at subpixel
        aperture coordinates. This keeps the Stage 1 objective smooth enough for
        continuous optimization.
        
        Args:
            img: Img processed by this function.
            x_img: X img processed by this function.
            y_img: Y img processed by this function.
            aperture_dx: Aperture dx processed by this function.
            aperture_dy: Aperture dy processed by this function.
            height: Height processed by this function.
            width: Width processed by this function.
            min_valid: Min valid processed by this function.
        
        Returns:
            object: Computed intensity interpolated Numba.
        """
        total_intensity = 0.0
        n_valid = 0

        for idx in range(len(x_img)):
            center_x = x_img[idx]
            center_y = y_img[idx]

            spot_valid = True
            spot_total = 0.0

            for offset_idx in range(len(aperture_dx)):
                x = center_x + aperture_dx[offset_idx]
                y = center_y + aperture_dy[offset_idx]

                if x < 0.0 or y < 0.0 or x >= (width - 1) or y >= (height - 1):
                    spot_valid = False
                    break

                x0 = int(np.floor(x))
                y0 = int(np.floor(y))
                x1 = x0 + 1
                y1 = y0 + 1

                if x1 >= width or y1 >= height:
                    spot_valid = False
                    break

                wx = x - x0
                wy = y - y0

                top = img[y0, x0] * (1.0 - wx) + img[y0, x1] * wx
                bottom = img[y1, x0] * (1.0 - wx) + img[y1, x1] * wx
                value = top * (1.0 - wy) + bottom * wy
                spot_total += value

            if spot_valid:
                total_intensity += spot_total
                n_valid += 1

        if n_valid < min_valid:
            return 1e10
        return -total_intensity
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_spot_metrics_numba(img, x_img, y_img, aperture_mask, half_ap, n_rows, n_cols):
        """Numba-compiled vectorized spot intensity computation.
        Processes all grid spots efficiently.
        Note: parallel=True removed to avoid conflict with joblib threading.
        
        Args:
            img: Img processed by this function.
            x_img: X img processed by this function.
            y_img: Y img processed by this function.
            aperture_mask: Aperture mask processed by this function.
            half_ap: Half ap processed by this function.
            n_rows: Number of rows used by this function.
            n_cols: Number of cols used by this function.
        
        Returns:
            tuple: Computed spot metrics Numba.
        """
        height, width = img.shape
        ap_h, ap_w = aperture_mask.shape
        
        intensities = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        max_intensities = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        median_intensities = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        saturation_fractions = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        
        n_pixels = int(np.sum(aperture_mask))
        
        for idx in range(len(x_img)):
            row = idx // n_cols
            col = idx % n_cols
            
            x, y = x_img[idx], y_img[idx]
            x_int = int(round(x))
            y_int = int(round(y))
            
            y_start = y_int - half_ap
            y_end = y_int + half_ap + 1
            x_start = x_int - half_ap
            x_end = x_int + half_ap + 1
            
            # Boundary check
            if y_start < 0 or y_end > height or x_start < 0 or x_end > width:
                continue
            
            # Size check
            if (y_end - y_start) != ap_h or (x_end - x_start) != ap_w:
                continue
            
            # Extract pixels within aperture
            region = img[y_start:y_end, x_start:x_end]
            aperture_pixels = np.empty(n_pixels, dtype=np.float32)
            pixel_idx = 0
            
            for i in range(ap_h):
                for j in range(ap_w):
                    if aperture_mask[i, j]:
                        aperture_pixels[pixel_idx] = region[i, j]
                        pixel_idx += 1
            
            # Compute metrics
            total = 0.0
            max_val = aperture_pixels[0]
            for k in range(n_pixels):
                total += aperture_pixels[k]
                if aperture_pixels[k] > max_val:
                    max_val = aperture_pixels[k]
            
            # Median calculation (simple sort for small arrays)
            sorted_pixels = np.sort(aperture_pixels)
            if n_pixels % 2 == 0:
                median_val = (sorted_pixels[n_pixels//2 - 1] + sorted_pixels[n_pixels//2]) / 2.0
            else:
                median_val = sorted_pixels[n_pixels//2]
            
            # Saturation fraction
            n_at_max = 0
            for k in range(n_pixels):
                if aperture_pixels[k] == max_val:
                    n_at_max += 1
            
            intensities[row, col] = total
            max_intensities[row, col] = max_val
            median_intensities[row, col] = median_val
            saturation_fractions[row, col] = float(n_at_max) / float(n_pixels)
        
        return intensities, max_intensities, median_intensities, saturation_fractions

    @staticmethod
    @jit(nopython=True, cache=True)
    def _grid_search_spot_position_numba(img, x0, y0, aperture_mask, half_ap, search_range, height, width):
        """Numba-compiled grid search for optimal spot position.
        Searches integer pixel positions within ±search_range of (x0, y0).
        
        Args:
            img: Img processed by this function.
            x0: X0 processed by this function.
            y0: Y0 processed by this function.
            aperture_mask: Aperture mask processed by this function.
            half_ap: Half ap processed by this function.
            search_range: Search range processed by this function.
            height: Height processed by this function.
            width: Width processed by this function.
        
        Returns:
            tuple: Grid search spot position Numba.
        """
        ap_h, ap_w = aperture_mask.shape
        
        # Convert search range to integer
        search_range_int = int(np.ceil(search_range))
        
        # Starting position in integer coordinates
        x0_int = int(round(x0))
        y0_int = int(round(y0))
        
        best_intensity = -1e10
        best_x = x0
        best_y = y0
        
        # Grid search over ±search_range pixels
        for dy in range(-search_range_int, search_range_int + 1):
            for dx in range(-search_range_int, search_range_int + 1):
                x_test = x0_int + dx
                y_test = y0_int + dy
                
                # Bounds check
                y_start = y_test - half_ap
                y_end = y_test + half_ap + 1
                x_start = x_test - half_ap
                x_end = x_test + half_ap + 1
                
                if y_start < 0 or y_end > height or x_start < 0 or x_end > width:
                    continue
                
                # Check region size
                if (y_end - y_start) != ap_h or (x_end - x_start) != ap_w:
                    continue
                
                # Extract region and compute intensity
                total_intensity = 0.0
                region = img[y_start:y_end, x_start:x_end]
                for i in range(ap_h):
                    for j in range(ap_w):
                        if aperture_mask[i, j]:
                            total_intensity += region[i, j]
                
                # Update best if this is better
                if total_intensity > best_intensity:
                    best_intensity = total_intensity
                    best_x = float(x_test)
                    best_y = float(y_test)
        
        return best_x, best_y, best_intensity

    @staticmethod
    @jit(nopython=True, cache=True)
    def _check_8_neighbors_numba(img, x0, y0, aperture_mask, half_ap, height, width):
        """Numba-compiled function to check if any of the 8-connected neighbors have better intensity.
        
        Args:
            img: Img processed by this function.
            x0: X0 processed by this function.
            y0: Y0 processed by this function.
            aperture_mask: Aperture mask processed by this function.
            half_ap: Half ap processed by this function.
            height: Height processed by this function.
            width: Width processed by this function.
        
        Returns:
            tuple: Check result for 8 neighbors Numba.
        """
        ap_h, ap_w = aperture_mask.shape
        
        x0_int = int(round(x0))
        y0_int = int(round(y0))
        
        # Compute current intensity
        y_start = y0_int - half_ap
        y_end = y0_int + half_ap + 1
        x_start = x0_int - half_ap
        x_end = x0_int + half_ap + 1
        
        # Bounds check for current position
        if (y_start < 0 or y_end > height or x_start < 0 or x_end > width or
            (y_end - y_start) != ap_h or (x_end - x_start) != ap_w):
            return False, 0.0
        
        # Current intensity
        current_intensity = 0.0
        region = img[y_start:y_end, x_start:x_end]
        for i in range(ap_h):
            for j in range(ap_w):
                if aperture_mask[i, j]:
                    current_intensity += region[i, j]
        
        # Check 8-connected neighbors
        best_neighbor_intensity = current_intensity
        has_better = False
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # Skip current position
                
                x_test = x0_int + dx
                y_test = y0_int + dy
                
                # Bounds check
                y_start_n = y_test - half_ap
                y_end_n = y_test + half_ap + 1
                x_start_n = x_test - half_ap
                x_end_n = x_test + half_ap + 1
                
                if (y_start_n < 0 or y_end_n > height or x_start_n < 0 or x_end_n > width or
                    (y_end_n - y_start_n) != ap_h or (x_end_n - x_start_n) != ap_w):
                    continue
                
                # Compute neighbor intensity
                neighbor_intensity = 0.0
                region_n = img[y_start_n:y_end_n, x_start_n:x_end_n]
                for i in range(ap_h):
                    for j in range(ap_w):
                        if aperture_mask[i, j]:
                            neighbor_intensity += region_n[i, j]
                
                if neighbor_intensity > best_neighbor_intensity:
                    best_neighbor_intensity = neighbor_intensity
                    has_better = True
        
        return has_better, best_neighbor_intensity

    def _compute_grid_intensity(
        self,
        img,
        center_x,
        center_y,
        angle,
        spacing,
        n_rows,
        n_cols,
        mask_width,
        mask_height,
        aperture_dx,
        aperture_dy,
        height,
        width,
        edge_offset,
    ):
        """Compute total integrated intensity for a grid with given parameters.
        Uses subpixel bilinear interpolation so Stage 1 optimization has a
        continuous per-image objective.
        
        Args:
            img: Img processed by this function.
            center_x: Center x processed by this function.
            center_y: Center y processed by this function.
            angle: Angle processed by this function.
            spacing: Spacing processed by this function.
            n_rows: Number of rows used by this function.
            n_cols: Number of cols used by this function.
            mask_width: Mask width processed by this function.
            mask_height: Mask height processed by this function.
            aperture_dx: Aperture dx processed by this function.
            aperture_dy: Aperture dy processed by this function.
            height: Height processed by this function.
            width: Width processed by this function.
            edge_offset: Edge offset processed by this function.
        
        Returns:
            object: Computed grid intensity.
        """
        col_positions = -mask_width/2 + edge_offset + np.arange(n_cols) * spacing
        row_positions = -mask_height/2 + edge_offset + np.arange(n_rows) * spacing
        
        col_grid, row_grid = np.meshgrid(col_positions, row_positions)
        x_rel, y_rel = col_grid.flatten(), row_grid.flatten()
        
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = x_rel * cos_a - y_rel * sin_a
        y_rot = x_rel * sin_a + y_rel * cos_a
        
        x_img = x_rot + center_x
        y_img = y_rot + center_y
        
        min_valid = int(n_rows * n_cols * 0.5)
        
        return self._compute_intensity_interpolated_numba(
            img,
            x_img,
            y_img,
            aperture_dx,
            aperture_dy,
            height,
            width,
            min_valid,
        )

    def _compute_positions_total_intensity(
        self,
        img,
        spot_positions,
        aperture_mask,
        half_ap,
    ) -> float:
        """Compute summed integrated intensity for one image's explicit spot positions.
        
        Args:
            img: Img processed by this function.
            spot_positions: Spot positions processed by this function.
            aperture_mask: Aperture mask processed by this function.
            half_ap: Half ap processed by this function.
        
        Returns:
            float: Computed positions total intensity.
        """
        if spot_positions is None:
            return np.nan

        x_img = spot_positions[:, :, 0].flatten()
        y_img = spot_positions[:, :, 1].flatten()
        height, width = img.shape
        min_valid = int(spot_positions.shape[0] * spot_positions.shape[1] * 0.5)
        return -self._compute_intensity_numba(
            img,
            x_img,
            y_img,
            aperture_mask,
            half_ap,
            height,
            width,
            min_valid,
        )

    def refine_grid_parameters(
        self,
        integration_radius: float = 5.0,
        position_range: float = 4.0,
        angle_range_deg: float = 0.5,
        spacing_range: float = 0.3,
        method: str = 'L-BFGS-B',
        n_rounds: int = 1,
        individual_spot_refinement: bool = True,
        individual_refinement_range: float = 2.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Refine grid parameters for each image to maximize total integrated intensity.
        
        Two-stage refinement (default):
        1. Global refinement: Optimize grid center, angle, and spacing for the entire grid
        2. Individual refinement: Refine each spot position independently (if individual_spot_refinement=True)
        
        Args:
            integration_radius (float): Integration radius used by this function.
            position_range (float): Position range used by this function.
            angle_range_deg (float): Angle range deg used by this function.
            spacing_range (float): Spacing range used by this function.
            method (str): Method used by this function.
            n_rounds (int): Number of rounds used by this function.
            individual_spot_refinement (bool): Individual spot refinement used by this function.
            individual_refinement_range (float): Individual refinement range used by this function.
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            Dict[str, Any]: Refined grid parameters.
        """
        if self._square_masked_images is None:
            raise RuntimeError("Square masks not applied. Call apply_square_mask() first.")
        
        # Scale dimensional parameters
        scaled_integration_radius = self._scale(integration_radius)
        scaled_position_range = self._scale(position_range)
        scaled_spacing_range = self._scale(spacing_range)
        scaled_edge_offset = self._scale(self.EDGE_OFFSET)
        
        n_images = len(self._square_masked_images.image_idx)
        height, width = self._square_masked_images.shape[1:]
        
        peptides_type = self._get_peptides_type()
        n_rows = self.PTK_GRID_ROWS if peptides_type == 'PTK' else self.STK_GRID_ROWS
        n_cols = self.PTK_GRID_COLS if peptides_type == 'PTK' else self.STK_GRID_COLS
        
        angle_range_rad = np.radians(angle_range_deg)
        base_spacing = self._scale(self.SPOT_SPACING)
        
        if verbose:
            print(f"\nRefining grid parameters for {n_images} images...")
            print(f"  Grid size: {n_rows}×{n_cols}")
            print(f"  STAGE 1: Global grid refinement")
            print(f"  Position range: ±{position_range}px → ±{scaled_position_range:.2f}px (scaled)")
            print(f"  Angle range: ±{angle_range_deg}°")
            print(f"  Spacing range: {self.SPOT_SPACING} ± {spacing_range}px → {base_spacing:.2f} ± {scaled_spacing_range:.2f}px (scaled)")
            print(f"  Edge offset: {self.EDGE_OFFSET}px → {scaled_edge_offset:.2f}px (scaled)")
            print(f"  Integration radius: {integration_radius}px → {scaled_integration_radius:.2f}px (scaled)")
            print(f"  Optimization method: {method}")
            print(f"  Number of rounds: {n_rounds}")
            if individual_spot_refinement:
                print("  STAGE 2: Individual spot refinement")
                print(f"  Individual refinement range: ±{individual_refinement_range:.0f} pixel(s) (array indices)")
        
        ap_size = int(np.ceil(scaled_integration_radius)) * 2 + 1
        yy, xx = np.ogrid[:ap_size, :ap_size]
        ap_center = ap_size // 2
        aperture_mask = ((xx - ap_center)**2 + (yy - ap_center)**2) <= scaled_integration_radius**2
        half_ap = ap_size // 2
        aperture_offsets = np.argwhere(aperture_mask)
        aperture_dy = aperture_offsets[:, 0].astype(np.float32) - float(ap_center)
        aperture_dx = aperture_offsets[:, 1].astype(np.float32) - float(ap_center)

        refined_params_list = []
        original_intensities = []
        refined_intensities = []
        self._refined_grid_positions = []
        self._stage1_grid_positions = None
        self._stage2_grid_stats = None
        
        # Convert to float32 once (2-3x faster than per-image conversion)
        if self._square_masked_images.dtype != np.float32:
            square_masked_images_float = self._square_masked_images.astype(np.float32)
        else:
            square_masked_images_float = self._square_masked_images
        
        # Helper function for parallel processing
        def refine_single_grid(i):
            """Refine single grid.
            
            Args:
                i: I processed by this function.
            
            Returns:
                object: Refined single grid.
            """
            img = square_masked_images_float.isel(image_idx=i).values
            mask_params = self._square_mask_params[i]
            
            if mask_params is None:
                return i, None, np.nan, np.nan, None, False
            
            orig_center_x = mask_params['center_x']
            orig_center_y = mask_params['center_y']
            orig_angle = mask_params['angle_rad']
            mask_width = mask_params['right_x'] - mask_params['left_x']
            mask_height = mask_params['bottom_y'] - mask_params['top_y']
            
            orig_intensity = -self._compute_grid_intensity(
                img, orig_center_x, orig_center_y, orig_angle, base_spacing,
                n_rows, n_cols, mask_width, mask_height, aperture_dx, aperture_dy, height, width, scaled_edge_offset
            )
            
            current_center_x = orig_center_x
            current_center_y = orig_center_y
            current_angle = orig_angle
            current_spacing = base_spacing
            
            total_iterations = 0
            
            for round_idx in range(n_rounds):
                def objective(params):
                    """Return objective.
                    
                    Args:
                        params: Params processed by this function.
                    
                    Returns:
                        object: Objective.
                    """
                    dx, dy, dangle, dspacing = params
                    return self._compute_grid_intensity(
                        img, 
                        current_center_x + dx, 
                        current_center_y + dy, 
                        current_angle + dangle, 
                        current_spacing + dspacing,
                        n_rows, n_cols, mask_width, mask_height, aperture_dx, aperture_dy, height, width, scaled_edge_offset
                    )
                
                x0 = [0.0, 0.0, 0.0, 0.0]
                
                bounds = [
                    (-scaled_position_range, scaled_position_range),
                    (-scaled_position_range, scaled_position_range),
                    (-angle_range_rad, angle_range_rad),
                    (-scaled_spacing_range, scaled_spacing_range)
                ]
                
                if method == 'L-BFGS-B':
                    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 50, 'ftol': 1e-4})
                else:
                    # Powell or other methods: use relaxed tolerance for speed
                    result = minimize(objective, x0, method=method,
                                    options={'maxiter': 200}, tol=1e-4)
                
                dx = np.clip(result.x[0], -scaled_position_range, scaled_position_range)
                dy = np.clip(result.x[1], -scaled_position_range, scaled_position_range)
                dangle = np.clip(result.x[2], -angle_range_rad, angle_range_rad)
                dspacing = np.clip(result.x[3], -scaled_spacing_range, scaled_spacing_range)
                
                current_center_x += dx
                current_center_y += dy
                current_angle += dangle
                current_spacing += dspacing
                
                total_iterations += result.nit if hasattr(result, 'nit') else 0
            
            refined_center_x = current_center_x
            refined_center_y = current_center_y
            refined_angle = current_angle
            refined_spacing = current_spacing
            
            ref_intensity = -self._compute_grid_intensity(
                img, refined_center_x, refined_center_y, refined_angle, refined_spacing,
                n_rows, n_cols, mask_width, mask_height, aperture_dx, aperture_dy, height, width, scaled_edge_offset
            )
            
            refined_params = {
                'center_x': refined_center_x,
                'center_y': refined_center_y,
                'angle_rad': refined_angle,
                'angle_deg': np.degrees(refined_angle),
                'spacing': refined_spacing,
                'dx': refined_center_x - orig_center_x,
                'dy': refined_center_y - orig_center_y,
                'dangle_rad': refined_angle - orig_angle,
                'dangle_deg': np.degrees(refined_angle - orig_angle),
                'dspacing': refined_spacing - base_spacing,
                'original_intensity': orig_intensity,
                'refined_intensity': ref_intensity,
                'intensity_improvement': ref_intensity - orig_intensity,
                'improvement_percent': 100 * (ref_intensity - orig_intensity) / orig_intensity if orig_intensity > 0 else 0,
                'optimization_success': True,
                'n_iterations': total_iterations,
                'n_rounds': n_rounds
            }
            
            col_positions = -mask_width/2 + scaled_edge_offset + np.arange(n_cols) * refined_spacing
            row_positions = -mask_height/2 + scaled_edge_offset + np.arange(n_rows) * refined_spacing
            col_grid, row_grid = np.meshgrid(col_positions, row_positions)
            x_rel, y_rel = col_grid.flatten(), row_grid.flatten()
            
            cos_a, sin_a = np.cos(refined_angle), np.sin(refined_angle)
            x_rot = x_rel * cos_a - y_rel * sin_a
            y_rot = x_rel * sin_a + y_rel * cos_a
            
            x_img = x_rot + refined_center_x
            y_img = y_rot + refined_center_y
            
            spot_positions = np.stack([x_img.reshape(n_rows, n_cols), 
                                       y_img.reshape(n_rows, n_cols)], axis=-1)
            
            return i, refined_params, orig_intensity, ref_intensity, spot_positions, True
        
        # Parallel processing
        if verbose:
            print("  Processing images in parallel...")
        
        results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(refine_single_grid)(i) for i in range(n_images)
        )
        
        # Sort and unpack results
        results.sort(key=lambda x: x[0])
        
        n_valid_images = 0
        for i, refined_params, orig_int, ref_int, positions, is_valid in results:
            refined_params_list.append(refined_params)
            original_intensities.append(orig_int)
            refined_intensities.append(ref_int)
            self._refined_grid_positions.append(positions)
            if is_valid:
                n_valid_images += 1
        
        self._grid_refinement_params = {
            'integration_radius': scaled_integration_radius,
            'position_range': scaled_position_range,
            'angle_range_deg': angle_range_deg,
            'spacing_range': scaled_spacing_range,
            'method': method,
            'n_rounds': n_rounds,
            'base_spacing': base_spacing,
            'edge_offset': scaled_edge_offset,
            'n_rows': n_rows,
            'n_cols': n_cols
        }
        self._refined_grid_params_list = refined_params_list
        
        valid_improvements = [p['improvement_percent'] for p in refined_params_list if p is not None]
        valid_dx = [p['dx'] for p in refined_params_list if p is not None]
        valid_dy = [p['dy'] for p in refined_params_list if p is not None]
        valid_dangle = [p['dangle_deg'] for p in refined_params_list if p is not None]
        valid_dspacing = [p['dspacing'] for p in refined_params_list if p is not None]
        
        summary = {
            'n_images': n_images,
            'n_valid_images': n_valid_images,
            'n_rounds': n_rounds,
            'mean_improvement_percent': np.mean(valid_improvements) if valid_improvements else 0,
            'max_improvement_percent': np.max(valid_improvements) if valid_improvements else 0,
            'mean_dx': np.mean(valid_dx) if valid_dx else 0,
            'std_dx': np.std(valid_dx) if valid_dx else 0,
            'mean_dy': np.mean(valid_dy) if valid_dy else 0,
            'std_dy': np.std(valid_dy) if valid_dy else 0,
            'mean_dangle_deg': np.mean(valid_dangle) if valid_dangle else 0,
            'std_dangle_deg': np.std(valid_dangle) if valid_dangle else 0,
            'mean_dspacing': np.mean(valid_dspacing) if valid_dspacing else 0,
            'std_dspacing': np.std(valid_dspacing) if valid_dspacing else 0,
        }
        
        if verbose:
            print(f"\nStage 1: Global grid refinement complete ({n_rounds} rounds)!")
            print(f"  Valid images: {n_valid_images}/{n_images}")
            print(f"  Mean intensity improvement: {summary['mean_improvement_percent']:.2f}%")
            print(f"  Max intensity improvement: {summary['max_improvement_percent']:.2f}%")
            print("  Position offsets (mean ± std):")
            print(f"    dx: {summary['mean_dx']:.3f} ± {summary['std_dx']:.3f} px")
            print(f"    dy: {summary['mean_dy']:.3f} ± {summary['std_dy']:.3f} px")
            print(f"  Angle offset: {summary['mean_dangle_deg']:.4f} ± {summary['std_dangle_deg']:.4f}°")
            print(f"  Spacing offset: {summary['mean_dspacing']:.4f} ± {summary['std_dspacing']:.4f} px")
        
        # Stage 2: Individual spot refinement (optional)
        if individual_spot_refinement:
            if verbose:
                print(f"\nStage 2: Refining individual spot positions...")
            
            # Store Stage 1 positions before Stage 2 modifies them
            self._stage1_grid_positions = [pos.copy() if pos is not None else None 
                                           for pos in self._refined_grid_positions]
            
            individual_summary = self._refine_individual_spot_positions(
                square_masked_images_float=square_masked_images_float,
                integration_radius=scaled_integration_radius,
                refinement_range=individual_refinement_range,
                aperture_mask=aperture_mask,
                half_ap=half_ap,
                verbose=verbose
            )
            
            summary['individual_refinement'] = individual_summary
            self._stage2_grid_stats = individual_summary.get('per_image')
            # Also store in _grid_refinement_params for visualization access
            self._grid_refinement_params['individual_refinement'] = individual_summary
        else:
            # No Stage 2, so Stage 1 = final positions
            self._stage1_grid_positions = None
            self._stage2_grid_stats = None
        
        return {'params': refined_params_list, 'summary': summary}

    def _refine_individual_spot_positions(
        self,
        square_masked_images_float,
        integration_radius: float,
        refinement_range: float,
        aperture_mask,
        half_ap: int,
        early_stopping: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Stage 2 refinement: Refine each spot position individually by optimizing local intensity.
        
        Uses numba-compiled grid search to check integer positions within refinement_range.
        For each spot in the grid, search for the position within refinement_range that
        maximizes the integrated intensity of that spot independently.
        
        Args:
            square_masked_images_float: Square masked images float processed by this function.
            integration_radius (float): Integration radius used by this function.
            refinement_range (float): Refinement range used by this function.
            aperture_mask: Aperture mask processed by this function.
            half_ap (int): Half ap used by this function.
            early_stopping (bool): Early stopping used by this function.
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            Dict[str, Any]: Refined individual spot positions.
        """
        
        n_images = len(self._refined_grid_positions)
        height, width = square_masked_images_float.shape[1:]
        
        total_displacements = []
        n_spots_improved = 0
        n_spots_skipped = 0
        total_spots = 0
        per_image_stats = [None] * n_images
        
        def refine_single_image_spots(i):
            """Refine all spots in a single image.
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Refined single image spots.
            """
            img = square_masked_images_float.isel(image_idx=i).values
            initial_positions = self._refined_grid_positions[i]
            
            if initial_positions is None:
                return i, None, 0, 0, 0, [], None
            
            n_rows, n_cols = initial_positions.shape[:2]
            refined_positions = initial_positions.copy()
            
            image_displacements = []
            image_improved = 0
            image_skipped = 0
            image_initial_total_intensity = 0.0
            image_final_total_intensity = 0.0
            
            for row in range(n_rows):
                for col in range(n_cols):
                    x0, y0 = initial_positions[row, col]
                    
                    # Early stopping: Check if any 8-connected neighbor has better intensity
                    if early_stopping:
                        has_better_neighbor, _ = self._check_8_neighbors_numba(
                            img, x0, y0, aperture_mask, half_ap, height, width
                        )
                        
                        if not has_better_neighbor:
                            # Current position is already a local maximum, skip optimization
                            refined_positions[row, col] = [x0, y0]
                            x0_int, y0_int = int(round(x0)), int(round(y0))
                            if (y0_int - half_ap >= 0 and y0_int + half_ap + 1 <= height and
                                x0_int - half_ap >= 0 and x0_int + half_ap + 1 <= width):
                                aperture = img[y0_int - half_ap:y0_int + half_ap + 1,
                                             x0_int - half_ap:x0_int + half_ap + 1]
                                initial_intensity = np.sum(aperture[aperture_mask])
                            else:
                                initial_intensity = 0.0
                            image_initial_total_intensity += initial_intensity
                            image_final_total_intensity += initial_intensity
                            image_displacements.append(0.0)
                            image_skipped += 1
                            continue
                    
                    # Use numba-compiled grid search
                    x_refined, y_refined, best_intensity = self._grid_search_spot_position_numba(
                        img, x0, y0, aperture_mask, half_ap, refinement_range, height, width
                    )
                    
                    # Compute initial intensity for comparison
                    x0_int, y0_int = int(round(x0)), int(round(y0))
                    if (y0_int - half_ap >= 0 and y0_int + half_ap + 1 <= height and
                        x0_int - half_ap >= 0 and x0_int + half_ap + 1 <= width):
                        aperture = img[y0_int - half_ap:y0_int + half_ap + 1,
                                     x0_int - half_ap:x0_int + half_ap + 1]
                        initial_intensity = np.sum(aperture[aperture_mask])
                    else:
                        initial_intensity = 0.0
                    
                    refined_intensity = best_intensity
                    image_initial_total_intensity += initial_intensity
                    image_final_total_intensity += refined_intensity
                    
                    # Update position
                    refined_positions[row, col] = [x_refined, y_refined]
                    
                    # Track displacement
                    displacement = np.sqrt((x_refined - x0)**2 + (y_refined - y0)**2)
                    image_displacements.append(displacement)
                    
                    # Track improvement
                    if refined_intensity > initial_intensity:
                        image_improved += 1

            image_total_spots = n_rows * n_cols
            image_mean_displacement = (
                float(np.mean(image_displacements)) if image_displacements else 0.0
            )
            image_max_displacement = (
                float(np.max(image_displacements)) if image_displacements else 0.0
            )
            image_improvement_percent = (
                100.0 * (image_final_total_intensity - image_initial_total_intensity)
                / image_initial_total_intensity
                if image_initial_total_intensity > 0
                else 0.0
            )
            image_skip_rate_percent = (
                100.0 * image_skipped / image_total_spots if image_total_spots > 0 else 0.0
            )
            image_improvement_rate_percent = (
                100.0 * image_improved / image_total_spots if image_total_spots > 0 else 0.0
            )

            image_stats = {
                'image_idx': i,
                'n_spots_refined': image_total_spots,
                'n_spots_improved': image_improved,
                'n_spots_skipped_early_stopping': image_skipped,
                'improvement_rate_percent': image_improvement_rate_percent,
                'skip_rate_percent': image_skip_rate_percent,
                'mean_displacement': image_mean_displacement,
                'max_displacement': image_max_displacement,
                'initial_total_intensity': float(image_initial_total_intensity),
                'final_total_intensity': float(image_final_total_intensity),
                'total_intensity_improvement': float(
                    image_final_total_intensity - image_initial_total_intensity
                ),
                'total_intensity_improvement_percent': float(image_improvement_percent),
            }

            return (
                i,
                refined_positions,
                image_improved,
                image_skipped,
                image_total_spots,
                image_displacements,
                image_stats,
            )
        
        # Parallel processing
        if verbose:
            early_stop_str = " with early stopping" if early_stopping else ""
            grid_size = int(refinement_range * 2 + 1)
            print(f"  Processing individual spot refinements in parallel ({grid_size}×{grid_size} grid search{early_stop_str})...")
        
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(refine_single_image_spots)(i) for i in range(n_images)
        )
        
        # Update positions and collect statistics
        results.sort(key=lambda x: x[0])
        
        for i, positions, improved, skipped, total, displacements, image_stats in results:
            if positions is not None:
                self._refined_grid_positions[i] = positions
                n_spots_improved += improved
                n_spots_skipped += skipped
                total_spots += total
                total_displacements.extend(displacements)
                per_image_stats[i] = image_stats

        # Summary statistics
        mean_displacement = np.mean(total_displacements) if total_displacements else 0
        max_displacement = np.max(total_displacements) if total_displacements else 0
        improvement_rate = 100 * n_spots_improved / total_spots if total_spots > 0 else 0
        skip_rate = 100 * n_spots_skipped / total_spots if total_spots > 0 else 0
        
        summary = {
            'n_spots_refined': total_spots,
            'n_spots_improved': n_spots_improved,
            'n_spots_skipped_early_stopping': n_spots_skipped,
            'improvement_rate_percent': improvement_rate,
            'skip_rate_percent': skip_rate,
            'mean_displacement': mean_displacement,
            'max_displacement': max_displacement,
            'per_image': per_image_stats,
        }
        
        if verbose:
            print("\nStage 2: Individual spot refinement complete!")
            print(f"  Total spots refined: {total_spots}")
            print(f"  Spots improved: {n_spots_improved} ({improvement_rate:.1f}%)")
            if early_stopping:
                print(f"  Spots skipped (early stopping): {n_spots_skipped} ({skip_rate:.1f}%)")
            print(f"  Mean displacement: {mean_displacement:.3f} px")
            print(f"  Max displacement: {max_displacement:.3f} px")
        
        return summary

    def get_refined_grid_positions(self, image_idx: int = 0):
        """Get refined spot grid positions for a specific image after grid refinement.
        
        Args:
            image_idx (int): Zero-based index selecting the image.
        
        Returns:
            object: Requested refined grid positions.
        """
        if self._refined_grid_positions is None:
            return None
        if image_idx >= len(self._refined_grid_positions):
            raise ValueError(f"image_idx {image_idx} out of range")
        return self._refined_grid_positions[image_idx]

    def compute_refined_spot_intensities(self, integration_radius: float = 5.0, verbose: bool = True):
        """Compute integrated, maximum, median intensities and saturation fraction using refined grid parameters.
        
        Saturation fraction is computed as the fraction of pixels at the maximum intensity
        within each spot aperture - a high value indicates potential signal clipping.
        
        Args:
            integration_radius (float): Integration radius used by this function.
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            object: Computed refined spot intensities.
        """
        if self._refined_grid_positions is None:
            raise RuntimeError("Grid not refined. Call refine_grid_parameters() first.")
        
        # Scale dimensional parameters
        scaled_integration_radius = self._scale(integration_radius)
        
        n_images = len(self._square_masked_images.image_idx)
        height, width = self._square_masked_images.shape[1:]
        
        # Convert to float32 once (2-3x faster than per-image conversion)
        if self._square_masked_images.dtype != np.float32:
            square_masked_images_float = self._square_masked_images.astype(np.float32)
        else:
            square_masked_images_float = self._square_masked_images
        
        peptides_type = self._get_peptides_type()
        n_rows = self.PTK_GRID_ROWS if peptides_type == 'PTK' else self.STK_GRID_ROWS
        n_cols = self.PTK_GRID_COLS if peptides_type == 'PTK' else self.STK_GRID_COLS
        
        if verbose:
            print(f"\nComputing refined spot intensities for {n_images} images...")
            print(f"  Integration radius: {integration_radius}px → {scaled_integration_radius:.2f}px (scaled)")
        
        ap_size = int(np.ceil(scaled_integration_radius)) * 2 + 1
        yy, xx = np.ogrid[:ap_size, :ap_size]
        ap_center = ap_size // 2
        aperture_mask = ((xx - ap_center)**2 + (yy - ap_center)**2) <= scaled_integration_radius**2
        half_ap = ap_size // 2
        
        # Helper function for parallel processing
        def compute_refined_intensities_single(i):
            """Compute refined intensities single.
            
            Args:
                i: I processed by this function.
            
            Returns:
                tuple: Computed refined intensities single.
            """
            img = square_masked_images_float.isel(image_idx=i).values
            spot_positions = self._refined_grid_positions[i]
            
            if spot_positions is None:
                return (i, np.full((n_rows, n_cols), np.nan), np.full((n_rows, n_cols), np.nan),
                        np.full((n_rows, n_cols), np.nan), np.full((n_rows, n_cols), np.nan))
            
            # Flatten spot positions for numba function
            x_img = spot_positions[:, :, 0].flatten()
            y_img = spot_positions[:, :, 1].flatten()
            
            # Use numba-compiled vectorized computation (5-10x faster than nested loops)
            intensities, max_intensities, median_intensities, saturation_fractions = \
                self._compute_spot_metrics_numba(img, x_img, y_img, aperture_mask, 
                                                 half_ap, n_rows, n_cols)
            
            return i, intensities, max_intensities, median_intensities, saturation_fractions
        
        # Parallel processing
        if verbose:
            print("  Processing images in parallel...")
        
        results = Parallel(n_jobs=-1, backend='threading', verbose=5 if verbose else 0)(
            delayed(compute_refined_intensities_single)(i) for i in range(n_images)
        )
        
        # Sort and unpack results
        results.sort(key=lambda x: x[0])
        
        intensities_array = np.zeros((n_images, n_rows, n_cols), dtype=np.float32)
        max_intensities_array = np.zeros((n_images, n_rows, n_cols), dtype=np.float32)
        median_intensities_array = np.zeros((n_images, n_rows, n_cols), dtype=np.float32)
        saturation_fractions_array = np.zeros((n_images, n_rows, n_cols), dtype=np.float32)
        
        for i, intensities, max_ints, median_ints, sat_fracs in results:
            intensities_array[i] = intensities
            max_intensities_array[i] = max_ints
            median_intensities_array[i] = median_ints
            saturation_fractions_array[i] = sat_fracs
        
        coords_dict = {'spot_row': np.arange(n_rows), 'spot_col': np.arange(n_cols)}
        for coord_name, coord_data in self._square_masked_images.coords.items():
            if coord_name in ['y', 'x']:
                continue
            if 'image_idx' in coord_data.dims:
                coords_dict[coord_name] = coord_data
            elif coord_name == 'image_idx':
                coords_dict['image_idx'] = coord_data
        
        self._refined_spot_intensities = xr.DataArray(
            data=intensities_array,
            dims=['image_idx', 'spot_row', 'spot_col'],
            coords=coords_dict,
            attrs={
                'processing': 'refined_spot_intensities',
                'integration_radius': scaled_integration_radius,
                'n_rows': n_rows, 'n_cols': n_cols,
                'peptides_type': peptides_type
            }
        )
        
        self._refined_spot_max_intensities = xr.DataArray(
            data=max_intensities_array,
            dims=['image_idx', 'spot_row', 'spot_col'],
            coords=coords_dict,
            attrs={
                'processing': 'refined_spot_max_intensities',
                'integration_radius': scaled_integration_radius,
                'n_rows': n_rows, 'n_cols': n_cols,
                'peptides_type': peptides_type
            }
        )
        
        self._refined_spot_median_intensities = xr.DataArray(
            data=median_intensities_array,
            dims=['image_idx', 'spot_row', 'spot_col'],
            coords=coords_dict,
            attrs={
                'processing': 'refined_spot_median_intensities',
                'integration_radius': scaled_integration_radius,
                'n_rows': n_rows, 'n_cols': n_cols,
                'peptides_type': peptides_type
            }
        )
        
        self._refined_spot_saturation_fractions = xr.DataArray(
            data=saturation_fractions_array,
            dims=['image_idx', 'spot_row', 'spot_col'],
            coords=coords_dict,
            attrs={
                'processing': 'refined_spot_saturation_fractions',
                'integration_radius': scaled_integration_radius,
                'description': 'Fraction of pixels at spot maximum intensity (indicates potential clipping)',
                'n_rows': n_rows, 'n_cols': n_cols,
                'peptides_type': peptides_type
            }
        )
        
        if verbose:
            valid_mask = ~np.isnan(intensities_array)
            valid_vals = intensities_array[valid_mask]
            valid_max_vals = max_intensities_array[valid_mask]
            valid_median_vals = median_intensities_array[valid_mask]
            valid_sat_vals = saturation_fractions_array[valid_mask]
            print("Refined spot intensity computation complete!")
            if len(valid_vals) > 0:
                print(f"  Integrated intensity - Mean: {np.mean(valid_vals):.2f}, Std: {np.std(valid_vals):.2f}")
                print(f"  Max intensity - Mean: {np.mean(valid_max_vals):.2f}, Std: {np.std(valid_max_vals):.2f}")
                print(f"  Median intensity - Mean: {np.mean(valid_median_vals):.2f}, Std: {np.std(valid_median_vals):.2f}")
                print(f"  Saturation fraction - Mean: {np.mean(valid_sat_vals):.4f}, Max: {np.max(valid_sat_vals):.4f}")
                n_potentially_saturated = np.sum(valid_sat_vals > 0.1)
                print(f"  Spots with >10% pixels at max: {n_potentially_saturated}/{len(valid_vals)}")
        
        return self._refined_spot_intensities
    
    @property
    def spot_max_intensities(self):
        """Get the spot max intensities DataArray.
        
        Args:
            None.
        
        Returns:
            object: Stored spot max intensities.
        """
        return self._spot_max_intensities
    
    @property
    def spot_median_intensities(self):
        """Get the spot median intensities DataArray.
        
        Args:
            None.
        
        Returns:
            object: Stored spot median intensities.
        """
        return self._spot_median_intensities
    
    @property
    def spot_saturation_fractions(self):
        """Get the spot saturation fractions DataArray.
        
        Args:
            None.
        
        Returns:
            object: Stored spot saturation fractions.
        """
        return self._spot_saturation_fractions
    
    @property
    def refined_spot_max_intensities(self):
        """Get the refined spot max intensities DataArray.
        
        Args:
            None.
        
        Returns:
            object: Stored refined spot max intensities.
        """
        return getattr(self, '_refined_spot_max_intensities', None)
    
    @property
    def refined_spot_median_intensities(self):
        """Get the refined spot median intensities DataArray.
        
        Args:
            None.
        
        Returns:
            object: Stored refined spot median intensities.
        """
        return getattr(self, '_refined_spot_median_intensities', None)
    
    @property
    def refined_spot_saturation_fractions(self):
        """Get the refined spot saturation fractions DataArray.
        
        Args:
            None.
        
        Returns:
            object: Stored refined spot saturation fractions.
        """
        return getattr(self, '_refined_spot_saturation_fractions', None)
    
    @property
    def refined_spot_intensities(self):
        """Get the refined spot intensities DataArray.
        
        Args:
            None.
        
        Returns:
            object: Stored refined spot intensities.
        """
        return getattr(self, '_refined_spot_intensities', None)

    @property
    def refined_grid_params(self):
        """Get the refined grid parameters list.
        
        Args:
            None.
        
        Returns:
            object: Stored refined grid params.
        """
        return getattr(self, '_refined_grid_params_list', None)

    def visualize_grid_refinement(
        self,
        image_idx: int = 0,
        figsize: Tuple[int, int] = (16, 7),
        spot_color: str = 'yellow',
        stage1_color: str = 'cyan',
        stage2_color: str = 'lime',
        spot_alpha: float = 0.8,
        save_images: bool = False
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize the spot grid before and after refinement.
        
        Shows:
        - Left panel: Original grid (before refinement)
        - Right panel: Refined grid
          * If two-stage refinement was used, shows both Stage 1 (global) and Stage 2 (individual)
        
        Args:
            image_idx (int): Zero-based index selecting the image.
            figsize (Tuple[int, int]): Figsize processed by this function.
            spot_color (str): Spot color used by this function.
            stage1_color (str): Stage1 color used by this function.
            stage2_color (str): Stage2 color used by this function.
            spot_alpha (float): Spot alpha used by this function.
            save_images (bool): Save images used by this function.
        
        Returns:
            Tuple[plt.Figure, np.ndarray]: Visualize grid refinement.
        """
        if self._square_masked_images is None:
            raise RuntimeError("Square masks not applied. Call apply_square_mask() first.")
        if self._refined_grid_positions is None:
            raise RuntimeError("Grid not refined. Call refine_grid_parameters() first.")
        
        if image_idx >= len(self._square_masked_images.image_idx):
            raise ValueError(f"image_idx {image_idx} out of range")
        
        # Get visualization directory if saving
        viz_dir = None
        if save_images:
            viz_dir = self._get_visualization_dir('visualize_grid_refinement')
        
        square_masked = self._square_masked_images.isel(image_idx=image_idx).values
        original_positions = self._spot_grid_positions[image_idx] if self._spot_grid_positions else None
        refined_positions = self._refined_grid_positions[image_idx]
        refined_params = self._refined_grid_params_list[image_idx] if self._refined_grid_params_list else None
        stage2_stats = None
        if self._stage2_grid_stats is not None and image_idx < len(self._stage2_grid_stats):
            stage2_stats = self._stage2_grid_stats[image_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        sq_disp = self._symmetric_log_transform(square_masked)
        vmax = np.max(np.abs(sq_disp[sq_disp != 0])) if np.any(sq_disp != 0) else 1
        
        integration_radius = self._grid_refinement_params.get('integration_radius', 5.0) if hasattr(self, '_grid_refinement_params') else 5.0
        
        ax = axes[0]
        ax.imshow(sq_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        if original_positions is not None:
            n_rows, n_cols = original_positions.shape[:2]
            for row in range(n_rows):
                for col in range(n_cols):
                    x, y = original_positions[row, col]
                    ax.add_patch(plt.Circle((x, y), integration_radius, color=spot_color,
                                           fill=False, linewidth=1.0, alpha=spot_alpha))
        
        orig_intensity = refined_params['original_intensity'] if refined_params else 'N/A'
        ax.set_title(f'Original Grid (idx={image_idx})\nTotal intensity: {orig_intensity:.0f}' if isinstance(orig_intensity, float) else f'Original Grid (idx={image_idx})', fontsize=14)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        ax = axes[1]
        ax.imshow(sq_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Check if two-stage refinement was used
        has_two_stages = hasattr(self, '_stage1_grid_positions') and self._stage1_grid_positions is not None
        stage1_positions = self._stage1_grid_positions[image_idx] if has_two_stages else None
        
        # Draw Stage 1 positions (global refinement) if available
        if has_two_stages and stage1_positions is not None:
            n_rows, n_cols = stage1_positions.shape[:2]
            for row in range(n_rows):
                for col in range(n_cols):
                    x, y = stage1_positions[row, col]
                    ax.add_patch(plt.Circle((x, y), integration_radius, color=stage1_color,
                                           fill=False, linewidth=1.5, alpha=spot_alpha, linestyle='--'))
        
        # Draw Stage 2 positions (individual refinement) or final positions
        if refined_positions is not None:
            n_rows, n_cols = refined_positions.shape[:2]
            # Use different color if two-stage refinement was used
            final_color = stage2_color if has_two_stages else spot_color
            for row in range(n_rows):
                for col in range(n_cols):
                    x, y = refined_positions[row, col]
                    ax.add_patch(plt.Circle((x, y), integration_radius, color=final_color,
                                           fill=False, linewidth=1.5, alpha=spot_alpha))
        
        if has_two_stages and stage2_stats is not None:
            ref_intensity = stage2_stats.get('final_total_intensity', np.nan)
            improvement = stage2_stats.get('total_intensity_improvement_percent', np.nan)
            title = f'Refined Grid\nStage 2 total intensity: {ref_intensity:.0f} (+{improvement:.2f}%)'
            title += '\n(Stage 1: cyan dash, Stage 2: green solid)'
            ax.set_title(title, fontsize=14)
        elif refined_params:
            ref_intensity = refined_params['refined_intensity']
            improvement = refined_params['improvement_percent']
            title = f'Refined Grid\nTotal intensity: {ref_intensity:.0f} (+{improvement:.2f}%)'
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Refined Grid', fontsize=14)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Update legend to show different stages
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor=spot_color, markersize=10, markeredgewidth=1.5,
                   alpha=spot_alpha, label='Original positions')
        ]
        
        if has_two_stages:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor=stage1_color, markersize=10, markeredgewidth=1.5,
                       alpha=spot_alpha, linestyle='--', label='Stage 1 (global)')
            )
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor=stage2_color, markersize=10, markeredgewidth=1.5,
                       alpha=spot_alpha, label='Stage 2 (individual)')
            )
        else:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor=spot_color, markersize=10, markeredgewidth=1.5,
                       alpha=spot_alpha, label='Refined positions')
            )
        
        axes[0].legend(handles=legend_elements[:1], loc='upper right', fontsize=9)
        axes[1].legend(handles=legend_elements[1:], loc='upper right', fontsize=9)
        
        if refined_params:
            info = (f"Stage 1 - Offsets: dx={refined_params['dx']:.2f}px, dy={refined_params['dy']:.2f}px, "
                   f"Δangle={refined_params['dangle_deg']:.3f}°, Δspacing={refined_params['dspacing']:.3f}px")
            if has_two_stages and stage2_stats is not None:
                info += (
                    f"\nStage 2 - Improved spots: {stage2_stats.get('improvement_rate_percent', 0):.1f}%, "
                    f"Mean displacement: {stage2_stats.get('mean_displacement', 0):.3f}px, "
                    f"Max displacement: {stage2_stats.get('max_displacement', 0):.3f}px"
                )
            fig.text(0.02, 0.02, info, fontsize=10, transform=fig.transFigure, color='darkgreen')
        
        plt.tight_layout()
        
        # Save image if requested
        if save_images and viz_dir is not None:
            
            self._save_visualization(fig, viz_dir, image_idx)
        
        return axes

    def _draw_rotated_rectangle(self, ax, center_x, center_y, width, height, angle, color='black', linewidth=2):
        """Draw rotated rectangle.
        
        Args:
            ax: Ax processed by this function.
            center_x: Center x processed by this function.
            center_y: Center y processed by this function.
            width: Width processed by this function.
            height: Height processed by this function.
            angle: Angle processed by this function.
            color: Color processed by this function.
            linewidth: Linewidth processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        half_w, half_h = width / 2, height / 2
        corners = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h], [-half_w, -half_h]])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_corners = corners @ rotation.T
        rotated_corners[:, 0] += center_x
        rotated_corners[:, 1] += center_y
        ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color, linewidth=linewidth)

    def visualize_square_mask(self, image_idx=0, figsize=(16, 7), show_reference_spots=True,
                              show_mask_outline=True, show_spot_grid=False, 
                              spot_grid_color='cyan', spot_grid_alpha=0.7, save_images=False):
        """Visualize square mask with optional spot grid overlay.
        
        Args:
            image_idx: Zero-based index selecting the image.
            figsize: Figsize processed by this function.
            show_reference_spots: Show reference spots processed by this function.
            show_mask_outline: Show mask outline processed by this function.
            show_spot_grid: Show spot grid processed by this function.
            spot_grid_color: Spot grid color processed by this function.
            spot_grid_alpha: Spot grid alpha processed by this function.
            save_images: Save images processed by this function.
        
        Returns:
            tuple: Visualize square mask.
        """
        if self._square_masked_images is None:
            raise RuntimeError("Square masks not applied. Call apply_square_mask() first.")
        
        # Get visualization directory if saving
        viz_dir = None
        if save_images:
            viz_dir = self._get_visualization_dir('visualize_square_mask')
        
        bg_subtracted = self._background_subtracted_images.isel(image_idx=image_idx).values
        square_masked = self._square_masked_images.isel(image_idx=image_idx).values
        mask_params = self._square_mask_params[image_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        ax = axes[0]
        bg_disp = self._symmetric_log_transform(bg_subtracted)
        vmax = np.max(np.abs(bg_disp[bg_disp != 0])) if np.any(bg_disp != 0) else 1
        ax.imshow(bg_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Background Subtracted (idx={image_idx})', fontsize=14)
        
        if show_reference_spots and self._refined_reference_spots is not None:
            t_shape = self._refined_reference_spots['t_shape'][image_idx]
            j_shape = self._refined_reference_spots['j_shape'][image_idx]
            for shape in [t_shape, j_shape]:
                if shape is not None:
                    for x, y in shape:
                        ax.add_patch(plt.Circle((x, y), 5, color='yellow', fill=False, linewidth=2))
        
        if show_mask_outline and mask_params is not None:
            self._draw_rotated_rectangle(ax, mask_params['center_x'], mask_params['center_y'],
                                        mask_params['right_x'] - mask_params['left_x'],
                                        mask_params['bottom_y'] - mask_params['top_y'],
                                        mask_params['angle_rad'], color='lime', linewidth=2)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        ax = axes[1]
        sq_disp = self._symmetric_log_transform(square_masked)
        vmax = np.max(np.abs(sq_disp[sq_disp != 0])) if np.any(sq_disp != 0) else 1
        ax.imshow(sq_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        title = f'Square Masked ({mask_params["peptides_type"]})\nAngle: {mask_params["angle_deg"]:.2f}°' if mask_params else 'Square Masked'
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        if show_spot_grid and self._spot_grid_positions is not None:
            spot_positions = self._spot_grid_positions[image_idx]
            if spot_positions is not None:
                integration_radius = self._spot_intensity_params.get('integration_radius', 5.0)
                n_rows, n_cols = spot_positions.shape[:2]
                for row in range(n_rows):
                    for col in range(n_cols):
                        x, y = spot_positions[row, col]
                        ax.add_patch(plt.Circle((x, y), integration_radius, color=spot_grid_color,
                                               fill=False, linewidth=1.0, alpha=spot_grid_alpha))
        
        legend_elements = []
        if show_reference_spots:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                         markeredgecolor='yellow', markersize=10, markeredgewidth=2, label='Reference spots'))
        if show_mask_outline and mask_params:
            legend_elements.append(Line2D([0], [0], color='lime', linewidth=2, label='Mask boundary'))
        if show_spot_grid and self._spot_grid_positions is not None and self._spot_grid_positions[image_idx] is not None:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                                         markeredgecolor=spot_grid_color, markersize=8, alpha=spot_grid_alpha, label='Spot grid'))
        if legend_elements:
            axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        if mask_params:
            info = f"Type: {mask_params['peptides_type']} | Angle: {mask_params['angle_deg']:.2f}°"
            if self._spot_intensity_params:
                info += f" | Grid: {self._spot_intensity_params['n_rows']}×{self._spot_intensity_params['n_cols']}"
            fig.text(0.02, 0.02, info, fontsize=10, transform=fig.transFigure, color='darkgreen')
        
        plt.tight_layout()
        
        # Save image if requested
        if save_images and viz_dir is not None:
            self._save_visualization(fig, viz_dir, image_idx)
        
        return fig, axes
        
    def visualize_circular_mask_detection(self, image_idx=0, figsize=(20, 16), save_images=False):
        """Visualize the circular mask detection results.
        
        Shows original image with detected center and circular mask overlay.
        For Canny method, also shows intermediate processing steps (gamma correction,
        median filtering, edge detection).
        
        Args:
            image_idx: Zero-based index selecting the image.
            figsize: Figsize processed by this function.
            save_images: Save images processed by this function.
        
        Returns:
            object: Visualize circular mask detection.
        """
        if self._masked_images is None:
            raise RuntimeError("Circular mask not applied. Call apply_circular_mask() first.")
        
        # Get visualization directory if saving
        viz_dir = None
        if save_images:
            viz_dir = self._get_visualization_dir('visualize_circular_mask_detection')
        
        if image_idx >= len(self.original_images.image_idx):
            raise ValueError(f"image_idx {image_idx} out of range")
        
        # Get images
        original = self.original_images.isel(image_idx=image_idx).values
        masked = self._masked_images.isel(image_idx=image_idx).values
        
        # Log transform for display
        original_log = self._log_transform_for_display(original)
        masked_log = self._log_transform_for_display(masked)
        
        # Get detection metadata
        metadata = self._detection_metadata[image_idx]
        center_x, center_y = self._centers[image_idx]
        radius = self.radius
        
        # Visualize using Canny detection
        return self._visualize_canny_detection(original, masked, original_log, masked_log, 
                                               metadata, center_x, center_y, radius, 
                                               image_idx, figsize, save_images, viz_dir)
    
    def _visualize_canny_detection(self, original, masked, original_log, masked_log, 
                                   metadata, center_x, center_y, radius, 
                                   image_idx, figsize, save_images, viz_dir):
        """Visualize 2D band-based Canny edge detection with intermediate steps.
        
        Args:
            original: Original processed by this function.
            masked: Masked processed by this function.
            original_log: Original log processed by this function.
            masked_log: Masked log processed by this function.
            metadata: Metadata processed by this function.
            center_x: Center x processed by this function.
            center_y: Center y processed by this function.
            radius: Radius processed by this function.
            image_idx: Zero-based index selecting the image.
            figsize: Figsize processed by this function.
            save_images: Save images processed by this function.
            viz_dir: Directory containing or receiving the viz.
        
        Returns:
            tuple: Visualize canny detection.
        """
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 6, hspace=0.45, wspace=0.35)

        ax = fig.add_subplot(gs[0, 0:2])
        ax.imshow(original_log, cmap='gray')
        circle = plt.Circle((center_x, center_y), radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
        band_specs = [
            ("top", metadata["top_grad_meta"], "cyan", "Top band"),
            ("bottom", metadata["bottom_grad_meta"], "magenta", "Bottom band"),
            ("left", metadata["left_grad_meta"], "yellow", "Left band"),
            ("right", metadata["right_grad_meta"], "orange", "Right band"),
        ]
        for side, grad_meta, color, label in band_specs:
            start, end = grad_meta['band_bounds']
            if side in {"top", "bottom"}:
                ax.axhspan(start, end - 1, color=color, alpha=0.14)
            else:
                ax.axvspan(start, end - 1, color=color, alpha=0.10)
            edge_x = grad_meta.get("edge_coords_image_x", np.empty(0, dtype=np.int32))
            edge_y = grad_meta.get("edge_coords_image_y", np.empty(0, dtype=np.int32))
            if len(edge_x) > 0:
                ax.scatter(edge_x, edge_y, s=6, c=color, alpha=0.85, linewidths=0, label=f"{label} edges")
        ax.set_title(f'Original Image With 4 Side Bands And Detected 2D Canny Edges (idx={image_idx})', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.tick_params(labelsize=10)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, loc='upper right')

        ax = fig.add_subplot(gs[0, 2:4])
        ax.imshow(masked_log, cmap='gray')
        ax.set_title(f'Masked Image\n(r={radius:.1f}px, log scale)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        ax.tick_params(labelsize=10)

        ax_info = fig.add_subplot(gs[0, 4:6])
        ax_info.axis('off')
        image_data = self.original_images.isel(image_idx=image_idx)
        info_lines = [
            "IMAGE INFORMATION",
            "=" * 35,
            "Detection: 2D Canny on finite bands",
            f"Band half-width: {self.edge_band_half_width}px",
            f"Gamma: {self.gamma:.2f}",
            f"Median Kernel: {self.median_kernel_size}",
            f"Canny Thresholds: [{self.canny_low_threshold:.2f}, {self.canny_high_threshold:.2f}]",
            "",
            "Detected Center:",
            f"  X: {center_x:.1f} px",
            f"  Y: {center_y:.1f} px",
            f"  Radius: {radius:.1f} px",
            "",
            "Coordinates:",
            "-" * 35,
        ]
        for coord_name, coord_val in image_data.coords.items():
            if coord_name in ['y', 'x', 'file_path']:
                continue
            val = coord_val.values
            if hasattr(val, 'item'):
                val = val.item()
            if isinstance(val, (float, np.floating)):
                info_lines.append(f"  {coord_name}: {val:.4f}")
            else:
                info_lines.append(f"  {coord_name}: {val}")
        ax_info.text(
            0.05,
            0.95,
            "\n".join(info_lines),
            transform=ax_info.transAxes,
            fontsize=9,
            fontfamily='monospace',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        )
        ax_info.set_title('Info', fontsize=12, fontweight='bold')

        self._plot_canny_band_pipeline(
            fig, gs, 1, 'Top Band', metadata['band_centers']['top'],
            metadata['top_grad_meta'], metadata['top_left_edge'], metadata['top_right_edge'],
            metadata['top_center_x'], orientation='horizontal'
        )
        self._plot_canny_band_pipeline(
            fig, gs, 2, 'Bottom Band', metadata['band_centers']['bottom'],
            metadata['bottom_grad_meta'], metadata['bottom_left_edge'], metadata['bottom_right_edge'],
            metadata['bottom_center_x'], orientation='horizontal'
        )
        self._plot_canny_band_pipeline(
            fig, gs, 3, 'Left Band', metadata['band_centers']['left'],
            metadata['left_grad_meta'], metadata['left_top_edge'], metadata['left_bottom_edge'],
            metadata['left_center_y'], orientation='vertical'
        )
        self._plot_canny_band_pipeline(
            fig, gs, 4, 'Right Band', metadata['band_centers']['right'],
            metadata['right_grad_meta'], metadata['right_top_edge'], metadata['right_bottom_edge'],
            metadata['right_center_y'], orientation='vertical'
        )

        if save_images and viz_dir is not None:
            self._save_visualization(fig, viz_dir, image_idx)

        return fig, fig.axes
    
    def _plot_canny_band_pipeline(self, fig, gs, row_start, title, band_position,
                                  grad_meta, left_edge, right_edge, center,
                                  orientation='horizontal'):
        """Plot the 2D band-based Canny pipeline for one representative band.
        
        Args:
            fig: Fig processed by this function.
            gs: Gs processed by this function.
            row_start: Row start processed by this function.
            title: Title processed by this function.
            band_position: Band position processed by this function.
            grad_meta: Grad meta processed by this function.
            left_edge: Left edge processed by this function.
            right_edge: Right edge processed by this function.
            center: Center processed by this function.
            orientation: Orientation processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        band_raw = grad_meta.get('band_raw')
        band_gamma = grad_meta.get('band_gamma_corrected')
        band_processed = grad_meta.get('band_processed')
        band_edges = grad_meta.get('band_edges')
        edge_projection = grad_meta.get('edge_projection')
        edge_indices = grad_meta.get('edge_indices', np.array([]))
        raw_profile = grad_meta.get('raw_profile', edge_projection)
        processed_profile = grad_meta.get('processed_profile', edge_projection)

        coord_label = 'X (pixels)' if orientation == 'horizontal' else 'Y (pixels)'
        coords = np.arange(len(edge_projection))

        ax = fig.add_subplot(gs[row_start, 0])
        ax.imshow(band_raw, cmap='gray', aspect='auto')
        ax.set_title(f'{title}\nRaw band @ {band_position}', fontsize=10)
        ax.set_ylabel('Band px', fontsize=9)
        ax.tick_params(labelsize=8)

        ax = fig.add_subplot(gs[row_start, 1])
        ax.imshow(band_gamma, cmap='gray', aspect='auto')
        ax.set_title(f'After Gamma\n(γ={self.gamma:.2f})', fontsize=10)
        ax.tick_params(labelsize=8)

        ax = fig.add_subplot(gs[row_start, 2])
        ax.imshow(band_processed, cmap='gray', aspect='auto')
        ax.set_title(f'After Median / Smooth\n(kernel={self.median_kernel_size})', fontsize=10)
        ax.tick_params(labelsize=8)

        ax = fig.add_subplot(gs[row_start, 3])
        ax.imshow(band_edges, cmap='gray', aspect='auto')
        ax.set_title('2D Canny Edge Map', fontsize=10)
        ax.tick_params(labelsize=8)

        ax = fig.add_subplot(gs[row_start, 4])
        ax.plot(coords, raw_profile, '0.5', linewidth=1.2, label='Raw profile')
        ax.plot(coords, processed_profile, 'b-', linewidth=2, alpha=0.8, label='Processed profile')
        ax.axvline(left_edge, color='red', linestyle='--', linewidth=2, label='Left/top edge')
        ax.axvline(right_edge, color='purple', linestyle='--', linewidth=2, label='Right/bottom edge')
        ax.axvline(center, color='blue', linestyle=':', linewidth=2, label='Center')
        ax.set_title(f'{title} Profiles', fontsize=10, fontweight='bold')
        ax.set_xlabel(coord_label, fontsize=9)
        ax.set_ylabel('Mean intensity', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
        ax.tick_params(labelsize=8)

        ax = fig.add_subplot(gs[row_start, 5])
        ax.plot(coords, edge_projection, 'purple', linewidth=1.8, alpha=0.85, label='Edge-pixel projection')
        if len(edge_indices) > 0:
            ax.scatter(edge_indices, edge_projection[edge_indices], color='red', s=20, alpha=0.7, label='Edge coords')
        left_range = grad_meta.get('left_search_range')
        right_range = grad_meta.get('right_search_range')
        if left_range is not None:
            ax.axvspan(left_range[0], left_range[1], alpha=0.12, color='blue')
        if right_range is not None:
            ax.axvspan(right_range[0], right_range[1], alpha=0.12, color='red')
        ax.axvline(left_edge, color='red', linestyle='--', linewidth=2)
        ax.axvline(right_edge, color='purple', linestyle='--', linewidth=2)
        ax.axvline(center, color='blue', linestyle=':', linewidth=2)
        ax.set_title('Collapsed 2D edge evidence used for picking bounds', fontsize=10)
        ax.set_xlabel(coord_label, fontsize=9)
        ax.set_ylabel('Edge count', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
        ax.tick_params(labelsize=8)
    
    def visualize_background_subtraction(
        self,
        image_idx: int = 0,
        cross_section_positions: Optional[Dict[str, List[int]]] = None,
        figsize: Tuple[int, int] = (24, 16),
        use_log_scale: bool = True,
        save_images=False
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize the background subtraction results with two-stage background estimation.
        
        Shows original image, stage 1 background, stage 2 background, and background-subtracted images.
        
        Args:
            image_idx (int): Zero-based index selecting the image.
            cross_section_positions (Optional[Dict[str, List[int]]]): Cross section positions processed by this function.
            figsize (Tuple[int, int]): Figsize processed by this function.
            use_log_scale (bool): Boolean flag controlling whether to use log scale.
            save_images: Save images processed by this function.
        
        Returns:
            Tuple[plt.Figure, np.ndarray]: Visualize background subtraction.
        """
        if self._background_subtracted_images is None:
            raise RuntimeError("Background not subtracted. Call subtract_background() first.")
        
        # Get visualization directory if saving
        viz_dir = None
        if save_images:
            viz_dir = self._get_visualization_dir('visualize_background_subtraction')
        
        original = self.original_images.isel(image_idx=image_idx).values.astype(np.float32)
        background = self._background_images.isel(image_idx=image_idx).values.astype(np.float32)
        background_stage2 = self._background_stage2_images.isel(image_idx=image_idx).values.astype(np.float32)
        subtracted = self._background_subtracted_images.isel(image_idx=image_idx).values.astype(np.float32)
        
        center_x, center_y = self._centers[image_idx]
        center_x, center_y = int(center_x), int(center_y)
        
        if cross_section_positions is None:
            cross_section_positions = {
                'horizontal': [center_y - 80, center_y, center_y + 80],
                'vertical': [center_x - 80, center_x, center_x + 80]
            }
        
        height, width = original.shape
        h_positions = [max(0, min(p, height-1)) for p in cross_section_positions['horizontal']]
        v_positions = [max(0, min(p, width-1)) for p in cross_section_positions['vertical']]
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        
        h_colors = ['#E41A1C', '#377EB8', '#4DAF4A']
        v_colors = ['#984EA3', '#FF7F00', '#A65628']
        
        # Prepare display transforms
        if use_log_scale:
            subtracted_disp = self._symmetric_log_transform(subtracted)
            scale_label_subtracted = "(sym. log)"
        else:
            subtracted_disp = subtracted
            scale_label_subtracted = ""
        
        # Row 1: Images - Original, Stage 1 BG, Stage 2 BG, Subtracted
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(original, cmap='gray')
        for i, y_pos in enumerate(h_positions):
            ax1.axhline(y_pos, color=h_colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        for i, x_pos in enumerate(v_positions):
            ax1.axvline(x_pos, color=v_colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.plot(center_x, center_y, 'r+', markersize=12, markeredgewidth=2)
        ax1.set_title('Original Image', fontsize=14)
        ax1.set_xlabel('X (pixels)', fontsize=12)
        ax1.set_ylabel('Y (pixels)', fontsize=12)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(background, cmap='gray')
        for i, y_pos in enumerate(h_positions):
            ax2.axhline(y_pos, color=h_colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        for i, x_pos in enumerate(v_positions):
            ax2.axvline(x_pos, color=v_colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_title('Background Stage 1', fontsize=14)
        ax2.set_xlabel('X (pixels)', fontsize=12)
        ax2.set_ylabel('Y (pixels)', fontsize=12)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(background_stage2, cmap='gray')
        for i, y_pos in enumerate(h_positions):
            ax3.axhline(y_pos, color=h_colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        for i, x_pos in enumerate(v_positions):
            ax3.axvline(x_pos, color=v_colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.set_title('Background Stage 2 (4x SE)', fontsize=14)
        ax3.set_xlabel('X (pixels)', fontsize=12)
        ax3.set_ylabel('Y (pixels)', fontsize=12)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        ax4 = fig.add_subplot(gs[0, 3])
        # Use diverging colormap for subtracted image (shows positive and negative)
        vmax = np.max(np.abs(subtracted_disp[subtracted_disp != 0])) if np.any(subtracted_disp != 0) else 1
        im4 = ax4.imshow(subtracted_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        for i, y_pos in enumerate(h_positions):
            ax4.axhline(y_pos, color=h_colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        for i, x_pos in enumerate(v_positions):
            ax4.axvline(x_pos, color=v_colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
        ax4.set_title(f'After Subtraction {scale_label_subtracted}', fontsize=14)
        ax4.set_xlabel('X (pixels)', fontsize=12)
        ax4.set_ylabel('Y (pixels)', fontsize=12)
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # Row 2: Horizontal cross-sections with both background stages
        for plot_idx, y_pos in enumerate(h_positions):
            ax = fig.add_subplot(gs[1, plot_idx])
            cs_original = original[y_pos, :]
            cs_background = background[y_pos, :]
            cs_background_stage2 = background_stage2[y_pos, :]
            x_coords = np.arange(len(cs_original))
            
            ax.plot(x_coords, cs_original, color=h_colors[plot_idx], linewidth=2, label='Original', alpha=0.8)
            ax.plot(x_coords, cs_background, 'b--', linewidth=2, label='BG Stage 1', alpha=0.7)
            ax.plot(x_coords, cs_background_stage2, 'k--', linewidth=2, label='BG Stage 2', alpha=0.7)
            ax.fill_between(x_coords, cs_background_stage2, alpha=0.2, color='gray')
            
            ax.set_title(f'Horizontal CS at y={y_pos}', fontsize=14)
            ax.set_xlabel('X (pixels)', fontsize=12)
            ax.set_ylabel('Intensity', fontsize=12)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            mask_left = center_x - int(self.radius)
            mask_right = center_x + int(self.radius)
            ax.axvline(mask_left, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(mask_right, color='gray', linestyle=':', alpha=0.5)
        
        # Plot difference between stage 1 and stage 2 backgrounds
        ax = fig.add_subplot(gs[1, 3])
        bg_diff = background - background_stage2
        for i, y_pos in enumerate(h_positions):
            cs_diff = bg_diff[y_pos, :]
            x_coords = np.arange(len(cs_diff))
            ax.plot(x_coords, cs_diff, color=h_colors[i], linewidth=2, label=f'y={y_pos}', alpha=0.8)
        ax.set_title('Stage1 - Stage2 Background', fontsize=14)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Intensity Difference', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Row 3: Vertical CS, Horizontal subtracted, Vertical subtracted, Info
        ax = fig.add_subplot(gs[2, 0])
        for i, x_pos in enumerate(v_positions):
            cs_original = original[:, x_pos]
            cs_background = background[:, x_pos]
            cs_background_stage2 = background_stage2[:, x_pos]
            y_coords = np.arange(len(cs_original))
            ax.plot(cs_original, y_coords, color=v_colors[i], linewidth=2, label=f'x={x_pos}', alpha=0.8)
            ax.plot(cs_background, y_coords, color=v_colors[i], linestyle='--', linewidth=1.5, alpha=0.5)
            ax.plot(cs_background_stage2, y_coords, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.invert_yaxis()
        ax.set_title('Vertical Cross-Sections', fontsize=14)
        ax.set_xlabel('Intensity', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        mask_top = center_y - int(self.radius)
        mask_bottom = center_y + int(self.radius)
        ax.axhline(mask_top, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(mask_bottom, color='gray', linestyle=':', alpha=0.5)
        
        ax = fig.add_subplot(gs[2, 1])
        for i, y_pos in enumerate(h_positions):
            cs_subtracted = subtracted[y_pos, :]
            x_coords = np.arange(len(cs_subtracted))
            ax.plot(x_coords, cs_subtracted, color=h_colors[i], linewidth=2, label=f'y={y_pos}', alpha=0.8)
        ax.set_title('Horizontal CS (Subtracted)', fontsize=14)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax = fig.add_subplot(gs[2, 2])
        for i, x_pos in enumerate(v_positions):
            cs_subtracted = subtracted[:, x_pos]
            y_coords = np.arange(len(cs_subtracted))
            ax.plot(cs_subtracted, y_coords, color=v_colors[i], linewidth=2, label=f'x={x_pos}', alpha=0.8)
        ax.invert_yaxis()
        ax.set_title('Vertical CS (Subtracted)', fontsize=14)
        ax.set_xlabel('Intensity', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Difference visualization in bottom right
        ax = fig.add_subplot(gs[2, 3])
        bg_diff_img = background - background_stage2
        vmax_diff = np.max(np.abs(bg_diff_img[bg_diff_img != 0])) if np.any(bg_diff_img != 0) else 1
        im_diff = ax.imshow(bg_diff_img, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
        ax.set_title('Stage1 - Stage2 (Image)', fontsize=14)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        plt.colorbar(im_diff, ax=ax, fraction=0.046, pad=0.04)
        
        params = self._background_subtraction_params
        fig.suptitle(
            f'Two-Stage Background Subtraction - Image {image_idx}\n'
            f'Method: {params["method"]}, Stage 1 SE: {params["structuring_element_size"]}px, '
            f'Stage 2 SE: {params["structuring_element_size"]*4}px, Pre-smooth σ: {params["pre_smooth_sigma"]}px',
            fontsize=16, y=1.00
        )
        
        plt.tight_layout()
        
        # Save image if requested
        if save_images and viz_dir is not None:
            self._save_visualization(fig, viz_dir, image_idx)
        
        return fig, fig.axes
    
    def visualize_reference_spots(self, image_idx=0, figsize=(12, 10), save_images=False):
        """Visualize the detected reference spots (T-shape and J-shape).
        
        Args:
            image_idx: Zero-based index selecting the image.
            figsize: Figsize processed by this function.
            save_images: Save images processed by this function.
        
        Returns:
            tuple: Visualize reference spots.
        """
        if self._reference_spots is None:
            raise RuntimeError("Reference spots not detected. Call detect_reference_spots() first.")
        
        # Get visualization directory if saving
        viz_dir = None
        if save_images:
            viz_dir = self._get_visualization_dir('visualize_reference_spots')
        
        bg_subtracted = self._filtered_images.isel(image_idx=image_idx).values
        t_shape = self._reference_spots['t_shape'][image_idx]
        j_shape = self._reference_spots['j_shape'][image_idx]
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Display with symmetric log scale
        bg_disp = self._symmetric_log_transform(bg_subtracted)
        vmax = np.max(np.abs(bg_disp[bg_disp != 0])) if np.any(bg_disp != 0) else 1
        ax.imshow(bg_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Plot T-shape spots
        if t_shape is not None:
            for i, (x, y) in enumerate(t_shape):
                ax.add_patch(plt.Circle((x, y), 8, color='yellow', fill=False, linewidth=2))
                ax.text(x + 10, y, f'T{i+1}', color='yellow', fontsize=10, fontweight='bold')
        
        # Plot J-shape spots
        if j_shape is not None:
            for i, (x, y) in enumerate(j_shape):
                ax.add_patch(plt.Circle((x, y), 8, color='cyan', fill=False, linewidth=2))
                ax.text(x + 10, y, f'J{i+1}', color='cyan', fontsize=10, fontweight='bold')
        
        n_t = len(t_shape) if t_shape is not None else 0
        n_j = len(j_shape) if j_shape is not None else 0
        ax.set_title(f'Reference Spots (idx={image_idx})\nT-shape: {n_t} spots, J-shape: {n_j} spots', fontsize=14)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='yellow', markersize=10, markeredgewidth=2, label='T-shape'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='cyan', markersize=10, markeredgewidth=2, label='J-shape')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save image if requested
        if save_images and viz_dir is not None:
            self._save_visualization(fig, viz_dir, image_idx)
        
        return fig, ax
    
    def visualize_refined_reference_spots(self, image_idx=0, figsize=(16, 7), save_images=False):
        """Visualize refined reference spots in comparison with initially found reference spots.
        
        Args:
            image_idx: Zero-based index selecting the image.
            figsize: Figsize processed by this function.
            save_images: Save images processed by this function.
        
        Returns:
            tuple: Visualize refined reference spots.
        """
        if self._reference_spots is None:
            raise RuntimeError("Reference spots not detected. Call detect_reference_spots() first.")
        if self._refined_reference_spots is None:
            raise RuntimeError("Reference spots not refined. Call refine_reference_spot_positions() first.")
        
        # Get visualization directory if saving
        viz_dir = None
        if save_images:
            viz_dir = self._get_visualization_dir('visualize_refined_reference_spots')
        
        bg_subtracted = self._background_subtracted_images.isel(image_idx=image_idx).values
        
        # Original spots
        orig_t_shape = self._reference_spots['t_shape'][image_idx]
        orig_j_shape = self._reference_spots['j_shape'][image_idx]
        
        # Refined spots
        ref_t_shape = self._refined_reference_spots['t_shape'][image_idx]
        ref_j_shape = self._refined_reference_spots['j_shape'][image_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Display with symmetric log scale
        bg_disp = self._symmetric_log_transform(bg_subtracted)
        vmax = np.max(np.abs(bg_disp[bg_disp != 0])) if np.any(bg_disp != 0) else 1
        
        # Left panel: Original spots
        ax = axes[0]
        ax.imshow(bg_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        if orig_t_shape is not None:
            for i, (x, y) in enumerate(orig_t_shape):
                ax.add_patch(plt.Circle((x, y), 8, color='yellow', fill=False, linewidth=2))
        if orig_j_shape is not None:
            for i, (x, y) in enumerate(orig_j_shape):
                ax.add_patch(plt.Circle((x, y), 8, color='cyan', fill=False, linewidth=2))
        
        n_t = len(orig_t_shape) if orig_t_shape is not None else 0
        n_j = len(orig_j_shape) if orig_j_shape is not None else 0
        ax.set_title(f'Original Reference Spots (idx={image_idx})\nT: {n_t}, J: {n_j}', fontsize=14)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Right panel: Refined spots with original for comparison
        ax = axes[1]
        ax.imshow(bg_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Plot original spots (smaller, dimmer)
        if orig_t_shape is not None:
            for x, y in orig_t_shape:
                ax.add_patch(plt.Circle((x, y), 5, color='yellow', fill=False, linewidth=1, alpha=0.4))
        if orig_j_shape is not None:
            for x, y in orig_j_shape:
                ax.add_patch(plt.Circle((x, y), 5, color='cyan', fill=False, linewidth=1, alpha=0.4))
        
        # Plot refined spots (larger, brighter)
        if ref_t_shape is not None:
            for i, (x, y) in enumerate(ref_t_shape):
                ax.add_patch(plt.Circle((x, y), 5, color='lime', fill=False, linewidth=2))
        if ref_j_shape is not None:
            for i, (x, y) in enumerate(ref_j_shape):
                ax.add_patch(plt.Circle((x, y), 5, color='magenta', fill=False, linewidth=2))
        
        # Draw arrows showing refinement shifts
        if orig_t_shape is not None and ref_t_shape is not None:
            for (ox, oy), (rx, ry) in zip(orig_t_shape, ref_t_shape):
                if np.sqrt((rx-ox)**2 + (ry-oy)**2) > 0.5:  # Only show if moved
                    ax.annotate('', xy=(rx, ry), xytext=(ox, oy),
                               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        if orig_j_shape is not None and ref_j_shape is not None:
            for (ox, oy), (rx, ry) in zip(orig_j_shape, ref_j_shape):
                if np.sqrt((rx-ox)**2 + (ry-oy)**2) > 0.5:
                    ax.annotate('', xy=(rx, ry), xytext=(ox, oy),
                               arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        
        ax.set_title('Refined Reference Spots\nGreen/Magenta: refined, Yellow/Cyan: original', fontsize=14)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='yellow', markersize=8, markeredgewidth=1, alpha=0.4, label='Original T-shape'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='cyan', markersize=8, markeredgewidth=1, alpha=0.4, label='Original J-shape'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='lime', markersize=10, markeredgewidth=2, label='Refined T-shape'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='magenta', markersize=10, markeredgewidth=2, label='Refined J-shape'),
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Save image if requested
        if save_images and viz_dir is not None:
            self._save_visualization(fig, viz_dir, image_idx)
        
        return fig, axes
    
    # ========== Layout Association Methods ==========
    
    def associate_layout(self, layout_loader: 'ArrayLayoutLoader') -> None:
        """Associate array layout information with the spot grid.
        Automatically loads enrichment data from the default enrichment file.
        
        Args:
            layout_loader ('ArrayLayoutLoader'): Layout loader processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self._spot_intensities is None and self._refined_grid_positions is None:
            raise RuntimeError("Spots not computed. Call compute_spot_intensities() or refine_grid_parameters() first.")
        
        peptides_type = self._get_peptides_type()
        expected_rows = self.PTK_GRID_ROWS if peptides_type == 'PTK' else self.STK_GRID_ROWS
        expected_cols = self.PTK_GRID_COLS if peptides_type == 'PTK' else self.STK_GRID_COLS
        
        layout_dims = layout_loader.grid_dimensions
        if layout_dims != (expected_rows, expected_cols):
            raise ValueError(
                f"Layout dimensions {layout_dims} don't match expected "
                f"{peptides_type} grid size ({expected_rows}×{expected_cols})"
            )
        
        if layout_loader.chip_type != peptides_type:
            raise ValueError(
                f"Layout chip type '{layout_loader.chip_type}' doesn't match "
                f"processor peptides_type '{peptides_type}'"
            )
        
        self._layout_loader = layout_loader
        original_spot_array = layout_loader.spot_array.copy()
        
        # Transpose the spot layout array to swap row/col interpretation
        # This means array[i,j] will contain what was originally at array[j,i]
        n_rows, n_cols = original_spot_array.shape
        transposed_data = np.empty((n_cols, n_rows), dtype=object)
        
        for i in range(n_rows):
            for j in range(n_cols):
                spot = original_spot_array.values[i, j]
                if spot is not None and isinstance(spot, dict):
                    # Make a copy and swap Row and Col values in the dictionary
                    spot_copy = spot.copy()
                    if 'Row' in spot_copy and 'Col' in spot_copy:
                        spot_copy['Row'], spot_copy['Col'] = spot_copy['Col'], spot_copy['Row']
                    transposed_data[j, i] = spot_copy
                else:
                    transposed_data[j, i] = spot
        
        # Create transposed xarray
        self._spot_layout = xr.DataArray(
            data=transposed_data,
            dims=['spot_row', 'spot_col'],
            coords={
                'spot_row': np.arange(n_cols),
                'spot_col': np.arange(n_rows)
            },
            attrs=original_spot_array.attrs.copy()
        )
        # Update grid dimensions in attrs to reflect transpose
        self._spot_layout.attrs['grid_dimensions'] = (n_cols, n_rows)
        
        print(f"Associated {peptides_type} layout: {expected_rows}×{expected_cols} grid, "
              f"{layout_loader.spot_array.attrs['n_spots']} spots (Row/Col transposed)")
        
        # Automatically load enrichment data
        self._load_enrichment_data(self._enrichment_file)
    
    @property
    def spot_layout(self) -> Optional[xr.DataArray]:
        """Get the associated spot layout array.
        
        Args:
            None.
        
        Returns:
            Optional[xr.DataArray]: Stored spot layout.
        """
        return self._spot_layout
    
    @property
    def layout_loader(self) -> Optional['ArrayLayoutLoader']:
        """Get the associated ArrayLayoutLoader instance.
        
        Args:
            None.
        
        Returns:
            Optional['ArrayLayoutLoader']: Stored layout loader.
        """
        return self._layout_loader
    
    # ========== Enrichment Data Methods ==========
    
    def _load_enrichment_data(self, filepath: Union[str, Path], verbose: bool = True) -> None:
        """Load enrichment data from a CSV file and associate it with spots by sequence.
        
        The CSV file must contain a 'Sequence' column for matching. All columns
        are read dynamically from the file.
        
        Note: One sequence may match multiple rows in the enrichment file (e.g.,
        one peptide matching multiple proteins). All matches are stored as a list.
        
        Args:
            filepath (Union[str, Path]): Path-like value for filepath.
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self._spot_layout is None:
            raise RuntimeError("Layout not associated. Call associate_layout() first.")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Enrichment file not found: {filepath}")
        
        if verbose:
            print(f"\nLoading enrichment data from: {filepath.name}")
        
        # Read CSV and get column names dynamically
        self._enrichment_df = pd.read_csv(filepath)
        self._enrichment_columns = list(self._enrichment_df.columns)
        
        if verbose:
            print(f"  Found {len(self._enrichment_df)} rows")
            print(f"  Columns: {', '.join(self._enrichment_columns)}")
        
        # Validate that 'Sequence' column exists
        if 'Sequence' not in self._enrichment_columns:
            raise ValueError("Enrichment CSV must contain a 'Sequence' column for matching")
        
        # Build lookup dictionary: sequence -> list of row dicts
        self._enrichment_by_sequence = {}
        for _, row in self._enrichment_df.iterrows():
            seq = row['Sequence']
            row_dict = row.to_dict()
            if seq not in self._enrichment_by_sequence:
                self._enrichment_by_sequence[seq] = []
            self._enrichment_by_sequence[seq].append(row_dict)
        
        if verbose:
            print(f"  Unique sequences in enrichment data: {len(self._enrichment_by_sequence)}")
        
        # Build array-format storage for each column
        self._build_enrichment_arrays(verbose)
        
        if verbose:
            print("Enrichment data loaded and associated successfully!")
    
    def _build_enrichment_arrays(self, verbose: bool = True) -> None:
        """Build 2D arrays (n_rows x n_cols) for each enrichment column.
        
        For columns with multiple matches per sequence, stores a list of values.
        
        Args:
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self._spot_layout is None or self._enrichment_by_sequence is None:
            return
        
        n_rows, n_cols = self._spot_layout.shape
        self._enrichment_arrays = {}
        
        # Get columns to store (all from enrichment file)
        columns_to_store = [c for c in self._enrichment_columns]
        
        # Count matches for statistics
        n_matched = 0
        n_unmatched = 0
        n_multi_match = 0
        
        for col_name in columns_to_store:
            arr = np.empty((n_rows, n_cols), dtype=object)
            
            for i in range(n_rows):
                for j in range(n_cols):
                    spot = self._spot_layout.values[i, j]
                    if spot is None:
                        arr[i, j] = None
                        continue
                    
                    seq = spot.get('Sequence')
                    if seq is None:
                        arr[i, j] = None
                        continue
                    
                    matches = self._enrichment_by_sequence.get(seq, [])
                    if len(matches) == 0:
                        arr[i, j] = None
                        if col_name == columns_to_store[0]:  # Count only once
                            n_unmatched += 1
                    elif len(matches) == 1:
                        arr[i, j] = matches[0].get(col_name)
                        if col_name == columns_to_store[0]:
                            n_matched += 1
                    else:
                        # Multiple matches - store as list
                        arr[i, j] = [m.get(col_name) for m in matches]
                        if col_name == columns_to_store[0]:
                            n_matched += 1
                            n_multi_match += 1
            
            self._enrichment_arrays[col_name] = arr
        
        if verbose:
            total_spots = n_matched + n_unmatched
            print(f"  Matched sequences: {n_matched}/{total_spots}")
            print(f"  Sequences with multiple enrichment matches: {n_multi_match}")
            if n_unmatched > 0:
                print(f"  Unmatched sequences: {n_unmatched}")
    
    def load_enrichment_data(self, filepath: Union[str, Path], verbose: bool = True) -> None:
        """Manually load enrichment data from a custom CSV file path.
        
        This can be used to override the default enrichment file or reload
        enrichment data with a different file.
        
        Args:
            filepath (Union[str, Path]): Path-like value for filepath.
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self._load_enrichment_data(filepath, verbose)
    
    def get_enrichment_array(self, column_name: str) -> Optional[np.ndarray]:
        """Get the 2D array of enrichment values for a specific column.
        
        Args:
            column_name (str): Column name used by this function.
        
        Returns:
            Optional[np.ndarray]: Requested enrichment array.
        """
        if self._enrichment_arrays is None:
            return None
        return self._enrichment_arrays.get(column_name)
    
    
    def get_spot_info(self, row: int, col: int, image_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get layout and enrichment information for a spot at a specific grid position.
        
        Args:
            row (int): Row used by this function.
            col (int): Col used by this function.
            image_idx (Optional[int]): Zero-based index selecting the image.
        
        Returns:
            Optional[Dict[str, Any]]: Requested spot info.
        """
        if self._spot_layout is None:
            return None
        
        n_rows, n_cols = self._spot_layout.shape
        
        if row < 0 or row >= n_rows:
            raise IndexError(f"Row {row} out of range [0, {n_rows - 1}]")
        if col < 0 or col >= n_cols:
            raise IndexError(f"Col {col} out of range [0, {n_cols - 1}]")
        
        spot_data = self._spot_layout.values[row, col]
        
        if spot_data is None:
            return None
        
        # Make a copy to avoid modifying the original
        result = spot_data.copy()
        
        # Add intensity, max intensity, median, saturation, and image metadata if image_idx is provided
        if image_idx is not None:
            # Integrated intensity
            if self._refined_spot_intensities is not None:
                result['Intensity'] = float(self._refined_spot_intensities.isel(
                    image_idx=image_idx, spot_row=row, spot_col=col).values)
            elif self._spot_intensities is not None:
                result['Intensity'] = float(self._spot_intensities.isel(
                    image_idx=image_idx, spot_row=row, spot_col=col).values)
            else:
                result['Intensity'] = None
            
            # Max intensity
            if self._refined_spot_max_intensities is not None:
                result['MaxIntensity'] = float(self._refined_spot_max_intensities.isel(
                    image_idx=image_idx, spot_row=row, spot_col=col).values)
            elif self._spot_max_intensities is not None:
                result['MaxIntensity'] = float(self._spot_max_intensities.isel(
                    image_idx=image_idx, spot_row=row, spot_col=col).values)
            else:
                result['MaxIntensity'] = None
            
            # Median intensity
            if self._refined_spot_median_intensities is not None:
                result['MedianIntensity'] = float(self._refined_spot_median_intensities.isel(
                    image_idx=image_idx, spot_row=row, spot_col=col).values)
            elif self._spot_median_intensities is not None:
                result['MedianIntensity'] = float(self._spot_median_intensities.isel(
                    image_idx=image_idx, spot_row=row, spot_col=col).values)
            else:
                result['MedianIntensity'] = None
            
            # Saturation fraction
            if self._refined_spot_saturation_fractions is not None:
                result['SaturationFraction'] = float(self._refined_spot_saturation_fractions.isel(
                    image_idx=image_idx, spot_row=row, spot_col=col).values)
            elif self._spot_saturation_fractions is not None:
                result['SaturationFraction'] = float(self._spot_saturation_fractions.isel(
                    image_idx=image_idx, spot_row=row, spot_col=col).values)
            else:
                result['SaturationFraction'] = None
            
            # Add image metadata from original_images coordinates
            image_metadata_fields = [
                'PamChip_barcode', 'Array', 'FoV', 'Exposure_time', 
                'Pump_cycle', 'Image_number', 'Temperature'
            ]
            for field in image_metadata_fields:
                if field in self.original_images.coords:
                    coord_data = self.original_images.coords[field]
                    if 'image_idx' in coord_data.dims:
                        value = coord_data.isel(image_idx=image_idx).values
                        # Convert numpy types to Python native types
                        if hasattr(value, 'item'):
                            value = value.item()
                        result[field] = value
                    else:
                        # Scalar coordinate (same for all images)
                        value = coord_data.values
                        if hasattr(value, 'item'):
                            value = value.item()
                        result[field] = value
        
        # Add enrichment data if available
        if self._enrichment_by_sequence is not None:
            seq = spot_data.get('Sequence')
            if seq is not None:
                enrichment_matches = self._enrichment_by_sequence.get(seq, [])
                result['Enrichment'] = enrichment_matches
            else:
                result['Enrichment'] = []
        
        return result
    
    def get_spot_field(self, row: int, col: int, field: str):
        """Get a specific field from a spot's layout information.
        
        Args:
            row (int): Row used by this function.
            col (int): Col used by this function.
            field (str): Field used by this function.
        
        Returns:
            object: Requested spot field.
        """
        spot_info = self.get_spot_info(row, col)
        if spot_info is None:
            return None
        return spot_info.get(field, None)
    
    def get_layout_field_array(self, field: str) -> Optional[np.ndarray]:
        """Get array of a specific field for all spots.
        
        Args:
            field (str): Field used by this function.
        
        Returns:
            Optional[np.ndarray]: Requested layout field array.
        """
        if self._spot_layout is None:
            return None
        
        n_rows, n_cols = self._spot_layout.shape
        field_array = np.empty((n_rows, n_cols), dtype=object)
        
        for i in range(n_rows):
            for j in range(n_cols):
                spot = self._spot_layout.values[i, j]
                field_array[i, j] = spot.get(field) if spot is not None else None
        
        return field_array
    
    def get_all_spot_ids(self) -> Optional[np.ndarray]:
        """Get array of all spot IDs.
        
        Args:
            None.
        
        Returns:
            Optional[np.ndarray]: Requested all spot IDs.
        """
        return self.get_layout_field_array('ID')
    
    def get_all_spot_sequences(self) -> Optional[np.ndarray]:
        """Get array of all spot sequences.
        
        Args:
            None.
        
        Returns:
            Optional[np.ndarray]: Requested all spot sequences.
        """
        return self.get_layout_field_array('Sequence')
    
    def get_spot_intensity_with_info(self, image_idx: int, row: int, col: int) -> Dict[str, Any]:
        """Get both intensity and layout information for a specific spot.
        
        Args:
            image_idx (int): Zero-based index selecting the image.
            row (int): Row used by this function.
            col (int): Col used by this function.
        
        Returns:
            Dict[str, Any]: Requested spot intensity with info.
        """
        result = {
            'intensity': None, 'max_intensity': None, 'median_intensity': None,
            'saturation_fraction': None, 'position': None, 'layout': None
        }
        
        if self._refined_spot_intensities is not None:
            result['intensity'] = float(self._refined_spot_intensities.isel(
                image_idx=image_idx, spot_row=row, spot_col=col).values)
        elif self._spot_intensities is not None:
            result['intensity'] = float(self._spot_intensities.isel(
                image_idx=image_idx, spot_row=row, spot_col=col).values)
        
        if self._refined_spot_max_intensities is not None:
            result['max_intensity'] = float(self._refined_spot_max_intensities.isel(
                image_idx=image_idx, spot_row=row, spot_col=col).values)
        elif self._spot_max_intensities is not None:
            result['max_intensity'] = float(self._spot_max_intensities.isel(
                image_idx=image_idx, spot_row=row, spot_col=col).values)
        
        if self._refined_spot_median_intensities is not None:
            result['median_intensity'] = float(self._refined_spot_median_intensities.isel(
                image_idx=image_idx, spot_row=row, spot_col=col).values)
        elif self._spot_median_intensities is not None:
            result['median_intensity'] = float(self._spot_median_intensities.isel(
                image_idx=image_idx, spot_row=row, spot_col=col).values)
        
        if self._refined_spot_saturation_fractions is not None:
            result['saturation_fraction'] = float(self._refined_spot_saturation_fractions.isel(
                image_idx=image_idx, spot_row=row, spot_col=col).values)
        elif self._spot_saturation_fractions is not None:
            result['saturation_fraction'] = float(self._spot_saturation_fractions.isel(
                image_idx=image_idx, spot_row=row, spot_col=col).values)
        
        if self._refined_grid_positions is not None and self._refined_grid_positions[image_idx] is not None:
            result['position'] = tuple(self._refined_grid_positions[image_idx][row, col])
        elif self._spot_grid_positions is not None and self._spot_grid_positions[image_idx] is not None:
            result['position'] = tuple(self._spot_grid_positions[image_idx][row, col])
        
        result['layout'] = self.get_spot_info(row, col)
        return result
    
    def get_intensity_by_spot_id(self, image_idx: int, spot_id: str) -> Optional[float]:
        """Get the intensity of a spot by its ID.
        
        Args:
            image_idx (int): Zero-based index selecting the image.
            spot_id (str): Spot ID used by this function.
        
        Returns:
            Optional[float]: Requested intensity by spot ID.
        """
        if self._spot_layout is None:
            return None
        
        n_rows, n_cols = self._spot_layout.shape
        for row in range(n_rows):
            for col in range(n_cols):
                spot = self._spot_layout.values[row, col]
                if spot is not None and spot.get('ID') == spot_id:
                    if self._refined_spot_intensities is not None:
                        return float(self._refined_spot_intensities.isel(
                            image_idx=image_idx, spot_row=row, spot_col=col).values)
                    elif self._spot_intensities is not None:
                        return float(self._spot_intensities.isel(
                            image_idx=image_idx, spot_row=row, spot_col=col).values)
        return None
    
    def get_intensities_dataframe(self, image_idx: int, include_enrichment: bool = True):
        """Get a DataFrame with spot intensities, layout, and enrichment information.
        
        Args:
            image_idx (int): Zero-based index selecting the image.
            include_enrichment (bool): Boolean flag controlling whether to include enrichment.
        
        Returns:
            object: Requested intensities dataframe.
        """
        if self._spot_layout is None:
            print("Warning: No layout associated. Call associate_layout() first.")
            return None
        
        if self._refined_spot_intensities is not None:
            intensities = self._refined_spot_intensities.isel(image_idx=image_idx).values
            max_intensities = self._refined_spot_max_intensities.isel(image_idx=image_idx).values if self._refined_spot_max_intensities is not None else None
            median_intensities = self._refined_spot_median_intensities.isel(image_idx=image_idx).values if self._refined_spot_median_intensities is not None else None
            saturation_fractions = self._refined_spot_saturation_fractions.isel(image_idx=image_idx).values if self._refined_spot_saturation_fractions is not None else None
            positions = self._refined_grid_positions[image_idx] if self._refined_grid_positions else None
        elif self._spot_intensities is not None:
            intensities = self._spot_intensities.isel(image_idx=image_idx).values
            max_intensities = self._spot_max_intensities.isel(image_idx=image_idx).values if self._spot_max_intensities is not None else None
            median_intensities = self._spot_median_intensities.isel(image_idx=image_idx).values if self._spot_median_intensities is not None else None
            saturation_fractions = self._spot_saturation_fractions.isel(image_idx=image_idx).values if self._spot_saturation_fractions is not None else None
            positions = self._spot_grid_positions[image_idx] if self._spot_grid_positions else None
        else:
            return None
        
        n_rows, n_cols = self._spot_layout.shape
        records = []
        
        for row in range(n_rows):
            for col in range(n_cols):
                record = {
                    'spot_row': row, 
                    'spot_col': col, 
                    'intensity': intensities[row, col],
                    'max_intensity': max_intensities[row, col] if max_intensities is not None else None,
                    'median_intensity': median_intensities[row, col] if median_intensities is not None else None,
                    'saturation_fraction': saturation_fractions[row, col] if saturation_fractions is not None else None
                }
                
                if positions is not None:
                    record['x'] = positions[row, col, 0]
                    record['y'] = positions[row, col, 1]
                
                spot_info = self._spot_layout.values[row, col]
                if spot_info is not None:
                    for key, value in spot_info.items():
                        if key not in ['Row', 'Col']:
                            record[key] = value
                
                # Add enrichment data (first match only for DataFrame simplicity)
                if include_enrichment and self._enrichment_by_sequence is not None:
                    seq = spot_info.get('Sequence') if spot_info else None
                    if seq is not None:
                        matches = self._enrichment_by_sequence.get(seq, [])
                        if len(matches) > 0:
                            first_match = matches[0]
                            for key, value in first_match.items():
                                if key != 'Sequence':  # Avoid duplicate
                                    record[f'Enr_{key}'] = value
                            record['Enr_n_matches'] = len(matches)
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    # ========== Statistics Methods ==========
    

    def get_statistics(self):
        """Get statistics.
        
        Args:
            None.
        
        Returns:
            dict: Requested statistics.
        """
        if self._centers is None:
            raise RuntimeError("Images not processed. Call apply_circular_mask() first.")
        return {
            'n_images': len(self._centers),
            'center_x_mean': float(np.mean(self._centers[:, 0])),
            'center_x_std': float(np.std(self._centers[:, 0])),
            'center_y_mean': float(np.mean(self._centers[:, 1])),
            'center_y_std': float(np.std(self._centers[:, 1])),
        }
    
    def get_grid_refinement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics of the refined grid parameters.
        
        Args:
            None.
        
        Returns:
            Dict[str, Any]: Requested grid refinement statistics.
        """
        if self._refined_grid_params_list is None:
            raise RuntimeError("Grid not refined. Call refine_grid_parameters() first.")
        
        valid_params = [p for p in self._refined_grid_params_list if p is not None]
        
        if len(valid_params) == 0:
            return {'n_valid': 0, 'n_total': len(self._refined_grid_params_list)}
        
        def compute_stats(values, name):
            """Compute stats.
            
            Args:
                values: Collection of input values processed by this helper.
                name: Name processed by this function.
            
            Returns:
                dict: Computed stats.
            """
            arr = np.array(values)
            return {
                f'{name}_mean': float(np.mean(arr)),
                f'{name}_median': float(np.median(arr)),
                f'{name}_std': float(np.std(arr)),
                f'{name}_min': float(np.min(arr)),
                f'{name}_max': float(np.max(arr)),
                f'{name}_q25': float(np.percentile(arr, 25)),
                f'{name}_q75': float(np.percentile(arr, 75)),
            }
        
        dx_vals = [p['dx'] for p in valid_params]
        dy_vals = [p['dy'] for p in valid_params]
        dangle_deg_vals = [p['dangle_deg'] for p in valid_params]
        dspacing_vals = [p['dspacing'] for p in valid_params]
        
        orig_intensity_vals = [p['original_intensity'] for p in valid_params]
        ref_intensity_vals = [p['refined_intensity'] for p in valid_params]
        improvement_vals = [p['intensity_improvement'] for p in valid_params]
        improvement_pct_vals = [p['improvement_percent'] for p in valid_params]
        
        center_x_vals = [p['center_x'] for p in valid_params]
        center_y_vals = [p['center_y'] for p in valid_params]
        angle_deg_vals = [p['angle_deg'] for p in valid_params]
        spacing_vals = [p['spacing'] for p in valid_params]
        
        stats = {
            'n_valid': len(valid_params),
            'n_total': len(self._refined_grid_params_list),
            'n_invalid': len(self._refined_grid_params_list) - len(valid_params),
        }
        
        stats.update(compute_stats(dx_vals, 'dx'))
        stats.update(compute_stats(dy_vals, 'dy'))
        stats.update(compute_stats(dangle_deg_vals, 'dangle_deg'))
        stats.update(compute_stats(dspacing_vals, 'dspacing'))
        
        stats.update(compute_stats(orig_intensity_vals, 'original_intensity'))
        stats.update(compute_stats(ref_intensity_vals, 'refined_intensity'))
        stats.update(compute_stats(improvement_vals, 'intensity_improvement'))
        stats.update(compute_stats(improvement_pct_vals, 'improvement_percent'))
        
        stats.update(compute_stats(center_x_vals, 'refined_center_x'))
        stats.update(compute_stats(center_y_vals, 'refined_center_y'))
        stats.update(compute_stats(angle_deg_vals, 'refined_angle_deg'))
        stats.update(compute_stats(spacing_vals, 'refined_spacing'))
        
        if hasattr(self, '_grid_refinement_params') and self._grid_refinement_params is not None:
            stats['optimization_params'] = self._grid_refinement_params.copy()
        
        return stats
    
    def get_reference_spot_statistics(self) -> Dict[str, Any]:
        """Get statistics on the detected reference spot positions (T-shape and J-shape).
        
        Args:
            None.
        
        Returns:
            Dict[str, Any]: Requested reference spot statistics.
        """
        if self._reference_spots is None:
            raise RuntimeError("Reference spots not detected. Call detect_reference_spots() first.")
        
        def compute_spot_stats(spots_list, shape_name):
            """Compute statistics for a list of spot arrays.
            
            Args:
                spots_list: Spots list processed by this function.
                shape_name: Shape name processed by this function.
            
            Returns:
                object: Computed spot stats.
            """
            # Filter out None entries
            valid_spots = [s for s in spots_list if s is not None]
            if len(valid_spots) == 0:
                return {f'{shape_name}_n_valid': 0}
            
            # Stack all spots: shape (n_images, n_spots, 2)
            stacked = np.array(valid_spots)
            n_spots = stacked.shape[1]
            
            stats = {
                f'{shape_name}_n_valid': len(valid_spots),
                f'{shape_name}_n_spots': n_spots,
            }
            
            for spot_idx in range(n_spots):
                x_vals = stacked[:, spot_idx, 0]
                y_vals = stacked[:, spot_idx, 1]
                
                stats[f'{shape_name}_spot{spot_idx+1}_x_mean'] = float(np.mean(x_vals))
                stats[f'{shape_name}_spot{spot_idx+1}_x_std'] = float(np.std(x_vals))
                stats[f'{shape_name}_spot{spot_idx+1}_x_min'] = float(np.min(x_vals))
                stats[f'{shape_name}_spot{spot_idx+1}_x_max'] = float(np.max(x_vals))
                stats[f'{shape_name}_spot{spot_idx+1}_y_mean'] = float(np.mean(y_vals))
                stats[f'{shape_name}_spot{spot_idx+1}_y_std'] = float(np.std(y_vals))
                stats[f'{shape_name}_spot{spot_idx+1}_y_min'] = float(np.min(y_vals))
                stats[f'{shape_name}_spot{spot_idx+1}_y_max'] = float(np.max(y_vals))
            
            return stats
        
        results = {'n_images': len(self._reference_spots['t_shape'])}
        
        # Initial reference spots
        results.update(compute_spot_stats(self._reference_spots['t_shape'], 'initial_t'))
        results.update(compute_spot_stats(self._reference_spots['j_shape'], 'initial_j'))
        
        # # Refined reference spots (if available)
        # if self._refined_reference_spots is not None:
        #     results['has_refined'] = True
        #     results.update(compute_spot_stats(self._refined_reference_spots['t_shape'], 'refined_t'))
        #     results.update(compute_spot_stats(self._refined_reference_spots['j_shape'], 'refined_j'))
        # else:
        #     results['has_refined'] = False
        
        return results

    def build_final_output(self, verbose: bool = True) -> pd.DataFrame:
        """Build a comprehensive table of all spot data across all images.
        
        Iterates through Arrays, then Cols, then Rows. For spots with multiple
        enrichment matches, each match is added as a separate row.
        
        Optimized for speed: vectorizes array access and builds DataFrame from dict of arrays.
        
        The result is stored in self.final_output attribute.
        
        Args:
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            pd.DataFrame: Constructed final output.
        """
        if self._spot_layout is None:
            raise RuntimeError("Layout not associated. Call associate_layout() first.")
        
        if self._spot_intensities is None and self._refined_spot_intensities is None:
            raise RuntimeError("Spot intensities not computed. Call compute_spot_intensities() first.")
        
        n_images = len(self.original_images.image_idx)
        n_rows, n_cols = self._spot_layout.shape
        
        if verbose:
            print("\nBuilding final output table (optimized)...")
            print(f"  Images: {n_images}, Grid: {n_rows}×{n_cols}")
        
        # OPTIMIZATION: Extract all intensity and metadata arrays once (vectorized)
        if verbose:
            print("  Extracting intensity arrays...")
        
        # Get intensity arrays (refined if available, otherwise original)
        if self._refined_spot_intensities is not None:
            intensities = self._refined_spot_intensities.values  # shape: (n_images, n_rows, n_cols)
            max_intensities = self._refined_spot_max_intensities.values if self._refined_spot_max_intensities is not None else None
            median_intensities = self._refined_spot_median_intensities.values if self._refined_spot_median_intensities is not None else None
            saturation_fracs = self._refined_spot_saturation_fractions.values if self._refined_spot_saturation_fractions is not None else None
        else:
            intensities = self._spot_intensities.values
            max_intensities = self._spot_max_intensities.values if self._spot_max_intensities is not None else None
            median_intensities = self._spot_median_intensities.values if self._spot_median_intensities is not None else None
            saturation_fracs = self._spot_saturation_fractions.values if self._spot_saturation_fractions is not None else None
        
        # Extract image metadata arrays
        image_metadata_fields = ['PamChip_barcode', 'Array', 'FoV', 'Exposure_time', 
                                 'Pump_cycle', 'Image_number', 'Temperature']
        metadata_arrays = {}
        for field in image_metadata_fields:
            if field in self.original_images.coords:
                coord_data = self.original_images.coords[field]
                if 'image_idx' in coord_data.dims:
                    metadata_arrays[field] = coord_data.values
                else:
                    # Scalar coordinate
                    val = coord_data.values
                    if hasattr(val, 'item'):
                        val = val.item()
                    metadata_arrays[field] = np.full(n_images, val)
        
        # Get all unique Array values and sort images by Array
        if 'Array' in metadata_arrays:
            array_values = [(metadata_arrays['Array'][i] if hasattr(metadata_arrays['Array'][i], 'item') 
                           else metadata_arrays['Array'][i], i) for i in range(n_images)]
        else:
            array_values = [(i, i) for i in range(n_images)]
        array_values.sort(key=lambda x: x[0])
        
        # Enrichment columns mapping
        enrichment_mapped = {'PepProtein_SeqMatch', 'PepProtein_UniprotName', 
                            'PepProtein_UniprotID', 'entrezid', 'PepProtein_SeqSimilarity',
                            'Sequence', 'ID'}
        
        # OPTIMIZATION: Build columns as lists, then construct DataFrame from dict
        if verbose:
            print("  Building table rows...")
        
        # Pre-allocate lists for all columns (dict of arrays approach)
        column_data = {
            'image_idx': [],
            'Barcode': [],
            'Row': [],
            'spotRow': [],
            'spotCol': [],
            'Exposure Time': [],
            'Cycle': [],
            'ID': [],
            'Sequence': [],
            'I_median': [],
            'Signal_Saturation': [],
        }
        
        for arr_val, image_idx in array_values:
            if verbose and image_idx % 100 == 0:
                print(f"    Processing image {image_idx + 1}/{n_images}...")
            
            # Iterate through Cols first, then Rows
            for col in range(n_cols):
                for row in range(n_rows):
                    spot_data = self._spot_layout.values[row, col]
                    
                    if spot_data is None:
                        continue
                    
                    # OPTIMIZATION: Direct array access instead of get_spot_info
                    intensity = intensities[image_idx, row, col]
                    max_int = max_intensities[image_idx, row, col] if max_intensities is not None else None
                    median_int = median_intensities[image_idx, row, col] if median_intensities is not None else None
                    sat_frac = saturation_fracs[image_idx, row, col] if saturation_fracs is not None else None
                    
                    # Get enrichment matches
                    seq = spot_data.get('Sequence')
                    if self._enrichment_by_sequence is not None and seq is not None:
                        enrichment_matches = self._enrichment_by_sequence.get(seq, [])
                    else:
                        enrichment_matches = []
                    
                    # OPTIMIZATION: Fast path for 0 or 1 enrichment matches (most common case)
                    if len(enrichment_matches) <= 1:
                        enr_match = enrichment_matches[0] if enrichment_matches else {}
                        
                        # Add core columns
                        column_data['image_idx'].append(image_idx)
                        column_data['Barcode'].append(metadata_arrays.get('PamChip_barcode', [None]*n_images)[image_idx])
                        column_data['Row'].append(metadata_arrays.get('Array', [None]*n_images)[image_idx])
                        column_data['spotRow'].append(spot_data.get('Col'))
                        column_data['spotCol'].append(spot_data.get('Row'))
                        column_data['Exposure Time'].append(metadata_arrays.get('Exposure_time', [None]*n_images)[image_idx])
                        column_data['Cycle'].append(metadata_arrays.get('Pump_cycle', [None]*n_images)[image_idx])
                        column_data['ID'].append(spot_data.get('ID'))
                        column_data['Sequence'].append(seq)
                        column_data['I_median'].append(float(median_int) if median_int is not None else None)
                        column_data['Signal_Saturation'].append(float(sat_frac) if sat_frac is not None else None)
                        
                        # Add enrichment columns if present
                        for key in ['PepProtein_SeqMatch', 'PepProtein_UniprotName', 
                                   'PepProtein_UniprotID', 'entrezid', 'PepProtein_SeqSimilarity']:
                            if key not in column_data:
                                column_data[key] = []
                            column_data[key].append(enr_match.get(key))
                        
                        # Add other spot_data fields
                        for key, value in spot_data.items():
                            if key not in ['Col', 'Row', 'ID', 'Sequence'] and key not in enrichment_mapped:
                                if key not in column_data:
                                    column_data[key] = []
                                column_data[key].append(value)
                        
                        # Add intensity metrics
                        if 'Intensity' not in column_data:
                            column_data['Intensity'] = []
                        column_data['Intensity'].append(float(intensity))
                        
                        if max_int is not None:
                            if 'MaxIntensity' not in column_data:
                                column_data['MaxIntensity'] = []
                            column_data['MaxIntensity'].append(float(max_int))
                        
                        # Add other metadata fields
                        for field in image_metadata_fields:
                            if field in metadata_arrays and field not in ['PamChip_barcode', 'Array', 'Exposure_time', 'Pump_cycle']:
                                if field not in column_data:
                                    column_data[field] = []
                                val = metadata_arrays[field][image_idx]
                                if hasattr(val, 'item'):
                                    val = val.item()
                                column_data[field].append(val)
                        
                        # Add remaining enrichment fields with Enr_ prefix
                        for key, value in enr_match.items():
                            if key not in enrichment_mapped:
                                enr_key = f'Enr_{key}'
                                if enr_key not in column_data:
                                    column_data[enr_key] = []
                                column_data[enr_key].append(value)
                    
                    else:
                        # Multiple enrichment matches - expand rows
                        for enr_match in enrichment_matches:
                            column_data['image_idx'].append(image_idx)
                            column_data['Barcode'].append(metadata_arrays.get('PamChip_barcode', [None]*n_images)[image_idx])
                            column_data['Row'].append(metadata_arrays.get('Array', [None]*n_images)[image_idx])
                            column_data['spotRow'].append(spot_data.get('Col'))
                            column_data['spotCol'].append(spot_data.get('Row'))
                            column_data['Exposure Time'].append(metadata_arrays.get('Exposure_time', [None]*n_images)[image_idx])
                            column_data['Cycle'].append(metadata_arrays.get('Pump_cycle', [None]*n_images)[image_idx])
                            column_data['ID'].append(spot_data.get('ID'))
                            column_data['Sequence'].append(seq)
                            column_data['I_median'].append(float(median_int) if median_int is not None else None)
                            column_data['Signal_Saturation'].append(float(sat_frac) if sat_frac is not None else None)
                            
                            # Add enrichment columns
                            for key in ['PepProtein_SeqMatch', 'PepProtein_UniprotName', 
                                       'PepProtein_UniprotID', 'entrezid', 'PepProtein_SeqSimilarity']:
                                if key not in column_data:
                                    column_data[key] = []
                                column_data[key].append(enr_match.get(key))
                            
                            # Add other spot_data fields
                            for key, value in spot_data.items():
                                if key not in ['Col', 'Row', 'ID', 'Sequence'] and key not in enrichment_mapped:
                                    if key not in column_data:
                                        column_data[key] = []
                                    column_data[key].append(value)
                            
                            # Add intensity metrics
                            if 'Intensity' not in column_data:
                                column_data['Intensity'] = []
                            column_data['Intensity'].append(float(intensity))
                            
                            if max_int is not None:
                                if 'MaxIntensity' not in column_data:
                                    column_data['MaxIntensity'] = []
                                column_data['MaxIntensity'].append(float(max_int))
                            
                            # Add other metadata fields
                            for field in image_metadata_fields:
                                if field in metadata_arrays and field not in ['PamChip_barcode', 'Array', 'Exposure_time', 'Pump_cycle']:
                                    if field not in column_data:
                                        column_data[field] = []
                                    val = metadata_arrays[field][image_idx]
                                    if hasattr(val, 'item'):
                                        val = val.item()
                                    column_data[field].append(val)
                            
                            # Add remaining enrichment fields with Enr_ prefix
                            for key, value in enr_match.items():
                                if key not in enrichment_mapped:
                                    enr_key = f'Enr_{key}'
                                    if enr_key not in column_data:
                                        column_data[enr_key] = []
                                    column_data[enr_key].append(value)
        
        # Pad all columns to the same length (handle missing values)
        if verbose:
            print("  Creating DataFrame...")
        
        max_len = len(column_data['image_idx'])
        for key in column_data:
            if len(column_data[key]) < max_len:
                column_data[key].extend([None] * (max_len - len(column_data[key])))
        
        # OPTIMIZATION: Create DataFrame from dict of arrays (much faster than list of dicts)
        self.final_output = pd.DataFrame(column_data)
        
        # Reorder columns to ensure primary columns come first
        if verbose:
            print("  Reordering and sorting...")
        
        primary_col_order = ['image_idx', 'Barcode', 'Row', 'spotRow', 'spotCol', 
                            'Exposure Time', 'Cycle', 'ID', 'Sequence',
                            'PepProtein_SeqMatch', 'PepProtein_UniprotName', 
                            'PepProtein_UniprotID', 'entrezid', 'PepProtein_SeqSimilarity',
                            'I_median', 'Signal_Saturation']
        existing_primary = [c for c in primary_col_order if c in self.final_output.columns]
        other_cols = [c for c in self.final_output.columns if c not in primary_col_order]
        self.final_output = self.final_output[existing_primary + other_cols]
        
        # Sort rows: Barcode → Row → spotRow → spotCol → Exposure Time → Cycle
        # This creates hierarchical ordering where Cycle changes most frequently
        sort_columns = ['Barcode', 'Row', 'spotRow', 'spotCol', 'Exposure Time', 'Cycle']
        existing_sort_cols = [col for col in sort_columns if col in self.final_output.columns]
        if existing_sort_cols:
            self.final_output = self.final_output.sort_values(
                by=existing_sort_cols,
                ascending=True
            ).reset_index(drop=True)
        
        if verbose:
            print("Final output table built!")
            print(f"  Total rows: {len(self.final_output)}")
            print(f"  Columns: {len(self.final_output.columns)}")
            print(f"  Sorted by: {' → '.join(existing_sort_cols)}")
        
        # Create filtered version based on peptides type
        if self.peptides_type == 'STK':
            # For STK: keep only Cycle == STK_FILTER_CYCLE
            self.filtered_final_output = self.final_output[
                self.final_output['Cycle'] == self.STK_FILTER_CYCLE
            ].copy()
            if verbose:
                print(f"  Filtered (STK, Cycle={self.STK_FILTER_CYCLE}): {len(self.filtered_final_output)} rows")
        else:
            # For PTK: keep only specific Cycle values and Exposure Time values
            self.filtered_final_output = self.final_output[
                self.final_output['Cycle'].isin(self.PTK_FILTER_CYCLES) & 
                self.final_output['Exposure Time'].isin(self.PTK_FILTER_EXPOSURE_TIMES)
            ].copy()
            if verbose:
                print(f"  Filtered (PTK, {len(self.PTK_FILTER_CYCLES)} cycles, exposure times {self.PTK_FILTER_EXPOSURE_TIMES}): {len(self.filtered_final_output)} rows")
        
        return self.final_output
    
    def export_final_output(self, verbose: bool = True) -> Optional[Path]:
        """Export the final_output table and the downstream bn subset to CSV files.
        
        The files are saved to: results/experiment_name/subfolder_name/
        Uses timestamp from loader for consistent naming.
        
        Args:
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            Optional[Path]: Export final output.
        """
        
        # Use timestamp from loader for consistent naming
        filename = self.timestamp + '_Export_image_analysis_' + self.peptides_type + '.csv'
        
        if self.final_output is None:
            raise RuntimeError("No final output to export. Call build_final_output() first.")
        
        # Ensure filename ends with .csv
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = self.results_dir / filename
        
        # Export to CSV
        self.final_output.to_csv(output_path, index=False)
        
        if verbose:
            print(f"\nExported final output to: {output_path}")
            print(f"  Rows: {len(self.final_output)}")
            print(f"  Columns: {len(self.final_output.columns)}")
        
        # Keep the filtered version in memory for downstream processing, but do not save
        # it as a separate CSV file.
        if self.filtered_final_output is not None and len(self.filtered_final_output) > 0:
            if verbose:
                print("\nPrepared filtered output in memory for downstream processing")
                print(f"  Rows: {len(self.filtered_final_output)}")
                print(f"  Columns: {len(self.filtered_final_output.columns)}")
            
            # Export subset version with columns from 'Barcode' to 'Signal_Saturation'
            columns_to_include = [
                'Barcode', 'Row', 'spotRow', 'spotCol', 'Exposure Time', 'Cycle',
                'ID', 'Sequence', 'PepProtein_SeqMatch', 'PepProtein_UniprotName',
                'PepProtein_UniprotID', 'entrezid', 'PepProtein_SeqSimilarity',
                'I_median', 'Signal_Saturation'
            ]
            
            # Filter to only include columns that exist in the filtered output
            existing_columns = [col for col in columns_to_include if col in self.filtered_final_output.columns]
            
            if existing_columns:
                self.final_output_bn = self.filtered_final_output[existing_columns].copy()
                bn_filename = filename.replace('.csv', '_bn.csv')
                bn_output_path = self.results_dir / bn_filename
                
                self.final_output_bn.to_csv(bn_output_path, index=False)
                
                if verbose:
                    print(f"\nExported subset output to: {bn_output_path}")
                    print(f"  Rows: {len(self.final_output_bn)}")
                    print(f"  Columns: {len(self.final_output_bn.columns)}")
            
        elif verbose:
            if self.filtered_final_output is None:
                print("\nNo filtered output to export (filtered_final_output is None)")
            else:
                print("\nNo filtered output to export (0 rows after filtering)")
                print("  Check that data contains expected Cycle values:")
                if self.peptides_type == 'STK':
                    print(f"    STK expects: Cycle == {self.STK_FILTER_CYCLE}")
                else:
                    print(f"    PTK expects: Cycle in {self.PTK_FILTER_CYCLES}")
                    print(f"    PTK expects: Exposure Time in {self.PTK_FILTER_EXPOSURE_TIMES}")
                print(f"  Unique Cycle values in data: {sorted(self.final_output['Cycle'].unique())}")
        
        return output_path

    def process(
        self,
        layout_loader: Optional['ArrayLayoutLoader'] = None,
        verbose: bool = True,
        wiggle_radius: float = 2.0,
        integration_radius: float = 5.0
    ) -> pd.DataFrame:
        """Run the complete image processing pipeline sequentially.
        
        This method executes all processing steps in the correct order:
        1. Apply circular mask
        2. Subtract background
        3. Detect reference spots
        4. Refine reference spot positions
        5. Estimate reference angles
        6. Apply square mask
        7. Compute spot intensities
        8. Refine grid parameters
        9. Compute refined spot intensities
        10. Associate layout
        11. Build final output
        12. Export final output
        
        Args:
            layout_loader (Optional['ArrayLayoutLoader']): Layout loader processed by this function.
            verbose (bool): Whether to emit progress output while running.
            wiggle_radius (float): Wiggle radius used by this function.
            integration_radius (float): Integration radius used by this function.
        
        Returns:
            pd.DataFrame: Process.
        """
        # Use provided layout_loader or fall back to stored one
        if layout_loader is None:
            layout_loader = self._layout_loader
        
        if layout_loader is None:
            raise RuntimeError(
                "No layout_loader available. Either pass layout_loader to process() "
                "or ensure loader.layout_loader was set before creating the processor."
            )
        
        if verbose:
            print("\n" + "=" * 80)
            print("STARTING IMAGE PROCESSING PIPELINE")
            print("=" * 80)
        
        # Step 1: Subtract background (on original images)
        if verbose:
            print("\n[1/12] Subtracting background (from original images)...")
        self.subtract_background(use_masked_images=False, verbose=verbose)
        
        # Step 2: Apply circular mask (using estimated background for edge detection)
        if verbose:
            print("\n[2/12] Applying circular mask (detecting edges on estimated background)...")
        self.apply_circular_mask(use_estimated_background=True, verbose=verbose)
        
        # Step 3: Detect reference spots
        if verbose:
            print("\n[3/12] Detecting reference spots...")
        self.detect_reference_spots(verbose=verbose)
        
        # Step 4: Refine reference spot positions
        if verbose:
            print("\n[4/12] Refining reference spot positions...")
        self.refine_reference_spot_positions(
            wiggle_radius=wiggle_radius,
            integration_radius=integration_radius,
            verbose=verbose
        )
        
        # Step 5: Estimate reference angles
        if verbose:
            print("\n[5/12] Estimating reference angles...")
        self.estimate_reference_angles(verbose=verbose)
        
        # Step 6: Apply square mask
        if verbose:
            print("\n[6/12] Applying square mask...")
        self.apply_square_mask(verbose=verbose)
        
        # Step 7: Compute spot intensities
        if verbose:
            print("\n[7/12] Computing spot intensities...")
        self.compute_spot_intensities(
            integration_radius=integration_radius,
            verbose=verbose
        )
        
        # Step 8: Refine grid parameters
        if verbose:
            print("\n[8/12] Refining grid parameters...")
        self.refine_grid_parameters(verbose=verbose)
        
        # Step 9: Compute refined spot intensities
        if verbose:
            print("\n[9/12] Computing refined spot intensities...")
        self.compute_refined_spot_intensities(
            integration_radius=integration_radius,
            verbose=verbose
        )
        
        # Step 10: Associate layout (must be done after spot computations)
        if verbose:
            print("\n[10/12] Associating layout...")
        self.associate_layout(layout_loader)
        
        # Step 11: Build final output
        if verbose:
            print("\n[11/12] Building final output...")
        self.build_final_output(verbose=verbose)
        
        # Step 12: Export final output
        if verbose:
            print("\n[12/12] Exporting final output...")
        output_path = self.export_final_output(verbose=verbose)
        
        if verbose:
            print("\n" + "=" * 80)
            print("IMAGE PROCESSING PIPELINE COMPLETED")
            print("=" * 80)
            print(f"Final output saved to: {output_path}")
        
        return self.final_output

    def create_publication_figure(
        self,
        image_idx: int = 0,
        figsize: Tuple[int, int] = (16, 16),
        dpi: int = 300,
        save_images: bool = True,
        cross_section_type: str = 'horizontal',
        cross_section_offset: int = -80
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create a publication-quality figure for Bioinformatics journal.
        
        Creates a 2x2 panel figure with:
        - Panel (0,0): Background subtraction with cross-section and 2D inset
        - Panel (0,1): Reference spots detection with square mask overlay
        - Panel (1,0): Circular mask application result
        - Panel (1,1): Square mask with refined grid spots
        
        Args:
            image_idx (int): Zero-based index selecting the image.
            figsize (Tuple[int, int]): Figsize processed by this function.
            dpi (int): Dpi used by this function.
            save_images (bool): Save images used by this function.
            cross_section_type (str): Cross section type used by this function.
            cross_section_offset (int): Cross section offset used by this function.
        
        Returns:
            Tuple[plt.Figure, np.ndarray]: Created publication figure.
        """
        if self._square_masked_images is None or self._refined_grid_positions is None:
            raise RuntimeError("Complete processing pipeline must be run first. Call process() method.")
        
        # Get visualization directory if saving
        viz_dir = None
        if save_images:
            viz_dir = self._get_visualization_dir('publication_figure')
        
        # Create figure with 2x2 grid - tight spacing
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=-0.35, wspace=0.15, 
                             left=0.08, right=0.96, bottom=0.08, top=0.94)
        
        # Panel labels
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        
        # --- Panel (0,0): Background subtraction with cross-section ---
        ax_bg = fig.add_subplot(gs[0, 0])
        
        # Get data - use original unmasked image
        original = self.original_images.isel(image_idx=image_idx).values
        bg_stage2 = self._background_stage2_images.isel(image_idx=image_idx).values
        center_x, center_y = self._centers[image_idx]
        
        # Take cross-section through center with offset (matching background visualization)
        if cross_section_type == 'horizontal':
            y_pos = int(center_y + cross_section_offset)
            y_pos = max(0, min(y_pos, original.shape[0] - 1))  # Bounds check
            x_coords = np.arange(original.shape[1])
            original_cs = original[y_pos, :]
            bg2_cs = bg_stage2[y_pos, :]
            coord_label = 'X (pixels)'
        else:
            x_pos = int(center_x + cross_section_offset)
            x_pos = max(0, min(x_pos, original.shape[1] - 1))  # Bounds check
            x_coords = np.arange(original.shape[0])
            original_cs = original[:, x_pos]
            bg2_cs = bg_stage2[:, x_pos]
            coord_label = 'Y (pixels)'
        
        # Plot cross-sections - only final BG (Stage 2)
        ax_bg.plot(x_coords, original_cs, 'r-', linewidth=2.5, label='Original', alpha=0.8)
        ax_bg.plot(x_coords, bg2_cs, 'k--', linewidth=2.5, label='BG', alpha=0.7)
        ax_bg.fill_between(x_coords, bg2_cs, alpha=0.2, color='gray')
        ax_bg.set_xlabel('', fontsize=20, fontweight='bold')  # Remove X-axis label
        ax_bg.set_ylabel('Intensity (a.u.)', fontsize=20, fontweight='bold')
        ax_bg.legend(loc='upper right', fontsize=16, framealpha=0.9)
        ax_bg.grid(True, alpha=0.3)
        ax_bg.tick_params(labelsize=18)
        ax_bg.tick_params(axis='x', labelbottom=False)  # Remove X-axis tick labels
        # Match panel dimensions to image aspect ratio (height/width)
        img_height, img_width = original.shape
        ax_bg.set_box_aspect(img_height / img_width)
        ax_bg.text(-0.12, 1.05, panel_labels[0], transform=ax_bg.transAxes, 
                  fontsize=28, fontweight='bold', va='top')
        
        # Add inset showing 2D image with cross-section line (no log scale)
        ax_inset = ax_bg.inset_axes([0.02, 0.68, 0.28, 0.28])
        vmax = np.percentile(original[original > 0], 99) if np.any(original > 0) else original.max()
        ax_inset.imshow(original, cmap='gray', vmin=0, vmax=vmax)
        if cross_section_type == 'horizontal':
            ax_inset.axhline(y_pos, color='yellow', linewidth=2.5, linestyle='--')
        else:
            ax_inset.axvline(x_pos, color='yellow', linewidth=2.5, linestyle='--')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        
        # --- Panel (0,1): Reference spots with square mask ---
        ax_ref = fig.add_subplot(gs[0, 1])
        
        # Get unmasked background-subtracted image by reconstructing it
        original_full = self.original_images.isel(image_idx=image_idx).values
        bg_stage1_full = self._background_images.isel(image_idx=image_idx).values
        bg_subtracted_full = original_full.astype(np.float32) - bg_stage1_full.astype(np.float32)
        
        bg_disp = self._symmetric_log_transform(bg_subtracted_full)
        vmax = np.max(np.abs(bg_disp[bg_disp != 0])) if np.any(bg_disp != 0) else 1
        im = ax_ref.imshow(bg_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Plot reference spots in yellow
        if self._refined_reference_spots is not None:
            t_shape = self._refined_reference_spots['t_shape'][image_idx]
            j_shape = self._refined_reference_spots['j_shape'][image_idx]
            for shape, color in [(t_shape, 'yellow'), (j_shape, 'yellow')]:
                if shape is not None:
                    for x, y in shape:
                        circle = plt.Circle((x, y), 6, color=color, fill=False, linewidth=3)
                        ax_ref.add_patch(circle)
        
        # Draw square mask in green
        if self._square_mask_params is not None:
            mask_params = self._square_mask_params[image_idx]
            center_x_sq = mask_params['center_x']
            center_y_sq = mask_params['center_y']
            width = mask_params['right_x'] - mask_params['left_x']
            height = mask_params['bottom_y'] - mask_params['top_y']
            angle = mask_params['angle_rad']
            self._draw_rotated_rectangle(ax_ref, center_x_sq, center_y_sq, width, height, 
                                         angle, color='lime', linewidth=3.5)
        
        # Draw circular mask outline
        center_x_c, center_y_c = self._centers[image_idx]
        circle = plt.Circle((center_x_c, center_y_c), self.radius, color='black', 
                           fill=False, linewidth=3.5, linestyle='--')
        ax_ref.add_patch(circle)
        
        # Fix axis limits to image dimensions (prevent auto-scaling from patches)
        ax_ref.set_xlim(0, bg_subtracted_full.shape[1])
        ax_ref.set_ylim(bg_subtracted_full.shape[0], 0)  # Inverted for image coordinates
        
        ax_ref.set_xlabel('', fontsize=20, fontweight='bold')  # Remove X-axis label
        ax_ref.set_ylabel('Y (pixels)', fontsize=20, fontweight='bold')
        ax_ref.tick_params(labelsize=18)
        ax_ref.tick_params(axis='x', labelbottom=False)  # Remove X-axis tick labels
        ax_ref.text(-0.12, 1.05, panel_labels[1], transform=ax_ref.transAxes, 
                   fontsize=28, fontweight='bold', va='top')
        
        # --- Panel (1,0): Circular mask result ---
        ax_circ = fig.add_subplot(gs[1, 0])
        
        # Apply circular mask to background-subtracted image
        center_x_c, center_y_c = self._centers[image_idx]
        height_img, width_img = bg_subtracted_full.shape
        mask = self._create_circular_mask((height_img, width_img), (center_x_c, center_y_c), self.radius)
        masked_bg_sub = bg_subtracted_full.copy()
        masked_bg_sub[~mask] = 0
        
        masked_disp = self._symmetric_log_transform(masked_bg_sub)
        vmax = np.max(np.abs(masked_disp[masked_disp != 0])) if np.any(masked_disp != 0) else 1
        im = ax_circ.imshow(masked_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        ax_circ.set_xlabel('X (pixels)', fontsize=20, fontweight='bold')
        ax_circ.set_ylabel('Y (pixels)', fontsize=20, fontweight='bold')
        ax_circ.tick_params(labelsize=18)
        ax_circ.text(-0.12, 1.05, panel_labels[2], transform=ax_circ.transAxes, 
                    fontsize=28, fontweight='bold', va='top')
        
        # --- Panel (1,1): Square mask with refined grid spots ---
        ax_grid = fig.add_subplot(gs[1, 1])
        
        square_masked = self._square_masked_images.isel(image_idx=image_idx).values
        sq_disp = self._symmetric_log_transform(square_masked)
        vmax = np.max(np.abs(sq_disp[sq_disp != 0])) if np.any(sq_disp != 0) else 1
        im = ax_grid.imshow(sq_disp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Plot refined grid spots
        refined_positions = self._refined_grid_positions[image_idx]
        integration_radius = self._grid_refinement_params.get('integration_radius', 5.0) if hasattr(self, '_grid_refinement_params') else 5.0
        
        if refined_positions is not None:
            n_rows, n_cols = refined_positions.shape[:2]
            for row in range(n_rows):
                for col in range(n_cols):
                    x, y = refined_positions[row, col]
                    circle = plt.Circle((x, y), integration_radius, color='lime', fill=False, 
                                      linewidth=2, alpha=0.8)
                    ax_grid.add_patch(circle)
        
        ax_grid.set_xlabel('X (pixels)', fontsize=20, fontweight='bold')
        ax_grid.set_ylabel('', fontsize=20, fontweight='bold')  # Remove Y-axis label
        ax_grid.tick_params(labelsize=18)
        ax_grid.tick_params(axis='y', labelleft=False)  # Remove Y-axis tick labels
        
        # Add colorbar inside the axes (taller and further inside the plot)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_grid, width="3%", height="85%", loc='lower right', 
                        bbox_to_anchor=(-0.04, 0.08, 1, 1), bbox_transform=ax_grid.transAxes, 
                        borderpad=0)
        cbar = plt.colorbar(im, cax=cax, label='Intensity (log)')
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Intensity (log)', fontsize=16)
        # Move ticks and labels to left side of colorbar
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        
        ax_grid.text(-0.12, 1.05, panel_labels[3], transform=ax_grid.transAxes, 
                    fontsize=28, fontweight='bold', va='top')
        
        # Remove top y-axis tick labels from panels (a), (b), and (c)
        # Force a complete draw to finalize tick locations before modifying them
        fig.canvas.draw()
        
        # Strategy: Set both yticks and yticklabels for persistence, 
        # but preserve y-limits for image panels to prevent expansion
        
        # Panel (a) - remove the highest/last tick label
        ytick_locs_a = ax_bg.get_yticks()
        ytick_labels_a = [f'{y:.0f}' for y in ytick_locs_a]
        if len(ytick_labels_a) > 0:
            ytick_labels_a[-1] = ''  # Remove last (top) label
        ax_bg.set_yticks(ytick_locs_a)
        ax_bg.set_yticklabels(ytick_labels_a)
        
        # Panel (b) - remove the "0" tick label (at top for images), preserve y-limits
        ylim_b = ax_ref.get_ylim()  # Save current limits
        ytick_locs_b = ax_ref.get_yticks()
        ytick_labels_b = []
        for y in ytick_locs_b:
            if abs(y - 0.0) < 0.1:  # This is the "0" tick
                ytick_labels_b.append('')
            else:
                ytick_labels_b.append(f'{y:.0f}')
        ax_ref.set_yticks(ytick_locs_b)
        ax_ref.set_yticklabels(ytick_labels_b)
        ax_ref.set_ylim(ylim_b)  # Restore original limits
        
        # Panel (c) - remove the "0" tick label (at top for images), preserve y-limits
        ylim_c = ax_circ.get_ylim()  # Save current limits
        ytick_locs_c = ax_circ.get_yticks()
        ytick_labels_c = []
        for y in ytick_locs_c:
            if abs(y - 0.0) < 0.1:  # This is the "0" tick
                ytick_labels_c.append('')
            else:
                ytick_labels_c.append(f'{y:.0f}')
        ax_circ.set_yticks(ytick_locs_c)
        ax_circ.set_yticklabels(ytick_labels_c)
        ax_circ.set_ylim(ylim_c)  # Restore original limits
        
        # Save figure if requested
        if save_images and viz_dir is not None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"{timestamp}_publication_figure_img_idx_{image_idx:03d}.png"
            filepath = viz_dir / filename
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Publication figure saved to: {filepath}")
            
            # Also save as high-quality PDF for publication
            pdf_filename = f"{timestamp}_publication_figure_img_idx_{image_idx:03d}.pdf"
            pdf_filepath = viz_dir / pdf_filename
            fig.savefig(pdf_filepath, dpi=dpi, bbox_inches='tight', format='pdf')
            print(f"Publication figure (PDF) saved to: {pdf_filepath}")
        
        return fig, fig.axes

    def visualize_processing_stages(
        self,
        image_idx: int = 0,
        save_images: bool = False,
        show_spot_grid: bool = True
    ) -> None:
        """Visualize all major processing stages for a single image.
        
        This convenience method displays:
        1. Circular mask detection
        2. Background subtraction
        3. Refined reference spots
        4. Square mask with optional spot grid
        5. Grid refinement
        
        Args:
            image_idx (int): Zero-based index selecting the image.
            save_images (bool): Save images used by this function.
            show_spot_grid (bool): Show spot grid used by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        # 1. Circular mask detection
        self.visualize_circular_mask_detection(image_idx=image_idx, save_images=save_images)
        
        # 2. Background subtraction
        self.visualize_background_subtraction(image_idx=image_idx, save_images=save_images)
        
        # 3. Refined reference spots
        self.visualize_refined_reference_spots(image_idx=image_idx, save_images=save_images)
        
        # 4. Square mask with spot grid
        self.visualize_square_mask(
            image_idx=image_idx,
            show_spot_grid=show_spot_grid,
            save_images=save_images
        )
        
        # 5. Grid refinement
        self.visualize_grid_refinement(image_idx=image_idx, save_images=save_images)

    def __repr__(self):
        """Return a concise string representation for debugging.
        
        Args:
            None.
        
        Returns:
            object: Repr.
        """
        processed = self._masked_images is not None
        bg_sub = self._background_subtracted_images is not None
        ref_spots = self._reference_spots is not None
        refined = self._refined_reference_spots is not None if hasattr(self, '_refined_reference_spots') else False
        angles = self._reference_angles is not None if hasattr(self, '_reference_angles') else False
        sq_masked = self._square_masked_images is not None if hasattr(self, '_square_masked_images') else False
        spot_int = self._spot_intensities is not None
        grid_refined = self._refined_grid_positions is not None
        layout_loaded = self._spot_layout is not None
        enrichment_loaded = self._enrichment_df is not None
        final_output_built = self.final_output is not None
        filtered_output_built = self.filtered_final_output is not None
        
        return (f"ImageProcessor(n_images={len(self.original_images.image_idx)}, "
                f"type='{self.peptides_type}', radius={self.radius}, "
                f"masked={processed}, bg_sub={bg_sub}, ref_spots={ref_spots}, "
                f"refined={refined}, angles={angles}, sq_masked={sq_masked}, "
                f"spot_int={spot_int}, grid_refined={grid_refined}, "
                f"layout={layout_loaded}, enrichment={enrichment_loaded}, "
                f"final_output={final_output_built}, "
                f"filtered_output={filtered_output_built})")


if __name__ == '__main__':
    print()
