"""
DataLoader - A class for loading and visualizing experimental data.

TO DO:
    Loading multiple runs (can be implemented, for example, as a function external 
    to DataLoader that creates several instances of DataLoader depedning on the number 
    of runs, and stores them in a dictionary).
"""

# Standard library imports
import re
import os
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local imports
from config.data_importer import (
    ARRAY_LAYOUT_LOADER_DEFAULTS,
    DATA_LOADER_DEFAULTS,
)


def _candidate_paths(path_values):
    """Expand configured candidate data directory strings into ``Path`` objects.
    
    Args:
        path_values: Path to the values.
    
    Returns:
        object: Candidate paths.
    """
    return [Path(path_value).expanduser() for path_value in path_values]


class DataLoader:
    """
    A class for loading and managing experimentally measured PamChip microarray data.
    
    This class provides the following functionality:
    1. Navigates and selects experiment folders and subfolders interactively
    2. Imports sample annotation information from experimental folders
    3. Loads array layout with information about peptide sequences, IDs, etc.
    4. Loads TIFF images from the experimental folder into an xarray DataArray
    5. Visualizes imported TIFF images
    
    Expected TIFF filename pattern: 
        {PamChip_barcode}_W{Array}_F{FoV}_T{Exposure_time}_P{Pump_cycle}_I{Image_number}_A{Temperature}.tif
    Example: 
        640208604_W1_F1_T10_P32_I1_A30.tif
    
    Attributes:
        base_data_dir (Path): Path to the base data directory ('../data')
        experiment_dir (Path): Path to the selected experiment directory ('../data/Experimental_data')
        experiment_name (str): Name of the experiment folder ('Experimental_data')
        data_dir (Path): Path to the specific data subfolder containing experimental run data
                                                                (e.g., '../data/Experimental_data/640208604_640208605-on 1200PTKlysv04-run <YYMMDDHHMMSS>', 
                                                                where <YYMMDDHHMMSS> is the date of the experiment). 
        subfolder_name (str): Name of the data subfolder (e.g., '640208604_640208605-on 1200PTKlysv04-run <YYMMDDHHMMSS>')
        image_results_dir (Path): Path to the ImageResults directory containing TIFF files
        peptides_type (str): Type of peptides - 'PTK' (Protein Tyrosine Kinase) or 
                           'STK' (Serine/Threonine Kinase) extracted from folder name
        timestamp (str): Timestamp for consistent naming across analysis sessions (format: YYYYMMDDHHMMSS)
        sample_annotation (pd.DataFrame): Loaded sample annotation table with chip locations, 
                                         barcodes, and sample information
        images (xr.DataArray): xarray DataArray containing all loaded TIFF images with metadata
                              coordinates (dimensions: image_idx, y, x)
        image_metadata (pd.DataFrame): DataFrame with parsed metadata for all loaded images
        layout_loader (ArrayLayoutLoader): Instance of ArrayLayoutLoader with loaded array layout
        spot_layout (xr.DataArray): 2D array (grid_row, grid_col) containing peptide information 
                                   (ID, Sequence, Position, etc.) for each spot on the PamChip
    
    Class Attributes:
        FILENAME_PATTERN (re.Pattern): Compiled regex pattern for parsing TIFF filenames with 
                                      7 metadata fields (barcode, array, FoV, exposure, pump cycle, 
                                      image number, temperature)
        COORDINATE_NAMES (List[str]): List of coordinate names corresponding to regex groups:
                                     ['PamChip_barcode', 'Array', 'FoV', 'Exposure_time', 
                                      'Pump_cycle', 'Image_number', 'Temperature']
    """
    
    # Regex pattern to extract metadata from TIFF filenames
    FILENAME_PATTERN = DATA_LOADER_DEFAULTS["filename_pattern"]
    
    # Mapping of regex groups to coordinate names
    COORDINATE_NAMES = list(DATA_LOADER_DEFAULTS["coordinate_names"])
    
    # Directory and file naming constants
    IMAGE_RESULTS_DIRNAME = DATA_LOADER_DEFAULTS["image_results_dirname"]
    SAMPLE_ANNOTATION_PATTERN = DATA_LOADER_DEFAULTS["sample_annotation_pattern"]
    
    # Folder validation constants
    FOLDER_DATETIME_PATTERN = DATA_LOADER_DEFAULTS["folder_datetime_pattern"]
    FOLDER_RUN_KEYWORD = DATA_LOADER_DEFAULTS["folder_run_keyword"]
    VALID_PEPTIDE_TYPES = list(DATA_LOADER_DEFAULTS["valid_peptide_types"])
    
    # Timestamp format for consistent naming
    TIMESTAMP_FORMAT = DATA_LOADER_DEFAULTS["timestamp_format"]
    
    # Progress reporting interval
    PROGRESS_INTERVAL = int(DATA_LOADER_DEFAULTS["progress_interval"])
    
    # Sample annotation display columns
    ANNOTATION_ALWAYS_DISPLAY = list(DATA_LOADER_DEFAULTS["annotation_always_display"])
    ANNOTATION_CONDITIONAL_DISPLAY = list(
        DATA_LOADER_DEFAULTS["annotation_conditional_display"]
    )
    ANNOTATION_EXCLUDE_COLUMNS = list(DATA_LOADER_DEFAULTS["annotation_exclude_columns"])
    
    def __init__(
        self, 
        data_dir: Union[str, Path] = None,
        experiment_name: Optional[str] = None,
        subfolder: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        """Initializes the DataLoader.
        
        Args:
            data_dir (Union[str, Path]): Directory containing or receiving the data.
            experiment_name (Optional[str]): Experiment name processed by this function.
            subfolder (Optional[str]): Subfolder processed by this function.
            timestamp (Optional[str]): Timestamp string associated with the current analysis run.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        
        # Try multiple possible data directory paths
        if data_dir is None:
            possible_paths = _candidate_paths(
                DATA_LOADER_DEFAULTS["candidate_data_paths"]
            )
            
            for path in possible_paths:
                if path.exists() and path.is_dir():
                    data_dir = path
                    print(f"Auto-detected data directory: {data_dir}")
                    break
                
        if data_dir is None:
            raise FileNotFoundError(
                "Could not find data directory. Tried:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\n\nPlease specify data_dir explicitly."
            )
            
        self.base_data_dir = Path(data_dir)
        
        # Validate that base data directory exists
        if not self.base_data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.base_data_dir}"
            )
        
        # Select or use provided experiment folder
        if experiment_name is None:
            self.experiment_dir = self._select_experiment_folder()
            self.experiment_name = self.experiment_dir.name
        else:
            # When experiment_name is provided, need to find it in the valid folders map
            valid_folders_map = self._find_all_valid_data_folders()
            
            # Try to find the experiment in the valid folders
            matched_key = None
            for parent_key in valid_folders_map.keys():
                # Check if experiment_name matches the parent key or is contained in it
                if experiment_name == parent_key or parent_key.endswith(experiment_name):
                    matched_key = parent_key
                    break
            
            if matched_key is None:
                raise FileNotFoundError(
                    f"Experiment '{experiment_name}' not found in valid folders.\\n"
                    f"Available experiments: {', '.join(sorted(valid_folders_map.keys()))}"
                )
            
            self.experiment_name = experiment_name
            self._valid_folders_map = valid_folders_map
            
            # Determine the experiment directory path
            if matched_key == str(self.base_data_dir.name):
                self.experiment_dir = self.base_data_dir
            else:
                self.experiment_dir = self.base_data_dir / matched_key
        
        # Select or use provided subfolder
        if subfolder is not None:
            self.data_dir = self.experiment_dir / subfolder
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Subfolder not found: {self.data_dir}")
            
            # Validate that it matches the expected pattern
            if not self._is_valid_image_folder(subfolder):
                print(f"Warning: Subfolder '{subfolder}' does not match expected pattern "
                      f"(space + date-time, 'run', and 'PTK'/'STK').")
        else:
            # Interactive selection
            self.data_dir = self._select_data_folder()
        
        self.subfolder_name = self.data_dir.name
        self.image_results_dir = self.data_dir / self.IMAGE_RESULTS_DIRNAME
        
        # Extract peptides_type (PTK or STK) from subfolder name
        self.peptides_type = self._extract_peptides_type(self.subfolder_name)
        if self.peptides_type:
            print(f"Peptides type: {self.peptides_type}")
        
        self._sample_annotation: Optional[pd.DataFrame] = None
        self._images: Optional[xr.DataArray] = None
        self._image_metadata: Optional[pd.DataFrame] = None
        
        # Create or use provided timestamp for consistent naming across all related classes
        if timestamp is None:
            self.timestamp = datetime.now().strftime(self.TIMESTAMP_FORMAT)
        else:
            self.timestamp = timestamp
        
        self._validate_directory_structure()
        
        self._layout_loader: Optional['ArrayLayoutLoader'] = None
    
    def _extract_peptides_type(self, folder_name: str) -> Optional[str]:
        """Extract peptides type (PTK or STK) from folder name.
        
        Args:
            folder_name (str): Folder name used by this function.
        
        Returns:
            Optional[str]: Extracted peptides type.
        """
        # Look for PTK or STK in the folder name (case-insensitive)
        folder_upper = folder_name.upper()
        
        for peptide_type in self.VALID_PEPTIDE_TYPES:
            if peptide_type in folder_upper:
                return peptide_type
        
        print(f"Warning: Could not identify peptides_type ({'/'.join(self.VALID_PEPTIDE_TYPES)}) from folder name: {folder_name}")
        return None
    
    def _is_valid_image_folder(self, folder_name: str) -> bool:
        """Check if a folder name matches the pattern for image folders.
        
        Criteria:
        1. Ends with ' YYMMDDHHMMSS' (space followed by 12-digit date-time)
        2. Contains 'run' (case-insensitive)
        3. Contains either 'PTK' or 'STK' (case-insensitive)
        
        Args:
            folder_name (str): Folder name used by this function.
        
        Returns:
            bool: Is valid image folder.
        """
        folder_upper = folder_name.upper()
        
        # Criterion 1: Ends with space followed by 12-digit date-time pattern
        has_datetime = bool(self.FOLDER_DATETIME_PATTERN.search(folder_name))
        
        # Criterion 2: Contains 'run'
        has_run = self.FOLDER_RUN_KEYWORD in folder_upper
        
        # Criterion 3: Contains either 'PTK' or 'STK'
        has_peptide_type = any(peptide in folder_upper for peptide in self.VALID_PEPTIDE_TYPES)
        
        return has_datetime and has_run and has_peptide_type
    
    def _find_all_valid_data_folders(self) -> Dict[str, List[Path]]:
        """Recursively search data directory for valid image folders.
        
        Args:
            None.
        
        Returns:
            Dict[str, List[Path]]: Found all valid data folders.
        """
        valid_folders: Dict[str, List[Path]] = {}
        
        # Recursively walk through the data directory
        for root, dirs, files in os.walk(self.base_data_dir):
            root_path = Path(root)
            
            # Check each subdirectory
            for dir_name in dirs:
                if self._is_valid_image_folder(dir_name):
                    dir_path = root_path / dir_name
                    
                    # Get the relative path from base_data_dir to the parent of this valid folder
                    relative_parent = root_path.relative_to(self.base_data_dir)
                    
                    # Use just the immediate parent folder name as the key
                    parent_key = str(relative_parent) if str(relative_parent) != '.' else root_path.name
                    
                    if parent_key not in valid_folders:
                        valid_folders[parent_key] = []
                    valid_folders[parent_key].append(dir_path)
        
        return valid_folders
        
    def _select_experiment_folder(self) -> Path:
        """Interactively select an experiment folder from folders containing valid image subfolders.
        
        Args:
            None.
        
        Returns:
            Path: Selected experiment folder.
        """
        # Find all valid data folders organized by parent
        valid_folders_map = self._find_all_valid_data_folders()
        
        if not valid_folders_map:
            raise FileNotFoundError(
                f"No folders with valid image subfolders found in {self.base_data_dir}.\n"
                f"Expected subfolder names containing: space + 12-digit date-time, '{self.FOLDER_RUN_KEYWORD.lower()}', and {' or '.join(self.VALID_PEPTIDE_TYPES)}."
            )
        
        # Sort parent folders for consistent display
        parent_folders = sorted(valid_folders_map.keys())
        
        # Store the mapping for later use
        self._valid_folders_map = valid_folders_map
        
        # Display available parent folders
        print("\n" + "=" * 60)
        print("Available experiment folders (containing image data):")
        print("=" * 60)
        for i, parent_name in enumerate(parent_folders, start=1):
            num_subfolders = len(valid_folders_map[parent_name])
            # Add spaces around slashes for better readability
            display_name = parent_name.replace('/', ' / ')
            print(f"  {i}. {display_name} ({num_subfolders} data folder{'s' if num_subfolders > 1 else ''})")
        print("=" * 60)
        
        # Get user selection
        while True:
            try:
                user_input = input(f"\nEnter experiment number (1-{len(parent_folders)}): ").strip()
                selection = int(user_input)
                
                if 1 <= selection <= len(parent_folders):
                    selected_parent_name = parent_folders[selection - 1]
                    print(f"\nSelected experiment: {selected_parent_name}")
                    # Return the actual directory path
                    if selected_parent_name == str(self.base_data_dir.name):
                        return self.base_data_dir
                    else:
                        return self.base_data_dir / selected_parent_name
                else:
                    print(f"Please enter a number between 1 and {len(parent_folders)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                raise
                
    def _select_data_folder(self) -> Path:
        """Interactively select a data folder from valid subfolders in the experiment directory.
        
        Args:
            None.
        
        Returns:
            Path: Selected data folder.
        """
        # Get the relative path key for this experiment
        experiment_relative = self.experiment_dir.relative_to(self.base_data_dir)
        parent_key = str(experiment_relative) if str(experiment_relative) != '.' else self.experiment_dir.name
        
        # Get valid folders for this experiment from the cached map
        if hasattr(self, '_valid_folders_map') and parent_key in self._valid_folders_map:
            run_folders = sorted(self._valid_folders_map[parent_key])
        else:
            # Fallback: re-search if map not available
            valid_folders_map = self._find_all_valid_data_folders()
            run_folders = sorted(valid_folders_map.get(parent_key, []))
        
        if not run_folders:
            raise FileNotFoundError(
                f"No valid data folders found in {self.experiment_dir}.\n"
                "Expected folders matching pattern: space + 12-digit date-time, 'run', and 'PTK' or 'STK'."
            )
        
        # Display available folders
        print("\n" + "=" * 60)
        print(f"Available data folders in '{self.experiment_dir.name}':")
        print("=" * 60)
        for i, folder in enumerate(run_folders, start=1):
            print(f"  {i}. {folder.name}")
        print("=" * 60)
        
        # Get user selection
        while True:
            try:
                user_input = input(f"\nEnter folder number (1-{len(run_folders)}): ").strip()
                selection = int(user_input)
                
                if 1 <= selection <= len(run_folders):
                    selected_folder = run_folders[selection - 1]
                    print(f"\nSelected: {selected_folder.name}")
                    return selected_folder
                else:
                    print(f"Please enter a number between 1 and {len(run_folders)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                raise
    
    def _validate_directory_structure(self) -> None:
        """Validate that the expected directory structure exists.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if not self.image_results_dir.exists():
            raise FileNotFoundError(
                f"ImageResults directory not found: {self.image_results_dir}"
            )
    
    @property
    def sample_annotation(self) -> Optional[pd.DataFrame]:
        """Get the loaded sample annotation DataFrame.
        
        Args:
            None.
        
        Returns:
            Optional[pd.DataFrame]: Stored sample annotation.
        """
        return self._sample_annotation
    
    @property
    def images(self) -> Optional[xr.DataArray]:
        """Get the loaded images as an xarray DataArray.
        
        Args:
            None.
        
        Returns:
            Optional[xr.DataArray]: Stored images.
        """
        return self._images
    
    @property
    def image_metadata(self) -> Optional[pd.DataFrame]:
        """Get metadata DataFrame for all loaded images.
        
        Args:
            None.
        
        Returns:
            Optional[pd.DataFrame]: Stored image metadata.
        """
        return self._image_metadata
    
    def _find_sample_annotation_file(self) -> Path:
        """Find the Sample Annotation text file in the data directory.
        
        Args:
            None.
        
        Returns:
            Path: Found sample annotation file.
        """
        pattern = self.SAMPLE_ANNOTATION_PATTERN
        matches = list(self.data_dir.glob(pattern))
        
        if not matches:
            # Try case-insensitive search
            all_txt_files = list(self.data_dir.glob('*.txt'))
            matches = [
                f for f in all_txt_files 
                if 'sample annotation' in f.name.lower()
            ]
        
        if not matches:
            raise FileNotFoundError(
                f"No Sample Annotation file found in {self.data_dir}"
            )
        
        if len(matches) > 1:
            print(f"Warning: Multiple Sample Annotation files found. Using: {matches[0]}")
        
        return matches[0]
    
    def load_sample_annotation(
        self, 
        sep: str = '\t',
        verbose: bool = True,
        **pandas_kwargs
    ) -> pd.DataFrame:
        """Load the sample annotation table from the text file.
        
        Args:
            sep (str): Sep used by this function.
            verbose (bool): Whether to emit progress output while running.
            **pandas_kwargs: Additional keyword arguments forwarded by this function.
        
        Returns:
            pd.DataFrame: Loaded sample annotation.
        """
        annotation_file = self._find_sample_annotation_file()
        
        try:
            self._sample_annotation = pd.read_csv(
                annotation_file, 
                sep=sep,
                index_col=False,
                **pandas_kwargs
            )
        except Exception as e:
            raise IOError(f"Failed to load sample annotation: {e}")
        
        if verbose:
            print(f"Loaded sample annotation from: {annotation_file}")
            print(f"Shape: {self._sample_annotation.shape}")
        
        # Columns to always display if they exist
        always_display = self.ANNOTATION_ALWAYS_DISPLAY
        
        # Columns to display only if they exist AND have non-zero values
        conditional_display = self.ANNOTATION_CONDITIONAL_DISPLAY
        
        # Check which "always display" columns are present
        available_columns = [col for col in always_display if col in self._sample_annotation.columns]
        
        # Helper function to check if column has non-zero integer-like values
        def has_nonzero_integers(col_name):
            """Return whether nonzero integers.
            
            Args:
                col_name: Col name processed by this function.
            
            Returns:
                bool: True when nonzero integers is satisfied, otherwise False.
            """
            if col_name not in self._sample_annotation.columns:
                return False
            try:
                # Check if column contains numeric data
                if pd.api.types.is_numeric_dtype(self._sample_annotation[col_name]):
                    non_null_values = self._sample_annotation[col_name].dropna()
                    if len(non_null_values) > 0:
                        # Check if values are integer-like (handles both int and float like 1.0)
                        # For floats like 1.0, val == int(val) will be True
                        if all(val == int(val) for val in non_null_values):
                            # Check if at least one is non-zero
                            if any(int(val) != 0 for val in non_null_values):
                                return True
            except (TypeError, ValueError):
                pass
            return False
        
        # Add conditional columns if they have non-zero values
        for col in conditional_display:
            if has_nonzero_integers(col):
                available_columns.append(col)
        
        # Find additional columns with non-zero integer values (except excluded columns)
        exclude_columns = set(always_display + conditional_display + self.ANNOTATION_EXCLUDE_COLUMNS)
        for col in self._sample_annotation.columns:
            if col not in exclude_columns and col not in available_columns:
                if has_nonzero_integers(col):
                    available_columns.append(col)
        
        if verbose:
            if available_columns:
                print("\nSample Annotation Details:")
                print("=" * 80)
                # Set display options to show all rows
                with pd.option_context('display.max_rows', None, 'display.width', None):
                    print(self._sample_annotation[available_columns].to_string(index=False))
                print("=" * 80)
            else:
                print("\nWarning: None of the requested columns found in sample annotation file.")
                print(f"Available columns: {list(self._sample_annotation.columns)}")
        
        return self._sample_annotation
    
    def _parse_filename(self, filename: str) -> Optional[Dict[str, int]]:
        """Parse a TIFF filename to extract metadata.
        
        Args:
            filename (str): Filename used by this function.
        
        Returns:
            Optional[Dict[str, int]]: Parse filename.
        """
        match = self.FILENAME_PATTERN.match(filename)
        
        if not match:
            return None
        
        return {
            name: int(value) 
            for name, value in zip(self.COORDINATE_NAMES, match.groups())
        }
    
    def _get_tiff_files(self) -> List[Tuple[Path, Dict[str, int]]]:
        """Get all TIFF files and their parsed metadata.
        
        Args:
            None.
        
        Returns:
            List[Tuple[Path, Dict[str, int]]]: Requested tiff files.
        """
        tiff_files = []
        
        for tiff_path in self.image_results_dir.glob('*.tif'):
            metadata = self._parse_filename(tiff_path.name)
            if metadata is not None:
                tiff_files.append((tiff_path, metadata))
            else:
                print(f"Warning: Could not parse filename: {tiff_path.name}")
        
        # Also check for .tiff extension
        for tiff_path in self.image_results_dir.glob('*.tiff'):
            metadata = self._parse_filename(tiff_path.name.replace('.tiff', '.tif'))
            if metadata is not None:
                tiff_files.append((tiff_path, metadata))
        
        return tiff_files
    
    def _find_array_layout_file(self) -> Path:
        """Find the Array Layout text file in the data directory.
        
        Args:
            None.
        
        Returns:
            Path: Found array layout file.
        """
        # Look for files ending with "Array Layout.txt"
        matches = [f for f in self.data_dir.glob('*Array Layout.txt')]
        
        if not matches:
            # Try case-insensitive search
            all_txt_files = list(self.data_dir.glob('*.txt'))
            matches = [
                f for f in all_txt_files 
                if f.name.lower().endswith('array layout.txt')
            ]
        
        if not matches:
            raise FileNotFoundError(
                f"No Array Layout file found in {self.data_dir}"
            )
        
        if len(matches) > 1:
            print(f"Warning: Multiple Array Layout files found. Using: {matches[0]}")
        
        return matches[0]
    
    def load_array_layout(self, verbose: bool = True) -> 'ArrayLayoutLoader':
        """Load the array layout from the Array Layout text file.
        
        Uses ArrayLayoutLoader to parse the file and create a structured
        spot array with peptide information (ID, Sequence, etc.).
        
        Args:
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            'ArrayLayoutLoader': Loaded array layout.
        """
        layout_file = self._find_array_layout_file()
        
        # Create loader and load the file
        # Pass just the filename since data_dir is already the correct directory
        self._layout_loader = ArrayLayoutLoader(data_dir=self.data_dir)
        self._layout_loader.load(layout_file.name)  # <-- Changed from str(layout_file)
        
        if verbose:
            # Validate chip type matches
            if self._layout_loader.chip_type != self.peptides_type:
                print(f"Warning: Layout chip type '{self._layout_loader.chip_type}' "
                      f"differs from folder-derived type '{self.peptides_type}'")
            
            print(f"Loaded array layout from: {layout_file.name}")
            print(f"  Chip type: {self._layout_loader.chip_type}")
            print(f"  Grid: {self._layout_loader.grid_dimensions[0]}×{self._layout_loader.grid_dimensions[1]}")
            print(f"  Spots: {self._layout_loader.spot_array.attrs['n_spots']}")
        
        return self._layout_loader
    
    @property
    def layout_loader(self) -> Optional['ArrayLayoutLoader']:
        """Get the loaded ArrayLayoutLoader instance.
        
        Args:
            None.
        
        Returns:
            Optional['ArrayLayoutLoader']: Stored layout loader.
        """
        return getattr(self, '_layout_loader', None)
    
    @property
    def spot_layout(self) -> Optional[xr.DataArray]:
        """Get the spot layout array (xarray DataArray of dictionaries).
        
        Args:
            None.
        
        Returns:
            Optional[xr.DataArray]: Stored spot layout.
        """
        if hasattr(self, '_layout_loader') and self._layout_loader is not None:
            return self._layout_loader.spot_array
        return None
    
    def load_images(
        self, 
        dtype: Optional[np.dtype] = None,
        verbose: bool = True
    ) -> xr.DataArray:
        """Load all TIFF images into an xarray DataArray.
        
        The images are organized with coordinates:
        - PamChip_barcode: Unique barcode identifier
        - Array: Array number (W)
        - FoV: Field of View (F)
        - Exposure_time: Exposure time in ms (T)
        - Pump_cycle: Pump cycle number (P)
        - Image_number: Image number within sequence (I)
        - Temperature: Temperature in Celsius (A)
        
        Args:
            dtype (Optional[np.dtype]): Dtype processed by this function.
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            xr.DataArray: Loaded images.
        """
        tiff_files = self._get_tiff_files()
        
        if not tiff_files:
            raise FileNotFoundError(
                f"No valid TIFF files found in {self.image_results_dir}"
            )
        
        if verbose:
            print(f"Found {len(tiff_files)} TIFF images to load...")
        
        # Load first image to get dimensions
        first_image = np.array(Image.open(tiff_files[0][0]))
        img_height, img_width = first_image.shape[:2]
        
        if dtype is None:
            dtype = first_image.dtype
        
        # Prepare coordinate arrays
        coords_data = {name: [] for name in self.COORDINATE_NAMES}
        file_paths = []
        
        # Collect all metadata from filenames
        for file_path, metadata in tiff_files:
            for name in self.COORDINATE_NAMES:
                coords_data[name].append(metadata[name])
            file_paths.append(str(file_path))
        
        n_images = len(tiff_files)
        
        # Load images
        if verbose:
            print("Loading image data...")
        images_array = np.zeros(
            (n_images, img_height, img_width), 
            dtype=dtype
        )
        
        for i, (file_path, _) in enumerate(tiff_files):
            if verbose and ((i + 1) % self.PROGRESS_INTERVAL == 0 or i == 0):
                print(f"Loading image {i + 1}/{n_images}...")
            images_array[i] = np.array(Image.open(file_path))
        
        # Build coordinates dictionary
        all_coords = {
            'image_idx': np.arange(n_images),
            'y': np.arange(img_height),
            'x': np.arange(img_width),
            **{name: ('image_idx', values) for name, values in coords_data.items()},
            'file_path': ('image_idx', file_paths)
        }
        
        # Create xarray DataArray
        self._images = xr.DataArray(
            data=images_array,
            dims=['image_idx', 'y', 'x'],
            coords=all_coords,
            attrs={
                'description': 'PamChip microarray images',
                'source_directory': str(self.image_results_dir),
                'n_images': n_images,
                'image_shape': (img_height, img_width)
            }
        )
        
        # Store filename metadata as DataFrame for convenience
        self._image_metadata = pd.DataFrame(coords_data)
        self._image_metadata['file_path'] = file_paths
        
        print(f"Loaded {n_images} images with shape {img_height}x{img_width}")
        print(f"DataArray dimensions: {dict(self._images.sizes)}")
        
        return self._images
    
    def load_data(
        self,
        annotation_verbose: bool = True,
        layout_verbose: bool = True,
        images_verbose: bool = True,
        dtype: Optional[np.dtype] = None
    ) -> None:
        """Load all data components sequentially (sample annotation, array layout, and images).
        
        This convenience method runs the complete data loading pipeline:
        1. Load sample annotation from text file
        2. Load array layout with peptide information
        3. Load all TIFF images with metadata
        
        Args:
            annotation_verbose (bool): Annotation verbose used by this function.
            layout_verbose (bool): Layout verbose used by this function.
            images_verbose (bool): Images verbose used by this function.
            dtype (Optional[np.dtype]): Dtype processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Step 1: Load sample annotation
        print("\n[1/3] Loading sample annotation...")
        self.load_sample_annotation(verbose=annotation_verbose)
        
        # Step 2: Load array layout
        print("\n[2/3] Loading array layout...")
        self.load_array_layout(verbose=layout_verbose)
        
        # Step 3: Load images
        print("\n[3/3] Loading images...")
        self.load_images(dtype=dtype, verbose=images_verbose)
        
        print("\n" + "=" * 80)
        print("DATA LOADING COMPLETED")
        print("=" * 80)
    
    def get_image_by_coords(
        self,
        pamchip_barcode: int,
        array: int,
        fov: int,
        exposure_time: int,
        pump_cycle: int,
        image_number: int,
        temperature: int
    ) -> xr.DataArray:
        """Retrieve a specific image by its coordinate values.
        
        Args:
            pamchip_barcode (int): Pamchip barcode used by this function.
            array (int): Array used by this function.
            fov (int): Fov used by this function.
            exposure_time (int): Exposure time used by this function.
            pump_cycle (int): Pump cycle used by this function.
            image_number (int): Image number used by this function.
            temperature (int): Temperature used by this function.
        
        Returns:
            xr.DataArray: Requested image by coords.
        """
        if self._images is None:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        
        # Create selection mask
        mask = (
            (self._images.coords['PamChip_barcode'] == pamchip_barcode) &
            (self._images.coords['Array'] == array) &
            (self._images.coords['FoV'] == fov) &
            (self._images.coords['Exposure_time'] == exposure_time) &
            (self._images.coords['Pump_cycle'] == pump_cycle) &
            (self._images.coords['Image_number'] == image_number) &
            (self._images.coords['Temperature'] == temperature)
        )
        
        indices = np.where(mask.values)[0]
        
        if len(indices) == 0:
            raise ValueError(
                f"No image found with specified coordinates:\n"
                f"  PamChip_barcode={pamchip_barcode}, Array={array}, FoV={fov},\n"
                f"  Exposure_time={exposure_time}, Pump_cycle={pump_cycle},\n"
                f"  Image_number={image_number}, Temperature={temperature}"
            )
        
        if len(indices) > 1:
            print(f"Warning: Multiple images ({len(indices)}) match the criteria. Returning first.")
        
        return self._images.isel(image_idx=indices[0])
    
    def visualize_image(
        self,
        pamchip_barcode: Optional[int] = None,
        array: Optional[int] = None,
        fov: Optional[int] = None,
        exposure_time: Optional[int] = None,
        pump_cycle: Optional[int] = None,
        image_number: Optional[int] = None,
        temperature: Optional[int] = None,
        image_idx: Optional[int] = None,
        cmap: str = 'gray',
        figsize: Tuple[int, int] = (10, 10),
        title: Optional[str] = None,
        colorbar: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        ax: Optional[plt.Axes] = None,
        **imshow_kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize a chosen TIFF image.
        
        You can select an image either by specifying all coordinate values
        (pamchip_barcode, array, fov, exposure_time, pump_cycle, image_number, temperature)
        or by providing the image_idx directly.
        
        Args:
            pamchip_barcode (Optional[int]): Pamchip barcode processed by this function.
            array (Optional[int]): Array processed by this function.
            fov (Optional[int]): Fov processed by this function.
            exposure_time (Optional[int]): Exposure time processed by this function.
            pump_cycle (Optional[int]): Pump cycle processed by this function.
            image_number (Optional[int]): Image number processed by this function.
            temperature (Optional[int]): Temperature processed by this function.
            image_idx (Optional[int]): Zero-based index selecting the image.
            cmap (str): Cmap used by this function.
            figsize (Tuple[int, int]): Figsize processed by this function.
            title (Optional[str]): Title processed by this function.
            colorbar (bool): Colorbar used by this function.
            vmin (Optional[float]): Vmin processed by this function.
            vmax (Optional[float]): Vmax processed by this function.
            ax (Optional[plt.Axes]): Ax processed by this function.
            **imshow_kwargs: Additional keyword arguments forwarded by this function.
        
        Returns:
            Tuple[plt.Figure, plt.Axes]: Visualize image.
        """
        if self._images is None:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        
        # Get the image
        if image_idx is not None:
            image = self._images.isel(image_idx=image_idx)
            # Get coordinates for title
            coords = {
                name: int(self._images.coords[name].values[image_idx])
                for name in self.COORDINATE_NAMES
            }
        else:
            # Check that all coordinates are provided
            coord_values = {
                'pamchip_barcode': pamchip_barcode,
                'array': array,
                'fov': fov,
                'exposure_time': exposure_time,
                'pump_cycle': pump_cycle,
                'image_number': image_number,
                'temperature': temperature
            }
            
            missing = [k for k, v in coord_values.items() if v is None]
            if missing:
                raise ValueError(
                    f"Either provide image_idx or all coordinate values. "
                    f"Missing: {missing}"
                )
            
            image = self.get_image_by_coords(
                pamchip_barcode=pamchip_barcode,
                array=array,
                fov=fov,
                exposure_time=exposure_time,
                pump_cycle=pump_cycle,
                image_number=image_number,
                temperature=temperature
            )
            coords = {
                'PamChip_barcode': pamchip_barcode,
                'Array': array,
                'FoV': fov,
                'Exposure_time': exposure_time,
                'Pump_cycle': pump_cycle,
                'Image_number': image_number,
                'Temperature': temperature
            }
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Plot the image
        im = ax.imshow(
            np.log(np.log(image.values)), 
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax,
            **imshow_kwargs
        )
        
        # Add colorbar with height matching the image
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax, label='Intensity')
            cbar.ax.tick_params(labelsize=16)  # Colorbar tick labels
            cbar.set_label('Intensity', fontsize=18)  # Colorbar label
        
        # Set title
        if title is None:
            title = (
                f"PamChip: {coords.get('PamChip_barcode', coords.get('pamchip_barcode'))} | "
                f"Array: {coords.get('Array', coords.get('array'))} | "
                f"FoV: {coords.get('FoV', coords.get('fov'))}\n"
                f"Exp: {coords.get('Exposure_time', coords.get('exposure_time'))}ms | "
                f"Pump: {coords.get('Pump_cycle', coords.get('pump_cycle'))} | "
                f"Img: {coords.get('Image_number', coords.get('image_number'))} | "
                f"Temp: {coords.get('Temperature', coords.get('temperature'))}°C" )
        
        ax.set_title(title, fontsize=20)  # Increased title font size
        ax.set_xlabel('X (pixels)', fontsize=18)  # Increased xlabel font size
        ax.set_ylabel('Y (pixels)', fontsize=18)  # Increased ylabel font size
        ax.tick_params(axis='both', labelsize=16)  # Increased tick label font size
        
        plt.tight_layout()
        
        return fig, ax
    
    def visualize_image_grid(
        self,
        vary_coord: str,
        fixed_coords: Dict[str, int],
        ncols: int = 4,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = 'gray',
        **imshow_kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Visualize multiple images in a grid, varying one coordinate.
        
        Args:
            vary_coord (str): Vary coord used by this function.
            fixed_coords (Dict[str, int]): Fixed coords processed by this function.
            ncols (int): Ncols used by this function.
            figsize (Optional[Tuple[int, int]]): Figsize processed by this function.
            cmap (str): Cmap used by this function.
            **imshow_kwargs: Additional keyword arguments forwarded by this function.
        
        Returns:
            Tuple[plt.Figure, np.ndarray]: Visualize image grid.
        """
        if self._images is None:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        
        # Get unique values for the varying coordinate
        mask = np.ones(len(self._images.image_idx), dtype=bool)
        for name, value in fixed_coords.items():
            mask &= (self._images.coords[name].values == value)
        
        subset = self._images.isel(image_idx=mask)
        vary_values = np.unique(subset.coords[vary_coord].values)
        
        if len(vary_values) == 0:
            raise ValueError("No images match the specified fixed coordinates")
        
        # Create grid
        n_images = len(vary_values)
        nrows = int(np.ceil(n_images / ncols))
        
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_2d(axes)
        
        for idx, val in enumerate(vary_values):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col]
            
            # Get the image
            img_mask = mask & (self._images.coords[vary_coord].values == val)
            img_idx = np.where(img_mask)[0][0]
            
            ax.imshow(self._images.isel(image_idx=img_idx).values, cmap=cmap, **imshow_kwargs)
            ax.set_title(f"{vary_coord}={val}", fontsize=18)  # Increased title font size
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_images, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def get_unique_values(self, coordinate: str) -> np.ndarray:
        """Get unique values for a given coordinate.
        
        Args:
            coordinate (str): Coordinate used by this function.
        
        Returns:
            np.ndarray: Requested unique values.
        """
        if self._images is None:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        
        if coordinate not in self.COORDINATE_NAMES:
            raise ValueError(
                f"Unknown coordinate: {coordinate}. "
                f"Valid coordinates: {self.COORDINATE_NAMES}"
            )
        
        return np.unique(self._images.coords[coordinate].values)
    
    def summary(self) -> str:
        """Get a summary of the loaded data.
        
        Args:
            None.
        
        Returns:
            str: Summary.
        """
        lines = ["=" * 60, "DataLoader Summary", "=" * 60]
        
        lines.append(f"\nExperiment: {self.experiment_name}")
        lines.append(f"Subfolder: {self.subfolder_name}")
        lines.append(f"Peptides type: {self.peptides_type}")
        lines.append(f"Data directory: {self.data_dir}")
        
        if self._sample_annotation is not None:
            lines.append("\nSample Annotation:")
            lines.append(f"  Shape: {self._sample_annotation.shape}")
            lines.append(f"  Columns: {list(self._sample_annotation.columns)}")
        else:
            lines.append("\nSample Annotation: Not loaded")
        
        if self._layout_loader is not None:
            lines.append("\nArray Layout:")
            lines.append(f"  Chip type: {self._layout_loader.chip_type}")
            lines.append(f"  Grid dimensions: {self._layout_loader.grid_dimensions[0]}×{self._layout_loader.grid_dimensions[1]}")
            lines.append(f"  Spots: {self._layout_loader.spot_array.attrs['n_spots']}")
            lines.append(f"  Columns: {self._layout_loader.spot_array.attrs['columns']}")
        else:
            lines.append("\nArray Layout: Not loaded")
        
        if self._images is not None:
            lines.append("\nImages:")
            lines.append(f"  Number of images: {len(self._images.image_idx)}")
            lines.append(f"  Image shape: {self._images.attrs['image_shape']}")
            lines.append(f"  Data type: {self._images.dtype}")
            
            lines.append("\nFilename-derived coordinates:")
            for coord in self.COORDINATE_NAMES:
                unique_vals = self.get_unique_values(coord)
                lines.append(f"  {coord}: {len(unique_vals)} unique values "
                           f"(min={unique_vals.min()}, max={unique_vals.max()})")
        else:
            lines.append("\nImages: Not loaded")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Return a concise string representation for debugging.
        
        Args:
            None.
        
        Returns:
            str: Repr.
        """
        return (
            f"DataLoader(\n"
            f"  experiment_name='{self.experiment_name}',\n"
            f"  subfolder='{self.subfolder_name}',\n"
            f"  peptides_type='{self.peptides_type}',\n"
            f"  data_dir='{self.data_dir}',\n"
            f"  sample_annotation_loaded={self._sample_annotation is not None},\n"
            f"  array_layout_loaded={self._layout_loader is not None},\n"
            f"  images_loaded={self._images is not None}\n"
            f")"
        )


class ArrayLayoutLoader:
    """Loader class for PamChip array layout data."""

    CHIP_GRID_DIMENSIONS = dict(
        ARRAY_LAYOUT_LOADER_DEFAULTS["chip_grid_dimensions"]
    )
    NUMERIC_FLOAT_FIELDS = tuple(
        ARRAY_LAYOUT_LOADER_DEFAULTS["numeric_float_fields"]
    )
    ENCODINGS_TO_TRY = tuple(ARRAY_LAYOUT_LOADER_DEFAULTS["encodings_to_try"])
    DEFAULT_OUTPUT_DIR = ARRAY_LAYOUT_LOADER_DEFAULTS["default_output_dir"]
    FALLBACK_DATA_DIR = ARRAY_LAYOUT_LOADER_DEFAULTS["fallback_data_dir"]

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the ArrayLayoutLoader instance.
        
        Args:
            data_dir (Optional[str]): Directory containing or receiving the data.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            possible_paths = _candidate_paths(
                ARRAY_LAYOUT_LOADER_DEFAULTS["candidate_data_paths"]
            )
            self.data_dir = None
            for path in possible_paths:
                if path.exists() and path.is_dir():
                    self.data_dir = path
                    break

            if self.data_dir is None:
                self.data_dir = Path(self.FALLBACK_DATA_DIR)

        self._layout_data = None
        self._column_names = []
        self._spot_array = None
        self._grid_dimensions = None
        self._chip_type = None

    def load(
        self,
        filepath: str,
        encoding: str = "auto",
        saveToFile: bool = False,
    ) -> pd.DataFrame:
        """Handle load.
        
        Args:
            filepath (str): Filepath used by this function.
            encoding (str): Encoding used by this function.
            saveToFile (bool): SaveToFile used by this function.
        
        Returns:
            pd.DataFrame: Load.
        """
        file_path = self._resolve_path(filepath)

        print()
        print("Loading array layout...")

        if encoding == "auto":
            df = None
            last_error = None
            for enc in self.ENCODINGS_TO_TRY:
                try:
                    df = pd.read_csv(file_path, sep="\t", dtype=str, encoding=enc)
                    print(f"Successfully loaded array layout (using {enc} encoding)")
                    break
                except (UnicodeDecodeError, UnicodeError) as exc:
                    last_error = exc
                    continue

            if df is None:
                raise ValueError(f"Could not decode file. Last error: {last_error}")
        else:
            df = pd.read_csv(file_path, sep="\t", dtype=str, encoding=encoding)

        self._column_names = list(df.columns)
        self._layout_data = df
        self._build_spot_array()

        if saveToFile:
            self.save_layout_to_csv()

        return df

    def _resolve_path(self, filepath: str) -> Path:
        """Resolve path.
        
        Args:
            filepath (str): Filepath used by this function.
        
        Returns:
            Path: Resolved path.
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.data_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Array layout file not found: {path}")

        return path

    def _build_spot_array(self):
        """Build spot array.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if self._layout_data is None:
            raise ValueError("No data loaded. Call load() first.")

        row_col = None
        col_col = None
        for col in self._column_names:
            if col.lower() == "row":
                row_col = col
            elif col.lower() == "col":
                col_col = col

        if not row_col or not col_col:
            raise ValueError("Could not find 'Row' and 'Col' columns in the data")

        df = self._layout_data.copy()
        df[col_col] = pd.to_numeric(df[col_col], errors="coerce")
        df[row_col] = pd.to_numeric(df[row_col], errors="coerce")

        spot_df = df[df[col_col] >= 0].copy()

        max_row = int(spot_df[row_col].max())
        max_col = int(spot_df[col_col].max())
        min_row = int(spot_df[row_col].min())
        min_col = int(spot_df[col_col].min())

        n_rows = max_row - min_row + 1
        n_cols = max_col - min_col + 1
        self._grid_dimensions = (n_rows, n_cols)

        if self.CHIP_GRID_DIMENSIONS.get("PTK") == self._grid_dimensions:
            self._chip_type = "PTK"
        elif self.CHIP_GRID_DIMENSIONS.get("STK") == self._grid_dimensions:
            self._chip_type = "STK"
        else:
            self._chip_type = "Unknown"

        spot_array_data = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                spot_array_data[i, j] = None

        for _, row_data in spot_df.iterrows():
            file_row = int(row_data[row_col])
            file_col = int(row_data[col_col])
            array_row = file_row - min_row
            array_col = file_col - min_col

            spot_dict = {}
            for col_name in self._column_names:
                value = row_data[col_name]
                if col_name in [row_col, col_col]:
                    spot_dict[col_name] = (
                        int(float(value)) if pd.notna(value) else None
                    )
                elif col_name in self.NUMERIC_FLOAT_FIELDS:
                    try:
                        spot_dict[col_name] = float(value) if pd.notna(value) else None
                    except (ValueError, TypeError):
                        spot_dict[col_name] = value
                else:
                    spot_dict[col_name] = value if pd.notna(value) else None

            spot_array_data[array_row, array_col] = spot_dict

        self._spot_array = xr.DataArray(
            data=spot_array_data,
            dims=["spot_row", "spot_col"],
            coords={
                "spot_row": np.arange(n_rows),
                "spot_col": np.arange(n_cols),
            },
            attrs={
                "chip_type": self._chip_type,
                "grid_dimensions": self._grid_dimensions,
                "n_spots": int(
                    np.sum([spot is not None for spot in spot_array_data.flatten()])
                ),
                "columns": self._column_names,
                "row_offset": min_row,
                "col_offset": min_col,
            },
        )

        print(
            f"Built spot array: {n_rows}×{n_cols} ({self._chip_type}), "
            f"{self._spot_array.attrs['n_spots']} spots"
        )

    @property
    def spot_array(self) -> xr.DataArray:
        """Return spot array.
        
        Args:
            None.
        
        Returns:
            xr.DataArray: Stored spot array.
        """
        if self._spot_array is None:
            raise ValueError("No data loaded. Call load() first.")
        return self._spot_array

    @property
    def grid_dimensions(self) -> tuple:
        """Return grid dimensions.
        
        Args:
            None.
        
        Returns:
            tuple: Stored grid dimensions.
        """
        return self._grid_dimensions

    @property
    def chip_type(self) -> str:
        """Return chip type.
        
        Args:
            None.
        
        Returns:
            str: Stored chip type.
        """
        return self._chip_type

    def get_spot(self, row: int, col: int) -> Optional[dict]:
        """Get spot.
        
        Args:
            row (int): Row used by this function.
            col (int): Col used by this function.
        
        Returns:
            Optional[dict]: Requested spot.
        """
        if self._spot_array is None:
            raise ValueError("No data loaded. Call load() first.")

        if row < 0 or row >= self._grid_dimensions[0]:
            raise IndexError(f"Row {row} out of range [0, {self._grid_dimensions[0] - 1}]")
        if col < 0 or col >= self._grid_dimensions[1]:
            raise IndexError(f"Col {col} out of range [0, {self._grid_dimensions[1] - 1}]")

        return self._spot_array.values[row, col]

    def get_spot_by_coords(self, file_row: int, file_col: int) -> Optional[dict]:
        """Get spot by coords.
        
        Args:
            file_row (int): File row used by this function.
            file_col (int): File col used by this function.
        
        Returns:
            Optional[dict]: Requested spot by coords.
        """
        if self._spot_array is None:
            raise ValueError("No data loaded. Call load() first.")

        row_offset = self._spot_array.attrs["row_offset"]
        col_offset = self._spot_array.attrs["col_offset"]
        return self.get_spot(file_row - row_offset, file_col - col_offset)

    def get_spot_field(self, row: int, col: int, field: str):
        """Get spot field.
        
        Args:
            row (int): Row used by this function.
            col (int): Col used by this function.
            field (str): Field used by this function.
        
        Returns:
            object: Requested spot field.
        """
        spot = self.get_spot(row, col)
        if spot is None:
            return None
        return spot.get(field, None)

    def get_all_ids(self) -> np.ndarray:
        """Get all IDs.
        
        Args:
            None.
        
        Returns:
            np.ndarray: Requested all IDs.
        """
        if self._spot_array is None:
            raise ValueError("No data loaded. Call load() first.")

        n_rows, n_cols = self._grid_dimensions
        ids = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                spot = self._spot_array.values[i, j]
                ids[i, j] = spot.get("ID") if spot is not None else None
        return ids

    def get_all_sequences(self) -> np.ndarray:
        """Get all sequences.
        
        Args:
            None.
        
        Returns:
            np.ndarray: Requested all sequences.
        """
        if self._spot_array is None:
            raise ValueError("No data loaded. Call load() first.")

        n_rows, n_cols = self._grid_dimensions
        sequences = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                spot = self._spot_array.values[i, j]
                sequences[i, j] = spot.get("Sequence") if spot is not None else None
        return sequences

    def get_field_array(self, field: str) -> np.ndarray:
        """Get field array.
        
        Args:
            field (str): Field used by this function.
        
        Returns:
            np.ndarray: Requested field array.
        """
        if self._spot_array is None:
            raise ValueError("No data loaded. Call load() first.")

        n_rows, n_cols = self._grid_dimensions
        field_array = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                spot = self._spot_array.values[i, j]
                field_array[i, j] = spot.get(field) if spot is not None else None
        return field_array

    def get_layout_summary(self) -> dict:
        """Get layout summary.
        
        Args:
            None.
        
        Returns:
            dict: Requested layout summary.
        """
        if self._layout_data is None:
            raise ValueError("No data loaded. Call load() first.")

        return {
            "total_spots": (
                self._spot_array.attrs["n_spots"]
                if self._spot_array is not None
                else len(self._layout_data) - 8
            ),
            "grid_dimensions": self._grid_dimensions,
            "chip_type": self._chip_type,
            "columns": self._column_names,
        }

    def save_layout_to_csv(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        verbose: bool = True,
    ) -> Optional[Path]:
        """Save layout to CSV.
        
        Args:
            output_dir (str): Directory containing or receiving the output.
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            Optional[Path]: Saved layout to CSV.
        """
        if self._spot_array is None:
            if verbose:
                print("No data loaded. Cannot save to CSV.")
            return None

        script_dir = Path(__file__).parent
        output_path = script_dir.parent / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"array_layout_{self._chip_type}.csv"
        csv_path = output_path / filename

        n_rows, n_cols = self._grid_dimensions
        row_offset = self._spot_array.attrs["row_offset"]
        col_offset = self._spot_array.attrs["col_offset"]
        col_headers = [f"Col_{col_offset + j}" for j in range(n_cols)]

        rows_data = []
        for i in range(n_rows):
            row_values = [f"Row_{row_offset + i}"]
            for j in range(n_cols):
                spot = self._spot_array.values[i, j]
                if spot is not None:
                    spot_id = spot.get("ID", "")
                    sequence = spot.get("Sequence", "")
                    cell_value = f"{spot_id}\n{sequence}"
                else:
                    cell_value = ""
                row_values.append(cell_value)
            rows_data.append(row_values)

        df = pd.DataFrame(rows_data, columns=["Row"] + col_headers)
        df.to_csv(csv_path, index=False)

        if verbose:
            print(f"Saved {self._chip_type} array layout to: {csv_path}")
            print(f"  Grid: {n_rows}×{n_cols}")
            print(f"  Total spots: {self._spot_array.attrs['n_spots']}")

        return csv_path

    def __repr__(self) -> str:
        """Return a concise string representation for debugging.
        
        Args:
            None.
        
        Returns:
            str: Repr.
        """
        if self._spot_array is None:
            return "ArrayLayoutLoader(not loaded)"
        return (
            f"ArrayLayoutLoader(chip_type='{self._chip_type}', "
            f"grid={self._grid_dimensions[0]}×{self._grid_dimensions[1]}, "
            f"spots={self._spot_array.attrs['n_spots']})"
        )


if __name__ == '__main__':
    
    # Initialize loader (will prompt for experiment and subfolder selection)
    loader = DataLoader()
    
    # Load data
    print("\n--- Loading Sample Annotation ---")
    loader.load_sample_annotation()
    
    print("\n--- Loading Images ---")
    loader.load_images()
    
    # Print summary
    print("\n" + loader.summary())
    
    # Example visualization
    print("\n--- Example Visualization ---")
    for i in range(len(loader.images.image_idx)):#[0:2])):
        if loader.images is not None and len(loader.images.image_idx) > 0:
            loader.visualize_image(image_idx=i)
            plt.show()
            plt.pause(0.00000001)
            
            
            
