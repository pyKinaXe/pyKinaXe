"""
DataEnricher - A class for enriching and combining sample annotation data.

This module provides the following functionality:
1. Combines sample annotations from two DataLoader instances (corresponding PTK and STK)
2. Parses and analyzes sample names given in sample annotation files
3. Generates test condition assignments based on sample naming patterns (from two separated numbers in the end)
4. Outputs experimental design enrichment tables
"""

# Standard library imports
import argparse
import os
from datetime import datetime
import io
from io import StringIO
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

# Third-party imports
import pandas as pd
import requests

# Local imports
from config.data_enricher import (
    DATA_ENRICHER_DEFAULTS,
    KINASE_LIVER_EXTRACTOR_DEFAULTS,
    OMNIPATH_PTM_EXTRACTOR_DEFAULTS,
    UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS,
)
from kx_data_importer import DataLoader


class DataEnricher:
    """
    A class for enriching experimental data with sample annotations made by the user prior to performing PTK- and STK-based experiments.
    
    This class combines sample annotation tables from the two different experiments (PTK and STK),
    parses sample names, and automatically assigns test conditions based on
    naming patterns and grouping logic.
    
    Attributes:
        sample_annotation_1 (pd.DataFrame): First sample annotation table.
        sample_annotation_2 (pd.DataFrame): Second sample annotation table.
        enriched_table (Optional[pd.DataFrame]): Generated enriched table with test conditions;
            initialized to None until generate_enriched_table() is called.
        experiment_name_1 (str): Name of the experiment for loader1.
        subfolder_name_1 (str): Name of the subfolder for loader1.
        experiment_name_2 (str): Name of the experiment for loader2.
        subfolder_name_2 (str): Name of the subfolder for loader2.
        timestamp_1 (str): Timestamp associated with loader1 (YYYYMMDDHHMMSS format).
        timestamp_2 (str): Timestamp associated with loader2 (YYYYMMDDHHMMSS format).
        source_data_path_1 (Path): Path to the source data directory used by loader1 (for PTK).
        source_data_path_2 (Path): Path to the source data directory used by loader2 (for STK).
        results_dir_1 (Path): Path to the results directory created for loader1 (for PTK).
        results_dir_2 (Path): Path to the results directory created for loader2 (for STK).
        peptides_type_1 (str): Type of peptides for loader1 ('PTK' or 'STK').
        peptides_type_2 (str): Type of peptides for loader2 ('PTK' or 'STK').
        cell_line (Optional[str]): Cell line name to be added to the enriched table, if provided.
    
    """
    
    # ========== Column Name Constants ==========
    COLUMN_PAMCHIP_LOCATION = DATA_ENRICHER_DEFAULTS["columns"]["pamchip_location"]
    COLUMN_BARCODE = DATA_ENRICHER_DEFAULTS["columns"]["barcode"]
    COLUMN_ROW = DATA_ENRICHER_DEFAULTS["columns"]["row"]
    COLUMN_ARRAY = DATA_ENRICHER_DEFAULTS["columns"]["array"]
    COLUMN_ARTICLE_NUMBER = DATA_ENRICHER_DEFAULTS["columns"]["article_number"]
    COLUMN_STRIP = DATA_ENRICHER_DEFAULTS["columns"]["strip"]
    COLUMN_SAMPLE_NAME = DATA_ENRICHER_DEFAULTS["columns"]["sample_name"]
    COLUMN_TECHNICAL_REPLICATE = DATA_ENRICHER_DEFAULTS["columns"][
        "technical_replicate"
    ]
    COLUMN_BIOLOGICAL_REPLICATE = DATA_ENRICHER_DEFAULTS["columns"][
        "biological_replicate"
    ]
    COLUMN_ASSAY_VOLUME = DATA_ENRICHER_DEFAULTS["columns"]["assay_volume"]
    COLUMN_TEST_CONDITION = DATA_ENRICHER_DEFAULTS["columns"]["test_condition"]
    
    # ========== Display Column Configuration ==========
    # Columns to always display in sample annotation tables
    ANNOTATION_ALWAYS_DISPLAY = list(
        DATA_ENRICHER_DEFAULTS["annotation_always_display"]
    )
    
    # Columns to display conditionally (if they exist in the dataframe)
    ANNOTATION_CONDITIONAL_DISPLAY = list(
        DATA_ENRICHER_DEFAULTS["annotation_conditional_display"]
    )
    
    # Columns to explicitly exclude from display
    ANNOTATION_EXCLUDE_COLUMNS = list(
        DATA_ENRICHER_DEFAULTS["annotation_exclude_columns"]
    )
    
    # ========== Roman Numeral Conversion ==========
    # Dictionary for converting Roman numerals to integers
    ROMAN_VALUES = dict(DATA_ENRICHER_DEFAULTS["roman_values"])
    
    # ========== File Naming Patterns ==========
    ENRICHMENT_FILENAME_SUFFIX = DATA_ENRICHER_DEFAULTS["enrichment_filename_suffix"]
    
    # ========== Test Condition Naming ==========
    TEST_CONDITION_CONTROL = DATA_ENRICHER_DEFAULTS["test_condition"]["control"]
    TEST_CONDITION_PREFIX = DATA_ENRICHER_DEFAULTS["test_condition"]["prefix"]
    
    def __init__(self, loader1: DataLoader, loader2: DataLoader, cell_line: Optional[str] = None):
        """Initializes the DataEnricher with two DataLoader instances.
        
        Args:
            loader1 (DataLoader): Loader1 processed by this function.
            loader2 (DataLoader): Loader2 processed by this function.
            cell_line (Optional[str]): Cell line processed by this function.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        # Store experiment and subfolder names for both loaders
        self.experiment_name_1 = loader1.experiment_name
        self.subfolder_name_1 = loader1.subfolder_name
        self.experiment_name_2 = loader2.experiment_name
        self.subfolder_name_2 = loader2.subfolder_name
        
        # Store timestamps from loaders for consistent naming
        self.timestamp_1 = loader1.timestamp
        self.timestamp_2 = loader2.timestamp
        
        # Store source data paths for reference
        self.source_data_path_1 = loader1.data_dir
        self.source_data_path_2 = loader2.data_dir
        
        # Store peptides types from both loaders
        self.peptides_type_1 = loader1.peptides_type
        self.peptides_type_2 = loader2.peptides_type
        self.results_parent_relpath_1 = Path(
            getattr(loader1, 'results_parent_relpath', '.')
        )
        self.results_parent_relpath_2 = Path(
            getattr(loader2, 'results_parent_relpath', '.')
        )
        self.results_experiment_relpath_1 = Path(
            getattr(loader1, 'results_experiment_relpath', self.experiment_name_1)
        )
        self.results_experiment_relpath_2 = Path(
            getattr(loader2, 'results_experiment_relpath', self.experiment_name_2)
        )
        
        # Store cell line if provided
        self.cell_line = cell_line
        
        # Extract sample annotations
        self.sample_annotation_1 = loader1._sample_annotation.copy()
        self.sample_annotation_2 = loader2._sample_annotation.copy()
        self.enriched_table = None
        
        # Create results directories for both loaders (same logic as ImageProcessor)
        self.results_dir_1 = self._get_results_dir(
            self.experiment_name_1,
            self.subfolder_name_1,
            self.timestamp_1,
            self.source_data_path_1,
            self.results_parent_relpath_1,
            self.results_experiment_relpath_1,
        )
        self.results_dir_2 = self._get_results_dir(
            self.experiment_name_2,
            self.subfolder_name_2,
            self.timestamp_2,
            self.source_data_path_2,
            self.results_parent_relpath_2,
            self.results_experiment_relpath_2,
        )
    
    def _get_results_dir(
        self,
        experiment_name: str,
        subfolder_name: str,
        timestamp: str,
        source_data_path: Path,
        results_parent_relpath: Path,
        results_experiment_relpath: Path,
    ) -> Path:
        """Get the results directory path:
            results/<YYYYMMDDHHMMSS>_<experiment_name>/<subfolder_name>/
        Creates the directory if it doesn't exist.
        
        Args:
            experiment_name (str): Experiment name used by this function.
            subfolder_name (str): Subfolder name used by this function.
            timestamp (str): Timestamp string associated with the current analysis run.
            source_data_path (Path): Path to the source data.
            results_parent_relpath (Path): Path-like value for results parent relpath.
            results_experiment_relpath (Path): Path-like value for results experiment relpath.
        
        Returns:
            Path: Requested results dir.
        """
        override_root = os.environ.get("PYKINAXE_RESULTS_ROOT")
        if override_root:
            root_dir = Path(override_root).expanduser().resolve().parent
            results_root = Path(override_root).expanduser().resolve()
        else:
            root_dir = Path(__file__).parent.parent.resolve()
            results_root = root_dir / 'results'
        
        # Use timestamp (from loader) for consistent naming
        experiment_folder = (
            results_experiment_relpath.parent
            / f"{timestamp}_{results_experiment_relpath.name}"
        )
        
        # Results directory path
        results_dir = (
            results_root
            / results_parent_relpath
            / experiment_folder
            / subfolder_name
        )
        
        # Create directory if it doesn't exist
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experimental data source path info
        path_info_file = results_dir / f"{timestamp}_source_data_path.txt"
        if not path_info_file.exists():  # Only write if not already created
            with open(path_info_file, 'w') as f:
                f.write("Source Data Path:\n")
                f.write(f"{source_data_path}\n")
                f.write(f"\nExperiment Name: {experiment_name}\n")
                f.write(f"Subfolder Name: {subfolder_name}\n")
                f.write(f"Analysis Timestamp: {timestamp}\n")
        
        return results_dir
    
    def _parse_sample_name(self, sample_name: str) -> tuple:
        """Parse sample name to extract the last two numbers (biological and technical replicates) and 'first_part' (sample name without those numbers).
        
        The last two numbers can be in any format (Roman or Arabic) and are separated
        by various possible separators (., , _ - / \ | : space).
        Separators can be different (e.g., "puc18_1.1" has _ then .).
        
        Args:
            sample_name (str): Sample name used by this function.
        
        Returns:
            tuple: Parse sample name.
        """
        # Convert to string in case of NaN or other types
        sample_name = str(sample_name).strip()
        
        # Define number patterns - Roman or Arabic
        roman_num = r'[IVXLCDM]+'
        arabic_num = r'\d+'
        
        # Create a separator pattern that matches any of the possible separators
        # Using character class for all separators
        sep_pattern = r'[_\-., /\\|:]'
        
        # Try all combinations: Roman+Roman, Roman+Arabic, Arabic+Roman, Arabic+Arabic
        # The greedy .+ ensures we capture the LAST two numbers in the string
        patterns = [
            (rf'^(.+){sep_pattern}({roman_num}){sep_pattern}({roman_num})$', 'RR'),
            (rf'^(.+){sep_pattern}({roman_num}){sep_pattern}({arabic_num})$', 'RA'),
            (rf'^(.+){sep_pattern}({arabic_num}){sep_pattern}({roman_num})$', 'AR'),
            (rf'^(.+){sep_pattern}({arabic_num}){sep_pattern}({arabic_num})$', 'AA'),
        ]
        
        for pattern, type in patterns:
            match = re.match(pattern, sample_name, re.IGNORECASE)
            
            if match:
                first_part = match.group(1)
                num1_raw = match.group(2)
                num2_raw = match.group(3)
                                
                # Convert both to integers
                num1 = self._convert_to_int(num1_raw)
                num2 = self._convert_to_int(num2_raw)
                
                # Only return if both conversions succeeded
                if num1 is not None and num2 is not None:
                    return first_part, num1, num2
        
        # If no pattern matched, return the whole string as first part
        # print(f"DEBUG: No pattern matched for '{sample_name}'")
        return sample_name, None, None
    
    def _convert_to_int(self, num_str: str) -> int:
        """Convert a number string (Roman or Arabic) to integer.
        
        Args:
            num_str (str): Num str used by this function.
        
        Returns:
            int: Convert to int.
        """
        # First try to parse as Arabic number
        try:
            return int(num_str)
        except ValueError:
            pass
        
        # If that fails, try Roman numeral
        num_str = num_str.upper()
        total = 0
        prev_value = 0
        
        # Roman numbers deciphering logic: process from right to left
        for char in reversed(num_str):
            value = self.ROMAN_VALUES.get(char, 0)
            if value >= prev_value:
                total += value
            else:
                total -= value
            prev_value = value
        
        return total if total > 0 else None
    
    
    def _has_valid_replicate_column(self, df: pd.DataFrame, col_name: str) -> bool:
        """Check if a replicate column exists and has non-zero values.
        
        Args:
            df (pd.DataFrame): Input pandas DataFrame used by this function.
            col_name (str): Col name used by this function.
        
        Returns:
            bool: Has valid replicate column.
        """
        if col_name not in df.columns:
            return False
        
        try:
            # Check if column contains numeric data
            if pd.api.types.is_numeric_dtype(df[col_name]):
                non_null_values = df[col_name].dropna()
                if len(non_null_values) > 0:
                    # Check if values are integer-like
                    if all(val == int(val) for val in non_null_values):
                        # Check if at least one is greater than zero
                        if any(int(val) > 0 for val in non_null_values):
                            return True
        except (TypeError, ValueError):
            pass
        
        return False
    
    def _determine_test_conditions(self, df_with_source: pd.DataFrame) -> list:
        """Determine test condition for each row based on sample name grouping.
        
        Logic:
        - For each Technical Replicate within the same source table
        - Check if there are multiple Biological Replicates within the Technical Replicate group
        - If yes, perform assignment separately for each (Technical Replicate, Biological Replicate) combination
        - Parse sample names to extract first parts (ignoring the last two numbers)
        - Find the alphabetically smallest (in ASCII order) first_part → ALL entries with this first_part get 'Control'
        - For remaining unique first_parts (sorted alphabetically), assign 'Test1', 'Test2', etc.
        - All entries with the same first_part get the same test condition value
        
        Args:
            df_with_source (pd.DataFrame): Input pandas DataFrame containing with source.
        
        Returns:
            list: Determine test conditions.
        """
        # Create a copy to avoid modifying original
        df = df_with_source.copy()
        df[self.COLUMN_TEST_CONDITION] = ''
        
        # Extract technical and biological replicate information for all rows
        tech_reps = []
        bio_reps = []
        for idx in df.index:
            sample_name = df.loc[idx, self.COLUMN_SAMPLE_NAME]
            _, bio_rep, tech_rep = self._parse_sample_name(sample_name)
            
            # If tech_rep not parsed from name, check if column exists
            if tech_rep is None and self.COLUMN_TECHNICAL_REPLICATE in df.columns:
                tech_rep_val = df.loc[idx, self.COLUMN_TECHNICAL_REPLICATE]
                if pd.notna(tech_rep_val):
                    tech_rep = int(tech_rep_val)
            
            # If bio_rep not parsed from name, check if column exists
            if bio_rep is None and self.COLUMN_BIOLOGICAL_REPLICATE in df.columns:
                bio_rep_val = df.loc[idx, self.COLUMN_BIOLOGICAL_REPLICATE]
                if pd.notna(bio_rep_val):
                    bio_rep = int(bio_rep_val)
            
            tech_reps.append(tech_rep)
            bio_reps.append(bio_rep)
        
        df['_tech_rep'] = tech_reps
        df['_bio_rep'] = bio_reps
        
        # Process each source table separately
        for source in df['_source'].unique():
            source_mask = df['_source'] == source
            
            # Get unique Technical Replicates for this source
            tech_rep_values = df[source_mask]['_tech_rep'].unique()
            
            for tech_rep_val in tech_rep_values:
                # Get rows for this technical replicate and source
                # Handle None/NaN comparison properly
                if pd.isna(tech_rep_val):
                    tech_rep_mask = source_mask & df['_tech_rep'].isna()
                else:
                    tech_rep_mask = source_mask & (df['_tech_rep'] == tech_rep_val)
                
                # Get unique Biological Replicates within this Technical Replicate group
                bio_rep_values = df[tech_rep_mask]['_bio_rep'].unique()
                
                # Process each biological replicate separately
                for bio_rep_val in bio_rep_values:
                    # Get rows for this biological replicate within the technical replicate group
                    # Handle None/NaN comparison properly
                    if pd.isna(bio_rep_val):
                        bio_rep_mask = tech_rep_mask & df['_bio_rep'].isna()
                    else:
                        bio_rep_mask = tech_rep_mask & (df['_bio_rep'] == bio_rep_val)
                    
                    group_indices = df[bio_rep_mask].index.tolist()
                    
                    # Skip if no rows for this group (shouldn't happen, but safety check)
                    if len(group_indices) == 0:
                        continue
                    
                    # Parse sample names to get first parts
                    first_parts = []
                    for idx in group_indices:
                        sample_name = df.loc[idx, self.COLUMN_SAMPLE_NAME]
                        first_part, _, _ = self._parse_sample_name(sample_name)
                        first_parts.append(first_part)
                    
                    # Identify unique groups and sort alphabetically
                    unique_groups = sorted(set(first_parts))
                    
                    # Smallest alphabetically → Control
                    # Others → Test1, Test2, etc. based on their alphabetical position
                    test_condition_map = {}
                    test_condition_map[unique_groups[0]] = self.TEST_CONDITION_CONTROL
                    
                    # Assign Test1, Test2, etc. to remaining groups
                    for i, group in enumerate(unique_groups[1:], start=1):
                        test_condition_map[group] = f'{self.TEST_CONDITION_PREFIX}{i}'
                    
                    # Assign test conditions to all rows based on their first_part
                    for i, idx in enumerate(group_indices):
                        first_part = first_parts[i]
                        df.loc[idx, self.COLUMN_TEST_CONDITION] = test_condition_map[first_part]
        
        return df[self.COLUMN_TEST_CONDITION].tolist()
    
    def generate_enriched_table(self) -> pd.DataFrame:
        """Generate the enriched table combining both sample annotations.
        
        Args:
            None.
        
        Returns:
            pd.DataFrame: Generate enriched table.
        """
        # Check if replicate columns are available in input tables
        has_bio_rep = (self._has_valid_replicate_column(self.sample_annotation_1, self.COLUMN_BIOLOGICAL_REPLICATE) and
                       self._has_valid_replicate_column(self.sample_annotation_2, self.COLUMN_BIOLOGICAL_REPLICATE))
        
        has_tech_rep = (self._has_valid_replicate_column(self.sample_annotation_1, self.COLUMN_TECHNICAL_REPLICATE) and
                        self._has_valid_replicate_column(self.sample_annotation_2, self.COLUMN_TECHNICAL_REPLICATE))
        
        # Select and prepare data from both tables
        cols_to_select = [self.COLUMN_PAMCHIP_LOCATION, self.COLUMN_BARCODE, self.COLUMN_ROW, self.COLUMN_SAMPLE_NAME]
        
        if has_bio_rep:
            cols_to_select.append(self.COLUMN_BIOLOGICAL_REPLICATE)
        if has_tech_rep:
            cols_to_select.append(self.COLUMN_TECHNICAL_REPLICATE)
        
        df1 = self.sample_annotation_1[cols_to_select].copy()
        df1['_source'] = 1
        
        df2 = self.sample_annotation_2[cols_to_select].copy()
        df2['_source'] = 2
        
        # Concatenate both tables
        combined = pd.concat([df1, df2], ignore_index=True)
        
        # Add Supergroup column (all 'Sgroup1')
        combined['Supergroup'] = 'Sgroup1'
        
        # Determine Test Condition based on logic
        combined['Test Condition'] = self._determine_test_conditions(combined)
        
        # Parse sample names to extract construct names and replicate information
        bio_reps = []
        tech_reps = []
        constructs = []
        standardized_sample_names = []
        
        for idx in combined.index:
            sample_name = combined.loc[idx, self.COLUMN_SAMPLE_NAME]
            first_part, num1, num2 = self._parse_sample_name(sample_name)
            
            # Determine final biological replicate value
            if num1 is not None:
                bio_rep = num1
            elif has_bio_rep:
                bio_rep = int(combined.loc[idx, self.COLUMN_BIOLOGICAL_REPLICATE])
            else:
                bio_rep = None
            
            # Determine final technical replicate value
            if num2 is not None:
                tech_rep = num2
            elif has_tech_rep:
                tech_rep = int(combined.loc[idx, self.COLUMN_TECHNICAL_REPLICATE])
            else:
                tech_rep = None
            
            # Store final replicate values
            bio_reps.append(bio_rep)
            tech_reps.append(tech_rep)
            constructs.append(first_part)
            
            # Standardize sample name to 'FirstPart_X_Y' format
            if bio_rep is not None and tech_rep is not None:
                standardized_name = f"{first_part}_{bio_rep}_{tech_rep}"
            else:
                # Keep original if numbers aren't available
                standardized_name = sample_name
            standardized_sample_names.append(standardized_name)
        
        # Update Sample name column with standardized format
        combined[self.COLUMN_SAMPLE_NAME] = standardized_sample_names
        
        # Add Construct column (first part of sample name)
        combined['Construct'] = constructs
        
        # Add Type column based on source (PTK or STK)
        combined['Type'] = combined['_source'].apply(
            lambda x: self.peptides_type_1 if x == 1 else self.peptides_type_2
        )
        
        # Add Cell line column if provided
        if self.cell_line is not None:
            combined['Cell line'] = self.cell_line
        
        # Add Biological and Technical Replicate columns
        combined['Biological Replicate'] = bio_reps
        combined['Technical Replicate'] = tech_reps
        
        # Drop original replicate columns if they existed
        if has_bio_rep:
            combined.drop(self.COLUMN_BIOLOGICAL_REPLICATE, axis=1, inplace=True, errors='ignore')
        if has_tech_rep:
            combined.drop(self.COLUMN_TECHNICAL_REPLICATE, axis=1, inplace=True, errors='ignore')
        
        # Select final columns in desired order
        final_columns = [
            self.COLUMN_BARCODE, 
            self.COLUMN_ROW, 
            self.COLUMN_SAMPLE_NAME,
            'Construct',
            'Type'
        ]
        
        # Add Cell line column if it exists
        if 'Cell line' in combined.columns:
            final_columns.append('Cell line')
        
        # Add remaining columns
        final_columns.extend(['Supergroup', 'Test Condition'])
        
        # Add replicate columns if they exist
        if 'Biological Replicate' in combined.columns:
            final_columns.append('Biological Replicate')
        if 'Technical Replicate' in combined.columns:
            final_columns.append('Technical Replicate')
        
        self.enriched_table = combined[final_columns]
        
        return self.enriched_table
    
    def _display_sample_annotation(self, df: pd.DataFrame, title: str):
        """Display sample annotation columns (similar logic as in kx_data_importer.py).
        
        Args:
            df (pd.DataFrame): Input pandas DataFrame used by this function.
            title (str): Title used by this function.
        
        Returns:
            object: Display sample annotation.
        """
        # Columns to always display if they exist
        always_display = [
            'PamChip Location', 'Barcode', 'Row', 'Array', 
            'Article number', 'Strip', 'Sample name'
        ]
        
        # Columns to display only if they exist AND have non-zero values
        conditional_display = ['Technical replicate', 'Biological replicate']
        
        # Check which "always display" columns are present
        available_columns = [col for col in always_display if col in df.columns]
        
        # Helper function to check if column has non-zero integer-like values
        def has_nonzero_integers(col_name):
            """Return whether nonzero integers.
            
            Args:
                col_name: Col name processed by this function.
            
            Returns:
                bool: True when nonzero integers is satisfied, otherwise False.
            """
            if col_name not in df.columns:
                return False
            try:
                # Check if column contains numeric data
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    non_null_values = df[col_name].dropna()
                    if len(non_null_values) > 0:
                        # Check if values are integer-like (handles both int and float like 1.0)
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
        
        # Find additional columns with non-zero integer values (except 'Assay volume')
        exclude_columns = set(always_display + conditional_display + ['Assay volume'])
        for col in df.columns:
            if col not in exclude_columns and col not in available_columns:
                if has_nonzero_integers(col):
                    available_columns.append(col)
        
        if available_columns:
            print(f"\n{title}:")
            print("=" * 80)
            # Set display options to show all rows
            with pd.option_context('display.max_rows', None, 'display.width', None):
                print(df[available_columns].to_string(index=False))
            print("=" * 80)
        else:
            print(f"\n{title}:")
            print("Warning: No displayable columns found.")
    
    def save_enriched_table(self, enriched_df: Optional[pd.DataFrame] = None, verbose: bool = True) -> tuple:
        """Save the enriched table to CSV and TXT formats with timestamp.
        
        Files are saved to both loader directories automatically.
        Uses timestamp from loader1 for consistent naming.
        
        Filenames: YYYYMMDDHHMMSS_data_enrichment.csv and YYYYMMDDHHMMSS_data_enrichment.txt
        
        Args:
            enriched_df (Optional[pd.DataFrame]): Input pandas DataFrame containing enriched.
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            tuple: Saved enriched table.
        """
        # Generate enriched table if not provided
        if enriched_df is None:
            enriched_df = self.generate_enriched_table()
        
        # Use timestamp from loader1 for consistent naming
        timestamp = self.timestamp_1
        
        # Define filenames
        csv_filename = f"{timestamp}{self.ENRICHMENT_FILENAME_SUFFIX}.csv"
        txt_filename = f"{timestamp}{self.ENRICHMENT_FILENAME_SUFFIX}.txt"
        
        # Save to loader1 directory
        csv_path_1 = self.results_dir_1 / csv_filename
        txt_path_1 = self.results_dir_1 / txt_filename
        
        enriched_df.to_csv(csv_path_1, index=False)
        enriched_df.to_csv(txt_path_1, index=False, sep='\t')
        
        if verbose:
            print("\nSaved enriched data for loader1:")
            print(f"  CSV: {csv_path_1}")
            print(f"  TXT: {txt_path_1}")
        
        # Save to loader2 directory
        csv_path_2 = self.results_dir_2 / csv_filename
        txt_path_2 = self.results_dir_2 / txt_filename
        
        enriched_df.to_csv(csv_path_2, index=False)
        enriched_df.to_csv(txt_path_2, index=False, sep='\t')
        
        if verbose:
            print("\nSaved enriched data for loader2:")
            print(f"  CSV: {csv_path_2}")
            print(f"  TXT: {txt_path_2}")
        
        return {'csv': csv_path_1, 'txt': txt_path_1}, {'csv': csv_path_2, 'txt': txt_path_2}
    
    def enrich_data(self, display_verbose: bool = True, save_verbose: bool = True) -> tuple:
        """Generate, display, and save enriched data tables.
        
        This convenience method runs the complete data enrichment pipeline:
        1. Display both input sample annotation tables
        2. Display the generated enriched table
        3. Save enriched table to CSV and TXT formats in both loader directories
        
        Args:
            display_verbose (bool): Display verbose used by this function.
            save_verbose (bool): Save verbose used by this function.
        
        Returns:
            tuple: Enrich data.
        """
        print("\n" + "=" * 80)
        print("DATA ENRICHMENT")
        print("=" * 80)
        
        # Display all tables
        self.display_all(verbose=display_verbose)
        
        # Save enriched table
        result = self.save_enriched_table(verbose=save_verbose)
        
        print("\n" + "=" * 80)
        print("DATA ENRICHMENT COMPLETED")
        print("=" * 80)
        
        return result
    
    def display_all(self, verbose: bool = True):
        """Display both input sample annotation tables and the enriched table.
        
        This method outputs:
        1. Sample Annotation 1 (with same column selection as in load_sample_annotation)
        2. Sample Annotation 2 (with same column selection as in load_sample_annotation)
        3. Enriched Table (newly generated with test conditions)
        
        Args:
            verbose (bool): Whether to emit progress output while running.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        if not verbose:
            return
            
        # Display input sample annotations
        self._display_sample_annotation(self.sample_annotation_1, "Sample Annotation 1")
        self._display_sample_annotation(self.sample_annotation_2, "Sample Annotation 2")
        
        # Generate enriched table if not already done
        if self.enriched_table is None:
            self.generate_enriched_table()
        
        # Display enriched table
        print("\n" + "=" * 80)
        print("ENRICHED TABLE:")
        print("=" * 80)
        with pd.option_context('display.max_rows', None, 'display.width', None):
            print(self.enriched_table.to_string(index=False))
        print("=" * 80)

class UniProt_BLAST_API_data_collector:
    """
    BLAST API data collector using the EBI NCBI BLAST+ REST API against UniProtKB/SwissProt.
    API docs: https://www.ebi.ac.uk/Tools/services/rest/ncbiblast
    """

    def __init__(
        self,
        input_dict,
        output_path=UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["output_path"],
        email=UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["email"],
    ):
        """Initialize the UniProt_BLAST_API_data_collector instance.
        
        Args:
            input_dict: Input value for dict.
            output_path: Path to the output.
            email: Email processed by this function.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        self.input_dict = input_dict
        self.output_path = output_path
        self.email = email
        self.base_url = UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["base_url"]

    def submit_blast(self, sequence):
        """Submit BLAST.
        
        Args:
            sequence: Sequence processed by this function.
        
        Returns:
            object: Submission result for BLAST.
        """
        params = {
            'email': self.email,
            'program': 'blastp',
            'database': 'uniprotkb_swissprot',
            'stype': 'protein',
            'sequence': sequence,
            'taxids': '9606',
            'alignments': 250,
            'scores': 250,
        }
        for attempt in range(3):
            try:
                r = requests.post(f'{self.base_url}/run', data=params, timeout=30)
                if r.status_code == 200:
                    return r.text.strip()  # job ID
                else:
                    print(f"Submit error {r.status_code}: {r.text[:200]}")
                    return None
            except requests.exceptions.ConnectionError:
                wait = (attempt + 1) * 30
                print(f"Connection error, retrying in {wait}s ({attempt+2}/3)")
                time.sleep(wait)
        return None

    def poll_status(self, job_id, max_wait=600, interval=30):
        """Poll status.
        
        Args:
            job_id: Unique identifier of the web-analysis job.
            max_wait: Max wait processed by this function.
            interval: Interval processed by this function.
        
        Returns:
            object: Polled status.
        """
        elapsed = 0
        while elapsed < max_wait:
            try:
                r = requests.get(f'{self.base_url}/status/{job_id}', timeout=30)
                status = r.text.strip()
                if status == 'FINISHED':
                    return True
                elif status in ('RUNNING', 'QUEUED'):
                    print(f"  Status: {status} ({elapsed}s)")
                    time.sleep(interval)
                    elapsed += interval
                elif status == 'FAILURE' or status == 'ERROR':
                    print(f"  Job failed: {status}")
                    return False
                else:
                    print(f"  Unknown status: {status}")
                    time.sleep(interval)
                    elapsed += interval
            except requests.exceptions.ConnectionError:
                print("  Connection error during poll, retrying...")
                time.sleep(interval)
                elapsed += interval
        print("  Timeout waiting for results")
        return False

    def get_results(self, job_id):
        """Retrieve TSV results.
        
        Args:
            job_id: Unique identifier of the web-analysis job.
        
        Returns:
            object: Requested results.
        """
        r = requests.get(f'{self.base_url}/result/{job_id}/tsv', timeout=30)
        if r.status_code == 200:
            return r.text
        print(f"  Error fetching results: {r.status_code}")
        return None

    def get_processed_sequences(self):
        """Get processed sequences.
        
        Args:
            None.
        
        Returns:
            object: Requested processed sequences.
        """
        if not os.path.exists(self.output_path):
            return set()
        try:
            df = pd.read_csv(self.output_path)
            if 'source_uniprot_id' in df.columns:
                processed = set(df['source_uniprot_id'].unique())
                print(f"Found {len(processed)} already processed sequences")
                return processed
        except Exception as e:
            print(f"Error reading existing file: {e}")
        return set()

    def run_blast_peptides(self, skip_existing=True):
        """Run BLAST peptides.
        
        Args:
            skip_existing: Skip existing processed by this function.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        peptide_dict = dict(self.input_dict)

        if skip_existing:
            processed = self.get_processed_sequences()
            peptide_dict = {k: v for k, v in peptide_dict.items() if k not in processed}
            print(f"{len(self.input_dict) - len(peptide_dict)} skipped, {len(peptide_dict)} remaining")
            if not peptide_dict:
                print("All sequences already processed!")
                return

        header_written = os.path.exists(self.output_path)

        for idx, (uniprot_id, sequence) in enumerate(peptide_dict.items(), 1):
            print(f"\n[{idx}/{len(peptide_dict)}] {uniprot_id}: {sequence}")

            try:
                # Submit
                job_id = self.submit_blast(sequence)
                if not job_id:
                    print("  Skipping (no job ID)")
                    continue
                print(f"  Job ID: {job_id}")

                # Poll
                time.sleep(15)  # initial wait
                if not self.poll_status(job_id):
                    continue

                # Get results
                tsv_text = self.get_results(job_id)
                if not tsv_text or not tsv_text.strip():
                    print("  No hits")
                    continue

                df_temp = pd.read_csv(StringIO(tsv_text), sep='\t')
                df_temp.insert(0, 'source_uniprot_id', uniprot_id)
                df_temp['original_sequence'] = sequence
                print(f"  {len(df_temp)} hits found")

                # Append to CSV
                if not header_written:
                    df_temp.to_csv(self.output_path, mode='w', index=False)
                    header_written = True
                else:
                    df_temp.to_csv(self.output_path, mode='a', index=False, header=False)

            except Exception as e:
                print(f"  Error: {e}")
                continue

            # Rate limit: EBI asks for ~1 req/sec, be generous
            if idx < len(peptide_dict):
                time.sleep(5)

        print(f"\nDone. Results: {self.output_path}")
        if os.path.exists(self.output_path):
            df = pd.read_csv(self.output_path)
            print(f"Total: {len(df)} hits from {df['source_uniprot_id'].nunique()} sequences")


class OmniPathPTMExtractor:
    """Fetch OmniPath enzyme-substrate PTM data and write pyKinaXe PTK/STK inputs."""

    BASE_COLUMNS = ["uniprot_id", "ptm_enzyme", "site", "ptm_type", "score", "source"]
    EVIDENCE_COLUMNS = [
        "evidence_level",
        "has_curated_source",
        "n_references",
        "references",
        "raw_sources",
    ]
    COLUMNS = BASE_COLUMNS + EVIDENCE_COLUMNS

    CURATED_SOURCE_LABELS = set(
        OMNIPATH_PTM_EXTRACTOR_DEFAULTS["curated_source_labels"]
    )
    EVIDENCE_RANK = dict(OMNIPATH_PTM_EXTRACTOR_DEFAULTS["evidence_rank"])

    OMNIPATH_ENZSUB_URL = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["omnipath_enzsub_url"]
    IPTMNET_API_URL = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["iptmnet_api_url"]
    UNIPROT_KINASE_URL = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["uniprot_kinase_url"]
    DEFAULT_UNIPROT_STK_PATH = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
        "default_uniprot_stk_path"
    ]
    DEFAULT_UNIPROT_PTK_PATH = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
        "default_uniprot_ptk_path"
    ]
    MANUAL_INTERACTIONS_FILENAME = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
        "manual_interactions_filename"
    ]

    def __init__(
        self,
        output_dir: Path = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["output_dir"],
        organism: int = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["organism"],
        databases: Iterable[str] | None = None,
        license_filter: str | None = None,
        timeout: int = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["timeout"],
        raw_input: Path | None = None,
        save_raw: Path | None = None,
        include_unknown_sites: bool = False,
        filter_human_kinases: bool = True,
        include_uniprot_interactions: bool = True,
        uniprot_stk_input: Path | None = None,
        uniprot_ptk_input: Path | None = None,
        include_iptmnet_rest: bool = True,
        iptmnet_api_url: str | None = None,
        iptmnet_batch_size: int = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
            "iptmnet_batch_size"
        ],
        iptmnet_site_input: Path | None = None,
        include_manual_interactions: bool = True,
        manual_interactions_input: Path | None = None,
        overwrite: bool = False,
    ):
        """Initialize the OmniPathPTMExtractor instance.
        
        Args:
            output_dir (Path): Directory containing or receiving the output.
            organism (int): Organism used by this function.
            databases (Iterable[str] | None): Databases processed by this function.
            license_filter (str | None): License filter processed by this function.
            timeout (int): Timeout used by this function.
            raw_input (Path | None): Path-like value for raw input.
            save_raw (Path | None): Path-like value for save raw.
            include_unknown_sites (bool): Boolean flag controlling whether to include unknown sites.
            filter_human_kinases (bool): Filter human kinases used by this function.
            include_uniprot_interactions (bool): Boolean flag controlling whether to include UniProt interactions.
            uniprot_stk_input (Path | None): Path-like value for UniProt STK input.
            uniprot_ptk_input (Path | None): Path-like value for UniProt PTK input.
            include_iptmnet_rest (bool): Boolean flag controlling whether to include iptmnet REST.
            iptmnet_api_url (str | None): Iptmnet API URL processed by this function.
            iptmnet_batch_size (int): Iptmnet batch size used by this function.
            iptmnet_site_input (Path | None): Path-like value for iptmnet site input.
            include_manual_interactions (bool): Boolean flag controlling whether to include manual interactions.
            manual_interactions_input (Path | None): Path-like value for manual interactions input.
            overwrite (bool): Overwrite used by this function.
        
        Returns:
            None: Constructors initialize object state in place.
        """
        self.output_dir = Path(output_dir)
        self.organism = int(organism)
        self.databases = tuple(databases or ())
        self.license_filter = license_filter
        self.timeout = int(timeout)
        self.raw_input = Path(raw_input) if raw_input else None
        self.save_raw = Path(save_raw) if save_raw else None
        self.include_unknown_sites = bool(include_unknown_sites)
        self.filter_human_kinases = bool(filter_human_kinases)
        self.include_uniprot_interactions = bool(include_uniprot_interactions)
        self.uniprot_stk_input = Path(
            uniprot_stk_input or self.DEFAULT_UNIPROT_STK_PATH
        )
        self.uniprot_ptk_input = Path(
            uniprot_ptk_input or self.DEFAULT_UNIPROT_PTK_PATH
        )
        self.include_iptmnet_rest = bool(include_iptmnet_rest)
        self.iptmnet_api_url = str(iptmnet_api_url or self.IPTMNET_API_URL).rstrip("/")
        self.iptmnet_batch_size = int(iptmnet_batch_size)
        self.iptmnet_site_input = Path(iptmnet_site_input) if iptmnet_site_input else None
        self.include_manual_interactions = bool(include_manual_interactions)
        self.manual_interactions_input = (
            Path(manual_interactions_input)
            if manual_interactions_input
            else self.output_dir / self.MANUAL_INTERACTIONS_FILENAME
        )
        self.overwrite = bool(overwrite)

        if self.iptmnet_batch_size <= 0:
            raise ValueError("iptmnet_batch_size must be > 0.")

    @staticmethod
    def _clean_uniprot_id(uid: str) -> str:
        """Return clean UniProt ID.
        
        Args:
            uid (str): Uid used by this function.
        
        Returns:
            str: Cleaned UniProt ID.
        """
        return str(uid).split("-")[0].strip()

    @staticmethod
    def _looks_like_uniprot_accession(uid: str) -> bool:
        """Return looks like UniProt accession.
        
        Args:
            uid (str): Uid used by this function.
        
        Returns:
            bool: Looks like UniProt accession.
        """
        uid = str(uid).strip().upper()
        return bool(
            re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", uid)
            or re.fullmatch(r"[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]", uid)
        )

    @staticmethod
    def _normalize_source_label(source: str) -> str:
        """Map OmniPath resource variants onto labels already used by the UKA filters.
        
        Args:
            source (str): Source used by this function.
        
        Returns:
            str: Normalized source label.
        """
        label = str(source).strip()
        normalized = re.sub(r"[^A-Za-z0-9]+", "", label).upper()

        if normalized.startswith("PHOSPHOSITE"):
            return "PhosphoSitePlus"
        if normalized == "PSP":
            return "PhosphoSitePlus"
        if normalized.startswith("PHOSPHOELM"):
            return "PhosphoELM"
        if normalized.startswith("SIGNOR"):
            return "SIGNOR"
        if normalized.startswith("HPRD"):
            return "HPRD"
        if normalized.startswith("IPTMNET"):
            return "iPTMnet"
        if normalized.startswith("NEXTPRO"):
            return "neXtProt"

        return label

    @classmethod
    def _normalize_sources(cls, raw_sources: str) -> str:
        """Normalize sources.
        
        Args:
            raw_sources (str): Raw sources used by this function.
        
        Returns:
            str: Normalized sources.
        """
        labels = []
        for source in cls._split_tokens(raw_sources):
            labels.append(cls._normalize_source_label(source))
        return ";".join(sorted(set(labels))) if labels else "OmniPath"

    @staticmethod
    def _split_tokens(raw_value: str) -> tuple[str, ...]:
        """Split tokens.
        
        Args:
            raw_value (str): Raw value used by this function.
        
        Returns:
            tuple[str, ...]: Split tokens.
        """
        tokens = []
        for token in re.split(r"[;,]", str(raw_value)):
            token = token.strip()
            if not token or token.lower() in {"nan", "none"}:
                continue
            tokens.append(token)
        return tuple(tokens)

    @classmethod
    def _normalize_raw_sources(cls, raw_sources: str) -> str:
        """Normalize raw sources.
        
        Args:
            raw_sources (str): Raw sources used by this function.
        
        Returns:
            str: Normalized raw sources.
        """
        sources = sorted(set(cls._split_tokens(raw_sources)))
        return ";".join(sources) if sources else ""

    @classmethod
    def _join_source_labels(cls, *raw_values: str) -> str:
        """Join source labels.
        
        Args:
            *raw_values (str): Additional positional arguments forwarded by this function.
        
        Returns:
            str: Join source labels.
        """
        labels = []
        for raw_value in raw_values:
            labels.extend(cls._split_tokens(raw_value))
        return ";".join(sorted(set(labels))) if labels else ""

    @classmethod
    def _normalize_references(cls, raw_references: str) -> tuple[str, ...]:
        """Normalize references.
        
        Args:
            raw_references (str): Raw references used by this function.
        
        Returns:
            tuple[str, ...]: Normalized references.
        """
        references = set()
        for token in cls._split_tokens(raw_references):
            if ":" in token:
                token = token.rsplit(":", 1)[-1]
            token = token.strip()
            if token:
                references.add(token)
        return tuple(sorted(references))

    @classmethod
    def _classify_evidence(cls, raw_sources: str, references: Iterable[str]) -> str:
        """Classify evidence.
        
        Args:
            raw_sources (str): Raw sources used by this function.
            references (Iterable[str]): References processed by this function.
        
        Returns:
            str: Classification for evidence.
        """
        normalized_sources = {
            cls._normalize_source_label(source) for source in cls._split_tokens(raw_sources)
        }
        has_curated_source = bool(normalized_sources & cls.CURATED_SOURCE_LABELS)
        has_references = bool(tuple(references))

        if has_curated_source and has_references:
            return "curated_literature"
        if has_curated_source:
            return "curated_source"
        if has_references:
            return "literature_supported"
        return "predicted_or_inferred"

    @classmethod
    def _best_evidence_level(cls, levels: Iterable[str]) -> str:
        """Return best evidence level.
        
        Args:
            levels (Iterable[str]): Levels processed by this function.
        
        Returns:
            str: Best evidence level.
        """
        best_level = "predicted_or_inferred"
        best_rank = cls.EVIDENCE_RANK[best_level]
        for level in levels:
            rank = cls.EVIDENCE_RANK.get(str(level), -1)
            if rank > best_rank:
                best_level = str(level)
                best_rank = rank
        return best_level

    @classmethod
    def _score_row(cls, row: pd.Series) -> float:
        """Return score row.
        
        Args:
            row (pd.Series): Row processed by this function.
        
        Returns:
            float: Score row.
        """
        curation_effort = pd.to_numeric(row.get("curation_effort"), errors="coerce")
        if pd.notna(curation_effort) and float(curation_effort) > 0:
            return float(curation_effort)

        n_refs = len(cls._normalize_references(row.get("references", "")))
        if n_refs:
            return float(n_refs)

        sources = cls._normalize_sources(row.get("sources", ""))
        return float(max(1, len(sources.split(";"))))

    @staticmethod
    def _format_modification(raw_modification: str) -> str:
        """Format modification.
        
        Args:
            raw_modification (str): Raw modification used by this function.
        
        Returns:
            str: Formatted modification.
        """
        modification = str(raw_modification).strip().lower()
        if modification == "phosphorylation":
            return "Phosphorylation"
        if not modification or modification in {"nan", "none"}:
            return "unknown"
        return modification.title()

    def _format_site(self, row: pd.Series) -> str | None:
        """Format site.
        
        Args:
            row (pd.Series): Row processed by this function.
        
        Returns:
            str | None: Formatted site.
        """
        residue = str(row.get("residue_type", "")).strip().upper()
        offset = str(row.get("residue_offset", "")).strip()

        if residue in {"Y", "S", "T"} and re.fullmatch(r"\d+", offset):
            return f"{residue}{offset}"

        if not self.include_unknown_sites:
            return None

        if residue == "Y":
            return "Y"
        if residue in {"S", "T"}:
            return "S/T"
        return "unknown"

    @staticmethod
    def _classify_site(site: str) -> str | None:
        """Classify site.
        
        Args:
            site (str): Site used by this function.
        
        Returns:
            str | None: Classification for site.
        """
        site = str(site).strip().upper()
        if site.startswith("Y"):
            return "ptk"
        if site.startswith(("S", "T")):
            return "stk"
        return None

    def _fetch_human_kinases(self) -> set[str]:
        """Fetch human kinases.
        
        Args:
            None.
        
        Returns:
            set[str]: Fetched human kinases.
        """
        print("Fetching reviewed human protein kinase list from UniProt...")
        response = requests.get(self.UNIPROT_KINASE_URL, timeout=self.timeout)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), sep="\t", dtype=str)
        kinase_ids = {
            self._clean_uniprot_id(uid)
            for uid in df.get("Entry", pd.Series(dtype=str)).dropna()
        }
        print(f"Found {len(kinase_ids)} reviewed human protein kinases.")
        return kinase_ids

    def _fetch_omnipath(self) -> pd.DataFrame:
        """Fetch OmniPath.
        
        Args:
            None.
        
        Returns:
            pd.DataFrame: Fetched OmniPath.
        """
        if self.raw_input:
            print(f"Loading OmniPath enzyme-substrate data from {self.raw_input}...")
            return pd.read_csv(self.raw_input, sep="\t", dtype=str)

        params = {
            "genesymbols": "1",
            "fields": "sources,references,curation_effort",
            "organisms": str(self.organism),
        }
        if self.databases:
            params["databases"] = ",".join(self.databases)
        if self.license_filter:
            params["license"] = self.license_filter

        print("Fetching OmniPath enzyme-substrate data...")
        response = requests.get(
            self.OMNIPATH_ENZSUB_URL,
            params=params,
            timeout=self.timeout,
            headers={"User-Agent": "pyKinaXe OmniPath PTM extractor"},
        )
        response.raise_for_status()

        text = response.text
        if text.lstrip().startswith("Something is not entirely good"):
            raise RuntimeError(text.strip())

        if self.save_raw:
            self.save_raw.parent.mkdir(parents=True, exist_ok=True)
            self.save_raw.write_text(text)
            print(f"Saved raw OmniPath TSV to {self.save_raw}")

        return pd.read_csv(io.StringIO(text), sep="\t", dtype=str)

    def _to_pipeline_format(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Convert pipeline format.
        
        Args:
            raw_df (pd.DataFrame): Input pandas DataFrame containing raw.
        
        Returns:
            pd.DataFrame: Converted pipeline format.
        """
        required = {"enzyme", "substrate", "residue_type", "residue_offset", "modification"}
        missing = sorted(required - set(raw_df.columns))
        if missing:
            raise ValueError(f"OmniPath response is missing required columns: {missing}")

        df = raw_df.copy()
        before = len(df)
        df = df[df["modification"].fillna("").str.lower().eq("phosphorylation")].copy()
        print(f"Kept phosphorylation rows: {before} -> {len(df)}")

        rows = []
        for _, row in df.iterrows():
            substrate = self._clean_uniprot_id(row.get("substrate", ""))
            enzyme = self._clean_uniprot_id(row.get("enzyme", ""))

            if not (
                self._looks_like_uniprot_accession(substrate)
                and self._looks_like_uniprot_accession(enzyme)
            ):
                continue

            site = self._format_site(row)
            if site is None:
                continue

            raw_sources = self._normalize_raw_sources(row.get("sources", ""))
            references = self._normalize_references(row.get("references", ""))
            evidence_level = self._classify_evidence(raw_sources, references)
            rows.append(
                {
                    "uniprot_id": substrate,
                    "ptm_enzyme": enzyme,
                    "site": site,
                    "ptm_type": self._format_modification(row.get("modification", "")),
                    "score": self._score_row(row),
                    "source": self._normalize_sources(row.get("sources", "")),
                    "evidence_level": evidence_level,
                    "has_curated_source": (
                        evidence_level in {"curated_literature", "curated_source"}
                    ),
                    "n_references": len(references),
                    "references": ";".join(references),
                    "raw_sources": raw_sources,
                }
            )

        out = pd.DataFrame(rows, columns=self.COLUMNS)
        print(f"Formatted site-specific kinase-substrate rows: {len(out)}")
        return out

    @staticmethod
    def _split_positioned_sites(raw_site: str) -> tuple[str, ...]:
        """Split positioned sites.
        
        Args:
            raw_site (str): Raw site used by this function.
        
        Returns:
            tuple[str, ...]: Split positioned sites.
        """
        sites = []
        for token in re.split(r"[;,]", str(raw_site)):
            site = token.strip().upper()
            if re.fullmatch(r"[YST]\d+", site):
                sites.append(site)
        return tuple(sorted(set(sites)))

    def _build_iptmnet_site_queries(
        self,
        interactions: pd.DataFrame,
    ) -> list[dict[str, str]]:
        """Build iptmnet site queries.
        
        Args:
            interactions (pd.DataFrame): Pandas DataFrame containing interactions.
        
        Returns:
            list[dict[str, str]]: Constructed iptmnet site queries.
        """
        if self.iptmnet_site_input:
            df_sites = pd.read_csv(
                self.iptmnet_site_input,
                sep=r"[\t, ]+",
                engine="python",
                header=None,
                dtype=str,
                usecols=[0, 1, 2],
                names=["substrate_ac", "site_residue", "site_position"],
            )
            site_rows = (
                df_sites[["substrate_ac", "site_residue", "site_position"]]
                .dropna()
                .drop_duplicates()
            )
            return [
                {
                    "substrate_ac": self._clean_uniprot_id(row["substrate_ac"]),
                    "site_residue": str(row["site_residue"]).strip().upper(),
                    "site_position": str(row["site_position"]).strip(),
                }
                for _, row in site_rows.iterrows()
                if self._looks_like_uniprot_accession(row["substrate_ac"])
                and str(row["site_residue"]).strip().upper() in {"Y", "S", "T"}
                and re.fullmatch(r"\d+", str(row["site_position"]).strip())
            ]

        seen = set()
        queries = []
        for _, row in interactions.iterrows():
            substrate = self._clean_uniprot_id(row.get("uniprot_id", ""))
            if not self._looks_like_uniprot_accession(substrate):
                continue

            for site in self._split_positioned_sites(row.get("site", "")):
                key = (substrate, site[0], site[1:])
                if key in seen:
                    continue
                seen.add(key)
                queries.append(
                    {
                        "substrate_ac": substrate,
                        "site_residue": site[0],
                        "site_position": site[1:],
                    }
                )
        return queries

    def _fetch_iptmnet_rest_rows(
        self,
        site_queries: list[dict[str, str]],
    ) -> pd.DataFrame:
        """Fetch iptmnet REST rows.
        
        Args:
            site_queries (list[dict[str, str]]): Site queries processed by this function.
        
        Returns:
            pd.DataFrame: Fetched iptmnet REST rows.
        """
        if not site_queries:
            return pd.DataFrame()

        url = f"{self.iptmnet_api_url}/batch_ptm_enzymes"
        frames = []
        print(
            "Fetching iPTMnet PTM enzyme-site data via REST API "
            f"for {len(site_queries)} substrate sites..."
        )

        for start in range(0, len(site_queries), self.iptmnet_batch_size):
            batch = site_queries[start : start + self.iptmnet_batch_size]
            response = requests.post(
                url,
                data=json.dumps(batch),
                headers={
                    "Accept": "text/plain",
                    "Content-Type": "application/json",
                    "User-Agent": "pyKinaXe iPTMnet REST PTM extractor",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            text = response.text.strip()
            if not text:
                continue

            frame = pd.read_csv(io.StringIO(text), dtype=str)
            if not frame.empty:
                frames.append(frame)

            print(
                "  iPTMnet batch "
                f"{start // self.iptmnet_batch_size + 1}: "
                f"{len(batch)} sites -> {0 if frame.empty else len(frame)} rows"
            )

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    def _format_iptmnet_rest_site(self, row: pd.Series) -> str | None:
        """Format iptmnet rest site.
        
        Args:
            row (pd.Series): Row processed by this function.
        
        Returns:
            str | None: Formatted iptmnet REST site.
        """
        site = str(row.get("site", "")).strip().upper()
        if re.fullmatch(r"[YST]\d+", site):
            return site

        residue = str(row.get("site_residue", "")).strip().upper()
        position = str(row.get("site_position", "")).strip()
        if residue in {"Y", "S", "T"} and re.fullmatch(r"\d+", position):
            return f"{residue}{position}"
        return None

    def _iptmnet_rest_to_pipeline_format(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Return iptmnet REST to pipeline format.
        
        Args:
            raw_df (pd.DataFrame): Input pandas DataFrame containing raw.
        
        Returns:
            pd.DataFrame: Iptmnet REST to pipeline format.
        """
        if raw_df.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        rows = []
        for _, row in raw_df.iterrows():
            if str(row.get("ptm_type", "")).strip().lower() != "phosphorylation":
                continue

            substrate = self._clean_uniprot_id(row.get("sub_id", ""))
            enzyme = self._clean_uniprot_id(row.get("enz_id", ""))
            if not (
                self._looks_like_uniprot_accession(substrate)
                and self._looks_like_uniprot_accession(enzyme)
            ):
                continue

            site = self._format_iptmnet_rest_site(row)
            if site is None:
                continue

            raw_source_labels = self._normalize_raw_sources(row.get("source", ""))
            normalized_sources = self._normalize_sources(raw_source_labels)
            source = self._join_source_labels("iPTMnet", normalized_sources)
            raw_sources = self._join_source_labels("iPTMnet_REST", raw_source_labels)
            references = self._normalize_references(
                row.get("pmids", row.get("pmid", ""))
            )
            evidence_level = self._classify_evidence(source, references)

            score = pd.to_numeric(row.get("score", ""), errors="coerce")
            if pd.isna(score):
                score = max(1.0, float(len(references)))

            rows.append(
                {
                    "uniprot_id": substrate,
                    "ptm_enzyme": enzyme,
                    "site": site,
                    "ptm_type": "Phosphorylation",
                    "score": float(score),
                    "source": source,
                    "evidence_level": evidence_level,
                    "has_curated_source": (
                        evidence_level in {"curated_literature", "curated_source"}
                    ),
                    "n_references": len(references),
                    "references": ";".join(references),
                    "raw_sources": raw_sources,
                }
            )

        out = pd.DataFrame(rows, columns=self.COLUMNS)
        print(f"Formatted iPTMnet REST kinase-substrate-site rows: {len(out)}")
        return out

    def _load_iptmnet_rest_interactions(
        self,
        seed_interactions: pd.DataFrame,
        existing_interactions: pd.DataFrame | None = None,
        kinase_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """Load iptmnet rest interactions.
        
        Args:
            seed_interactions (pd.DataFrame): Pandas DataFrame containing seed interactions.
            existing_interactions (pd.DataFrame | None): Pandas DataFrame containing existing interactions.
            kinase_ids (set[str] | None): Kinase IDs processed by this function.
        
        Returns:
            pd.DataFrame: Loaded iptmnet REST interactions.
        """
        if not self.include_iptmnet_rest:
            return pd.DataFrame(columns=self.COLUMNS)

        site_queries = self._build_iptmnet_site_queries(seed_interactions)
        api_rows = self._fetch_iptmnet_rest_rows(site_queries)
        interactions = self._iptmnet_rest_to_pipeline_format(api_rows)

        if kinase_ids is not None and not interactions.empty:
            before = len(interactions)
            interactions = interactions[
                interactions["ptm_enzyme"].isin(kinase_ids)
            ].reset_index(drop=True)
            print(
                "Filtered iPTMnet REST rows to UniProt human kinases: "
                f"{before} -> {len(interactions)}"
            )

        if existing_interactions is not None and not interactions.empty:
            key_columns = ["uniprot_id", "ptm_enzyme", "site"]
            existing_keys = set(
                map(
                    tuple,
                    existing_interactions[key_columns]
                    .fillna("")
                    .astype(str)
                    .itertuples(index=False, name=None),
                )
            )
            before = len(interactions)
            interaction_keys = interactions[key_columns].fillna("").astype(str).apply(tuple, axis=1)
            interactions = interactions[
                ~interaction_keys.isin(existing_keys)
            ].reset_index(drop=True)
            print(
                "Kept novel iPTMnet REST rows not already present in active "
                f"OmniPath rows: {before} -> {len(interactions)}"
            )

        return interactions

    def _process_uniprot_interactions(
        self,
        path: Path,
        kinase_type: str,
        kinase_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """Convert a saved UniProt STK/PTK TSV into generic interaction edges.
        
        Args:
            path (Path): Path value processed by this helper.
            kinase_type (str): Kinase type used by this function.
            kinase_ids (set[str] | None): Kinase IDs processed by this function.
        
        Returns:
            pd.DataFrame: Processed UniProt interactions.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing UniProt {kinase_type.upper()} file: {path}")

        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        required = {"Entry", "Interacts with"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"UniProt file {path} is missing columns: {missing}")

        kinase_type = str(kinase_type).lower()
        if kinase_type not in {"ptk", "stk"}:
            raise ValueError("kinase_type must be 'ptk' or 'stk'")
        generic_site = "Y" if kinase_type == "ptk" else "S/T"

        rows = []
        for _, row in df.iterrows():
            enzyme = self._clean_uniprot_id(row.get("Entry", ""))
            if not self._looks_like_uniprot_accession(enzyme):
                continue
            if kinase_ids is not None and enzyme not in kinase_ids:
                continue

            partners = set()
            for raw_partner in self._split_tokens(row.get("Interacts with", "")):
                if raw_partner.startswith("PRO_"):
                    partners.update(
                        self._clean_uniprot_id(match)
                        for match in re.findall(r"\[([A-Z0-9-]+)\]", raw_partner)
                    )
                else:
                    partners.add(self._clean_uniprot_id(raw_partner))

            for partner in sorted(partners):
                if partner == enzyme or not self._looks_like_uniprot_accession(partner):
                    continue
                rows.append(
                    {
                        "uniprot_id": partner,
                        "ptm_enzyme": enzyme,
                        "site": generic_site,
                        "ptm_type": "Phosphorylation",
                        "score": 1.0,
                        "source": "UniProt_InteractsWith",
                        "evidence_level": "curated_source",
                        "has_curated_source": True,
                        "n_references": 0,
                        "references": "",
                        "raw_sources": "UniProt_InteractsWith",
                    }
                )

        out = pd.DataFrame(rows, columns=self.COLUMNS)
        print(
            f"Formatted UniProt {kinase_type.upper()} interaction rows: {len(out)}"
        )
        return out

    def _load_uniprot_interactions(
        self,
        kinase_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """Load UniProt interactions.
        
        Args:
            kinase_ids (set[str] | None): Kinase IDs processed by this function.
        
        Returns:
            pd.DataFrame: Loaded UniProt interactions.
        """
        if not self.include_uniprot_interactions:
            return pd.DataFrame(columns=self.COLUMNS)

        frames = [
            self._process_uniprot_interactions(
                self.uniprot_ptk_input,
                "ptk",
                kinase_ids=kinase_ids,
            ),
            self._process_uniprot_interactions(
                self.uniprot_stk_input,
                "stk",
                kinase_ids=kinase_ids,
            ),
        ]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=self.COLUMNS)

        out = pd.concat(frames, ignore_index=True)
        print(f"Loaded local UniProt interaction rows: {len(out)}")
        return out

    def _ensure_manual_interactions_file(self) -> None:
        """Return ensure manual interactions file.
        
        Args:
            None.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        path = self.manual_interactions_input
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return

        pd.DataFrame(columns=self.COLUMNS).to_csv(path, index=False)
        print(f"Created empty manual interactions CSV: {path}")

    def _load_manual_interactions(
        self,
        kinase_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """Load manual interactions.
        
        Args:
            kinase_ids (set[str] | None): Kinase IDs processed by this function.
        
        Returns:
            pd.DataFrame: Loaded manual interactions.
        """
        if not self.include_manual_interactions:
            return pd.DataFrame(columns=self.COLUMNS)

        self._ensure_manual_interactions_file()
        manual = pd.read_csv(self.manual_interactions_input, dtype=str).fillna("")
        missing = sorted(set(self.BASE_COLUMNS) - set(manual.columns))
        if missing:
            raise ValueError(
                f"Manual interactions file {self.manual_interactions_input} "
                f"is missing required columns: {missing}"
            )

        for column in self.COLUMNS:
            if column not in manual.columns:
                manual[column] = ""

        manual = manual[self.COLUMNS].copy()
        manual = manual[
            manual["uniprot_id"].astype(str).str.strip().ne("")
            & manual["ptm_enzyme"].astype(str).str.strip().ne("")
            & manual["site"].astype(str).str.strip().ne("")
        ].copy()
        if manual.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        manual["uniprot_id"] = manual["uniprot_id"].apply(self._clean_uniprot_id)
        manual["ptm_enzyme"] = manual["ptm_enzyme"].apply(self._clean_uniprot_id)
        manual["site"] = manual["site"].astype(str).str.strip().str.upper()
        manual["ptm_type"] = manual["ptm_type"].replace("", "Phosphorylation")
        manual["score"] = pd.to_numeric(manual["score"], errors="coerce").fillna(1.0)
        manual["source"] = manual["source"].mask(manual["source"].eq(""), "Manual")
        manual["evidence_level"] = manual["evidence_level"].mask(
            manual["evidence_level"].eq(""),
            "curated_source",
        )
        manual["has_curated_source"] = manual["has_curated_source"].mask(
            manual["has_curated_source"].eq(""),
            "true",
        )
        manual["has_curated_source"] = (
            manual["has_curated_source"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
        manual["n_references"] = (
            pd.to_numeric(manual["n_references"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        manual["raw_sources"] = manual["raw_sources"].mask(
            manual["raw_sources"].eq(""),
            "Manual",
        )

        manual = manual[
            manual["uniprot_id"].apply(self._looks_like_uniprot_accession)
            & manual["ptm_enzyme"].apply(self._looks_like_uniprot_accession)
        ].copy()

        if kinase_ids is not None and not manual.empty:
            before = len(manual)
            manual = manual[manual["ptm_enzyme"].isin(kinase_ids)].reset_index(drop=True)
            print(
                "Filtered manual interaction rows to UniProt human kinases: "
                f"{before} -> {len(manual)}"
            )

        print(f"Loaded manual interaction rows: {len(manual)}")
        return manual

    @staticmethod
    def _merge_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Merge duplicates.
        
        Args:
            df (pd.DataFrame): Input pandas DataFrame used by this function.
        
        Returns:
            pd.DataFrame: Merged duplicates.
        """
        if df.empty:
            return pd.DataFrame(columns=OmniPathPTMExtractor.COLUMNS)

        def aggregate(group: pd.DataFrame) -> dict[str, object]:
            """Return aggregate.
            
            Args:
                group (pd.DataFrame): Pandas DataFrame containing group.
            
            Returns:
                dict[str, object]: Aggregate.
            """
            sites = sorted(set(group["site"].dropna().astype(str)))
            sources = sorted(
                {
                    source.strip()
                    for raw_sources in group["source"].dropna().astype(str)
                    for source in raw_sources.split(";")
                    if source.strip()
                }
            )
            raw_sources = sorted(
                {
                    source.strip()
                    for raw in group["raw_sources"].dropna().astype(str)
                    for source in raw.split(";")
                    if source.strip()
                }
            )
            references = sorted(
                {
                    reference.strip()
                    for raw in group["references"].dropna().astype(str)
                    for reference in raw.split(";")
                    if reference.strip()
                }
            )
            evidence_level = OmniPathPTMExtractor._best_evidence_level(
                group["evidence_level"].dropna().astype(str)
            )
            return {
                "site": ";".join(sites) if sites else "unknown",
                "ptm_type": "Phosphorylation",
                "score": float(pd.to_numeric(group["score"], errors="coerce").max()),
                "source": ";".join(sources) if sources else "OmniPath",
                "evidence_level": evidence_level,
                "has_curated_source": bool(group["has_curated_source"].fillna(False).any()),
                "n_references": len(references),
                "references": ";".join(references),
                "raw_sources": ";".join(raw_sources),
            }

        rows = []
        for (substrate, enzyme), group in df.groupby(["uniprot_id", "ptm_enzyme"]):
            rows.append(
                {
                    "uniprot_id": substrate,
                    "ptm_enzyme": enzyme,
                    **aggregate(group),
                }
            )

        return pd.DataFrame(rows, columns=OmniPathPTMExtractor.COLUMNS)

    def _write_outputs(self, ptk: pd.DataFrame, stk: pd.DataFrame) -> None:
        """Write outputs.
        
        Args:
            ptk (pd.DataFrame): Pandas DataFrame containing PTK.
            stk (pd.DataFrame): Pandas DataFrame containing STK.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ptk_path = self.output_dir / "ptk_interactions.csv"
        stk_path = self.output_dir / "stk_interactions.csv"

        existing = [path for path in (ptk_path, stk_path) if path.exists()]
        if existing and not self.overwrite:
            existing_str = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Refusing to overwrite existing file(s): {existing_str}. "
                "Pass --overwrite if this is intentional."
            )

        ptk.to_csv(ptk_path, index=False)
        stk.to_csv(stk_path, index=False)
        print(f"Wrote PTK interactions: {len(ptk)} -> {ptk_path}")
        print(f"Wrote STK interactions: {len(stk)} -> {stk_path}")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Handle run.
        
        Args:
            None.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Run.
        """
        raw_df = self._fetch_omnipath()
        interactions = self._to_pipeline_format(raw_df)
        iptmnet_seed_interactions = interactions.copy()

        kinase_ids = None
        if self.filter_human_kinases:
            kinase_ids = self._fetch_human_kinases()
            before = len(interactions)
            interactions = interactions[
                interactions["ptm_enzyme"].isin(kinase_ids)
            ].reset_index(drop=True)
            print(f"Filtered to UniProt human kinases: {before} -> {len(interactions)}")

        iptmnet_rest_interactions = self._load_iptmnet_rest_interactions(
            iptmnet_seed_interactions,
            existing_interactions=interactions,
            kinase_ids=kinase_ids,
        )
        if not iptmnet_rest_interactions.empty:
            before = len(interactions)
            interactions = pd.concat(
                [interactions, iptmnet_rest_interactions],
                ignore_index=True,
            )
            print(
                "Added iPTMnet REST rows to OmniPath rows: "
                f"{before} -> {len(interactions)}"
            )

        uniprot_interactions = self._load_uniprot_interactions(kinase_ids=kinase_ids)
        if not uniprot_interactions.empty:
            before = len(interactions)
            interactions = pd.concat(
                [interactions, uniprot_interactions],
                ignore_index=True,
            )
            print(
                "Added local UniProt interaction rows to OmniPath rows: "
                f"{before} -> {len(interactions)}"
            )

        manual_interactions = self._load_manual_interactions(kinase_ids=kinase_ids)
        if not manual_interactions.empty:
            before = len(interactions)
            interactions = pd.concat(
                [interactions, manual_interactions],
                ignore_index=True,
            )
            print(
                "Added manual interaction rows to PTM rows: "
                f"{before} -> {len(interactions)}"
            )

        interactions["_array_type"] = interactions["site"].apply(self._classify_site)
        unknown_array_type = interactions["_array_type"].isna().sum()
        if unknown_array_type:
            print(f"Dropping rows without PTK/STK-compatible sites: {unknown_array_type}")

        ptk = interactions[interactions["_array_type"] == "ptk"][self.COLUMNS].copy()
        stk = interactions[interactions["_array_type"] == "stk"][self.COLUMNS].copy()

        ptk_before = len(ptk)
        stk_before = len(stk)
        ptk = self._merge_duplicates(ptk)
        stk = self._merge_duplicates(stk)
        print(f"PTK merged duplicate substrate-kinase pairs: {ptk_before} -> {len(ptk)}")
        print(f"STK merged duplicate substrate-kinase pairs: {stk_before} -> {len(stk)}")

        self._write_outputs(ptk, stk)
        return ptk, stk


def _split_csv_values(raw_values: str | None) -> tuple[str, ...]:
    """Split CSV values.
    
    Args:
        raw_values (str | None): Raw values processed by this function.
    
    Returns:
        tuple[str, ...]: Split CSV values.
    """
    if not raw_values:
        return tuple()
    return tuple(value.strip() for value in raw_values.split(",") if value.strip())


class Kinase_Liver_Extractor:
    """
    Class to retriev a list of all human kinases in the liver.

    Attributes:
        path_output (Path): Path to save the output file with the list of human kinases in the liver.
        
    Example usage:
        Kinase_Liver_Extractor = Kinase_Liver_Extractor()

        df_kinases_in_liver = Kinase_Liver_Extractor.create_list(flag_save=True)
        
    """

    def __init__(
        self,
        path_output=KINASE_LIVER_EXTRACTOR_DEFAULTS["output_path"],
    ):
        """Initializes the KinaseProteinInteractionExtractor with paths to the raw data files of the various PTM databases and the output path for the processed interactions. The constructor checks if the provided paths are valid and sets them as attributes of the class instance.
        
        :param path_output: Path to save the output file with the list of human kinases in the liver.
        
        Args:
            path_output: Path to the output.
        
        Returns:
            None: Constructors initialize object state in place.
        """

        self.path_output = Path(path_output)  

        #parameters for uniprot API calls
        self.api_format = KINASE_LIVER_EXTRACTOR_DEFAULTS["api_format"]
        self.api_taxonomy_id = KINASE_LIVER_EXTRACTOR_DEFAULTS["api_taxonomy_id"]
        self.api_keyword = KINASE_LIVER_EXTRACTOR_DEFAULTS["api_keyword"]
        self.api_fields = KINASE_LIVER_EXTRACTOR_DEFAULTS["api_fields"]

        # Search query for human liver kinases
        self.url = KINASE_LIVER_EXTRACTOR_DEFAULTS["url"]
        self.params = dict(KINASE_LIVER_EXTRACTOR_DEFAULTS["params"])

    def _get_human_kinases(self):
        """Loads list of human kinases from uniprot.org.
        This list is used for later comparison with the kinase-protein interactions from the PTM databases, to filter for interactions that involve human kinases.
        
        Args:
            None.
        
        Returns:
            object: Requested human kinases.
        """

        try:
            response = requests.get(self.url, params=self.params)
            response.raise_for_status()
            
            # Convert TSV response to DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), sep='\t')
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from UniProt: {e}")
            return pd.DataFrame(columns=['Entry', 'Gene Names', 'Protein names', 'Tissue specificity'])
        
    def _save_data(self, df_kinases_in_liver):
        """Function to save the list of human kinases in the liver to the specified output path and also to an archive directory with a timestamp.
        
        :param df_kinases_in_liver: DataFrame containing the list of human kinases in the liver.
        
        Args:
            df_kinases_in_liver: Input pandas DataFrame containing kinases in liver.
        
        Returns:
            None: This function is used for side effects and does not return a value.
        """

        # Create archive directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(self.path_output), 'Archive'), exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # Save to main output path
        df_kinases_in_liver.to_csv(self.path_output, index=False)

        # Save to archiv with timestamp
        archiv_filename = f"{timestamp}_{os.path.basename(self.path_output)}"
        archiv_path = os.path.join('data/external/UniProt/Archive', archiv_filename)
        df_kinases_in_liver.to_csv(archiv_path, index=False)

        return None
        
        
    def create_list(self, flag_save=True):
        """Creates a list of human kinases in the liver and saves it to the specified output path if flag_save is True.
        
        :param flag_save: Boolean flag to indicate whether to save the output file.
        
        :return: DataFrame with the list of human kinases in the liver.
        
        Args:
            flag_save: Flag save processed by this function.
        
        Returns:
            object: Created list.
        """

        df_kinases_in_liver = self._get_human_kinases()

        if flag_save:
            self._save_data(df_kinases_in_liver)

        return df_kinases_in_liver




def _build_blast_arg_parser(subparsers):
    """Build BLAST arg parser.
    
    Args:
        subparsers: Subparsers processed by this function.
    
    Returns:
        object: Constructed BLAST arg parser.
    """
    parser = subparsers.add_parser(
        "blast",
        help="Run the UniProt BLAST API collector on enrichment_peptides.csv.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["input_path"],
        help="CSV with peptide IDs and sequences.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["output_path"],
        help="Output CSV path for appended BLAST hits.",
    )
    parser.add_argument(
        "--email",
        default=UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["email"],
        help="Email address required by the EBI BLAST API.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip sequences that are already present in the output CSV.",
    )
    return parser


def _run_blast_cli(args) -> None:
    """Run BLAST CLI.
    
    Args:
        args: Args processed by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    df_raw = pd.read_csv(args.input)
    sequence_dict = dict(zip(df_raw['ID'], df_raw['Sequence']))
    collector = UniProt_BLAST_API_data_collector(
        input_dict=sequence_dict,
        output_path=str(args.output),
        email=args.email,
    )
    collector.run_blast_peptides(skip_existing=not args.no_skip_existing)


def _build_omnipath_arg_parser(subparsers):
    """Build OmniPath arg parser.
    
    Args:
        subparsers: Subparsers processed by this function.
    
    Returns:
        object: Constructed OmniPath arg parser.
    """
    parser = subparsers.add_parser(
        "omnipath",
        help=(
            "Fetch OmniPath enzyme-substrate phosphorylation data and create "
            "ptk_interactions.csv/stk_interactions.csv for the UKA/KPEA pipeline."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OMNIPATH_PTM_EXTRACTOR_DEFAULTS["output_dir"],
        help="Output directory for ptk_interactions.csv and stk_interactions.csv.",
    )
    parser.add_argument(
        "--organism",
        type=int,
        default=OMNIPATH_PTM_EXTRACTOR_DEFAULTS["organism"],
        help="NCBI taxonomy ID passed to OmniPath as organisms=...",
    )
    parser.add_argument(
        "--databases",
        default=None,
        help="Optional comma-separated OmniPath resource filter.",
    )
    parser.add_argument(
        "--license",
        dest="license_filter",
        default=None,
        help="Optional OmniPath license filter, e.g. commercial.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=OMNIPATH_PTM_EXTRACTOR_DEFAULTS["timeout"],
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=None,
        help="Use an already downloaded OmniPath enz_sub TSV instead of fetching.",
    )
    parser.add_argument(
        "--save-raw",
        type=Path,
        default=None,
        help="Optional path where the raw OmniPath TSV should be saved.",
    )
    parser.add_argument(
        "--include-unknown-sites",
        action="store_true",
        help="Keep phosphorylation rows without exact residue offsets as generic Y or S/T sites.",
    )
    parser.add_argument(
        "--no-kinase-filter",
        action="store_true",
        help="Do not filter enzymes against the UniProt human kinase keyword list.",
    )
    uniprot_group = parser.add_mutually_exclusive_group()
    uniprot_group.add_argument(
        "--include-uniprot-interactions",
        dest="include_uniprot_interactions",
        action="store_true",
        help="Append locally saved UniProt STK/PTK interaction partners as generic phosphorylation edges.",
    )
    uniprot_group.add_argument(
        "--no-uniprot-interactions",
        dest="include_uniprot_interactions",
        action="store_false",
        help="Do not append locally saved UniProt STK/PTK interaction partners.",
    )
    parser.set_defaults(include_uniprot_interactions=True)
    parser.add_argument(
        "--uniprot-stk-input",
        type=Path,
        default=OmniPathPTMExtractor.DEFAULT_UNIPROT_STK_PATH,
        help="Saved UniProt STK TSV used with --include-uniprot-interactions.",
    )
    parser.add_argument(
        "--uniprot-ptk-input",
        type=Path,
        default=OmniPathPTMExtractor.DEFAULT_UNIPROT_PTK_PATH,
        help="Saved UniProt PTK TSV used with --include-uniprot-interactions.",
    )
    iptmnet_group = parser.add_mutually_exclusive_group()
    iptmnet_group.add_argument(
        "--include-iptmnet-rest",
        dest="include_iptmnet_rest",
        action="store_true",
        help="Append iPTMnet PTM enzyme-site relationships from the REST API.",
    )
    iptmnet_group.add_argument(
        "--no-iptmnet-rest",
        dest="include_iptmnet_rest",
        action="store_false",
        help="Do not query iPTMnet REST for additional kinase-substrate-site edges.",
    )
    parser.set_defaults(include_iptmnet_rest=True)
    parser.add_argument(
        "--iptmnet-api-url",
        default=OmniPathPTMExtractor.IPTMNET_API_URL,
        help="Base iPTMnet API URL used for REST requests.",
    )
    parser.add_argument(
        "--iptmnet-batch-size",
        type=int,
        default=OMNIPATH_PTM_EXTRACTOR_DEFAULTS["iptmnet_batch_size"],
        help="Number of substrate sites per iPTMnet batch_ptm_enzymes request.",
    )
    parser.add_argument(
        "--iptmnet-site-input",
        type=Path,
        default=None,
        help="Optional headerless substrate-site file for iPTMnet REST queries.",
    )
    manual_group = parser.add_mutually_exclusive_group()
    manual_group.add_argument(
        "--include-manual-interactions",
        dest="include_manual_interactions",
        action="store_true",
        help="Append manually curated/test interactions from manual_interactions.csv in the output directory.",
    )
    manual_group.add_argument(
        "--no-manual-interactions",
        dest="include_manual_interactions",
        action="store_false",
        help="Do not create or append manual interaction rows.",
    )
    parser.set_defaults(include_manual_interactions=True)
    parser.add_argument(
        "--manual-interactions-input",
        type=Path,
        default=None,
        help="Optional manual interaction CSV path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing ptk_interactions.csv/stk_interactions.csv.",
    )
    return parser


def _run_omnipath_cli(args) -> None:
    """Run OmniPath CLI.
    
    Args:
        args: Args processed by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    extractor = OmniPathPTMExtractor(
        output_dir=args.output,
        organism=args.organism,
        databases=_split_csv_values(args.databases),
        license_filter=args.license_filter,
        timeout=args.timeout,
        raw_input=args.raw_input,
        save_raw=args.save_raw,
        include_unknown_sites=args.include_unknown_sites,
        filter_human_kinases=not args.no_kinase_filter,
        include_uniprot_interactions=args.include_uniprot_interactions,
        uniprot_stk_input=args.uniprot_stk_input,
        uniprot_ptk_input=args.uniprot_ptk_input,
        include_iptmnet_rest=args.include_iptmnet_rest,
        iptmnet_api_url=args.iptmnet_api_url,
        iptmnet_batch_size=args.iptmnet_batch_size,
        iptmnet_site_input=args.iptmnet_site_input,
        include_manual_interactions=args.include_manual_interactions,
        manual_interactions_input=args.manual_interactions_input,
        overwrite=args.overwrite,
    )
    extractor.run()


def _build_liver_kinases_arg_parser(subparsers):
    """Build liver kinases arg parser.
    
    Args:
        subparsers: Subparsers processed by this function.
    
    Returns:
        object: Constructed liver kinases arg parser.
    """
    parser = subparsers.add_parser(
        "liver-kinases",
        help="Fetch the UniProt list of human kinases with liver tissue specificity.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=KINASE_LIVER_EXTRACTOR_DEFAULTS["output_path"],
        help="CSV output path.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the fetched kinase list to disk.",
    )
    return parser


def _run_liver_kinases_cli(args) -> None:
    """Run liver kinases CLI.
    
    Args:
        args: Args processed by this function.
    
    Returns:
        None: This function is used for side effects and does not return a value.
    """
    extractor = Kinase_Liver_Extractor(path_output=args.output)
    extractor.create_list(flag_save=not args.no_save)


def build_cli_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arg parser.
    
    Args:
        None.
    
    Returns:
        argparse.ArgumentParser: Constructed CLI arg parser.
    """
    parser = argparse.ArgumentParser(
        description="Data enrichment and supporting data-collection utilities for the KX pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")
    _build_blast_arg_parser(subparsers)
    _build_omnipath_arg_parser(subparsers)
    _build_liver_kinases_arg_parser(subparsers)
    return parser


def main() -> None:
    """Run the module as a command-line entry point.
    
    Args:
        None.
    
    Returns:
        None: The command-line entry point runs for its side effects.
    """
    parser = build_cli_arg_parser()
    args = parser.parse_args()

    if args.command == "blast":
        _run_blast_cli(args)
    elif args.command == "omnipath":
        _run_omnipath_cli(args)
    elif args.command == "liver-kinases":
        _run_liver_kinases_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
