"""Build the experimental-design table from the PTK and STK sample annotations.

DataEnricher combines the sample annotations of the two DataLoader instances,
parses the sample names (construct, biological and technical replicate),
assigns the control/test roles from the c/t prefix (falling back to
alphabetical construct order) and writes the enrichment table the downstream
stages read.

The module also contains the collectors that build the reference tables in
data/external (UniProt_BLAST_API_data_collector, OmniPathPTMExtractor and the
liver-kinase list extraction), exposed through the command-line interface at
the end of the module.
"""

import argparse
import os
from datetime import datetime
import io
from io import StringIO
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

# Allow running this file directly (``python src/kx_data_enricher.py omnipath``)
# without installing the package or setting PYTHONPATH: put the repository root
# (for the ``config`` package) and ``src`` (for sibling modules) on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_dir in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

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
    COLUMN_CONDITION_ROLE = DATA_ENRICHER_DEFAULTS["columns"]["condition_role"]
    
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
    # A Test Condition IS the sample name (the sample name minus the c/t prefix
    # and minus the trailing replicate numbers), so 'c2_STR_CTL_2_2' and
    # 't1_pSHDAg' become 'STR_CTL' and 'pSHDAg'. Samples are therefore matched
    # BY NAME: the same name is the same condition, no matter which well, chip
    # or array it sits in. The control/test ROLE is carried separately, in the
    # Condition Role column.
    CONDITION_ROLE_CONTROL = "control"
    CONDITION_ROLE_TEST = "test"
    # Name of the role column in the FINAL enriched table (COLUMN_CONDITION_ROLE
    # above is the internal working column).
    COLUMN_CONDITION_ROLE_OUTPUT = "Condition Role"

    # ========== Control/Test Chip Prefix ==========
    # Optional prefix at the START of a sample name that marks a control
    # ('c') or test ('t') well, an optional chip number, then a REQUIRED
    # underscore separator (e.g. 'c1_', 't2_', 'c_', 't_'). Matched
    # case-insensitively. The trailing underscore avoids matching ordinary
    # names that merely start with 'c' or 't'.
    #
    # Only the LETTER is used: it supplies the control/test role. The number is
    # the chip index and is parsed for completeness -- conditions are matched by
    # sample name, so 't1_pSHDAg' and 't2_pSHDAg' are the same condition on two
    # chips and the number never enters the grouping.
    CT_PREFIX_PATTERN = re.compile(r'^([ct])(\d*)_', re.IGNORECASE)
    
    def __init__(self, loader1: DataLoader, loader2: DataLoader, cell_line: Optional[str] = None):
        """Initializes the DataEnricher with two DataLoader instances."""
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
            timestamp (str): Timestamp string associated with the current analysis run.
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
        r"""Parse sample name to extract the last two numbers (biological and technical replicates) and 'first_part' (sample name without those numbers).
        
        The last two numbers can be in any format (Roman or Arabic) and are separated
        by various possible separators (., , _ - / \ | : space).
        Separators can be different (e.g., "puc18_1.1" has _ then .).
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
        """Convert a number string (Roman or Arabic) to integer."""
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
        """Check if a replicate column exists and has non-zero values."""
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
    
    def _parse_ct_prefix(self, sample_name: str) -> Optional[tuple]:
        """Parse an optional control/test chip prefix at the start of a name.

        Recognizes 'c'/'t' (case-insensitive), an optional chip number, and a
        REQUIRED underscore separator, e.g. 'c1_Mock', 't2_pORF3', 'c_Mock',
        't_pORF3'. The role comes from the letter; the number is returned for
        completeness but does not affect how conditions are grouped (that is
        done by sample name).

        Args:
            sample_name (str): Raw sample name to inspect.

        Returns:
            Optional[tuple]: ``(role, chip_number)`` where ``role`` is
            ``'control'`` or ``'test'`` and ``chip_number`` is the parsed chip
            integer (or ``None`` when no digits were given). Returns ``None``
            when the name has no valid c/t prefix.
        """
        match = self.CT_PREFIX_PATTERN.match(str(sample_name).strip())
        if not match:
            return None
        role = 'control' if match.group(1).lower() == 'c' else 'test'
        digits = match.group(2)
        chip_number = int(digits) if digits else None
        return role, chip_number

    def _strip_ct_prefix(self, name: str) -> str:
        """Remove a leading control/test chip prefix from a construct string.

        The prefix (e.g. 'c1_', 't2_', 'c_', 't_') marks the control/test role
        and the chip; it is not part of the biological construct itself. If
        removing the prefix would leave an empty string, the original value is
        returned unchanged.

        Args:
            name (str): Construct string (a sample name minus any trailing
                replicate numbers).

        Returns:
            str: The construct with any leading c/t prefix removed.
        """
        if name is None:
            return name
        stripped = self.CT_PREFIX_PATTERN.sub('', str(name), count=1)
        return stripped if stripped else str(name)

    def _source_uses_ct_logic(self, source_df: pd.DataFrame) -> bool:
        """Decide whether a chip type consistently uses the c/t prefix scheme.

        The logical-chip logic is applied only when EVERY non-empty sample name
        in the source carries a valid c/t prefix; otherwise the source falls
        back to the alphabetical construct-based assignment.

        Args:
            source_df (pd.DataFrame): Rows for a single ``_source`` value.

        Returns:
            bool: True when all non-empty sample names have a c/t prefix.
        """
        names = [
            str(name).strip()
            for name in source_df[self.COLUMN_SAMPLE_NAME]
            if str(name).strip() and str(name).strip().lower() != 'nan'
        ]
        if not names:
            return False
        return all(self._parse_ct_prefix(name) is not None for name in names)
    
    def _assign_logical_chip_conditions(
        self, df: pd.DataFrame, source_indices: list
    ) -> None:
        """Label wells by their sample name, with the role from the c/t prefix.

        The prefix carries the ROLE and nothing else: ``c`` marks a control
        well, ``t`` a test well, and the number after it is the chip index
        (``t1_pSHDAg`` and ``t2_pSHDAg`` are the same condition measured on two
        chips). Everything after the prefix, minus any trailing replicate
        numbers, is the sample name and becomes the Test Condition:

            c2_STR_CTL_2_2  ->  Test Condition 'STR_CTL',  role control
            t1_pSHDAg       ->  Test Condition 'pSHDAg',   role test
            t2_Exer_HPC_1_2 ->  Test Condition 'Exer_HPC', role test

        Because the label comes from the name, samples match across wells,
        chips and BOTH arrays even when the two annotation files list their
        wells in a different order -- which the earlier positional numbering
        (Control1, Control2, ...) got wrong whenever the orders disagreed.

        Args:
            df (pd.DataFrame): Working table; modified in place.
            source_indices (list): Row indices for one source, in file order.
        """
        for idx in source_indices:
            sample_name = df.loc[idx, self.COLUMN_SAMPLE_NAME]
            parsed = self._parse_ct_prefix(sample_name)
            if parsed is None:
                # Guarded for safety; _source_uses_ct_logic guarantees every
                # name in this source has a prefix before we get here.
                continue
            role, _chip_number = parsed
            first_part, _, _ = self._parse_sample_name(sample_name)
            df.loc[idx, self.COLUMN_TEST_CONDITION] = self._strip_ct_prefix(first_part)
            df.loc[idx, self.COLUMN_CONDITION_ROLE] = (
                self.CONDITION_ROLE_CONTROL
                if role == 'control'
                else self.CONDITION_ROLE_TEST
            )
    
    def _assign_fallback_conditions(
        self, df: pd.DataFrame, source_mask: pd.Series
    ) -> None:
        """Label wells by sample name, inferring the role alphabetically.

        Used for chip types whose sample names carry no c/t prefix, so the role
        cannot be read off the name. The Test Condition is the sample name here
        too (samples still match by name); only the ROLE has to be guessed:
        within each (Technical Replicate, Biological Replicate) group the
        alphabetically smallest construct is taken as the control, the rest as
        tests. That is the historical rule -- with a c/t prefix nothing is
        guessed, which is why the prefix scheme is preferred.

        Args:
            df (pd.DataFrame): Working table with ``_tech_rep``/``_bio_rep``
                helper columns; modified in place.
            source_mask (pd.Series): Boolean mask selecting one source.
        """
        # Get unique Technical Replicates for this source
        tech_rep_values = df[source_mask]['_tech_rep'].unique()

        for tech_rep_val in tech_rep_values:
            # NaN never compares equal, so match it with isna()
            if pd.isna(tech_rep_val):
                tech_rep_mask = source_mask & df['_tech_rep'].isna()
            else:
                tech_rep_mask = source_mask & (df['_tech_rep'] == tech_rep_val)

            # Get unique Biological Replicates within this Technical Replicate group
            bio_rep_values = df[tech_rep_mask]['_bio_rep'].unique()

            for bio_rep_val in bio_rep_values:
                if pd.isna(bio_rep_val):
                    bio_rep_mask = tech_rep_mask & df['_bio_rep'].isna()
                else:
                    bio_rep_mask = tech_rep_mask & (df['_bio_rep'] == bio_rep_val)

                group_indices = df[bio_rep_mask].index.tolist()

                # Skip if no rows for this group (shouldn't happen, but safety)
                if len(group_indices) == 0:
                    continue

                # Parse sample names to get first parts (constructs)
                first_parts = []
                for idx in group_indices:
                    sample_name = df.loc[idx, self.COLUMN_SAMPLE_NAME]
                    first_part, _, _ = self._parse_sample_name(sample_name)
                    first_parts.append(first_part)

                # The label is the sample name itself; only the role is
                # inferred, and the alphabetically smallest construct gets it.
                unique_groups = sorted(set(first_parts))
                role_map = {
                    group: (
                        self.CONDITION_ROLE_CONTROL
                        if group == unique_groups[0]
                        else self.CONDITION_ROLE_TEST
                    )
                    for group in unique_groups
                }

                for i, idx in enumerate(group_indices):
                    df.loc[idx, self.COLUMN_TEST_CONDITION] = first_parts[i]
                    df.loc[idx, self.COLUMN_CONDITION_ROLE] = role_map[first_parts[i]]
    
    def _determine_test_conditions(self, df_with_source: pd.DataFrame) -> tuple:
        """Determine the Test Condition and its role for every row.

        In BOTH schemes the Test Condition is the SAMPLE NAME -- the sample name
        minus a leading c/t prefix and minus the trailing replicate numbers
        ('t2_Exer_HPC_1_2' -> 'Exer_HPC'). Samples are matched by that name, so
        one name is one condition across wells, chips and both arrays. The
        schemes differ only in how the control/test ROLE is established, and the
        choice is made independently per chip type (``_source``):

        1. c/t prefix scheme (preferred): used when every sample name in the
           source starts with 'c'/'t', an optional chip number, then '_'
           (e.g. 'c1_Mock', 't2_pORF3'). The prefix states the role outright --
           nothing is inferred (see _assign_logical_chip_conditions).

        2. Alphabetical fallback: used when the source does not consistently use
           the c/t prefix. Within each (Technical Replicate, Biological
           Replicate) group the alphabetically smallest construct is taken as
           the control, the rest as tests (see _assign_fallback_conditions).

        Args:
            df_with_source (pd.DataFrame): Combined table with a ``_source``
                column and the sample-name/replicate information.

        Returns:
            tuple: ``(labels, roles)`` -- the Test Condition label and the
            control/test role for each row, in the input row order.
        """
        # Create a copy to avoid modifying original
        df = df_with_source.copy()
        df[self.COLUMN_TEST_CONDITION] = ''
        df[self.COLUMN_CONDITION_ROLE] = ''

        # Extract technical and biological replicate information for all rows.
        # These feed the alphabetical fallback path and mirror the values later
        # written to the enriched table.
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

        # Assign conditions independently per source, using the logical-chip
        # scheme when the source consistently uses the c/t prefix, otherwise the
        # alphabetical construct fallback.
        for source in df['_source'].unique():
            source_mask = df['_source'] == source
            source_df = df[source_mask]
            if self._source_uses_ct_logic(source_df):
                self._assign_logical_chip_conditions(df, source_df.index.tolist())
            else:
                self._assign_fallback_conditions(df, source_mask)

        return (
            df[self.COLUMN_TEST_CONDITION].tolist(),
            df[self.COLUMN_CONDITION_ROLE].tolist(),
        )
    
    def generate_enriched_table(self) -> pd.DataFrame:
        """Generate the enriched table combining both sample annotations."""
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
        
        # Determine Test Condition (the sample name) and its control/test role
        (
            combined['Test Condition'],
            combined[self.COLUMN_CONDITION_ROLE_OUTPUT],
        ) = self._determine_test_conditions(combined)
        
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
            
            # Store final replicate values. The Construct excludes any leading
            # control/test chip prefix (c1_, t1_, ...) since that marks the
            # role/logical chip rather than the biological construct.
            bio_reps.append(bio_rep)
            tech_reps.append(tech_rep)
            constructs.append(self._strip_ct_prefix(first_part))
            
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
        final_columns.extend(
            ['Supergroup', 'Test Condition', self.COLUMN_CONDITION_ROLE_OUTPUT]
        )
        
        # Add replicate columns if they exist
        if 'Biological Replicate' in combined.columns:
            final_columns.append('Biological Replicate')
        if 'Technical Replicate' in combined.columns:
            final_columns.append('Technical Replicate')
        
        self.enriched_table = combined[final_columns]
        
        return self.enriched_table
    
    def _display_sample_annotation(self, df: pd.DataFrame, title: str):
        """Display sample annotation columns (similar logic as in kx_data_importer.py)."""
        # Columns to always display if they exist
        always_display = [
            'PamChip Location', 'Barcode', 'Row', 'Array', 
            'Article number', 'Strip', 'Sample name'
        ]
        
        # Columns to display only if they exist AND have non-zero values
        conditional_display = ['Technical replicate', 'Biological replicate']
        
        # Check which "always display" columns are present
        available_columns = [col for col in always_display if col in df.columns]
        
        def has_nonzero_integers(col_name):
            """True if the column holds at least one non-zero integer-like value."""
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
        """Store the query dictionary, output path and contact e-mail for the BLAST and UniProt requests."""
        self.input_dict = input_dict
        self.output_path = output_path
        self.email = email
        self.base_url = UNIPROT_BLAST_API_DATA_COLLECTOR_DEFAULTS["base_url"]

    def submit_blast(self, sequence):
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
        """
        r = requests.get(f'{self.base_url}/result/{job_id}/tsv', timeout=30)
        if r.status_code == 200:
            return r.text
        print(f"  Error fetching results: {r.status_code}")
        return None

    def get_processed_sequences(self):
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
    DEFAULT_UNIPROT_MODRES_PATH = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
        "default_uniprot_modres_path"
    ]
    DEFAULT_UNIPROT_GENE_SYMBOLS_PATH = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
        "default_uniprot_gene_symbols_path"
    ]
    UNIPROT_STREAM_URL = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["uniprot_stream_url"]
    UNIPROT_MODRES_QUERY = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["uniprot_modres_query"]
    UNIPROT_GENE_SYMBOLS_QUERY = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
        "uniprot_gene_symbols_query"
    ]
    UNIPROT_KINASE_GROUP_NAMES = frozenset(
        str(name).upper()
        for name in OMNIPATH_PTM_EXTRACTOR_DEFAULTS["uniprot_kinase_group_names"]
    )
    UNIPROT_MODRES_SOURCE = "UniProt_ModRes"
    # Retired source label: UniProt 'Interacts with' binding partners that were
    # once added as site-unspecific phosphorylation edges. Removed by
    # refresh_uniprot_modres().
    RETIRED_UNIPROT_SOURCES = ("UniProt_InteractsWith",)
    UNIPROT_MODRES_RESIDUES = {
        "Phosphoserine": "S",
        "Phosphothreonine": "T",
        "Phosphotyrosine": "Y",
    }
    # Canonical-sequence features only. Isoform-specific features are written
    # "MOD_RES P12345-2:17"; their position belongs to another sequence and they
    # do not match.
    UNIPROT_MODRES_PATTERN = re.compile(
        r'MOD_RES\s+(\d+)(?:\.\.\d+)?;\s*/note="([^"]*)"(?:;\s*/evidence="([^"]*)")?'
    )
    GREEK_LETTER_SUFFIXES = {
        "alpha": "A",
        "beta": "B",
        "gamma": "G",
        "delta": "D",
        "epsilon": "E",
        "zeta": "Z",
        "eta": "H",
        "theta": "Q",
        "iota": "I",
    }
    MANUAL_INTERACTIONS_FILENAME = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
        "manual_interactions_filename"
    ]
    UNIPROT_ACCESSIONS_URL = OMNIPATH_PTM_EXTRACTOR_DEFAULTS[
        "uniprot_accessions_url"
    ]
    ENZYME_ANNOTATION_BATCH_SIZE = int(
        OMNIPATH_PTM_EXTRACTOR_DEFAULTS["enzyme_annotation_batch_size"]
    )
    # Extra per-enzyme annotation columns added to the output, placed right
    # after ``ptm_enzyme``. Purely informational (no rows are filtered).
    ENZYME_ANNOTATION_COLUMNS = ["ptm_enzyme_family", "ptm_enzyme_class"]

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
        include_uniprot_modres: bool = True,
        uniprot_modres_input: Path | None = None,
        uniprot_gene_symbols_input: Path | None = None,
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
        overwrite: bool = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["overwrite"],
        annotate_enzymes: bool = True,
        filter_enzyme_class: bool = OMNIPATH_PTM_EXTRACTOR_DEFAULTS["filter_enzyme_class"],
    ):
        """Store the OmniPath, UniProt and iPTMnet query settings and the output paths."""
        self.output_dir = Path(output_dir)
        self.organism = int(organism)
        self.databases = tuple(databases or ())
        self.license_filter = license_filter
        self.timeout = int(timeout)
        self.raw_input = Path(raw_input) if raw_input else None
        self.save_raw = Path(save_raw) if save_raw else None
        self.include_unknown_sites = bool(include_unknown_sites)
        self.filter_human_kinases = bool(filter_human_kinases)
        self.include_uniprot_modres = bool(include_uniprot_modres)
        self.uniprot_modres_input = Path(
            uniprot_modres_input or self.DEFAULT_UNIPROT_MODRES_PATH
        )
        self.uniprot_gene_symbols_input = Path(
            uniprot_gene_symbols_input or self.DEFAULT_UNIPROT_GENE_SYMBOLS_PATH
        )
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
        self.annotate_enzymes = bool(annotate_enzymes)
        self.filter_enzyme_class = bool(filter_enzyme_class)

        if self.filter_enzyme_class and not self.annotate_enzymes:
            raise ValueError(
                "filter_enzyme_class requires annotate_enzymes=True "
                "(the ptm_enzyme_class column is needed for filtering)."
            )
        if self.iptmnet_batch_size <= 0:
            raise ValueError("iptmnet_batch_size must be > 0.")

    @staticmethod
    def _clean_uniprot_id(uid: str) -> str:
        return str(uid).split("-")[0].strip()

    @staticmethod
    def _looks_like_uniprot_accession(uid: str) -> bool:
        uid = str(uid).strip().upper()
        return bool(
            re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", uid)
            or re.fullmatch(r"[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]", uid)
        )

    @staticmethod
    def _normalize_source_label(source: str) -> str:
        """Map OmniPath resource variants onto labels already used by the UKA filters."""
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
        labels = []
        for source in cls._split_tokens(raw_sources):
            labels.append(cls._normalize_source_label(source))
        return ";".join(sorted(set(labels))) if labels else "OmniPath"

    @staticmethod
    def _split_tokens(raw_value: str) -> tuple[str, ...]:
        tokens = []
        for token in re.split(r"[;,]", str(raw_value)):
            token = token.strip()
            if not token or token.lower() in {"nan", "none"}:
                continue
            tokens.append(token)
        return tuple(tokens)

    @classmethod
    def _normalize_raw_sources(cls, raw_sources: str) -> str:
        sources = sorted(set(cls._split_tokens(raw_sources)))
        return ";".join(sources) if sources else ""

    @classmethod
    def _join_source_labels(cls, *raw_values: str) -> str:
        labels = []
        for raw_value in raw_values:
            labels.extend(cls._split_tokens(raw_value))
        return ";".join(sorted(set(labels))) if labels else ""

    @classmethod
    def _normalize_references(cls, raw_references: str) -> tuple[str, ...]:
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
        modification = str(raw_modification).strip().lower()
        if modification == "phosphorylation":
            return "Phosphorylation"
        if not modification or modification in {"nan", "none"}:
            return "unknown"
        return modification.title()

    def _format_site(self, row: pd.Series) -> str | None:
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
        site = str(site).strip().upper()
        if site.startswith("Y"):
            return "ptk"
        if site.startswith(("S", "T")):
            return "stk"
        return None

    @staticmethod
    def _extract_kinase_family(similarities: Iterable[str]) -> str:
        """Best kinase-family label across all UniProt SIMILARITY comments.

        Prefers the most specific level (subfamily), falling back to the family
        group. Only kinase-related similarities are considered. Multidomain
        proteins (e.g. myosins) carry several SIMILARITY comments; the kinase
        one wins. Returns "" when no kinase family is annotated.

        Args:
            similarities (Iterable[str]): UniProt SIMILARITY comment texts.

        Returns:
            str: Kinase-family label, or "" if none.
        """
        prefix = "Belongs to the "
        rank = {"subfamily": 2, "family": 1, "": 0}
        best = ("", "")
        for similarity in similarities:
            if not similarity or "kinase" not in similarity.lower():
                continue
            text = similarity[len(prefix):] if similarity.startswith(prefix) else similarity
            segments = [
                segment.strip()
                for part in text.split(";")
                for segment in part.split(".")
                if segment.strip()
            ]
            subfamilies = [s for s in segments if s.lower().endswith(" subfamily")]
            if subfamilies:
                candidate = (
                    re.sub(r"\s+subfamily$", "", subfamilies[-1], flags=re.I),
                    "subfamily",
                )
            else:
                families = [
                    s
                    for s in segments
                    if s.lower().endswith(" family") and "superfamily" not in s.lower()
                ]
                candidate = (
                    (re.sub(r"\s+family$", "", families[-1], flags=re.I), "family")
                    if families
                    else ("", "")
                )
            if rank[candidate[1]] > rank[best[1]]:
                best = candidate
        return best[0]

    @staticmethod
    def _classify_kinase(
        ec_numbers: Iterable[str],
        keywords: Iterable[str],
        similarities: Iterable[str],
    ) -> str:
        """Classify a kinase by catalytic substrate specificity.

        Uses EC number first (most precise: 2.7.10 = Tyr, 2.7.11 = Ser/Thr,
        2.7.12 = dual), then UniProt keywords, then the SIMILARITY family text
        as a last resort.

        Args:
            ec_numbers (Iterable[str]): EC numbers from the UniProt entry.
            keywords (Iterable[str]): UniProt keyword names.
            similarities (Iterable[str]): SIMILARITY comment texts.

        Returns:
            str: One of "tyr", "ser_thr", "dual", "unknown".
        """
        ecs = [str(ec) for ec in ec_numbers]
        has_tyr_ec = any(ec.startswith("2.7.10") for ec in ecs)
        has_st_ec = any(ec.startswith("2.7.11") for ec in ecs)
        has_dual_ec = any(ec.startswith("2.7.12") for ec in ecs)
        if has_dual_ec or (has_tyr_ec and has_st_ec):
            return "dual"
        if has_tyr_ec:
            return "tyr"
        if has_st_ec:
            return "ser_thr"
        # An enzyme carrying EC number(s) but none in the protein-kinase range
        # (2.7.10/11/12) is a small-molecule kinase, e.g. choline kinase
        # (EC 2.7.1.32). A "Tyrosine-protein kinase" keyword on such an entry is
        # a legacy/spurious annotation, so we do not infer a protein-residue
        # specificity from keywords here.
        if ecs:
            return "unknown"

        kw = {str(k).strip().lower() for k in keywords}
        kw_tyr = "tyrosine-protein kinase" in kw
        kw_st = "serine/threonine-protein kinase" in kw
        if kw_tyr and kw_st:
            return "dual"
        if kw_tyr:
            return "tyr"
        if kw_st:
            return "ser_thr"

        joined = " ".join(similarities).lower()
        if "tyr protein kinase" in joined:
            return "tyr"
        if "ser/thr protein kinase" in joined or "serine/threonine" in joined:
            return "ser_thr"
        return "unknown"

    @classmethod
    def _parse_enzyme_entry(cls, entry: dict) -> tuple[str, str]:
        """Extract (family, class) from one UniProt JSON entry.

        Args:
            entry (dict): A single UniProtKB JSON result object.

        Returns:
            tuple[str, str]: (kinase family label, kinase class).
        """
        similarities = [
            comment["texts"][0]["value"]
            for comment in entry.get("comments", [])
            if comment.get("commentType") == "SIMILARITY" and comment.get("texts")
        ]
        ec_numbers: list[str] = []
        description = entry.get("proteinDescription", {})
        for block in ("recommendedName", "submissionNames", "alternativeNames"):
            value = description.get(block)
            if isinstance(value, dict):
                value = [value]
            for named in value or []:
                ec_numbers += [ec.get("value", "") for ec in named.get("ecNumbers", [])]
        keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]
        return (
            cls._extract_kinase_family(similarities),
            cls._classify_kinase(ec_numbers, keywords, similarities),
        )

    def _fetch_enzyme_annotations(
        self, accessions: Iterable[str]
    ) -> dict[str, tuple[str, str]]:
        """Look up (family, class) per kinase accession from UniProt.

        Batched against the UniProt ``accessions`` endpoint, with an individual
        fallback for accessions the batch omits. Network failures fall back
        to ("", "unknown") so the pipeline still completes.

        Args:
            accessions (Iterable[str]): UniProt accessions to annotate.

        Returns:
            dict[str, tuple[str, str]]: accession -> (family, class).
        """
        fields = "accession,protein_families,cc_similarity,ec,keyword"
        unique = sorted(
            {acc for acc in accessions if self._looks_like_uniprot_accession(acc)}
        )
        lookup: dict[str, tuple[str, str]] = {}
        if not unique:
            return lookup

        print(
            f"Annotating {len(unique)} unique kinases with UniProt "
            "family/class information..."
        )
        batch = max(1, self.ENZYME_ANNOTATION_BATCH_SIZE)
        for start in range(0, len(unique), batch):
            chunk = unique[start:start + batch]
            try:
                response = requests.get(
                    self.UNIPROT_ACCESSIONS_URL,
                    params={"accessions": ",".join(chunk), "fields": fields, "format": "json"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                results = response.json().get("results", [])
            except Exception as exc:  # noqa: BLE001
                print(f"  UniProt annotation batch failed ({exc}); leaving blank.")
                results = []

            for entry in results:
                family_class = self._parse_enzyme_entry(entry)
                primary = entry.get("primaryAccession", "")
                if primary:
                    lookup[primary] = family_class
                for secondary in entry.get("secondaryAccessions", []) or []:
                    lookup.setdefault(secondary, family_class)

            for acc in chunk:
                if acc in lookup:
                    continue
                try:
                    response = requests.get(
                        f"https://rest.uniprot.org/uniprotkb/{acc}.json",
                        params={"fields": fields},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    lookup[acc] = self._parse_enzyme_entry(response.json())
                except Exception:  # noqa: BLE001
                    lookup[acc] = ("", "unknown")
        return lookup

    def _annotate_enzymes(
        self,
        ptk: pd.DataFrame,
        stk: pd.DataFrame,
        known: dict[str, tuple[str, str]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Insert ptm_enzyme_family/ptm_enzyme_class columns (no rows removed).

        Args:
            ptk (pd.DataFrame): Final PTK interaction table.
            stk (pd.DataFrame): Final STK interaction table.
            known (dict | None): accession -> (family, class) to reuse; only the
                remaining enzymes are looked up in UniProt.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Annotated (ptk, stk).
        """
        if not self.annotate_enzymes:
            return ptk, stk

        enzymes: set[str] = set()
        for frame in (ptk, stk):
            if not frame.empty:
                enzymes.update(frame["ptm_enzyme"].dropna().astype(str))
        lookup = dict(known or {})
        lookup.update(self._fetch_enzyme_annotations(enzymes - set(lookup)))

        def _apply(frame: pd.DataFrame, label: str) -> pd.DataFrame:
            frame = frame.copy()
            annotations = frame["ptm_enzyme"].map(
                lambda acc: lookup.get(str(acc), ("", "unknown"))
            )
            family = annotations.map(lambda pair: pair[0])
            kinase_class = annotations.map(lambda pair: pair[1])
            insert_at = frame.columns.get_loc("ptm_enzyme") + 1
            frame.insert(insert_at, "ptm_enzyme_class", kinase_class.to_numpy())
            frame.insert(insert_at, "ptm_enzyme_family", family.to_numpy())
            if not frame.empty:
                filled = int((frame["ptm_enzyme_family"] != "").sum())
                print(
                    f"{label} enzyme annotation: family filled for "
                    f"{filled}/{len(frame)} rows"
                )
            return frame

        return _apply(ptk, "PTK"), _apply(stk, "STK")

    def _filter_enzyme_class(
        self, ptk: pd.DataFrame, stk: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drop enzymes whose catalytic class is opposite to the array type.

        Only active when ``filter_enzyme_class`` is set. PTK drops pure
        ``ser_thr`` enzymes; STK drops pure ``tyr`` enzymes. Dual-specificity
        (``dual``) and unclassified (``unknown``) enzymes are always kept, since
        dual kinases legitimately act on both residue types and ``unknown``
        should not be discarded on weak evidence.

        Args:
            ptk (pd.DataFrame): Annotated PTK interaction table.
            stk (pd.DataFrame): Annotated STK interaction table.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Class-filtered (ptk, stk).
        """
        if not self.filter_enzyme_class:
            return ptk, stk

        def _drop(frame: pd.DataFrame, opposite_class: str, label: str) -> pd.DataFrame:
            if frame.empty or "ptm_enzyme_class" not in frame.columns:
                return frame
            before = len(frame)
            kept = frame[frame["ptm_enzyme_class"] != opposite_class].reset_index(drop=True)
            print(
                f"{label} enzyme-class filter: dropped {before - len(kept)} "
                f"'{opposite_class}' rows -> {len(kept)}"
            )
            return kept

        return _drop(ptk, "ser_thr", "PTK"), _drop(stk, "tyr", "STK")

    def _get_with_retry(self, url, params=None, headers=None, attempts=4):
        """HTTP GET with backoff retries for transient server errors.

        OmniPath / UniProt occasionally return 502/503/504 under load. This
        retries on those status codes and on connection errors (with linear
        backoff), then raises. Non-retryable responses (e.g. 404) raise at once.

        Args:
            url (str): Request URL.
            params (dict | None): Query parameters.
            headers (dict | None): Request headers.
            attempts (int): Maximum number of attempts.

        Returns:
            requests.Response: The successful response.
        """
        retry_statuses = (429, 500, 502, 503, 504)
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
            except requests.exceptions.RequestException as exc:
                if attempt >= attempts:
                    raise
                wait = 5 * attempt
                print(
                    f"  Request error ({exc}); retrying in {wait}s "
                    f"({attempt}/{attempts - 1})..."
                )
                time.sleep(wait)
                continue
            if response.status_code in retry_statuses and attempt < attempts:
                wait = 5 * attempt
                print(
                    f"  Server returned {response.status_code}; retrying in {wait}s "
                    f"({attempt}/{attempts - 1})..."
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response

    def _fetch_human_kinases(self) -> set[str]:
        print("Fetching reviewed human protein kinase list from UniProt...")
        response = self._get_with_retry(self.UNIPROT_KINASE_URL)
        df = pd.read_csv(io.StringIO(response.text), sep="\t", dtype=str)
        kinase_ids = {
            self._clean_uniprot_id(uid)
            for uid in df.get("Entry", pd.Series(dtype=str)).dropna()
        }
        print(f"Found {len(kinase_ids)} reviewed human protein kinases.")
        return kinase_ids

    def _fetch_omnipath(self) -> pd.DataFrame:
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
        response = self._get_with_retry(
            self.OMNIPATH_ENZSUB_URL,
            params=params,
            headers={"User-Agent": "pyKinaXe OmniPath PTM extractor"},
        )

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
        site = str(row.get("site", "")).strip().upper()
        if re.fullmatch(r"[YST]\d+", site):
            return site

        residue = str(row.get("site_residue", "")).strip().upper()
        position = str(row.get("site_position", "")).strip()
        if residue in {"Y", "S", "T"} and re.fullmatch(r"\d+", position):
            return f"{residue}{position}"
        return None

    def _iptmnet_rest_to_pipeline_format(self, raw_df: pd.DataFrame) -> pd.DataFrame:
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

    @staticmethod
    def _read_uniprot_tsv(path: Path, required: set[str]) -> pd.DataFrame:
        """Read a saved UniProt TSV and check that it has the required columns."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing UniProt file: {path}. Run `python src/kx_data_enricher.py "
                "uniprot-modres --download` to fetch the current release."
            )
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"UniProt file {path} is missing columns: {missing}")
        return df

    @classmethod
    def _normalize_kinase_name(cls, name: str) -> str:
        """Return a kinase name from a modified-residue note in lookup form.

        Upper case without spaces; a trailing Greek letter becomes the letter
        suffix of the gene symbol (GSK3-beta -> GSK3B, PKC-theta -> PKCQ).
        """
        token = str(name).strip().strip(".").replace(" ", "")
        match = re.fullmatch(
            r"(.+?)-?(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota)",
            token,
            flags=re.IGNORECASE,
        )
        if match:
            token = match.group(1) + cls.GREEK_LETTER_SUFFIXES[match.group(2).lower()]
        return token.upper()

    def _uniprot_kinase_name_index(self) -> dict[str, object]:
        """Build the lookup that resolves kinase names to kinase accessions.

        Returns:
            dict: ``symbol`` official gene symbol -> kinase accession,
                ``names`` any gene name (symbol or synonym) -> kinase accessions,
                ``official`` official symbol of any reviewed human protein ->
                accessions, ``kinases`` the kinase accessions.
        """
        symbol_to_kinase: dict[str, str] = {}
        name_to_kinases: dict[str, set[str]] = {}
        for path in (self.uniprot_ptk_input, self.uniprot_stk_input):
            table = self._read_uniprot_tsv(path, {"Entry", "Gene Names"})
            for entry, gene_names in zip(table["Entry"], table["Gene Names"]):
                accession = self._clean_uniprot_id(entry)
                names = gene_names.split()
                if not names:
                    continue
                symbol_to_kinase.setdefault(names[0].upper(), accession)
                for name in names:
                    name_to_kinases.setdefault(name.upper(), set()).add(accession)

        symbols = self._read_uniprot_tsv(
            self.uniprot_gene_symbols_input, {"Entry", "Gene Names (primary)"}
        )
        official: dict[str, set[str]] = {}
        for entry, primary in zip(symbols["Entry"], symbols["Gene Names (primary)"]):
            for symbol in self._split_tokens(primary):
                official.setdefault(symbol.upper(), set()).add(entry)

        return {
            "symbol": symbol_to_kinase,
            "names": name_to_kinases,
            "official": official,
            "kinases": set(symbol_to_kinase.values()),
        }

    def _resolve_uniprot_kinase(
        self, name: str, substrate: str, index: dict[str, object]
    ) -> list[str]:
        """Resolve one kinase name of a modified-residue note to accession(s).

        A name is used only when it denotes exactly one human protein kinase:

        - an official gene symbol wins over synonyms (UniProt also lists PAK1 as
          a synonym of PKN1 and PASK as one of STK39);
        - family and group names (``uniprot_kinase_group_names``) are never
          assigned to one member, and they are checked before synonyms because
          UniProt lists e.g. AMPK as a synonym of PRKAA2;
        - a synonym counts when it belongs to one kinase only and is not the
          official symbol of another human protein (PDK1 is pyruvate
          dehydrogenase kinase 1 and also a PDPK1 synonym);
        - "A/B" spellings resolve through their unique part (PKB/AKT1 -> AKT1);
        - "autocatalysis" is the modified protein itself.

        Returns:
            list[str]: The kinase accession, or an empty list.
        """
        if name.strip().lower() == "autocatalysis":
            return [substrate] if substrate in index["kinases"] else []

        parts = [name] + (name.split("/") if "/" in name else [])
        tokens = [self._normalize_kinase_name(part) for part in parts]
        for token in tokens:
            if token in index["symbol"]:
                return [index["symbol"][token]]
        if any(token in self.UNIPROT_KINASE_GROUP_NAMES for token in tokens):
            return []
        for token in tokens:
            kinases = index["names"].get(token, set())
            if len(kinases) == 1 and not index["official"].get(token, set()) - kinases:
                return sorted(kinases)
        return []

    def _process_uniprot_modres(
        self,
        kinase_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """Convert UniProt modified-residue annotations into kinase-site rows.

        Uses the phosphoserine, phosphothreonine and phosphotyrosine features of
        reviewed human entries whose note names the kinase
        ("Phosphothreonine; by IKKE, PDPK1 and TBK1"), at canonical-sequence
        positions. Every named kinase gives one substrate-kinase-site row; names
        that do not denote exactly one protein kinase (PKA, CK2, ...) are
        skipped, see ``_resolve_uniprot_kinase``.

        Returns:
            pd.DataFrame: Rows in ``COLUMNS`` format.
        """
        table = self._read_uniprot_tsv(
            self.uniprot_modres_input, {"Entry", "Modified residue"}
        )
        index = self._uniprot_kinase_name_index()

        rows = []
        unresolved = 0
        for entry, features in zip(table["Entry"], table["Modified residue"]):
            substrate = self._clean_uniprot_id(entry)
            for position, note, evidence in self.UNIPROT_MODRES_PATTERN.findall(features):
                parts = [part.strip() for part in note.split(";")]
                residue = self.UNIPROT_MODRES_RESIDUES.get(parts[0])
                named = [part[3:] for part in parts if part.startswith("by ")]
                if residue is None or not named:
                    continue
                references = sorted(set(re.findall(r"PubMed:(\d+)", evidence)))
                for name in re.split(r",\s*|\s+and\s+|\s+or\s+", named[0]):
                    if not name.strip():
                        continue
                    kinases = self._resolve_uniprot_kinase(name, substrate, index)
                    if not kinases:
                        unresolved += 1
                    for kinase in kinases:
                        if kinase_ids is not None and kinase not in kinase_ids:
                            continue
                        rows.append(
                            {
                                "uniprot_id": substrate,
                                "ptm_enzyme": kinase,
                                "site": f"{residue}{position}",
                                "ptm_type": "Phosphorylation",
                                "score": 1.0,
                                "source": self.UNIPROT_MODRES_SOURCE,
                                "evidence_level": (
                                    "curated_literature" if references else "curated_source"
                                ),
                                "has_curated_source": True,
                                "n_references": len(references),
                                "references": ";".join(references),
                                "raw_sources": self.UNIPROT_MODRES_SOURCE,
                            }
                        )

        out = pd.DataFrame(rows, columns=self.COLUMNS)
        print(
            f"Formatted UniProt modified-residue kinase-site rows: {len(out)} "
            f"({unresolved} kinase names not attributable to one kinase skipped)"
        )
        return out

    def _load_uniprot_modres(
        self,
        kinase_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        """Load the UniProt modified-residue kinase-site rows when enabled."""
        if not self.include_uniprot_modres:
            return pd.DataFrame(columns=self.COLUMNS)
        return self._process_uniprot_modres(kinase_ids=kinase_ids)

    def download_uniprot_modres(self, target_dir: Path) -> tuple[Path, Path]:
        """Save the current UniProt release of the two modified-residue inputs.

        Writes ``uniprot_modres_human_<release>.tsv`` (reviewed human entries with
        phospho features: accession, primary gene name, modified residues) and
        ``uniprot_gene_symbols_human_<release>.tsv`` (primary gene names of all
        reviewed human entries) and points this extractor at both files.

        Returns:
            tuple[Path, Path]: The modified-residue and the gene-symbol file.
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for query, fields, stem in (
            (self.UNIPROT_MODRES_QUERY, "accession,gene_primary,ft_mod_res", "uniprot_modres_human"),
            (self.UNIPROT_GENE_SYMBOLS_QUERY, "accession,gene_primary", "uniprot_gene_symbols_human"),
        ):
            response = self._get_with_retry(
                self.UNIPROT_STREAM_URL,
                params={"query": query, "fields": fields, "format": "tsv"},
            )
            release = response.headers.get("X-UniProt-Release", "unknown")
            path = target_dir / f"{stem}_{release}.tsv"
            path.write_text(response.text)
            print(f"Saved UniProt release {release}: {path}")
            paths.append(path)
        self.uniprot_modres_input, self.uniprot_gene_symbols_input = paths
        return paths[0], paths[1]

    def refresh_uniprot_modres(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Replace the UniProt-derived rows of the existing PTK/STK tables.

        Re-derives the UniProt part of ``ptk_interactions.csv`` and
        ``stk_interactions.csv`` in ``output_dir`` without fetching OmniPath or
        iPTMnet again. Every contribution of a UniProt source (the
        modified-residue rows and the retired 'Interacts with' binding-partner
        edges) is removed first: rows supported only by it are dropped, mixed
        rows lose its label and the site-unspecific "Y" / "S/T" placeholder and
        get their evidence level re-derived from the remaining sources. The
        current modified-residue rows are then merged in. Enzyme family/class
        annotations already in the tables are kept; only new enzymes are looked
        up in UniProt.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The rewritten (ptk, stk) tables.
        """
        uniprot_sources = {self.UNIPROT_MODRES_SOURCE, *self.RETIRED_UNIPROT_SOURCES}
        generic_sites = {"Y", "S/T"}
        tables = []
        known_annotations: dict[str, tuple[str, str]] = {}
        for name in ("ptk_interactions.csv", "stk_interactions.csv"):
            path = self.output_dir / name
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            if set(self.ENZYME_ANNOTATION_COLUMNS) <= set(table.columns):
                known_annotations.update(
                    zip(
                        table["ptm_enzyme"],
                        zip(table["ptm_enzyme_family"], table["ptm_enzyme_class"]),
                    )
                )
            sources = table["source"].map(lambda value: set(self._split_tokens(value)))
            kept = sources.map(lambda labels: bool(labels - uniprot_sources))
            touched = kept & sources.map(lambda labels: bool(labels & uniprot_sources))
            before = len(table)
            table = table[kept].copy()
            touched = touched[kept]
            for column in ("source", "raw_sources"):
                table.loc[touched, column] = table.loc[touched, column].map(
                    lambda value: ";".join(
                        token for token in self._split_tokens(value) if token not in uniprot_sources
                    )
                )
            table.loc[touched, "site"] = table.loc[touched, "site"].map(
                lambda value: ";".join(
                    token for token in self._split_tokens(value) if token.upper() not in generic_sites
                ) or "unknown"
            )
            table.loc[touched, "evidence_level"] = [
                self._classify_evidence(raw_sources, self._split_tokens(references))
                for raw_sources, references in zip(
                    table.loc[touched, "raw_sources"], table.loc[touched, "references"]
                )
            ]
            table["has_curated_source"] = table["has_curated_source"].astype(str).eq("True")
            table.loc[touched, "has_curated_source"] = table.loc[
                touched, "evidence_level"
            ].isin({"curated_literature", "curated_source"})
            print(
                f"{name}: removed {before - len(table)} UniProt-only rows, "
                f"stripped UniProt from {int(touched.sum())} mixed rows"
            )
            tables.append(table[self.COLUMNS])

        kinase_ids = self._fetch_human_kinases() if self.filter_human_kinases else None
        modres = self._load_uniprot_modres(kinase_ids=kinase_ids)
        array_type = modres["site"].apply(self._classify_site)
        ptk = pd.concat([tables[0], modres[array_type == "ptk"]], ignore_index=True)
        stk = pd.concat([tables[1], modres[array_type == "stk"]], ignore_index=True)
        ptk = self._merge_duplicates(ptk)
        stk = self._merge_duplicates(stk)
        ptk, stk = self._annotate_enzymes(ptk, stk, known=known_annotations)
        ptk, stk = self._filter_enzyme_class(ptk, stk)
        self._write_outputs(ptk, stk)
        return ptk, stk

    def _ensure_manual_interactions_file(self) -> None:
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
        
        Returns:
            pd.DataFrame: Merged duplicates.
        """
        if df.empty:
            return pd.DataFrame(columns=OmniPathPTMExtractor.COLUMNS)

        def aggregate(group: pd.DataFrame) -> dict[str, object]:
            sites = sorted(
                {
                    site.strip()
                    for raw_sites in group["site"].dropna().astype(str)
                    for site in raw_sites.split(";")
                    if site.strip()
                }
            )
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
        """Handle run."""
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

        uniprot_modres = self._load_uniprot_modres(kinase_ids=kinase_ids)
        if not uniprot_modres.empty:
            before = len(interactions)
            interactions = pd.concat(
                [interactions, uniprot_modres],
                ignore_index=True,
            )
            print(
                "Added UniProt modified-residue rows to PTM rows: "
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

        ptk, stk = self._annotate_enzymes(ptk, stk)
        ptk, stk = self._filter_enzyme_class(ptk, stk)

        self._write_outputs(ptk, stk)
        return ptk, stk


def _split_csv_values(raw_values: str | None) -> tuple[str, ...]:
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
        """

        df_kinases_in_liver = self._get_human_kinases()

        if flag_save:
            self._save_data(df_kinases_in_liver)

        return df_kinases_in_liver




def _build_blast_arg_parser(subparsers):
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
    df_raw = pd.read_csv(args.input)
    sequence_dict = dict(zip(df_raw['ID'], df_raw['Sequence']))
    collector = UniProt_BLAST_API_data_collector(
        input_dict=sequence_dict,
        output_path=str(args.output),
        email=args.email,
    )
    collector.run_blast_peptides(skip_existing=not args.no_skip_existing)


def _add_uniprot_modres_input_arguments(parser) -> None:
    """Add the saved UniProt inputs of the modified-residue kinase-site rows."""
    parser.add_argument(
        "--uniprot-modres-input",
        type=Path,
        default=OmniPathPTMExtractor.DEFAULT_UNIPROT_MODRES_PATH,
        help="Saved UniProt TSV with the modified-residue annotations (field ft_mod_res).",
    )
    parser.add_argument(
        "--uniprot-gene-symbols-input",
        type=Path,
        default=OmniPathPTMExtractor.DEFAULT_UNIPROT_GENE_SYMBOLS_PATH,
        help="Saved UniProt TSV with the primary gene names of all reviewed human entries.",
    )
    parser.add_argument(
        "--uniprot-stk-input",
        type=Path,
        default=OmniPathPTMExtractor.DEFAULT_UNIPROT_STK_PATH,
        help="Saved UniProt TSV of the human Ser/Thr kinases with all gene names.",
    )
    parser.add_argument(
        "--uniprot-ptk-input",
        type=Path,
        default=OmniPathPTMExtractor.DEFAULT_UNIPROT_PTK_PATH,
        help="Saved UniProt TSV of the human Tyr kinases with all gene names.",
    )


def _build_uniprot_modres_arg_parser(subparsers):
    """Build the arg parser that refreshes the UniProt rows of the PTM tables."""
    parser = subparsers.add_parser(
        "uniprot-modres",
        help=(
            "Replace the UniProt rows of the existing ptk/stk_interactions.csv by "
            "the kinase-site rows of the saved UniProt modified-residue annotations "
            "(no OmniPath or iPTMnet request)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OMNIPATH_PTM_EXTRACTOR_DEFAULTS["output_dir"],
        help="Directory with the ptk_interactions.csv and stk_interactions.csv to update.",
    )
    parser.add_argument(
        "--download",
        type=Path,
        nargs="?",
        const=OmniPathPTMExtractor.DEFAULT_UNIPROT_MODRES_PATH.parent,
        default=None,
        help=(
            "Fetch the current UniProt release of both inputs into this directory "
            "first (default: the directory of the saved inputs) and use it."
        ),
    )
    parser.add_argument(
        "--no-kinase-filter",
        action="store_true",
        help="Do not filter enzymes against the UniProt human kinase keyword list.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=OMNIPATH_PTM_EXTRACTOR_DEFAULTS["timeout"],
        help="Request timeout in seconds.",
    )
    _add_uniprot_modres_input_arguments(parser)
    return parser


def _run_uniprot_modres_cli(args) -> None:
    """Run the UniProt modified-residue refresh CLI."""
    extractor = OmniPathPTMExtractor(
        output_dir=args.output,
        timeout=args.timeout,
        filter_human_kinases=not args.no_kinase_filter,
        uniprot_modres_input=args.uniprot_modres_input,
        uniprot_gene_symbols_input=args.uniprot_gene_symbols_input,
        uniprot_stk_input=args.uniprot_stk_input,
        uniprot_ptk_input=args.uniprot_ptk_input,
        overwrite=True,
    )
    if args.download is not None:
        extractor.download_uniprot_modres(args.download)
    extractor.refresh_uniprot_modres()


def _build_omnipath_arg_parser(subparsers):
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
        "--include-uniprot-modres",
        dest="include_uniprot_modres",
        action="store_true",
        help=(
            "Append the kinase-site rows of the saved UniProt modified-residue "
            "annotations ('Phosphoserine; by <kinase>')."
        ),
    )
    uniprot_group.add_argument(
        "--no-uniprot-modres",
        dest="include_uniprot_modres",
        action="store_false",
        help="Do not append UniProt modified-residue kinase-site rows.",
    )
    parser.set_defaults(include_uniprot_modres=True)
    _add_uniprot_modres_input_arguments(parser)
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
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Allow overwriting existing ptk_interactions.csv/stk_interactions.csv.",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Refuse to overwrite existing output files.",
    )
    parser.set_defaults(overwrite=OMNIPATH_PTM_EXTRACTOR_DEFAULTS["overwrite"])
    parser.add_argument(
        "--no-enzyme-annotation",
        dest="annotate_enzymes",
        action="store_false",
        help="Do not add the ptm_enzyme_family/ptm_enzyme_class columns from UniProt.",
    )
    parser.set_defaults(annotate_enzymes=True)
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--filter-enzyme-class",
        dest="filter_enzyme_class",
        action="store_true",
        help=(
            "Drop enzymes whose catalytic class is opposite to the array type "
            "(PTK drops ser_thr, STK drops tyr; dual/unknown kept). "
            "Requires enzyme annotation."
        ),
    )
    filter_group.add_argument(
        "--no-filter-enzyme-class",
        dest="filter_enzyme_class",
        action="store_false",
        help="Keep all enzymes regardless of catalytic class.",
    )
    parser.set_defaults(
        filter_enzyme_class=OMNIPATH_PTM_EXTRACTOR_DEFAULTS["filter_enzyme_class"]
    )
    return parser


def _run_omnipath_cli(args) -> None:
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
        include_uniprot_modres=args.include_uniprot_modres,
        uniprot_modres_input=args.uniprot_modres_input,
        uniprot_gene_symbols_input=args.uniprot_gene_symbols_input,
        uniprot_stk_input=args.uniprot_stk_input,
        uniprot_ptk_input=args.uniprot_ptk_input,
        include_iptmnet_rest=args.include_iptmnet_rest,
        iptmnet_api_url=args.iptmnet_api_url,
        iptmnet_batch_size=args.iptmnet_batch_size,
        iptmnet_site_input=args.iptmnet_site_input,
        include_manual_interactions=args.include_manual_interactions,
        manual_interactions_input=args.manual_interactions_input,
        overwrite=args.overwrite,
        annotate_enzymes=args.annotate_enzymes,
        filter_enzyme_class=args.filter_enzyme_class,
    )
    extractor.run()


def _build_liver_kinases_arg_parser(subparsers):
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
    extractor = Kinase_Liver_Extractor(path_output=args.output)
    extractor.create_list(flag_save=not args.no_save)


def build_cli_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data enrichment and supporting data-collection utilities for the KX pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")
    _build_blast_arg_parser(subparsers)
    _build_omnipath_arg_parser(subparsers)
    _build_uniprot_modres_arg_parser(subparsers)
    _build_liver_kinases_arg_parser(subparsers)
    return parser


def main() -> None:
    """Run the module as a command-line entry point."""
    parser = build_cli_arg_parser()
    args = parser.parse_args()

    if args.command == "blast":
        _run_blast_cli(args)
    elif args.command == "omnipath":
        _run_omnipath_cli(args)
    elif args.command == "uniprot-modres":
        _run_uniprot_modres_cli(args)
    elif args.command == "liver-kinases":
        _run_liver_kinases_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
