"""YAML-backed defaults for ``kx_data_importer.py``."""

from __future__ import annotations

import re

from config._loader import load_yaml_config


_CONFIG = load_yaml_config(__file__)

DATA_LOADER_DEFAULTS = {
    **_CONFIG["data_loader"],
    "filename_pattern": re.compile(
        _CONFIG["data_loader"]["filename_pattern"],
        re.IGNORECASE,
    ),
    "folder_datetime_pattern": re.compile(
        _CONFIG["data_loader"]["folder_datetime_pattern"]
    ),
    "coordinate_names": tuple(_CONFIG["data_loader"]["coordinate_names"]),
    "valid_peptide_types": tuple(_CONFIG["data_loader"]["valid_peptide_types"]),
    "candidate_data_paths": tuple(_CONFIG["data_loader"]["candidate_data_paths"]),
    "annotation_always_display": tuple(
        _CONFIG["data_loader"]["annotation_always_display"]
    ),
    "annotation_conditional_display": tuple(
        _CONFIG["data_loader"]["annotation_conditional_display"]
    ),
    "annotation_exclude_columns": tuple(
        _CONFIG["data_loader"]["annotation_exclude_columns"]
    ),
}

ARRAY_LAYOUT_LOADER_DEFAULTS = {
    **_CONFIG["array_layout_loader"],
    "candidate_data_paths": tuple(
        _CONFIG["array_layout_loader"]["candidate_data_paths"]
    ),
    "encodings_to_try": tuple(_CONFIG["array_layout_loader"]["encodings_to_try"]),
    "numeric_float_fields": tuple(
        _CONFIG["array_layout_loader"]["numeric_float_fields"]
    ),
    "chip_grid_dimensions": {
        chip_type: tuple(dimensions)
        for chip_type, dimensions in _CONFIG["array_layout_loader"][
            "chip_grid_dimensions"
        ].items()
    },
}
