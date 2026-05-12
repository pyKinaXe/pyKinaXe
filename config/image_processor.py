"""YAML-backed defaults for ``kx_image_processor.py``."""

from __future__ import annotations

from config._loader import load_yaml_config, resolve_repo_path


IMAGE_PROCESSOR_CONFIG = load_yaml_config(__file__)
IMAGE_PROCESSOR_CONFIG["paths"]["default_enrichment_file"] = resolve_repo_path(
    IMAGE_PROCESSOR_CONFIG["paths"]["default_enrichment_file"]
)
