"""Helpers for YAML-backed configuration modules in ``config/``."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(module_file: str | Path) -> dict:
    """Load the YAML file that shares the same stem as the given module.
    
    Args:
        module_file (str | Path): Python module file whose sibling YAML configuration should be loaded.
    
    Returns:
        dict: Loaded yaml config.
    """
    config_path = Path(module_file).with_suffix(".yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_repo_path(value):
    """Resolve a repo-relative path value to an absolute ``Path``.
    
    Args:
        value: Input value processed by this helper.
    
    Returns:
        object: Resolved repo path.
    """
    if value is None:
        return None
    return REPO_ROOT / str(value)


def expand_user_path(value):
    """Convert a string-like path to ``Path`` and expand ``~`` if present.
    
    Args:
        value: Input value processed by this helper.
    
    Returns:
        object: Expand user path.
    """
    return Path(str(value)).expanduser()
