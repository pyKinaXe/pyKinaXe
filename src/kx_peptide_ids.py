"""Normalisation of peptide ids shared by the peptide stage and the UKA.
"""

import re

import pandas as pd

_SEPARATORS = re.compile(r"[\s/]+")


def normalize_peptide_id(value):
    """Return one peptide id with whitespace and slashes removed.

    The scalar counterpart of :func:`normalize_peptide_ids`, for the per-peptide
    lookups in the family analysis -- building a ``pd.Series`` per call would put
    a frame constructor inside a loop over every peptide of every comparison.

    Args:
        value: A single peptide id.

    Returns:
        str: The normalised id.
    """
    return _SEPARATORS.sub("", str(value))


def normalize_peptide_ids(values):
    """Return the peptide ids with whitespace and slashes removed.

    Args:
        values: Peptide-id column (anything ``pd.Series`` accepts).

    Returns:
        pd.Series: The ids as strings, normalised, index preserved.
    """
    return pd.Series(values).astype(str).str.replace(r"[\s/]+", "", regex=True)


def normalize_peptide_id_column(df, column="ID"):
    """Normalise ``column`` in ``df``, copying only when something changes.

    Args:
        df: Frame carrying peptide ids, or ``None``.
        column: Name of the id column.

    Returns:
        The frame with normalised ids -- the ORIGINAL object when the ids were
        already clean, so a clean run pays no copy of a multi-million-row export.
    """
    if df is None or column not in df.columns:
        return df
    normalized = normalize_peptide_ids(df[column])
    if normalized.equals(df[column].astype(str)):
        return df
    df = df.copy()
    df[column] = normalized
    return df
