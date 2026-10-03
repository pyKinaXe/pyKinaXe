import pandas as pd

from kx_peptide_ids import (
    normalize_peptide_id,
    normalize_peptide_id_column,
    normalize_peptide_ids,
)


def test_normalize_peptide_ids_remove_spaces_and_slashes():
    assert normalize_peptide_id(" JAK1_ 1027/1039 ") == "JAK1_10271039"

    values = pd.Series(["A / B", "clean"], index=["first", "second"])
    normalized = normalize_peptide_ids(values)
    assert normalized.to_dict() == {"first": "AB", "second": "clean"}


def test_normalize_column_copies_only_when_needed():
    clean = pd.DataFrame({"ID": ["AKT1_1_2"]})
    assert normalize_peptide_id_column(clean) is clean

    dirty = pd.DataFrame({"ID": ["JAK1_ 1027/1039"], "value": [1]})
    normalized = normalize_peptide_id_column(dirty)
    assert normalized is not dirty
    assert normalized["ID"].tolist() == ["JAK1_10271039"]
    assert dirty["ID"].tolist() == ["JAK1_ 1027/1039"]
