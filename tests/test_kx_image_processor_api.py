from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_dir in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from kx_image_processor import ImageProcessor


def _build_layout() -> xr.DataArray:
    layout = np.empty((1, 2), dtype=object)
    layout[0, 0] = {"ID": "S1", "Sequence": "SEQ1", "Row": 1, "Col": 1}
    layout[0, 1] = {"ID": "S2", "Sequence": "SEQ2", "Row": 1, "Col": 2}
    return xr.DataArray(
        layout,
        dims=["spot_row", "spot_col"],
        coords={"spot_row": [0], "spot_col": [0, 1]},
    )


def _build_images() -> xr.DataArray:
    return xr.DataArray(
        np.zeros((1, 2, 2), dtype=np.float32),
        dims=["image_idx", "y", "x"],
        coords={
            "image_idx": [0],
            "PamChip_barcode": ("image_idx", ["BC-1"]),
            "Array": ("image_idx", [1]),
            "FoV": ("image_idx", [2]),
            "Exposure_time": ("image_idx", [50]),
            "Pump_cycle": ("image_idx", [3]),
            "Image_number": ("image_idx", [4]),
            "Temperature": ("image_idx", [25.0]),
        },
    )


def _build_processor() -> ImageProcessor:
    processor = object.__new__(ImageProcessor)
    processor._spot_layout = _build_layout()
    processor._layout_loader = None
    processor._enrichment_df = None
    processor._enrichment_columns = None
    processor._enrichment_by_sequence = None
    processor._enrichment_arrays = None
    processor._spot_intensities = xr.DataArray(
        np.array([[[1.5, 2.5]]], dtype=np.float32),
        dims=["image_idx", "spot_row", "spot_col"],
        coords={"image_idx": [0], "spot_row": [0], "spot_col": [0, 1]},
    )
    processor._spot_max_intensities = xr.DataArray(
        np.array([[[5.0, 6.0]]], dtype=np.float32),
        dims=["image_idx", "spot_row", "spot_col"],
        coords={"image_idx": [0], "spot_row": [0], "spot_col": [0, 1]},
    )
    processor._spot_median_intensities = xr.DataArray(
        np.array([[[1.0, 2.0]]], dtype=np.float32),
        dims=["image_idx", "spot_row", "spot_col"],
        coords={"image_idx": [0], "spot_row": [0], "spot_col": [0, 1]},
    )
    processor._spot_saturation_fractions = xr.DataArray(
        np.array([[[0.1, 0.2]]], dtype=np.float32),
        dims=["image_idx", "spot_row", "spot_col"],
        coords={"image_idx": [0], "spot_row": [0], "spot_col": [0, 1]},
    )
    processor._refined_spot_intensities = None
    processor._refined_spot_max_intensities = None
    processor._refined_spot_median_intensities = None
    processor._refined_spot_saturation_fractions = None
    processor._spot_grid_positions = [
        np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)
    ]
    processor._refined_grid_positions = None
    processor._centers = np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    processor._refined_grid_params_list = [
        {
            "dx": 1.0,
            "dy": 2.0,
            "dangle_deg": 0.1,
            "dspacing": 0.2,
            "original_intensity": 100.0,
            "refined_intensity": 110.0,
            "intensity_improvement": 10.0,
            "improvement_percent": 10.0,
            "center_x": 25.0,
            "center_y": 35.0,
            "angle_deg": 1.5,
            "spacing": 8.0,
        },
        {
            "dx": 3.0,
            "dy": 4.0,
            "dangle_deg": 0.3,
            "dspacing": 0.4,
            "original_intensity": 120.0,
            "refined_intensity": 132.0,
            "intensity_improvement": 12.0,
            "improvement_percent": 10.0,
            "center_x": 26.0,
            "center_y": 36.0,
            "angle_deg": 1.7,
            "spacing": 8.2,
        },
    ]
    processor._grid_refinement_params = {"method": "L-BFGS-B"}
    processor._reference_spots = {
        "t_shape": [
            np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
            np.array([[2.0, 1.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32),
        ],
        "j_shape": [
            np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [4.0, 3.0]], dtype=np.float32),
            np.array([[6.0, 1.0], [6.0, 2.0], [6.0, 3.0], [5.0, 3.0]], dtype=np.float32),
        ],
    }
    processor.original_images = _build_images()
    return processor


def test_enrichment_loading_and_accessors(tmp_path: Path) -> None:
    processor = _build_processor()
    enrichment_csv = tmp_path / "enrichment.csv"
    enrichment_csv.write_text(
        "\n".join(
            [
                "Sequence,Kinase,Score",
                "SEQ1,K1,1.0",
                "SEQ1,K2,2.0",
                "SEQ2,K3,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    processor.load_enrichment_data(enrichment_csv, verbose=False)

    kinase_array = processor.get_enrichment_array("Kinase")
    assert kinase_array.shape == (1, 2)
    assert kinase_array[0, 0] == ["K1", "K2"]
    assert kinase_array[0, 1] == "K3"
    assert processor.get_all_spot_ids().tolist() == [["S1", "S2"]]
    assert processor.get_all_spot_sequences().tolist() == [["SEQ1", "SEQ2"]]


def test_dataframe_and_statistics_helpers(tmp_path: Path) -> None:
    processor = _build_processor()
    enrichment_csv = tmp_path / "enrichment.csv"
    enrichment_csv.write_text(
        "\n".join(
            [
                "Sequence,Kinase,Score",
                "SEQ1,K1,1.0",
                "SEQ2,K3,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    processor.load_enrichment_data(enrichment_csv, verbose=False)

    df = processor.get_intensities_dataframe(0, include_enrichment=True)
    assert list(df["ID"]) == ["S1", "S2"]
    assert list(df["intensity"]) == [1.5, 2.5]
    assert list(df["Enr_Kinase"]) == ["K1", "K3"]

    stats = processor.get_statistics()
    assert stats["n_images"] == 2
    assert stats["center_x_mean"] == 2.0
    assert stats["center_y_mean"] == 4.0

    grid_stats = processor.get_grid_refinement_statistics()
    assert grid_stats["n_valid"] == 2
    assert grid_stats["dx_mean"] == 2.0
    assert grid_stats["optimization_params"]["method"] == "L-BFGS-B"

    reference_stats = processor.get_reference_spot_statistics()
    assert reference_stats["n_images"] == 2
    assert reference_stats["initial_t_n_valid"] == 2
    assert reference_stats["initial_j_n_spots"] == 4
