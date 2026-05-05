"""
data_loader.py
Load and preprocess SFM and IFM data from Excel files.
Normalizes column names and handles nulls so the matcher works cleanly.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ── SFM columns we care about ────────────────────────────────────────────────
SFM_COLS = [
    "nav_name", "nexus_id", "nav_type", "equip_type",
    "nav_path", "meta", "enabled",
]

# ── IFM columns we care about ─────────────────────────────────────────────────
IFM_COLS = [
    "asset_id", "asset_alternate_id", "asset_name", "asset_status",
    "manufacturer", "serial_number", "equip_part_description",
    "asset_type", "model",
    "position_id", "position_alternate_id", "position_name",
    "position_status", "position_uniformat_code", "position_type_description",
    "region_id", "region_name", "region_status",
    "building_id", "building_name", "building_status",
    "floor_id", "floor_name",
    "room_id", "room_name",
    "customer_name", "country_name",
]


def _clean_str(val) -> str:
    """Return stripped lowercase string or empty string for nulls."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all string columns and fill NaN with ''."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(_clean_str)
    df.fillna("", inplace=True)
    return df


def load_sfm_ifm_from_excel(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the combined SFM+IFM Excel (the hackathon format where SFM is on the
    left and IFM is on the right, with two header rows).
    Returns (sfm_df, ifm_df).
    """
    raw = pd.read_excel(filepath, header=None)

    # Row 0 has section labels ('SFM', 'IFM'), row 1 has column headers
    col_headers = raw.iloc[1].tolist()
    data = raw.iloc[2:].copy()
    data.columns = col_headers
    data = data.reset_index(drop=True)

    # SFM block: first 8 columns
    sfm_df = data.iloc[:, :8].copy()
    sfm_df.columns = [str(c) for c in sfm_df.columns]

    # IFM block: columns 9 onwards
    ifm_df = data.iloc[:, 9:].copy()
    ifm_df.columns = [str(c) for c in ifm_df.columns]

    sfm_df = _normalize_df(sfm_df)
    ifm_df = _normalize_df(ifm_df)

    return sfm_df, ifm_df


def load_separate_files(
    sfm_path: str, ifm_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load SFM and IFM from two separate Excel files (used for test/demo uploads).
    Expects one header row in each file.
    """
    sfm_df = pd.read_excel(sfm_path)
    ifm_df = pd.read_excel(ifm_path)

    sfm_df.columns = [str(c).strip().lower().replace(" ", "_") for c in sfm_df.columns]
    ifm_df.columns = [str(c).strip().lower().replace(" ", "_") for c in ifm_df.columns]

    sfm_df = _normalize_df(sfm_df)
    ifm_df = _normalize_df(ifm_df)

    return sfm_df, ifm_df


def records_to_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of row dicts."""
    return df.to_dict(orient="records")
