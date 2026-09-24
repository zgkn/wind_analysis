"""Shared config, calendar-window math, and dataset naming helpers used by
both download_winds.py and plot_winds.py.
"""

from datetime import date, timedelta
from pathlib import Path

import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_YEARS = [2015, 2019, 2023, 2026]

LEVELS = [925, 850, 700]  # hPa

# CDS area is [North, West, South, East]
AREA = [10, 95, -10, 120]  # covers Singapore, Peninsular Malaysia, Sumatra, Kalimantan

# Shared comparison window, applied identically to every year (month, day).
WINDOW_START = (8, 1)

# ERA5T daily data lags behind real time; this is a conservative buffer so
# --window-end defaults to a date CDS is actually likely to have.
ERA5T_LAG_DAYS = 6

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# Calendar window helpers
# ---------------------------------------------------------------------------

def default_window_end(today: date | None = None, lag_days: int = ERA5T_LAG_DAYS) -> tuple[int, int]:
    """Latest (month, day) CDS is likely to have daily data for, applied to
    every comparison year regardless of that year's own calendar position."""
    today = today or date.today()
    cutoff = today - timedelta(days=lag_days)
    return (cutoff.month, cutoff.day)


def days_in_window(year: int, start: tuple[int, int], end: tuple[int, int]) -> dict[int, list[int]]:
    """Return {month: [days]} for every calendar day from start to end
    (inclusive) in the given year. start/end must not cross a year boundary."""
    start_date = date(year, *start)
    end_date = date(year, *end)
    if end_date < start_date:
        raise ValueError(f"[{year}] window end {end} is before window start {start}")

    months_days: dict[int, list[int]] = {}
    d = start_date
    while d <= end_date:
        months_days.setdefault(d.month, []).append(d.day)
        d += timedelta(days=1)
    return months_days


def parse_month_day(arg: str) -> tuple[int, int]:
    """Parse 'MM-DD' into (month, day)."""
    month_str, day_str = arg.split("-")
    return (int(month_str), int(day_str))


def window_label(start: tuple[int, int], end: tuple[int, int]) -> str:
    return f"{date(2000, *start).strftime('%b %d')}–{date(2000, *end).strftime('%b %d')}"


def daily_file_stub(data_dir: Path, year: int, month: int) -> Path:
    """Base name (no extension) for one (year, month) request. CDS may
    deliver that request as a single .nc or as several (e.g. one per
    variable), so this isn't necessarily one physical file -- see
    download_winds.py's _download_and_extract."""
    return data_dir / f"era5_daily_winds_{year}_{month:02d}"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def standardize_names(ds: xr.Dataset) -> xr.Dataset:
    """Handle variable/dim naming differences across CDS API versions."""
    rename = {}
    candidates = {
        "u": ["u", "u_component_of_wind"],
        "v": ["v", "v_component_of_wind"],
    }
    for target, options in candidates.items():
        for opt in options:
            if opt in ds.data_vars:
                if opt != target:
                    rename[opt] = target
                break
    if rename:
        ds = ds.rename(rename)

    dim_rename = {}
    if "pressure_level" in ds.dims:
        dim_rename["pressure_level"] = "level"
    if "valid_time" in ds.dims:
        dim_rename["valid_time"] = "time"
    if dim_rename:
        ds = ds.rename(dim_rename)

    return ds


def open_year_dataset(paths: list[Path]) -> xr.Dataset:
    """Open and standardize (but don't average) one year's daily files.

    Uses plain xr.open_dataset() per file + xr.combine_by_coords() instead
    of xr.open_mfdataset(), which routes through dask-backed chunking
    internals even when chunks=None -- these files are small enough
    (single-digit MB per region/month) that eager, non-dask loading is
    simpler than adding a dask dependency for no real benefit.
    """
    datasets = [xr.open_dataset(p) for p in paths]
    ds = xr.combine_by_coords(datasets, combine_attrs="override")
    return standardize_names(ds)


def time_dim_name(ds: xr.Dataset) -> str:
    return "time" if "time" in ds.dims else [d for d in ds.dims if "time" in d][0]
