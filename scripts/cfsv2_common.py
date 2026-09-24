"""Shared config and calendar-window helpers for the CFSv2 pipeline.

Deliberately independent from wind_common.py (the ERA5 pipeline's shared
module) -- this track must not touch or depend on anything ERA5-specific,
even though the window-math logic is conceptually identical.
"""

from datetime import date, timedelta
from pathlib import Path

DEFAULT_YEARS = [2015, 2019, 2023, 2026]

LEVELS = [925, 850, 700]  # hPa

# [North, West, South, East] -- same domain as the ERA5 pipeline, kept
# consistent so the two are visually comparable.
AREA = [10, 95, -10, 120]

WINDOW_START = (8, 1)

# CFSv2's operational analysis is near-real-time (unlike ERA5T's ~5-6 day
# reprocessing lag), but the exact publication delay on NOAA's public
# archive is unverified from this environment -- treat as a starting
# guess to correct once we see real results.
PUBLICATION_LAG_DAYS = 2

# The four synoptic analysis cycles per day; a "daily mean" here is the
# vector average of these four instantaneous analyses, since CFSv2's
# archive has no server-side daily-statistic endpoint the way CDS's
# derived-era5-pressure-levels-daily-statistics does.
CYCLES_UTC = [0, 6, 12, 18]

DATA_DIR = Path("data_cfsv2")
OUTPUT_DIR = Path("outputs_cfsv2")
MANIFEST_FILENAME = "manifest.json"

# NOTE: this base URL/dataset ID is a best-effort identification of NOAA's
# public CFSv2 operational analysis archive on NCEI's THREDDS server, from
# training knowledge -- NOT verified against a live server (this sandbox's
# network policy blocks noaa.gov domains; the download workflow's runner
# will have real access). If catalog fetches 404, this ID is the first
# thing to correct -- see download_winds_cfsv2.py's discover_catalog().
THREDDS_CATALOG_BASE = "https://www.ncei.noaa.gov/thredds/catalog/model-cfs_v2_anl_6h"
THREDDS_FILESERVER_BASE = "https://www.ncei.noaa.gov/thredds/fileServer/model-cfs_v2_anl_6h"


def default_window_end(today: date | None = None, lag_days: int = PUBLICATION_LAG_DAYS) -> tuple[int, int]:
    today = today or date.today()
    cutoff = today - timedelta(days=lag_days)
    return (cutoff.month, cutoff.day)


def days_in_window(year: int, start: tuple[int, int], end: tuple[int, int]) -> dict[int, list[int]]:
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
    month_str, day_str = arg.split("-")
    return (int(month_str), int(day_str))


def window_label(start: tuple[int, int], end: tuple[int, int]) -> str:
    return f"{date(2000, *start).strftime('%b %d')}–{date(2000, *end).strftime('%b %d')}"


def daily_file_stub(data_dir: Path, year: int, month: int, day: int) -> Path:
    return data_dir / f"cfsv2_daily_winds_{year}{month:02d}{day:02d}"
