#!/usr/bin/env python3
"""
Download raw ERA5 synoptic-hour (00/06/12/18 UTC) pressure-level wind at a
single level, for a range of September days in one year -- NOT averaged
into a daily mean, unlike download_winds.py. For inspecting day-to-day (and
within-day) synoptic evolution, rather than a seasonal comparison.

Data source: Copernicus Climate Data Store (CDS)
Dataset:     reanalysis-era5-pressure-levels (same fast archive dataset as
             download_winds.py -- no toolbox compute queue)

This writes its own small manifest.json, separate from the seasonal
pipeline's data/manifest.json (different shape: one year, one level, raw
hourly timesteps rather than a multi-year daily-mean comparison).

Usage:
    python download_synoptic_hourly.py
    python download_synoptic_hourly.py --year 2026 --level 925 --start-day 8 --end-day 18
"""

import argparse
import json
import zipfile
from pathlib import Path

import xarray as xr

from wind_common import AREA, default_window_end, standardize_names

SYNOPTIC_HOURS = ["00:00", "06:00", "12:00", "18:00"]
DATASET = "reanalysis-era5-pressure-levels"
DEFAULT_LAG_DAYS = 6  # matches download_winds.py's ERA5T_LAG_DAYS

DATA_DIR = Path("data_daily")
MANIFEST_FILENAME = "manifest.json"


def _download_and_extract(client, request: dict, stub: Path) -> Path:
    """Retrieve one CDS request; return the real NetCDF path, unwrapping a
    zip if CDS delivers one (observed behavior for this dataset family)."""
    raw = stub.with_suffix(".download")
    client.retrieve(DATASET, request, str(raw))

    if not zipfile.is_zipfile(raw):
        final = stub.with_suffix(".nc")
        raw.replace(final)
        return final

    with zipfile.ZipFile(raw) as zf:
        nc_members = [n for n in zf.namelist() if n.endswith(".nc")]
        if len(nc_members) != 1:
            raise RuntimeError(
                f"{raw} is a zip archive but doesn't contain exactly one .nc "
                f"file (found {zf.namelist()}) -- this script requests a "
                "single pressure level so expects one combined file; if CDS "
                "splits u/v into separate members the way it did for the "
                "old multi-level daily-statistics dataset, this needs the "
                "same multi-member handling download_winds.py has."
            )
        extracted = zf.read(nc_members[0])
    final = stub.with_suffix(".nc")
    final.write_bytes(extracted)
    raw.unlink()
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2026, help="Year to fetch (default: 2026)")
    parser.add_argument("--level", type=int, default=925, help="Pressure level in hPa (default: 925)")
    parser.add_argument("--start-day", type=int, default=8, help="First September day to include (default: 8)")
    parser.add_argument(
        "--end-day", type=int, default=None,
        help="Last September day to include (default: latest day CDS likely has, today minus --lag-days)",
    )
    parser.add_argument(
        "--lag-days", type=int, default=DEFAULT_LAG_DAYS,
        help=f"Days behind today to use as --end-day when it isn't given (default: {DEFAULT_LAG_DAYS})",
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Where to save the downloaded NetCDF + manifest")
    args = parser.parse_args()

    if args.end_day is not None:
        end_day = args.end_day
    else:
        month, day = default_window_end(lag_days=args.lag_days)
        if month != 9:
            raise ValueError(
                f"today minus --lag-days ({args.lag_days}) lands in month {month}, not "
                "September -- pass --end-day explicitly."
            )
        end_day = day

    if end_day < args.start_day:
        raise ValueError(f"--end-day {end_day} is before --start-day {args.start_day}")

    days = list(range(args.start_day, end_day + 1))
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    request = {
        "product_type": "reanalysis",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(args.level)],
        "year": str(args.year),
        "month": "09",
        "day": [f"{d:02d}" for d in days],
        "time": SYNOPTIC_HOURS,
        "area": AREA,
        "format": "netcdf",
    }

    print(
        f"Requesting {args.year}-09-{args.start_day:02d} to {args.year}-09-{end_day:02d} "
        f"at {args.level} hPa, hours {SYNOPTIC_HOURS} ...", flush=True,
    )

    import cdsapi
    client = cdsapi.Client()
    stub = data_dir / f"era5_synoptic_{args.year}_09_{args.start_day:02d}-{end_day:02d}_{args.level}hpa"
    out_path = _download_and_extract(client, request, stub)
    print(f"saved -> {out_path}")

    ds = xr.open_dataset(out_path)
    ds = standardize_names(ds)
    time_dim = "time" if "time" in ds.dims else [d for d in ds.dims if "time" in d][0]
    actual_timesteps = ds.sizes[time_dim]
    expected_timesteps = len(days) * len(SYNOPTIC_HOURS)
    ds.close()
    if actual_timesteps != expected_timesteps:
        raise RuntimeError(
            f"Expected {expected_timesteps} timesteps ({len(days)} days x "
            f"{len(SYNOPTIC_HOURS)} hours) but got {actual_timesteps} -- CDS "
            "likely doesn't have all requested hours yet for the most recent "
            "day(s). Try a smaller --end-day."
        )
    print(f"validated: {actual_timesteps} timesteps OK")

    manifest = {
        "year": args.year,
        "level": args.level,
        "start_day": args.start_day,
        "end_day": end_day,
        "hours": SYNOPTIC_HOURS,
        "area": AREA,
        "file": out_path.name,
    }
    manifest_path = data_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
