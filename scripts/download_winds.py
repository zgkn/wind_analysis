#!/usr/bin/env python3
"""
Download ERA5 daily-mean pressure-level winds (925/850/700 hPa) over
Southeast Asia for a set of years, all sharing the same calendar window.

Data source: Copernicus Climate Data Store (CDS)
Dataset:     derived-era5-pressure-levels-daily-statistics
Statistic:   daily_mean

This is the download-only half of the pipeline: it fetches data and writes
a manifest.json describing exactly what was fetched, but does no plotting.
Run plot_winds.py separately against its output to (re)generate the figure
without re-hitting CDS -- handy when only the plot styling needs changing.

Requires a CDS API key configured either via:
  - a ~/.cdsapirc file, or
  - the CDSAPI_URL and CDSAPI_KEY environment variables

Usage:
    python download_winds.py
    python download_winds.py --years 2015,2019,2023,2026 --window-end 09-17
"""

import argparse
import json
import zipfile
from pathlib import Path

from wind_common import (
    AREA,
    DATA_DIR,
    DEFAULT_YEARS,
    ERA5T_LAG_DAYS,
    LEVELS,
    MANIFEST_FILENAME,
    WINDOW_START,
    daily_file_stub,
    days_in_window,
    default_window_end,
    open_year_dataset,
    parse_month_day,
    time_dim_name,
    window_label,
)


def _download_and_extract(client, dataset: str, request: dict, stub: Path) -> list[Path]:
    """Retrieve one CDS request and return the real NetCDF file(s) it
    contains. derived-era5-pressure-levels-daily-statistics sometimes
    delivers a zip archive even though format=netcdf was requested (the
    older reanalysis-*-monthly-means dataset never did this), and when it
    does, u and v each come back as a separate .nc member rather than one
    combined file. Handle both shapes rather than assuming either one.
    """
    raw = stub.with_suffix(".download")
    client.retrieve(dataset, request, str(raw))

    if not zipfile.is_zipfile(raw):
        final = stub.with_suffix(".nc")
        raw.replace(final)
        return [final]

    extracted = []
    with zipfile.ZipFile(raw) as zf:
        nc_members = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_members:
            raise RuntimeError(f"{raw} is a zip archive with no .nc members: {zf.namelist()}")
        for member in nc_members:
            suffix = Path(member).stem
            out_path = stub.parent / f"{stub.name}_{suffix}.nc"
            out_path.write_bytes(zf.read(member))
            extracted.append(out_path)
    raw.unlink()
    return extracted


def download_year(client, year: int, months_days: dict[int, list[int]], data_dir: Path) -> list[Path]:
    """Download one year's daily-mean u/v wind at the target levels, one CDS
    request per calendar month in the window (a month may need a full set of
    days or a partial one, depending on where it falls in the window). CDS
    may return each month's request as one file or several (see
    _download_and_extract) -- callers shouldn't assume a fixed count.

    Returns the list of NetCDF file paths for that year. Skips a month's
    request if matching file(s) already exist (handy for re-runs / manual
    triggers).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for month, days in sorted(months_days.items()):
        stub = daily_file_stub(data_dir, year, month)
        existing = sorted(data_dir.glob(f"{stub.name}*.nc"))
        if existing:
            print(f"[{year}-{month:02d}] already downloaded -> {existing}")
            paths.extend(existing)
            continue

        request = {
            "product_type": "reanalysis",
            "variable": ["u_component_of_wind", "v_component_of_wind"],
            "pressure_level": [str(lv) for lv in LEVELS],
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in days],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": AREA,
            "format": "netcdf",
        }

        print(f"[{year}-{month:02d}] requesting {len(days)} day(s) at levels {LEVELS} hPa ...")
        month_paths = _download_and_extract(
            client, "derived-era5-pressure-levels-daily-statistics", request, stub
        )
        print(f"[{year}-{month:02d}] saved -> {month_paths}")
        paths.extend(month_paths)

    return paths


def validate_year(year: int, paths: list[Path], expected_num_days: int) -> None:
    """Fail loudly if a year's downloaded day count doesn't match what was
    requested, rather than letting a short fetch pass silently through to
    the plot stage as if it were a same-length comparison."""
    ds = open_year_dataset(paths)
    actual = ds.sizes[time_dim_name(ds)]
    ds.close()
    if actual != expected_num_days:
        raise RuntimeError(
            f"[{year}] expected {expected_num_days} daily values but got "
            f"{actual} -- CDS likely doesn't have all requested days yet "
            "(check ERA5T publication lag / --lag-days). Delete the "
            f"partial file(s) under {paths[0].parent} for this year and re-run."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years", default=",".join(str(y) for y in DEFAULT_YEARS),
        help=f"Comma-separated years to include (default: {','.join(str(y) for y in DEFAULT_YEARS)})",
    )
    parser.add_argument(
        "--window-start", default=f"{WINDOW_START[0]:02d}-{WINDOW_START[1]:02d}",
        help="Start of the shared calendar window, as MM-DD (default: 08-01)",
    )
    parser.add_argument(
        "--window-end", default=None,
        help="End of the shared calendar window, as MM-DD (default: today minus --lag-days)",
    )
    parser.add_argument(
        "--lag-days", type=int, default=ERA5T_LAG_DAYS,
        help=f"Days to subtract from today when --window-end isn't given, to stay behind "
             f"the ERA5T publication lag (default: {ERA5T_LAG_DAYS})",
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help="Where to save downloaded NetCDF files and the manifest",
    )
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",")]
    window_start = parse_month_day(args.window_start)
    window_end = parse_month_day(args.window_end) if args.window_end else default_window_end(lag_days=args.lag_days)
    data_dir = Path(args.data_dir)

    year_months_days = {year: days_in_window(year, window_start, window_end) for year in years}
    expected_num_days = sum(len(days) for days in year_months_days[years[0]].values())
    label = window_label(window_start, window_end)
    print(f"Comparison window: {label} ({expected_num_days} days), years: {years}")

    import cdsapi
    client = cdsapi.Client()

    year_files: dict[int, list[str]] = {}
    for year, months_days in year_months_days.items():
        paths = download_year(client, year, months_days, data_dir)
        validate_year(year, paths, expected_num_days)
        year_files[year] = [p.name for p in paths]

    manifest = {
        "years": years,
        "window_start": args.window_start,
        "window_end": f"{window_end[0]:02d}-{window_end[1]:02d}",
        "window_label": label,
        "expected_num_days": expected_num_days,
        "levels": LEVELS,
        "area": AREA,
        "files": year_files,
    }
    manifest_path = data_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
