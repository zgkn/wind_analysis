#!/usr/bin/env python3
"""
Download ERA5 daily-mean pressure-level winds (925/850/700 hPa) over
Southeast Asia for a set of years, all sharing the same calendar window.

Data source: Copernicus Climate Data Store (CDS)
Dataset:     reanalysis-era5-pressure-levels (plain archived hourly data)

This is the download-only half of the pipeline: it fetches data and writes
a manifest.json describing exactly what was fetched, but does no plotting.
Run plot_winds.py separately against its output to (re)generate the figure
without re-hitting CDS -- handy when only the plot styling needs changing.

Earlier versions of this script used derived-era5-pressure-levels-daily-
statistics, a CDS "toolbox" dataset that computes the daily mean on demand
server-side. That dataset's compute queue was observed taking 90 minutes to
2+ hours per request under normal load -- not a bug on our end, just how
backed up that queue was. reanalysis-era5-pressure-levels is a plain
*archived* dataset (CDS just hands back a pre-existing file, no server-side
computation), which is why the original monthly-means dataset always
returned in seconds -- so this fetches the 4 synoptic hours per day
(00/06/12/18 UTC) directly from the archive and computes the daily vector-
mean ourselves, the same technique already used for the CFSv2 pipeline
(download_winds_cfsv2.py). Downstream files still end up as one value per
day, so plot_winds.py needs no changes.

Since every comparison year shares the identical (month, day) window by
construction, each month is first tried as a single combined request across
all years (year passed as a list); if CDS rejects that, it falls back to
one request per year automatically.

Requires a CDS API key configured either via:
  - a ~/.cdsapirc file, or
  - the CDSAPI_URL and CDSAPI_KEY environment variables

Usage:
    python download_winds.py
    python download_winds.py --years 2015,2019,2023,2026 --window-end 09-17
"""

import argparse
import json
import threading
import time
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

DATASET = "reanalysis-era5-pressure-levels"
SYNOPTIC_HOURS = ["00:00", "06:00", "12:00", "18:00"]

HEARTBEAT_INTERVAL_SECONDS = 120


def _run_with_heartbeat(label: str, fn, *args, interval: int = HEARTBEAT_INTERVAL_SECONDS, **kwargs):
    """Run a blocking call while printing a heartbeat periodically. Kept
    from the daily-statistics version even though this archive dataset is
    expected to return quickly -- cheap insurance against another
    unexpectedly slow CDS queue."""
    done = threading.Event()
    start = time.monotonic()

    def _heartbeat():
        while not done.wait(interval):
            elapsed = int(time.monotonic() - start)
            print(f"[{label}] still waiting on CDS ({elapsed // 60}m{elapsed % 60:02d}s elapsed) ...", flush=True)

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    try:
        return fn(*args, **kwargs)
    finally:
        done.set()
        t.join()


def _download_and_extract(client, dataset: str, request: dict, stub: Path, label: str) -> list[Path]:
    """Retrieve one CDS request and return the real NetCDF file(s) it
    contains. Some CDS datasets deliver a zip archive even with
    format=netcdf requested, sometimes splitting u/v into separate .nc
    members -- handle both shapes rather than assuming either one."""
    raw = stub.with_suffix(".download")
    _run_with_heartbeat(label, client.retrieve, dataset, request, str(raw))

    if not zipfile.is_zipfile(raw):
        final = stub.with_suffix(".rawnc")
        raw.replace(final)
        return [final]

    extracted = []
    with zipfile.ZipFile(raw) as zf:
        nc_members = [n for n in zf.namelist() if n.endswith(".nc")]
        if not nc_members:
            raise RuntimeError(f"{raw} is a zip archive with no .nc members: {zf.namelist()}")
        for member in nc_members:
            suffix = Path(member).stem
            out_path = stub.parent / f"{stub.name}_{suffix}.rawnc"
            out_path.write_bytes(zf.read(member))
            extracted.append(out_path)
    raw.unlink()
    return extracted


def _finalize_daily_mean(raw_paths: list[Path], out_path: Path) -> None:
    """Collapse raw hourly archive files (4 synoptic times/day) into a
    single daily-mean NetCDF at out_path, replacing the raw files.
    Downstream code (validate_year, plot_winds.py) then sees exactly one
    time-value per day, same as when CDS's toolbox endpoint computed this
    for us server-side -- just done client-side now against the fast
    plain archive dataset instead.
    """
    ds = open_year_dataset(raw_paths)
    time_dim = time_dim_name(ds)
    daily = ds.resample({time_dim: "1D"}).mean(keep_attrs=True).load()
    ds.close()
    daily.to_netcdf(out_path)
    for p in raw_paths:
        p.unlink()


def _base_request(month: int, days: list[int]) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(lv) for lv in LEVELS],
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in days],
        "time": SYNOPTIC_HOURS,
        "area": AREA,
        "format": "netcdf",
    }


def _month_already_downloaded(data_dir: Path, years: list[int], month: int) -> bool:
    return all(
        daily_file_stub(data_dir, year, month).with_suffix(".nc").exists()
        for year in years
    )


def _existing_month_files(data_dir: Path, years: list[int], month: int) -> dict[int, list[Path]]:
    return {year: [daily_file_stub(data_dir, year, month).with_suffix(".nc")] for year in years}


def download_month_per_year(client, month: int, days: list[int], years: list[int], data_dir: Path) -> dict[int, list[Path]]:
    """Fallback path: one CDS request per year for this month."""
    result = {}
    for year in years:
        out_path = daily_file_stub(data_dir, year, month).with_suffix(".nc")
        if out_path.exists():
            print(f"[{year}-{month:02d}] already downloaded -> {out_path}")
            result[year] = [out_path]
            continue

        stub = daily_file_stub(data_dir, year, month)
        request = _base_request(month, days)
        request["year"] = str(year)
        label = f"{year}-{month:02d}"
        print(f"[{label}] requesting {len(days)} day(s) x {len(SYNOPTIC_HOURS)} synoptic hour(s) ...")
        raw_paths = _download_and_extract(client, DATASET, request, stub, label)
        _finalize_daily_mean(raw_paths, out_path)
        print(f"[{label}] saved -> {out_path}")
        result[year] = [out_path]
    return result


def download_month(client, month: int, days: list[int], years: list[int], data_dir: Path) -> dict[int, list[Path]]:
    """Download one month's data across all comparison years, preferring a
    single combined request (year as a list) over one request per year.
    Falls back to download_month_per_year if CDS rejects a multi-year
    request (observed to happen on the old toolbox dataset over its "cost"
    limit; unconfirmed whether this archive dataset has the same limit).
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    if _month_already_downloaded(data_dir, years, month):
        existing = _existing_month_files(data_dir, years, month)
        print(f"[month {month:02d}] already downloaded for {years}")
        return existing

    label = f"month {month:02d} x {years}"
    stub = data_dir / f"era5_daily_winds_multiyear_{month:02d}"
    request = _base_request(month, days)
    request["year"] = [str(y) for y in years]

    try:
        print(f"[{label}] requesting {len(days)} day(s) x {len(years)} year(s) as one combined request ...")
        raw_paths = _download_and_extract(client, DATASET, request, stub, label)
    except Exception as exc:
        print(f"[{label}] combined multi-year request failed ({exc}); falling back to one request per year")
        stub.with_suffix(".download").unlink(missing_ok=True)
        return download_month_per_year(client, month, days, years, data_dir)

    combined = open_year_dataset(raw_paths)
    time_dim = time_dim_name(combined)
    result = {}
    for year in years:
        year_ds = combined.where(combined[time_dim].dt.year == year, drop=True)
        daily = year_ds.resample({time_dim: "1D"}).mean(keep_attrs=True).load()
        out_path = daily_file_stub(data_dir, year, month).with_suffix(".nc")
        daily.to_netcdf(out_path)
        result[year] = [out_path]
    combined.close()
    for p in raw_paths:
        p.unlink()

    print(f"[{label}] saved -> {[str(p) for paths in result.values() for p in paths]}")
    return result


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

    # Every year shares the identical (month, day) window by construction,
    # so the day list per month is the same for all years -- take it from
    # the first year and reuse it for the combined multi-year requests.
    shared_months_days = days_in_window(years[0], window_start, window_end)
    expected_num_days = sum(len(days) for days in shared_months_days.values())
    label = window_label(window_start, window_end)
    print(f"Comparison window: {label} ({expected_num_days} days), years: {years}", flush=True)

    import cdsapi
    client = cdsapi.Client()

    year_paths: dict[int, list[Path]] = {year: [] for year in years}
    for month, days in sorted(shared_months_days.items()):
        month_result = download_month(client, month, days, years, data_dir)
        for year, paths in month_result.items():
            year_paths[year].extend(paths)

    year_files: dict[int, list[str]] = {}
    for year in years:
        validate_year(year, year_paths[year], expected_num_days)
        year_files[year] = [p.name for p in year_paths[year]]
        print(f"[{year}] validated: {expected_num_days} days OK", flush=True)

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
