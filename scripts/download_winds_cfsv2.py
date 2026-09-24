#!/usr/bin/env python3
"""
Download NOAA CFSv2 (Climate Forecast System version 2) reanalysis daily-mean
pressure-level winds (925/850/700 hPa) over Southeast Asia, as an alternative
data source to the ERA5/CDS pipeline (scripts/download_winds.py, untouched by
this file).

Why CFSv2 instead of raw GFS: CFSv2 has been NOAA's single, consistent
operational analysis since 2011, so every comparison year here (2015, 2019,
2023, 2026) is drawn from the same model version -- unlike raw GFS, which
had a major model change in 2019 that would confound a multi-year
comparison. Resolution (~0.5 deg) is comparable to ERA5's 0.25 deg.

Data source: NOAA NCEI public THREDDS archive of CFSv2 operational
analysis, 4 synoptic cycles/day (00/06/12/18 UTC). There is no server-side
daily-mean product here (unlike CDS's derived-era5-pressure-levels-daily-
statistics) -- this script downloads all 4 cycles for each day and
computes the vector-mean itself.

IMPORTANT / UNVERIFIED: this environment's network policy blocks noaa.gov
domains, so the THREDDS catalog base URL and dataset ID in cfsv2_common.py
(THREDDS_CATALOG_BASE) could not be verified against a live server before
writing this. Expect the first real run to need at least one round of
fixing the exact URL/catalog structure -- discover_day_catalog() and
find_cycle_entry() raise with the raw server response / catalog contents
specifically so that's easy to do.

Usage:
    python download_winds_cfsv2.py
    python download_winds_cfsv2.py --years 2015,2019,2023,2026 --window-end 09-17
"""

import argparse
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
import xarray as xr

from cfsv2_common import (
    AREA,
    CYCLES_UTC,
    DATA_DIR,
    DEFAULT_YEARS,
    LEVELS,
    MANIFEST_FILENAME,
    PUBLICATION_LAG_DAYS,
    THREDDS_CATALOG_BASE,
    WINDOW_START,
    daily_file_stub,
    days_in_window,
    default_window_end,
    parse_month_day,
    window_label,
)

THREDDS_NS = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"


def discover_day_catalog(year: int, month: int, day: int, session: requests.Session) -> list[tuple[str, str]]:
    """Return [(name, urlPath), ...] for every dataset entry in that day's
    THREDDS catalog. Raises with the raw response if the catalog doesn't
    exist or doesn't have the expected structure, rather than guessing --
    see this file's module docstring.
    """
    catalog_url = f"{THREDDS_CATALOG_BASE}/{year}/{year}{month:02d}/{year}{month:02d}{day:02d}/catalog.xml"
    resp = session.get(catalog_url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"CFSv2 catalog fetch failed for {year}-{month:02d}-{day:02d}: "
            f"HTTP {resp.status_code} at {catalog_url}\n"
            "The dataset ID in cfsv2_common.THREDDS_CATALOG_BASE "
            "('model-cfs_v2_anl_6h') is unverified -- this is the first "
            "thing to check/correct."
        )

    root = ET.fromstring(resp.content)
    entries = [
        (ds.attrib.get("name"), ds.attrib.get("urlPath"))
        for ds in root.iter(f"{THREDDS_NS}dataset")
        if ds.attrib.get("urlPath")
    ]
    if not entries:
        raise RuntimeError(
            f"CFSv2 catalog for {year}-{month:02d}-{day:02d} at {catalog_url} "
            "returned no dataset entries with a urlPath -- the catalog XML "
            f"structure may differ from expected. Raw response (first 2000 chars):\n"
            f"{resp.text[:2000]}"
        )
    return entries


def find_cycle_entry(entries: list[tuple[str, str]], year: int, month: int, day: int, cycle: int) -> str:
    """Pick the pressure-level analysis file for one cycle out of a day's
    catalog entries. Matching is heuristic (by date+cycle in the filename,
    preferring names containing 'pgb') since the exact naming convention
    is unverified -- raises with the full entry list if it can't uniquely
    identify one, so the heuristic can be corrected against real names.
    """
    cycle_tag = f"{year}{month:02d}{day:02d}{cycle:02d}"
    candidates = [url_path for name, url_path in entries if cycle_tag in (name or "") or cycle_tag in url_path]
    pgb_candidates = [c for c in candidates if "pgb" in c.lower()]
    if len(pgb_candidates) == 1:
        return pgb_candidates[0]
    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Could not uniquely identify the {cycle:02d}Z pressure-level analysis "
        f"file for {year}-{month:02d}-{day:02d}. Catalog entries: "
        f"{[name for name, _ in entries]}\n"
        "Update find_cycle_entry()'s matching heuristic to match these "
        "real filenames."
    )


def build_fileserver_url(url_path: str) -> str:
    """Standard THREDDS Data Server convention: the fileServer service for
    any catalog entry lives at {server_root}/thredds/fileServer/{urlPath}.
    This part is generic TDS behavior, not dataset-specific, so it's on
    firmer ground than the catalog base URL/dataset ID itself.
    """
    parsed = urlparse(THREDDS_CATALOG_BASE)
    return f"{parsed.scheme}://{parsed.netloc}/thredds/fileServer/{url_path}"


def download_file(url: str, dest: Path, session: requests.Session) -> None:
    resp = session.get(url, timeout=180, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed: HTTP {resp.status_code} at {url}")
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def load_cycle_uv(path: Path, levels: list[int], area: list[float]) -> xr.Dataset:
    """Open one cycle's GRIB2 file, keep only u/v wind at our pressure
    levels, and subset to our lat/lon box."""
    datasets = []
    for short_name in ("u", "v"):
        ds = xr.open_dataset(
            path, engine="cfgrib",
            backend_kwargs={
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": short_name},
                "indexpath": "",
            },
        )
        datasets.append(ds)
    merged = xr.merge(datasets)

    level_dim = "isobaricInhPa"
    if level_dim in merged.coords:
        merged = merged.sel({level_dim: levels})

    north, west, south, east = area
    lat_name = "latitude" if "latitude" in merged.coords else "lat"
    lon_name = "longitude" if "longitude" in merged.coords else "lon"
    lat_ascending = float(merged[lat_name][0]) < float(merged[lat_name][-1])
    lat_slice = slice(south, north) if lat_ascending else slice(north, south)
    merged = merged.sel({lat_name: lat_slice, lon_name: slice(west, east)})

    return merged.load()


def daily_mean_for_day(
    session: requests.Session, year: int, month: int, day: int,
    levels: list[int], area: list[float], tmp_dir: Path,
) -> xr.Dataset:
    """Fetch all 4 synoptic cycles for one day and return their vector
    mean -- the CFSv2 equivalent of CDS's daily_mean statistic, computed
    ourselves since the archive doesn't offer it server-side."""
    entries = discover_day_catalog(year, month, day, session)
    cycle_datasets = []
    for cycle in CYCLES_UTC:
        url_path = find_cycle_entry(entries, year, month, day, cycle)
        url = build_fileserver_url(url_path)
        raw_path = tmp_dir / f"cfsv2_{year}{month:02d}{day:02d}{cycle:02d}.grib2"
        download_file(url, raw_path, session)
        ds = load_cycle_uv(raw_path, levels, area)
        cycle_datasets.append(ds.expand_dims(cycle=[cycle]))
        raw_path.unlink()

    combined = xr.concat(cycle_datasets, dim="cycle")
    return combined.mean(dim="cycle", keep_attrs=True)


def download_day(year: int, month: int, day: int, data_dir: Path, tmp_dir: Path) -> Path:
    out_path = daily_file_stub(data_dir, year, month, day).with_suffix(".nc")
    if out_path.exists():
        print(f"[{year}-{month:02d}-{day:02d}] already downloaded -> {out_path}")
        return out_path

    session = requests.Session()
    print(f"[{year}-{month:02d}-{day:02d}] fetching {len(CYCLES_UTC)} cycles ...", flush=True)
    daily_mean = daily_mean_for_day(session, year, month, day, LEVELS, AREA, tmp_dir)
    daily_mean.to_netcdf(out_path)
    print(f"[{year}-{month:02d}-{day:02d}] saved -> {out_path}", flush=True)
    return out_path


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
        "--lag-days", type=int, default=PUBLICATION_LAG_DAYS,
        help=f"Days to subtract from today when --window-end isn't given (default: {PUBLICATION_LAG_DAYS})",
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help="Where to save downloaded NetCDF files and the manifest",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Parallel day-downloads (default: 4) -- this archive is plain HTTP file "
             "serving with no request queue, so parallelizing is safe/fast, but keep "
             "this modest to be a reasonable citizen of NOAA's public server.",
    )
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",")]
    window_start = parse_month_day(args.window_start)
    window_end = parse_month_day(args.window_end) if args.window_end else default_window_end(lag_days=args.lag_days)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = data_dir / "_tmp_grib"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shared_months_days = days_in_window(years[0], window_start, window_end)
    expected_num_days = sum(len(days) for days in shared_months_days.values())
    label = window_label(window_start, window_end)
    print(f"Comparison window: {label} ({expected_num_days} days), years: {years}", flush=True)

    jobs = [
        (year, month, day)
        for year in years
        for month, days in shared_months_days.items()
        for day in days
    ]

    year_files: dict[int, list[str]] = {year: [] for year in years}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(download_day, year, month, day, data_dir, tmp_dir): (year, month, day)
            for year, month, day in jobs
        }
        for future in as_completed(futures):
            year, month, day = futures[future]
            path = future.result()  # re-raises with full context if this day failed
            year_files[year].append(path.name)

    for year in years:
        if len(year_files[year]) != expected_num_days:
            raise RuntimeError(
                f"[{year}] expected {expected_num_days} daily files but got "
                f"{len(year_files[year])} -- a day's download must have failed silently."
            )
        print(f"[{year}] validated: {expected_num_days} days OK", flush=True)

    manifest = {
        "years": years,
        "window_start": args.window_start,
        "window_end": f"{window_end[0]:02d}-{window_end[1]:02d}",
        "window_label": label,
        "expected_num_days": expected_num_days,
        "levels": LEVELS,
        "area": AREA,
        "files": {year: sorted(names) for year, names in year_files.items()},
    }
    manifest_path = data_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest -> {manifest_path}")

    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # non-empty (a failed run left raw GRIB behind) -- leave it for inspection


if __name__ == "__main__":
    main()
