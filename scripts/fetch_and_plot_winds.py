#!/usr/bin/env python3
"""
Fetch ERA5 daily-mean pressure-level winds (925/850/700 hPa) over
Southeast Asia (Singapore, Peninsular Malaysia, Sumatra, Kalimantan)
for a set of years, and plot a year x level grid of mean wind maps.

Data source: Copernicus Climate Data Store (CDS)
Dataset:     derived-era5-pressure-levels-daily-statistics
Statistic:   daily_mean

Every year is averaged over the *same* calendar window (month/day to
month/day), not whole calendar months. This matters because whole-month
averaging silently mismatches years once the current year is still
in-progress (e.g. comparing an Aug-Oct mean against an Aug-Sep mean is
not an apples-to-apples comparison of the same seasonal signal). Using
daily data lets every year -- including one that isn't finished yet --
share the exact same day range.

The window's end date defaults to "today minus a lag", since ERA5T
(preliminary near-real-time ERA5) daily data is typically only
available a few days after the fact; see --lag-days.

Requires a CDS API key configured either via:
  - a ~/.cdsapirc file, or
  - the CDSAPI_URL and CDSAPI_KEY environment variables
(the GitHub Actions workflow in this repo sets up ~/.cdsapirc from
secrets before calling this script).

Usage:
    python fetch_and_plot_winds.py
    python fetch_and_plot_winds.py --years 2015,2019,2023,2026 --skip-download
    python fetch_and_plot_winds.py --window-start 08-01 --window-end 09-17
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
# --window-end defaults to a date CDS is actually likely to have. Override
# with --lag-days or an explicit --window-end if CDS has caught up further.
ERA5T_LAG_DAYS = 6

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


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


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_year(client, year: int, months_days: dict[int, list[int]], data_dir: Path) -> list[Path]:
    """Download one year's daily-mean u/v wind at the target levels, one CDS
    request per calendar month in the window (since each month may need a
    different subset of days -- a full month, or a partial one).

    Returns the list of NetCDF file paths for that year. Skips a request if
    its file already exists (handy for re-runs / manual triggers).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for month, days in sorted(months_days.items()):
        target = data_dir / f"era5_daily_winds_{year}_{month:02d}.nc"
        if target.exists():
            print(f"[{year}-{month:02d}] already downloaded -> {target}")
            paths.append(target)
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
        client.retrieve(
            "derived-era5-pressure-levels-daily-statistics",
            request,
            str(target),
        )
        print(f"[{year}-{month:02d}] saved -> {target}")
        paths.append(target)

    return paths


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _standardize_names(ds: xr.Dataset) -> xr.Dataset:
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


def load_year_mean(year: int, paths: list[Path], expected_num_days: int) -> xr.Dataset:
    """Load one year's daily files, verify the day count matches what was
    requested, and return the time-mean u/v at each level.

    Raises loudly rather than silently averaging over fewer days than
    requested -- CDS can return a short file without erroring (e.g. when a
    requested day's daily-mean doesn't exist yet), and that must not pass
    as a same-length comparison against other years.
    """
    ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords")
    ds = _standardize_names(ds)
    time_dim = "time" if "time" in ds.dims else [d for d in ds.dims if "time" in d][0]

    actual_num_days = ds.sizes[time_dim]
    if actual_num_days != expected_num_days:
        raise RuntimeError(
            f"[{year}] expected {expected_num_days} daily values but got "
            f"{actual_num_days} -- CDS likely doesn't have all requested days "
            "yet (check ERA5T publication lag / --lag-days), or a partial "
            "download is cached under data/. Delete the stale file(s) and "
            "re-run rather than silently comparing a shorter mean."
        )

    return ds.mean(dim=time_dim, keep_attrs=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(year_means: dict[int, xr.Dataset], levels: list[int], window_label: str, out_path: Path):
    """Plot a (years x levels) grid of mean wind maps: speed shading + vectors."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        have_cartopy = True
    except ImportError:
        have_cartopy = False

    years = sorted(year_means.keys())
    nrows, ncols = len(years), len(levels)

    proj = ccrs.PlateCarree() if have_cartopy else None
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        subplot_kw={"projection": proj} if have_cartopy else {},
        squeeze=False,
    )

    # find a common speed color scale across all panels for fair comparison
    all_speeds = []
    for y in years:
        ds = year_means[y]
        for lv in levels:
            if "level" in ds["u"].dims:
                u = ds["u"].sel(level=lv)
                v = ds["v"].sel(level=lv)
            else:
                u, v = ds["u"], ds["v"]
            all_speeds.append(np.sqrt(u**2 + v**2).values)
    vmax = np.nanmax([np.nanmax(s) for s in all_speeds])

    mesh = None
    for i, y in enumerate(years):
        ds = year_means[y]
        for j, lv in enumerate(levels):
            ax = axes[i][j]
            if "level" in ds["u"].dims:
                u = ds["u"].sel(level=lv)
                v = ds["v"].sel(level=lv)
            else:
                u, v = ds["u"], ds["v"]
            speed = np.sqrt(u**2 + v**2)

            lon = ds["longitude"].values
            lat = ds["latitude"].values

            kwargs = {"transform": ccrs.PlateCarree()} if have_cartopy else {}
            mesh = ax.pcolormesh(
                lon, lat, speed.values,
                cmap="viridis", vmin=0, vmax=vmax, shading="auto", **kwargs
            )

            # subsample the vector field so arrows aren't overcrowded
            step = max(1, len(lon) // 20)
            ax.quiver(
                lon[::step], lat[::step],
                u.values[::step, ::step], v.values[::step, ::step],
                color="white", scale=200, width=0.003, **kwargs
            )

            if have_cartopy:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())

            if i == 0:
                ax.set_title(f"{lv} hPa", fontsize=11)
            if j == 0:
                ax.text(
                    -0.25, 0.5, str(y), transform=ax.transAxes,
                    fontsize=12, fontweight="bold",
                    va="center", ha="center", rotation=90,
                )

    fig.suptitle(
        f"Mean {window_label} wind, 925/850/700 hPa — Singapore / Peninsular Malaysia / Sumatra / Kalimantan\n"
        "(ERA5 daily-mean reanalysis; same calendar window per year; ERA5T preliminary for the current year)",
        fontsize=13,
    )
    fig.subplots_adjust(right=0.9, top=0.90, left=0.08)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(mesh, cax=cbar_ax, label="Wind speed (m/s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"saved figure -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_month_day(arg: str) -> tuple[int, int]:
    """Parse 'MM-DD' into (month, day)."""
    month_str, day_str = arg.split("-")
    return (int(month_str), int(day_str))


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
        "--skip-download", action="store_true",
        help="Reuse existing files in data/ instead of calling the CDS API",
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help="Where to cache downloaded NetCDF files",
    )
    parser.add_argument(
        "--out", default=str(OUTPUT_DIR / "wind_grid_925_850_700.png"),
        help="Output figure path",
    )
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",")]
    window_start = parse_month_day(args.window_start)
    window_end = parse_month_day(args.window_end) if args.window_end else default_window_end(lag_days=args.lag_days)
    data_dir = Path(args.data_dir)

    year_months_days = {year: days_in_window(year, window_start, window_end) for year in years}
    expected_num_days = sum(len(days) for days in year_months_days[years[0]].values())

    window_label = (
        f"{date(2000, *window_start).strftime('%b %d')}"
        f"–{date(2000, *window_end).strftime('%b %d')}"
    )
    print(f"Comparison window: {window_label} ({expected_num_days} days), years: {years}")

    year_paths: dict[int, list[Path]] = {}
    if not args.skip_download:
        import cdsapi
        client = cdsapi.Client()
        for year, months_days in year_months_days.items():
            year_paths[year] = download_year(client, year, months_days, data_dir)
    else:
        print("skip-download set: reusing files already in", data_dir)
        for year, months_days in year_months_days.items():
            year_paths[year] = [
                data_dir / f"era5_daily_winds_{year}_{month:02d}.nc"
                for month in sorted(months_days)
            ]

    year_means = {}
    for year, paths in year_paths.items():
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"[{year}] missing {missing} -- run without --skip-download first, "
                "or check the CDS request succeeded."
            )
        year_means[year] = load_year_mean(year, paths, expected_num_days)

    plot_grid(year_means, LEVELS, window_label, Path(args.out))


if __name__ == "__main__":
    main()
