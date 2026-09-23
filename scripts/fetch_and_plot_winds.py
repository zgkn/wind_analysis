#!/usr/bin/env python3
"""
Fetch ERA5 monthly-mean pressure-level winds (925/850/700 hPa) over
Southeast Asia (Singapore, Peninsular Malaysia, Sumatra, Kalimantan)
for a set of years, and plot a year x level grid of mean wind maps.

Data source: Copernicus Climate Data Store (CDS)
Dataset:     reanalysis-era5-pressure-levels-monthly-means
Product:     monthly_averaged_reanalysis

Requires a CDS API key configured either via:
  - a ~/.cdsapirc file, or
  - the CDSAPI_URL and CDSAPI_KEY environment variables
(the GitHub Actions workflow in this repo sets up ~/.cdsapirc from
secrets before calling this script).

Usage:
    python fetch_and_plot_winds.py
    python fetch_and_plot_winds.py --years 2015,2019,2023,2026 --skip-download
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Years to compare, and which months (Aug-Oct) are actually available for
# each. 2026 is a partial year at the time this was written -- only include
# months that have already occurred. Update this as more months land.
DEFAULT_YEAR_MONTHS = {
    2015: [8, 9, 10],
    2019: [8, 9, 10],
    2023: [8, 9, 10],
    2026: [8, 9],  # partial season -- update once Oct 2026 data exists
}

LEVELS = [925, 850, 700]  # hPa

# CDS area is [North, West, South, East]
AREA = [10, 95, -10, 120]  # covers Singapore, Peninsular Malaysia, Sumatra, Kalimantan

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_year(client, year: int, months: list[int], data_dir: Path) -> Path:
    """Download monthly-mean u/v wind at the target levels for one year.

    Returns the path to the downloaded NetCDF file. Skips download if the
    file already exists (handy for re-runs / manual triggers).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"era5_winds_{year}.nc"
    if target.exists():
        print(f"[{year}] already downloaded -> {target}")
        return target

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(lv) for lv in LEVELS],
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "time": "00:00",
        "area": AREA,
        "format": "netcdf",
    }

    print(f"[{year}] requesting months {months} at levels {LEVELS} hPa ...")
    client.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        request,
        str(target),
    )
    print(f"[{year}] saved -> {target}")
    return target


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


def load_year_mean(path: Path) -> xr.Dataset:
    """Load one year's file and return the Aug-Oct time-mean u/v at each level."""
    ds = xr.open_dataset(path)
    ds = _standardize_names(ds)
    time_dim = "time" if "time" in ds.dims else [d for d in ds.dims if "time" in d][0]
    return ds.mean(dim=time_dim, keep_attrs=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grid(year_means: dict[int, xr.Dataset], levels: list[int], out_path: Path):
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
        "Mean Aug\u2013Oct wind, 925/850/700 hPa \u2014 Singapore / Peninsular Malaysia / Sumatra / Kalimantan\n"
        "(ERA5 reanalysis; 2026 partial season, ERA5T preliminary)",
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

def parse_year_months(arg: str) -> dict[int, list[int]]:
    """Parse '--years 2015,2019,2023,2026' using DEFAULT_YEAR_MONTHS for months."""
    years = [int(y.strip()) for y in arg.split(",")]
    result = {}
    for y in years:
        if y not in DEFAULT_YEAR_MONTHS:
            raise ValueError(f"No default month list for {y}; edit DEFAULT_YEAR_MONTHS.")
        result[y] = DEFAULT_YEAR_MONTHS[y]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years", default="2015,2019,2023,2026",
        help="Comma-separated years to include (default: 2015,2019,2023,2026)",
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

    year_months = parse_year_months(args.years)
    data_dir = Path(args.data_dir)

    if not args.skip_download:
        import cdsapi
        client = cdsapi.Client()
        for year, months in year_months.items():
            download_year(client, year, months, data_dir)
    else:
        print("skip-download set: reusing files already in", data_dir)

    year_means = {}
    for year in year_months:
        path = data_dir / f"era5_winds_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} -- run without --skip-download first, "
                "or check the CDS request succeeded."
            )
        year_means[year] = load_year_mean(path)

    plot_grid(year_means, LEVELS, Path(args.out))


if __name__ == "__main__":
    main()
