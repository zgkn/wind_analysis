#!/usr/bin/env python3
"""
Plot a (years x levels) grid of mean wind maps from CFSv2 data already
fetched by download_winds_cfsv2.py -- reads data_cfsv2/manifest.json and the
per-day NetCDF files it lists. Never touches NOAA's servers, so re-run this
freely while tweaking the figure.

Each day's file is already the vector mean of that day's 4 synoptic cycles
(computed at download time); this script averages across all days in the
window to get the seasonal mean per year, then plots as speed + vectors.

Usage:
    python plot_winds_cfsv2.py
    python plot_winds_cfsv2.py --data-dir data_cfsv2 --out outputs_cfsv2/wind_grid_cfsv2.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from cfsv2_common import AREA, DATA_DIR, MANIFEST_FILENAME, OUTPUT_DIR


def _level_dim(ds: xr.Dataset) -> str | None:
    for candidate in ("level", "isobaricInhPa"):
        if candidate in ds.dims or candidate in ds.coords:
            return candidate
    return None


def load_year_mean(year: int, paths: list[Path], expected_num_days: int) -> xr.Dataset:
    """Load one year's daily files (each already a same-day 4-cycle mean)
    and average across days. Fails loudly if the file count doesn't match
    the manifest, rather than silently averaging over fewer days."""
    if len(paths) != expected_num_days:
        raise RuntimeError(
            f"[{year}] manifest expects {expected_num_days} daily files but "
            f"found {len(paths)} -- data_cfsv2/ may be stale or incomplete "
            "relative to manifest.json. Re-run download_winds_cfsv2.py."
        )
    datasets = [xr.open_dataset(p) for p in paths]
    combined = xr.concat(datasets, dim="day")
    return combined.mean(dim="day", keep_attrs=True)


def plot_grid(year_means: dict[int, xr.Dataset], levels: list[int], label: str, out_path: Path):
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

    sample_ds = year_means[years[0]]
    level_dim = _level_dim(sample_ds)

    def uv_at_level(ds: xr.Dataset, lv: int):
        if level_dim and level_dim in ds["u"].dims:
            return ds["u"].sel({level_dim: lv}), ds["v"].sel({level_dim: lv})
        return ds["u"], ds["v"]

    all_speeds = []
    for y in years:
        ds = year_means[y]
        for lv in levels:
            u, v = uv_at_level(ds, lv)
            all_speeds.append(np.sqrt(u**2 + v**2).values)
    vmax = np.nanmax([np.nanmax(s) for s in all_speeds])

    mesh = None
    for i, y in enumerate(years):
        ds = year_means[y]
        for j, lv in enumerate(levels):
            ax = axes[i][j]
            u, v = uv_at_level(ds, lv)
            speed = np.sqrt(u**2 + v**2)

            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            lon = ds[lon_name].values
            lat = ds[lat_name].values

            kwargs = {"transform": ccrs.PlateCarree()} if have_cartopy else {}
            mesh = ax.pcolormesh(
                lon, lat, speed.values,
                cmap="viridis", vmin=0, vmax=vmax, shading="auto", **kwargs
            )

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
        f"Mean {label} wind, 925/850/700 hPa — Singapore / Peninsular Malaysia / Sumatra / Kalimantan\n"
        "(NOAA CFSv2 reanalysis; same calendar window per year; daily means from 4 synoptic cycles)",
        fontsize=13,
    )
    fig.subplots_adjust(right=0.9, top=0.90, left=0.08)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(mesh, cax=cbar_ax, label="Wind speed (m/s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"saved figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help="Directory containing manifest.json and the downloaded NetCDF files",
    )
    parser.add_argument(
        "--out", default=str(OUTPUT_DIR / "wind_grid_cfsv2_925_850_700.png"),
        help="Output figure path",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path} -- run download_winds_cfsv2.py first "
            "(or point --data-dir at a directory that has its output)."
        )
    manifest = json.loads(manifest_path.read_text())

    years = manifest["years"]
    levels = manifest["levels"]
    label = manifest["window_label"]
    expected_num_days = manifest["expected_num_days"]

    year_means = {}
    for year in years:
        paths = [data_dir / name for name in manifest["files"][str(year)]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"[{year}] missing {missing} listed in manifest.json -- "
                "re-run download_winds_cfsv2.py to regenerate data_cfsv2/."
            )
        year_means[year] = load_year_mean(year, paths, expected_num_days)

    plot_grid(year_means, levels, label, Path(args.out))


if __name__ == "__main__":
    main()
