#!/usr/bin/env python3
"""
Plot one PNG per day, each showing that day's synoptic-hour (00/06/12/18
UTC) snapshots at a single pressure level -- from data already fetched by
download_synoptic_hourly.py. Never touches the CDS API.

Usage:
    python plot_daily_synoptic.py
    python plot_daily_synoptic.py --data-dir data_daily --out-dir outputs_daily
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

from wind_common import AREA, standardize_names

DATA_DIR = Path("data_daily")
OUTPUT_DIR = Path("outputs_daily")
MANIFEST_FILENAME = "manifest.json"

# Approximation of ECMWF's wind-speed chart style (white/blue at calm,
# through green/yellow, to orange/red/purple at the highest speeds). Not
# pixel-exact to ECMWF's published color table -- this environment has no
# way to fetch that definition -- but matches the same visual convention.
ECMWF_WIND_STOPS = [
    (0.00, "#FFFFFF"),
    (0.10, "#C9E7F5"),
    (0.20, "#7EB6D9"),
    (0.32, "#4F9DC4"),
    (0.42, "#5CB88A"),
    (0.52, "#9ED96C"),
    (0.62, "#F5E642"),
    (0.72, "#F5A623"),
    (0.82, "#E8542A"),
    (0.90, "#C31F1F"),
    (0.96, "#8B1FA2"),
    (1.00, "#D183D1"),
]
ECMWF_WIND_CMAP = LinearSegmentedColormap.from_list("ecmwf_wind", ECMWF_WIND_STOPS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path} -- run download_synoptic_hourly.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    level = manifest["level"]

    ds = xr.open_dataset(data_dir / manifest["file"])
    ds = standardize_names(ds)
    if "level" in ds.dims:
        ds = ds.sel(level=level)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        have_cartopy = True
    except ImportError:
        have_cartopy = False

    vmax = float(np.sqrt(ds["u"] ** 2 + ds["v"] ** 2).max())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times = pd.to_datetime(ds["time"].values)
    days = sorted({t.date() for t in times})

    proj = ccrs.PlateCarree() if have_cartopy else None
    for day in days:
        day_times = [t for t in times if t.date() == day]
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(9.0, 8.6),
            subplot_kw={"projection": proj} if have_cartopy else {},
            squeeze=False,
        )
        mesh = None
        for j, t in enumerate(day_times):
            ax = axes[j // ncols][j % ncols]
            frame = ds.sel(time=t)
            u, v = frame["u"], frame["v"]
            speed = np.sqrt(u**2 + v**2)
            lon = frame["longitude"].values
            lat = frame["latitude"].values

            kwargs = {"transform": ccrs.PlateCarree()} if have_cartopy else {}
            mesh = ax.pcolormesh(
                lon, lat, speed.values,
                cmap=ECMWF_WIND_CMAP, vmin=0, vmax=vmax, shading="auto", **kwargs
            )
            step = max(1, len(lon) // 20)
            ax.quiver(
                lon[::step], lat[::step],
                u.values[::step, ::step], v.values[::step, ::step],
                color="black", scale=200, width=0.004, **kwargs
            )
            if have_cartopy:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())
            ax.set_title(t.strftime("%H:%M UTC"), fontsize=11)

        for j in range(len(day_times), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.suptitle(
            f"{level} hPa wind, {day.isoformat()} — Singapore / Peninsular Malaysia / Sumatra / Kalimantan\n"
            "(ERA5 synoptic-hour snapshots, not daily-averaged)",
            fontsize=13, y=0.98,
        )
        fig.subplots_adjust(right=0.88, top=0.88, left=0.06, wspace=0.15, hspace=0.2)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])
        fig.colorbar(mesh, cax=cbar_ax, label="Wind speed (m/s)")

        out_path = out_dir / f"wind_{level}hpa_{day.isoformat()}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
