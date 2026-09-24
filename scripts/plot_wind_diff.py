#!/usr/bin/env python3
"""
Plot wind SPEED and DIRECTION differences between a reference year (default:
the most recent year in the manifest, i.e. 2026) and each other comparison
year, from data already fetched by download_winds.py. Never touches the CDS
API -- reads data/manifest.json and the NetCDF files it lists.

Produces two separate figures, each a (comparison years x levels) grid:
  - wind_speed_diff.png: scalar speed difference (m/s), reference minus
    each comparison year. Diverging color scale centered at zero: red =
    reference year faster, blue = reference year slower.
  - wind_direction_diff.png: direction difference (degrees), same layout.
    Direction here is the vector angle wind is blowing TOWARD (atan2(v,u):
    0=East, 90=North, counterclockwise-positive) -- NOT the meteorological
    "wind is FROM" convention. Positive = reference year's wind rotated
    counterclockwise relative to the comparison year.

    Grid cells where either year's wind speed is below --weak-threshold
    (default 1 m/s) are masked out (shown blank/grey): direction is
    numerically unstable -- close to meaningless -- wherever wind is near
    calm, so an unmasked direction-difference field would show large,
    noisy "changes" in regions where there's barely any wind to have a
    direction at all.

Usage:
    python plot_wind_diff.py
    python plot_wind_diff.py --reference-year 2026 --weak-threshold 1.5
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from wind_common import AREA, DATA_DIR, MANIFEST_FILENAME, OUTPUT_DIR
from plot_winds import load_year_mean

DEFAULT_WEAK_THRESHOLD_MS = 1.0


def _uv_at_level(ds: xr.Dataset, lv: int):
    if "level" in ds["u"].dims:
        return ds["u"].sel(level=lv), ds["v"].sel(level=lv)
    return ds["u"], ds["v"]


def compute_diffs(
    ref_ds: xr.Dataset, other_ds: xr.Dataset, lv: int, weak_threshold: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (speed_diff, direction_diff) at one level: ref minus other.
    direction_diff is wrapped to (-180, 180] degrees and masked to NaN
    wherever either year's speed at that cell is below weak_threshold."""
    u_ref, v_ref = _uv_at_level(ref_ds, lv)
    u_other, v_other = _uv_at_level(other_ds, lv)

    speed_ref = np.sqrt(u_ref**2 + v_ref**2)
    speed_other = np.sqrt(u_other**2 + v_other**2)
    speed_diff = speed_ref - speed_other

    dir_ref = np.degrees(np.arctan2(v_ref, u_ref))
    dir_other = np.degrees(np.arctan2(v_other, u_other))
    dir_diff = ((dir_ref - dir_other + 180) % 360) - 180

    weak = (speed_ref < weak_threshold) | (speed_other < weak_threshold)
    dir_diff = dir_diff.where(~weak)

    return speed_diff, dir_diff


def plot_diff_grid(
    fields: dict[int, dict[int, xr.DataArray]],
    levels: list[int],
    ref_year: int,
    label: str,
    title_prefix: str,
    colorbar_label: str,
    out_path: Path,
    bad_color: str | None = None,
):
    """Plot a (comparison years x levels) grid of a single diverging field."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        have_cartopy = True
    except ImportError:
        have_cartopy = False

    years = sorted(fields.keys())
    nrows, ncols = len(years), len(levels)

    proj = ccrs.PlateCarree() if have_cartopy else None
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        subplot_kw={"projection": proj} if have_cartopy else {},
        squeeze=False,
    )

    all_vals = [
        fields[y][lv].values for y in years for lv in levels
    ]
    vmax = np.nanmax([np.nanmax(np.abs(v)) for v in all_vals])

    cmap = plt.get_cmap("RdBu_r").copy()
    if bad_color:
        cmap.set_bad(bad_color)

    mesh = None
    for i, y in enumerate(years):
        for j, lv in enumerate(levels):
            ax = axes[i][j]
            field = fields[y][lv]

            lon = field["longitude"].values
            lat = field["latitude"].values

            kwargs = {"transform": ccrs.PlateCarree()} if have_cartopy else {}
            mesh = ax.pcolormesh(
                lon, lat, field.values,
                cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto", **kwargs
            )

            if have_cartopy:
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.set_extent([AREA[1], AREA[3], AREA[2], AREA[0]], crs=ccrs.PlateCarree())

            if i == 0:
                ax.set_title(f"{lv} hPa", fontsize=11)
            if j == 0:
                ax.text(
                    -0.25, 0.5, f"{ref_year} − {y}", transform=ax.transAxes,
                    fontsize=12, fontweight="bold",
                    va="center", ha="center", rotation=90,
                )

    fig.suptitle(
        f"{title_prefix}, 925/850/700 hPa — Singapore / Peninsular Malaysia / Sumatra / Kalimantan\n"
        f"(ERA5 daily-mean reanalysis; {label} window; {ref_year} vs each comparison year)",
        fontsize=13,
    )
    fig.subplots_adjust(right=0.9, top=0.90, left=0.08)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(mesh, cax=cbar_ax, label=colorbar_label)

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
        "--reference-year", type=int, default=None,
        help="Year to compare every other year against (default: the most recent year in the manifest)",
    )
    parser.add_argument(
        "--weak-threshold", type=float, default=DEFAULT_WEAK_THRESHOLD_MS,
        help=f"Wind speed (m/s) below which direction is masked as unstable/undefined (default: {DEFAULT_WEAK_THRESHOLD_MS})",
    )
    parser.add_argument(
        "--speed-out", default=str(OUTPUT_DIR / "wind_speed_diff.png"),
        help="Output path for the speed-difference figure",
    )
    parser.add_argument(
        "--direction-out", default=str(OUTPUT_DIR / "wind_direction_diff.png"),
        help="Output path for the direction-difference figure",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path} -- run download_winds.py first "
            "(or point --data-dir at a directory that has its output)."
        )
    manifest = json.loads(manifest_path.read_text())

    years = manifest["years"]
    levels = manifest["levels"]
    label = manifest["window_label"]
    expected_num_days = manifest["expected_num_days"]
    ref_year = args.reference_year or max(years)
    if ref_year not in years:
        raise ValueError(f"--reference-year {ref_year} isn't in the manifest's years: {years}")
    other_years = [y for y in years if y != ref_year]
    if not other_years:
        raise ValueError(f"No comparison years left after excluding reference year {ref_year} from {years}")

    year_means = {}
    for year in years:
        paths = [data_dir / name for name in manifest["files"][str(year)]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"[{year}] missing {missing} listed in manifest.json -- "
                "re-run download_winds.py to regenerate data/."
            )
        year_means[year] = load_year_mean(year, paths, expected_num_days)

    ref_ds = year_means[ref_year]
    speed_fields: dict[int, dict[int, xr.DataArray]] = {}
    dir_fields: dict[int, dict[int, xr.DataArray]] = {}
    for year in other_years:
        speed_fields[year] = {}
        dir_fields[year] = {}
        for lv in levels:
            speed_diff, dir_diff = compute_diffs(ref_ds, year_means[year], lv, args.weak_threshold)
            speed_fields[year][lv] = speed_diff
            dir_fields[year][lv] = dir_diff

    plot_diff_grid(
        speed_fields, levels, ref_year, label,
        title_prefix="Wind speed difference",
        colorbar_label="Speed difference (m/s)",
        out_path=Path(args.speed_out),
    )
    plot_diff_grid(
        dir_fields, levels, ref_year, label,
        title_prefix="Wind direction difference (masked where either year's speed "
                      f"< {args.weak_threshold} m/s)",
        colorbar_label="Direction change (degrees, counterclockwise-positive)",
        out_path=Path(args.direction_out),
        bad_color="lightgrey",
    )


if __name__ == "__main__":
    main()
