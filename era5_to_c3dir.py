"""
Convert ERA5 pressure-level GRIB file(s) into C3DIR-like features (code from 02_era_explore.ipynb).

Reads one or more .grib files (or directories of them), concatenates them along time,
runs era5_to_c3dir() to interpolate IWC / LWC / RWC (g m-3) and occurrence masks onto the
C3DIR altitude grid (80 bins x 0.25 km), saves a vector (PDF/SVG) occurrence plot as a
sanity check, and writes everything to a single .nc file.

Examples:
    python era5_to_c3dir.py data/era5_boston/chunks/era5_20260601_20260603_N45.5_W74_S39.5_E68.grib
    python era5_to_c3dir.py data/era5_boston/chunks/            # every .grib in the folder
    python era5_to_c3dir.py a.grib b.grib -o c3dir_june.nc --plot c3dir_june.svg
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.interpolate import interpolate_1d
from metpy.units import units

# constants
G0, RD = 9.80665, 287.05          # gravity (m s-2), dry-air gas constant (J kg-1 K-1)
THRESH = 1e-5                     # C3DIR occurrence threshold (g m-3), the paper uses 1e-5 g m-3
ALT = np.arange(80) * 0.25        # C3DIR vertical levels grid: 80 bins, 0.25 km, centers 0..19.75 km
ALT_M = ALT * 1000


def find_gribs(inputs):
    """Expand the CLI inputs (files and/or directories) into a sorted list of .grib files."""
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(p.glob("*.grib")) + sorted(p.glob("*.grib2"))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    if not files:
        raise SystemExit("No .grib files found in the given inputs.")
    return sorted(set(files))


def open_gribs(files):
    """Open each GRIB file and concatenate along time (sorted, duplicate times dropped)."""
    parts = [xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""}) for f in files]
    if len(parts) == 1:
        return parts[0]
    # single-time files have a scalar time coord; give them a time dim so concat works
    parts = [d if "time" in d.dims else d.expand_dims("time") for d in parts]
    ds = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override", join="override")
    ds = ds.sortby("time")
    return ds.isel(time=~ds.get_index("time").duplicated())


def era5_to_c3dir(ds):
    ds = ds.rename(isobaricInhPa="plev")
    dims = ds.t.dims # (time, plev, lat, lon)

    #* Air density (kg m-3)
    # q -> mixing ratio, then density with virtual temperature (unit-checked by Pint).
    mr = mpcalc.mixing_ratio_from_specific_humidity(ds.q.values * units("kg/kg")) # get the mixing ratio
    p = ds.plev.values[None, :, None, None] * units.hPa
    rho = xr.DataArray(mpcalc.density(p, ds.t.values * units.K, mr).to("kg/m^3").magnitude,
                       coords=ds.t.coords, dims=dims)

    #* Convert mixing ratio (kg/kg) -> water content (g m-3)
    gm3 = lambda x: (x * rho * 1000).clip(min=0)
    water = {
        "IWC": gm3(ds.ciwc + ds.cswc),    # ice = cloud ice + snow
        "LWC": gm3(ds.clwc),              # cloud liquid
        "RWC": gm3(ds.crwc),              # rain
    }

    #* Convert geopotential -> height (m)
    h = (ds.z / G0).transpose(*dims).values

    #* Interpolate pressure levels onto C3DIR altitude grid  [MetPy]
    # Heights differ per column, so h is passed as xp. Bins outside the column's range -> NaN.
    ax = dims.index("plev")
    out = xr.Dataset()
    for k, v in water.items():
        res = interpolate_1d(ALT_M, h, v.transpose(*dims).values, axis=ax)
        out[k] = (("time", "alt", "latitude", "longitude"), np.moveaxis(np.asarray(res), ax, 1))

    out = out.assign_coords(time=ds.time, alt=ALT, latitude=ds.latitude, longitude=ds.longitude)
    out = out.clip(min=0) # remove interpolation undershoot

    #* Occurrence masks
    valid = out.IWC.notnull()
    for k, n in [("IWC", "ice"), ("LWC", "liquid"), ("RWC", "rain")]:
        out[f"{n}_occ"] = ((out[k] > THRESH) & valid).astype("uint8")

    out["hydrometeor_occ"] = out[["ice_occ", "liquid_occ", "rain_occ"]].to_array().max("variable")

    # metadata
    for k in water: out[k].attrs = {"units": "g m-3"}
    out.alt.attrs = {"units": "km"}
    return out


def plot_occurrence(c3, path):
    """Fraction of lat/lon cells with each hydrometeor, time x altitude. Saved as vector graphics."""
    ok = c3.IWC.notnull()
    names = {"Ice": ("ice_occ", "tab:blue"), "Liquid": ("liquid_occ", "tab:green"),
             "Rain": ("rain_occ", "tab:red"), "Any hydrometeor": ("hydrometeor_occ", "k")}

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True, sharey=True)
    for ax, (n, (v, color)) in zip(axes, names.items()):
        frac = c3[v].where(ok).mean(["latitude", "longitude"]) # fraction of grid cells with the hydrometeor, 0..1
        frac.plot.pcolormesh(x="time", y="alt", ax=ax, cmap=LinearSegmentedColormap.from_list("", ["white", color]),
                             vmin=0, vmax=1, cbar_kwargs={"label": "fraction of cells"})
        ax.set(title=f"{n} occurrence, fraction of all lat/lon grid cells", xlabel="", ylabel="Altitude (km)")
        ax.set_ylim([0, 15])

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".grib file(s) and/or directories containing them")
    ap.add_argument("-o", "--out", type=Path,
                    help="output .nc (default: c3dir_like_<first file stem>.nc next to the first input)")
    ap.add_argument("--plot", type=Path,
                    help="vector plot path, .pdf or .svg (default: output path with .pdf suffix)")
    args = ap.parse_args()

    files = find_gribs(args.inputs)
    print(f"Reading {len(files)} GRIB file(s):", *(f"  {f}" for f in files), sep="\n")
    ds = open_gribs(files)
    print(f"Concatenated: {ds.sizes['time']} times, {ds.time.values[0]} -> {ds.time.values[-1]}")

    c3 = era5_to_c3dir(ds)

    out = args.out or files[0].with_name(f"c3dir_like_{files[0].stem}.nc")
    plot = args.plot or out.with_suffix(".pdf")
    plot_occurrence(c3, plot)
    print(f"Plot   -> {plot}")

    c3.to_netcdf(out)
    print(f"NetCDF -> {out}")


if __name__ == "__main__":
    main()
