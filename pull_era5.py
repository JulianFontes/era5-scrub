"""
Pull one month of hourly ERA5 pressure-level data around Boston during the 2026 World Cup month
(June 2026 by default), in the same layout as era5_subset.nc
(valid_time x pressure_level x latitude x longitude, 16 variables, 15 levels).

Default domain: 10 x 10 degrees centered on Boston (~42.4N, 71.1W), 41 x 41 grid points.

Setup (once):
    uv sync   (cdsapi, xarray, netCDF4 are project dependencies)
    Put your key from https://cds.climate.copernicus.eu/profile in era5_secrets.txt
    (gitignored; same format as ~/.cdsapirc, which is used if the file is absent):
        url: https://cds.climate.copernicus.eu/api
        key: <your-personal-access-token>
    Accept the ERA5 licence on the dataset page once (the API refuses otherwise).

Spacing: every timestamp is exactly --step-hours apart, starting 00 UTC on
--start. Steps that divide 24 (1, 2, 3, 4, 6, 8, 12) keep the same hours every
day. For the default 30-day window (June 2026):
    --step-hours 1  -> 720 timestamps (default, every hour)
    --step-hours 3  -> 240
    --step-hours 6  -> 120
Pick a different window with --start/--end, e.g. --start 2026-07-01 --end 2026-07-31.

Examples:
    python pull_era5.py                                   # hourly, June 2026, Boston box
    python pull_era5.py --step-hours 3
    python pull_era5.py --c3dir-only                      # 8 variables, ~3x fewer requests
    python pull_era5.py --area 34.0 -84.75 33.5 -84.0     # the original 3x4 Atlanta box
    python pull_era5.py --dry-run                         # print the plan, download nothing

Downloads go in chunks of as many days as fit under the CDS request cost limit
(1 day for hourly, all 16 variables; 3 days with --c3dir-only) to --workdir. Chunk
filenames include the area, so different regions never collide. Reruns skip chunks
already on disk, so an interrupted pull resumes.
"""
import argparse, os, sys, zipfile, shutil
from datetime import date, timedelta
import numpy as np
import pandas as pd

DATASET = "reanalysis-era5-pressure-levels"
VARIABLES = {  # CDS name -> short name in the netCDF
    "divergence": "d", "fraction_of_cloud_cover": "cc", "geopotential": "z",
    "ozone_mass_mixing_ratio": "o3", "potential_vorticity": "pv",
    "relative_humidity": "r", "specific_cloud_ice_water_content": "ciwc",
    "specific_cloud_liquid_water_content": "clwc", "specific_humidity": "q",
    "specific_rain_water_content": "crwc", "specific_snow_water_content": "cswc",
    "temperature": "t", "u_component_of_wind": "u", "v_component_of_wind": "v",
    "vertical_velocity": "w", "vorticity": "vo",
}
C3DIR_VARS = ["fraction_of_cloud_cover", "geopotential", "specific_cloud_ice_water_content",
              "specific_cloud_liquid_water_content", "specific_rain_water_content",
              "specific_snow_water_content", "specific_humidity", "temperature"]
SECRETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "era5_secrets.txt")
# CDS rejects requests whose cost exceeds 60000; for this dataset cost is
# 6 x (days x hours x variables x levels), measured via the /costing endpoint.
COST_LIMIT, COST_PER_FIELD = 60000, 6
LEVELS = ["1000", "950", "900", "850", "800", "750", "650", "550",
          "450", "350", "250", "200", "150", "100", "50"]

BOSTON_AREA = [45.5, -74, 39.5, -68]   # N, W, S, E: 6 x 6 deg around Boston

def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2026-06-01", help="first day (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-06-30", help="last day, inclusive")
    p.add_argument("--step-hours", type=int, default=1,
                   help="fixed spacing between timestamps; a divisor of 24 keeps the same hours each day")
    p.add_argument("--area", type=float, nargs=4, metavar=("N", "W", "S", "E"),
                   default=BOSTON_AREA,
                   help="bounding box; default = 10x10 deg centered on Boston")
    p.add_argument("--c3dir-only", action="store_true",
                   help="only the 8 C3DIR-related variables (smaller download)")
    p.add_argument("--chunk-days", type=int, default=None,
                   help="days per CDS request; default = most that fit under the CDS cost limit")
    p.add_argument("--workdir", default="data/era5_boston/chunks")
    p.add_argument("--out", default="data/era5_boston/era5_boston_20260601_20260630.nc")
    p.add_argument("--format", choices=["nc", "grib"], default="nc",
                   help="download format; grib saves the raw chunks and skips the netCDF merge")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def pick_timestamps(start, end, step):
    if step < 1:
        sys.exit("--step-hours must be at least 1")
    return pd.date_range(f"{start} 00:00", f"{end} 23:00", freq=f"{step}h")


def chunks(start, end, days):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    while d0 <= d1:
        e = min(d0 + timedelta(days=days - 1), d1)
        yield d0, e
        d0 = e + timedelta(days=1)


def area_tag(area):
    """Short string for filenames, e.g. [47.5, -76, 37.5, -66] -> 'N47.5_W76_S37.5_E66'."""
    n, w, s, e = area
    return f"N{n:g}_W{abs(w):g}_S{s:g}_E{abs(e):g}"


def request_for(d0, d1, hours_needed, variables, area, fmt="nc"):
    days = pd.date_range(d0, d1, freq="D")
    # CDS requests are a product of year x month x day x time, so every listed hour
    # comes back for every listed day; the result is filtered to exact stamps later.
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": sorted({f"{x.year}" for x in days}),
        "month": sorted({f"{x.month:02d}" for x in days}),
        "day": sorted({f"{x.day:02d}" for x in days}),
        "time": sorted({f"{h:02d}:00" for h in hours_needed}),
        "pressure_level": LEVELS,
        "data_format": "grib" if fmt == "grib" else "netcdf",
        "download_format": "unarchived",
        "area": area,
    }


def make_client():
    import cdsapi
    if not os.path.exists(SECRETS):
        return cdsapi.Client()               # falls back to ~/.cdsapirc / env vars
    cfg = {}
    with open(SECRETS) as f:
        for line in f:
            k, sep, v = line.partition(":")
            if sep and v.strip():
                cfg[k.strip()] = v.strip()
    if "key" not in cfg:
        sys.exit(f"No 'key:' line in {SECRETS}")
    return cdsapi.Client(url=cfg.get("url", "https://cds.climate.copernicus.eu/api"), key=cfg["key"])


def fetch(client, req, target):
    tmp = target + ".part"
    client.retrieve(DATASET, req).download(tmp)
    if zipfile.is_zipfile(tmp):              # CDS sometimes zips multi-stream netCDF
        with zipfile.ZipFile(tmp) as z:
            ncs = [m for m in z.namelist() if m.endswith((".nc", ".grib"))]
            if len(ncs) != 1:
                sys.exit(f"Unexpected zip contents in {tmp}: {z.namelist()}")
            with z.open(ncs[0]) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.remove(tmp)
    else:
        os.replace(tmp, target)


def main():
    a = parse()
    stamps = pick_timestamps(a.start, a.end, a.step_hours)
    variables = C3DIR_VARS if a.c3dir_only else list(VARIABLES)
    gaps = pd.Series(stamps).diff().dropna().value_counts().sort_index()
    print(f"{len(stamps)} timestamps, {stamps[0]} -> {stamps[-1]}, every {a.step_hours} h")
    print("gap sizes:", ", ".join(f"{g} x{c}" for g, c in gaps.items()))
    print(f"{len(variables)} variables x {len(LEVELS)} levels, area N/W/S/E = {a.area}")

    if a.chunk_days is None:
        hours_per_day = len(set(stamps.hour))
        a.chunk_days = max(1, COST_LIMIT // (COST_PER_FIELD * hours_per_day * len(variables) * len(LEVELS)))
    tag = area_tag(a.area)
    plan = []
    for d0, d1 in chunks(a.start, a.end, a.chunk_days):
        sel = stamps[(stamps >= pd.Timestamp(d0)) & (stamps < pd.Timestamp(d1) + pd.Timedelta(days=1))]
        if len(sel):
            target = os.path.join(a.workdir, f"era5_{d0:%Y%m%d}_{d1:%Y%m%d}_{tag}.{a.format}")
            plan.append((d0, d1, sel, target, request_for(d0, d1, sel.hour, variables, a.area, a.format)))
    print(f"{len(plan)} CDS requests of up to {a.chunk_days} days each")
    if a.dry_run:
        for d0, d1, sel, target, req in plan:
            print(f"  {d0}..{d1}: {len(sel)} timestamps, {len(req['time'])} hours/day requested -> {target}")
        return

    import xarray as xr
    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    client = make_client()
    for i, (d0, d1, sel, target, req) in enumerate(plan, 1):
        if os.path.exists(target):
            print(f"[{i}/{len(plan)}] {target} exists, skipping"); continue
        print(f"[{i}/{len(plan)}] requesting {d0}..{d1} ({len(sel)} timestamps)")
        fetch(client, req, target)

    if a.format == "grib":
        print(f"GRIB chunks are in {a.workdir}; the merge below is netCDF-only, so skipping it")
        return

    parts = []
    for d0, d1, sel, target, req in plan:
        ds = xr.open_dataset(target)
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        parts.append(ds.sel({tdim: sel.values}).load())   # exact timestamps only
        ds.close()
    out = xr.concat(parts, dim=tdim, data_vars="all", coords="minimal", compat="override")
    out = out.sortby(tdim).sortby("pressure_level", ascending=False)
    missing = len(stamps) - out.sizes[tdim]
    if missing:
        print(f"WARNING: {missing} requested timestamps not returned (e.g. beyond ERA5's latest date)")
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.attrs["selection"] = f"step={a.step_hours}h, n={len(stamps)}, window={a.start}..{a.end}, area={a.area}"
    out.to_netcdf(a.out, encoding=enc)
    print(f"wrote {a.out}: {dict(out.sizes)}")


if __name__ == "__main__":
    main()