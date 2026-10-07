"""
Parallel version of pull_era5.py: same data, same layout, same chunk files, but the
CDS requests are submitted concurrently (--workers at a time) instead of one by one.

Pull one month of hourly ERA5 pressure-level data around Boston during the 2026 World Cup month
(June 2026 by default), in the same layout as era5_subset.nc
(valid_time x pressure_level x latitude x longitude, 16 variables, 15 levels).

Default domain: 6 x 6 degrees centered on Boston (~42.4N, 71.1W), 25 x 25 grid points.

Setup (once):
    uv sync   (cdsapi, xarray, netCDF4 are project dependencies)
    Put your key from https://cds.climate.copernicus.eu/profile in era5_secrets.txt
    (gitignored; same format as ~/.cdsapirc, which is used if the file is absent):
        url: https://cds.climate.copernicus.eu/api
        key: <your-personal-access-token>
    Accept the ERA5 licence on the dataset page once (the API refuses otherwise).

Parallelism: each worker thread has its own cdsapi client and submits one chunk
at a time. CDS caps how many requests per user run at once (extra ones just wait
in its queue), so going far above ~4-8 workers mostly adds queued requests rather
than speed. Failed chunks are retried --retries times, then reported; the merge
only runs if every chunk is on disk. Chunk files are interchangeable with
pull_era5.py, so either script can resume the other's partial pull.

Examples:
    python pull_era5_parallel.py                          # hourly, June 2026, 4 workers
    python pull_era5_parallel.py --workers 8
    python pull_era5_parallel.py --step-hours 3 --c3dir-only
    python pull_era5_parallel.py --dry-run                # print the plan, download nothing
"""
import argparse, os, sys, zipfile, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
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

_print_lock = threading.Lock()
_local = threading.local()


def log(*args):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), *args, flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2026-06-01", help="first day (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-06-30", help="last day, inclusive")
    p.add_argument("--step-hours", type=int, default=1,
                   help="fixed spacing between timestamps; a divisor of 24 keeps the same hours each day")
    p.add_argument("--area", type=float, nargs=4, metavar=("N", "W", "S", "E"),
                   default=BOSTON_AREA, help="bounding box; default = 6x6 deg centered on Boston")
    p.add_argument("--c3dir-only", action="store_true",
                   help="only the 8 C3DIR-related variables (smaller download)")
    p.add_argument("--chunk-days", type=int, default=None,
                   help="days per CDS request; default = most that fit under the CDS cost limit")
    p.add_argument("--workers", type=int, default=4, help="concurrent CDS requests (default 4)")
    p.add_argument("--retries", type=int, default=2, help="extra attempts per failed chunk (default 2)")
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


def read_secrets():
    """Returns (url, key) from era5_secrets.txt, or None to let cdsapi use ~/.cdsapirc / env vars."""
    if not os.path.exists(SECRETS):
        return None
    cfg = {}
    with open(SECRETS) as f:
        for line in f:
            k, sep, v = line.partition(":")
            if sep and v.strip():
                cfg[k.strip()] = v.strip()
    if "key" not in cfg:
        sys.exit(f"No 'key:' line in {SECRETS}")
    return cfg.get("url", "https://cds.climate.copernicus.eu/api"), cfg["key"]


def thread_client(creds):
    """One cdsapi client per worker thread (the client holds a requests session)."""
    if getattr(_local, "client", None) is None:
        import cdsapi
        kw = dict(quiet=True, progress=False)   # per-thread progress bars would interleave
        _local.client = cdsapi.Client(**kw) if creds is None else cdsapi.Client(url=creds[0], key=creds[1], **kw)
    return _local.client


def fetch(client, req, target):
    tmp = target + ".part"
    client.retrieve(DATASET, req).download(tmp)
    if zipfile.is_zipfile(tmp):              # CDS sometimes zips multi-stream netCDF
        with zipfile.ZipFile(tmp) as z:
            ncs = [m for m in z.namelist() if m.endswith((".nc", ".grib"))]
            if len(ncs) != 1:
                raise RuntimeError(f"Unexpected zip contents in {tmp}: {z.namelist()}")
            with z.open(ncs[0]) as src, open(target + ".unzip", "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.replace(target + ".unzip", target)
        os.remove(tmp)
    else:
        os.replace(tmp, target)


def worker(job, creds, retries):
    i, n, d0, d1, sel, target, req = job
    for attempt in range(retries + 1):
        try:
            log(f"[{i}/{n}] submitting {d0}..{d1} ({len(sel)} timestamps)"
                + (f", attempt {attempt + 1}" if attempt else ""))
            t0 = time.time()
            fetch(thread_client(creds), req, target)
            log(f"[{i}/{n}] done {d0}..{d1} in {(time.time() - t0) / 60:.1f} min -> {target}")
            return
        except Exception as e:
            log(f"[{i}/{n}] {d0}..{d1} failed: {type(e).__name__}: {e}")
            _local.client = None             # fresh session for the retry
            if attempt < retries:
                time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"{d0}..{d1} failed after {retries + 1} attempts")


def main():
    a = parse()
    if a.workers < 1:
        sys.exit("--workers must be at least 1")
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
    todo = [(i, len(plan), *p) for i, p in enumerate(plan, 1) if not os.path.exists(p[3])]
    print(f"{len(plan)} CDS requests of up to {a.chunk_days} days each; "
          f"{len(plan) - len(todo)} already on disk, {len(todo)} to fetch with {a.workers} workers")
    if a.dry_run:
        for d0, d1, sel, target, req in plan:
            state = "on disk" if os.path.exists(target) else "to fetch"
            print(f"  {d0}..{d1}: {len(sel)} timestamps, {len(req['time'])} hours/day -> {target} [{state}]")
        return

    os.makedirs(a.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    creds = read_secrets()

    failed = []
    if todo:
        t0 = time.time()
        pool = ThreadPoolExecutor(max_workers=min(a.workers, len(todo)))
        futures = {pool.submit(worker, job, creds, a.retries): job for job in todo}
        try:
            for done, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    failed.append((job[2], job[3], e))
                log(f"progress: {done}/{len(todo)} finished, {len(failed)} failed, "
                    f"{(time.time() - t0) / 60:.1f} min elapsed")
        except KeyboardInterrupt:
            log("interrupted: cancelling queued chunks (in-flight ones finish or are dropped); rerun to resume")
            pool.shutdown(wait=False, cancel_futures=True)
            os._exit(130)
        pool.shutdown()

    if failed:
        print(f"\n{len(failed)} chunk(s) failed; rerun to retry just these:")
        for d0, d1, e in sorted(failed, key=lambda x: x[0]):
            print(f"  {d0}..{d1}: {e}")
        sys.exit(1)

    if a.format == "grib":
        print(f"GRIB chunks are in {a.workdir}; the merge below is netCDF-only, so skipping it")
        return

    import xarray as xr
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