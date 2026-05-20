import os
os.environ["PROJ_LIB"] = "/opt/conda/share/proj"

import random
import numpy as np
import boto3
from osgeo import gdal, osr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')

# ── config ────────────────────────────────────────────────────────────────────

ANIMALS       = ["horse", "cattle", "sheep", "goat", "buffalo"]

YEARS         = range(2000, 2023)
NODATA_OUT    = -9999
OUTPUT_RES    = 0.008333333333
RASTER_WIDTH  = int(360 / OUTPUT_RES)
RASTER_HEIGHT = int(180 / OUTPUT_RES)
N_WORKERS     = 4

OUT_DIR         = Path(".")
MASK_4326_PATH  = OUT_DIR / "country_mask_4326.tif"
MASK_54052_PATH = OUT_DIR / "country_mask_54052.tif"

SRS_4326     = osr.SpatialReference()
SRS_4326.ImportFromEPSG(4326)
SRS_4326_WKT = SRS_4326.ExportToWkt()
GT_4326      = (-180.0, OUTPUT_RES, 0.0, 90.0, 0.0, -OUTPUT_RES)

def make_s3():
    addr = random.choice(GAIA_S3_PARAMS['s3_addresses']).replace('http://', '')
    return boto3.client(
        's3',
        endpoint_url=f'http://{addr}',
        aws_access_key_id=GAIA_S3_PARAMS['s3_access_key'],
        aws_secret_access_key=GAIA_S3_PARAMS['s3_secret_key'],
    )


def s3_key(animal, year):
    fname = f"gpw_{animal}.headcount.faostat_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1.tif"
    return f"{GAIA_S3_PARAMS['s3_prefix']}/{fname}"


def s3_exists(key):
    try:
        make_s3().head_object(Bucket=GAIA_S3_PARAMS['s3_bucket'], Key=key)
        return True
    except Exception:
        return False

# ── functions ─────────────────────────────────────────────────────────────────

def build_faostat_url(animal, year):
    host  = "http://192.168.49.30:8333"
    fname = f"gpw_{animal}.headcount.faostat_rf_m_1km_s_{year}0101_{year}1231_go_esri.54052_v1.tif"
    return f"/vsicurl/{host}/gpw/arco/{fname}"


def load_mask(path):
    ds  = gdal.Open(str(path), gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    print(f"  Loaded {path.name}  (countries: {int(arr.max())})")
    return arr


def index_country_pixels(mask, n):
    """Pre-index pixel locations per country — avoids 198x full-raster scans per year."""
    flat   = mask.ravel()
    nz     = np.where(flat > 0)[0].astype("int32")
    ids    = flat[nz].astype("int32")
    order  = np.argsort(ids, kind="stable")
    nz_s   = nz[order]
    ids_s  = ids[order]
    bounds = np.searchsorted(ids_s, np.arange(0, n + 2))
    return [nz_s[bounds[c]: bounds[c + 1]] for c in range(1, n + 1)]


def get_src_nodata(ds):
    nd = ds.GetRasterBand(1).GetNoDataValue()
    if nd is None:
        raise ValueError("Source file has no nodata value set.")
    return float(nd)


def warp_to_4326(src_ds, src_nodata):
    return gdal.Warp(
        "", src_ds, format="MEM",
        srcSRS=src_ds.GetProjection(),
        dstSRS=SRS_4326_WKT,
        xRes=OUTPUT_RES, yRes=OUTPUT_RES,
        outputBounds=(-180, -90, 180, 90),
        outputBoundsSRS=SRS_4326_WKT,
        resampleAlg=gdal.GRA_Bilinear,
        srcNodata=src_nodata,
        dstNodata=NODATA_OUT,
        outputType=gdal.GDT_Float32,
        warpOptions=["NUM_THREADS=ALL_CPUS"],
        multithread=True,
    )


def largest_remainder(vals, target):
    """Distribute `target` integer units across pixels proportional to `vals`."""
    n = len(vals)
    if n == 0 or target <= 0:
        return np.zeros(n, dtype="int32")

    total = float(vals.sum())
    if total <= 0:
        base   = target // n
        result = np.full(n, base, dtype="int64")
        result[np.random.choice(n, target % n, replace=False)] += 1
        return result.astype("int32")

    scaled  = vals.astype("float64") * (target / total)
    floored = np.floor(scaled).astype("int64")
    deficit = target - int(floored.sum())
    fracs   = scaled - floored

    if deficit > 0:
        boundary = np.partition(fracs, n - deficit)[n - deficit]
        above    = np.where(fracs > boundary)[0]
        floored[above] += 1
        need = deficit - len(above)
        if need > 0:
            tied = np.where(fracs == boundary)[0]
            floored[np.random.choice(tied, need, replace=False)] += 1

    elif deficit < 0:
        overshoot = -deficit
        boundary  = np.partition(fracs, overshoot - 1)[overshoot - 1]
        below     = np.where(fracs < boundary)[0]
        floored[below] -= 1
        need = overshoot - len(below)
        if need > 0:
            tied = np.where(fracs == boundary)[0]
            floored[np.random.choice(tied, need, replace=False)] -= 1
        floored = np.maximum(floored, 0)

    return floored.astype("int32")


def save_tif_to_s3(arr, animal, year, key):
    """Write int32 COG to vsimem then upload to Gaia S3."""
    vsimem = f"/vsimem/{animal}_{year}.tif"

    mem_ds = gdal.GetDriverByName("MEM").Create(
        "", RASTER_WIDTH, RASTER_HEIGHT, 1, gdal.GDT_Int32)
    mem_ds.SetGeoTransform(GT_4326)
    mem_ds.SetProjection(SRS_4326_WKT)
    mem_ds.GetRasterBand(1).SetNoDataValue(NODATA_OUT)
    mem_ds.GetRasterBand(1).WriteArray(arr)
    gdal.Translate(
        vsimem, mem_ds,
        format="COG",
        creationOptions=[
            "COMPRESS=ZSTD",
            "PREDICTOR=2",
            "BLOCKSIZE=512",
            "OVERVIEW_RESAMPLING=NEAREST",
            "BIGTIFF=YES",
            "NUM_THREADS=ALL_CPUS",
        ],
    )
    mem_ds = None

    # read COG bytes from vsimem
    vsi_f = gdal.VSIFOpenL(vsimem, "rb")
    size  = gdal.VSIStatL(vsimem).size
    data  = gdal.VSIFReadL(1, size, vsi_f)
    gdal.VSIFCloseL(vsi_f)
    gdal.Unlink(vsimem)

    make_s3().put_object(
        Bucket=GAIA_S3_PARAMS['s3_bucket'],
        Key=key,
        Body=data,
    )


def process_year(animal, year, px_4326, px_54052, n):
    key = s3_key(animal, year)

    if s3_exists(key):
        print(f"  [{animal} {year}] already exists on S3, skipping.")
        return animal, year, True

    try:
        fao_ds = gdal.Open(build_faostat_url(animal, year), gdal.GA_ReadOnly)
    except RuntimeError as e:
        print(f"  [{animal} {year}] SKIPPED — {e}")
        return animal, year, False

    src_nodata  = get_src_nodata(fao_ds)
    arr_native  = fao_ds.GetRasterBand(1).ReadAsArray().astype("float64")

    wgs84_ds = warp_to_4326(fao_ds, src_nodata)
    if wgs84_ds is None:
        print(f"  [{animal} {year}] SKIPPED — warp failed")
        return animal, year, False

    arr_4326  = wgs84_ds.GetRasterBand(1).ReadAsArray().astype("float64")
    land_mask = arr_4326 != float(NODATA_OUT)
    arr_4326  = np.where(land_mask, np.maximum(arr_4326, 0.0), float(NODATA_OUT))

    out       = np.full((RASTER_HEIGHT, RASTER_WIDTH), NODATA_OUT, dtype="int32")
    out[land_mask] = 0

    flat_native = arr_native.ravel()
    flat_4326   = arr_4326.ravel()
    flat_out    = out.ravel()
    total_in    = 0
    total_out   = 0

    for cid in range(1, n + 1):
        px54    = px_54052[cid - 1]
        vals54  = flat_native[px54]
        valid54 = vals54 != src_nodata
        target  = int(round(vals54[valid54].sum()))
        total_in += target

        if target <= 0:
            continue

        px4  = px_4326[cid - 1]
        vals = flat_4326[px4]
        pos  = vals > 0
        if not pos.any():
            continue

        result = largest_remainder(vals[pos], target)
        flat_out[px4[pos]] = result
        total_out += int(result.sum())

    save_tif_to_s3(out, animal, year, key)
    print(f"  [{animal} {year}] native={total_in:,.0f}  out={total_out:,.0f}  -> s3://{GAIA_S3_PARAMS['s3_bucket']}/{key}")
    return animal, year, True

# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading country masks...")
    mask_4326   = load_mask(MASK_4326_PATH)
    mask_54052  = load_mask(MASK_54052_PATH)
    n_countries = int(mask_4326.max())

    print("Indexing country pixels (once)...")
    px_4326  = index_country_pixels(mask_4326,  n_countries)
    px_54052 = index_country_pixels(mask_54052, n_countries)

    total_tasks = len(ANIMALS) * len(YEARS)
    print(f"\nProcessing {len(ANIMALS)} animals × {len(YEARS)} years = {total_tasks} tasks with {N_WORKERS} workers...")
    print(f"Animals: {', '.join(ANIMALS)}")
    print(f"Years: {min(YEARS)}-{max(YEARS)}\n")
    
    failed_tasks = []
    completed_tasks = 0
    
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {}
        for animal in ANIMALS:
            for year in YEARS:
                futures[executor.submit(process_year, animal, year, px_4326, px_54052, n_countries)] = (animal, year)
        
        for f in as_completed(futures):
            animal, year, ok = f.result()
            completed_tasks += 1
            if not ok:
                failed_tasks.append((animal, year))

    print(f"\n{'='*60}")
    print(f"Completed: {completed_tasks}/{total_tasks} tasks")
    if failed_tasks:
        print(f"\n  Failed tasks ({len(failed_tasks)}):")
        for animal, year in failed_tasks:
            print(f"   - {animal} {year}")
    else:
        print("\n✓ All tasks completed successfully!")
    print(f"{'='*60}\n")