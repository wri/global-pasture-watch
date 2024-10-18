import geopandas as gpd
from typing import Callable, Iterator, List,  Union
from osgeo import gdal, gdal_array
import numpy as np
import bottleneck as bn
import numexpr as ne
from pathlib import Path
import pandas as pd

executor = None

gdal_opts = {
 'GDAL_HTTP_VERSION': '1.0',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
}

def ttprint(*args, **kwargs):
  from datetime import datetime
  import sys

  print(f'[{datetime.now():%H:%M:%S}] ', end='')
  print(*args, **kwargs, flush=True)

def ProcessGeneratorLazy(
  worker:Callable,
  args:Iterator[tuple],
  max_workers:int = None
):
  import concurrent.futures
  import multiprocessing
  from itertools import islice

  global executor

  if executor is None:
    max_workers = multiprocessing.cpu_count()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

  futures = { executor.submit(worker, **arg) for arg in args }

  done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
  for task in done:
    err = task.exception()
    if err is not None:
      raise err
    else:
        yield task.result()

def read_raster(raster_file, array, i, band=1, minx = None, maxy = None):
    
  for key in gdal_opts.keys():
      gdal.SetConfigOption(key,gdal_opts[key])
  
  ds = gdal.Open(raster_file)

  xoff, yoff = 0, 0
  win_xsize, win_ysize = None, None
  if minx is not None and maxy is not None:
    gt = ds.GetGeoTransform()
    gti = gdal.InvGeoTransform(gt)

    xoff, yoff = gdal.ApplyGeoTransform(gti, minx, maxy)
    xoff, yoff = int(xoff), int(yoff)
    win_xsize, win_ysize = array.shape[0], array.shape[1]
  
  band = ds.GetRasterBand(1)
  nodata = band.GetNoDataValue()
  
  gdal_array.BandReadAsArray(band, buf_obj=array[:,:,i],
          xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)

  array[:,:,i][array[:,:,i] == nodata] = np.nan

def save_raster(base_raster, out_file, mask, i, minx = None, maxy = None, co = []):
    
  base_ds = gdal.Open(base_raster)

  driver = gdal.GetDriverByName('GTiff')

  out_ds = driver.CreateCopy(out_file, base_ds, options=co)
  out_band = out_ds.GetRasterBand(1)

  xoff, yoff = 0, 0
  if minx is not None and maxy is not None:
    gt = base_ds.GetGeoTransform()
    gti = gdal.InvGeoTransform(gt)

    xoff, yoff = gdal.ApplyGeoTransform(gti, minx, maxy)
    xoff, yoff = round(xoff), round(yoff)
  
  gdal_array.BandWriteArray(out_band, np.zeros((4004,4004)))
  gdal_array.BandWriteArray(out_band, mask, xoff=xoff, yoff=yoff)

def read_rasters(raster_files, idx_list, array, minx = None, maxy = None, gdal_opts = []):
  n_files = len(raster_files)

  ttprint(f"Reading {len(raster_files)} raster files.")
  
  args = []
  for raster_file, i in zip(raster_files, idx_list):
    args.append({
      'raster_file': raster_file, 
      'array': array, 
      'i': i,
      'minx': minx,
      'maxy': maxy
    })

  for arg in args:
    read_raster(**arg)

def landmask(raster_files, tile, minx, maxy, mask_vals, outdir):

  n_raster = len(raster_files)
  array = np.empty((4000,4000,n_raster))

  read_rasters(raster_files, range(0,n_raster), array, minx, maxy, gdal_opts)

  values = sum([ list(range(v1,v2+1)) for v1,v2 in mask_vals ],[])
  mask_cond = ' | '.join([ f'({v} == array)' for v in values ])
  expression = f"where({mask_cond}, 0, 1)"

  mask = (np.nansum(ne.evaluate(expression), axis=-1) >= 1).astype('int')

  base_raster = f'http://192.168.49.30:8333/prod-landsat-ard2/{tile}/seasconv/swir2_glad.SeasConv.ard2_m_30m_s_20220101_20220228_go_epsg.4326_v20230908.tif'
  out_file = str(Path(outdir).joinpath(f'{tile}.tif'))
  co = ['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE', 'BLOCKXSIZE=1024', 'BLOCKYSIZE=1024']
  save_raster(base_raster, out_file, mask, 0, minx, maxy, co)

  return tile, np.nanmax(mask)

outdir = 'predmask'
mask_vals = [[254,255],[197,239]]

raster_files = [
    '/mnt/inca/GPW/vrts/lc_glad.glcluc_c_30m_s_20000101_20001231_go_epsg.4326_v20230901.vrt',
    '/mnt/inca/GPW/vrts/lc_glad.glcluc_c_30m_s_20050101_20051231_go_epsg.4326_v20230901.vrt',
    '/mnt/inca/GPW/vrts/lc_glad.glcluc_c_30m_s_20100101_20101231_go_epsg.4326_v20230901.vrt',
    '/mnt/inca/GPW/vrts/lc_glad.glcluc_c_30m_s_20150101_20151231_go_epsg.4326_v20230901.vrt',
    '/mnt/inca/GPW/vrts/lc_glad.glcluc_c_30m_s_20200101_20201231_go_epsg.4326_v20230901.vrt'
]
missing_tiles = pd.read_csv('/mnt/slurm/jobs/wri_pasture_class/gpw_pasture.class_ids.csv')['TILE']

tiles = gpd.read_file('ard2_final_status.gpkg')
tiles = tiles[tiles['TILE'].isin(missing_tiles)]

args = []
for _,row in tiles.iterrows():
    minx, miny, maxx, maxy = row.geometry.bounds
    args.append({
        'raster_files': raster_files,
        'tile': row['TILE'],
        'minx': minx,
        'maxy': maxy,
        'mask_vals': mask_vals,
        'outdir': outdir
    })

Path(outdir).mkdir(parents=True, exist_ok=True)
for tile, max_val in ProcessGeneratorLazy(landmask, args, max_workers=96):
  print(f"TILE {tile} max {max_val}")
  continue