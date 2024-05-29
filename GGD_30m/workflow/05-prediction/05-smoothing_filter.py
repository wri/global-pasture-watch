import skmap_bindings
import SharedArray as sa
from skmap.misc import make_tempfile, make_tempdir, ttprint
from skmap.parallel import job
from scipy.signal import savgol_filter
import numpy as np
import bottleneck as bn
import time
import cv2
import sys
from pathlib import Path
import tempfile
import pandas as pd
import requests
import gc
import os
import savitzkygolay
from scipy.stats import theilslopes
import numexpr as ne
from typing import Callable, Iterator, List,  Union

executor = None

def _urls(tile, p, base_url = 'http://192.168.49.40:8333', years=range(2000,2023,2)):
  return [
    f'{base_url}/tmp-gpw/v20240418_ultimate_prod/{tile}/gpw_rf.{p}_30m_m_{y}0101_{y}1231_go_epsg.4326_v20240206.tif'
    for y in years 
  ]

def smoothing_3d(i0, i1, array_fn, array_fn_slop):
  array = sa.attach(array_fn)
  array[0:23,:,i0:i1] = savitzkygolay.filter3D(array[0:23,:,i0:i1], 5, 3).round()
  array[23:46,:,i0:i1] = savitzkygolay.filter3D(array[23:46,:,i0:i1], 5, 3).round()

  array0 = array[0,:,i0:i1]
  array1 = array[22,:,i0:i1]
  array_slop[0,:,i0:i1] = array1 - array0

  array0 = array[22,:,i0:i1]
  array1 = array[45,:,i0:i1]
  array_slop[1,:,i0:i1] = array1 - array0
  
def normalize_3d(array_prob):
  array_prob[array_prob < 0] = 0
  array_prob_t = array_prob.transpose(2,1,0)
  array_prob_t_r = array_prob_t.reshape(-1,3)
  t_r_sum = bn.nansum(array_prob_t_r,axis=-1).reshape(-1,1)
  return (((array_prob_t_r / t_r_sum).reshape(array_prob_t.shape).transpose(2,1,0)) * 100).round(0)

def _processed(tile):
  url = f'http://192.168.49.30:8333/tmp-gpw/v20240418_ultimate_prod/{tile}/gpw_nat.semi.grassland_rf.savgol_p_30m_20030101_20031231_go_epsg.4326_v1.tif'
  r = requests.head(url)
  return (r.status_code == 200)

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

  #with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
  futures = { executor.submit(worker, *arg) for arg in args }

  done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
  for task in done:
    err = task.exception()
    if err is not None:
      raise err
    else:
        yield task.result()

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])
server_name=sys.argv[3]

TMP_DIR = tempfile.gettempdir()

base_dir = Path('/mnt/slurm/jobs/wri_pasture_class')
ids_fn = str(base_dir.joinpath('gpw_pasture.class_ids.csv'))
tiles_id = pd.read_csv(ids_fn)['TILE'][start_tile:end_tile]
years = range(2000,2023)

subnet = '192.168.49'
hosts = [ f'{subnet}.{i}:8333' for i in range(30,43) ]

gdal_opts = {
   'GDAL_HTTP_VERSION': '1.0',
   'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
  }
n_threads = 96
bands_list = [1,]
x_off = 2
y_off = 2
x_size, y_size = (4000, 4000)
bands_list = [1]
nodata_val = 255.

mask_prefix = 'http://192.168.1.30:8333/gpw/landmask'
umask_prefix = 'http://192.168.1.30:8333/gpw/urbanmask'

#tiles_id = ['055W_28S', '001W_11N','001W_12N','002E_45N','002W_11N','002W_12N','071W_19N','130E_69N','148E_25S','148E_26S','149E_25S','149E_26S']
#tiles_id = ['111E_65N']
for tile_id in tiles_id:

  if _processed(tile_id):
    ttprint(f"Tile {tile_id} already exists.")
    continue

  try:
    mask_file = f'{mask_prefix}/{tile_id}.tif'
    umask_file1 = f'{umask_prefix}/WSFEVO_{tile_id}.tif'
    umask_file2 = f'{umask_prefix}/WSF2019_{tile_id}.tif'

    ttprint(f"Tile {tile_id} - Start")
    files_list = _urls(tile_id, 'seeded.grass', years=years) + _urls(tile_id, 'nat.semi.grass', years=years)

    n_rasters = len(files_list)
    file_order = np.arange(n_rasters)
    shape = ((23 * 3) + 4, x_size * y_size)
    array = np.zeros(shape, dtype=np.float32)

    start = time.time()
    skmap_bindings.readData(array, n_threads, files_list, file_order, x_off, y_off, x_size, y_size, bands_list, gdal_opts)
    array[array == 255] = 0
    
    mask = np.zeros((3, x_size * y_size), dtype=np.float32)
    skmap_bindings.readData(mask, n_threads, [mask_file, ], [0], x_off, y_off, x_size, y_size, [1,], gdal_opts)
    skmap_bindings.readData(mask, n_threads, [umask_file1, umask_file2, ], [1,2], 0, 0, x_size, y_size, [1,], gdal_opts)

    ttprint(f"Tile {tile_id} - Reading probabilities: {(time.time() - start):.2f} segs")

    start = time.time()
    array_fn_prob = 'file://' + str(make_tempfile(prefix='shm_array'))
    array_prob = sa.create(array_fn_prob, (shape[0], x_size, y_size), dtype=np.float32)
    array_prob[0:23,:,:] = array[0:23].reshape(23, x_size, y_size)
    array_prob[23:46,:,:] = array[23:46].reshape(23, x_size, y_size)

    array_fn_slop = 'file://' + str(make_tempfile(prefix='shm_array'))
    array_slop = sa.create(array_fn_slop, (2, x_size, y_size), dtype=np.float32)

    start = time.time()
    n_pixels = array_prob.shape[-1]
    n_threads = 96
    batch = int(n_pixels / n_threads)
    args = []
    for i0 in range(0, n_pixels, batch):
      i1 = n_pixels if (i0 + batch) > n_pixels else (i0 + batch)
      #print((i0, i1, array_fn_prob))
      args.append((i0, i1, array_fn_prob, array_fn_slop))
        
    #for r in job(smoothing_3d, args,  n_jobs=n_threads, joblib_args={'backend': 'multiprocessing'}):
    #    continue
    for result in ProcessGeneratorLazy(smoothing_3d, args, len(args)):
      continue
    ttprint(f"Tile {tile_id} - Smoothing probabilities: {(time.time() - start):.2f} segs")
    
    start = time.time()
    array_prob[array_prob < 0] = 0
    array_prob[array_prob > 100] = 100
    ttprint(f"Tile {tile_id} - Cliping probabilities: {(time.time() - start):.2f} segs")

    array_slop = (array_slop + 101)
    #array_slop[np.isnan(array_slop)] = 255

    start = time.time()
    array_sg = array_prob[0:23,:,:].reshape(23, x_size * y_size)
    #array[69,:] = bn.nanmean(np.absolute(array[0:23,:] - array_sg), axis=0)
    array[69,:] = np.mean(ne.evaluate('abs(raw - sg)', local_dict={'raw':array[0:23,:] , 'sg':array_sg}), axis=0)
    array[0:23,:]  = array_sg

    array_sg = array_prob[23:46,:,:].reshape(23, x_size * y_size)
    #array[70,:] = bn.nanmean(np.absolute(array[23:46,:] - array_sg), axis=0)
    array[70,:] = np.mean(ne.evaluate('abs(raw - sg)', local_dict={'raw':array[23:46,:] , 'sg':array_sg}), axis=0)
    array[23:46,:]  = array_prob[23:46,:,:].reshape(23, x_size * y_size)
    ttprint(f"Tile {tile_id} - Computing madi: {(time.time() - start):.2f} segs")

    start = time.time()
    see_mask = (array[0:23,:] >= 32)
    nat_mask = (array[23:46,:] >= 42)

    array[46:69,:][see_mask] = 1
    array[46:69,:][nat_mask] = 2
    ttprint(f"Tile {tile_id} - Computing hard classes: {(time.time() - start):.2f} segs")

    start = time.time()
    array[71,:] = array_slop[0,:,:].reshape(x_size * y_size)
    array[72,:] = array_slop[1,:,:].reshape(x_size * y_size)
    ttprint(f"Tile {tile_id} - Transposing data: {(time.time() - start):.2f} segs")
    
    start = time.time()
    mask = np.logical_or.reduce([mask[0,:] == 0, mask[1,:] >= 1985, mask[2,:] == 255])
    mask_exp = np.repeat(mask[np.newaxis, :], array.shape[0], axis=0)
    array[mask_exp] = 255
    ttprint(f"Tile {tile_id} - Aplying mask: {(time.time() - start):.2f} segs")
    
    write_idx = list(range(0, array.shape[0]))
    tmp_dir = str(make_tempdir(tile_id))
    base_raster = files_list[-1]
    s3_prefix = 'tmp-gpw/v20240418_ultimate_prod'

    out_files = [ f'gpw_cultiv.grassland_rf.savgol_p_30m_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
             [ f'gpw_nat.semi.grassland_rf.savgol_p_30m_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
             [ f'gpw_grassland_rf.savgol.bthr_c_30m_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
             [ 
                f'gpw_cultiv.grassland_rf.savgol.madi_p_30m_20000101_20221231_go_epsg.4326_v1',
                f'gpw_nat.semi.grassland_rf.savgol.madi_p_30m_20000101_20221231_go_epsg.4326_v1'
             ] + \
             [ 
                f'gpw_cultiv.grassland_rf.savgol.diff_p_30m_20000101_20221231_go_epsg.4326_v1',
                f'gpw_nat.semi.grassland_rf.savgol.diff_p_30m_20000101_20221231_go_epsg.4326_v1'
             ]
             #[ f'gpw_eml.grass.type_30m_m_{year}0101_{year}1231_go_epsg.4326_v20240203' for year in years ]
    
    #out_s3 = [ f'gaia/{s3_prefix}/{tile_id}' for o in out_files ]
    out_s3 = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % len(hosts)}/{s3_prefix}/{tile_id}" for o in out_files ]

    x_off_d, y_off_d = (x_off, y_off)

    nodata_val = 255
    compression_command = f"gdal_translate -a_nodata {nodata_val} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
    base_raster = [ base_raster for o in out_files ]
      
    skmap_bindings.writeByteData(array, n_threads, gdal_opts, base_raster, tmp_dir, out_files, write_idx,
        x_off_d, y_off_d, x_size, y_size, nodata_val, compression_command, out_s3)
    ttprint(f"Tile {tile_id} - Result available in gaia {out_s3[-1]}")

    os.remove(array_fn_prob.replace("file://",""))
    del array
    gc.collect()

    ttprint(f"Tile {tile_id} - End")
  except:
    ttprint(f"Tile {tile_id} - Error")