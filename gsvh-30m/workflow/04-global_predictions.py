import os
import shutil
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'

from datetime import datetime
from hummingbird.ml import load
from itertools import islice
from joblib import Parallel, delayed
from multiprocessing import Pool
from osgeo import gdal, gdal_array
from pathlib import Path
from skmap.misc import find_files
from skmap.parallel import job
from typing import Callable, Iterator, List,  Union

import bottleneck as bn
import concurrent.futures
import gc
import geopandas as gpd
import joblib
import lleaves
import multiprocessing
import numexpr as ne
import numpy as np
import pandas as pd
import requests
import SharedArray as sa
import skmap_bindings
import sys
import tempfile
import time
import traceback
import treelite_runtime

gdal_opts = {
 #'GDAL_HTTP_MULTIRANGE': 'SINGLE_GET',
 #'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'NO',
 'GDAL_HTTP_VERSION': '1.0',
 #'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
 #'VSI_CACHE': 'FALSE',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
 #'GDAL_HTTP_CONNECTTIMEOUT': '320',
 #'CPL_VSIL_CURL_USE_HEAD': 'NO',
 #'GDAL_HTTP_TIMEOUT': '320',
 #'CPL_CURL_GZIP': 'NO'
}

co = ['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE', 'BLOCKXSIZE=1024', 'BLOCKYSIZE=1024']

executor = None

def ttprint(*args, **kwargs):
  from datetime import datetime
  import sys

  print(f'[{datetime.now():%H:%M:%S}] ', end='')
  print(*args, **kwargs, flush=True)

def make_tempdir(basedir='skmap', make_subdir = True):
  tempdir = Path(TMP_DIR).joinpath(basedir)
  if make_subdir: 
    name = Path(tempfile.NamedTemporaryFile().name).name
    tempdir = tempdir.joinpath(name)
  tempdir.mkdir(parents=True, exist_ok=True)
  return tempdir

def make_tempfile(basedir='skmap', prefix='', suffix='', make_subdir = False):
  tempdir = make_tempdir(basedir, make_subdir=make_subdir)
  return tempdir.joinpath(
    Path(tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix).name).name
  )

def _features(csv_file, years, tile_id, rfe):
  
  df_features = pd.read_csv(csv_file,index_col=0)

  df_list = []
  df_list += [ df_features[(df_features['type'] == 'static')] ]

  for year in years:
    mask = (df_features['type'] == 'landsat')
    df = df_features[mask].copy()
    df['path'] = df['path'].apply(lambda p: p.replace('{tile}', tile_id).replace('{year}', str(year)))
    df_list += [ df ]

  otf_mask = df_features['type'] == 'on-the-fly'
  otf_sel = df_features[otf_mask]['name'].apply(lambda f: '_'.join(f.split('_')[0:2])).unique()
  for year in years:
    df_list += [ df_features[np.logical_and(otf_mask, df_features['name'].str.contains('|'.join(otf_sel)))] ]

  df_features = pd.concat(df_list)
  df_features = df_features.sort_values(['idx', 'path']).reset_index(drop=True)
  df_features['idx'] = df_features.index

  matrix_idx = []

  for c in rfe:
    sel_mask = df_features['name'] == c
    idx = list(df_features[sel_mask]['idx'])
    if len(idx) == 1:
      idx = [ idx[0] for i in range(0,len(years)) ]
    #print(c, idx)
    matrix_idx.append(idx)

  matrix_idx = np.array(matrix_idx)
  
  return df_features, matrix_idx


  mask = (df_features['type'] == ftype)
  ids_list = list(df_features[mask]['idx'])
  raster_files = list(df_features[mask]['path'])

  return raster_files, ids_list

def _raster_paths(df_features, ftype):

  mask = (df_features['type'] == ftype)
  ids_list = list(df_features[mask]['idx'])
  raster_files = list(df_features[mask]['path'])

  return raster_files, ids_list

def _get_static_layers_info(df_features, tiles, tile):
  
  min_x, _, _, max_y = tiles[tiles['TILE'] == tile].iloc[0].geometry.bounds
  static_files, static_idx = _raster_paths(df_features, 'static')
  
  gidal_ds = gdal.Open(static_files[0]) # It is assumed to be the same for all static layers
  gt = gidal_ds.GetGeoTransform()
  gti = gdal.InvGeoTransform(gt)
  x_off_s, y_off_s = gdal.ApplyGeoTransform(gti, min_x, max_y)
  x_off_s, y_off_s = int(x_off_s), int(y_off_s)
  
  return static_files, static_idx, x_off_s, y_off_s

def _geom_temperature(df_features, array, n_threads):

  elev_idx = list(df_features[df_features['name'].str.contains('dtm.bareearth_ensemble')].index)
  lst_min_geo_idx = list(df_features[df_features['name'].str.contains('clm_lst_min.geom.temp')].index)
  lst_max_geo_idx = list(df_features[df_features['name'].str.contains('clm_lst_max.geom.temp')].index)

  x_off, y_off = (0, 0)
  base_landsat = landsat_files[-1]
  lon_lat = np.zeros((2, array.shape[1]), dtype=np.float32)
  skmap_bindings.getLatLonArray(lon_lat, n_threads, gdal_opts, base_landsat, x_off, y_off, x_size, y_size)
  latitude = lon_lat[1,:].copy()

  doys = [ datetime.strptime(f'2000-{m}-15', '%Y-%m-%d').timetuple().tm_yday for m in range(1,13) ]
  doys_all = sum([ doys for i in range(0, len(years)) ],[])

  elevation = array[elev_idx[0],:]

  #skmap_bindings.computeGeometricTemperature(array, n_threads, latitude, elevation, 0.1, 24.16453, -15.71751, 100., lst_min_geo_idx, doys_all)
  skmap_bindings.computeGeometricTemperature(array, n_threads, latitude, elevation, 0.1, 37.03043, -15.43029, 100., lst_max_geo_idx, doys_all)

def in_mem_calc(data, df_features, n_threads):
    
  band_scaling = 0.004
  result_scaling = 125.
  result_offset = 125.

  blue_idx = list(df_features[df_features['name'].str.contains('blue_glad')].index)
  red_idx = list(df_features[df_features['name'].str.contains('red_glad')].index)
  nir_idx = list(df_features[df_features['name'].str.contains('nir_glad')].index)

  swir1_idx = list(df_features[df_features['name'].str.contains('swir1_glad')].index)
  swir2_idx = list(df_features[df_features['name'].str.contains('swir2_glad')].index)
  bsf_idx = list(df_features[df_features['name'].str.contains('bsf')].index)

  ndvi_idx = list(df_features[df_features['name'].str.contains('ndvi_glad')].index)
  ndwi_idx = list(df_features[df_features['name'].str.contains('ndwi_glad')].index)
  bsi_idx = list(df_features[df_features['name'].str.contains('bsi_glad')].index)
  ndti_idx = list(df_features[df_features['name'].str.contains('ndti_glad')].index)
  nirv_idx = list(df_features[df_features['name'].str.contains('nirv_glad')].index)
  evi_idx = list(df_features[df_features['name'].str.contains('evi_glad')].index)
  fapar_idx = list(df_features[df_features['name'].str.contains('fapar_glad')].index)

  # NDVI
  #skmap_bindings.computeNormalizedDifference(data, n_threads,
  #                          nir_idx, red_idx, ndvi_idx,
  #                          band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # NDWI
  #skmap_bindings.computeNormalizedDifference(data, n_threads,
  #                          nir_idx, swir1_idx, ndwi_idx,
  #                          band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # BSI
  #skmap_bindings.computeBsi(data, n_threads,
  #                          swir1_idx, red_idx, nir_idx, blue_idx, bsi_idx,
  #                          band_scaling, band_scaling, band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # NDTI
  #skmap_bindings.computeNormalizedDifference(data, n_threads,
  #                          swir1_idx, swir2_idx, ndti_idx,
  #                          band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])

  # NIRV
  #expr = '( ( ( ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) ) ) - 0.08) *  (nir * 0.004) ) * 125 + 125'
  #data[nir_idx,:] = ne.evaluate(expr, local_dict={ 'red':data[red_idx,:], 'nir':data[nir_idx,:] }).round()
  #skmap_bindings.computeNirv(data, n_threads,
  #                          nir_idx, red_idx, nirv_idx,
  #                          band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  # EVI
  #expr = '( 2.5 * ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + 6 * (red * 0.004) - 7.5 * (blue * 0.004) + 1) ) * 125 + 125'
  #data[evi_idx,:] = ne.evaluate(expr, local_dict={ 'red':data[red_idx,:], 'nir':data[nir_idx,:], 'blue': data[blue_idx,:]  }).round()
  skmap_bindings.computeEvi(data, n_threads,
                            red_idx, nir_idx, blue_idx, evi_idx,
                            band_scaling, band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])

  # FAPAR
  #skmap_bindings.computeFapar(data, n_threads,
  #                          red_idx, nir_idx, fapar_idx,
  #                          band_scaling, band_scaling, result_scaling, result_offset, [0., 250.])
  
  _geom_temperature(df_features, array, n_threads)

def _processed(s3_prefix, tile):
  url = f'http://192.168.49.30:8333/{s3_prefix}{tile}/gpw_short.veg.height_lgb_m_30m_s_20200101_20201231_go_epsg.4326_v20240809.tif'
  r = requests.head(url)
  return (r.status_code == 200)

start_tile=int(sys.argv[1])
end_tile=int(sys.argv[2])
server_name=sys.argv[3]

TMP_DIR = tempfile.gettempdir()

tiles_fn = '/mnt/slurm/jobs/wri_shv/ard2_final_status.gpkg' 
model_dir = f'/mnt/{server_name}/wri_svh/models'
features_fn = f'{model_dir}/features_all.csv'

years = range(2000,2022 + 1)
x_size, y_size = (4004, 4004)
n_threads = 96
n_classes = 3
s3_prefix = 'tmp-gpw/svh_prod_20250201/'

mask_prefix = 'http://192.168.49.30:8333/gpw/landmask'

subnet = '192.168.49'
hosts = [ f'{subnet}.{i}:8333' for i in range(30,43) ]
tiles = gpd.read_file(tiles_fn)

ids_fn='/mnt/slurm/jobs/wri_shv/global_tiles.csv'
tiles_id = pd.read_csv(ids_fn)['TILE'][start_tile:end_tile]

rfe = list(joblib.load(f'{model_dir}/veg.height.med_ht_rfecv.lz4')['covs_rfe'])
n_features = len(rfe)

models = []

for mod_o, mod_txt in zip(find_files(model_dir,'*.o'), find_files(model_dir,'*.txt')):
    start = time.time()
    llvm_model = lleaves.Model(model_file=mod_txt)
    llvm_model.compile(cache=mod_o, use_fp64=False)
    ttprint(f"Reading {mod_o}")
    models.append(llvm_model)

scale = 10
nodata_val = -32000

for tile_id in tiles_id:
    
    if _processed(s3_prefix, tile_id):
        ttprint(f"Tile {tile_id} is processed. Ignoring it.")
        continue

    try:
        ttprint(f"Tile {tile_id} - Start")
        
        minx, miny, maxx, maxy = tiles[tiles['TILE'] == tile_id].iloc[0].geometry.bounds
        df_features, matrix_idx = _features(features_fn, years, tile_id, rfe)

        bands_list = [1,]
        n_rasters = df_features.shape[0]
        
        shape = (n_rasters, x_size * y_size)
        array = np.empty(shape, dtype=np.float32)
        
        landsat_files, landsat_idx = _raster_paths(df_features, 'landsat')
        landsat_files = [str(r).replace(f"{subnet}.30", f"{subnet}.{30 + int.from_bytes(Path(r).stem.encode(), 'little') % len(hosts)}") for r in landsat_files]
        
        static_files, static_idx, x_off_s, y_off_s = _get_static_layers_info(df_features, tiles, tile_id)

        start = time.time()
        skmap_bindings.readData(array, n_threads, static_files, static_idx, x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
        ttprint(f"Tile {tile_id} - Reading static: {(time.time() - start):.2f} segs")
        
        start = time.time()
        x_off_d, y_off_d = (0, 0)
        skmap_bindings.readData(array, n_threads, landsat_files, landsat_idx, x_off_d, y_off_d, x_size, y_size, bands_list, gdal_opts, 255., np.nan)
        ttprint(f"Tile {tile_id} - Total reading landsat: {(time.time() - start):.2f} segs")
        
        start = time.time()
        in_mem_calc(array, df_features, n_threads)
        ttprint(f"Tile {tile_id} - In memory calc: {(time.time() - start):.2f} segs")
        
        start = time.time()
        mask_file = f'{mask_prefix}/{tile_id}.tif'
        mask = np.zeros((1,x_size * y_size), dtype=np.float32)
        skmap_bindings.readData(mask, n_threads, [mask_file,], [0,], x_off_d, x_off_d, x_size, y_size, [1,], gdal_opts)

        n_data = int(np.sum(mask)) * len(years)

        selected_pix = np.arange(0, x_size * y_size)[mask[0,:] == 1]
        selected_rows = np.concatenate([ selected_pix + (x_size * y_size) * i for i in range(0,len(years)) ]).tolist()
        ttprint(f"Tile {tile_id} - Reading mask and allocating memory: {(time.time() - start):.2f} segs")
        
        start = time.time()
        n_pix = len(years) * x_size * y_size
        array_mem_t = np.empty((n_pix, n_features), dtype=np.float32)
        array_mem = np.empty((n_features, n_pix), dtype=np.float32)
        print(array_mem_t.shape)

        skmap_bindings.reorderArray(array, n_threads, array_mem, matrix_idx)
        skmap_bindings.transposeArray(array_mem, n_threads, array_mem_t)
        ttprint(f"Tile {tile_id} - Transposing data: {(time.time() - start):.2f} segs")

        start = time.time()
        array_t = np.empty((n_data, n_features), dtype=np.float32)
        skmap_bindings.selArrayRows(array_mem_t, n_threads, array_t, selected_rows)
        ttprint(f"Tile {tile_id} - Masking data: {(time.time() - start):.2f} segs")
        
        #del array
        
        start = time.time()
        percentiles = [5., 50., 95.]
        percentiles_codes = ['q.05', 'm', 'q.95']
        n_out_bands = len(percentiles)
        shape = (n_data, n_out_bands)
        out = np.empty(shape, dtype=np.float32)
        out_pi = np.empty((n_data, len(models)), dtype=np.float32)
        
        for i in range(0,len(models)):
            start = time.time()
            out_pi[:,i] = models[i].predict(array_t, n_jobs=96) * scale
            ttprint(f"Tile {tile_id} - Running model {i}: {(time.time() - start):.2f} segs")
            print(out_pi[0:5,i], out_pi.shape)
            
        start = time.time()
        col_in_select = range(0,out_pi.shape[1])
        col_out_select = range(0,out.shape[1])
        skmap_bindings.computePercentiles(out_pi, 96, col_in_select, out, col_out_select, percentiles)
        ttprint(f"Tile {tile_id} - Computing percentiles: {(time.time() - start):.2f} segs")
        
        out_mean = np.empty((n_data,), dtype=np.float32)
        skmap_bindings.nanMean(out_pi, n_threads, out_mean)
        
        out[:,1] = out_mean[:]
        
        print(out[0:5,1], out.shape)
        
        ttprint(f"Tile {tile_id} - Average: {(time.time() - start):.2f} segs")
        
        start = time.time()  
        out_exp = np.empty((array_mem_t.shape[0], out.shape[1]), dtype=np.float32)
        skmap_bindings.fillArray(out_exp, n_threads, nodata_val)
        skmap_bindings.expandArrayRows(out, n_threads, out_exp, selected_rows)
        ttprint(f"Tile {tile_id} - Reversing mask: {(time.time() - start):.2f} segs")

        start = time.time()
        out_idx = range(0, n_out_bands)
        n_out_bands_years = len(out_idx) * len(years)
        out_t = np.empty((out_exp.shape[1],out_exp.shape[0]), dtype=np.float32)
        out_gdal = np.empty((n_out_bands_years, x_size * y_size), dtype=np.float32)
        skmap_bindings.fillArray(out_gdal, n_threads, nodata_val)
        skmap_bindings.transposeArray(out_exp, n_threads, out_t)

        subrows = np.arange(0, len(years))
        rows = out_idx
        subrows_grid, rows_grid = np.meshgrid(subrows, rows)
        inverse_idx = np.empty((n_out_bands_years,2), dtype=np.uintc)
        inverse_idx[:,0] = rows_grid.flatten()
        inverse_idx[:,1] = subrows_grid.flatten()

        skmap_bindings.inverseReorderArray(out_t, n_threads, out_gdal[0:n_out_bands_years,:], inverse_idx)
        ttprint(f"Tile {tile_id} - Transposing output: {(time.time() - start):.2f} segs")
        print(out_gdal.shape)
        
        start = time.time()
        write_idx = range(0, out_gdal.shape[0])
        tmp_dir = str(make_tempdir(tile_id))
        base_raster = landsat_files[-1]

        out_files = []
        for q in percentiles_codes:
            out_files += [ 
                f'gpw_short.veg.height_lgb_{q}_30m_s_{year}0101_{year}1231_go_epsg.4326_v20240809' for year in years 
            ]
            
        out_s3 = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % len(hosts)}/{s3_prefix}/{tile_id}" for o in out_files ]
        base_raster = [ base_raster for o in out_files ]

        x_off_d, y_off_d = (0, 0)

        compression_command = f"gdal_translate -a_scale {1/scale} -a_nodata {nodata_val} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"

        skmap_bindings.writeInt16Data(out_gdal, n_threads, gdal_opts, base_raster, tmp_dir, out_files, write_idx,
        x_off_d, y_off_d, x_size, y_size, nodata_val, compression_command, out_s3)
        ttprint(f"Tile {tile_id} - Exporting output to S3: {(time.time() - start):.2f} segs")

        start = time.time()
        del array
        del array_mem
        del array_mem_t
        del array_t
        del inverse_idx
        del out
        del out_exp
        del out_t
        del out_pi
        del out_mean
        del out_gdal
        gc.collect()
        ttprint(f"Tile {tile_id} - Cleaning arrays: {(time.time() - start):.2f} segs")
        
    except:
        tb = traceback.format_exc()
        ttprint(f"Tile {tile_id} - Prediction error ")
        ttprint(tb)
        
        array = None
        array_mem = None
        array_mem_t = None
        array_t = None
        inverse_idx = None
        out = None
        out_exp = None
        out_t = None
        out_pi = None
        out_mean = None
        out_gdal = None

        del array
        del array_mem
        del array_mem_t
        del array_t
        del inverse_idx
        del out
        del out_exp
        del out_t
        del out_pi
        del out_mean
        del out_gdal
        gc.collect()
        continue

