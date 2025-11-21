import sys
#sys.path.insert(0,'./scikit-map')

import math
import os
import shutil
#os.environ['USE_PYGEOS'] = '0'
#os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
#os.environ['NUMEXPR_MAX_THREADS'] = '96'
#os.environ['NUMEXPR_NUM_THREADS'] = '96'
#os.environ['OMP_THREAD_LIMIT'] = '96'
#os.environ['OMP_NUM_THREADS'] = '96'

import pandas as pd
import geopandas as gpd

from datetime import datetime
from osgeo import gdal, gdal_array
from pathlib import Path
from typing import Callable, Iterator, List,  Union
import bottleneck as bn
import geopandas as gpd
import numexpr as ne
import numpy as np
import pandas as pd
#import SharedArray as sa
import tempfile
import time
import sys
import requests
import joblib
import traceback
#import treelite_runtime
import gc
from skmap.parallel import job


import concurrent.futures
import multiprocessing
from itertools import islice
from multiprocessing import Pool
from joblib import Parallel, delayed

import rasterio 
from affine import Affine

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
import threading
import numpy as np

import skmap_bindings as skb
from skmap.io import save_rasters_cpp

import time
#import lleaves

#ne.set_num_threads(96)

GDAL_OPTS = {
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

def _features(csv_file, years, tile_id, rfe_name_base, land_mask_feat, pot_mask_feat):
  
  df_features = pd.read_csv(csv_file,index_col=0)
  rfe_names = rfe_name_base + [ pot_mask_feat, land_mask_feat ]
  ttprint(f"Features: {rfe_names}")
    
  df_list = []
  df_list += [ df_features[
      np.logical_or(
          np.logical_and.reduce([
              df_features['type'] == 'static',
              df_features['name'].isin(rfe_names)
          ]),
          df_features['name'].str.contains('filtered.dtm_edtm_m_960m')
      )
  ] ]

  for year in years:
    mask = np.logical_and(
        df_features['type'] == 'temporal',
        df_features['name'].isin(rfe_names)
    )
    df = df_features[mask].copy()
    df['path'] = df['path'].apply(lambda p: p.replace('{year}', str(year)))
    df_list += [ df ]

  otf_mask = np.logical_and(
      df_features['type'] == 'on-the-fly',
      df_features['name'].isin(rfe_names)
  )
  otf_sel = df_features[otf_mask]['name'].apply(lambda f: '_'.join(f.split('_')[0:2])).unique()
  for year in years:
    df_list += [ df_features[np.logical_and(otf_mask, df_features['name'].str.contains('|'.join(otf_sel)))] ]

  df_features = pd.concat(df_list)
  df_features = df_features.sort_values(['idx', 'path']).reset_index(drop=True)
  df_features['idx'] = df_features.index

  matrix_idx = []

  for c in rfe_names:
    sel_mask = df_features['name'] == c
    idx = list(df_features[sel_mask]['idx'])
    if len(idx) == 1:
      idx = [ idx[0] for i in range(0,len(years)) ]
    matrix_idx.append(idx)

  matrix_idx = np.array(matrix_idx)
  
  return df_features, matrix_idx

def _raster_paths(df_features, ftype):

  mask = (df_features['type'] == ftype)
  ids_list = list(df_features[mask]['idx'])
  raster_files = list(df_features[mask]['path'])
  
  return raster_files, ids_list

def _get_static_layers_info(df_features, tiles, tile):
  
  min_x, _, _, max_y = tiles[tiles['tile_id'] == tile].iloc[0].geometry.bounds
  static_files, static_idx = _raster_paths(df_features, 'static')
  
  gidal_ds = gdal.Open(static_files[0]) # It is assumed to be the same for all static layers
  gt = gidal_ds.GetGeoTransform()
  gti = gdal.InvGeoTransform(gt)
  x_off_s, y_off_s = gdal.ApplyGeoTransform(gti, min_x, max_y)
  x_off_s, y_off_s = int(x_off_s), int(y_off_s)
  
  return static_files, static_idx, x_off_s, y_off_s

def _geom_temperature(df_features, array, x_size, y_size, x_off_s, y_off_s, base_landsat):
  
  from pyproj import Proj, transform
  
  elev_idx = list(df_features[df_features['name'].str.contains('filtered.dtm_edtm_m_960m')].index)
  lst_min_geo_idx = list(df_features[df_features['name'].str.contains('clm_lst_min.geom.temp')].index)
  lst_max_geo_idx = list(df_features[df_features['name'].str.contains('clm_lst_max.geom.temp')].index)

  lon_lat = np.zeros((2, array.shape[1]), dtype=np.float32)
  skb.getLatLonArray(lon_lat, N_THREADS, GDAL_OPTS, base_landsat, x_off_s, y_off_s, x_size, y_size)
  latitude = lon_lat[1,:].copy()

  inProj = Proj('+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs')
  outProj = Proj(init='epsg:4326')

  _, latitude = transform(inProj, outProj, lon_lat[0,:], lon_lat[1,:])
  latitude = latitude.astype('float32')

  months_min = sorted(set(df_features[df_features['name'].str.contains('clm_lst_min.geom.temp')]['name'].apply(lambda f: int(f.split('_')[6].replace('m','')))))
  doys_min = [ datetime.strptime(f'2000-{m}-15', '%Y-%m-%d').timetuple().tm_yday for m in months_min ]
  doys_min_all = sum([ doys_min for i in range(0, len(years)) ],[])
    
  months_max = sorted(set(df_features[df_features['name'].str.contains('clm_lst_max.geom.temp')]['name'].apply(lambda f: int(f.split('_')[6].replace('m','')))))
  doys_max = [ datetime.strptime(f'2000-{m}-15', '%Y-%m-%d').timetuple().tm_yday for m in months_max ]
  doys_max_all = sum([ doys_max for i in range(0, len(years)) ],[])

  elevation = array[elev_idx[0],:]
  
  skb.computeGeometricTemperature(array, N_THREADS, latitude, elevation, 0.1, 24.16453, -15.71751, 100., lst_min_geo_idx, doys_min_all)
  skb.computeGeometricTemperature(array, N_THREADS, latitude, elevation, 0.1, 37.03043, -15.43029, 100., lst_max_geo_idx, doys_max_all)

def in_mem_calc(array, df_features, x_size, y_size, x_off_s, y_off_s, base_landsat):
  _geom_temperature(df_features, array, x_size, y_size, x_off_s, y_off_s, base_landsat)

def _raster_template(base_raster, tmp_raster, x_off_s, y_off_s, x_size, y_size, dtype):
    
    ds = rasterio.open(base_raster)

    transform = np.array(ds.transform)
    transform[2] = x_off_s
    transform[5] = y_off_s
    transform = Affine(*transform)

    new_dataset = rasterio.open(tmp_raster, 'w', driver='GTiff',
                            height = y_size, width = x_size,
                            count=1, dtype=dtype,
                            crs=ds.crs,
                            transform=transform)

    new_dataset.close()

def _processed(s3_prefix, tile, animal):
  url = f'http://192.168.49.30:8333/{s3_prefix}/{tile}/gpw_{animal}.total_rf_m_1km_s_20210101_20211231_go_epsg.4326_v1.tif'
  r = requests.head(url)
  return (r.status_code == 200)
    
def _single_prediction(predict, X, out, i, lock):
    prediction = predict(X, check_input=False)
    with lock:
        out[i, :] = prediction

def cast_tree_rf(model):
    model.__class__ = TreesRandomForestRegressor
    return model

class TreesRandomForestRegressor(RandomForestRegressor):
    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed according
        to a list of functions that receives the predicted regression targets of each 
        single tree in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        s : an ndarray of shape (n_estimators, n_samples)
            The predicted values for each single tree.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # store the output of every estimator
        assert(self.n_outputs_ == 1)
        pred_t = np.empty((len(self.estimators_), X.shape[0]), dtype=np.float32)
        # Assign chunk of trees to jobs
        n_jobs = min(self.n_estimators, self.n_jobs)
        # Parallel loop prediction
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_single_prediction)(self.estimators_[i].predict, X, pred_t, i, lock)
            for i in range(len(self.estimators_))
        )
        return pred_t

def _grs_url(y1 = 2000, y2=2022):
    return [
        f'/mnt/tupi/WRI/livestock_global_modeling/vrt/gpw_livestock.pot.land_grassland.cropland.rules_p_1km_{y}0101_{y}1231_go_epsg.4326_v1.vrt'
        for y in range(y1, y2+1)
    ]

def _xy_size(tile):
  minx, miny, maxx, maxy = tile.geometry.bounds
  
  if tile_id >= 119:
    y_size = int(math.ceil(abs((maxx - minx) / 1000)))
    x_size = int(math.ceil(abs((maxy - miny) / 1000)))
  else:
    x_size, y_size = int(tile['x_size']), int(tile['y_size'])

  return x_size, y_size

def read_data(df_features, tiles, tile_id, animal, land_mask_url):
  
  bands_list = [1,]
  n_rasters = df_features.shape[0]
  
  tile = tiles[tiles['tile_id'] == tile_id].iloc[0]
  x_size, y_size = _xy_size(tile)
  xy_size = x_size * y_size

  shape = (n_rasters, xy_size)
  array = np.empty(shape, dtype=np.float32)

  temporal_files, temporal_idx = _raster_paths(df_features, 'temporal')
  static_files, static_idx, x_off_s, y_off_s = _get_static_layers_info(df_features, tiles, tile_id)

  start = time.time()
  skb.readData(array, N_THREADS, static_files, static_idx, x_off_s, y_off_s, x_size, y_size, bands_list, GDAL_OPTS)
  ttprint(f"Tile {tile_id} {animal} - Reading static ({x_size}x{y_size} pixels): {(time.time() - start):.2f} segs")

  start = time.time()
  skb.readData(array, N_THREADS, temporal_files, temporal_idx, x_off_s, y_off_s, x_size, y_size, bands_list, GDAL_OPTS)
  ttprint(f"Tile {tile_id} {animal} - Reading temporal ({x_size}x{y_size} pixels): {(time.time() - start):.2f} segs")
  
  start = time.time()
  base_landsat = temporal_files[-1]
  in_mem_calc(array, df_features, x_size, y_size, x_off_s, y_off_s, base_landsat)
  ttprint(f"Tile {tile_id} {animal} - In memory calc: {(time.time() - start):.2f} segs")
  
  start = time.time()
  grs_urls = _grs_url()
  grs_array = np.empty((len(grs_urls), xy_size), dtype=np.float32)
  skb.readData(grs_array, N_THREADS, grs_urls, range(0, len(grs_urls)), x_off_s, y_off_s, x_size, y_size, bands_list, GDAL_OPTS, 255, np.nan)
  ttprint(f"Tile {tile_id} {animal} - Reading potential livestock land mask: {(time.time() - start):.2f} segs")
  
  return array, grs_array

def temporal_mask(grs_array, years, animal):
  
  start = time.time()
  xy_size = grs_array.shape[1]

  mask = np.zeros(xy_size, dtype=np.float32)
  grs_array_t = np.empty((grs_array.shape[1], grs_array.shape[0]), dtype='float32')
  
  skb.transposeArray(grs_array, N_THREADS, grs_array_t)
  skb.nanMean(grs_array_t, N_THREADS, mask)

  mask_pix = np.logical_and(mask > 0, mask <= 100)
  n_data = int(np.sum(mask_pix)) * len(years)
  
  selected_pix = np.arange(0, xy_size)[mask_pix]
  selected_rows = np.concatenate([ selected_pix + (xy_size) * i for i in range(0,len(years)) ]).tolist()
  ttprint(f"Tile {tile_id} {animal} - Producing temporal mask: {(time.time() - start):.2f} segs")

  del grs_array_t, mask, selected_pix

  return n_data, selected_rows

def read_features(tiles, tile_id, years, animal, features_fn, rfe_name, land_mask_feat, pot_mask_feat):
  df_features, matrix_idx = _features(features_fn, years, tile_id, rfe_name, land_mask_feat, pot_mask_feat)
  n_feat = matrix_idx.shape[0]

  array, grs_array = read_data(df_features, tiles, tile_id, animal, land_mask_url)
  n_data, selected_rows = temporal_mask(grs_array, years, animal)
  xy_size = grs_array.shape[1]

  start = time.time()
  n_pix = len(years) * xy_size
  
  array_t_size = (n_pix, n_feat)
  array_mem = np.empty((n_feat, n_pix), dtype=np.float32)
  array_mem_t = np.empty(array_t_size, dtype=np.float32)
  land_mask = np.empty(n_pix, dtype=np.float32)

  skb.reorderArray(array, N_THREADS, array_mem, matrix_idx)
  skb.transposeArray(array_mem, N_THREADS, array_mem_t)
  land_mask[:] = array_mem_t[:, -1]
  ttprint(f"Tile {tile_id} {animal} - Transposing data: {(time.time() - start):.2f} segs")

  start = time.time()
  feat_array = np.empty((n_data, n_feat), dtype=np.float32)
  if n_data > 0:
      skb.selArrayRows(array_mem_t, N_THREADS, feat_array, selected_rows)
      ttprint(f"Tile {tile_id} {animal} - Masking data: {(time.time() - start):.2f} segs")

      feat_array[np.isnan(feat_array)] = 0 # Problems in ['12', '19', '20']

  del array, array_mem, array_mem_t

  return feat_array, land_mask, grs_array, selected_rows

def run_model(prod_rf, pt, feat_array, scale, tile_id, animal):
  
  start = time.time()
  
  n_out_bands = 6
  percentiles = [2.5, 97.5]
  n_trees = prod_rf.n_estimators

  n_pix_val = feat_array.shape[0]
  shape = (n_pix_val, n_out_bands)

  ttprint(f"Tile {tile_id} {animal} - Running prod_rf for {feat_array.shape}")
  #land_mask = feat_array[:,-1]
  pot_mask = feat_array[:,-2]
  rf_pred = prod_rf.predict(feat_array[:,0:-2]).astype('float32')

  if pt is not None:
    rf_pred = pt.inverse_transform(rf_pred.reshape(-1,1)).reshape(rf_pred.shape).astype('float32')
  ttprint("End")

  ttprint(f"Tile {tile_id} {animal} - Transposing predictions")
  rf_pred_t = np.empty((n_pix_val, n_trees), dtype=np.float32)
  skb.transposeArray(rf_pred, N_THREADS, rf_pred_t)
  ttprint(f"Tile {tile_id} {animal} - End")

  ttprint(f"Tile {tile_id} {animal} - Percentile calculation over predictions")
  rf_pred_perc = np.empty((n_pix_val, len(percentiles)), dtype=np.float32)
  skb.computePercentiles(rf_pred_t, N_THREADS, range(rf_pred_t.shape[1]), rf_pred_perc, range(0,len(percentiles)), percentiles)
  ttprint("End")
    
  ttprint(f"Tile {tile_id} {animal} - Averaging predictions")
  rf_pred_avg = np.empty((n_pix_val,), dtype=np.float32)
  skb.nanMean(rf_pred_t, N_THREADS, rf_pred_avg)
  ttprint("End")

  out = np.empty(shape, dtype=np.float32)
  out[:, 0] = rf_pred_avg * (pot_mask > 0) * scale
  out[:, 1] = rf_pred_perc[:,0] * (pot_mask > 0) * scale
  out[:, 2] = rf_pred_perc[:,1] * (pot_mask > 0) * scale
  out[:, 3] = rf_pred_avg * (pot_mask / 100)
  out[:, 4] = rf_pred_perc[:,0] * (pot_mask / 100)
  out[:, 5] = rf_pred_perc[:,1] * (pot_mask / 100)

  del rf_pred, rf_pred_t, rf_pred_perc, rf_pred_avg

  return out

def transp_out(out, land_mask, xy_size, selected_rows, nodata_val):

  start = time.time()

  n_pix = len(years) * xy_size
  n_out = (out.shape[1])

  out_exp = np.empty((n_pix, n_out), dtype=np.float32)
  skb.fillArray(out_exp, N_THREADS, nodata_val)
  if len(selected_rows) > 0:
      skb.expandArrayRows(out, N_THREADS, out_exp, selected_rows)
  ttprint(f"Tile {tile_id} {animal} - Reversing mask: {(time.time() - start):.2f} segs")
  
  start = time.time()
  for i in range(0,out_exp.shape[1]):
      out_exp[:,i][np.logical_and(out_exp[:,i] == nodata_val, land_mask != 2)] = 0
  ttprint(f"Tile {tile_id} {animal} - Applying zeros into land mask: {(time.time() - start):.2f} segs")

  start = time.time()
  
  out_idx = range(0, n_out)
  n_out_years = len(out_idx) * len(years)

  out_t = np.empty((out_exp.shape[1],out_exp.shape[0]), dtype=np.float32)
  out_gdal = np.empty((n_out_years, xy_size), dtype=np.float32)
  skb.fillArray(out_gdal, N_THREADS, nodata_val)
  skb.transposeArray(out_exp, N_THREADS, out_t)

  subrows = np.arange(0, len(years))
  rows = out_idx
  subrows_grid, rows_grid = np.meshgrid(subrows, rows)
  inverse_idx = np.empty((n_out_years, 2), dtype=np.uintc)
  inverse_idx[:,0] = rows_grid.flatten()
  inverse_idx[:,1] = subrows_grid.flatten()

  skb.inverseReorderArray(out_t, N_THREADS, out_gdal, inverse_idx)
  ttprint(f"Tile {tile_id} {animal} - Transposing output: {(time.time() - start):.2f} segs")

  del out_exp, out_t, inverse_idx

  return out_gdal

def save_files(base_raster, tile, animal, out_gdal, out_files, out_s3, nodata_val):
  start = time.time()
  write_idx = range(0, out_gdal.shape[0])
  tmp_dir = str(make_tempdir(str(tile_id)))
  
  base_raster = land_mask_url
  out_files = [ f'gpw_{animal}.density_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
              [ f'gpw_{animal}.density_rf_p.025_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
              [ f'gpw_{animal}.density_rf_p.975_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
              [ f'gpw_{animal}.total_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
              [ f'gpw_{animal}.total_rf_p.025_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
              [ f'gpw_{animal}.total_rf_p.975_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ]

  out_s3 = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % len(hosts)}/{s3_prefix}/{tile_id}" for o in out_files ]
  
  x_off_d, y_off_d = (0, 0)

  compression_command = f"gdal_translate -a_nodata {nodata_val} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
  
  fn_tile_raster = f'/tmp/base_{tile_id}.tif'
  min_x, _, _, max_y = tile.geometry.bounds
  _raster_template(base_raster, fn_tile_raster, min_x, max_y, x_size, y_size, np.uint16)
  
  base_raster = [ fn_tile_raster for o in out_files ]
  skb.writeInt16Data(out_gdal, N_THREADS, GDAL_OPTS, base_raster, tmp_dir, out_files, write_idx,
  0, 0, x_size, y_size, nodata_val, compression_command, out_s3)
  ttprint(f"Tile {tile_id} {animal} - Exporting densities to S3: {(time.time() - start):.2f} segs")

TMP_DIR = tempfile.gettempdir()

wd = '/mnt/tupi/WRI/livestock_global_modeling'
tiles_fn = f'{wd}/gpw_livestock_tile_igh.gpkg' 
features_fn = f'{wd}/livestock_census_ard/features_livestock_all_zonal.csv'

models_dir = f'{wd}/livestock_census_ard/zonal_models_zeros_nowei_prod_v20250924'


pot_mask_feat = f'gpw_livestock.pot.land_grassland.cropland.rules_p_1km_year0101_year1231_go_epsg.4326_v1'
land_mask_feat = f'lcv_landmask_esacci.lc.l4_c_1km_s0..0cm_2000..2015_v1.0'
land_mask_url = f'/mnt/tupi/WRI/livestock_global_modeling/vrt/lcv_landmask_esacci.lc.l4_c_1km_s0..0cm_2000..2015_v1.0.vrt'

years = range(2000,2022 + 1)
N_THREADS = 96
s3_prefix = 'tmp-gpw/livestock_prod_zeros_nowei_v20250924_ultimate_bx'

subnet = '192.168.49'
hosts = [ f'{subnet}.{i}:8333' for i in range(30,43) ]
tiles = gpd.read_file(tiles_fn)

tiles_id = [44, 83, 68, 19, 99]

animals = ['cattle', 'horse', 'sheep', 'goat', 'buffalo']
model_types = ['density_boxcox.rf', 'density_boxcox.rf', 'density_boxcox.rf', 'density_boxcox.rf', 'density_boxcox.rf']
models = []

scale = 10
nodata_val = -32000

for mod_type, animal in zip(model_types, animals):
  ttprint(f"Loading models for {animal}")
  model = joblib.load(f'{models_dir}/{animal}.{animal}_{mod_type}_prod.lz4')

  rf_estimator = model[f"prod_rf"]
  rf_estimator.__class__ = TreesRandomForestRegressor
  
  pt = None
  if 'target_pt' in model:
    ttprint(f"Loading pt for {animal} {mod_type}")
    pt = model[f"target_pt"]

  rfe_name = list(joblib.load(f'{models_dir}/{animal}.{animal}_density_rfecv.lz4')['covs_rfe'])

  models.append([
    animal,
    rf_estimator,
    pt,
    rfe_name
  ])

for (animal, prod_rf, pt, rfe_name) in models:

    for tile_id in tiles_id:
      try:
        ttprint(f"Tile {tile_id} {animal} - Start")
        
        if False: #_processed(s3_prefix, tile_id, animal):
          ttprint(f"Tile {tile_id} {animal} is processed. Ignoring it.")
          continue
        
        tile = tiles[tiles['tile_id'] == tile_id].iloc[0]
        x_size, y_size = _xy_size(tile)
        xy_size = x_size * y_size
        
        feat_array, land_mask, grs_array, selected_rows = read_features(
          tiles, tile_id, years, 
          animal, features_fn, rfe_name, land_mask_feat, pot_mask_feat
        )
        
        if feat_array.shape[0] == 0:
            ttprint(f"Tile {tile_id} {animal} has no land to predict. Ignoring it.")
            
            n_pix_val = feat_array.shape[0]
            shape = (n_pix_val, 6)
            
            out = np.empty(shape, dtype=np.float32)
        else:
            out = run_model(prod_rf, pt, feat_array, scale, tile_id, animal)
        
        out_gdal = transp_out(out, land_mask, xy_size, selected_rows, nodata_val)
        #skb.maskData(out_gdal, N_THREADS, range(0,out_gdal.shape[0]), mas_array, 0, 0)
        
        start = time.time()
        base_raster = f'/tmp/base_{tile_id}.tif'
        min_x, _, _, max_y = tile.geometry.bounds
        _raster_template(land_mask_url, base_raster, min_x, max_y, x_size, y_size, np.uint16)

        out_files = [ f'gpw_{animal}.density_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
                    [ f'gpw_{animal}.density_rf_p.025_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
                    [ f'gpw_{animal}.density_rf_p.975_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
                    [ f'gpw_{animal}.total_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
                    [ f'gpw_{animal}.total_rf_p.025_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
                    [ f'gpw_{animal}.total_rf_p.975_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ]

        out_s3 = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % len(hosts)}/{s3_prefix}/{tile_id}" for o in out_files ]
        save_rasters_cpp(base_raster, out_gdal, out_files, out_s3=out_s3, nodata=nodata_val, dtype=np.int16, n_jobs=N_THREADS, gdal_opts=GDAL_OPTS)
        ttprint(f"Tile {tile_id} {animal} - Exporting output to S3: {(time.time() - start):.2f} segs")
        
        start = time.time()
        del feat_array, grs_array, out, out_gdal
        gc.collect()
        os.unlink(base_raster) 
        ttprint(f"Tile {tile_id} {animal} - Cleaning arrays: {(time.time() - start):.2f} segs")
          
      except:
        tb = traceback.format_exc()
        ttprint(f"Tile {tile_id} {animal} - Prediction error ")
        ttprint(tb)