import sys
sys.path.insert(0,'./scikit-map')

import math
import os
import shutil
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
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
import SharedArray as sa
import tempfile
import time
import sys
import requests
import joblib
from hummingbird.ml import load
import traceback
import treelite_runtime
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

import time
import lleaves

#ne.set_num_threads(96)

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

def _features(csv_file, years, tile_id, rfe_names):
  
  df_features = pd.read_csv(csv_file,index_col=0)
    
  df_list = []
  df_list += [ df_features[
      np.logical_or(
          np.logical_and.reduce([
              df_features['type'] == 'static',
              df_features['name'].isin(rfe_names)
          ]),
          df_features['name'].str.contains('filtered.dtm_edtm_m_240m')
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
  #print(otf_sel)
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
  
  min_x, _, _, max_y = tiles[tiles['tile_id'] == tile].iloc[0].geometry.bounds
  static_files, static_idx = _raster_paths(df_features, 'static')
  
  gidal_ds = gdal.Open(static_files[0]) # It is assumed to be the same for all static layers
  gt = gidal_ds.GetGeoTransform()
  gti = gdal.InvGeoTransform(gt)
  x_off_s, y_off_s = gdal.ApplyGeoTransform(gti, min_x, max_y)
  x_off_s, y_off_s = int(x_off_s), int(y_off_s)
  
  return static_files, static_idx, x_off_s, y_off_s

def _geom_temperature(df_features, array, n_threads, x_off_s, y_off_s):
  
  from pyproj import Proj, transform
  
  elev_idx = list(df_features[df_features['name'].str.contains('filtered.dtm_edtm_m_240m')].index)
  lst_min_geo_idx = list(df_features[df_features['name'].str.contains('clm_lst_min.geom.temp')].index)
  lst_max_geo_idx = list(df_features[df_features['name'].str.contains('clm_lst_max.geom.temp')].index)

  base_landsat = temporal_files[-1]
  lon_lat = np.zeros((2, array.shape[1]), dtype=np.float32)
  skb.getLatLonArray(lon_lat, n_threads, gdal_opts, base_landsat, x_off_s, y_off_s, x_size, y_size)
  latitude = lon_lat[1,:].copy()
  #longitude = lon_lat[0,:].copy()

  #inProj = Proj(init='epsg:3035')
  inProj = Proj('+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs')
  outProj = Proj(init='epsg:4326')

  #lon_grid_nat, lat_grid_nat = ds.transform * np.meshgrid(lon_lat[0,:], lon_lat[1,:])
  _, latitude = transform(inProj, outProj, lon_lat[0,:], lon_lat[1,:])
  latitude = latitude.astype('float32')
  print(lon_lat[1,:][0:5])
  print(latitude[0:5])

  #months_min = range(1,13)
  months_min = sorted(set(df_features[df_features['name'].str.contains('clm_lst_min.geom.temp')]['name'].apply(lambda f: int(f.split('_')[6].replace('m','')))))
  doys_min = [ datetime.strptime(f'2000-{m}-15', '%Y-%m-%d').timetuple().tm_yday for m in months_min ]
  doys_min_all = sum([ doys_min for i in range(0, len(years)) ],[])
    
  #months_max = range(1,13)
  months_max = sorted(set(df_features[df_features['name'].str.contains('clm_lst_max.geom.temp')]['name'].apply(lambda f: int(f.split('_')[6].replace('m','')))))
  doys_max = [ datetime.strptime(f'2000-{m}-15', '%Y-%m-%d').timetuple().tm_yday for m in months_max ]
  doys_max_all = sum([ doys_max for i in range(0, len(years)) ],[])

  elevation = array[elev_idx[0],:]
  
  print(months_min, len(lst_min_geo_idx), len(doys_min_all))
  print(months_max, len(lst_max_geo_idx), len(doys_max_all))

  skb.computeGeometricTemperature(array, n_threads, latitude, elevation, 0.1, 24.16453, -15.71751, 100., lst_min_geo_idx, doys_min_all)
  skb.computeGeometricTemperature(array, n_threads, latitude, elevation, 0.1, 37.03043, -15.43029, 100., lst_max_geo_idx, doys_max_all)

def in_mem_calc(data, df_features, n_threads, x_off_s, y_off_s):
  _geom_temperature(df_features, array, n_threads, x_off_s, y_off_s)

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
  print(url)
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
        #f'http://192.168.49.30:8333/gpw/arco/gpw_grassland_rf.savgol.bthr_c12_1km_{y}0101_{y}1231_go_epsg.4326_v1.tif'
        f'/mnt/tupi/WRI/livestock_global_modeling/vrt/gpw_livestock.pot.land_grassland.cropland.rules_p_1km_{y}0101_{y}1231_go_epsg.4326_v1.vrt'
        for y in range(y1, y2+1)
    ]
      
TMP_DIR = tempfile.gettempdir()

wd = '/mnt/tupi/WRI/livestock_global_modeling'
tiles_fn = f'{wd}/gpw_livestock_tile_igh.gpkg' 
features_fn = f'{wd}/livestock_census_ard/features_livestock_all_zonal.csv'

#mask_file = f'{wd}/gpw_grassland.mask_rf.savgol.bthr_max_1km_20000101_20221231_go_epsg.4326_v1.tif'
mask_file = f'/mnt/tupi/WRI/livestock_global_modeling/vrt/gpw_livestock.pot.land_grassland.cropland.rules_p_1km_20220101_20221231_go_epsg.4326_v1.vrt'

models_dir = f'{wd}/livestock_census_ard/zonal_models_ultimate'

#land_mask_url = 'http://192.168.49.30:8333/gpw/arco/lcv_landmask_esacci.lc.l4_c_1km_s0..0cm_2000..2015_v1.0.tif'
land_mask_url = f'/mnt/tupi/WRI/livestock_global_modeling/vrt/lcv_landmask_esacci.lc.l4_c_1km_s0..0cm_2000..2015_v1.0.vrt'

years = range(2000,2022 + 1)
#x_size, y_size = (2250, 2250)
n_threads = 96
n_classes = 3
s3_prefix = 'tmp-gpw/livestock_prod_zeros_nowei_v20250217'

subnet = '192.168.49'
hosts = [ f'{subnet}.{i}:8333' for i in range(30,43) ]
tiles = gpd.read_file(tiles_fn)

#007E_46N
#tiles_id = [44] # 65
#tiles_id = [44, 83, 68, 19, 99]
#tiles_id = [ 119, 122, 123, 125 ]
tiles_id = list(tiles['tile_id'])

animals = ['cattle', 'sheep', 'goat', 'horse']
model_type = ['rf']
models = []

for animal in animals:
  ttprint(f"Loading models for {animal}")
  rf_estimator = joblib.load(f'{models_dir}/{animal}.{animal}_density.rf_prod.lz4')[f"prod_rf"]
  rf_estimator.__class__ = TreesRandomForestRegressor
  
  models.append([
    animal,
    rf_estimator,
    #joblib.load(f'{models_dir}/{animal}.{animal}_density.lgb_prod.lz4')[f"prod_lgb"],
    list(joblib.load(f'{models_dir}/{animal}.{animal}_density_rfecv.lz4')['covs_rfe'])
  ])

scale = 10
nodata_val = -32000

for (animal, prod_rf, rfe_name) in models:

    for tile_id in tiles_id:
      try:
        ttprint(f"Tile {tile_id} {animal} - Start")
        n_features = len(rfe_name)
        
        if _processed(s3_prefix, tile_id, animal):
          ttprint(f"Tile {tile_id} {animal} is processed. Ignoring it.")
          continue
        
        minx, miny, maxx, maxy = tiles[tiles['tile_id'] == tile_id].iloc[0].geometry.bounds
        x_size, y_size = int(tiles.iloc[0]['x_size']), int(tiles.iloc[0]['y_size'])
        if tile_id >= 119:
          y_size = int(math.ceil(abs((maxx - minx) / 1000)))
          x_size = int(math.ceil(abs((maxy - miny) / 1000)))
        
        df_features, matrix_idx = _features(features_fn, years, tile_id, rfe_name)

        bands_list = [1,]
        n_rasters = df_features.shape[0]
        
        shape = (n_rasters, x_size * y_size)
        array = np.empty(shape, dtype=np.float32)

        temporal_files, temporal_idx = _raster_paths(df_features, 'temporal')
        static_files, static_idx, x_off_s, y_off_s = _get_static_layers_info(df_features, tiles, tile_id)

        start = time.time()
        skb.readData(array, n_threads, static_files, static_idx, x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
        ttprint(f"Tile {tile_id} {animal} - Reading static ({x_size}x{y_size} pixels): {(time.time() - start):.2f} segs")

        start = time.time()
        skb.readData(array, n_threads, temporal_files, temporal_idx, x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts)
        ttprint(f"Tile {tile_id} {animal} - Reading temporal ({x_size}x{y_size} pixels): {(time.time() - start):.2f} segs")
        
        start = time.time()
        in_mem_calc(array, df_features, n_threads, x_off_s, y_off_s)
        ttprint(f"Tile {tile_id} {animal} - In memory calc: {(time.time() - start):.2f} segs")
        
        start = time.time()
        grs_urls = _grs_url()
        grs_array = np.empty((len(grs_urls), x_size * y_size), dtype=np.float32)
        skb.readData(grs_array, n_threads, grs_urls, range(0, len(grs_urls)), x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts, 255, np.nan)
        ttprint(f"Tile {tile_id} {animal} - Reading potential livestock land mask: {(time.time() - start):.2f} segs")
        
        start = time.time()
        mas_array_2d = np.empty((1, x_size * y_size), dtype=np.float32)
        mas_array = np.empty(x_size * y_size, dtype=np.float32)
        skb.readData(mas_array_2d, n_threads, [land_mask_url,], [0,], x_off_s, y_off_s, x_size, y_size, bands_list, gdal_opts, 2, np.nan)
        mas_array[:] = mas_array_2d[0,:]
        del mas_array_2d
        ttprint(f"Tile {tile_id} {animal} - Reading land mask: {(time.time() - start):.2f} segs")
    
        mask = np.zeros((x_size * y_size), dtype=np.float32)
        grs_array_t = np.empty((grs_array.shape[1], grs_array.shape[0]), dtype='float32')
        skb.transposeArray(grs_array, n_threads, grs_array_t)
        print(grs_array_t.shape, mask.shape)
        skb.nanMean(grs_array_t, n_threads, mask)
        #skb.readData(mask, n_threads, [mask_file,], [0,], x_off_s, y_off_s, x_size, y_size, [1,], gdal_opts)
        mask_mask = np.logical_and(mask > 0, mask <= 100)
        n_data = int(np.sum(mask_mask)) * len(years)
        print(n_data, np.min(mask), np.mean(mask), np.max(mask))
        print(mask[0:5])

        selected_pix = np.arange(0, x_size * y_size)[mask_mask]
        selected_rows = np.concatenate([ selected_pix + (x_size * y_size) * i for i in range(0,len(years)) ]).tolist()
        ttprint(f"Tile {tile_id} {animal} - Reading mask and allocating memory: {(time.time() - start):.2f} segs")
        
        start = time.time()
        #n_features = len(df_features[df_features[model_names[0]] > -1]['name'].unique()) #df_features[model_name].max() + 1
        n_pix = len(years) * x_size * y_size
        array_mem_t = np.empty((n_pix, n_features), dtype=np.float32)
        array_mem = np.empty((n_features, n_pix), dtype=np.float32)
        skb.reorderArray(array, n_threads, array_mem, matrix_idx)
        skb.transposeArray(array_mem, n_threads, array_mem_t)
        ttprint(f"Tile {tile_id} {animal} - Transposing data: {(time.time() - start):.2f} segs")

        start = time.time()
        array_t = np.empty((n_data, n_features), dtype=np.float32)
        skb.selArrayRows(array_mem_t, n_threads, array_t, selected_rows)
        ttprint(f"Tile {tile_id} {animal} - Masking data: {(time.time() - start):.2f} segs")
        
        del array
        
        start = time.time()
        n_out_bands = 3
        shape = (n_data, n_out_bands)
        out = np.empty(shape, dtype=np.float32)
        n_threads = 96
        percentiles = [2.5, 97.5]
        n_trees = prod_rf.n_estimators
        n_pix_val = array_t.shape[0]

        array_t[np.isnan(array_t)] = 0 # Problems in ['12', '19', '20']
        
        ttprint(f"Tile {tile_id} {animal} - Running prod_rf for {array_t.shape}")
        rf_avg_t = np.empty((n_pix_val, n_trees), dtype=np.float32)
        pred_rf = np.empty((n_pix_val,), dtype=np.float32)
        rf_avg = prod_rf.predict(array_t).astype('float32')
        ttprint("End")

        ttprint(f"Tile {tile_id} {animal} - Averaging prod_rf")
        skb.transposeArray(rf_avg, n_threads, rf_avg_t)
        skb.nanMean(rf_avg_t, n_threads, pred_rf)
        print(pred_rf[0:5])
        ttprint("End")

        ttprint(f"Tile {tile_id} {animal} - Running percentile calculation")
        pred_interval = np.empty((n_pix_val, len(percentiles)), dtype=np.float32)
        skb.computePercentiles(rf_avg_t, n_threads, range(rf_avg_t.shape[1]), pred_interval, range(0,len(percentiles)), percentiles)
        print(pred_interval[0:5,:])
        ttprint("End")
        
        #del scalings
        #del rf_avg
        del rf_avg_t
        
        out[:, 0] = pred_rf * scale
        out[:, 1] = pred_interval[:,0] * scale
        out[:, 2] = pred_interval[:,1] * scale
        
        #del pred_eml
        del pred_rf
        #del pred_lgm
        del pred_interval
        
        start = time.time()
        out_exp = np.empty((array_mem_t.shape[0], out.shape[1]), dtype=np.float32)
        skb.fillArray(out_exp, n_threads, nodata_val)
        skb.expandArrayRows(out, n_threads, out_exp, selected_rows)
        ttprint(f"Tile {tile_id} {animal} - Reversing mask: {(time.time() - start):.2f} segs")
        
        start = time.time()
        out_idx = range(0, n_out_bands)
        n_out_bands_years = len(out_idx) * len(years)
        
        out_t = np.empty((out_exp.shape[1],out_exp.shape[0]), dtype=np.float32)
        out_gdal = np.empty((n_out_bands_years, x_size * y_size), dtype=np.float32)
        skb.fillArray(out_gdal, n_threads, nodata_val)
        
        skb.transposeArray(out_exp, n_threads, out_t)
        
        subrows = np.arange(0, len(years))
        rows = out_idx
        subrows_grid, rows_grid = np.meshgrid(subrows, rows)
        inverse_idx = np.empty((n_out_bands_years,2), dtype=np.uintc)
        inverse_idx[:,0] = rows_grid.flatten()
        inverse_idx[:,1] = subrows_grid.flatten()

        #skb.inverseReorderArray(out_t, n_threads, out_gdal[0:n_out_bands_years,:], inverse_idx)
        skb.inverseReorderArray(out_t, n_threads, out_gdal, inverse_idx)
        print(out_gdal[0,0:5])
        ttprint(f"Tile {tile_id} {animal} - Transposing output: {(time.time() - start):.2f} segs")
        
        out_gdal_t0 = np.empty((out_gdal.shape[1], out_gdal.shape[0]), dtype=np.float32)
        skb.transposeArray(out_gdal, n_threads, out_gdal_t0)
        
        ## Masking the data
        scalings = np.empty(mas_array.shape, dtype=np.float32)
        scalings.fill(1)
        skb.offsetsAndScales(out_gdal_t0, 96, range(out_gdal_t0.shape[0]), mas_array, scalings)
        print(out_gdal_t0[0:5,0])
        
        grs_array_t = np.empty((grs_array.shape[1], grs_array.shape[0]), dtype=np.float32)
        skb.transposeArray(grs_array, n_threads, grs_array_t)
        
        n_years = 23
        grs_array_t = (grs_array_t / 100)

        for i1 in range(0, out_gdal_t0.shape[1], n_years):
            i2 = i1 + n_years
            print("Densities", i1, i2, out_gdal_t0[:,i1:i2].shape, out_gdal_t0.shape)
            out_gdal_t0[:,i1:i2] = ( (grs_array_t > 0).astype('int') * out_gdal_t0[:,i1:i2])
        
        print(out_gdal_t0[0:5,0])
        out_gdal_t0[np.isnan(out_gdal_t0)] = nodata_val
        print(out_gdal_t0[0:5,0])
        
        skb.transposeArray(out_gdal_t0, n_threads, out_gdal)
        
        #out_t_tot = np.empty((out_exp.shape[1],out_exp.shape[0]), dtype=np.float32)
        #out_gdal_tot = np.empty((n_out_bands_years, x_size * y_size), dtype=np.float32)
        #skb.fillArray(out_gdal_tot, n_threads, nodata_val)
        
        #scalings = np.empty(mas_array.shape, dtype=np.float32)
        #scalings.fill(1)
        
        #skb.offsetsAndScales(out_exp, n_threads, range(out_exp.shape[0]), mas_array, scalings)
        #skb.transposeArray(out_exp, n_threads, out_t_tot)
        #out_t_tot = ((grs_array / 100) * out_t_tot)
        
        del array_mem
        del array_mem_t
        del array_t
        del inverse_idx
        del out
        del out_exp
        del out_t
        
        start = time.time()
        write_idx = range(0, out_gdal.shape[0])
        tmp_dir = str(make_tempdir(str(tile_id)))
        base_raster = temporal_files[-1]
        
        out_files = [ f'gpw_{animal}.density_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
          [ f'gpw_{animal}.density_rf_p.025_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
          [ f'gpw_{animal}.density_rf_p.975_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ]

        out_s3 = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % len(hosts)}/{s3_prefix}/{tile_id}" for o in out_files ]
        
        x_off_d, y_off_d = (0, 0)

        compression_command = f"gdal_translate -a_scale {1/scale} -a_nodata {nodata_val} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
        
        fn_tile_raster = f'/tmp/base_{tile_id}.tif'
        min_x, _, _, max_y = tiles[tiles['tile_id'] == tile_id].iloc[0].geometry.bounds
        _raster_template(base_raster, fn_tile_raster, min_x, max_y, x_size, y_size, np.uint16)
        
        base_raster = [ fn_tile_raster for o in out_files ]
        skb.writeInt16Data(out_gdal, n_threads, gdal_opts, base_raster, tmp_dir, out_files, write_idx,
        0, 0, x_size, y_size, nodata_val, compression_command, out_s3)
        ttprint(f"Tile {tile_id} {animal} - Exporting densities to S3: {(time.time() - start):.2f} segs")
        
        #### Total animals
        out_gdal_t = np.empty((out_gdal.shape[1], out_gdal.shape[0]), dtype=np.float32)
        skb.transposeArray(out_gdal, n_threads, out_gdal_t)
        
        out_gdal_t[out_gdal_t == nodata_val] = np.nan
        
        #scalings = np.empty(mas_array.shape, dtype=np.float32)
        #scalings.fill(1)
        #skb.offsetsAndScales(out_gdal_t, 96, range(out_gdal_t.shape[0]), mas_array, scalings)
        #print(out_gdal_t[0:5,0])
        
        #grs_array_t = np.empty((grs_array.shape[1], grs_array.shape[0]), dtype=np.float32)
        #skb.transposeArray(grs_array, n_threads, grs_array_t)
        
        #n_years = 23
        #grs_array_t = (grs_array_t / 100)

        for i1 in range(0, out_gdal_t.shape[1], n_years):
            i2 = i1 + n_years
            print("Total animals", i1, i2, out_gdal_t[:,i1:i2].shape)
            out_gdal_t[:,i1:i2] = np.round(grs_array_t * out_gdal_t[:,i1:i2] * 1/scale)
        
        print(out_gdal_t[0:5,0])
        #out_gdal_t[out_gdal_t < 0] = nodata_val
        out_gdal_t[np.isnan(out_gdal_t)] = nodata_val
        print(out_gdal_t[0:5,0])
        
        out_gdal_2 = np.empty((out_gdal_t.shape[1], out_gdal_t.shape[0]), dtype=np.float32)
        skb.transposeArray(out_gdal_t, n_threads, out_gdal_2)
        
        start = time.time()
        write_idx = range(0, out_gdal_2.shape[0])
        tmp_dir = str(make_tempdir(str(tile_id)))
        base_raster = temporal_files[-1]

        out_files = [ f'gpw_{animal}.total_rf_m_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
          [ f'gpw_{animal}.total_rf_p.025_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ] + \
          [ f'gpw_{animal}.total_rf_p.975_1km_s_{year}0101_{year}1231_go_epsg.4326_v1' for year in years ]

        out_s3 = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % len(hosts)}/{s3_prefix}/{tile_id}" for o in out_files ]

        x_off_d, y_off_d = (0, 0)
        base_raster = [ fn_tile_raster for o in out_files ]

        compression_command = f"gdal_translate -a_nodata {nodata_val} -co COMPRESS=deflate -co ZLEVEL=9 -co TILED=TRUE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024"
        #print(n_threads, gdal_opts, base_raster, tmp_dir, out_files, write_idx, 0, 0, x_size, y_size, nodata_val, compression_command)
        skb.writeInt16Data(out_gdal_2, 96, gdal_opts, base_raster, tmp_dir, out_files, write_idx,
        0, 0, x_size, y_size, nodata_val, compression_command, out_s3)
        ttprint(f"Tile {tile_id} {animal} - Exporting total to S3: {(time.time() - start):.2f} segs")
        
        start = time.time()
        del out_gdal
        del out_gdal_2
        del out_gdal_t
        del scalings
        del grs_array_t
        del grs_array
        del mas_array
        gc.collect()
        os.unlink(fn_tile_raster) 
        ttprint(f"Tile {tile_id} {animal} - Cleaning arrays: {(time.time() - start):.2f} segs")
          
      except:
        tb = traceback.format_exc()
        ttprint(f"Tile {tile_id} {animal} - Prediction error ")
        ttprint(tb)
        #continue
      