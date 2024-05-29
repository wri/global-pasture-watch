from typing import Callable, Iterator, List,  Union
from osgeo import gdal, gdal_array
import numpy as np
import SharedArray as sa

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
  futures = { executor.submit(worker, **arg) for arg in args }

  done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
  for task in done:
    err = task.exception()
    if err is not None:
      raise err
    else:
        yield task.result()

def read_raster(raster_file, array_fn, i, band=1, minx = None, maxy = None):
    
  for key in gdal_opts.keys():
      gdal.SetConfigOption(key,gdal_opts[key])
  
  array = sa.attach(array_fn, False)
  
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

def save_raster(base_raster, out_file, array_fn, i, minx = None, maxy = None, co = []):
    
  #for key in gdal_opts.keys():
  #  gdal.SetConfigOption(key,gdal_opts[key])
  
  array = sa.attach(array_fn, False)
  
  base_ds = gdal.Open(base_raster)

  driver = gdal.GetDriverByName('GTiff')
  #cols, rows = ds.RasterXSize, ds.RasterYSize

  out_ds = driver.CreateCopy(out_file, base_ds, options=co)
  #out_ds = driver.Create(out_file, cols, rows, 1, dtype)
  #out_ds.SetGeoTransform(*base_ds.GetGeoTransform())
  out_band = out_ds.GetRasterBand(1)

  xoff, yoff = 0, 0
  if minx is not None and maxy is not None:
    gt = base_ds.GetGeoTransform()
    gti = gdal.InvGeoTransform(gt)

    xoff, yoff = gdal.ApplyGeoTransform(gti, minx, maxy)
    xoff, yoff = int(xoff), int(yoff)

  gdal_array.BandWriteArray(out_band, array[:,:,i], xoff=xoff, yoff=yoff)
  
def _model_input(tile, start_year = 2000, end_year = 2022, bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'], base_url='http://192.168.49.30:8333'):
  prediction_layers = []
  
  for year in range(start_year, end_year + 1):
    for band in bands:
      prediction_layers += [
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0101_{year}0228_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0301_{year}0430_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0501_{year}0630_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0701_{year}0831_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0901_{year}1031_go_epsg.4326_v20230908.tif',
        f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}1101_{year}1231_go_epsg.4326_v20230908.tif'
      ]
  
  raster_files = []
  dict_layers_newnames = {}
  for l in prediction_layers:

    key = Path(l).stem.replace('{year}', '')
    value = Path(l).stem.replace('{year}', str(year))
    dict_layers_newnames[key] = value
    raster_files.append('/vsicurl/' + l.replace('{year}', str(year)))
  
  hosts = [ f'192.168.49.{i}:8333' for i in range(30,43) ]
  raster_files = [str(r).replace("192.168.49.30", f"192.168.49.{30 + int.from_bytes(Path(r).stem.encode(), 'little') % len(hosts)}") for r in raster_files]
  
  return raster_files, dict_layers_newnames

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

def read_rasters(raster_files, idx_list, array_fn = None, minx = None, maxy = None, gdal_opts = []):
  n_files = len(raster_files)

  #if array_fn is None:
  #  ds = gdal.Open(raster_files[0])
  #  cols, rows = ds.RasterXSize, ds.RasterYSize
    
  #  shape = (cols, rows, n_files * 10)
  #  array_fn = 'file://' + str(make_tempfile(prefix='shm_array'))
  #  print(array_fn)
    
  #  sa.create(array_fn, shape, dtype=np.float32)
  
  ttprint(f"Reading {len(raster_files)} raster files.")
  
  args = []
  for raster_file, i in zip(raster_files, idx_list):
    args.append({
      'raster_file': raster_file, 
      'array_fn': array_fn, 
      'i': i,
      'minx': minx,
      'maxy': maxy
    })

  for result in ProcessGeneratorLazy(read_raster, args, len(args)):
    continue

def save_rasters(base_raster, out_files, idx_list, array_fn, minx = None, maxy = None):

  ttprint(f"Saving {len(out_files)} raster files.")
  
  args = []
  for out_file, i in zip(out_files, idx_list):
    args.append({
      'base_raster': base_raster, 
      'out_file': out_file, 
      'array_fn': array_fn, 
      'i': i,
      'minx': minx,
      'maxy': maxy,
      'co': co
    })

  for result in ProcessGeneratorLazy(save_raster, args, len(args)):
    continue

def read_rasters(raster_files, idx_list, array_fn = None, minx = None, maxy = None, gdal_opts = []):
  n_files = len(raster_files)

  #if array_fn is None:
  #  ds = gdal.Open(raster_files[0])
  #  cols, rows = ds.RasterXSize, ds.RasterYSize
    
  #  shape = (cols, rows, n_files * 10)
  #  array_fn = 'file://' + str(make_tempfile(prefix='shm_array'))
  #  ttprint(array_fn)
    
  #  sa.create(array_fn, shape, dtype=np.float32)
  
  ttprint(f"Reading {len(raster_files)} raster files.")
  
  args = []
  for raster_file, i in zip(raster_files, idx_list):
    args.append({
      'raster_file': raster_file, 
      'array_fn': array_fn, 
      'i': i,
      'minx': minx,
      'maxy': maxy
    })

  for result in ProcessGeneratorLazy(read_raster, args, len(args)):
    continue

def eval_calc(array_fn, feature, expr, local_dict, idx):
  array = sa.attach(array_fn, False)

  local_dict = { b: array[:,:,local_dict[b]:local_dict[b]+1]  for b in local_dict.keys() }

  #ttprint(f'Calculating {feature}')
  array[:,:,idx:idx+1] = ne.evaluate(expr, local_dict=local_dict).round()
  array[:,:,idx:idx+1][array[:,:,idx:idx+1] == -np.inf] = 0
  array[:,:,idx:idx+1][array[:,:,idx:idx+1] == +np.inf] = 250

  return feature

def in_mem_calc(lookup, array):
  pref = 'glad.SeasConv.ard2_m_30m_s'
  suff = 'go_epsg.4326_v20230908'
  dates = ['0101_0228','0301_0430','0501_0630','0701_0831','0901_1031','1101_1231']
  bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']

  features = {}

  for dt in dates:
    #local_dict = { b: array[:, :, lookup[f'{b}_{pref}_{dt}_{suff}']:lookup[f'{b}_{pref}_{dt}_{suff}']+1 ] for b in bands}
    local_dict = { b: lookup[f'{b}_{pref}_{dt}_{suff}'] for b in bands}
    features[f'ndvi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) ) ) * 125 + 125', local_dict
    features[f'ndwi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (swir1 * 0.004) ) / ( (nir * 0.004) + (swir1 * 0.004) ) ) * 125 + 125', local_dict
    #features[f'savi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (red * 0.004) )*1.5 / ( (nir * 0.004) + (red * 0.004)  + 0.5) ) * 125 + 125', local_dict
    #features[f'msavi_{pref}_{dt}_{suff}'] = f'( (2 *  (nir * 0.004) + 1 - sqrt((2 *  (nir * 0.004) + 1)**2 - 8 * ( (nir * 0.004) - (red * 0.004) ))) / 2 ) * 125 + 125', local_dict
    #features[f'nbr_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - ( swir2 * 0.004) ) / ( (nir * 0.004) + ( swir2 * 0.004) ) ) * 125 + 125', local_dict
    #features[f'ndmi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) -  (swir1 * 0.004)) / ( (nir * 0.004) +  (swir1 * 0.004)) ) * 125 + 125', local_dict
    #features[f'nbr2_{pref}_{dt}_{suff}'] = f'( ( (swir1 * 0.004) - ( thermal * 0.004) ) / ( (swir1 * 0.004) + ( thermal * 0.004) ) ) * 125 + 125', local_dict
    #features[f'rei_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - blue)/( (nir * 0.004) + blue *  (nir * 0.004)) ) * 125 + 125', local_dict
    features[f'bsi_{pref}_{dt}_{suff}'] = f'( ( ( (swir1 * 0.004) + (red * 0.004) ) - ( (nir * 0.004) + blue) ) / ( ( (swir1 * 0.004) + (red * 0.004) ) + ( (nir * 0.004) + blue) ) ) * 125 + 125', local_dict
    features[f'ndti_{pref}_{dt}_{suff}'] = f'( ( (swir1 * 0.004) - (swir2 * 0.004) )  / ( (swir1 * 0.004) + (swir2 * 0.004) )  ) * 125 + 125', local_dict
    #features[f'ndsi_{pref}_{dt}_{suff}'] = f'( ( (green * 0.004) -  (swir1 * 0.004) ) / ( (green * 0.004) +  (swir1 * 0.004) ) ) * 125 + 125', local_dict
    #features[f'ndsmi_{pref}_{dt}_{suff}'] = f'( ( (nir * 0.004) - (swir2 * 0.004) )  / ( (nir * 0.004) + (swir2 * 0.004) )  ) * 125 + 125', local_dict
    features[f'nirv_{pref}_{dt}_{suff}'] = f'( ( ( ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) ) ) - 0.08) *  (nir * 0.004) ) * 125 + 125', local_dict
    features[f'evi_{pref}_{dt}_{suff}'] = f'( 2.5 * ( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + 6 * (red * 0.004) - 7.5 * ( blue * 0.004) + 1) ) * 125 + 125', local_dict
    features[f'fapar_{pref}_{dt}_{suff}'] = f'( ((( (( (nir * 0.004) - (red * 0.004) ) / ( (nir * 0.004) + (red * 0.004) )) - 0.03) * (0.95 - 0.001)) / (0.96 - 0.03)) + 0.001 ) * 125 + 125', local_dict

  new_lookup = []
  bcf_local_dict = {}
  args = []
  for feature, (expr, local_dict) in  features.items():
    args.append({
      'array_fn': array.base.name,
      'feature': feature,
      'expr': expr,
      'local_dict': local_dict,
      'idx': lookup[feature]
    })
    
    if 'ndvi_' in feature:
      _feature = feature.replace(f'_{pref}','').replace(f'_{suff}','')
      bcf_local_dict[_feature] = lookup[feature] #array[:,:,gi:gi+1]

  ttprint(f"Calculating {len(args) + 1} in mem features.")
  for feature in ProcessGeneratorLazy(eval_calc, args, len(args)):
    continue

  expr = f'( where( ndvi_0101_0228 <= 169, 100, 0) + where( ndvi_0301_0430 <= 169, 100, 0) + ' + \
         f'  where( ndvi_0501_0630 <= 169, 100, 0) + where( ndvi_0701_0831 <= 169, 100, 0) + ' + \
         f'  where( ndvi_0501_0630 <= 169, 100, 0) + where( ndvi_0701_0831 <= 169, 100, 0) ) / 6'

  feature = f'bsf_{pref}_{suff}'
  eval_calc(array.base.name, feature, expr, bcf_local_dict, lookup[feature])
  #ttprint(f'Calculating {feature}')
  #newdata = ne.evaluate(expr, local_dict=bcf_local_dict).round()
  #array[:,:,gi:gi+1] = newdata
  #new_lookup.append(feature)

  #new_lookup = {new_lookup[i]: len(lookup) + i for i in range(0, len(new_lookup))}
  #lookup = {**lookup, **new_lookup}

def run_model(model_type, model, model_fn, array_fn, out_fn, i0, i1):

  array = sa.attach(array_fn, False)
  start = time.time()
  #array = array[0:169,:].transpose((1,0))
  #array = array[:,0:169]
  #ttprint(f"Not so silly array copy / transpose: {(time.time() - start):.2f} segs")
  out = sa.attach(out_fn, False)
  #n_features = array.shape[-1]

  start = time.time()
  if model_type == 'tl2cgen':
    import tl2cgen
    model = tl2cgen.Predictor(libpath=model_fn)
    out[:,:,i0:i1] = model.predict(tl2cgen.DMatrix(array)).reshape((out.shape[0],out.shape[1],-1))

  elif model_type == 'xgboost':
    #ttprint(f"Running a {model_type} model ({model_fn})")
    #import xgboost
    #model = xgboost.XGBClassifier() 
    #model.load_model("model/xgb_model.bin")
    out[:,:,i0:i1] = model.predict_proba(array).reshape((out.shape[0],out.shape[1],-1))

  elif model_type == 'hummingbird':
    #ttprint(f"Running a {model_type} model ({model_fn})")
    from hummingbird.ml import load
    model = load(model_fn)
    out[:,:,i0:i1] = model.predict_proba(array).reshape((out.shape[0],out.shape[1],-1))

  else:
    raise Exception(f'Invalid model type {model_type}')
  ttprint(f"Model {model_type} ({model_fn}): {(time.time() - start):.2f} segs")

def _raster_paths(df_features, ftype, tile = None, year = None):

  mask = (df_features['type'] == ftype)

  path_col = 'path'
  if ftype == 'landsat':
    df_features['ypath'] = df_features[mask]['path'].apply(lambda p: p.replace('{tile}', tile).replace('{year}', str(year)))
    path_col = 'ypath'

  ids_list = list(df_features[mask]['idx'])
  raster_files = list(df_features[mask][path_col])

  return raster_files, ids_list

if __name__ == '__main__':

  import rasterio
  from osgeo import gdal, gdal_array

  import numpy as np
  import multiprocessing
  import ctypes
  from multiprocessing import RawArray
  import time
  import concurrent
  from concurrent.futures.process import ProcessPoolExecutor
  from concurrent.futures.thread import ThreadPoolExecutor
  from multiprocessing import shared_memory
  import math
  import tempfile
  from pathlib import Path
  import SharedArray as sa

  import multiprocessing as mp

  TMP_DIR = tempfile.gettempdir()

  from skmap.mapper import LandMapper
  from skmap.misc import find_files
  from skmap.mapper import LandMapper
  from skmap.misc import find_files
  import pandas as pd
  import geopandas as gpd
  import numexpr as ne
  import numpy as np
  import traceback
  import os

  from pathlib import Path
  import bottleneck as bn
  import time

  ttprint("Reading tiles gpkg")
  tiles = gpd.read_file('ard2_final_status.gpkg')
  df_features = pd.read_csv('./model/features.csv')

  tile = '029E_51N'
  minx, miny, maxx, maxy = tiles[tiles['TILE'] == tile].iloc[0].geometry.bounds

  static_files, static_idx = _raster_paths(df_features, 'static')
  n_features = df_features.shape[0]

  shape = (4000, 4000, n_features)
  #array_fn = 'file://' + str(make_tempfile(prefix='shm_array'))
  array_fn = 'file:///tmp/skmap/shm_arraysq5u197x'
  #array = sa.create(array_fn, shape, dtype=np.float32)
  array = sa.attach(array_fn)

  start = time.time()
  #read_rasters(static_files, static_idx, array_fn=array_fn, minx=minx, maxy=maxy)
  ttprint(f"Reading static: {(time.time() - start):.2f} segs")

  import xgboost
  model_xgb = xgboost.XGBClassifier() 
  model_xgb.load_model('model/xgb_model.bin')

  from hummingbird.ml import load
  model_ann = load('model/ann.torch')

  from hummingbird.ml import load
  model_meta = load('model/log-reg.torch.zip')
  ttprint(f"Loading models: {(time.time() - start):.2f} segs")

  for year in [2020]: #range(2000,2022,2):
    
    landsat_files, landsat_idx = _raster_paths(df_features, 'landsat', tile, year)

    start = time.time()
    i = len(static_files) 
    #read_rasters(landsat_files, landsat_idx, array_fn=array_fn, minx=minx, maxy=maxy)
    ttprint(f"Reading landsat: {(time.time() - start):.2f} segs")

    lockup = { row['name']: row['idx'] for _, row in df_features[['idx','name']].iterrows() }

    start = time.time()
    #in_mem_calc(lockup, array)
    #ttprint(f"In memory calc: {(time.time() - start):.2f} segs")
    #ttprint(f"Number of feature: {n_features}")
    #ttprint(f"Array shape: {array.shape}")

    start = time.time()
    #import tl2cgen
    #model_rf = tl2cgen.Predictor(libpath='model/skl_rf_intel.so')

    n_classes = 3
    shape = (4000, 4000, n_classes * 5 + 1)
    out_fn = 'file://' + str(make_tempfile(prefix='shm_output'))
    out = sa.create(out_fn, shape, dtype=np.float32)

    args = [
      {
        'model_type': 'tl2cgen',
        'model_fn': 'model/skl_rf_intel.so',
        'model': None,
        'array_fn': array_fn,
        'out_fn': out_fn,
        'i0': 0,
        'i1': 3
      },
      {
        'model_type': 'xgboost',
        'model_fn': 'model/xgb_model.bin',
        'model': model_xgb,
        'array_fn': array_fn,
        'out_fn': out_fn,
        'i0': 3,
        'i1': 6
      },
      {
        'model_type': 'hummingbird',
        'model_fn': 'model/ann.torch',
        'model': model_ann,
        'array_fn': array_fn,
        'out_fn': out_fn,
        'i0': 6,
        'i1': 9
      }
    ]

    n_models = len(args)

    start = time.time()
    for index in ProcessGeneratorLazy(run_model, args, len(args)):
      continue
    #for arg in args:
    #  run_model(**arg)
    ttprint(f"Running models: {(time.time() - start):.2f} segs")

    mi = n_classes * n_models

    start = time.time()
    # EML probabilities
    out[:,:,9:12] = model_meta.predict_proba(out[:,:,0:mi].reshape(-1,mi)).reshape(out.shape[0],out.shape[1],-1).round(2) * 100
    
    # Model deviance
    out[:,:,12:15] = bn.nanstd(out[:,:,0:mi].reshape(out.shape[0],out.shape[1], n_models, n_classes), axis=-2).round(2) * 100

    # Final map
    out[:,:,15] = np.argmax(out[:,:,9:12], axis=-1) + 1

    ttprint(f"Running meta-learner: {(time.time() - start):.2f} segs")
    ttprint(f'Probabilities are in {out_fn}')

    start = time.time()
    out_files = [
      f'./gpw_eml.seeded.grass_30m_m_{year}0101_{year}1231_go_epsg.4326_v20240206_py.tif',
      f'./gpw_eml.semi.nat.grass_30m_m_{year}0101_{year}1231_go_epsg.4326_v20240206_py.tif',
      f'./gpw_eml.seeded.grass_30m_md_{year}0101_{year}1231_go_epsg.4326_v20240206_py.tif',
      f'./gpw_eml.semi.nat.grass_30m_md_{year}0101_{year}1231_go_epsg.4326_v20240206_py.tif',
      f'./gpw_eml.grass.type_30m_c_{year}0101_{year}1231_go_epsg.4326_v20240206_py.tif'
    ]

    out_idx = [ 9, 10, 12, 13, 15 ]

    save_rasters(landsat_files[0], out_files, out_idx, out_fn, minx = minx, maxy = maxy)
    ttprint(f"Writing files: {(time.time() - start):.2f} segs")
