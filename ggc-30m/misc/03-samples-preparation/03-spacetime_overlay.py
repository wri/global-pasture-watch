import geopandas as gpd
import pandas as pd
from skmap.misc import find_files, make_tempdir
#from skmap.misc import vrt_warp
from pathlib import Path
from osgeo.gdal import BuildVRT, SetConfigOption
import os

from skmap.mapper import SpaceTimeOverlay, SpaceOverlay
from skmap import parallel 

def _build_vrt(raster_file, te):
  outdir = make_tempdir()
  outfile_1 = str(Path(outdir).joinpath(str(Path(str(raster_file).split('?')[0]).stem + '.vrt')))
  ds_1 = BuildVRT(outfile_1, f'{raster_file}', bandList = [1], outputBounds=te)
  ds_1.FlushCache()

  return outfile_1

def _gdal_clip(raster_file, te):
  outdir = make_tempdir()
  outfile_1 = str(Path(outdir).joinpath(str(Path(str(raster_file).split('?')[0]).stem + '.tif')))
  #ds_1 = BuildVRT(outfile_1, f'{raster_file}', bandList = [1], outputBounds=te)
  #ds_1.FlushCache()
  minx, miny, maxx, maxy = te
  os.system(f"gdal_translate -projwin {minx} {maxy} {maxx} {miny} -co TILED=YES -co BIGTIFF=YES -co COMPRESS=DEFLATE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 {raster_file} {outfile_1} > /dev/null")
  return outfile_1


def clip_raster(raster_files, te):
  outfiles = []
  args = [ (r,te) for r in raster_files ]
  for outfile in parallel.job(_gdal_clip, args, n_jobs=-1, joblib_args={'backend': 'multiprocessing'}):
    outfiles.append(outfile)
  return outfiles

def _raster_files(tile, bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'], base_url='http://192.168.49.30:8333'):
  result = []

  itile = int.from_bytes(tile.encode(), 'little')
  base_url = base_url.replace('192.168.49.30', f'192.168.49.{30 + itile % 13}')

  for band in bands:
    result += [
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0101_{year}0228_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0301_{year}0430_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0501_{year}0630_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0701_{year}0831_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0901_{year}1031_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}1101_{year}1231_go_epsg.4326_v20230908.tif'
    ]
  
  return result

def _raster_files_coarse(tile, bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'], base_url='http://192.168.49.30:8333'):
  result = []

  itile = int.from_bytes(tile.encode(), 'little')
  base_url = base_url.replace('192.168.49.30', f'192.168.49.{30 + itile % 13}')
  'global/wilderness/wilderness_li2022.human.footprint_p_1km_s_{year}0101_{year}1231_go_epsg.4326_v16022022.tif'
  for band in bands:
    result += [
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0101_{year}0228_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0301_{year}0430_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0501_{year}0630_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0701_{year}0831_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}0901_{year}1031_go_epsg.4326_v20230908.tif',
      f'{base_url}/prod-landsat-ard2/{tile}/seasconv/{band}_glad.SeasConv.ard2_m_30m_s_' + '{year}1101_{year}1231_go_epsg.4326_v20230908.tif'
    ]
  
  return result

#def overlay(glad_tile_id, rows):
#  print(f"Overlay for tile {glad_tile_id} shape={rows.shape}")
#  raster_files = _raster_files(glad_tile_id)
#  spt_overlay = SpaceTimeOverlay(rows, 'ref_date', raster_files, verbose=False)
#  result = spt_overlay.run()
#  return result

def overlay(glad_tile_id, rows, static_raster, bounds):
  try:
    SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE','1000')
    print(f"Overlay for tile {glad_tile_id} shape={rows.shape}")
    landsat_files = _raster_files(glad_tile_id)
    spt_overlay = SpaceTimeOverlay(rows, 'ref_date', landsat_files, verbose=False)
    spt_result = spt_overlay.run()
    
    t_files = [   
        'http://192.168.49.30:8333/global/wilderness/wilderness_li2022.human.footprint_p_1km_s_{year}0101_{year}1231_go_epsg.4326_v16022022.tif',
        'http://192.168.49.30:8333/global/veg/veg/veg_ndvi_mod13q1.v061_p50_250m_s0..0cm_{year}.10.01..{year}.10.31_v2.tif'
    ]
    
    #t_raster = []
    #for year in range(2000,2022)
    #    for f in t_files:
    #        t_raster.append(f.replace('{year}',year))
    
    vrt_files = [ Path(f) for f in clip_raster(static_raster, te = bounds) ]    
    
    #s_files = [
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm/pos.openness_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm/neg.openness_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm/slope_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm/geomorphon_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm/hillshade_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm/neg.openness_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm/pos.openness_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm/slope_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    #    '/vsicurl/http://192.168.49.30:8333/global/dtm.bareearth/dtm.bareearth_ensemble_p10_30m_s_2018_go_epsg4326_v20230210.tif'
    #]
    
    vrt_files = [ Path(f) for f in clip_raster(static_raster, te = bounds) ]
    print(f'{vrt_files[0]}')
    spc_overlay = SpaceOverlay(spt_result, vrt_files, verbose=False)
    spc_result = spc_overlay.run()
    dummy = [ os.unlink(f) for f in vrt_files ]
    
    return spc_result
  except:
    return pd.DataFrame()


SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE','1000')
#input_sampl = 'global_samples_v20240210_interp'
input_sampl = 'glance_all_gpw_samples'

tiles = gpd.read_file('ard2_final_status.gpkg')

samples = pd.read_parquet(f'{input_sampl}.pq')
samples = gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples['x'], samples['y']))
samples = samples.set_crs('EPSG:4326')

static_raster = find_files('./static', '*.vrt')

args = []
for glad_tile_id, rows in samples.groupby('glad_tile_id'):
  bounds = tiles[tiles['TILE'] == glad_tile_id].iloc[0].geometry.bounds
  args.append((glad_tile_id, rows, static_raster, bounds))

#args = []
#for glad_tile_id, rows in samples.groupby('glad_tile_id'):
#  args.append((glad_tile_id, rows))

result = []
for df in parallel.job(overlay, args, n_jobs=96):
  result.append(df)

result = pd.concat(result)
result.drop(columns=['geometry']).to_parquet(f'{input_sampl}_overlaid.pq')