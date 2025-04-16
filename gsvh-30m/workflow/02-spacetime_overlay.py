import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo import gdal
from skmap.misc import find_files, make_tempdir

from osgeo.gdal import BuildVRT, SetConfigOption
from skmap.mapper import SpaceOverlay, SpaceTimeOverlay
from skmap.misc import find_files
from skmap import misc, parallel
from pathlib import Path
import os

def _landsat(tile, bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'], base_url='http://192.168.49.30:8333'):
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

def _gdal_clip(raster_file, te):
  outdir = make_tempdir()
  outfile_1 = str(Path(outdir).joinpath(str(Path(str(raster_file).split('?')[0]).stem + '.tif')))
  minx, miny, maxx, maxy = te
  os.system(f"gdal_translate -projwin {minx} {maxy} {maxx} {miny} -co TILED=YES -co BIGTIFF=YES -co COMPRESS=DEFLATE -co BLOCKXSIZE=1024 -co BLOCKYSIZE=1024 {raster_file} {outfile_1} > /dev/null")
  return outfile_1

def clip_raster(raster_files, te):
  outfiles = []
  args = [ (r,te) for r in raster_files ]
  for outfile in parallel.job(_gdal_clip, args, n_jobs=-1, joblib_args={'backend': 'multiprocessing'}):
    outfiles.append(outfile)
  return outfiles

def overlay(glad_tile_id, rows, bounds):
  
  try:
      SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE','1000')
      print(f"Overlay for tile {glad_tile_id} shape={rows.shape}")

      temporal_files = _landsat(glad_tile_id)
      spt_overlay = SpaceTimeOverlay(rows, 'survey_dt', temporal_files, verbose=False)
      r1 = spt_overlay.run()
      
      s_files = [
        '/vsicurl/http://192.168.49.30:8333/global/dtm/pos.openness_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
        '/vsicurl/http://192.168.49.30:8333/global/dtm/neg.openness_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
        '/vsicurl/http://192.168.49.30:8333/global/dtm/slope_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
        '/vsicurl/http://192.168.49.30:8333/global/dtm/geomorphon_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
        '/vsicurl/http://192.168.49.30:8333/global/dtm/hillshade_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
        '/vsicurl/http://192.168.49.30:8333/global/dtm/neg.openness_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
        '/vsicurl/http://192.168.49.30:8333/global/dtm/pos.openness_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
        '/vsicurl/http://192.168.49.30:8333/global/dtm/slope_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
        '/vsicurl/http://192.168.49.30:8333/global/dtm.bareearth/dtm.bareearth_ensemble_p10_30m_s_2018_go_epsg4326_v20230210.tif',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m02_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m03_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m04_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m05_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m06_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m07_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m08_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m09_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m10_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m11_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.daytime.m12_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m02_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m03_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m04_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m05_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m06_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m07_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m08_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m09_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m10_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m11_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/clm_lst_mod11a2.nighttime.m12_p50_1km_s0..0cm_2000..2021_v1.2.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m01_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m02_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m03_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m04_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m05_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m06_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m07_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m08_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m09_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m10_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m11_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/mnt/gaia/tmp/static/wv_mcd19a2v061.seasconv.m.m12_p50_1km_s_20000101_20221231_go_epsg.4326_v20230619.vrt',
        '/vsicurl/http://192.168.49.30:8333/gpw/glcluc/lc_glad.glcluc_c_30m_s_20200101_20201231_go_epsg.4326_v20240917_sparse.tif'
      ]
      
      
      vrt_files = [ Path(f) for f in clip_raster(s_files, te = bounds) ]
      print(f'{vrt_files[0]}')
      spc_overlay = SpaceOverlay(r1, vrt_files, verbose=False)
      r2 = spc_overlay.run()
      dummy = [ os.unlink(f) for f in vrt_files ]
      print("Finished.")
      return r2
  except:
      print(f'ERROR {glad_tile_id}')
      return pd.DataFrame([])

samples_fn = 'gpw_short.veg.height_icesat.atl08_point.samples.10.batches_20190101_20221231_go_epsg.4326_v1'

samples = pd.read_parquet(f'{samples_fn}.pq')
samples = gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples['lon_20m'], samples['lat_20m']))
samples['survey_dt'] = pd.to_datetime(samples['year'], format='%Y')
samples = samples.set_crs('EPSG:4326')

tiles = gpd.read_file('ard2_final_status.gpkg')

samples = samples.sjoin(tiles[['TILE','geometry']], how="left")
samples = samples.rename(columns={'TILE':'glad_tile_id'})
samples = samples[(samples['survey_dt'] > '2000-01-01') & (samples['survey_dt'] <= '2022-12-31')]
#samples = samples.sample(100)

args = []
for glad_tile_id, rows in samples.groupby('glad_tile_id'):
  #x1, y1, x2, y2 = tiles[tiles['TILE'] == glad_tile_id].iloc[0]['geometry'].bounds
  #xmin, ymin, xmax, ymax = np.min([x1, x2]), np.min([y1, y2]), np.max([x1, x2]), np.max([y1, y2])
  #print(glad_tile_id, tiles[tiles['TILE'] == glad_tile_id])
  bounds = tiles[tiles['TILE'] == glad_tile_id].iloc[0].geometry.bounds
  args.append((glad_tile_id, rows, bounds))

result = []
for df in parallel.job(overlay, args, n_jobs=96):
  result.append(df)

result = pd.concat(result)
result.drop(columns=['geometry']).to_parquet(f'{samples_fn}_overlaid.pq')