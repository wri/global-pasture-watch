import sys
sys.path.insert(0,'./scikit-map')
import skmap

import pyarrow.parquet as pq
from affine import Affine
from pathlib import Path
from pyarrow import Table
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from rasterio.windows import Window
from skmap import io
from skmap.misc import make_tempdir, ttprint
import bottleneck as bn
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
import skmap_bindings as skb
import time
import traceback

def livestock_maps(animals, suff='total_rf_m', y1=2000, y2=2022):
  return [
    f'http://192.168.49.30:8333/gpw/livestock_final/gpw_{animal}.{suff}_1km_s_{y}0101_{y}1231_go_epsg.4326_v1.tif'
    for animal in animals
    for y in range(y1,y2+1)
  ]

def _window(polygon_samp, raster_layer):
  minx, miny, maxx, maxy = polygon_samp.total_bounds
  return  from_bounds(minx, miny, maxx, maxy, rasterio.open(raster_layer).transform).round_lengths().round_offsets()

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


animals = ['cattle', 'sheep', 'goat', 'horse']
y1, y2 = 2000, 2022

nodata_val = -32000
n_threads = 96

crs_igg = '+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'

s3_prefix = 'tmp-gpw/livestock_prod_zeros_nowei_mp_v20250217'

subnet = '192.168.49'
hosts = [ f'{subnet}.{i}:8333' for i in range(30,43) ]
out_pq_file = './faostat_correction_factor.pq'

livestock_total_url = livestock_maps(animals, 'total_rf_m', y1, y2)
livestock_densi_url = livestock_maps(animals, 'density_rf_m', y1, y2) + \
                      livestock_maps(animals, 'density_rf_p.025', y1, y2) + \
                      livestock_maps(animals, 'density_rf_p.975', y1, y2)

start = time.time()
faostat_db = joblib.load('faostat_livestock_all.lz4')
ttprint(f"FAOSTAT db reading ({faostat_db.shape}) {(time.time() - start):.2f} secs")

processed_countries = None
try:
  ttprint(f"Reading file {out_pq_file}")
  processed_countries = list(pd.read_parquet(out_pq_file)['country'].unique())
  ttprint(f"End.")
except:
  pass

start_i = int(sys.argv[1])
end_i = int(sys.argv[2])

for index in range(start_i, end_i):
  try:
    ttprint(f"Country {index} - Start")
    
    #faostat_db[faostat_db['gazName'] == 'Russian Federation']
    polygon_samp = faostat_db[faostat_db.index == index]
    country = polygon_samp['gazName'].iloc[0]
    
    if processed_countries is not None:
      if country in processed_countries:
        ttprint(f"Country {country} already processed.")
        continue

    raster_bounds = rasterio.open(livestock_total_url[0]).bounds
    polygon_samp = polygon_samp.to_crs(crs_igg).clip(raster_bounds)

    start = time.time()
    window = _window(polygon_samp, livestock_total_url[0])
    livestock_data = io.read_rasters_cpp(livestock_total_url, window=window, verbose=False, n_jobs=n_threads)
    livestock_densi_data = io.read_rasters_cpp(livestock_densi_url, window=window, verbose=False, n_jobs=n_threads)
    n_densi = int(len(livestock_densi_url) / 3)
    ttprint(f"Country {index} - Reading livestock layers ({livestock_data.shape}) {(time.time() - start):.2f} secs")

    start = time.time()
    ds = rasterio.open(livestock_total_url[0])
    transform = (
      ds.transform[0], 0.0,  ds.transform[2] + window.col_off * ds.transform[0],
      0.0, ds.transform[4],  ds.transform[5] + window.row_off * ds.transform[4]
    )
    
    out_shape = (window.height,window.width)
    polygon_mask_temp = rasterize(polygon_samp.to_crs(crs_igg).geometry, fill=0, out_shape=out_shape, transform=transform, dtype='float32', all_touched=True)
    polygon_mask = np.empty((livestock_data.shape[1],), dtype='float32')
    polygon_mask[:] = polygon_mask_temp.flatten()
    
    skb.maskDataRows(livestock_data, n_threads, range(0, livestock_data.shape[0]), polygon_mask, 0, np.nan)
    skb.maskDataRows(livestock_densi_data, n_threads, range(0, livestock_densi_data.shape[0]), polygon_mask, 0, np.nan)
    ttprint(f"Country {index} - Masking livestock layers {(time.time() - start):.2f} secs")

    start = time.time()
    livestock_total = bn.nansum(livestock_data, axis=-1)
    livestock_densi = bn.nanmean(livestock_densi_data, axis=-1)
    ttprint(f"Country {index} - Adding all values {(time.time() - start):.2f} secs")

    start = time.time()
    animal_cols = [ f'{animal}_{y}' for animal in animals for y in range(y1,y2+1) ]
    livestock_faostat = polygon_samp[animal_cols].to_numpy()

    faostat_factor = (livestock_faostat / livestock_total).flatten()

    livestock_total_adj = np.empty(livestock_data.shape, dtype=np.float32)
    skb.fillArray(livestock_total_adj, n_threads, nodata_val)
    for i in range(0,len(faostat_factor)):
      nan_mask = np.logical_not(np.isnan(livestock_data[i,:]))
      livestock_total_adj[i,:][nan_mask] = livestock_data[i,:][nan_mask] * faostat_factor[i]
    ttprint(f"Country {index} - Applying mass-preservation adjustment {(time.time() - start):.2f} secs")
    
    start = time.time()
    out_files = [ Path(url).stem.replace('total', f'num.heads.faostat.{index}').replace('rf', f'rf.mp') for url in livestock_total_url ]

    fn_tile_raster = f'/tmp/base_{index}.tif'
    min_x, _, _, max_y = polygon_samp.iloc[0].geometry.bounds
    x_size, y_size = window.width, window.height
    _raster_template(livestock_total_url[0], fn_tile_raster, min_x, max_y, x_size, y_size, np.uint16)

    out_s3 = [ f"g{1 + int.from_bytes(Path(o).stem.encode(), 'little') % len(hosts)}/{s3_prefix}" for o in out_files ]
    tmp_dir = str(make_tempdir(str(index)))

    _ = io.save_rasters_cpp(fn_tile_raster, livestock_total_adj, out_files, out_s3=out_s3, nodata=nodata_val, out_dir=tmp_dir, verbose=False, n_jobs=n_threads)
    ttprint(f"Country {index} - Saving adjusted livestock layers {(time.time() - start):.2f} secs")
    
    #print(livestock_densi[0:n_densi].shape)
    #print(livestock_densi[n_densi:n_densi*2].shape)
    #print(livestock_densi[n_densi*2:n_densi*3].shape)
    #print(livestock_densi.shape)
    #print(livestock_total.shape)
    
    start = time.time()
    faostat_factor_df = pd.DataFrame({
        'country': country,
        'col': animal_cols,
        'adj_factor_faostat': faostat_factor,
        'livestock_total_faostat': livestock_faostat.flatten(),
        'livestock_total_raw': livestock_total.flatten(),
        'livestock_density' : livestock_densi[0:n_densi].flatten(),
        'livestock_density_lower' : livestock_densi[n_densi:n_densi*2].flatten(),
        'livestock_density_upper' : livestock_densi[n_densi*2:n_densi*3].flatten()
    })
    faostat_factor_df['animal'] = faostat_factor_df.col.str.split('_', expand=True)[0]
    faostat_factor_df['year'] = faostat_factor_df.col.str.split('_', expand=True)[1].astype('int')
    faostat_factor_df = faostat_factor_df.drop(columns=['col'])

    print(
      faostat_factor_df[faostat_factor_df['year'] == 2020]
    )
    
    pq.write_to_dataset(
      Table.from_pandas(faostat_factor_df),
      out_pq_file,
      partition_cols=['country'],
      compression="snappy",
      version="2.4",
    )
    ttprint(f"Country {index} - Saving adjustment parameters in {out_pq_file} {(time.time() - start):.2f} secs")

  except:
    tb = traceback.format_exc()
    ttprint(tb)
    ttprint(f"Country {index} {country} - ERROR")
    

  ttprint(f"Country {index} - End")